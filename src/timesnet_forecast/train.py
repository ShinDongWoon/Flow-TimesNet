from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from .config import Config, save_yaml
from .utils.logging import console, print_config
from .utils.seed import seed_everything
from .utils.torch_opt import (
    amp_autocast,
    maybe_channels_last,
    maybe_compile,
    move_to_device,
    clean_state_dict,
)
from .utils.metrics import wsmape_grouped, smape_mean
from .utils import io as io_utils
from .data.split import make_holdout_slices, make_rolling_slices
from .data.dataset import SlidingWindowDataset
from .models.timesnet import TimesNet
from .predict import forecast_recursive_batch


class WSMAPELoss(nn.Module):
    """Differentiable approximation of weighted SMAPE.

    Computes ``2 * |y_pred - y_true| / (|y_true| + |y_pred| + eps)`` and
    averages across all elements. Points where the actual value is exactly
    zero contribute a zero weight to avoid unstable gradients.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        denom = torch.abs(y_true) + torch.abs(y_pred) + self.eps
        diff = torch.abs(y_pred - y_true)
        loss = 2.0 * diff / denom
        mask = (torch.abs(y_true) > self.eps).float()
        if mask.sum() == 0:
            return torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)
        return (loss * mask).sum() / mask.sum()


def _select_device(req: str) -> torch.device:
    if req == "cuda" and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _build_dataloader(
    arrays: List[np.ndarray],
    input_len: int,
    pred_len: int,
    mode: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    shuffle: bool,
    drop_last: bool,
    recursive_pred_len: int | None = None,
    augment: Dict | None = None,
) -> DataLoader:
    datasets = [
        SlidingWindowDataset(a, input_len, pred_len, mode, recursive_pred_len, augment)
        for a in arrays
    ]
    if len(datasets) == 1:
        ds = datasets[0]
    else:
        ds = ConcatDataset(datasets)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else 2,
        drop_last=drop_last,
    )


def _eval_wsmape(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mode: str,
    ids: List[str],
    pred_len: int,
    weights: Dict[str, float] | None = None,
) -> float:
    model.eval()
    ys: List[np.ndarray] = []
    ps: List[np.ndarray] = []
    with torch.inference_mode(), amp_autocast(True if device.type == "cuda" else False):
        for xb, yb in loader:
            xb = move_to_device(xb, device)  # [B, L, N]
            yb = move_to_device(yb, device)  # [B, H_or_1, N]
            if mode == "direct":
                out = model(xb)  # [B, H, N]
            else:
                # recursive multi-step forecast during validation
                out = forecast_recursive_batch(model, xb, pred_len)  # [B, H, N]
                out = out[:, : yb.shape[1], :]  # align horizon with provided targets
            ys.append(yb.detach().float().cpu().numpy())
            ps.append(out.detach().float().cpu().numpy())
    Y = np.concatenate(ys, axis=0).reshape(-1, len(ids))
    P = np.concatenate(ps, axis=0).reshape(-1, len(ids))
    return wsmape_grouped(Y, P, ids=ids, weights=None)


def _eval_smape(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mode: str,
    ids: List[str],
    pred_len: int,
) -> float:
    model.eval()
    ys: List[np.ndarray] = []
    ps: List[np.ndarray] = []
    with torch.inference_mode(), amp_autocast(True if device.type == "cuda" else False):
        for xb, yb in loader:
            xb = move_to_device(xb, device)  # [B, L, N]
            yb = move_to_device(yb, device)  # [B, H_or_1, N]
            if mode == "direct":
                out = model(xb)  # [B, H, N]
            else:
                out = forecast_recursive_batch(model, xb, pred_len)
                out = out[:, : yb.shape[1], :]
            ys.append(yb.detach().float().cpu().numpy())
            ps.append(out.detach().float().cpu().numpy())
    Y = np.concatenate(ys, axis=0).reshape(-1, len(ids))
    P = np.concatenate(ps, axis=0).reshape(-1, len(ids))
    return smape_mean(Y, P)


def train_once(cfg: Dict) -> Tuple[float, Dict]:
    # --- bootstrap
    device = _select_device(cfg["train"]["device"])
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision(cfg["train"]["matmul_precision"])
    seed_everything(int(cfg.get("tuning", {}).get("seed", 2025)))
    console().print(f"[bold green]Device:[/bold green] {device}")

    # --- data loading
    schema = io_utils.resolve_schema(cfg)
    enc = cfg["data"]["encoding"]
    train_path = cfg["data"]["train_csv"]
    df = pd.read_csv(train_path, encoding=enc)
    wide = io_utils.pivot_long_to_wide(
        df, date_col=schema["date"], id_col=schema["id"], target_col=schema["target"],
        fill_missing_dates=cfg["data"]["fill_missing_dates"], fillna0=True
    )
    if cfg.get("preprocess", {}).get("clip_negative", False):
        wide = wide.clip(lower=0.0)
    ids = list(wide.columns)

    # --- normalization (FIT ONLY ON TRAIN PART to avoid leakage)
    # We'll split first to determine train-only fit.
    if cfg["train"]["val"]["strategy"] == "holdout":
        trn_df, val_df = make_holdout_slices(wide, cfg["train"]["val"]["holdout_days"])
        scaler, trn_norm = io_utils.fit_series_scaler(
            trn_df, cfg["preprocess"]["normalize"], cfg["preprocess"]["normalize_per_series"], cfg["preprocess"]["eps"]
        )
        val_norm = pd.DataFrame(
            trn_norm.values[-len(val_df):],  # dummy slice to align shape; we'll properly transform below
            index=val_df.index, columns=ids
        )
        # Proper transform for val using train scaler:
        V = val_df.values.astype(np.float32)
        Vn = io_utils.inverse_transform(V, ids, scaler, method=cfg["preprocess"]["normalize"])  # identity if 'none'?
        # Wait: inverse_transform applies inverse; for transform we reimplement quickly:
        def _transform(X: np.ndarray) -> np.ndarray:
            out = np.zeros_like(X, dtype=np.float32)
            for j, c in enumerate(ids):
                if cfg["preprocess"]["normalize"] == "zscore":
                    mu, sd = scaler[c]
                    out[:, j] = (X[:, j] - mu) / (sd if sd != 0 else 1.0)
                elif cfg["preprocess"]["normalize"] == "minmax":
                    mn, mx = scaler[c]
                    rng = (mx - mn) if (mx - mn) != 0 else 1.0
                    out[:, j] = (X[:, j] - mn) / rng
                else:
                    out[:, j] = X[:, j]
            return out
        val_norm = pd.DataFrame(_transform(val_df.values.astype(np.float32)), index=val_df.index, columns=ids)
        train_arrays = [trn_norm.values.astype(np.float32)]
        val_arrays = [val_norm.values.astype(np.float32)]
    else:
        val_cfg = cfg["train"]["val"]
        fold_iter = make_rolling_slices(
            wide, val_cfg["rolling_folds"], val_cfg["rolling_step_days"], val_cfg["holdout_days"]
        )
        try:
            first_tr, _ = next(fold_iter)
        except StopIteration:
            raise ValueError("No folds produced; check rolling validation configuration")

        scaler, _ = io_utils.fit_series_scaler(
            first_tr, cfg["preprocess"]["normalize"], cfg["preprocess"]["normalize_per_series"], cfg["preprocess"]["eps"]
        )

        def _transform_df(df_: pd.DataFrame) -> pd.DataFrame:
            X = df_.to_numpy(dtype=np.float32, copy=True)
            out = np.zeros_like(X, dtype=np.float32)
            for j, c in enumerate(ids):
                if cfg["preprocess"]["normalize"] == "zscore":
                    mu, sd = scaler[c]
                    out[:, j] = (X[:, j] - mu) / (sd if sd != 0 else 1.0)
                elif cfg["preprocess"]["normalize"] == "minmax":
                    mn, mx = scaler[c]
                    rng = (mx - mn) if (mx - mn) != 0 else 1.0
                    out[:, j] = (X[:, j] - mn) / rng
                else:
                    out[:, j] = X[:, j]
            return pd.DataFrame(out, index=df_.index, columns=ids)

        wide_norm = _transform_df(wide)
        train_arrays: List[np.ndarray] = []
        val_arrays: List[np.ndarray] = []
        for tr_df, va_df in make_rolling_slices(
            wide_norm, val_cfg["rolling_folds"], val_cfg["rolling_step_days"], val_cfg["holdout_days"]
        ):
            train_arrays.append(tr_df.to_numpy(copy=False))
            val_arrays.append(va_df.to_numpy(copy=False))

    # --- dataloaders
    input_len = int(cfg["model"]["input_len"])
    pred_len = int(cfg["model"]["pred_len"])
    mode = cfg["model"]["mode"]
    dl_train = _build_dataloader(
        train_arrays, input_len, pred_len, mode, cfg["train"]["batch_size"],
        cfg["train"]["num_workers"], cfg["train"]["pin_memory"], cfg["train"]["persistent_workers"],
        cfg["train"]["prefetch_factor"], shuffle=True, drop_last=True,
        augment=cfg["data"].get("augment"),
    )
    dl_val = _build_dataloader(
        val_arrays, input_len, pred_len, mode, batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"], pin_memory=cfg["train"]["pin_memory"],
        persistent_workers=cfg["train"]["persistent_workers"], prefetch_factor=cfg["train"]["prefetch_factor"],
        shuffle=False, drop_last=False,
        recursive_pred_len=(pred_len if mode == "recursive" else None),
        augment=None,
    )
    if len(dl_val.dataset) == 0:
        raise ValueError("Validation split has no windows; increase train.val.holdout_days or adjust model.input_len/pred_len.")

    # --- model
    model = TimesNet(
        input_len=input_len,
        pred_len=pred_len,
        d_model=int(cfg["model"]["d_model"]),
        n_layers=int(cfg["model"]["n_layers"]),
        k_periods=int(cfg["model"]["k_periods"]),
        kernel_set=list(cfg["model"]["kernel_set"]),
        dropout=float(cfg["model"]["dropout"]),
        activation=str(cfg["model"]["activation"]),
        mode=mode,
        series_chunk=int(cfg["model"].get("series_chunk", 128)),
    ).to(device)

    # Lazily build model parameters so that downstream utilities see them
    with torch.no_grad():
        dummy = torch.zeros(1, input_len, len(ids), device=device)
        model(dummy)

    if cfg["train"]["channels_last"]:
        model = maybe_channels_last(model, True)
    if cfg["train"]["compile"]:
        model = maybe_compile(model, True)

    # --- optimizer / scheduler / loss
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    sched_cfg = cfg["train"].get("lr_scheduler", {})
    scheduler: torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
    sched_type = sched_cfg.get("type")
    if sched_type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim,
            mode="min",
            factor=float(sched_cfg.get("factor", 0.1)),
            patience=int(sched_cfg.get("patience", 10)),
            threshold=float(sched_cfg.get("threshold", 1e-4)),
            min_lr=float(sched_cfg.get("min_lr", 0.0)),
        )
    elif sched_type == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim,
            step_size=int(sched_cfg.get("step_size", 10)),
            gamma=float(sched_cfg.get("gamma", 0.1)),
        )

    try:
        grad_scaler = torch.amp.GradScaler(
            device_type=device.type,
            enabled=cfg["train"]["amp"] and device.type == "cuda",
        )
    except TypeError:
        # For older PyTorch versions where ``device_type`` is unsupported.
        grad_scaler = torch.amp.GradScaler(
            device=device.type,
            enabled=cfg["train"]["amp"] and device.type == "cuda",
        )
    loss_fn = WSMAPELoss()

    # --- training loop
    best_smape = float("inf")
    best_state = None
    epochs = int(cfg["train"]["epochs"])
    grad_clip = float(cfg["train"]["grad_clip_norm"]) if cfg["train"]["grad_clip_norm"] else 0.0
    patience_limit = cfg["train"].get("early_stopping_patience")
    patience = 0
    best_epoch = 0

    print_config(cfg)
    for ep in range(1, epochs + 1):
        model.train()
        losses: List[float] = []
        for xb, yb in tqdm(dl_train, desc=f"Epoch {ep}/{epochs}", leave=False):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with amp_autocast(cfg["train"]["amp"] and device.type == "cuda"):
                out = model(xb)  # [B, H, N] or [B,1,N]
                loss = loss_fn(out, yb)
            grad_scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                grad_scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            grad_scaler.step(optim)
            grad_scaler.update()
            losses.append(loss.item())

        val_smape = _eval_smape(model, dl_val, device, mode, ids, pred_len)
        console().print(f"[bold]Epoch {ep}[/bold] loss={np.mean(losses):.6f}  val_smape={val_smape:.6f}")
        if scheduler is not None:
            if sched_type == "ReduceLROnPlateau":
                scheduler.step(val_smape)
            else:
                scheduler.step()
        if val_smape < best_smape:
            best_smape = val_smape
            best_state = clean_state_dict(
                {k: v.detach().cpu() for k, v in model.state_dict().items()}
            )
            best_epoch = ep
            patience = 0
        else:
            patience += 1
            if patience_limit is not None and patience > patience_limit:
                console().print(
                    f"[yellow]Early stopping at epoch {ep}; best epoch was {best_epoch} with val_smape={best_smape:.6f}[/yellow]"
                )
                break

    console().print(
        f"[bold]Best epoch {best_epoch} with val_smape={best_smape:.6f}[/bold]"
    )

    # --- save artifacts
    art_dir = cfg["artifacts"]["dir"]
    os.makedirs(art_dir, exist_ok=True)
    model_path = os.path.join(art_dir, cfg["artifacts"]["model_file"])
    torch.save(
        best_state if best_state is not None else clean_state_dict(model.state_dict()),
        model_path,
    )

    # Save scaler/schema/config
    scaler_path = os.path.join(art_dir, cfg["artifacts"]["scaler_file"])
    schema_path = os.path.join(art_dir, cfg["artifacts"]["schema_file"])
    cfg_path = os.path.join(art_dir, cfg["artifacts"]["config_file"])
    io_utils.save_pickle(
        {"scaler": scaler, "method": cfg["preprocess"]["normalize"], "ids": ids},
        scaler_path,
    )
    io_utils.save_json({"date": schema["date"], "target": schema["target"], "id": schema["id"]}, schema_path)
    save_yaml(cfg, cfg_path)
    console().print(f"[green]Saved:[/green] {model_path}, {scaler_path}, {schema_path}, {cfg_path}")
    return best_smape, {"model": model_path, "scaler": scaler_path, "schema": schema_path, "config": cfg_path}


def main() -> None:
    import argparse
    from .config import Config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()
    cfg = Config.from_files(args.config, overrides=args.override).to_dict()
    best_smape, paths = train_once(cfg)
    console().print(f"[bold magenta]Final best SMAPE: {best_smape:.6f}[/bold magenta]")


if __name__ == "__main__":
    main()
