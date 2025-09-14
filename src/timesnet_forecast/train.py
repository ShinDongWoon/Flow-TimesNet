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
from .utils.metrics import smape_grouped
from .utils import io as io_utils
from .data.split import make_holdout_slices, make_rolling_slices
from .data.dataset import SlidingWindowDataset
from .models.timesnet import TimesNet
from .predict import forecast_recursive_batch


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
) -> DataLoader:
    datasets = [
        SlidingWindowDataset(a, input_len, pred_len, mode, recursive_pred_len)
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
                # recursive multi-step forecast during validation
                out = forecast_recursive_batch(model, xb, pred_len)  # [B, H, N]
                out = out[:, : yb.shape[1], :]  # align horizon with provided targets
            ys.append(yb.detach().float().cpu().numpy())
            ps.append(out.detach().float().cpu().numpy())
    Y = np.concatenate(ys, axis=0).reshape(-1, len(ids))
    P = np.concatenate(ps, axis=0).reshape(-1, len(ids))
    return smape_grouped(Y, P, ids=ids)


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
        folds = make_rolling_slices(
            wide, cfg["train"]["val"]["rolling_folds"], cfg["train"]["val"]["rolling_step_days"], cfg["train"]["val"]["holdout_days"]
        )
        train_arrays, val_arrays = [], []
        # Fit scaler on full concatenated train parts (union) to avoid leakage into their val
        trn_concat = pd.concat([tr for tr, _ in folds], axis=0)
        scaler, trn_concat_norm = io_utils.fit_series_scaler(
            trn_concat, cfg["preprocess"]["normalize"], cfg["preprocess"]["normalize_per_series"], cfg["preprocess"]["eps"]
        )
        # Recompute per fold using fitted scaler:
        def _transform_df(df_: pd.DataFrame) -> pd.DataFrame:
            X = df_.values.astype(np.float32)
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

        for tr, va in folds:
            train_arrays.append(_transform_df(tr).values.astype(np.float32))
            val_arrays.append(_transform_df(va).values.astype(np.float32))

    # --- dataloaders
    input_len = int(cfg["model"]["input_len"])
    pred_len = int(cfg["model"]["pred_len"])
    mode = cfg["model"]["mode"]
    dl_train = _build_dataloader(
        train_arrays, input_len, pred_len, mode, cfg["train"]["batch_size"],
        cfg["train"]["num_workers"], cfg["train"]["pin_memory"], cfg["train"]["persistent_workers"],
        cfg["train"]["prefetch_factor"], shuffle=True, drop_last=True
    )
    dl_val = _build_dataloader(
        val_arrays, input_len, pred_len, mode, batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"], pin_memory=cfg["train"]["pin_memory"],
        persistent_workers=cfg["train"]["persistent_workers"], prefetch_factor=cfg["train"]["prefetch_factor"],
        shuffle=False, drop_last=False,
        recursive_pred_len=(pred_len if mode == "recursive" else None)
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
        kernel_set=list(cfg["model"]["inception_kernel_set"]),
        dropout=float(cfg["model"]["dropout"]),
        activation=str(cfg["model"]["activation"]),
        mode=mode,
    ).to(device)

    # Lazily build model parameters so that downstream utilities see them
    with torch.no_grad():
        dummy = torch.zeros(1, input_len, len(ids), device=device)
        model(dummy)

    if cfg["train"]["channels_last"]:
        model = maybe_channels_last(model, True)
    if cfg["train"]["compile"]:
        model = maybe_compile(model, True)

    # --- optimizer / loss
    optim = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
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
    loss_fn = nn.MSELoss()

    # --- training loop
    best_smape = float("inf")
    best_state = None
    epochs = int(cfg["train"]["epochs"])
    grad_clip = float(cfg["train"]["grad_clip_norm"]) if cfg["train"]["grad_clip_norm"] else 0.0

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
        if val_smape < best_smape:
            best_smape = val_smape
            best_state = clean_state_dict(
                {k: v.detach().cpu() for k, v in model.state_dict().items()}
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
