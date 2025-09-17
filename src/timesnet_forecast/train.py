from __future__ import annotations

import os
import time
import math
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .config import Config, save_yaml
from .utils.logging import console, print_config
from .utils.seed import seed_everything
from .utils.torch_opt import (
    amp_autocast,
    maybe_compile,
    move_to_device,
    clean_state_dict,
)
from .utils.metrics import wsmape_grouped, smape_mean
from .utils import io as io_utils
from .data.split import make_holdout_slices, make_rolling_slices
from .data.dataset import SlidingWindowDataset
from .models.timesnet import TimesNet, PeriodicityTransform
from .predict import forecast_recursive_batch


LOG_2PI = math.log(2.0 * math.pi)


def gaussian_nll_loss(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    min_sigma: float = 0.0,
) -> torch.Tensor:
    """Element-wise Gaussian negative log-likelihood.

    Args:
        mu: Predicted mean ``[B, H, N]``.
        sigma: Predicted standard deviation ``[B, H, N]``.
        target: Ground truth ``[B, H, N]``.
        min_sigma: Optional clamp ensuring strictly positive variance.

    Returns:
        Tensor of per-element losses with the same shape as ``mu``.
    """

    if min_sigma > 0.0:
        sigma = torch.clamp(sigma, min=min_sigma)
    log_sigma = torch.log(sigma)
    z = (target - mu) / sigma
    return 0.5 * (z**2 + 2.0 * log_sigma + LOG_2PI)


def _select_device(req: str) -> torch.device:
    if req == "cuda" and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _should_use_cuda_graphs(train_cfg: Dict, device: torch.device) -> bool:
    """Determine whether manual CUDA graph capture should be enabled."""

    return bool(
        train_cfg.get("cuda_graphs", False)
        and device.type == "cuda"
        and not train_cfg.get("compile", False)
    )


def _build_dataloader(
    arrays: List[np.ndarray],
    masks: List[np.ndarray],
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
    pmax_global: int | None = None,
) -> DataLoader:
    if len(arrays) != len(masks):
        raise ValueError("arrays and masks must have the same length")
    datasets = [
        SlidingWindowDataset(
            a,
            input_len,
            pred_len,
            mode,
            recursive_pred_len,
            augment,
            pmax_global=pmax_global,
            valid_mask=m,
        )
        for a, m in zip(arrays, masks)
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


def _assert_min_len(x: torch.Tensor, pmax: int) -> None:
    """Ensure the sequence length meets the model's minimum requirement."""
    if x.size(1) < pmax:
        raise ValueError(
            f"Sequence length {x.size(1)} is shorter than required pmax {pmax}."
        )


def _compute_pmax_global(arrays: List[np.ndarray], k: int, cap: int) -> int:
    """Compute global maximum period length across training arrays.

    Uses :meth:`PeriodicityTransform._topk_freq` to determine dominant
    frequencies for each series and derives the corresponding period lengths.

    Args:
        arrays: List of training arrays shaped ``[T, N]``.
        k: Number of top frequencies to consider.
        cap: Maximum allowed value for the inferred period length.

    Returns:
        Maximum period length observed across all arrays clipped to ``cap``.
    """

    pmax = 1
    if k <= 0:
        return min(pmax, cap)
    for arr in arrays:
        if arr.size == 0:
            continue
        x = torch.from_numpy(arr.T.astype(np.float32))  # [N, T]
        kidx = PeriodicityTransform._topk_freq(x, k)
        if kidx.numel() == 0:
            continue
        periods = torch.clamp(x.shape[-1] // torch.clamp(kidx, min=1), min=1)
        pmax = max(pmax, int(periods.max().item()))
    return min(pmax, cap)


def _masked_mean(loss_tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute the mean of ``loss_tensor`` over valid points given ``mask``."""

    mask = mask.to(loss_tensor.dtype)
    denom = torch.clamp(mask.sum(), min=1.0)
    return (loss_tensor * mask).sum() / denom


def _transform_dataframe(
    df: pd.DataFrame,
    ids: List[str],
    scaler: Optional[Dict[str, Tuple[float, float]]],
    method: str,
) -> pd.DataFrame:
    """Apply a fitted scaler to a wide-format DataFrame."""

    if method == "none" or scaler is None:
        return df.copy()
    X = df.to_numpy(dtype=np.float32, copy=True)
    out = np.zeros_like(X, dtype=np.float32)
    for j, c in enumerate(ids):
        if method == "zscore":
            mu, sd = scaler[c]
            denom = sd if sd != 0 else 1.0
            out[:, j] = (X[:, j] - mu) / denom
        elif method == "minmax":
            mn, mx = scaler[c]
            rng = (mx - mn) if (mx - mn) != 0 else 1.0
            out[:, j] = (X[:, j] - mn) / rng
        else:
            out[:, j] = X[:, j]
    return pd.DataFrame(out, index=df.index, columns=ids)


def _eval_wsmape(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mode: str,
    ids: List[str],
    pred_len: int,
    weights: Dict[str, float] | None = None,
    channels_last: bool = False,
    use_loss_mask: bool = False,
) -> float:
    model.eval()
    ys: List[np.ndarray] = []
    ps: List[np.ndarray] = []
    with torch.inference_mode(), amp_autocast(True if device.type == "cuda" else False):
        for xb, yb, mask in loader:
            xb = move_to_device(xb, device)  # [B, L, N]
            yb = move_to_device(yb, device)  # [B, H_or_1, N]
            loss_mask = move_to_device(mask, device) if use_loss_mask else None
            if loss_mask is not None:
                loss_mask = loss_mask.to(yb.dtype)
            if channels_last and xb.dim() == 4:
                xb = xb.to(memory_format=torch.channels_last)
            _assert_min_len(xb, model.period.pmax)
            if mode == "direct":
                mu, _ = model(xb)
            else:
                # recursive multi-step forecast during validation
                mu, _ = forecast_recursive_batch(model, xb, pred_len)
                mu = mu[:, : yb.shape[1], :]
            if loss_mask is not None:
                yb_eval = yb * loss_mask
                mu_eval = mu * loss_mask
            else:
                yb_eval = yb
                mu_eval = mu
            ys.append(yb_eval.detach().float().cpu().numpy())
            ps.append(mu_eval.detach().float().cpu().numpy())
    Y = np.concatenate(ys, axis=0).reshape(-1, len(ids))
    P = np.concatenate(ps, axis=0).reshape(-1, len(ids))
    return wsmape_grouped(Y, P, ids=ids, weights=None)


def _eval_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mode: str,
    ids: List[str],
    pred_len: int,
    channels_last: bool = False,
    use_loss_mask: bool = False,
    min_sigma: float = 0.0,
) -> Dict[str, float]:
    model.eval()
    ys: List[np.ndarray] = []
    ps: List[np.ndarray] = []
    nll_num = 0.0
    nll_den = 0.0
    with torch.inference_mode(), amp_autocast(True if device.type == "cuda" else False):
        for xb, yb, mask in loader:
            xb = move_to_device(xb, device)  # [B, L, N]
            yb = move_to_device(yb, device)  # [B, H_or_1, N]
            loss_mask = move_to_device(mask, device) if use_loss_mask else None
            if loss_mask is not None:
                loss_mask = loss_mask.to(yb.dtype)
            if channels_last and xb.dim() == 4:
                xb = xb.to(memory_format=torch.channels_last)
            _assert_min_len(xb, model.period.pmax)
            if mode == "direct":
                mu, sigma = model(xb)
            else:
                mu, sigma = forecast_recursive_batch(model, xb, pred_len)
                mu = mu[:, : yb.shape[1], :]
                sigma = sigma[:, : yb.shape[1], :]
            if loss_mask is not None:
                mask_for_loss = loss_mask.to(yb.dtype)
                yb_eval = yb * mask_for_loss
                mu_eval = mu * mask_for_loss
            else:
                mask_for_loss = torch.ones_like(yb, dtype=yb.dtype, device=yb.device)
                yb_eval = yb
                mu_eval = mu
            loss_tensor = gaussian_nll_loss(mu, sigma, yb, min_sigma=min_sigma)
            nll_num += float((loss_tensor * mask_for_loss).sum().item())
            nll_den += float(mask_for_loss.sum().item())
            ys.append(yb_eval.detach().float().cpu().numpy())
            ps.append(mu_eval.detach().float().cpu().numpy())
    Y = np.concatenate(ys, axis=0).reshape(-1, len(ids))
    P = np.concatenate(ps, axis=0).reshape(-1, len(ids))
    smape = smape_mean(Y, P)
    denom = nll_den if nll_den > 0 else 1.0
    return {"nll": nll_num / denom, "smape": smape}


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
    wide_raw = io_utils.pivot_long_to_wide(
        df, date_col=schema["date"], id_col=schema["id"], target_col=schema["target"],
        fill_missing_dates=cfg["data"]["fill_missing_dates"], fillna0=False
    )
    mask_wide = (~wide_raw.isna()).astype(np.float32)
    wide = wide_raw.fillna(0.0)
    if cfg.get("preprocess", {}).get("clip_negative", False):
        wide = wide.clip(lower=0.0)
    ids = list(wide.columns)

    # --- normalization (FIT ONLY ON TRAIN PART to avoid leakage)
    # We'll split first to determine train-only fit.
    norm_method = cfg["preprocess"]["normalize"]
    norm_per_series = cfg["preprocess"]["normalize_per_series"]
    eps = cfg["preprocess"]["eps"]

    if cfg["train"]["val"]["strategy"] == "holdout":
        trn_df, val_df = make_holdout_slices(wide, cfg["train"]["val"]["holdout_days"])
        trn_mask_df, val_mask_df = make_holdout_slices(mask_wide, cfg["train"]["val"]["holdout_days"])
        if norm_method == "none":
            scaler = None
            trn_norm = trn_df.copy()
            val_norm = val_df.copy()
        else:
            scaler, trn_norm = io_utils.fit_series_scaler(
                trn_df, norm_method, norm_per_series, eps
            )
            val_norm = _transform_dataframe(val_df, ids, scaler, norm_method)
        train_arrays = [trn_norm.to_numpy(dtype=np.float32, copy=False)]
        val_arrays = [val_norm.to_numpy(dtype=np.float32, copy=False)]
        train_mask_arrays = [trn_mask_df.to_numpy(dtype=np.float32, copy=False)]
        val_mask_arrays = [val_mask_df.to_numpy(dtype=np.float32, copy=False)]
    else:
        val_cfg = cfg["train"]["val"]
        fold_iter = make_rolling_slices(
            wide, val_cfg["rolling_folds"], val_cfg["rolling_step_days"], val_cfg["holdout_days"]
        )
        try:
            first_tr, _ = next(fold_iter)
        except StopIteration:
            raise ValueError("No folds produced; check rolling validation configuration")

        if norm_method == "none":
            scaler = None
            wide_norm = wide.copy()
        else:
            scaler, _ = io_utils.fit_series_scaler(
                first_tr, norm_method, norm_per_series, eps
            )

            wide_norm = _transform_dataframe(wide, ids, scaler, norm_method)
        train_arrays: List[np.ndarray] = []
        val_arrays: List[np.ndarray] = []
        train_mask_arrays: List[np.ndarray] = []
        val_mask_arrays: List[np.ndarray] = []
        for (tr_df, va_df), (tr_mask_df, va_mask_df) in zip(
            make_rolling_slices(
                wide_norm, val_cfg["rolling_folds"], val_cfg["rolling_step_days"], val_cfg["holdout_days"]
            ),
            make_rolling_slices(
                mask_wide, val_cfg["rolling_folds"], val_cfg["rolling_step_days"], val_cfg["holdout_days"]
            ),
        ):
            train_arrays.append(tr_df.to_numpy(dtype=np.float32, copy=False))
            val_arrays.append(va_df.to_numpy(dtype=np.float32, copy=False))
            train_mask_arrays.append(tr_mask_df.to_numpy(dtype=np.float32, copy=False))
            val_mask_arrays.append(va_mask_df.to_numpy(dtype=np.float32, copy=False))

    # --- compute global period length
    k_periods = int(cfg["model"].get("k_periods", 0))
    pmax_cap = int(cfg["model"].get("pmax_cap", 730))
    pmax_global = _compute_pmax_global(train_arrays, k_periods, pmax_cap)
    cfg.setdefault("model", {})
    cfg["model"]["pmax"] = int(pmax_global)
    min_period_threshold = int(cfg["model"].get("min_period_threshold", 1))
    cfg["model"]["min_period_threshold"] = min_period_threshold

    # --- dataloaders
    input_len = int(cfg["model"]["input_len"])
    pred_len = int(cfg["model"]["pred_len"])
    mode = cfg["model"]["mode"]
    dl_train = _build_dataloader(
        train_arrays, train_mask_arrays, input_len, pred_len, mode, cfg["train"]["batch_size"],
        cfg["train"]["num_workers"], cfg["train"]["pin_memory"], cfg["train"]["persistent_workers"],
        cfg["train"]["prefetch_factor"], shuffle=True, drop_last=True,
        augment=cfg["data"].get("augment"), pmax_global=pmax_global,
    )
    dl_val = _build_dataloader(
        val_arrays, val_mask_arrays, input_len, pred_len, mode, batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"], pin_memory=cfg["train"]["pin_memory"],
        persistent_workers=cfg["train"]["persistent_workers"], prefetch_factor=cfg["train"]["prefetch_factor"],
        shuffle=False, drop_last=False,
        recursive_pred_len=(pred_len if mode == "recursive" else None),
        augment=None, pmax_global=pmax_global,
    )
    if len(dl_val.dataset) == 0:
        raise ValueError("Validation split has no windows; increase train.val.holdout_days or adjust model.input_len/pred_len.")

    use_loss_masking = bool(cfg["train"].get("use_loss_masking", False))
    min_sigma = float(cfg["train"].get("min_sigma", 1e-3))

    # --- model
    model = TimesNet(
        input_len=input_len,
        pred_len=pred_len,
        d_model=int(cfg["model"]["d_model"]),
        n_layers=int(cfg["model"]["n_layers"]),
        k_periods=int(cfg["model"]["k_periods"]),
        pmax=int(cfg["model"]["pmax"]),
        min_period_threshold=min_period_threshold,
        kernel_set=list(cfg["model"]["kernel_set"]),
        dropout=float(cfg["model"]["dropout"]),
        activation=str(cfg["model"]["activation"]),
        mode=mode,
        channels_last=cfg["train"]["channels_last"],
        use_checkpoint=not cfg["train"].get("cuda_graphs", False),
        min_sigma=min_sigma,
    ).to(device)

    # Lazily build model parameters so that downstream utilities see them
    with torch.no_grad():
        dummy = torch.zeros(1, input_len, len(ids), device=device)
        model(dummy)
        if cfg["train"]["channels_last"]:
            model.to(memory_format=torch.channels_last)
    if cfg["train"]["compile"]:
        model = maybe_compile(model, True)

    # --- optimizer / scheduler / loss
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    epochs = int(cfg["train"]["epochs"])

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
    elif sched_type == "cosine":
        t_max_val = sched_cfg.get("T_max", epochs)
        try:
            t_max = int(t_max_val)
        except (TypeError, ValueError):
            t_max = epochs
        scheduler = CosineAnnealingLR(
            optim,
            T_max=t_max,
            eta_min=float(sched_cfg.get("eta_min", 1e-5)),
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
    # --- training loop
    best_nll = float("inf")
    best_smape = float("inf")
    best_state = None
    grad_clip = float(cfg["train"]["grad_clip_norm"]) if cfg["train"]["grad_clip_norm"] else 0.0
    patience_limit = cfg["train"].get("early_stopping_patience")
    patience = 0
    best_epoch = 0
    accum_steps = int(cfg["train"].get("accumulation_steps", 1))

    use_graphs = _should_use_cuda_graphs(cfg["train"], device)
    if use_graphs:
        warmup_iters = 3
        train_iter = iter(dl_train)
        for w in range(warmup_iters):
            try:
                xb_w, yb_w, mb_w = next(train_iter)
            except StopIteration:
                break
            xb_w = xb_w.to(device, non_blocking=True)
            yb_w = yb_w.to(device, non_blocking=True)
            if use_loss_masking:
                mb_w = mb_w.to(device, non_blocking=True).to(yb_w.dtype)
            else:
                mb_w = torch.ones_like(yb_w)
            if cfg["train"]["channels_last"] and xb_w.dim() == 4:
                xb_w = xb_w.to(memory_format=torch.channels_last)
            _assert_min_len(xb_w, model.period.pmax)
            with amp_autocast(cfg["train"]["amp"] and device.type == "cuda"):
                mu_w, sigma_w = model(xb_w)
                loss_tensor = gaussian_nll_loss(mu_w, sigma_w, yb_w, min_sigma=min_sigma)
                loss_w = _masked_mean(loss_tensor, mb_w) / accum_steps
            grad_scaler.scale(loss_w).backward()
            if (w + 1) % accum_steps == 0:
                if grad_clip and grad_clip > 0:
                    grad_scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                grad_scaler.step(optim)
                grad_scaler.update()
                optim.zero_grad(set_to_none=True)

        xb0, yb0, mb0 = next(train_iter)
        xb0 = xb0.to(device, non_blocking=True)
        yb0 = yb0.to(device, non_blocking=True)
        if use_loss_masking:
            mb0 = mb0.to(device, non_blocking=True).to(yb0.dtype)
        else:
            mb0 = torch.ones_like(yb0)
        if cfg["train"]["channels_last"] and xb0.dim() == 4:
            xb0 = xb0.to(memory_format=torch.channels_last)
        static_x = torch.empty_like(xb0)
        static_y = torch.empty_like(yb0)
        static_m = torch.empty_like(mb0)
        static_x.copy_(xb0)
        static_y.copy_(yb0)
        static_m.copy_(mb0)
        capture_stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        optim.zero_grad(set_to_none=True)
        model.eval()
        with torch.cuda.stream(capture_stream):
            graph.capture_begin()
            with amp_autocast(cfg["train"]["amp"] and device.type == "cuda"):
                _assert_min_len(static_x, model.period.pmax)
                static_mu, static_sigma = model(static_x)
                static_loss_tensor = gaussian_nll_loss(
                    static_mu, static_sigma, static_y, min_sigma=min_sigma
                )
                static_loss = _masked_mean(static_loss_tensor, static_m)
                static_scaled = static_loss / accum_steps
            grad_scaler.scale(static_scaled).backward()
            graph.capture_end()
        torch.cuda.current_stream().wait_stream(capture_stream)
        model.train()
        optim.zero_grad(set_to_none=True)

        def graph_step(xb: torch.Tensor, yb: torch.Tensor, mb: torch.Tensor) -> float:
            static_x.copy_(xb)
            static_y.copy_(yb)
            static_m.copy_(mb)
            graph.replay()
            return float(static_loss.item())

    print_config(cfg)
    for ep in range(1, epochs + 1):
        model.train()
        losses: List[float] = []
        optim.zero_grad(set_to_none=True)
        num_batches = len(dl_train)
        for i, (xb, yb, mb) in enumerate(tqdm(dl_train, desc=f"Epoch {ep}/{epochs}", leave=False)):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if use_loss_masking:
                mb = mb.to(device, non_blocking=True).to(yb.dtype)
            else:
                mb = torch.ones_like(yb)
            if cfg["train"]["channels_last"] and xb.dim() == 4:
                xb = xb.to(memory_format=torch.channels_last)
            _assert_min_len(xb, model.period.pmax)
            if use_graphs:
                loss_val = graph_step(xb, yb, mb)
            else:
                with amp_autocast(cfg["train"]["amp"] and device.type == "cuda"):
                    mu, sigma = model(xb)
                    loss_tensor = gaussian_nll_loss(mu, sigma, yb, min_sigma=min_sigma)
                    masked_loss = _masked_mean(loss_tensor, mb)
                    loss = masked_loss / accum_steps
                grad_scaler.scale(loss).backward()
                loss_val = float(masked_loss.item())
            if (i + 1) % accum_steps == 0 or (i + 1) == num_batches:
                if grad_clip and grad_clip > 0:
                    grad_scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                grad_scaler.step(optim)
                grad_scaler.update()
                optim.zero_grad(set_to_none=True)
            losses.append(float(loss_val))

        eval_metrics = _eval_metrics(
            model,
            dl_val,
            device,
            mode,
            ids,
            pred_len,
            cfg["train"]["channels_last"],
            use_loss_mask=use_loss_masking,
            min_sigma=min_sigma,
        )
        val_nll = eval_metrics["nll"]
        val_smape = eval_metrics["smape"]
        console().print(
            f"[bold]Epoch {ep}[/bold] loss={np.mean(losses):.6f}  val_nll={val_nll:.6f}  val_smape={val_smape:.6f}"
        )
        if scheduler is not None:
            if sched_type == "ReduceLROnPlateau":
                scheduler.step(val_nll)
            else:
                scheduler.step()
        if val_nll < best_nll:
            best_nll = val_nll
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
                    f"[yellow]Early stopping at epoch {ep}; best epoch was {best_epoch} with val_nll={best_nll:.6f} (val_smape={best_smape:.6f})[/yellow]"
                )
                break

    console().print(
        f"[bold]Best epoch {best_epoch} with val_nll={best_nll:.6f} (val_smape={best_smape:.6f})[/bold]"
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
    return best_nll, {
        "model": model_path,
        "scaler": scaler_path,
        "schema": schema_path,
        "config": cfg_path,
        "metrics": {"nll": best_nll, "smape": best_smape},
    }


def main() -> None:
    import argparse
    from .config import Config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()
    cfg = Config.from_files(args.config, overrides=args.override).to_dict()
    best_nll, paths = train_once(cfg)
    metrics = paths.get("metrics") if isinstance(paths, dict) else None
    if isinstance(metrics, dict) and "smape" in metrics:
        console().print(
            f"[bold magenta]Final best NLL: {best_nll:.6f} (SMAPE={metrics['smape']:.6f})[/bold magenta]"
        )
    else:
        console().print(f"[bold magenta]Final best NLL: {best_nll:.6f}[/bold magenta]")


if __name__ == "__main__":
    main()
