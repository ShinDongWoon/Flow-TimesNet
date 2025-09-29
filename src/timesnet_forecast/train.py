from __future__ import annotations

import os
import time
import math
from typing import Any, Dict, List, Tuple, Optional, Iterable
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .config import PipelineConfig, save_yaml
from .utils.logging import console, print_config
from .utils.seed import seed_everything
from .losses import negative_binomial_mask, negative_binomial_nll
from .utils.torch_opt import (
    amp_autocast,
    maybe_compile,
    move_to_device,
    clean_state_dict,
)
from .utils.metrics import wsmape_grouped, smape_mean
from .utils import io as io_utils
from .utils.static_features import compute_series_features
from .data.split import make_holdout_slices, make_rolling_slices
from .data.dataset import SlidingWindowDataset
from .models.timesnet import TimesNet
from .predict import forecast_recursive_batch


LOG_2PI = math.log(2.0 * math.pi)


def gaussian_nll_loss(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    min_sigma: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Element-wise Gaussian negative log-likelihood.

    Args:
        mu: Predicted mean ``[B, H, N]``.
        sigma: Predicted standard deviation ``[B, H, N]``.
        target: Ground truth ``[B, H, N]``.
        min_sigma: Optional clamp ensuring strictly positive variance. May be a
            scalar ``float`` or a tensor shaped ``[1, 1, N]`` to provide
            per-series floors.

    Returns:
        Tensor of per-element losses with the same shape as ``mu`` stored in
        ``float32`` precision to preserve numerical stability under AMP.
    """

    mu_f32 = mu.to(torch.float32)
    sigma_f32 = sigma.to(torch.float32)
    target_f32 = target.to(torch.float32)

    if isinstance(min_sigma, torch.Tensor):
        if min_sigma.numel() > 0:
            floor = min_sigma.to(device=sigma_f32.device, dtype=sigma_f32.dtype)
            sigma_f32 = torch.maximum(sigma_f32, floor)
    else:
        min_sigma_val = float(min_sigma)
        if min_sigma_val > 0.0:
            sigma_f32 = torch.clamp(sigma_f32, min=min_sigma_val)

    log_sigma = torch.log(sigma_f32)
    z = (target_f32 - mu_f32) / sigma_f32
    loss_f32 = 0.5 * (z**2 + 2.0 * log_sigma + mu_f32.new_tensor(LOG_2PI))

    return loss_f32


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
    masks: List[np.ndarray | None],
    input_len: int,
    pred_len: int,
    stride: int,
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
    series_static: List[np.ndarray | None] | None = None,
    series_ids: List[np.ndarray | None] | None = None,
    time_indices: List[pd.DatetimeIndex | np.ndarray | None] | None = None,
    time_features: List[np.ndarray | None] | None = None,
    time_feature_config: Dict | None = None,
) -> DataLoader:
    if len(arrays) != len(masks):
        raise ValueError("arrays and masks must have the same length")
    if series_static is not None and len(series_static) != len(arrays):
        raise ValueError("series_static must match arrays length when provided")
    if series_ids is not None and len(series_ids) != len(arrays):
        raise ValueError("series_ids must match arrays length when provided")
    if time_indices is not None and len(time_indices) != len(arrays):
        raise ValueError("time_indices must match arrays length when provided")
    if time_features is not None and len(time_features) != len(arrays):
        raise ValueError("time_features must match arrays length when provided")
    datasets = [
        SlidingWindowDataset(
            a,
            input_len,
            pred_len,
            mode,
            recursive_pred_len,
            augment,
            valid_mask=m,
            series_static=series_static[i] if series_static is not None else None,
            series_ids=series_ids[i] if series_ids is not None else None,
            time_index=time_indices[i] if time_indices is not None else None,
            time_features=time_features[i] if time_features is not None else None,
            time_feature_config=time_feature_config,
            stride=stride,
        )
        for i, (a, m) in enumerate(zip(arrays, masks))
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


def _iter_datasets(dataset) -> Iterable[SlidingWindowDataset]:
    if isinstance(dataset, ConcatDataset):
        for sub in dataset.datasets:
            yield from _iter_datasets(sub)
    elif isinstance(dataset, SlidingWindowDataset):
        yield dataset


def _time_feature_dim_from_dataset(dataset) -> int:
    for ds in _iter_datasets(dataset):
        dim = getattr(ds, "time_feature_dim", 0)
        if dim:
            return int(dim)
    return 0


def _time_frequency_from_dataset(dataset) -> str | None:
    for ds in _iter_datasets(dataset):
        freq = getattr(ds, "time_frequency", None)
        if freq:
            return str(freq)
    return None


def _periods_to_day_counts(periods: List[int], freq: str | None) -> List[Optional[float]]:
    if not freq:
        return [None for _ in periods]
    try:
        offset = pd.tseries.frequencies.to_offset(freq)
    except (TypeError, ValueError):
        return [None for _ in periods]

    nanos: Optional[int] = None
    try:
        nanos = int(getattr(offset, "nanos"))
    except (TypeError, AttributeError, ValueError):
        nanos = None

    if not nanos:
        delta = getattr(offset, "delta", None)
        if delta is not None:
            try:
                nanos = int(pd.to_timedelta(delta).value)
            except (ValueError, TypeError):
                nanos = None

    if not nanos or nanos == 0:
        return [None for _ in periods]

    day_scale = nanos / pd.Timedelta(days=1).value
    return [period * day_scale for period in periods]


def _log_selected_periods(
    model: "TimesNet", freq: str | None, epoch: int, batch: int
) -> None:
    selector = getattr(model, "period_selector", None)
    if selector is None:
        return
    periods_tensor = getattr(selector, "last_selected_periods", None)
    if not isinstance(periods_tensor, torch.Tensor) or periods_tensor.numel() == 0:
        return

    periods_unique = (
        torch.unique(periods_tensor.detach().to(device="cpu", dtype=torch.long))
        .tolist()
    )
    if not periods_unique:
        return

    periods_unique.sort()
    day_counts = _periods_to_day_counts(periods_unique, freq)

    parts: List[str] = []
    for period, days in zip(periods_unique, day_counts):
        if days is None:
            parts.append(f"{period}")
        else:
            if abs(days - round(days)) < 1e-6:
                day_repr = f"{int(round(days))}"
            else:
                day_repr = f"{days:.2f}"
            parts.append(f"{period} (~{day_repr}d)")

    message = ", ".join(parts)
    console().print(
        f"[cyan]Epoch {epoch} batch {batch}: selected periods {message}[/cyan]"
    )


def _normalize_optional(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        if all(v is None for v in value):
            return None
    if isinstance(value, torch.Tensor) and value.numel() == 0:
        return None
    return value


def _stack_series_columns(
    per_id_values: Dict[int, List[np.ndarray]], n_ids: int
) -> np.ndarray:
    if n_ids <= 0:
        return np.zeros((0, 0), dtype=np.float32)
    columns: List[np.ndarray] = []
    expected_len: int | None = None
    for sid in range(n_ids):
        series_list = per_id_values.get(sid, [])
        if series_list:
            flat_values = [np.asarray(v, dtype=np.float32).reshape(-1) for v in series_list]
            col = np.concatenate(flat_values, axis=0)
        else:
            col = np.zeros(0, dtype=np.float32)
        if expected_len is None:
            expected_len = int(col.shape[0])
        elif int(col.shape[0]) != expected_len:
            raise ValueError("Mismatched series lengths detected during evaluation")
        columns.append(col.reshape(-1, 1))
    if expected_len is None:
        return np.zeros((0, n_ids), dtype=np.float32)
    return np.concatenate(columns, axis=1)


def _unpack_batch(
    batch,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    if not isinstance(batch, (list, tuple)):
        raise TypeError("batch must be a tuple or list of tensors")
    if len(batch) < 3:
        raise ValueError(f"Unexpected batch size: {len(batch)}")
    xb, yb, mask = batch[0], batch[1], batch[2]
    next_idx = 3
    x_mark: torch.Tensor | None = None
    y_mark: torch.Tensor | None = None
    if len(batch) >= 5:
        x_mark = _normalize_optional(batch[3])
        y_mark = _normalize_optional(batch[4])
        next_idx = 5
    static: torch.Tensor | None = None
    series_ids: torch.Tensor | None = None
    if len(batch) > next_idx:
        static = batch[next_idx]
        next_idx += 1
    if len(batch) > next_idx:
        series_ids = batch[next_idx]
        next_idx += 1
    if len(batch) != next_idx:
        raise ValueError(f"Unexpected batch size: {len(batch)}")
    return xb, yb, mask, x_mark, y_mark, static, series_ids


def _call_model_optional(
    model: nn.Module,
    xb: torch.Tensor,
    x_mark: torch.Tensor | None,
    series_static: torch.Tensor | None,
    series_ids: torch.Tensor | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    kwargs: Dict[str, torch.Tensor] = {}
    if x_mark is not None:
        kwargs["x_mark"] = x_mark
    if series_static is not None:
        kwargs["series_static"] = series_static
    if series_ids is not None:
        kwargs["series_ids"] = series_ids
    try:
        return model(xb, **kwargs)
    except TypeError as err:
        err_str = str(err)
        fallback_keys = ["series_static", "series_ids", "x_mark"]
        for key in fallback_keys:
            if key in kwargs and key in err_str:
                kwargs.pop(key)
                try:
                    return model(xb, **kwargs)
                except TypeError as err_inner:
                    err_str = str(err_inner)
                    continue
        raise


def _assert_min_len(x: torch.Tensor, required_len: int) -> None:
    """Ensure the sequence length meets the model's minimum requirement."""
    if x.size(1) < required_len:
        raise ValueError(
            f"Sequence length {x.size(1)} is shorter than required input_len {required_len}."
        )


def _model_input_len(model: nn.Module) -> int:
    """Retrieve the ``input_len`` attribute from ``model`` for compatibility."""

    if hasattr(model, "input_len"):
        return int(getattr(model, "input_len"))
    raise AttributeError("model does not expose an input_len attribute")


def _masked_mean(loss_tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute the mean of ``loss_tensor`` over valid points given ``mask``."""

    mask = mask.to(loss_tensor.dtype)
    denom = torch.clamp(mask.sum(), min=1.0)
    return (loss_tensor * mask).sum() / denom


def _masked_std(
    arrays: List[np.ndarray],
    masks: List[np.ndarray],
    method: str = "global",
) -> Tuple[float, np.ndarray | None]:
    """Compute a standard deviation summary over ``arrays`` respecting ``masks``.

    Args:
        arrays: Sequence of arrays shaped ``[T, N]`` containing observations.
        masks: Sequence of masks with the same shapes as ``arrays`` where values
            greater than zero mark valid entries.
        method: Aggregation strategy used to summarise the variability. ``"global"``
            reproduces the historical behaviour by computing a single standard
            deviation over all valid points. ``"per_series_median"`` computes the
            per-series standard deviation (again using the mask) and returns the
            median across series, providing a robust alternative that down-weights
            outlier series.

    Returns:
        Tuple containing the aggregate standard deviation summary and an optional
        array with the per-series standard deviations. ``None`` is returned for
        the per-series component when ``method`` does not require it or when the
        number of series cannot be inferred.
    """

    if len(arrays) == 0:
        return 0.0, None

    method_normalized = method.lower()

    if method_normalized == "global":
        total = 0.0
        total_sq = 0.0
        count = 0
        for arr, mask in zip(arrays, masks):
            if arr.size == 0:
                continue
            if mask is None:
                values = arr.reshape(-1)
            else:
                valid = mask > 0.0
                if not np.any(valid):
                    continue
                values = arr[valid]
            if values.size == 0:
                continue
            values64 = values.astype(np.float64, copy=False)
            total += float(values64.sum())
            total_sq += float(np.square(values64).sum())
            count += int(values.size)

        if count == 0:
            return 0.0, None

        mean = total / count
        variance = max(total_sq / count - mean * mean, 0.0)
        return float(math.sqrt(variance)), None

    if method_normalized == "per_series_median":
        n_series: int | None = None
        per_sum: np.ndarray | None = None
        per_sum_sq: np.ndarray | None = None
        per_count: np.ndarray | None = None

        for arr, mask in zip(arrays, masks):
            if arr.size == 0:
                continue
            arr2d = np.asarray(arr)
            if arr2d.ndim == 1:
                arr2d = arr2d.reshape(-1, 1)
            if arr2d.shape[0] == 0 or arr2d.shape[1] == 0:
                continue

            if mask is None:
                mask_bool = np.ones(arr2d.shape, dtype=bool)
            else:
                mask_arr = np.asarray(mask)
                if mask_arr.shape != arr2d.shape:
                    raise ValueError("Mask shape must match array shape for per-series std computation")
                mask_bool = mask_arr > 0.0
            if not np.any(mask_bool):
                continue

            arr64 = arr2d.astype(np.float64, copy=False)
            mask_float = mask_bool.astype(np.float64, copy=False)

            if n_series is None:
                n_series = arr2d.shape[1]
                per_sum = np.zeros(n_series, dtype=np.float64)
                per_sum_sq = np.zeros(n_series, dtype=np.float64)
                per_count = np.zeros(n_series, dtype=np.float64)
            elif n_series != arr2d.shape[1]:
                raise ValueError("All arrays must have the same number of series")

            per_sum += (arr64 * mask_float).sum(axis=0)
            per_sum_sq += (np.square(arr64) * mask_float).sum(axis=0)
            per_count += mask_float.sum(axis=0)

        if n_series is None or per_sum is None or per_sum_sq is None or per_count is None:
            return 0.0, None

        valid_series = per_count > 0.0
        per_series_std = np.zeros(n_series, dtype=np.float64)
        if not np.any(valid_series):
            return 0.0, per_series_std

        means = np.zeros_like(per_sum)
        means[valid_series] = per_sum[valid_series] / per_count[valid_series]
        variances = np.zeros_like(per_sum)
        variances[valid_series] = np.maximum(
            per_sum_sq[valid_series] / per_count[valid_series] - means[valid_series] ** 2,
            0.0,
        )
        per_series_std[valid_series] = np.sqrt(variances[valid_series])
        stds = per_series_std[valid_series]
        if stds.size == 0:
            return 0.0, per_series_std
        return float(np.median(stds)), per_series_std

    raise ValueError(f"Unsupported min_sigma_method '{method}'. Expected 'global' or 'per_series_median'.")


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
    per_id_targets: Dict[int, List[np.ndarray]] = {i: [] for i in range(len(ids))}
    per_id_preds: Dict[int, List[np.ndarray]] = {i: [] for i in range(len(ids))}
    default_series_ids = torch.arange(len(ids), dtype=torch.long, device=device)
    with torch.inference_mode(), amp_autocast(True if device.type == "cuda" else False):
        for batch in loader:
            xb, yb, mask, x_mark, y_mark, static, series_idx = _unpack_batch(batch)
            xb = move_to_device(xb, device)  # [B, L, N]
            yb = move_to_device(yb, device)  # [B, H_or_1, N]
            mask_dev = move_to_device(mask, device)
            if use_loss_mask:
                base_mask = mask_dev > 0.0
            else:
                base_mask = torch.ones_like(yb, dtype=torch.bool, device=device)
            if x_mark is not None:
                x_mark = x_mark.to(device=device, non_blocking=True)
            if y_mark is not None:
                y_mark = y_mark.to(device=device, non_blocking=True)
            if static is not None:
                static = move_to_device(static, device)
            if series_idx is not None:
                series_idx = series_idx.to(device=device, dtype=torch.long, non_blocking=True)
            else:
                series_idx = default_series_ids
            if channels_last and xb.dim() == 4:
                xb = xb.to(memory_format=torch.channels_last)
            _assert_min_len(xb, _model_input_len(model))
            if mode == "direct":
                rate, _ = _call_model_optional(
                    model, xb, x_mark, static, series_idx
                )
            else:
                # recursive multi-step forecast during validation
                rate, _ = forecast_recursive_batch(
                    model,
                    xb,
                    pred_len,
                    x_mark=x_mark,
                    y_mark=y_mark,
                    series_static=static,
                    series_ids=series_idx,
                )
                rate = rate[:, : yb.shape[1], :]
            nb_mask = negative_binomial_mask(
                yb,
                rate,
                torch.ones_like(rate, dtype=rate.dtype, device=rate.device),
                base_mask,
            )
            mask_for_eval = nb_mask.to(yb.dtype)
            yb_eval = yb * mask_for_eval
            rate_eval = rate * mask_for_eval

            series_idx = series_idx if series_idx is not None else default_series_ids
            if series_idx.dim() == 1:
                series_idx = series_idx.unsqueeze(0).expand(yb_eval.size(0), -1)
            series_idx_cpu = series_idx.detach().cpu()
            y_cpu = yb_eval.detach().float().cpu()
            rate_cpu = rate_eval.detach().float().cpu()
            for b in range(y_cpu.size(0)):
                for n in range(y_cpu.size(2)):
                    sid = int(series_idx_cpu[b, n].item())
                    per_id_targets.setdefault(sid, []).append(y_cpu[b, :, n].numpy().reshape(-1))
                    per_id_preds.setdefault(sid, []).append(rate_cpu[b, :, n].numpy().reshape(-1))
    Y = _stack_series_columns(per_id_targets, len(ids))
    P = _stack_series_columns(per_id_preds, len(ids))
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
    min_sigma: float | torch.Tensor = 0.0,
) -> Dict[str, float]:
    model.eval()
    per_id_targets: Dict[int, List[np.ndarray]] = {i: [] for i in range(len(ids))}
    per_id_preds: Dict[int, List[np.ndarray]] = {i: [] for i in range(len(ids))}
    nll_num = 0.0
    nll_den = 0.0
    default_series_ids = torch.arange(len(ids), dtype=torch.long, device=device)
    with torch.inference_mode(), amp_autocast(True if device.type == "cuda" else False):
        for batch in loader:
            xb, yb, mask, x_mark, y_mark, static, series_idx = _unpack_batch(batch)
            xb = move_to_device(xb, device)  # [B, L, N]
            yb = move_to_device(yb, device)  # [B, H_or_1, N]
            mask_dev = move_to_device(mask, device)
            if use_loss_mask:
                base_mask = mask_dev > 0.0
            else:
                base_mask = torch.ones_like(yb, dtype=torch.bool, device=device)
            if x_mark is not None:
                x_mark = x_mark.to(device=device, non_blocking=True)
            if y_mark is not None:
                y_mark = y_mark.to(device=device, non_blocking=True)
            if static is not None:
                static = move_to_device(static, device)
            if series_idx is not None:
                series_idx = series_idx.to(device=device, dtype=torch.long, non_blocking=True)
            else:
                series_idx = default_series_ids
            if channels_last and xb.dim() == 4:
                xb = xb.to(memory_format=torch.channels_last)
            _assert_min_len(xb, _model_input_len(model))
            if mode == "direct":
                rate, dispersion = _call_model_optional(
                    model,
                    xb,
                    x_mark,
                    static,
                    series_idx,
                )
            else:
                rate, dispersion = forecast_recursive_batch(
                    model,
                    xb,
                    pred_len,
                    x_mark=x_mark,
                    y_mark=y_mark,
                    series_static=static,
                    series_ids=series_idx,
                )
                rate = rate[:, : yb.shape[1], :]
                dispersion = dispersion[:, : yb.shape[1], :]
            nb_mask = negative_binomial_mask(yb, rate, dispersion, base_mask)
            mask_for_loss = nb_mask.to(yb.dtype)
            yb_eval = yb * mask_for_loss
            rate_eval = rate * mask_for_loss
            nb_loss = negative_binomial_nll(
                y=yb,
                rate=rate,
                dispersion=dispersion,
                mask=nb_mask,
            )
            mask_total = float(mask_for_loss.sum().item())
            if mask_total <= 0.0:
                mask_total = float(yb.numel()) if yb.numel() > 0 else 1.0
            nll_num += float(nb_loss.item()) * mask_total
            nll_den += mask_total
            series_idx = series_idx if series_idx is not None else default_series_ids
            if series_idx.dim() == 1:
                series_idx = series_idx.unsqueeze(0).expand(yb_eval.size(0), -1)
            series_idx_cpu = series_idx.detach().cpu()
            y_cpu = yb_eval.detach().float().cpu()
            rate_cpu = rate_eval.detach().float().cpu()
            for b in range(y_cpu.size(0)):
                for n in range(y_cpu.size(2)):
                    sid = int(series_idx_cpu[b, n].item())
                    per_id_targets.setdefault(sid, []).append(y_cpu[b, :, n].numpy().reshape(-1))
                    per_id_preds.setdefault(sid, []).append(rate_cpu[b, :, n].numpy().reshape(-1))
    Y = _stack_series_columns(per_id_targets, len(ids))
    P = _stack_series_columns(per_id_preds, len(ids))
    smape = smape_mean(Y, P)
    denom = nll_den if nll_den > 0 else 1.0
    return {"nll": nll_num / denom, "smape": smape}


def train_once(cfg: PipelineConfig | Dict[str, Any]) -> Tuple[float, Dict]:
    # --- bootstrap
    if isinstance(cfg, PipelineConfig):
        pipeline_cfg = cfg
        cfg = pipeline_cfg.to_dict()
    elif isinstance(cfg, dict):
        pipeline_cfg = PipelineConfig.from_mapping(cfg)
        cfg = pipeline_cfg.to_dict()
    else:
        raise TypeError("cfg must be a PipelineConfig or mapping")

    window_cfg = pipeline_cfg.window
    cfg.setdefault("window", {}).update(window_cfg.to_dict())
    cfg.setdefault("model", {}).update(pipeline_cfg.model.to_dict(window_cfg))
    cfg.setdefault("artifacts", {}).setdefault("signature_file", "model_signature.json")

    train_section = cfg.setdefault("train", {})
    val_section = train_section.setdefault("val", {})
    if pipeline_cfg.train.val_holdout_days is not None:
        val_section.setdefault("holdout_days", int(pipeline_cfg.train.val_holdout_days))
    val_section.setdefault("strategy", pipeline_cfg.train.val_strategy)
    if pipeline_cfg.train.val_rolling_folds is not None:
        val_section.setdefault("rolling_folds", int(pipeline_cfg.train.val_rolling_folds))
    if pipeline_cfg.train.val_rolling_step_days is not None:
        val_section.setdefault("rolling_step_days", int(pipeline_cfg.train.val_rolling_step_days))

    device = _select_device(cfg["train"]["device"])
    deterministic = bool(cfg["train"].get("deterministic", False))
    torch.set_float32_matmul_precision(cfg["train"]["matmul_precision"])
    seed_everything(int(cfg.get("tuning", {}).get("seed", 2025)), deterministic=deterministic)
    console().print(f"[bold green]Device:[/bold green] {device}")

    use_graphs = _should_use_cuda_graphs(cfg["train"], device)
    requested_checkpoint = bool(cfg["train"].get("use_checkpoint", False))
    use_checkpoint = requested_checkpoint
    if use_graphs and use_checkpoint:
        console().print(
            "[yellow]train.use_checkpoint disabled because train.cuda_graphs is enabled.[/yellow]"
        )
        use_checkpoint = False
    cfg["train"]["use_checkpoint"] = bool(use_checkpoint)

    # --- data loading
    data_cfg = cfg.setdefault("data", {})
    time_feature_cfg_raw = data_cfg.get("time_features") or {}
    time_feature_cfg = dict(time_feature_cfg_raw)
    time_feature_cfg.setdefault("enabled", False)
    time_features_enabled = bool(time_feature_cfg.get("enabled", False))
    data_cfg["time_features"] = time_feature_cfg
    enc = cfg["data"]["encoding"]
    train_path = cfg["data"]["train_csv"]
    df = pd.read_csv(train_path, encoding=enc)
    schema = io_utils.DataSchema.from_config(data_cfg, sample_df=df)
    data_cfg.setdefault("schema", schema.as_dict())
    wide_raw = io_utils.pivot_long_to_wide(
        df,
        date_col=schema["date"],
        id_col=schema["id"],
        target_col=schema["target"],
        fill_missing_dates=cfg["data"]["fill_missing_dates"], fillna0=False
    )
    mask_wide = (~wide_raw.isna()).astype(np.float32)
    wide = wide_raw.fillna(0.0)
    series_static_np, static_feature_names = compute_series_features(wide, mask_wide)
    if cfg.get("preprocess", {}).get("clip_negative", False):
        wide = wide.clip(lower=0.0)
    ids = list(wide.columns)

    # --- normalization (FIT ONLY ON TRAIN PART to avoid leakage)
    # We'll split first to determine train-only fit.
    norm_method = cfg["preprocess"]["normalize"]
    norm_per_series = cfg["preprocess"]["normalize_per_series"]
    eps = cfg["preprocess"]["eps"]

    train_time_indices: List[pd.DatetimeIndex | None] | None = None
    val_time_indices: List[pd.DatetimeIndex | None] | None = None
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
        if time_features_enabled:
            train_time_indices = [pd.DatetimeIndex(trn_norm.index)]
            val_time_indices = [pd.DatetimeIndex(val_norm.index)]
        else:
            train_time_indices = None
            val_time_indices = None
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
        if time_features_enabled:
            train_time_indices = []
            val_time_indices = []
        else:
            train_time_indices = None
            val_time_indices = None
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
            if time_features_enabled:
                assert train_time_indices is not None and val_time_indices is not None
                train_time_indices.append(pd.DatetimeIndex(tr_df.index))
                val_time_indices.append(pd.DatetimeIndex(va_df.index))

    # --- model hyper-parameters derived from config
    cfg.setdefault("model", {})
    min_period_threshold = int(cfg["model"].get("min_period_threshold", 1))
    cfg["model"]["min_period_threshold"] = min_period_threshold

    # --- dataloaders
    input_len = int(window_cfg.input_len)
    pred_len = int(window_cfg.pred_len)
    cfg["model"]["input_len"] = input_len
    cfg["model"]["pred_len"] = pred_len
    mode = cfg["model"]["mode"]
    series_id_array = np.arange(len(ids), dtype=np.int64)
    train_static_list = [series_static_np] * len(train_arrays)
    val_static_list = [series_static_np] * len(val_arrays)
    train_id_list = [series_id_array] * len(train_arrays)
    val_id_list = [series_id_array] * len(val_arrays)
    dl_train = _build_dataloader(
        train_arrays, train_mask_arrays, input_len, pred_len, window_cfg.stride, mode, cfg["train"]["batch_size"],
        cfg["train"]["num_workers"], cfg["train"]["pin_memory"], cfg["train"]["persistent_workers"],
        cfg["train"]["prefetch_factor"], shuffle=True, drop_last=True,
        augment=cfg["data"].get("augment"),
        series_static=train_static_list,
        series_ids=train_id_list,
        time_indices=train_time_indices,
        time_feature_config=time_feature_cfg if time_features_enabled else None,
    )
    dl_val = _build_dataloader(
        val_arrays, val_mask_arrays, input_len, pred_len, window_cfg.stride, mode, batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"], pin_memory=cfg["train"]["pin_memory"],
        persistent_workers=cfg["train"]["persistent_workers"], prefetch_factor=cfg["train"]["prefetch_factor"],
        shuffle=False, drop_last=False,
        recursive_pred_len=(pred_len if mode == "recursive" else None),
        augment=None,
        series_static=val_static_list,
        series_ids=val_id_list,
        time_indices=val_time_indices,
        time_feature_config=time_feature_cfg if time_features_enabled else None,
    )
    if len(dl_val.dataset) == 0:
        raise ValueError("Validation split has no windows; increase train.val.holdout_days or adjust model.input_len/pred_len.")
    time_feature_dim = _time_feature_dim_from_dataset(dl_train.dataset)
    dataset_freq = _time_frequency_from_dataset(dl_train.dataset)
    base_index = wide.index if isinstance(wide.index, pd.DatetimeIndex) else None
    inferred_freq = dataset_freq
    if inferred_freq is None and base_index is not None:
        inferred_freq = getattr(base_index, "freqstr", None)
        if inferred_freq is None:
            inferred_freq = pd.infer_freq(base_index)
    cfg["data"]["time_features"]["feature_dim"] = int(time_feature_dim)
    if inferred_freq is not None:
        cfg["data"]["time_features"]["freq"] = inferred_freq
    time_feature_meta = {
        "enabled": bool(time_features_enabled and time_feature_dim > 0),
        "feature_dim": int(time_feature_dim),
        "config": dict(time_feature_cfg),
        "freq": inferred_freq,
    }
    freq_for_logging = inferred_freq
    warmup_series_static = torch.from_numpy(series_static_np).to(
        device=device, dtype=torch.float32
    )
    warmup_series_static_single: torch.Tensor | None
    if warmup_series_static.numel() > 0:
        warmup_series_static_single = warmup_series_static[:1, :]
    else:
        warmup_series_static_single = None
    series_ids_default = torch.from_numpy(series_id_array).to(
        device=device, dtype=torch.long
    )

    use_loss_masking = bool(cfg["train"].get("use_loss_masking", False))

    min_sigma_method = str(cfg["train"].get("min_sigma_method", "global"))
    target_std, per_series_std = _masked_std(
        train_arrays, train_mask_arrays, method=min_sigma_method
    )
    min_sigma_cfg = float(cfg["train"].get("min_sigma", 1e-3))
    min_sigma_scale = float(cfg["train"].get("min_sigma_scale", 0.1))
    scaled_min_sigma = target_std * min_sigma_scale if target_std > 0.0 else 0.0
    min_sigma_scalar = max(min_sigma_cfg, scaled_min_sigma)

    cfg.setdefault("train", {})
    per_series_floor_list: List[float] | None = None
    if per_series_std is not None and per_series_std.size > 0:
        per_series_scaled = np.asarray(per_series_std, dtype=np.float64) * min_sigma_scale
        per_series_scaled = np.maximum(per_series_scaled, min_sigma_scalar)
        per_series_floor_list = [float(x) for x in per_series_scaled]
        cfg["train"]["min_sigma_vector"] = per_series_floor_list
    else:
        cfg["train"].pop("min_sigma_vector", None)

    cfg["train"]["min_sigma_effective"] = float(min_sigma_scalar)
    msg = (
        "[bold green]min_sigma calibrated:[/bold green] "
        f"{min_sigma_scalar:.6f} (target std={target_std:.6f}, scale={min_sigma_scale})"
    )
    if per_series_floor_list:
        msg += f"  per-series max={max(per_series_floor_list):.6f}"
    console().print(msg)

    if per_series_floor_list is not None:
        min_sigma_vector_tensor = torch.tensor(
            per_series_floor_list, dtype=torch.float32
        ).view(1, 1, -1)
    else:
        min_sigma_vector_tensor = None

    # --- model
    model_cfg = cfg["model"]
    d_model = int(model_cfg["d_model"])
    d_ff = int(model_cfg.get("d_ff", 4 * d_model))
    model_cfg["d_ff"] = d_ff
    bottleneck_ratio = float(model_cfg.get("bottleneck_ratio", 1.0))
    model_cfg["bottleneck_ratio"] = bottleneck_ratio

    id_embed_dim = int(model_cfg.get("id_embed_dim", 32))
    static_proj_cfg = model_cfg.get("static_proj_dim", 32)
    static_proj_dim = None if static_proj_cfg is None else int(static_proj_cfg)
    static_layernorm = bool(model_cfg.get("static_layernorm", True))
    model_cfg["id_embed_dim"] = id_embed_dim
    model_cfg["static_proj_dim"] = static_proj_dim
    model_cfg["static_layernorm"] = static_layernorm

    model = TimesNet(
        input_len=input_len,
        pred_len=pred_len,
        d_model=d_model,
        d_ff=d_ff,
        n_layers=int(model_cfg["n_layers"]),
        k_periods=int(model_cfg["k_periods"]),
        min_period_threshold=min_period_threshold,
        kernel_set=list(model_cfg["kernel_set"]),
        dropout=float(model_cfg["dropout"]),
        activation=str(model_cfg["activation"]),
        mode=mode,
        bottleneck_ratio=bottleneck_ratio,
        channels_last=cfg["train"]["channels_last"],
        use_checkpoint=use_checkpoint,
        use_embedding_norm=bool(model_cfg.get("use_embedding_norm", True)),
        min_sigma=min_sigma_scalar,
        min_sigma_vector=min_sigma_vector_tensor,
        id_embed_dim=id_embed_dim,
        static_proj_dim=static_proj_dim,
        static_layernorm=static_layernorm,
    ).to(device)

    # Lazily build model parameters so that downstream utilities see them
    warmup_ids_single: torch.Tensor | None
    if series_ids_default.numel() > 0:
        # Use the maximum identifier so the embedding initializes capacity for all IDs.
        max_series_id = torch.max(series_ids_default)
        warmup_ids_single = max_series_id.reshape(1)
    else:
        warmup_ids_single = None
    warmup_kwargs = {
        "series_static": warmup_series_static_single,
        "series_ids": warmup_ids_single,
    }
    warmup_kwargs = {k: v for k, v in warmup_kwargs.items() if v is not None}
    if time_features_enabled and time_feature_dim > 0:
        warmup_kwargs["x_mark"] = torch.zeros(
            1,
            input_len,
            time_feature_dim,
            device=device,
            dtype=torch.float32,
        )
    original_min_sigma_buffer: torch.Tensor | None = None
    if (
        isinstance(model.min_sigma_vector, torch.Tensor)
        and model.min_sigma_vector.numel() > 0
    ):
        original_min_sigma_buffer = model.min_sigma_vector
        model.min_sigma_vector = model.min_sigma_vector[..., :1]

    with torch.no_grad():
        dummy = torch.zeros(1, input_len, 1, device=device)
        model(dummy, **warmup_kwargs)
        if cfg["train"]["channels_last"]:
            model.to(memory_format=torch.channels_last)
            dummy_cl = dummy.to(memory_format=torch.channels_last) if dummy.dim() == 4 else dummy
            model(dummy_cl, **warmup_kwargs)
    if cfg["train"]["compile"]:
        model = maybe_compile(
            model,
            True,
            warmup_args=(dummy,),
            warmup_kwargs=warmup_kwargs,
        )

    if original_min_sigma_buffer is not None:
        model.min_sigma_vector = original_min_sigma_buffer

    if isinstance(getattr(model, "min_sigma_vector", None), torch.Tensor) and model.min_sigma_vector.numel() > 0:
        min_sigma: float | torch.Tensor = model.min_sigma_vector
    else:
        min_sigma = min_sigma_scalar

    # --- optimizer / scheduler / loss
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    epochs = int(cfg["train"]["epochs"])
    accum_steps = max(int(cfg["train"].get("accumulation_steps", 1)), 1)

    batches_per_epoch = len(dl_train)
    updates_per_epoch = max(1, math.ceil(batches_per_epoch / accum_steps)) if batches_per_epoch > 0 else 1

    warmup_steps_cfg = cfg["train"].get("lr_warmup_steps")
    warmup_epochs_cfg = cfg["train"].get("lr_warmup_epochs")
    if warmup_steps_cfg is not None and warmup_epochs_cfg is not None:
        raise ValueError("Specify only one of train.lr_warmup_steps or train.lr_warmup_epochs")

    warmup_steps = 0
    warmup_epochs = 0
    if warmup_steps_cfg is not None:
        warmup_steps = max(int(warmup_steps_cfg), 0)
        if warmup_steps > 0:
            warmup_epochs = max(1, math.ceil(warmup_steps / updates_per_epoch)) if updates_per_epoch > 0 else warmup_steps
    elif warmup_epochs_cfg is not None:
        warmup_epochs = max(int(warmup_epochs_cfg), 0)
        warmup_steps = warmup_epochs * updates_per_epoch

    warmup_start_factor = 1.0
    warmup_length = warmup_steps if warmup_steps > 0 else warmup_epochs
    if warmup_length > 0:
        if warmup_length <= 1:
            warmup_start_factor = 0.5
        else:
            warmup_start_factor = max(1e-4, min(1.0, 1.0 / warmup_length))

    cfg["train"]["lr_warmup_steps_effective"] = warmup_steps
    cfg["train"]["lr_warmup_epochs_effective"] = warmup_epochs
    cfg["train"]["lr_warmup_start_factor_effective"] = warmup_start_factor

    sched_cfg = cfg["train"].get("lr_scheduler", {})
    scheduler: torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
    sched_type = sched_cfg.get("type") or "cosine"
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
        cosine_t_max = max(1, t_max - warmup_epochs) if warmup_epochs > 0 else t_max
        cosine_sched = CosineAnnealingLR(
            optim,
            T_max=cosine_t_max,
            eta_min=float(sched_cfg.get("eta_min", 1e-5)),
        )
        scheduler = cosine_sched
        if warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optim,
                start_factor=warmup_start_factor,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optim,
                schedulers=[warmup_scheduler, cosine_sched],
                milestones=[warmup_epochs],
            )
    elif warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optim,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        scheduler = warmup_scheduler

    if warmup_epochs > 0 and sched_type == "ReduceLROnPlateau":
        console().print(
            "[yellow]Warmup requested but not supported with ReduceLROnPlateau scheduler; skipping warmup.[/yellow]"
        )
        cfg["train"]["lr_warmup_steps_effective"] = 0
        cfg["train"]["lr_warmup_epochs_effective"] = 0
        cfg["train"]["lr_warmup_start_factor_effective"] = 1.0
    elif sched_type == "cosine" and warmup_epochs > 0:
        warmup_lr = float(cfg["train"]["lr"]) * warmup_start_factor
        for param_group in optim.param_groups:
            param_group["lr"] = warmup_lr
        if scheduler is not None and hasattr(scheduler, "_last_lr"):
            scheduler._last_lr = [warmup_lr for _ in scheduler._last_lr]
        if isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR):
            for sub_scheduler in scheduler._schedulers:
                if hasattr(sub_scheduler, "_last_lr"):
                    sub_scheduler._last_lr = [warmup_lr for _ in sub_scheduler._last_lr]

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
    if use_graphs:
        warmup_iters = 3
        train_iter = iter(dl_train)
        graph_captured = False
        static_series_buf: torch.Tensor | None = None
        static_ids_buf: torch.Tensor | None = None
        static_mark_buf: torch.Tensor | None = None
        for w in range(warmup_iters):
            try:
                batch_w = next(train_iter)
            except StopIteration:
                break
            xb_w, yb_w, mask_w, x_mark_w, y_mark_w, static_w, series_ids_w = _unpack_batch(batch_w)
            xb_w = xb_w.to(device, non_blocking=True)
            yb_w = yb_w.to(device, non_blocking=True)
            if use_loss_masking:
                mask_w = mask_w.to(device, non_blocking=True)
                base_mask_w = mask_w > 0.0
            else:
                base_mask_w = torch.ones_like(yb_w, dtype=torch.bool, device=device)
            if x_mark_w is not None:
                x_mark_w = x_mark_w.to(device=device, non_blocking=True)
            if y_mark_w is not None:
                y_mark_w = y_mark_w.to(device=device, non_blocking=True)
            if static_w is not None:
                static_w = static_w.to(device=device, non_blocking=True)
            if series_ids_w is not None:
                series_ids_w = series_ids_w.to(
                    device=device, dtype=torch.long, non_blocking=True
                )
            else:
                series_ids_w = series_ids_default
            if cfg["train"]["channels_last"] and xb_w.dim() == 4:
                xb_w = xb_w.to(memory_format=torch.channels_last)
            _assert_min_len(xb_w, _model_input_len(model))
            with amp_autocast(cfg["train"]["amp"] and device.type == "cuda"):
                rate_w, dispersion_w = model(
                    xb_w,
                    x_mark=x_mark_w,
                    series_static=static_w,
                    series_ids=series_ids_w,
                )
                if w == 0:
                    _log_selected_periods(model, freq_for_logging, 0, w + 1)
                nb_mask_w = negative_binomial_mask(yb_w, rate_w, dispersion_w, base_mask_w)
                loss_w = (
                    negative_binomial_nll(
                        y=yb_w,
                        rate=rate_w,
                        dispersion=dispersion_w,
                        mask=nb_mask_w,
                    )
                    / accum_steps
                )
            grad_scaler.scale(loss_w).backward()
            if (w + 1) % accum_steps == 0:
                if grad_clip and grad_clip > 0:
                    grad_scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                grad_scaler.step(optim)
                grad_scaler.update()
                optim.zero_grad(set_to_none=True)

        try:
            batch0 = next(train_iter)
        except StopIteration:
            train_iter = iter(dl_train)
            batch0 = next(train_iter)
        xb0, yb0, mask0, x_mark0, y_mark0, static0, series_ids0 = _unpack_batch(batch0)
        xb0 = xb0.to(device, non_blocking=True)
        yb0 = yb0.to(device, non_blocking=True)
        if use_loss_masking:
            mask0 = mask0.to(device, non_blocking=True)
            base_mask0 = mask0 > 0.0
        else:
            base_mask0 = torch.ones_like(yb0, dtype=torch.bool, device=device)
        if x_mark0 is not None:
            x_mark0 = x_mark0.to(device=device, non_blocking=True)
        if y_mark0 is not None:
            y_mark0 = y_mark0.to(device=device, non_blocking=True)
        if static0 is not None:
            static0 = static0.to(device=device, non_blocking=True)
        if series_ids0 is not None:
            series_ids0 = series_ids0.to(
                device=device, dtype=torch.long, non_blocking=True
            )
        else:
            series_ids0 = series_ids_default
        if cfg["train"]["channels_last"] and xb0.dim() == 4:
            xb0 = xb0.to(memory_format=torch.channels_last)
        static_x = torch.empty_like(xb0)
        static_y = torch.empty_like(yb0)
        static_m = torch.empty_like(base_mask0)
        static_x.copy_(xb0)
        static_y.copy_(yb0)
        static_m.copy_(base_mask0)
        if x_mark0 is not None:
            static_mark_buf = torch.empty_like(x_mark0)
            static_mark_buf.copy_(x_mark0)
        else:
            static_mark_buf = None
        if static0 is not None:
            static_series_buf = torch.empty_like(static0)
            static_series_buf.copy_(static0)
        if series_ids0 is not None:
            static_ids_buf = torch.empty_like(series_ids0)
            static_ids_buf.copy_(series_ids0)
        capture_stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        mask_stats_buf = torch.zeros(1, dtype=torch.float64, device=device)
        optim.zero_grad(set_to_none=True)
        model.eval()
        with torch.cuda.stream(capture_stream):
            graph.capture_begin()
            with amp_autocast(cfg["train"]["amp"] and device.type == "cuda"):
                _assert_min_len(static_x, _model_input_len(model))
                static_rate, static_dispersion = model(
                    static_x,
                    x_mark=static_mark_buf,
                    series_static=static_series_buf,
                    series_ids=static_ids_buf,
                )
                static_nb_mask = negative_binomial_mask(
                    static_y, static_rate, static_dispersion, static_m
                )
                static_loss = negative_binomial_nll(
                    y=static_y,
                    rate=static_rate,
                    dispersion=static_dispersion,
                    mask=static_nb_mask,
                )
                mask_stats_buf[0] = static_nb_mask.sum().to(mask_stats_buf.dtype)
                static_scaled = static_loss / accum_steps
            grad_scaler.scale(static_scaled).backward()
            graph.capture_end()
            graph_captured = True
        torch.cuda.current_stream().wait_stream(capture_stream)
        model.train()
        optim.zero_grad(set_to_none=True)
        assert graph_captured, "CUDA graph capture did not complete successfully"
        console().print("[green]CUDA graph captured successfully.[/green]")

        def graph_step(
            xb: torch.Tensor,
            yb: torch.Tensor,
            base_mask: torch.Tensor,
            x_mark: torch.Tensor | None,
            static_feat: torch.Tensor | None,
            series_idx: torch.Tensor,
        ) -> tuple[float, float, float]:
            static_x.copy_(xb)
            static_y.copy_(yb)
            static_m.copy_(base_mask)
            if static_mark_buf is not None:
                if x_mark is None:
                    raise RuntimeError(
                        "x_mark must be provided during CUDA graph replay when time features are enabled"
                    )
                static_mark_buf.copy_(x_mark)
            elif x_mark is not None:
                raise RuntimeError(
                    "x_mark provided but CUDA graph captured without temporal buffer"
                )
            if static_series_buf is not None:
                if static_feat is None:
                    raise RuntimeError(
                        "series_static buffer captured but no features provided"
                    )
                static_series_buf.copy_(static_feat)
            elif static_feat is not None:
                raise RuntimeError(
                    "series_static provided but CUDA graph captured without buffer"
                )
            if static_ids_buf is not None:
                static_ids_buf.copy_(series_idx)
            graph.replay()
            mask_true = float(mask_stats_buf[0].item())
            mask_total = float(static_y.numel())
            return float(static_loss.item()), mask_true, mask_total

    print_config(cfg, current_lr=optim.param_groups[0]["lr"])
    for ep in range(1, epochs + 1):
        model.train()
        losses: List[float] = []
        optim.zero_grad(set_to_none=True)
        num_batches = len(dl_train)
        copy_time_total = 0.0
        iter_time_total = 0.0
        mask_true_total = 0.0
        mask_total = 0.0
        for i, batch in enumerate(tqdm(dl_train, desc=f"Epoch {ep}/{epochs}", leave=False)):
            iter_start = time.perf_counter()
            xb, yb, mask, x_mark, y_mark, static_feat, series_idx = _unpack_batch(batch)
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if use_loss_masking:
                mask = mask.to(device, non_blocking=True)
                base_mask_batch = mask > 0.0
            else:
                base_mask_batch = torch.ones_like(yb, dtype=torch.bool, device=device)
            if x_mark is not None:
                x_mark = x_mark.to(device=device, non_blocking=True)
            if y_mark is not None:
                y_mark = y_mark.to(device=device, non_blocking=True)
            if static_feat is not None:
                static_feat = static_feat.to(device=device, non_blocking=True)
            if series_idx is not None:
                series_idx = series_idx.to(
                    device=device, dtype=torch.long, non_blocking=True
                )
            else:
                series_idx = series_ids_default
            if cfg["train"]["channels_last"] and xb.dim() == 4:
                xb = xb.to(memory_format=torch.channels_last)
            _assert_min_len(xb, _model_input_len(model))
            after_copy = time.perf_counter()
            if use_graphs:
                loss_val, mask_true_inc, mask_total_inc = graph_step(
                    xb, yb, base_mask_batch, x_mark, static_feat, series_idx
                )
                mask_true_total += mask_true_inc
                mask_total += mask_total_inc
                if i == 0:
                    _log_selected_periods(model, freq_for_logging, ep, i + 1)
            else:
                with amp_autocast(cfg["train"]["amp"] and device.type == "cuda"):
                    rate, dispersion = model(
                        xb,
                        x_mark=x_mark,
                        series_static=static_feat,
                        series_ids=series_idx,
                    )
                    if i == 0:
                        _log_selected_periods(model, freq_for_logging, ep, i + 1)
                    nb_mask_batch = negative_binomial_mask(
                        yb, rate, dispersion, base_mask_batch
                    )
                    loss_value = negative_binomial_nll(
                        y=yb,
                        rate=rate,
                        dispersion=dispersion,
                        mask=nb_mask_batch,
                    )
                    loss = loss_value / accum_steps
                grad_scaler.scale(loss).backward()
                loss_val = float(loss_value.item())
                mask_true_total += float(nb_mask_batch.sum().item())
                mask_total += float(nb_mask_batch.numel())
            iter_end = time.perf_counter()
            copy_time_total += after_copy - iter_start
            iter_time_total += max(iter_end - iter_start, 1e-12)
            if (i + 1) % accum_steps == 0 or (i + 1) == num_batches:
                if grad_clip and grad_clip > 0:
                    grad_scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                grad_scaler.step(optim)
                grad_scaler.update()
                optim.zero_grad(set_to_none=True)
            losses.append(float(loss_val))

        if num_batches > 0 and iter_time_total > 0.0:
            overhead_pct = (copy_time_total / iter_time_total) * 100.0
            console().print(
                (
                    f"[cyan]Epoch {ep} data prep overhead: {overhead_pct:.2f}% "
                    f"(prep {copy_time_total:.4f}s / iter {iter_time_total:.4f}s)[/cyan]"
                )
            )

        if mask_total > 0.0:
            coverage = mask_true_total / mask_total
        else:
            coverage = 0.0
        console().print(f"[blue]Epoch {ep} loss mask coverage: {coverage:.4f}[/blue]")

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
    signature_path = os.path.join(art_dir, cfg["artifacts"]["signature_file"])
    normalization_meta = {
        "method": cfg["preprocess"]["normalize"],
        "per_series": cfg["preprocess"]["normalize_per_series"],
        "eps": cfg["preprocess"]["eps"],
    }
    io_utils.save_pickle(
        {
            "scaler": scaler,
            "method": normalization_meta["method"],
            "ids": ids,
            "static_features": series_static_np,
            "feature_names": static_feature_names,
            "time_features": time_feature_meta,
        },
        scaler_path,
    )
    io_utils.save_schema_artifact(
        schema_path,
        schema,
        normalization=normalization_meta,
        extras={"time_features": time_feature_meta},
    )
    save_yaml(cfg, cfg_path)
    preprocess_signature = dict(normalization_meta)
    preprocess_signature["schema_artifact_version"] = io_utils.SCHEMA_ARTIFACT_VERSION
    static_feature_dim = (
        int(series_static_np.shape[1])
        if isinstance(series_static_np, np.ndarray) and series_static_np.size > 0
        else 0
    )
    signature_payload = {
        "signature_version": 1,
        "window": window_cfg.to_dict(),
        "model": {
            "mode": str(cfg["model"]["mode"]),
            "d_model": int(cfg["model"]["d_model"]),
            "d_ff": int(cfg["model"]["d_ff"]),
            "n_layers": int(cfg["model"]["n_layers"]),
            "k_periods": int(cfg["model"]["k_periods"]),
            "min_period_threshold": int(cfg["model"]["min_period_threshold"]),
            "id_embed_dim": int(cfg["model"]["id_embed_dim"]),
            "static_proj_dim": (
                None if cfg["model"]["static_proj_dim"] is None else int(cfg["model"]["static_proj_dim"])
            ),
        },
        "train": {
            "batch_size": int(cfg["train"]["batch_size"]),
            "channels_last": bool(cfg["train"]["channels_last"]),
            "use_checkpoint": bool(cfg["train"]["use_checkpoint"]),
            "min_sigma_effective": float(cfg["train"].get("min_sigma_effective", 0.0)),
            "min_sigma_method": cfg["train"].get("min_sigma_method"),
            "min_sigma_scale": float(cfg["train"].get("min_sigma_scale", 0.0)),
        },
        "data": {
            "num_series": len(ids),
            "static_feature_dim": static_feature_dim,
            "time_feature_dim": int(time_feature_dim),
            "time_features_enabled": bool(time_features_enabled and time_feature_dim > 0),
            "time_feature_freq": inferred_freq,
        },
        "preprocess": preprocess_signature,
    }
    io_utils.save_json(signature_payload, signature_path)
    console().print(
        f"[green]Saved:[/green] {model_path}, {scaler_path}, {schema_path}, {cfg_path}, {signature_path}"
    )
    return best_nll, {
        "model": model_path,
        "scaler": scaler_path,
        "schema": schema_path,
        "config": cfg_path,
        "metrics": {"nll": best_nll, "smape": best_smape},
    }


def main() -> None:
    import argparse
    from .config import PipelineConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()
    cfg = PipelineConfig.from_files(args.config, overrides=args.override)
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
