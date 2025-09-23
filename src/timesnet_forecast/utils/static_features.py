from __future__ import annotations

import numpy as np
import pandas as pd

_F32_EPS = np.float32(1e-6)


def _safe_divide(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """Element-wise division with float32 outputs and EPS-denominator clamp."""

    denom_safe = np.maximum(denom.astype(np.float32, copy=False), _F32_EPS)
    result = numer.astype(np.float32, copy=False) / denom_safe
    return result.astype(np.float32, copy=False)


def compute_series_features(
    wide_df: pd.DataFrame, mask_df: pd.DataFrame
) -> tuple[np.ndarray, list[str]]:
    """Compute per-series static features for wide-form time series data.

    Args:
        wide_df: Wide-form dataframe shaped ``[T, N]`` with series in columns.
        mask_df: DataFrame with the same shape as ``wide_df`` indicating valid
            observations (``>0``) for each timestamp and series.

    Returns:
        Tuple ``(features, feature_names)`` where ``features`` is an array shaped
        ``[N, F]`` containing ``float32`` features and ``feature_names`` lists the
        column names in ``features``.
    """

    if wide_df.shape != mask_df.shape:
        raise ValueError("wide_df and mask_df must have the same shape")

    values = wide_df.to_numpy(dtype=np.float32, copy=False)
    mask = mask_df.to_numpy(dtype=np.float32, copy=False)
    time_steps, num_series = values.shape

    feature_names = [
        "mean",
        "std",
        "diff_std",
        "seasonal_strength",
        "dominant_period",
    ]

    if num_series == 0:
        return np.zeros((0, len(feature_names)), dtype=np.float32), feature_names

    counts = mask.sum(axis=0, dtype=np.float32)
    sum_vals = (values * mask).sum(axis=0, dtype=np.float32)
    mean = _safe_divide(sum_vals, counts)

    centered = (values - mean[np.newaxis, :]) * mask
    var = _safe_divide(
        (centered * centered).sum(axis=0, dtype=np.float32),
        np.maximum(counts, np.float32(1.0)),
    )
    std = np.sqrt(np.clip(var, 0.0, None)).astype(np.float32)

    if time_steps > 1:
        diffs = values[1:, :] - values[:-1, :]
        diff_mask = mask[1:, :] * mask[:-1, :]
        diff_counts = diff_mask.sum(axis=0, dtype=np.float32)
        diff_sum = (diffs * diff_mask).sum(axis=0, dtype=np.float32)
        diff_mean = _safe_divide(diff_sum, diff_counts)
        diff_centered = (diffs - diff_mean[np.newaxis, :]) * diff_mask
        diff_var = _safe_divide(
            (diff_centered * diff_centered).sum(axis=0, dtype=np.float32),
            np.maximum(diff_counts, np.float32(1.0)),
        )
        diff_std = np.sqrt(np.clip(diff_var, 0.0, None)).astype(np.float32)
    else:
        diff_std = np.zeros(num_series, dtype=np.float32)

    if time_steps > 1:
        demeaned = np.where(mask > 0.0, values - mean[np.newaxis, :], 0.0)
        fft_vals = np.fft.rfft(demeaned, axis=0)
        power = np.abs(fft_vals) ** 2
        if power.shape[0] > 1:
            power_no_dc = power[1:, :]
            peak_indices = np.argmax(power_no_dc, axis=0)
            cols = np.arange(num_series)
            peak_power = power_no_dc[peak_indices, cols]
            total_power = power_no_dc.sum(axis=0)
            seasonal_strength = _safe_divide(peak_power, total_power)
            dominant_period = np.where(
                total_power > _F32_EPS,
                (time_steps / np.maximum(peak_indices + 1, 1)).astype(np.float32),
                0.0,
            ).astype(np.float32)
        else:
            seasonal_strength = np.zeros(num_series, dtype=np.float32)
            dominant_period = np.zeros(num_series, dtype=np.float32)
    else:
        seasonal_strength = np.zeros(num_series, dtype=np.float32)
        dominant_period = np.zeros(num_series, dtype=np.float32)

    features = np.stack(
        [mean, std, diff_std, seasonal_strength, dominant_period], axis=1
    ).astype(np.float32, copy=False)
    return features, feature_names
