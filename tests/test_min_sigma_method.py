import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the project src is on the path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.train import _masked_std


def _collect_valid_values(arrays, masks):
    values = []
    for arr, mask in zip(arrays, masks):
        if arr.size == 0:
            continue
        if mask is None:
            values.append(arr.reshape(-1))
        else:
            valid = mask > 0.0
            if np.any(valid):
                values.append(arr[valid])
    if not values:
        return np.array([], dtype=np.float64)
    return np.concatenate([v.astype(np.float64, copy=False) for v in values])


def test_masked_std_global_matches_numpy():
    arrays = [
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        np.array([[7.0, 8.0], [9.0, 10.0]], dtype=np.float32),
    ]
    masks = [
        np.array([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32),
    ]

    summary, per_series = _masked_std(arrays, masks, method="global")

    valid = _collect_valid_values(arrays, masks)
    expected = 0.0 if valid.size == 0 else float(np.std(valid))

    assert summary == pytest.approx(expected)
    assert per_series is None


def test_masked_std_per_series_median_is_robust():
    arrays = [
        np.array(
            [
                [1.0, 10.0, 1.0],
                [2.0, 10.0, 1.0],
                [3.0, 10.0, 1.0],
                [4.0, 10.0, 1.0],
                [100.0, 10.0, 100.0],
            ],
            dtype=np.float32,
        )
    ]
    masks = [
        np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
    ]

    global_std, _ = _masked_std(arrays, masks, method="global")
    robust_std, per_series = _masked_std(arrays, masks, method="per_series_median")

    assert per_series is not None
    assert per_series.shape[0] == arrays[0].shape[1]

    per_series_values = []
    arr = arrays[0]
    mask = masks[0] > 0.0
    for j in range(arr.shape[1]):
        col_values = arr[mask[:, j], j]
        if col_values.size > 0:
            per_series_values.append(np.std(col_values.astype(np.float64, copy=False)))
    expected_robust = 0.0 if not per_series_values else float(np.median(per_series_values))

    assert robust_std == pytest.approx(expected_robust)
    assert robust_std < global_std


def test_per_series_floor_retains_high_variance_channels():
    arrays = [
        np.array(
            [
                [0.0, 0.00, 0.01],
                [10.0, 0.05, 0.02],
                [20.0, 0.10, 0.03],
                [30.0, 0.15, 0.04],
            ],
            dtype=np.float32,
        )
    ]
    masks = [np.ones_like(arrays[0], dtype=np.float32)]

    summary, per_series = _masked_std(arrays, masks, method="per_series_median")
    assert per_series is not None
    scale = 0.2
    min_sigma_cfg = 1e-3

    scalar_floor = max(min_sigma_cfg, summary * scale)
    per_series_floor = np.maximum(per_series * scale, scalar_floor)

    high_idx = int(np.argmax(per_series))
    assert per_series_floor[high_idx] > scalar_floor * 10
    assert np.all(per_series_floor >= scalar_floor)

