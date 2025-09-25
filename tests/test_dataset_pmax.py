import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Ensure the project src is on the path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.data.dataset import SlidingWindowDataset
from timesnet_forecast.utils.time_features import build_time_features


def test_sliding_window_returns_raw_input_len():
    values = np.arange(1, 9, dtype=np.float32).reshape(-1, 1)
    ds = SlidingWindowDataset(
        values,
        input_len=3,
        pred_len=2,
        mode="direct",
        recursive_pred_len=None,
        augment=None,
    )

    assert len(ds) == len(values) - 3 - 2 + 1
    x0, y0, m0, x_mark, y_mark = ds[0][:5]
    assert x0.shape == (3, 1)
    assert y0.shape == (2, 1)
    assert m0.shape == (2, 1)
    np.testing.assert_allclose(x0.squeeze(-1).numpy(), np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(y0.squeeze(-1).numpy(), np.array([4.0, 5.0], dtype=np.float32))
    assert isinstance(x_mark, torch.Tensor) and x_mark.numel() == 0
    assert isinstance(y_mark, torch.Tensor) and y_mark.numel() == 0


def test_sliding_window_last_chunk_matches_source():
    values = np.arange(10, dtype=np.float32).reshape(-1, 1)
    mask = np.ones_like(values, dtype=np.float32)
    ds = SlidingWindowDataset(
        values,
        input_len=4,
        pred_len=3,
        mode="direct",
        recursive_pred_len=None,
        augment=None,
        valid_mask=mask,
    )

    x_last, y_last, m_last, x_mark, y_mark = ds[len(ds) - 1][:5]
    np.testing.assert_allclose(x_last.squeeze(-1).numpy(), np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32))
    np.testing.assert_allclose(y_last.squeeze(-1).numpy(), np.array([7.0, 8.0, 9.0], dtype=np.float32))
    np.testing.assert_allclose(m_last.squeeze(-1).numpy(), np.array([1.0, 1.0, 1.0], dtype=np.float32))
    assert isinstance(x_mark, torch.Tensor) and x_mark.numel() == 0
    assert isinstance(y_mark, torch.Tensor) and y_mark.numel() == 0


def test_sliding_window_static_features_collate_shape():
    values = np.arange(20, dtype=np.float32).reshape(-1, 2)
    mask = np.ones_like(values, dtype=np.float32)
    static = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    series_ids = np.array([0, 1], dtype=np.int64)
    ds = SlidingWindowDataset(
        values,
        input_len=3,
        pred_len=2,
        mode="direct",
        recursive_pred_len=None,
        augment=None,
        valid_mask=mask,
        series_static=static,
        series_ids=series_ids,
    )

    x0, y0, m0, x_mark, y_mark, s0, ids0 = ds[0]
    assert x0.dtype == torch.float32
    assert y0.dtype == torch.float32
    assert m0.dtype == torch.float32
    assert s0.dtype == torch.float32
    assert ids0.dtype == torch.long
    assert s0.shape == (1, static.shape[1])
    assert ids0.shape == (1,)
    assert isinstance(x_mark, torch.Tensor) and x_mark.numel() == 0
    assert isinstance(y_mark, torch.Tensor) and y_mark.numel() == 0

    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)
    xb, yb, mb, xmb, ymb, sb, idb = next(iter(loader))
    assert xb.shape == (2, 3, 1)
    assert yb.shape == (2, 2, 1)
    assert sb.shape == (2, 1, static.shape[1])
    assert idb.shape == (2, 1)
    assert isinstance(xmb, torch.Tensor) and xmb.numel() == 0
    assert isinstance(ymb, torch.Tensor) and ymb.numel() == 0


def test_sliding_window_series_are_isolated_per_sample():
    values = np.stack(
        [
            np.linspace(1.0, 6.0, num=6, dtype=np.float32),
            np.linspace(10.0, 60.0, num=6, dtype=np.float32),
        ],
        axis=1,
    )
    ds = SlidingWindowDataset(
        values,
        input_len=2,
        pred_len=1,
        mode="direct",
        augment=None,
    )

    x_first, y_first, *_ = ds[0]
    x_second, y_second, *_ = ds[1]
    np.testing.assert_allclose(x_first.squeeze(-1).numpy(), [1.0, 2.0])
    np.testing.assert_allclose(y_first.squeeze(-1).numpy(), [3.0])
    np.testing.assert_allclose(x_second.squeeze(-1).numpy(), [10.0, 20.0])
    np.testing.assert_allclose(y_second.squeeze(-1).numpy(), [30.0])

    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)
    xb, _, _, _, _ = next(iter(loader))[:5]
    assert xb.shape == (2, 2, 1)
    np.testing.assert_allclose(xb[0].squeeze(-1).numpy(), [1.0, 2.0])
    np.testing.assert_allclose(xb[1].squeeze(-1).numpy(), [10.0, 20.0])


def test_dataframe_loader_preserves_single_series_shape():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    records = []
    for name, values in {"A": [1, 2, 3, 4], "B": [10, 11, 12, 13]}.items():
        for d, v in zip(dates, values):
            records.append({"date": d, "id": name, "target": v})
    df = pd.DataFrame(records)
    wide = df.pivot(index="date", columns="id", values="target").sort_index(axis=1)
    values = wide.to_numpy(dtype=np.float32)
    ds = SlidingWindowDataset(values, input_len=2, pred_len=1, mode="direct")

    x_sample, y_sample, *_ = ds[0]
    assert x_sample.shape == (2, 1)
    assert y_sample.shape == (1, 1)


def test_sliding_window_time_features_marks():
    idx = pd.date_range("2023-01-01", periods=8, freq="D")
    values = np.arange(8, dtype=np.float32).reshape(-1, 1)
    cfg = {"enabled": True, "features": ["day_of_week"], "encoding": "numeric", "normalize": False}
    ds = SlidingWindowDataset(
        values,
        input_len=3,
        pred_len=2,
        mode="direct",
        time_index=idx,
        time_feature_config=cfg,
    )
    x0, y0, m0, x_mark, y_mark = ds[0][:5]
    assert isinstance(x_mark, torch.Tensor)
    assert isinstance(y_mark, torch.Tensor)
    features = build_time_features(idx, cfg)
    np.testing.assert_allclose(x_mark.numpy(), features[:3].reshape(3, -1))
    np.testing.assert_allclose(y_mark.numpy(), features[3:5].reshape(2, -1))


def test_sliding_window_time_features_collate_shapes():
    idx = pd.date_range("2023-02-01", periods=10, freq="D")
    values = np.linspace(0.0, 1.0, num=10, dtype=np.float32).reshape(-1, 1)
    cfg = {"enabled": True, "features": ["day_of_month"], "encoding": "numeric", "normalize": True}
    ds = SlidingWindowDataset(
        values,
        input_len=4,
        pred_len=2,
        mode="direct",
        time_index=idx,
        time_feature_config=cfg,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)
    xb, yb, mb, xmb, ymb = next(iter(loader))[:5]
    assert xmb.shape == (2, 4, 1)
    assert ymb.shape == (2, 2, 1)
