import sys
from pathlib import Path

import numpy as np

# Ensure the project src is on the path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.data.dataset import SlidingWindowDataset


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
    x0, y0, m0 = ds[0]
    assert x0.shape == (3, 1)
    assert y0.shape == (2, 1)
    assert m0.shape == (2, 1)
    np.testing.assert_allclose(x0.squeeze(-1).numpy(), np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(y0.squeeze(-1).numpy(), np.array([4.0, 5.0], dtype=np.float32))


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

    x_last, y_last, m_last = ds[len(ds) - 1]
    np.testing.assert_allclose(x_last.squeeze(-1).numpy(), np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32))
    np.testing.assert_allclose(y_last.squeeze(-1).numpy(), np.array([7.0, 8.0, 9.0], dtype=np.float32))
    np.testing.assert_allclose(m_last.squeeze(-1).numpy(), np.array([1.0, 1.0, 1.0], dtype=np.float32))
