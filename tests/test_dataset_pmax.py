import sys
from pathlib import Path

import numpy as np

# Ensure the project src is on the path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.data.dataset import SlidingWindowDataset


def test_sliding_window_left_pads_to_pmax():
    values = np.arange(1, 8, dtype=np.float32).reshape(-1, 1)
    ds = SlidingWindowDataset(
        values,
        input_len=3,
        pred_len=1,
        mode="direct",
        recursive_pred_len=None,
        augment=None,
        pmax_global=5,
    )

    x0, y0, m0 = ds[0]
    assert x0.shape == (5, 1)
    assert y0.shape == (1, 1)
    np.testing.assert_allclose(x0.squeeze(-1).numpy(), np.array([0.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(y0.squeeze(-1).numpy(), np.array([4.0], dtype=np.float32))
    np.testing.assert_allclose(m0.squeeze(-1).numpy(), np.array([1.0], dtype=np.float32))

    x_last, _, m_last = ds[len(ds) - 1]
    assert x_last.shape == (5, 1)
    np.testing.assert_allclose(
        x_last.squeeze(-1).numpy(),
        np.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
    )
    np.testing.assert_allclose(m_last.squeeze(-1).numpy(), np.array([1.0], dtype=np.float32))


def test_sliding_window_truncates_to_pmax():
    values = np.arange(1, 8, dtype=np.float32).reshape(-1, 1)
    ds = SlidingWindowDataset(
        values,
        input_len=4,
        pred_len=1,
        mode="direct",
        recursive_pred_len=None,
        augment=None,
        pmax_global=2,
    )

    x0, y0, m0 = ds[0]
    assert x0.shape == (2, 1)
    np.testing.assert_allclose(x0.squeeze(-1).numpy(), np.array([3.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(y0.squeeze(-1).numpy(), np.array([5.0], dtype=np.float32))
    np.testing.assert_allclose(m0.squeeze(-1).numpy(), np.array([1.0], dtype=np.float32))

    x1, _, m1 = ds[1]
    np.testing.assert_allclose(x1.squeeze(-1).numpy(), np.array([4.0, 5.0], dtype=np.float32))
    np.testing.assert_allclose(m1.squeeze(-1).numpy(), np.array([1.0], dtype=np.float32))
