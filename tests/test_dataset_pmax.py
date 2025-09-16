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

    x0, y0, sid0 = ds[0]
    assert sid0 == 0
    assert x0.shape == (5, 1)
    assert y0.shape == (1, 1)
    np.testing.assert_allclose(x0.squeeze(-1).numpy(), np.array([0.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(y0.squeeze(-1).numpy(), np.array([4.0], dtype=np.float32))

    x_last, _, sid_last = ds[len(ds) - 1]
    assert sid_last == 0
    assert x_last.shape == (5, 1)
    np.testing.assert_allclose(
        x_last.squeeze(-1).numpy(),
        np.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
    )


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

    x0, y0, sid0 = ds[0]
    assert sid0 == 0
    assert x0.shape == (2, 1)
    np.testing.assert_allclose(x0.squeeze(-1).numpy(), np.array([3.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(y0.squeeze(-1).numpy(), np.array([5.0], dtype=np.float32))

    x1, _, sid1 = ds[1]
    assert sid1 == 0
    np.testing.assert_allclose(x1.squeeze(-1).numpy(), np.array([4.0, 5.0], dtype=np.float32))


def test_min_history_filters_zero_padded_windows():
    values = np.array(
        [
            [0.0, 0.0],
            [0.0, 5.0],
            [1.0, 6.0],
            [2.0, 7.0],
            [3.0, 8.0],
            [4.0, 9.0],
        ],
        dtype=np.float32,
    )
    ds = SlidingWindowDataset(
        values,
        input_len=3,
        pred_len=1,
        mode="direct",
        recursive_pred_len=None,
        augment=None,
        pmax_global=4,
        min_history_for_training=3,
    )

    assert len(ds) == 3
    # The valid windows should correspond to series 1 at start=1, and both series at start=2.
    sample0 = ds[0]
    assert sample0[2] == 1  # series index
    np.testing.assert_array_equal(sample0[0].numpy().squeeze(-1), np.array([0.0, 5.0, 6.0, 7.0], dtype=np.float32))

    sample1 = ds[1]
    assert sample1[2] == 0
    np.testing.assert_array_equal(sample1[0].numpy().squeeze(-1), np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32))

    sample2 = ds[2]
    assert sample2[2] == 1
    np.testing.assert_array_equal(sample2[0].numpy().squeeze(-1), np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32))
