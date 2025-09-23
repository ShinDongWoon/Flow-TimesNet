from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.utils.static_features import compute_series_features


def test_compute_series_features_small_sample():
    T = 12
    t = np.arange(T, dtype=np.float32)
    series_a = 5.0 + np.sin(2 * np.pi * t / 3.0)
    series_b = np.full(T, 2.0, dtype=np.float32)
    series_c = np.array(
        [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, np.nan, 8.0, 9.0, 10.0, np.nan, 12.0],
        dtype=np.float32,
    )

    wide_raw = pd.DataFrame({"A": series_a, "B": series_b, "C": series_c})
    mask_df = (~wide_raw.isna()).astype(np.float32)
    wide = wide_raw.fillna(0.0)

    features, names = compute_series_features(wide, mask_df)

    assert names == [
        "mean",
        "std",
        "diff_std",
        "seasonal_strength",
        "dominant_period",
    ]
    assert features.shape == (3, len(names))
    assert features.dtype == np.float32

    expected_mean_a = float(series_a.mean())
    expected_std_a = float(series_a.std(ddof=0))
    expected_diff_std_a = float(np.diff(series_a).std(ddof=0))

    valid_c = wide_raw["C"].dropna().to_numpy(dtype=np.float32)
    expected_mean_c = float(valid_c.mean())
    expected_std_c = float(valid_c.std(ddof=0))

    np.testing.assert_allclose(features[0, 0], expected_mean_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(features[0, 1], expected_std_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        features[0, 2], expected_diff_std_a, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(features[0, 4], 3.0, rtol=1e-5, atol=1e-5)
    assert 0.9 <= float(features[0, 3]) <= 1.0

    np.testing.assert_allclose(
        features[1], np.array([2.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    )

    np.testing.assert_allclose(features[2, 0], expected_mean_c, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(features[2, 1], expected_std_c, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(features[2, 2], 0.0, atol=1e-6)
    assert 0.0 <= float(features[2, 3]) <= 1.0
    assert float(features[2, 4]) >= 0.0
