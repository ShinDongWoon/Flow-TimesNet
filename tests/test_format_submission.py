from pathlib import Path
import sys
import pandas as pd

# Ensure src path is available for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.utils.io import format_submission


def test_format_submission_invalid_row_key():
    sample_df = pd.DataFrame(
        {
            "date": ["TEST_A+Day 1", "BAD_KEY"],
            "MENU_1": [0.0, 0.0],
            "MENU_2": [0.0, 0.0],
        }
    )

    pred_df = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=pd.date_range("2023-01-01", periods=2, freq="D"),
        columns=["MENU_1", "MENU_2"],
    )
    pred_df["date"] = pred_df.index
    pred_df["row_key"] = ["TEST_A+D1", "TEST_A+D2"]
    preds = pred_df.set_index("row_key")

    out = format_submission(sample_df, preds, date_col="date")

    # First row parsed successfully
    assert out.loc[0, "date"] == pd.Timestamp("2023-01-01")
    assert out.loc[0, "MENU_1"] == 1.0
    assert out.loc[0, "MENU_2"] == 2.0

    # Second row had invalid key and should be filled with defaults
    assert pd.isna(out.loc[1, "date"])
    assert out.loc[1, "MENU_1"] == 0.0
    assert out.loc[1, "MENU_2"] == 0.0
