from pathlib import Path
import sys
import pandas as pd

# Ensure src path is available for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.utils.io import format_submission


def test_format_submission_invalid_row_key():
    sample_df = pd.DataFrame(
        {
            "row_key": ["TEST_A+Day 1", "BAD_KEY"],
            "MENU 1": [0.0, 0.0],
            "MENU 2": [0.0, 0.0],
        }
    )

    preds = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=["TEST_A+D1", "TEST_A+D2"],
        columns=["MENU_1", "MENU_2"],
    )

    out = format_submission(sample_df, preds)

    # First row parsed successfully
    assert out.loc[0, "MENU 1"] == 1.0
    assert out.loc[0, "MENU 2"] == 2.0

    # Second row had invalid key and should be filled with defaults
    assert out.loc[1, "MENU 1"] == 0.0
    assert out.loc[1, "MENU 2"] == 0.0

    # Column order and names preserved
    assert list(out.columns) == ["row_key", "MENU 1", "MENU 2"]
