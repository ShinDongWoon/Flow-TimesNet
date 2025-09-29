from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.utils.submission import (  # noqa: E402
    DateMenuWriter,
    RowKeyLongWriter,
    SubmissionRowMeta,
    build_submission_context,
    get_submission_writer,
)


def _build_context(
    *,
    predictions: pd.DataFrame,
    sample_df: pd.DataFrame | None,
    row_meta: dict[str, SubmissionRowMeta],
    row_order: list[str],
    submission_cfg: dict | None = None,
) -> object:
    return build_submission_context(
        predictions=predictions,
        sample_df=sample_df,
        row_meta=row_meta,
        row_order=row_order,
        test_parts={"TEST_A": row_order},
        ids=list(predictions.columns),
        new_ids=[],
        missing_ids=[],
        missing_by_part={"TEST_A": []},
        submission_cfg=submission_cfg or {},
    )


def test_row_key_writer_parses_variants():
    sample_df = pd.DataFrame(
        {
            "row_key": ["TEST_A+Day 1", "TEST_A+D2", "TEST_A+3일"],
            "MENU 1": [0.0, 0.0, 0.0],
            "MENU 2": [0.0, 0.0, 0.0],
        }
    )

    preds = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        index=["TEST_A+D1", "TEST_A+D2", "TEST_A+D3"],
        columns=["MENU_1", "MENU_2"],
    )

    row_meta = {
        "TEST_A+D1": SubmissionRowMeta("TEST_A", 1),
        "TEST_A+D2": SubmissionRowMeta("TEST_A", 2),
        "TEST_A+D3": SubmissionRowMeta("TEST_A", 3),
    }
    row_order = ["TEST_A+D1", "TEST_A+D2", "TEST_A+D3"]

    context = _build_context(
        predictions=preds,
        sample_df=sample_df,
        row_meta=row_meta,
        row_order=row_order,
    )
    writer = RowKeyLongWriter()
    out = writer.render(preds, context)

    assert list(out["MENU 1"]) == [1.0, 3.0, 5.0]
    assert list(out["MENU 2"]) == [2.0, 4.0, 6.0]
    assert list(out.columns) == ["row_key", "MENU 1", "MENU 2"]


def test_row_key_writer_handles_invalid_and_missing_rows():
    sample_df = pd.DataFrame(
        {
            "row_key": ["TEST_A+Day 1", "BAD_KEY", "TEST_A+Day 3"],
            "MENU 1": [0.0, 0.0, 0.0],
            "MENU 2": [0.0, 0.0, 0.0],
        }
    )

    preds = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=["TEST_A+D1", "TEST_A+D2"],
        columns=["MENU_1", "MENU_2"],
    )

    row_meta = {
        "TEST_A+D1": SubmissionRowMeta("TEST_A", 1),
        "TEST_A+D2": SubmissionRowMeta("TEST_A", 2),
        "TEST_A+D3": SubmissionRowMeta("TEST_A", 3),
    }
    row_order = ["TEST_A+D1", "TEST_A+D2", "TEST_A+D3"]

    context = _build_context(
        predictions=preds,
        sample_df=sample_df,
        row_meta=row_meta,
        row_order=row_order,
    )
    writer = RowKeyLongWriter()
    out = writer.render(preds, context)

    assert out.loc[0, "MENU 1"] == 1.0
    assert out.loc[0, "MENU 2"] == 2.0
    assert out.loc[1, "MENU 1"] == 0.0
    assert out.loc[1, "MENU 2"] == 0.0
    assert out.loc[2, "MENU 1"] == 0.0
    assert out.loc[2, "MENU 2"] == 0.0


def test_date_menu_writer_synthesizes_template_without_sample():
    preds = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=["TEST_A+D1", "TEST_A+D2"],
        columns=["MENU_1", "MENU_2"],
    )
    row_order = ["TEST_A+D1", "TEST_A+D2", "TEST_A+D3"]
    row_meta = {
        "TEST_A+D1": SubmissionRowMeta("TEST_A", 1, pd.Timestamp("2024-01-01")),
        "TEST_A+D2": SubmissionRowMeta("TEST_A", 2, pd.Timestamp("2024-01-02")),
        "TEST_A+D3": SubmissionRowMeta("TEST_A", 3, pd.Timestamp("2024-01-03")),
    }

    context = _build_context(
        predictions=preds,
        sample_df=None,
        row_meta=row_meta,
        row_order=row_order,
        submission_cfg={"date_col": "영업일자"},
    )
    writer = DateMenuWriter()
    out = writer.render(preds, context)

    assert list(out.columns) == ["영업일자", "MENU_1", "MENU_2"]
    assert list(out["영업일자"]) == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    assert out.loc[0, "MENU_1"] == 1.0
    assert out.loc[0, "MENU_2"] == 2.0
    assert out.loc[1, "MENU_1"] == 3.0
    assert out.loc[1, "MENU_2"] == 4.0
    assert out.loc[2, "MENU_1"] == 0.0
    assert out.loc[2, "MENU_2"] == 0.0


def test_get_submission_writer_dispatches_and_validates():
    assert get_submission_writer("row_key") is RowKeyLongWriter
    assert get_submission_writer("date_menu") is DateMenuWriter
    with pytest.raises(KeyError):
        get_submission_writer("unknown_format")
