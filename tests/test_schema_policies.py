import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure the project src is on the path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.utils import io as io_utils


def _build_ambiguous_dataframe() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    return pd.DataFrame(
        {
            "주문일자": dates,
            "배송시작일": dates + pd.Timedelta(days=1),
            "영업장명_메뉴명": ["StoreA_Menu1"] * len(dates),
            "매출수량": [1.0, 2.0, 3.0, 4.0],
        }
    )


def _build_evolving_dataframe() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    rows = []
    for idx, d in enumerate(dates):
        rows.append(
            {
                "date": d,
                "id": "StoreA",
                "target": float(idx + 1),
                "promo_type": None if idx < 3 else "BUNDLE",
            }
        )
    return pd.DataFrame(rows)


def test_schema_detection_strict_rejects_ambiguity():
    df = _build_ambiguous_dataframe()
    with pytest.raises(ValueError) as exc:
        io_utils.DataSchema.from_config(
            {"schema_detection_policy": "strict"},
            sample_df=df,
        )
    assert "Ambiguous auto-detection" in str(exc.value)


def test_schema_detection_manual_requires_overrides():
    df = _build_ambiguous_dataframe()
    with pytest.raises(ValueError):
        io_utils.DataSchema.from_config(
            {"schema_detection_policy": "manual"},
            sample_df=df,
        )

    schema = io_utils.DataSchema.from_config(
        {
            "schema_detection_policy": "manual",
            "date_col": "주문일자",
            "id_col": "영업장명_메뉴명",
            "target_col": "매출수량",
        },
        sample_df=df,
    )
    assert schema["date"] == "주문일자"
    assert schema.sources["date"] == "override"


def test_schema_evolution_warns(caplog):
    df = _build_evolving_dataframe()
    caplog.set_level("WARNING")
    schema = io_utils.DataSchema.from_config({}, sample_df=df)
    assert any("Schema evolution detected" in rec.message for rec in caplog.records)
    coverage = schema.detection.get("coverage", {})
    promo_meta = coverage.get("promo_type")
    assert promo_meta and promo_meta.get("missing_prefix") is True


def test_schema_evolution_error_policy():
    df = _build_evolving_dataframe()
    with pytest.raises(ValueError):
        io_utils.DataSchema.from_config(
            {"schema_evolution_policy": "error"},
            sample_df=df,
        )
