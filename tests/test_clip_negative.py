import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the project src is on the path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.config import PipelineConfig
from timesnet_forecast import train
from timesnet_forecast.utils import io as io_utils


def test_clip_negative(tmp_path, monkeypatch):
    dates = pd.date_range("2023-01-01", periods=6, freq="D")
    vals1 = [-5, -1, 2, -3, 4, 5]
    vals2 = [1, -2, 3, -4, 5, -6]
    rows = []
    for d, v1, v2 in zip(dates, vals1, vals2):
        rows.append({"date": d, "id": "StoreA_Menu1", "target": v1})
        rows.append({"date": d, "id": "StoreA_Menu2", "target": v2})
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "train.csv"
    df.to_csv(csv_path, index=False)

    overrides = [
        f"data.train_csv={csv_path}",
        "data.date_col=date",
        "data.id_col=id",
        "data.target_col=target",
        "data.encoding=utf-8",
        "data.fill_missing_dates=False",
        "train.device=cpu",
        "train.epochs=1",
        "train.batch_size=2",
        "train.num_workers=1",
        "train.pin_memory=False",
        "train.persistent_workers=False",
        "train.prefetch_factor=2",
        "train.amp=False",
        "train.compile=False",
        "train.val.strategy=holdout",
        "train.val.holdout_days=3",
        "preprocess.normalize=none",
        "preprocess.normalize_per_series=True",
        "preprocess.eps=1e-8",
        "preprocess.clip_negative=True",
        "model.mode=direct",
        "model.input_len=2",
        "model.pred_len=1",
        "model.d_model=8",
        "model.n_layers=1",
        "model.dropout=0.0",
        "model.k_periods=2",
        "model.min_period_threshold=1",
        "model.kernel_set=[[3,3]]",
        "train.lr=1e-3",
        "train.weight_decay=0.0",
        "train.grad_clip_norm=0.0",
        f"artifacts.dir={tmp_path}/artifacts",
        "artifacts.model_file=model.pth",
        "artifacts.scaler_file=scaler.pkl",
        "artifacts.schema_file=schema.json",
        "artifacts.config_file=config.yaml",
    ]
    cfg = PipelineConfig.from_files("configs/default.yaml", overrides=overrides)

    captured = []
    orig_build = train._build_dataloader

    def capture_build(arrays, masks, *args, **kwargs):
        for a in arrays:
            captured.append(a)
        return orig_build(arrays, masks, *args, **kwargs)

    monkeypatch.setattr(train, "_build_dataloader", capture_build)

    train.train_once(cfg)

    assert captured, "No data arrays captured"
    for arr in captured:
        assert np.all(arr >= 0.0), "Negative values remain after preprocessing"

    schema_path = tmp_path / "artifacts" / "schema.json"
    schema_obj, schema_meta = io_utils.load_schema_artifact(str(schema_path))
    assert schema_obj["date"] == "date"
    assert schema_obj.sources["target"] == "override"
    normalization_meta = schema_meta.get("normalization")
    assert normalization_meta["method"] == "none"
    assert normalization_meta["per_series"] is True


def test_train_auto_schema_detection(tmp_path):
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    rows = []
    for d in dates:
        rows.append({"timestamp": d, "series_id": "StoreA", "sales": 1.0})
        rows.append({"timestamp": d, "series_id": "StoreB", "sales": 2.0})
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "train.csv"
    df.to_csv(csv_path, index=False)

    overrides = [
        f"data.train_csv={csv_path}",
        "data.date_col=null",
        "data.id_col=null",
        "data.target_col=null",
        "data.encoding=utf-8",
        "data.fill_missing_dates=False",
        "train.device=cpu",
        "train.epochs=1",
        "train.batch_size=2",
        "train.num_workers=1",
        "train.pin_memory=False",
        "train.persistent_workers=False",
        "train.prefetch_factor=2",
        "train.amp=False",
        "train.compile=False",
        "train.val.strategy=holdout",
        "train.val.holdout_days=3",
        "preprocess.normalize=none",
        "preprocess.normalize_per_series=True",
        "preprocess.eps=1e-8",
        "preprocess.clip_negative=False",
        "model.mode=direct",
        "model.input_len=2",
        "model.pred_len=1",
        "model.d_model=8",
        "model.n_layers=1",
        "model.dropout=0.0",
        "model.k_periods=1",
        "model.min_period_threshold=1",
        "model.kernel_set=[[3,3]]",
        "train.lr=1e-3",
        "train.weight_decay=0.0",
        "train.grad_clip_norm=0.0",
        f"artifacts.dir={tmp_path}/artifacts",
        "artifacts.model_file=model.pth",
        "artifacts.scaler_file=scaler.pkl",
        "artifacts.schema_file=schema.json",
        "artifacts.config_file=config.yaml",
    ]
    cfg = PipelineConfig.from_files("configs/default.yaml", overrides=overrides)

    train.train_once(cfg)

    schema_obj, schema_meta = io_utils.load_schema_artifact(
        str(tmp_path / "artifacts" / "schema.json")
    )
    assert schema_obj["date"] == "timestamp"
    assert schema_obj.sources["date"] != "override"
    assert schema_obj["id"] == "series_id"
    assert schema_obj["target"] == "sales"
    assert schema_meta.get("version") == io_utils.SCHEMA_ARTIFACT_VERSION
