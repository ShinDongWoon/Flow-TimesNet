from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Ensure the project src is on the path for imports
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.config import PipelineConfig  # noqa: E402
from timesnet_forecast.utils import io as io_utils  # noqa: E402
from timesnet_forecast.utils import metadata as metadata_utils  # noqa: E402


_BASE_CONFIG = {
    "model": {
        "mode": "direct",
        "input_len": 4,
        "pred_len": 2,
        "d_model": 8,
        "d_ff": 16,
        "n_layers": 1,
        "k_periods": 1,
        "min_period_threshold": 1,
        "kernel_set": [3],
        "dropout": 0.0,
        "activation": "gelu",
        "bottleneck_ratio": 1.0,
        "use_embedding_norm": True,
        "id_embed_dim": 4,
        "static_proj_dim": 2,
        "static_layernorm": True,
    },
    "data": {
        "train_csv": "train.csv",
        "test_dir": "test",
        "sample_submission": "sample.csv",
        "date_col": "date",
        "target_col": "target",
        "id_col": "series_id",
        "min_context_days": 4,
        "horizon": 2,
        "fill_missing_dates": True,
        "encoding": "utf-8",
        "schema_detection_policy": "manual",
        "schema_evolution_policy": "warn",
        "time_features": {
            "enabled": True,
            "features": ["day_of_week", "month"],
            "encoding": "cyclical",
            "normalize": True,
            "freq": "D",
            "feature_dim": 4,
        },
    },
    "train": {
        "device": "cpu",
        "epochs": 1,
        "batch_size": 4,
        "accumulation_steps": 1,
        "lr_warmup_steps": 0,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "grad_clip_norm": 0.0,
        "amp": False,
        "compile": False,
        "deterministic": False,
        "cuda_graphs": False,
        "use_checkpoint": False,
        "min_sigma": 1e-3,
        "min_sigma_method": "global",
        "min_sigma_scale": 0.1,
        "matmul_precision": "medium",
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": 2,
        "channels_last": False,
        "use_loss_masking": False,
        "val": {"strategy": "holdout", "holdout_days": 8},
    },
    "window": {
        "input_len": 4,
        "pred_len": 2,
        "stride": 1,
        "short_series_strategy": "error",
        "pad_value": 0.0,
    },
    "artifacts": {
        "dir": "artifacts",
        "model_file": "model.pth",
        "scaler_file": "scaler.pkl",
        "schema_file": "schema.json",
        "config_file": "config_used.yaml",
        "signature_file": "model_signature.json",
        "metadata_file": "metadata.json",
    },
}


@pytest.fixture()
def base_config() -> PipelineConfig:
    return PipelineConfig.from_mapping(_BASE_CONFIG)


def test_metadata_artifact_validates_config_and_artifacts(base_config: PipelineConfig) -> None:
    schema = io_utils.DataSchema.from_fields(
        {
            "date": base_config.data.date_col,
            "id": base_config.data.id_col,
            "target": base_config.data.target_col,
        }
    )
    artifact = metadata_utils.MetadataArtifact.from_training(
        window=base_config.window,
        schema=schema,
        time_features={
            "config": base_config.data.time_features.to_dict(),
            "enabled": True,
            "feature_dim": 4,
            "freq": "D",
        },
        static_features={"feature_names": ["mean", "std"], "feature_dim": 2},
    )

    artifact.validate_config(base_config)

    scaler_meta = {
        "ids": ["S1", "S2"],
        "method": "none",
        "scaler": None,
        "static_features": np.zeros((2, 2), dtype=np.float32),
        "feature_names": ["mean", "std"],
        "time_features": {
            "enabled": True,
            "feature_dim": 4,
            "freq": "D",
            "config": base_config.data.time_features.to_dict(),
        },
    }

    artifact.validate_artifacts(
        schema=schema,
        scaler_meta=scaler_meta,
        num_series=len(scaler_meta["ids"]),
    )

    mismatch_cfg = base_config.apply_overrides(
        [
            f"window.pred_len={base_config.window.pred_len + 1}",
            f"data.horizon={base_config.window.pred_len + 1}",
        ]
    )
    with pytest.raises(ValueError, match="window.pred_len"):
        artifact.validate_config(mismatch_cfg)

    mismatch_scaler = dict(scaler_meta)
    mismatch_scaler["feature_names"] = ["foo", "bar"]
    with pytest.raises(ValueError, match="Static feature names"):
        artifact.validate_artifacts(
            schema=schema,
            scaler_meta=mismatch_scaler,
            num_series=len(scaler_meta["ids"]),
        )


def test_metadata_artifact_version_guard(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    io_utils.save_json({"meta_version": "999"}, str(path))
    with pytest.raises(ValueError, match="not supported"):
        metadata_utils.load_metadata_artifact(str(path))


def test_metadata_artifact_migrates_legacy_payload(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    legacy_payload = {
        "window": {"input_len": 4, "pred_len": 2},
        "schema": {"date": "date", "id": "series_id", "target": "value"},
        "time_features": {"enabled": False, "feature_dim": 0},
        "static_features": ["mean", "std"],
    }
    io_utils.save_json(legacy_payload, str(path))

    artifact = metadata_utils.load_metadata_artifact(str(path))
    assert artifact.meta_version == metadata_utils.METADATA_ARTIFACT_VERSION
    assert artifact.static_features["feature_names"] == ["mean", "std"]
    assert artifact.time_features["feature_dim"] == 0
