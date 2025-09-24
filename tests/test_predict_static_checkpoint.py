from __future__ import annotations

import json
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import yaml


# Ensure the project src is on the path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.models.timesnet import TimesNet
from timesnet_forecast.predict import predict_once


def test_predict_once_restores_static_checkpoint(tmp_path):
    torch.manual_seed(0)
    np.random.seed(0)

    ids = ["A", "B"]
    input_len = 4
    pred_len = 2

    static_features = torch.tensor(
        [[0.5, -0.25, 1.0], [1.25, 0.75, -0.5]], dtype=torch.float32
    )
    series_ids = torch.arange(len(ids), dtype=torch.long)

    model = TimesNet(
        input_len=input_len,
        pred_len=pred_len,
        d_model=8,
        d_ff=16,
        n_layers=1,
        k_periods=1,
        kernel_set=[3],
        dropout=0.0,
        activation="gelu",
        mode="direct",
        bottleneck_ratio=1.0,
        channels_last=True,
        use_checkpoint=False,
        use_embedding_norm=True,
        id_embed_dim=4,
        static_proj_dim=3,
        static_layernorm=True,
    )

    train_warmup = torch.randn(1, input_len, len(ids))
    with torch.no_grad():
        model(train_warmup, series_static=static_features, series_ids=series_ids)

    art_dir = tmp_path / "artifacts"
    art_dir.mkdir()

    model_path = art_dir / "timesnet.pth"
    torch.save(model.state_dict(), model_path)

    scaler_meta = {
        "ids": ids,
        "method": "none",
        "scaler": None,
        "static_features": static_features.numpy(),
    }
    with open(art_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler_meta, f)

    with open(art_dir / "schema.json", "w", encoding="utf-8") as f:
        json.dump({"date": "date", "id": "series_id", "target": "value"}, f)

    test_dir = tmp_path / "test_data"
    test_dir.mkdir()

    dates = pd.date_range("2024-01-01", periods=input_len, freq="D")
    series_a = np.linspace(0.1, 0.4, input_len, dtype=np.float32)
    series_b = np.linspace(1.0, 1.3, input_len, dtype=np.float32)
    values = np.stack([series_a, series_b], axis=1)

    rows = []
    for day_idx, ts in enumerate(dates):
        for series_idx, series_name in enumerate(ids):
            rows.append(
                {
                    "date": ts.strftime("%Y-%m-%d"),
                    "series_id": series_name,
                    "value": float(values[day_idx, series_idx]),
                }
            )
    pd.DataFrame(rows).to_csv(test_dir / "TEST_00.csv", index=False)

    sample_path = tmp_path / "sample_submission.csv"
    sample_df = pd.DataFrame(
        {
            "row_key": ["TEST_00+Day 1", "TEST_00+Day 2"],
            "A": [0.0, 0.0],
            "B": [0.0, 0.0],
        }
    )
    sample_df.to_csv(sample_path, index=False)

    cfg = {
        "artifacts": {
            "dir": str(art_dir),
            "model_file": model_path.name,
            "scaler_file": "scaler.pkl",
            "schema_file": "schema.json",
            "config_file": "config_used.yaml",
        },
        "data": {
            "test_dir": str(test_dir),
            "sample_submission": str(sample_path),
            "fill_missing_dates": False,
        },
        "preprocess": {"clip_negative": False},
        "train": {
            "device": "cpu",
            "matmul_precision": "medium",
            "channels_last": True,
            "amp": False,
            "use_checkpoint": False,
            "cuda_graphs": False,
        },
        "model": {
            "mode": "direct",
            "input_len": input_len,
            "pred_len": pred_len,
            "d_model": 8,
            "d_ff": 16,
            "n_layers": 1,
            "k_periods": 1,
            "kernel_set": [3],
            "dropout": 0.0,
            "activation": "gelu",
            "bottleneck_ratio": 1.0,
            "use_embedding_norm": True,
            "id_embed_dim": 4,
            "static_proj_dim": 3,
            "static_layernorm": True,
            "min_period_threshold": 1,
        },
        "submission": {"out_path": str(tmp_path / "outputs" / "submission.csv")},
    }

    config_payload = {
        "model": cfg["model"],
        "train": cfg["train"],
        "data": cfg["data"],
        "artifacts": cfg["artifacts"],
        "submission": cfg["submission"],
        "preprocess": cfg["preprocess"],
    }
    with open(art_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config_payload, f, allow_unicode=True)

    model.eval()
    with torch.no_grad():
        expected_mu, _ = model(
            torch.from_numpy(values).unsqueeze(0),
            series_static=static_features,
            series_ids=series_ids,
        )
    expected = np.clip(expected_mu.squeeze(0).numpy(), 0.0, None)

    out_path = predict_once(cfg)

    assert out_path == cfg["submission"]["out_path"]
    submission_df = pd.read_csv(out_path)
    row_col = submission_df.columns[0]
    for day_idx in range(pred_len):
        row_key = f"TEST_00+Day {day_idx + 1}"
        row = submission_df.loc[submission_df[row_col] == row_key, ids]
        assert not row.empty
        np.testing.assert_allclose(
            row.to_numpy(dtype=np.float32)[0],
            expected[day_idx],
            rtol=1e-5,
            atol=1e-5,
        )
