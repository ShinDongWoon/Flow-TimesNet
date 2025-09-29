from __future__ import annotations

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
from timesnet_forecast.utils import io as io_utils


def test_predict_once_restores_static_checkpoint(tmp_path):
    torch.manual_seed(0)
    np.random.seed(0)

    ids = ["B", "A"]
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
        min_sigma_vector=[0.05, 0.1],
    )

    warmup_series = torch.zeros(1, input_len, 1)
    warmup_static = static_features[:1, :]
    warmup_ids = series_ids[:1]
    full_sigma = None
    if isinstance(model.min_sigma_vector, torch.Tensor) and model.min_sigma_vector.numel() > 0:
        full_sigma = model.min_sigma_vector.detach().clone()
        model.min_sigma_vector = model.min_sigma_vector[..., :1]
    with torch.no_grad():
        model(warmup_series, series_static=warmup_static, series_ids=warmup_ids)
    if full_sigma is not None:
        model.min_sigma_vector = full_sigma.clone()
    if isinstance(model.series_embedding, torch.nn.Embedding):
        current_vocab = int(model.series_embedding.num_embeddings)
        required_vocab = len(ids)
        if required_vocab > current_vocab:
            new_embed = torch.nn.Embedding(
                required_vocab,
                model.series_embedding.embedding_dim,
            )
            with torch.no_grad():
                new_embed.weight.zero_()
                new_embed.weight[:current_vocab] = model.series_embedding.weight
            model.series_embedding = new_embed
        model._series_id_vocab = required_vocab
        model._series_id_reference = torch.arange(required_vocab, dtype=torch.long)

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

    io_utils.save_schema_artifact(
        str(art_dir / "schema.json"),
        io_utils.DataSchema.from_fields(
            {"date": "date", "id": "series_id", "target": "value"},
            sources={"date": "override", "id": "override", "target": "override"},
        ),
        normalization={"method": "none", "per_series": True, "eps": 1e-8},
    )

    test_dir = tmp_path / "test_data"
    test_dir.mkdir()

    dates = pd.date_range("2024-01-01", periods=input_len, freq="D")
    series_a = np.linspace(0.1, 0.4, input_len, dtype=np.float32)
    series_b = np.linspace(1.0, 1.3, input_len, dtype=np.float32)
    values_by_id = {"A": series_a, "B": series_b}
    data_order = ["A", "B"]

    rows = []
    for day_idx, ts in enumerate(dates):
        for series_name in data_order:
            rows.append(
                {
                    "date": ts.strftime("%Y-%m-%d"),
                    "series_id": series_name,
                    "value": float(values_by_id[series_name][day_idx]),
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
        "preprocess": {
            "clip_negative": False,
            "normalize": "none",
            "normalize_per_series": True,
            "eps": 1e-8,
        },
        "train": {
            "device": "cpu",
            "matmul_precision": "medium",
            "channels_last": True,
            "amp": False,
            "use_checkpoint": False,
            "cuda_graphs": False,
            "val": {"strategy": "holdout", "holdout_days": input_len + pred_len},
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
    id_position_map = {sid: idx for idx, sid in enumerate(ids)}
    gather_positions = [id_position_map[sid] for sid in data_order]
    xb_expected_np = np.stack(
        [values_by_id[sid] for sid in data_order], axis=0
    ).astype(np.float32)
    xb_expected = torch.from_numpy(xb_expected_np[:, :, None])
    static_expected = static_features[gather_positions].unsqueeze(1)
    ids_expected = torch.tensor(gather_positions, dtype=torch.long).view(-1, 1)
    with torch.no_grad():
        expected_mu, _ = model(
            xb_expected,
            series_static=static_expected,
            series_ids=ids_expected,
        )
    expected_partial = expected_mu.squeeze(-1).numpy()
    if expected_partial.ndim == 1:
        expected_partial = expected_partial.reshape(1, -1)
    expected = np.zeros((pred_len, len(ids)), dtype=np.float32)
    expected[:, np.asarray(gather_positions, dtype=np.int64)] = expected_partial.T
    expected = np.clip(expected, 0.0, None)
    if full_sigma is not None:
        model.min_sigma_vector = full_sigma.clone()

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
