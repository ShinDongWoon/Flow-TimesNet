import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Ensure the project src is on the path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.train import _compute_pmax_global
from timesnet_forecast.config import Config
from timesnet_forecast import train


def test_compute_pmax_global():
    T = 100
    t = np.arange(T, dtype=np.float32)
    s1 = np.sin(2 * math.pi * 5 * t / T)  # period 20
    s2 = np.sin(2 * math.pi * 2 * t / T)  # period 50
    arr = np.stack([s1, s2], axis=1)
    pmax = _compute_pmax_global([arr], k=1, cap=100)
    assert pmax == 50
    assert _compute_pmax_global([arr], k=1, cap=30) == 30


def test_pmax_cap_applied(tmp_path, monkeypatch):
    # Create a tiny training CSV
    dates = pd.date_range("2023-01-01", periods=6, freq="D")
    rows = []
    for d in dates:
        rows.append({"date": d, "id": "S1", "target": 1.0})
        rows.append({"date": d, "id": "S2", "target": 2.0})
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
        "train.cuda_graphs=False",
        "train.val.strategy=holdout",
        "train.val.holdout_days=3",
        "preprocess.normalize=none",
        "preprocess.normalize_per_series=True",
        "preprocess.eps=1e-8",
        "model.mode=direct",
        "model.input_len=2",
        "model.pred_len=1",
        "model.d_model=8",
        "model.n_layers=1",
        "model.dropout=0.0",
        "model.k_periods=1",
        "model.kernel_set=[3]",
        "model.pmax_cap=5",
        "train.lr=1e-3",
        "train.weight_decay=0.0",
        "train.grad_clip_norm=0.0",
        f"artifacts.dir={tmp_path}/artifacts",
        "artifacts.model_file=model.pth",
        "artifacts.scaler_file=scaler.pkl",
        "artifacts.schema_file=schema.json",
        "artifacts.config_file=config.yaml",
    ]
    cfg = Config.from_files("configs/default.yaml", overrides=overrides).to_dict()

    def _fake_compute(arrays, k, cap):
        assert cap == 5
        return min(50, cap)

    monkeypatch.setattr(train, "_compute_pmax_global", _fake_compute)
    train.train_once(cfg)
    assert cfg["model"]["pmax"] == 5

