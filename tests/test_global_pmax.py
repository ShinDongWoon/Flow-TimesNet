import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

# Ensure the project src is on the path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.config import Config
from timesnet_forecast import train
from timesnet_forecast.models.timesnet import TimesNet
from timesnet_forecast.utils import io as io_utils
from timesnet_forecast.data.split import make_holdout_slices


def test_timesnet_embedding_accepts_temporal_features():
    torch.manual_seed(0)
    B, L, H, N, mark_dim = 2, 12, 3, 4, 5
    model = TimesNet(
        input_len=L,
        pred_len=H,
        d_model=16,
        d_ff=32,
        n_layers=2,
        k_periods=2,
        kernel_set=[(3, 3)],
        dropout=0.1,
        activation="gelu",
        mode="direct",
    )
    x = torch.randn(B, L + 2, N)
    x_mark = torch.randn(B, L + 2, mark_dim)
    mu, sigma = model(x, x_mark=x_mark)
    assert mu.shape == (B, H, N)
    assert sigma.shape == (B, H, N)
    assert torch.all(sigma >= model.min_sigma)


def test_train_once_runs_without_pmax(tmp_path):
    periods = 40
    dates = pd.date_range("2023-01-01", periods=periods, freq="D")
    t = np.arange(periods, dtype=np.float32)
    s1 = np.sin(2 * math.pi * 5 * t / periods) + 10.0
    s2 = np.sin(2 * math.pi * 2 * t / periods) + 20.0
    rows = []
    for i, d in enumerate(dates):
        rows.append({"date": d, "id": "S1", "target": float(s1[i])})
        rows.append({"date": d, "id": "S2", "target": float(s2[i])})
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
        "train.channels_last=False",
        "train.val.strategy=holdout",
        "train.val.holdout_days=10",
        "preprocess.normalize=none",
        "preprocess.normalize_per_series=True",
        "preprocess.clip_negative=False",
        "preprocess.eps=1e-8",
        "model.mode=direct",
        "model.input_len=6",
        "model.pred_len=2",
        "model.d_model=8",
        "model.d_ff=16",
        "model.n_layers=1",
        "model.dropout=0.0",
        "model.k_periods=1",
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
    cfg = Config.from_files("configs/default.yaml", overrides=overrides).to_dict()

    df_loaded = pd.read_csv(csv_path)
    schema = io_utils.DataSchema.from_config(cfg["data"], sample_df=df_loaded)
    wide = io_utils.pivot_long_to_wide(
        df_loaded,
        date_col=schema["date"],
        id_col=schema["id"],
        target_col=schema["target"],
        fill_missing_dates=cfg["data"]["fill_missing_dates"],
        fillna0=True,
    )
    train_df, _ = make_holdout_slices(wide, cfg["train"]["val"]["holdout_days"])
    _, trn_norm = io_utils.fit_series_scaler(
        train_df,
        cfg["preprocess"]["normalize"],
        cfg["preprocess"]["normalize_per_series"],
        cfg["preprocess"]["eps"],
    )
    assert trn_norm.shape[0] >= cfg["model"]["input_len"]

    train.train_once(cfg)
    scaler_meta = io_utils.load_pickle(
        tmp_path / "artifacts" / cfg["artifacts"]["scaler_file"]
    )
    static_features = scaler_meta.get("static_features")
    feature_names = scaler_meta.get("feature_names")
    assert isinstance(static_features, np.ndarray)
    assert static_features.dtype == np.float32
    assert static_features.shape[0] == len(scaler_meta["ids"])
    assert feature_names == [
        "mean",
        "std",
        "diff_std",
        "seasonal_strength",
        "dominant_period",
    ]
    assert "pmax" not in cfg.get("model", {})

