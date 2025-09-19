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
from timesnet_forecast.utils import io as io_utils
from timesnet_forecast.data.split import make_holdout_slices


def test_compute_pmax_global():
    T = 100
    t = np.arange(T, dtype=np.float32)
    s1 = np.sin(2 * math.pi * 5 * t / T)  # period 20
    s2 = np.sin(2 * math.pi * 2 * t / T)  # period 50
    arr = np.stack([s1, s2], axis=1)
    pmax = _compute_pmax_global([arr], k=1, cap=100)
    assert pmax == 50
    assert _compute_pmax_global([arr], k=1, cap=30) == 30


def test_pmax_cap_applied(tmp_path):
    # Create a synthetic training CSV with strong periodic components
    periods = 60
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
        "train.val.holdout_days=3",
        "preprocess.normalize=none",
        "preprocess.normalize_per_series=True",
        "preprocess.clip_negative=False",
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

    schema = io_utils.resolve_schema(cfg)
    df_loaded = pd.read_csv(csv_path)
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
    train_arrays = [trn_norm.values.astype(np.float32)]
    expected_pmax = _compute_pmax_global(
        train_arrays,
        int(cfg["model"]["k_periods"]),
        int(cfg["model"]["pmax_cap"]),
    )
    assert expected_pmax == int(cfg["model"]["pmax_cap"])

    train.train_once(cfg)
    assert cfg["model"]["pmax"] == expected_pmax

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD

def test_short_periods_clamped_to_input_len(tmp_path):
    periods = 40
    dates = pd.date_range("2023-01-01", periods=periods, freq="D")
    pattern = (np.arange(periods, dtype=np.int64) % 2).astype(np.float32)
    rows = []
    for i, d in enumerate(dates):
        rows.append({"date": d, "id": "S1", "target": float(pattern[i])})
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "train_short.csv"
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
        "train.val.holdout_days=12",
        "preprocess.normalize=none",
        "preprocess.normalize_per_series=True",
        "preprocess.clip_negative=False",
        "preprocess.eps=1e-8",
        "model.mode=direct",
        "model.input_len=8",
        "model.pred_len=2",
        "model.d_model=8",
        "model.n_layers=1",
        "model.dropout=0.0",
        "model.k_periods=1",
        "model.kernel_set=[3]",
        "model.pmax_cap=64",
        "model.activation=gelu",
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

    schema = io_utils.resolve_schema(cfg)
    df_loaded = pd.read_csv(csv_path)
    wide_raw = io_utils.pivot_long_to_wide(
        df_loaded,
        date_col=schema["date"],
        id_col=schema["id"],
        target_col=schema["target"],
        fill_missing_dates=cfg["data"]["fill_missing_dates"],
        fillna0=False,
    )
    mask_wide = (~wide_raw.isna()).astype(np.float32)
    wide = wide_raw.fillna(0.0)

    trn_df, _ = make_holdout_slices(wide, cfg["train"]["val"]["holdout_days"])
    trn_mask_df, _ = make_holdout_slices(mask_wide, cfg["train"]["val"]["holdout_days"])

    train_arrays = [trn_df.to_numpy(dtype=np.float32, copy=False)]
    train_mask_arrays = [trn_mask_df.to_numpy(dtype=np.float32, copy=False)]

    inferred = _compute_pmax_global(
        train_arrays,
        int(cfg["model"]["k_periods"]),
        int(cfg["model"]["pmax_cap"]),
    )
    input_len = int(cfg["model"]["input_len"])
    assert inferred < input_len

    expected_pmax = min(int(cfg["model"]["pmax_cap"]), max(inferred, input_len))

    train.train_once(cfg)
    assert cfg["model"]["pmax"] == expected_pmax == input_len

    dl = train._build_dataloader(
        train_arrays,
        train_mask_arrays,
        input_len,
        int(cfg["model"]["pred_len"]),
        cfg["model"]["mode"],
        batch_size=1,
        num_workers=1,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        shuffle=False,
        drop_last=False,
        augment=None,
        pmax_global=int(cfg["model"]["pmax"]),
    )

    xb, _, _, _ = next(iter(dl))
    assert xb.shape[1] == input_len
    assert xb.shape[1] == int(cfg["model"]["pmax"])

=======
>>>>>>> parent of 11c1c96 (Fix duplicate import in test_global_pmax)
=======
>>>>>>> parent of 11c1c96 (Fix duplicate import in test_global_pmax)
=======
>>>>>>> parent of 11c1c96 (Fix duplicate import in test_global_pmax)
