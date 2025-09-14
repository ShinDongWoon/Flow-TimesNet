from __future__ import annotations

import os
from glob import glob
from typing import Dict, List
import numpy as np
import pandas as pd
import torch

from .config import Config
from .utils.logging import console
from .utils.torch_opt import amp_autocast, maybe_channels_last
from .utils import io as io_utils
from .models.timesnet import TimesNet


def _select_device(req: str) -> torch.device:
    if req == "cuda" and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _pad_left_zeros(arr: np.ndarray, need_len: int) -> np.ndarray:
    """
    arr: [T, N], pad zeros at top to reach need_len
    """
    T, N = arr.shape
    if T >= need_len:
        return arr[-need_len:, :]
    pad = np.zeros((need_len - T, N), dtype=arr.dtype)
    return np.concatenate([pad, arr], axis=0)


def forecast_direct_batch(model: TimesNet, last_seq: torch.Tensor) -> torch.Tensor:
    return model(last_seq)  # [B, H, N]


def forecast_recursive_batch(model: TimesNet, last_seq: torch.Tensor, H: int) -> torch.Tensor:
    outs = []
    seq = last_seq
    for _ in range(H):
        y1 = model(seq)  # [B, 1, N]
        outs.append(y1)
        seq = torch.cat([seq[:, 1:, :], y1], dim=1)
    return torch.cat(outs, dim=1)


def predict_once(cfg: Dict) -> str:
    art_dir = cfg["artifacts"]["dir"]
    cfg_used = io_utils.load_yaml(os.path.join(art_dir, cfg["artifacts"]["config_file"]))
    cfg_used.setdefault("artifacts", {}).update(cfg["artifacts"])
    for k, v in cfg.items():
        if k in {"model", "artifacts"}:
            continue
        if isinstance(v, dict):
            cfg_used.setdefault(k, {}).update(v)
        else:
            cfg_used[k] = v

    device = _select_device(cfg_used["train"]["device"])
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision(cfg_used["train"]["matmul_precision"])
    console().print(f"[bold green]Predict device:[/bold green] {device}")

    art_dir = cfg_used["artifacts"]["dir"]
    model_file = os.path.join(art_dir, cfg_used["artifacts"]["model_file"])
    scaler_meta = io_utils.load_pickle(os.path.join(art_dir, cfg_used["artifacts"]["scaler_file"]))
    schema = io_utils.load_json(os.path.join(art_dir, cfg_used["artifacts"]["schema_file"]))

    ids: List[str] = list(scaler_meta["ids"])
    method = scaler_meta["method"]
    scaler = scaler_meta["scaler"]

    # Build model
    model = TimesNet(
        input_len=int(cfg_used["model"]["input_len"]),
        pred_len=int(cfg_used["model"]["pred_len"]),
        d_model=int(cfg_used["model"]["d_model"]),
        n_layers=int(cfg_used["model"]["n_layers"]),
        k_periods=int(cfg_used["model"]["k_periods"]),
        kernel_set=list(cfg_used["model"]["inception_kernel_set"]),
        dropout=float(cfg_used["model"]["dropout"]),
        activation=str(cfg_used["model"]["activation"]),
        mode=str(cfg_used["model"]["mode"]),
    ).to(device)
    # Lazily constructed layers depend on number of series (channels).
    dummy = torch.zeros(1, len(ids), 1, device=device)
    model._build_lazy(N=len(ids), L=1, x=dummy)
    state = torch.load(model_file, map_location="cpu")
    # Checkpoints saved with torch.compile or DataParallel may prefix parameter names.
    clean_state = {
        k.replace("_orig_mod.", "").replace("module.", ""): v
        for k, v in state.items()
    }
    model.load_state_dict(clean_state)
    model.eval()
    if cfg_used["train"]["channels_last"]:
        model = maybe_channels_last(model, True)

    # Iterate test parts
    test_dir = cfg_used["data"]["test_dir"]
    sample = pd.read_csv(
        cfg_used["data"]["sample_submission"], encoding="utf-8-sig"
    )
    preds_by_test: Dict[str, pd.DataFrame] = {}

    test_files = sorted(glob(os.path.join(test_dir, "TEST_*.csv")))
    for fp in test_files:
        df = pd.read_csv(fp)
        wide = io_utils.pivot_long_to_wide(
            df, date_col=schema["date"], id_col=schema["id"], target_col=schema["target"],
            fill_missing_dates=cfg_used["data"]["fill_missing_dates"], fillna0=True
        )
        # align columns to training ids
        wide = wide.reindex(columns=ids).fillna(0.0)
        X = wide.values.astype(np.float32)
        # transform by scaler
        Xn = np.zeros_like(X, dtype=np.float32)
        for j, c in enumerate(ids):
            if method == "zscore":
                mu, sd = scaler[c]
                Xn[:, j] = (X[:, j] - mu) / (sd if sd != 0 else 1.0)
            elif method == "minmax":
                mn, mx = scaler[c]
                rng = (mx - mn) if (mx - mn) != 0 else 1.0
                Xn[:, j] = (X[:, j] - mn) / rng
            else:
                Xn[:, j] = X[:, j]

        # take last input_len with left zero padding if needed
        L = int(cfg_used["model"]["input_len"])
        H = int(cfg_used["model"]["pred_len"])
        last_seq = _pad_left_zeros(Xn, need_len=L)  # [L, N]
        xb = torch.from_numpy(last_seq).unsqueeze(0).to(device, non_blocking=True)  # [1, L, N]

        with torch.inference_mode(), amp_autocast(cfg_used["train"]["amp"] and device.type == "cuda"):
            if cfg_used["model"]["mode"] == "direct":
                out = forecast_direct_batch(model, xb)  # [1, H, N]
            else:
                out = forecast_recursive_batch(model, xb, H)  # [1, H, N]

        Pn = out.squeeze(0).float().cpu().numpy()  # [H, N]
        # inverse transform & clip
        P = io_utils.inverse_transform(Pn, ids, scaler, method=method)
        P = np.clip(P, 0.0, None)
        last_date = wide.index.max()
        forecast_idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=H, freq="D")
        pred_df = pd.DataFrame(P, index=forecast_idx, columns=ids)

        test_name = os.path.splitext(os.path.basename(fp))[0]
        preds_by_test[test_name] = pred_df

    # Format submission
    sub = io_utils.format_submission(
        sample,
        preds_by_test,
        cfg_used["submission"].get("date_col", schema["date"]),
    )
    os.makedirs(os.path.dirname(cfg_used["submission"]["out_path"]), exist_ok=True)
    sub.to_csv(cfg_used["submission"]["out_path"], index=False, encoding="utf-8-sig")
    console().print(f"[bold green]Saved submission:[/bold green] {cfg_used['submission']['out_path']}")
    return cfg_used["submission"]["out_path"]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()
    cfg = Config.from_files(args.config, overrides=args.override).to_dict()
    predict_once(cfg)


if __name__ == "__main__":
    main()
