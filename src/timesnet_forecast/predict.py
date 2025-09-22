from __future__ import annotations

import os
from glob import glob
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch

from .config import Config
from .utils.logging import console
from .utils.torch_opt import amp_autocast
from .utils import io as io_utils
from .models.timesnet import TimesNet


def _select_device(req: str) -> torch.device:
    if req == "cuda" and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")
def forecast_direct_batch(
    model: TimesNet, last_seq: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    return model(last_seq)


def forecast_recursive_batch(
    model: TimesNet, last_seq: torch.Tensor, H: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    mus: List[torch.Tensor] = []
    sigmas: List[torch.Tensor] = []
    seq = last_seq
    for _ in range(H):
        mu_step, sigma_step = model(seq)
        mus.append(mu_step)
        sigmas.append(sigma_step)
        seq = torch.cat([seq[:, 1:, :], mu_step], dim=1)
    return torch.cat(mus, dim=1), torch.cat(sigmas, dim=1)


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
    min_period_threshold = int(cfg_used["model"].get("min_period_threshold", 1))

    train_cfg = cfg_used["train"]
    use_checkpoint = bool(train_cfg.get("use_checkpoint", False))
    if cfg_used["train"].get("cuda_graphs", False):
        use_checkpoint = False

    min_sigma_scalar = float(train_cfg.get("min_sigma_effective", 1e-3))
    min_sigma_vector_cfg = train_cfg.get("min_sigma_vector")
    min_sigma_vector_tensor: torch.Tensor | None = None
    if min_sigma_vector_cfg is not None:
        min_sigma_vector_tensor = torch.as_tensor(
            min_sigma_vector_cfg, dtype=torch.float32
        )
        if min_sigma_vector_tensor.numel() == 0:
            min_sigma_vector_tensor = None
        else:
            min_sigma_vector_tensor = min_sigma_vector_tensor.reshape(1, 1, -1)

    input_len = int(cfg_used["model"]["input_len"])
    pred_len = int(cfg_used["model"]["pred_len"])
    model = TimesNet(
        input_len=input_len,
        pred_len=pred_len,
        d_model=int(cfg_used["model"]["d_model"]),
        n_layers=int(cfg_used["model"]["n_layers"]),
        k_periods=int(cfg_used["model"]["k_periods"]),
        min_period_threshold=min_period_threshold,
        kernel_set=list(cfg_used["model"]["kernel_set"]),
        dropout=float(cfg_used["model"]["dropout"]),
        activation=str(cfg_used["model"]["activation"]),
        mode=str(cfg_used["model"]["mode"]),
        channels_last=cfg_used["train"]["channels_last"],
        use_checkpoint=use_checkpoint,
        use_embedding_norm=bool(cfg_used["model"].get("use_embedding_norm", True)),
        min_sigma=min_sigma_scalar,
        min_sigma_vector=min_sigma_vector_tensor,
    ).to(device)
    # Lazily construct layers by mirroring the training warm-up.
    with torch.no_grad():
        dummy = torch.zeros(
            1,
            input_len,
            len(ids),
            device=device,
        )
        if cfg_used["train"]["channels_last"]:
            dummy = dummy.unsqueeze(-1).to(memory_format=torch.channels_last).squeeze(-1)
        model(dummy)
    state = torch.load(model_file, map_location="cpu")
    # Checkpoints saved with torch.compile or DataParallel may prefix parameter names.
    clean_state = {
        k.replace("_orig_mod.", "").replace("module.", ""): v
        for k, v in state.items()
    }
    min_sigma_buffer = getattr(model, "min_sigma_vector", None)
    if isinstance(min_sigma_buffer, torch.Tensor) and min_sigma_buffer.numel() > 0:
        checkpoint_value = clean_state.get("min_sigma_vector")
        buffer_cpu = min_sigma_buffer.detach().to("cpu")
        if isinstance(checkpoint_value, torch.Tensor):
            buffer_cpu = buffer_cpu.to(dtype=checkpoint_value.dtype)
        clean_state["min_sigma_vector"] = buffer_cpu
    else:
        clean_state.pop("min_sigma_vector", None)
    model.load_state_dict(clean_state, strict=True)
    if cfg_used["train"]["channels_last"]:
        model.to(memory_format=torch.channels_last)
    model.eval()

    # Iterate test parts
    test_dir = cfg_used["data"]["test_dir"]
    sample = pd.read_csv(
        cfg_used["data"]["sample_submission"], encoding="utf-8-sig"
    )
    pred_list: List[pd.DataFrame] = []

    test_files = sorted(glob(os.path.join(test_dir, "TEST_*.csv")))
    for fp in test_files:
        df = pd.read_csv(fp)
        wide = io_utils.pivot_long_to_wide(
            df, date_col=schema["date"], id_col=schema["id"], target_col=schema["target"],
            fill_missing_dates=cfg_used["data"]["fill_missing_dates"], fillna0=True
        )
        if cfg_used.get("preprocess", {}).get("clip_negative", False):
            wide = wide.clip(lower=0.0)
        # align columns to training ids
        wide = wide.reindex(columns=ids).fillna(0.0)
        X = wide.values.astype(np.float32)
        # transform by scaler
        if method == "none" or scaler is None:
            Xn = X
        else:
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

        if Xn.shape[0] < input_len:
            raise ValueError(
                f"Test series '{fp}' shorter than required input_len={input_len}"
            )
        last_seq = Xn[-input_len:, :]
        xb = torch.from_numpy(last_seq).unsqueeze(0)
        if cfg_used["train"]["channels_last"]:
            xb = xb.unsqueeze(-1).to(
                device=device,
                memory_format=torch.channels_last,
                non_blocking=True,
            ).squeeze(-1)
        else:
            xb = xb.to(device, non_blocking=True)

        with torch.inference_mode(), amp_autocast(cfg_used["train"]["amp"] and device.type == "cuda"):
            if cfg_used["model"]["mode"] == "direct":
                mu_pred, _ = forecast_direct_batch(model, xb)
            else:
                mu_pred, _ = forecast_recursive_batch(model, xb, pred_len)

        Pn = mu_pred.squeeze(0).float().cpu().numpy()  # [H, N]
        # inverse transform & clip
        P = io_utils.inverse_transform(Pn, ids, scaler, method=method)
        P = np.clip(P, 0.0, None)

        pred_df = pd.DataFrame(P, columns=ids)
        test_name = os.path.splitext(os.path.basename(fp))[0]
        # Attach row keys for later concatenation
        pred_df["row_key"] = [f"{test_name}+D{i+1}" for i in range(len(pred_df))]
        pred_df = pred_df.set_index("row_key")
        pred_list.append(pred_df)

    # Format submission
    preds = io_utils.merge_forecasts(pred_list)
    sub = io_utils.format_submission(sample, preds)
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
