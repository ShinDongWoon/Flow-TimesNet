from __future__ import annotations

import os
from glob import glob
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from pandas.tseries.frequencies import to_offset

from .config import Config
from .utils.logging import console
from .utils.torch_opt import amp_autocast
from .utils import io as io_utils
from .models.timesnet import TimesNet
from .utils.time_features import build_time_features


def _select_device(req: str) -> torch.device:
    if req == "cuda" and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")
def _invoke_model(
    model: TimesNet,
    xb: torch.Tensor,
    *,
    x_mark: torch.Tensor | None = None,
    series_static: torch.Tensor | None = None,
    series_ids: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    kwargs: Dict[str, torch.Tensor] = {}
    if x_mark is not None:
        kwargs["x_mark"] = x_mark
    if series_static is not None:
        kwargs["series_static"] = series_static
    if series_ids is not None:
        kwargs["series_ids"] = series_ids
    try:
        return model(xb, **kwargs)
    except TypeError as err:
        err_str = str(err)
        for key in ["series_static", "series_ids", "x_mark"]:
            if key in kwargs and key in err_str:
                kwargs.pop(key)
                try:
                    return model(xb, **kwargs)
                except TypeError as inner_err:
                    err_str = str(inner_err)
                    continue
        raise


def forecast_direct_batch(
    model: TimesNet,
    last_seq: torch.Tensor,
    x_mark: torch.Tensor | None = None,
    series_static: torch.Tensor | None = None,
    series_ids: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _invoke_model(
        model,
        last_seq,
        x_mark=x_mark,
        series_static=series_static,
        series_ids=series_ids,
    )


def forecast_recursive_batch(
    model: TimesNet,
    last_seq: torch.Tensor,
    H: int,
    x_mark: torch.Tensor | None = None,
    y_mark: torch.Tensor | None = None,
    series_static: torch.Tensor | None = None,
    series_ids: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rates: List[torch.Tensor] = []
    dispersions: List[torch.Tensor] = []
    seq = last_seq
    mark_seq = x_mark
    for step in range(H):
        rate_step, dispersion_step = _invoke_model(
            model,
            seq,
            x_mark=mark_seq,
            series_static=series_static,
            series_ids=series_ids,
        )
        rates.append(rate_step)
        dispersions.append(dispersion_step)
        seq = torch.cat([seq[:, 1:, :], rate_step], dim=1)
        if mark_seq is not None:
            if y_mark is None:
                raise ValueError(
                    "Temporal features provided for history but missing future marks during recursive forecast"
                )
            if y_mark.size(1) <= step:
                raise ValueError(
                    "y_mark does not provide enough future steps for recursive forecasting"
                )
            next_mark = y_mark[:, step : step + 1, :]
            mark_seq = torch.cat([mark_seq[:, 1:, :], next_mark], dim=1)
    return torch.cat(rates, dim=1), torch.cat(dispersions, dim=1)


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
    data_time_cfg = dict(cfg_used.get("data", {}).get("time_features") or {})
    time_feature_meta = scaler_meta.get("time_features") or {}
    meta_config = dict(time_feature_meta.get("config") or data_time_cfg)
    meta_enabled = bool(time_feature_meta.get("enabled", meta_config.get("enabled", False)))
    meta_dim = int(time_feature_meta.get("feature_dim", meta_config.get("feature_dim", 0)) or 0)
    meta_freq = time_feature_meta.get("freq") or meta_config.get("freq")
    meta_config.setdefault("enabled", meta_enabled)
    cfg_used.setdefault("data", {}).setdefault("time_features", {}).update(
        {"feature_dim": meta_dim, "freq": meta_freq, "enabled": meta_enabled}
    )
    time_features_enabled = bool(meta_enabled and meta_dim > 0)

    static_features_np = None
    static_feature_ids: List[str] | None = None
    static_file = cfg_used["artifacts"].get("static_file")
    if static_file:
        static_path = static_file
        if not os.path.isabs(static_path):
            static_path = os.path.join(art_dir, static_path)
        try:
            static_payload = io_utils.load_pickle(static_path)
        except FileNotFoundError:
            console().print(
                f"[yellow]Static feature artifact not found at {static_path}; falling back to scaler metadata.[/yellow]"
            )
        except OSError as err:
            console().print(
                f"[yellow]Failed to load static feature artifact {static_path}: {err}; falling back to scaler metadata.[/yellow]"
            )
        except Exception as err:  # noqa: BLE001
            console().print(
                f"[yellow]Error loading static feature artifact {static_path}: {err}; falling back to scaler metadata.[/yellow]"
            )
        else:
            if isinstance(static_payload, dict):
                static_features_np = static_payload.get("static_features")
                payload_ids = static_payload.get("ids") or static_payload.get("series_ids")
                if payload_ids is not None:
                    static_feature_ids = list(payload_ids)
            elif isinstance(static_payload, np.ndarray):
                static_features_np = static_payload
            else:
                console().print(
                    f"[yellow]Unsupported static feature artifact type {type(static_payload)!r}; falling back to scaler metadata.[/yellow]"
                )
                static_features_np = None
            if static_features_np is None:
                console().print(
                    f"[yellow]Static feature artifact {static_path} did not contain features; falling back to scaler metadata.[/yellow]"
                )

    if static_features_np is None:
        static_features_np = scaler_meta.get("static_features")
        static_feature_ids = static_feature_ids or ids

    static_tensor_full: torch.Tensor | None = None
    if static_features_np is not None:
        static_arr = np.asarray(static_features_np, dtype=np.float32)
        if static_arr.ndim == 1:
            static_arr = static_arr.reshape(-1, 1)
        if static_arr.ndim != 2:
            console().print(
                f"[yellow]Expected 2D static features but received shape {static_arr.shape}; ignoring static features.[/yellow]"
            )
        else:
            base_ids = list(static_feature_ids or ids)
            if len(base_ids) == 0 and static_arr.shape[0] > 0:
                base_ids = ids[: static_arr.shape[0]]
            if static_arr.shape[0] != len(base_ids):
                console().print(
                    f"[yellow]Static feature count ({static_arr.shape[0]}) does not match provided id list length ({len(base_ids)}); aligning with overlap and zero-filling missing ids.[/yellow]"
                )
            limit = min(static_arr.shape[0], len(base_ids))
            id_to_row = {base_ids[i]: i for i in range(limit)}
            static_tensor_full = torch.from_numpy(static_arr).to(
                device=device, dtype=torch.float32
            )
            feat_dim = int(static_tensor_full.size(1)) if static_tensor_full.ndim == 2 else 0
            aligned = torch.zeros(
                (len(ids), feat_dim),
                dtype=static_tensor_full.dtype,
                device=device,
            )
            missing_static_ids: List[str] = []
            for pos, series_id in enumerate(ids):
                row_idx = id_to_row.get(series_id)
                if row_idx is None:
                    missing_static_ids.append(series_id)
                    continue
                aligned[pos] = static_tensor_full[row_idx]
            if missing_static_ids:
                console().print(
                    f"[yellow]Static features missing for {len(missing_static_ids)} series; zero-filled values will be used.[/yellow]"
                )
            static_tensor_full = aligned

    id_position_map = {series_id: idx for idx, series_id in enumerate(ids)}
    full_series_ids_tensor = torch.arange(len(ids), dtype=torch.long, device=device)

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
    model_cfg = cfg_used["model"]
    d_model = int(model_cfg["d_model"])
    d_ff = int(model_cfg.get("d_ff", 4 * d_model))
    model_cfg["d_ff"] = d_ff
    bottleneck_ratio = float(model_cfg.get("bottleneck_ratio", 1.0))
    model_cfg["bottleneck_ratio"] = bottleneck_ratio

    id_embed_dim = int(model_cfg.get("id_embed_dim", 32))
    static_proj_cfg = model_cfg.get("static_proj_dim", 32)
    static_proj_dim = None if static_proj_cfg is None else int(static_proj_cfg)
    static_layernorm = bool(model_cfg.get("static_layernorm", True))
    model_cfg["id_embed_dim"] = id_embed_dim
    model_cfg["static_proj_dim"] = static_proj_dim
    model_cfg["static_layernorm"] = static_layernorm

    model = TimesNet(
        input_len=input_len,
        pred_len=pred_len,
        d_model=d_model,
        d_ff=d_ff,
        n_layers=int(model_cfg["n_layers"]),
        k_periods=int(model_cfg["k_periods"]),
        min_period_threshold=min_period_threshold,
        kernel_set=list(model_cfg["kernel_set"]),
        dropout=float(model_cfg["dropout"]),
        activation=str(model_cfg["activation"]),
        mode=str(model_cfg["mode"]),
        bottleneck_ratio=bottleneck_ratio,
        channels_last=cfg_used["train"]["channels_last"],
        use_checkpoint=use_checkpoint,
        use_embedding_norm=bool(model_cfg.get("use_embedding_norm", True)),
        min_sigma=min_sigma_scalar,
        min_sigma_vector=min_sigma_vector_tensor,
        id_embed_dim=id_embed_dim,
        static_proj_dim=static_proj_dim,
        static_layernorm=static_layernorm,
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
        warmup_kwargs = {}
        if time_features_enabled and meta_dim > 0:
            warmup_kwargs["x_mark"] = torch.zeros(
                1,
                input_len,
                meta_dim,
                device=device,
                dtype=torch.float32,
            )
        if static_tensor_full is not None:
            warmup_kwargs["series_static"] = static_tensor_full
        if full_series_ids_tensor is not None:
            warmup_kwargs["series_ids"] = full_series_ids_tensor
        if warmup_kwargs:
            try:
                model(dummy, **warmup_kwargs)
            except TypeError as err:
                err_str = str(err)
                if "series_static" in err_str or "series_ids" in err_str:
                    model(dummy)
                else:
                    raise
        else:
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
        columns = list(wide.columns)
        if columns == ids:
            series_ids_tensor = full_series_ids_tensor
            static_tensor = static_tensor_full
        else:
            gather_positions: List[int] = []
            unknown_cols: List[str] = []
            for col in columns:
                idx = id_position_map.get(col)
                if idx is None:
                    unknown_cols.append(col)
                else:
                    gather_positions.append(idx)
            if unknown_cols:
                raise ValueError(
                    f"Test series '{fp}' contains unknown ids not seen during training: {unknown_cols}"
                )
            if gather_positions:
                index_tensor = torch.tensor(
                    gather_positions, dtype=torch.long, device=device
                )
            else:
                index_tensor = torch.empty(0, dtype=torch.long, device=device)
            series_ids_tensor = index_tensor
            if static_tensor_full is not None:
                if index_tensor.numel() == 0:
                    static_tensor = static_tensor_full.new_zeros(
                        (0, static_tensor_full.size(1))
                    )
                else:
                    static_tensor = torch.index_select(
                        static_tensor_full, 0, index_tensor
                    )
            else:
                static_tensor = None
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
        x_mark_tensor: torch.Tensor | None = None
        y_mark_tensor: torch.Tensor | None = None
        if time_features_enabled:
            history_index = pd.DatetimeIndex(wide.index)
            recent_index = history_index[-input_len:]
            active_cfg = dict(meta_config)
            active_cfg["enabled"] = True
            freq_str = meta_freq or cfg_used.get("data", {}).get("time_features", {}).get("freq")
            if freq_str is None:
                freq_str = pd.infer_freq(history_index)
            if freq_str is None:
                console().print(
                    "[yellow]Unable to infer frequency for time features during prediction; temporal marks disabled for this batch.[/yellow]"
                )
            else:
                try:
                    offset = to_offset(freq_str)
                except (ValueError, TypeError) as err:
                    console().print(
                        f"[yellow]Invalid frequency '{freq_str}' for time features ({err}); disabling temporal marks for this batch.[/yellow]"
                    )
                else:
                    future_index = pd.date_range(
                        recent_index[-1] + offset, periods=pred_len, freq=offset
                    )
                    combined_index = recent_index.append(future_index)
                    marks_np = build_time_features(combined_index, active_cfg)
                    if marks_np.shape[1] != meta_dim:
                        console().print(
                            "[yellow]Time feature dimension mismatch during prediction; temporal marks disabled for this batch.[/yellow]"
                        )
                    else:
                        x_mark_np = marks_np[:input_len]
                        y_mark_np = marks_np[input_len:]
                        x_mark_tensor = torch.from_numpy(x_mark_np).unsqueeze(0)
                        y_mark_tensor = torch.from_numpy(y_mark_np).unsqueeze(0)
        if cfg_used["train"]["channels_last"]:
            xb = xb.unsqueeze(-1).to(
                device=device,
                memory_format=torch.channels_last,
                non_blocking=True,
            ).squeeze(-1)
        else:
            xb = xb.to(device, non_blocking=True)
        if x_mark_tensor is not None:
            x_mark_tensor = x_mark_tensor.to(
                device=device, dtype=xb.dtype, non_blocking=True
            )
        if y_mark_tensor is not None:
            y_mark_tensor = y_mark_tensor.to(
                device=device, dtype=xb.dtype, non_blocking=True
            )

        with torch.inference_mode(), amp_autocast(cfg_used["train"]["amp"] and device.type == "cuda"):
            if cfg_used["model"]["mode"] == "direct":
                rate_pred, _ = forecast_direct_batch(
                    model,
                    xb,
                    x_mark=x_mark_tensor,
                    series_static=static_tensor,
                    series_ids=series_ids_tensor,
                )
            else:
                rate_pred, _ = forecast_recursive_batch(
                    model,
                    xb,
                    pred_len,
                    x_mark=x_mark_tensor,
                    y_mark=y_mark_tensor,
                    series_static=static_tensor,
                    series_ids=series_ids_tensor,
                )

        Pn = rate_pred.squeeze(0).float().cpu().numpy()  # [H, N]
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
