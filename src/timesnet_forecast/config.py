from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple
import copy
import os
import textwrap
import yaml


def _deep_get(d: Dict[str, Any], path: Iterable[str]) -> Any:
    cur = d
    for p in path:
        cur = cur[p]
    return cur


def _deep_set(d: Dict[str, Any], path: Iterable[str], value: Any) -> None:
    cur = d
    path = list(path)
    for p in path[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[path[-1]] = value


def _parse_scalar(s: str) -> Any:
    # Try to parse booleans, null, ints, floats via YAML-safe loader heuristic
    try:
        v = yaml.safe_load(s)
        return v
    except Exception:
        return s


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def apply_overrides(cfg: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    """
    Apply CLI overrides like a.b.c=value into nested dict.
    """
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            continue
        key, val = item.split("=", 1)
        path = key.strip().split(".")
        _deep_set(out, path, _parse_scalar(val.strip()))
    return out


DEFAULT_TIME_FEATURES: List[str] = [
    "day_of_week",
    "day_of_month",
    "month",
    "day_of_year",
]


@dataclass
class TimeFeatureConfig:
    enabled: bool = False
    features: List[str] = field(default_factory=lambda: DEFAULT_TIME_FEATURES.copy())
    encoding: Any = "cyclical"
    normalize: bool = True
    freq: Optional[str] = None
    feature_dim: Optional[int] = None

    @classmethod
    def from_mapping(cls, mapping: Dict[str, Any] | None) -> "TimeFeatureConfig":
        if mapping is None:
            return cls()
        data = dict(mapping)
        enabled = bool(data.get("enabled", False))
        feats = data.get("features")
        if enabled:
            if not isinstance(feats, list) or not feats:
                raise ValueError(
                    "data.time_features.features must be a non-empty list when enabled is true"
                )
            features = [str(f) for f in feats]
        else:
            if isinstance(feats, list) and feats:
                features = [str(f) for f in feats]
            else:
                features = DEFAULT_TIME_FEATURES.copy()
        encoding = data.get("encoding", "cyclical")
        normalize = bool(data.get("normalize", True))
        freq = data.get("freq")
        feature_dim = data.get("feature_dim")
        return cls(
            enabled=enabled,
            features=features,
            encoding=encoding,
            normalize=normalize,
            freq=freq,
            feature_dim=feature_dim if feature_dim is None else int(feature_dim),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "enabled": self.enabled,
            "features": list(self.features),
            "encoding": self.encoding,
            "normalize": self.normalize,
        }
        if self.freq is not None:
            payload["freq"] = self.freq
        if self.feature_dim is not None:
            payload["feature_dim"] = int(self.feature_dim)
        return payload


@dataclass(frozen=True)
class WindowConfig:
    """Sliding window specification shared across training and inference."""

    input_len: int
    pred_len: int
    stride: int = 1
    short_series_strategy: str = "error"  # error|repeat|pad
    pad_value: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_len", int(self.input_len))
        object.__setattr__(self, "pred_len", int(self.pred_len))
        object.__setattr__(self, "stride", max(1, int(self.stride)))
        strategy = str(self.short_series_strategy).lower()
        if strategy not in {"error", "repeat", "pad"}:
            raise ValueError(
                "window.short_series_strategy must be one of {'error', 'repeat', 'pad'}"
            )
        object.__setattr__(self, "short_series_strategy", strategy)

    @property
    def total_length(self) -> int:
        return int(self.input_len + self.pred_len)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_len": int(self.input_len),
            "pred_len": int(self.pred_len),
            "stride": int(self.stride),
            "short_series_strategy": self.short_series_strategy,
            "pad_value": float(self.pad_value),
        }


@dataclass(frozen=True)
class ModelConfig:
    mode: str
    d_model: int
    d_ff: int
    n_layers: int
    k_periods: int
    min_period_threshold: int
    kernel_set: List[Any]
    dropout: float
    activation: str
    bottleneck_ratio: float
    use_embedding_norm: bool
    id_embed_dim: int
    static_proj_dim: Optional[int]
    static_layernorm: bool

    @classmethod
    def from_mapping(cls, mapping: Dict[str, Any], window: WindowConfig) -> "ModelConfig":
        data = dict(mapping)
        mode = str(data.get("mode", "direct"))
        if mode not in {"direct", "recursive"}:
            raise ValueError("model.mode must be one of {'direct', 'recursive'}")
        d_model = int(data.get("d_model", 128))
        d_ff = int(data.get("d_ff", 4 * d_model))
        kernel = data.get("kernel_set", [])
        if isinstance(kernel, tuple):
            kernel = list(kernel)
        if not isinstance(kernel, list) or not kernel:
            raise ValueError("model.kernel_set must be a non-empty list of kernel specifications")
        static_proj_cfg = data.get("static_proj_dim", 32)
        static_proj_dim = None if static_proj_cfg in {None, "null"} else int(static_proj_cfg)
        return cls(
            mode=mode,
            d_model=d_model,
            d_ff=d_ff,
            n_layers=int(data.get("n_layers", 2)),
            k_periods=int(data.get("k_periods", 2)),
            min_period_threshold=int(data.get("min_period_threshold", 1)),
            kernel_set=list(kernel),
            dropout=float(data.get("dropout", 0.1)),
            activation=str(data.get("activation", "gelu")),
            bottleneck_ratio=float(data.get("bottleneck_ratio", 1.0)),
            use_embedding_norm=bool(data.get("use_embedding_norm", True)),
            id_embed_dim=int(data.get("id_embed_dim", 32)),
            static_proj_dim=static_proj_dim,
            static_layernorm=bool(data.get("static_layernorm", True)),
        )

    def to_dict(self, window: WindowConfig) -> Dict[str, Any]:
        payload = {
            "mode": self.mode,
            "input_len": int(window.input_len),
            "pred_len": int(window.pred_len),
            "d_model": int(self.d_model),
            "d_ff": int(self.d_ff),
            "n_layers": int(self.n_layers),
            "k_periods": int(self.k_periods),
            "min_period_threshold": int(self.min_period_threshold),
            "kernel_set": list(self.kernel_set),
            "dropout": float(self.dropout),
            "activation": self.activation,
            "bottleneck_ratio": float(self.bottleneck_ratio),
            "use_embedding_norm": bool(self.use_embedding_norm),
            "id_embed_dim": int(self.id_embed_dim),
            "static_proj_dim": self.static_proj_dim,
            "static_layernorm": bool(self.static_layernorm),
        }
        return payload


@dataclass(frozen=True)
class DataConfig:
    train_csv: str
    test_dir: str
    sample_submission: str
    date_col: str
    target_col: str
    id_col: str
    min_context_days: Optional[int]
    horizon: Optional[int]
    fill_missing_dates: bool
    encoding: str
    schema_detection_policy: str
    schema_evolution_policy: str
    time_features: TimeFeatureConfig

    @classmethod
    def from_mapping(cls, mapping: Dict[str, Any]) -> "DataConfig":
        data = dict(mapping)
        time_cfg = TimeFeatureConfig.from_mapping(data.get("time_features"))
        return cls(
            train_csv=str(data.get("train_csv", "")),
            test_dir=str(data.get("test_dir", "")),
            sample_submission=str(data.get("sample_submission", "")),
            date_col=str(data.get("date_col", "date")),
            target_col=str(data.get("target_col", "target")),
            id_col=str(data.get("id_col", "id")),
            min_context_days=(
                None if data.get("min_context_days") is None else int(data.get("min_context_days"))
            ),
            horizon=(None if data.get("horizon") is None else int(data.get("horizon"))),
            fill_missing_dates=bool(data.get("fill_missing_dates", True)),
            encoding=str(data.get("encoding", "utf-8")),
            schema_detection_policy=str(data.get("schema_detection_policy", "infer")),
            schema_evolution_policy=str(data.get("schema_evolution_policy", "warn")),
            time_features=time_cfg,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "train_csv": self.train_csv,
            "test_dir": self.test_dir,
            "sample_submission": self.sample_submission,
            "date_col": self.date_col,
            "target_col": self.target_col,
            "id_col": self.id_col,
            "min_context_days": self.min_context_days,
            "horizon": self.horizon,
            "fill_missing_dates": self.fill_missing_dates,
            "encoding": self.encoding,
            "schema_detection_policy": self.schema_detection_policy,
            "schema_evolution_policy": self.schema_evolution_policy,
            "time_features": self.time_features.to_dict(),
        }
        return payload


@dataclass(frozen=True)
class TrainConfig:
    device: str
    epochs: int
    batch_size: int
    accumulation_steps: int
    lr_warmup_steps: int
    lr: float
    weight_decay: float
    grad_clip_norm: float
    early_stopping_patience: Optional[int]
    amp: bool
    compile: bool
    deterministic: bool
    cuda_graphs: bool
    use_checkpoint: bool
    min_sigma: float
    min_sigma_method: str
    min_sigma_scale: float
    matmul_precision: str
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int
    channels_last: bool
    use_loss_masking: bool
    val_strategy: str
    val_holdout_days: Optional[int]
    val_rolling_folds: Optional[int]
    val_rolling_step_days: Optional[int]

    @classmethod
    def from_mapping(cls, mapping: Dict[str, Any]) -> "TrainConfig":
        data = dict(mapping)
        val_cfg = dict(data.get("val") or {})
        strategy = str(val_cfg.get("strategy", "holdout"))
        holdout_days = val_cfg.get("holdout_days")
        rolling_folds = val_cfg.get("rolling_folds")
        rolling_step = val_cfg.get("rolling_step_days")
        return cls(
            device=str(data.get("device", "cpu")),
            epochs=int(data.get("epochs", 1)),
            batch_size=max(1, int(data.get("batch_size", 1))),
            accumulation_steps=max(1, int(data.get("accumulation_steps", 1))),
            lr_warmup_steps=int(data.get("lr_warmup_steps", 0)),
            lr=float(data.get("lr", 1e-3)),
            weight_decay=float(data.get("weight_decay", 0.0)),
            grad_clip_norm=float(data.get("grad_clip_norm", 0.0)),
            early_stopping_patience=(
                None
                if data.get("early_stopping_patience") is None
                else int(data.get("early_stopping_patience"))
            ),
            amp=bool(data.get("amp", False)),
            compile=bool(data.get("compile", False)),
            deterministic=bool(data.get("deterministic", False)),
            cuda_graphs=bool(data.get("cuda_graphs", False)),
            use_checkpoint=bool(data.get("use_checkpoint", False)),
            min_sigma=float(data.get("min_sigma", 1e-3)),
            min_sigma_method=str(data.get("min_sigma_method", "global")),
            min_sigma_scale=float(data.get("min_sigma_scale", 0.1)),
            matmul_precision=str(data.get("matmul_precision", "medium")),
            num_workers=int(data.get("num_workers", 0)),
            pin_memory=bool(data.get("pin_memory", False)),
            persistent_workers=bool(data.get("persistent_workers", False)),
            prefetch_factor=int(data.get("prefetch_factor", 2)),
            channels_last=bool(data.get("channels_last", False)),
            use_loss_masking=bool(data.get("use_loss_masking", False)),
            val_strategy=strategy,
            val_holdout_days=(None if holdout_days is None else int(holdout_days)),
            val_rolling_folds=(None if rolling_folds is None else int(rolling_folds)),
            val_rolling_step_days=(None if rolling_step is None else int(rolling_step)),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "device": self.device,
            "epochs": int(self.epochs),
            "early_stopping_patience": self.early_stopping_patience,
            "batch_size": int(self.batch_size),
            "accumulation_steps": int(self.accumulation_steps),
            "lr_warmup_steps": int(self.lr_warmup_steps),
            "lr": float(self.lr),
            "weight_decay": float(self.weight_decay),
            "grad_clip_norm": float(self.grad_clip_norm),
            "amp": self.amp,
            "compile": self.compile,
            "deterministic": self.deterministic,
            "cuda_graphs": self.cuda_graphs,
            "use_checkpoint": self.use_checkpoint,
            "min_sigma": float(self.min_sigma),
            "min_sigma_method": self.min_sigma_method,
            "min_sigma_scale": float(self.min_sigma_scale),
            "matmul_precision": self.matmul_precision,
            "num_workers": int(self.num_workers),
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "prefetch_factor": int(self.prefetch_factor),
            "channels_last": self.channels_last,
            "use_loss_masking": self.use_loss_masking,
            "val": {
                "strategy": self.val_strategy,
                "holdout_days": self.val_holdout_days,
                "rolling_folds": self.val_rolling_folds,
                "rolling_step_days": self.val_rolling_step_days,
            },
        }
        return payload


def _normalise_model_section(base: Dict[str, Any]) -> None:
    model_cfg = base.setdefault("model", {})
    if "inception_kernel_set" in model_cfg and "kernel_set" not in model_cfg:
        model_cfg["kernel_set"] = model_cfg.pop("inception_kernel_set")
    model_cfg.setdefault("id_embed_dim", 32)
    model_cfg.setdefault("static_proj_dim", None)
    model_cfg.setdefault("static_layernorm", True)


def _normalise_time_features(base: Dict[str, Any]) -> TimeFeatureConfig:
    data_cfg = base.setdefault("data", {})
    time_cfg = TimeFeatureConfig.from_mapping(data_cfg.get("time_features"))
    data_cfg["time_features"] = time_cfg.to_dict()
    return time_cfg


def _extract_window(base: Dict[str, Any]) -> Tuple[WindowConfig, Dict[str, Any]]:
    window_raw = dict(base.get("window") or {})
    model_raw = base.setdefault("model", {})
    input_len = window_raw.get("input_len", model_raw.get("input_len"))
    pred_len = window_raw.get("pred_len", model_raw.get("pred_len"))
    if input_len is None or pred_len is None:
        raise ValueError("Configuration must specify model.input_len and model.pred_len")
    stride = window_raw.get("stride", window_raw.get("step", 1))
    strategy = window_raw.get("short_series_strategy", "error")
    pad_value = window_raw.get("pad_value", 0.0)
    window_cfg = WindowConfig(
        input_len=int(input_len),
        pred_len=int(pred_len),
        stride=int(stride),
        short_series_strategy=strategy,
        pad_value=float(pad_value),
    )
    base.setdefault("window", {}).update(window_cfg.to_dict())
    model_raw["input_len"] = int(window_cfg.input_len)
    model_raw["pred_len"] = int(window_cfg.pred_len)
    return window_cfg, model_raw


@dataclass(frozen=True)
class PipelineConfig:
    """Normalised configuration with validation across dependent sections."""

    raw: Dict[str, Any]
    window: WindowConfig
    model: ModelConfig
    data: DataConfig
    train: TrainConfig

    @classmethod
    def from_files(
        cls, config_path: str, overrides: Iterable[str] | None = None
    ) -> "PipelineConfig":
        base = load_yaml(config_path)
        if overrides:
            base = apply_overrides(base, overrides)
        return cls.from_mapping(base)

    @classmethod
    def from_mapping(cls, mapping: Dict[str, Any]) -> "PipelineConfig":
        base = copy.deepcopy(mapping)
        _normalise_model_section(base)
        time_cfg = _normalise_time_features(base)
        artifacts_cfg = base.setdefault("artifacts", {})
        artifacts_cfg.setdefault("signature_file", "model_signature.json")
        artifacts_cfg.setdefault("metadata_file", "metadata.json")
        window_cfg, model_raw = _extract_window(base)
        model_cfg = ModelConfig.from_mapping(model_raw, window_cfg)
        data_cfg = DataConfig.from_mapping(base.get("data", {}))
        train_cfg = TrainConfig.from_mapping(base.get("train", {}))
        base.setdefault("window", {}).update(window_cfg.to_dict())
        base.setdefault("model", {}).update(model_cfg.to_dict(window_cfg))
        base.setdefault("data", {}).setdefault("time_features", time_cfg.to_dict())
        instance = cls(
            raw=base,
            window=window_cfg,
            model=model_cfg,
            data=data_cfg,
            train=train_cfg,
        )
        instance.validate()
        return instance

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.raw)

    def apply_overrides(self, overrides: Iterable[str]) -> "PipelineConfig":
        if not overrides:
            return self
        new_raw = apply_overrides(self.to_dict(), overrides)
        return PipelineConfig.from_mapping(new_raw)

    def validate(self) -> None:
        errors: List[str] = []
        if self.window.input_len <= 0:
            errors.append("window.input_len must be positive")
        if self.window.pred_len <= 0:
            errors.append("window.pred_len must be positive")
        if self.window.stride <= 0:
            errors.append("window.stride must be positive")
        if self.model.min_period_threshold > self.window.input_len:
            errors.append(
                "model.min_period_threshold cannot exceed window.input_len"
            )
        if self.data.min_context_days is not None and (
            self.data.min_context_days < self.window.input_len
        ):
            errors.append(
                "data.min_context_days must be at least window.input_len to ensure sufficient history"
            )
        if self.data.horizon is not None and self.data.horizon < self.window.pred_len:
            errors.append(
                "data.horizon must be at least window.pred_len to cover the forecast horizon"
            )
        if self.train.val_strategy in {"holdout", "rolling"}:
            if self.train.val_holdout_days is None:
                errors.append("train.val.holdout_days must be specified for holdout/rolling validation")
            elif self.train.val_holdout_days < self.window.total_length:
                errors.append(
                    "train.val.holdout_days must be >= window.input_len + window.pred_len"
                )
        if self.train.batch_size <= 0:
            errors.append("train.batch_size must be positive")
        if errors:
            raise ValueError(
                "\n".join(
                    [
                        "Configuration validation failed with the following issues:",
                        *[f"- {err}" for err in errors],
                    ]
                )
            )

    def describe(self) -> str:
        payload = {
            "window": self.window.to_dict(),
            "model": self.model.to_dict(self.window),
            "data": self.data.to_dict(),
            "train": self.train.to_dict(),
        }
        return textwrap.indent(yaml.safe_dump(payload, sort_keys=False), prefix="  ")


# Backwards compatibility: retain the old Config name.
Config = PipelineConfig
