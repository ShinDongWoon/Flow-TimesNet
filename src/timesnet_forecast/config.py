from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
import copy
import yaml
import os


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


@dataclass
class Config:
    raw: Dict[str, Any]

    @staticmethod
    def from_files(config_path: str, overrides: Iterable[str] | None = None) -> "Config":
        base = load_yaml(config_path)
        if overrides:
            base = apply_overrides(base, overrides)
        # Backward compatibility: rename inception_kernel_set -> kernel_set
        model_cfg = base.setdefault("model", {})
        if "inception_kernel_set" in model_cfg and "kernel_set" not in model_cfg:
            model_cfg["kernel_set"] = model_cfg.pop("inception_kernel_set")
        # Inject defaults for newly introduced static/id embedding knobs to
        # maintain backwards compatibility with older config files that do not
        # specify them explicitly.
        model_cfg.setdefault("id_embed_dim", 32)
        # Preserve the legacy fallback behaviour where omitting static_proj_dim
        # keeps the projection width tied to the input feature dimension.
        model_cfg.setdefault("static_proj_dim", None)
        model_cfg.setdefault("static_layernorm", True)
        data_cfg = base.setdefault("data", {})
        time_cfg = TimeFeatureConfig.from_mapping(data_cfg.get("time_features"))
        data_cfg["time_features"] = time_cfg.to_dict()
        return Config(raw=base)

    def get(self, path: str, default: Any = None) -> Any:
        cur = self.raw
        for p in path.split("."):
            if p not in cur:
                return default
            cur = cur[p]
        return cur

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.raw)
