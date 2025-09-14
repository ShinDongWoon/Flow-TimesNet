from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable
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


@dataclass
class Config:
    raw: Dict[str, Any]

    @staticmethod
    def from_files(config_path: str, overrides: Iterable[str] | None = None) -> "Config":
        base = load_yaml(config_path)
        if overrides:
            base = apply_overrides(base, overrides)
        # Backward compatibility: rename inception_kernel_set -> kernel_set
        model_cfg = base.get("model", {})
        if "inception_kernel_set" in model_cfg and "kernel_set" not in model_cfg:
            model_cfg["kernel_set"] = model_cfg.pop("inception_kernel_set")
            base["model"] = model_cfg
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
