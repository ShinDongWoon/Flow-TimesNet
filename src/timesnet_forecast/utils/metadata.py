from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Sequence

import numpy as np

from ..config import PipelineConfig
from .io import DataSchema, load_json, save_json


METADATA_ARTIFACT_VERSION = "1"
SUPPORTED_METADATA_VERSIONS: tuple[str, ...] = (METADATA_ARTIFACT_VERSION,)


def _upgrade_from_legacy(payload: Mapping[str, Any]) -> Dict[str, Any]:
    upgraded: Dict[str, Any] = dict(payload)
    time_meta = upgraded.get("time_features")
    if isinstance(time_meta, Mapping):
        config = dict(time_meta.get("config") or {})
        if "enabled" not in config and "enabled" in time_meta:
            config.setdefault("enabled", bool(time_meta["enabled"]))
        if time_meta.get("freq") is not None and "freq" not in config:
            config.setdefault("freq", time_meta.get("freq"))
        if time_meta.get("feature_dim") is not None and "feature_dim" not in config:
            config.setdefault("feature_dim", time_meta.get("feature_dim"))
        upgraded["time_features"] = {
            "config": config,
            "enabled": bool(time_meta.get("enabled", config.get("enabled", False))),
            "feature_dim": int(time_meta.get("feature_dim", config.get("feature_dim", 0)) or 0),
        }
        if time_meta.get("freq") is not None:
            upgraded["time_features"]["freq"] = time_meta.get("freq")
    static_meta = upgraded.get("static_features")
    if isinstance(static_meta, Sequence) and not isinstance(static_meta, Mapping):
        names = [str(name) for name in static_meta]
        upgraded["static_features"] = {
            "feature_names": names,
            "feature_dim": len(names),
        }
    upgraded["meta_version"] = METADATA_ARTIFACT_VERSION
    return upgraded


METADATA_MIGRATIONS: Dict[str, Callable[[Mapping[str, Any]], Dict[str, Any]]] = {
    "0": _upgrade_from_legacy,
}


def _ensure_mapping(value: Mapping[str, Any] | None, name: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Metadata artifact missing '{name}' object")
    return dict(value)


def _coerce_window(obj: Mapping[str, Any]) -> Dict[str, Any]:
    required = {"input_len", "pred_len"}
    missing = [key for key in required if key not in obj]
    if missing:
        raise ValueError(
            "Metadata artifact window section missing keys: "
            + ", ".join(sorted(missing))
        )
    strategy = str(obj.get("short_series_strategy", "error")).lower()
    return {
        "input_len": int(obj["input_len"]),
        "pred_len": int(obj["pred_len"]),
        "stride": int(obj.get("stride", 1)),
        "short_series_strategy": strategy,
        "pad_value": float(obj.get("pad_value", 0.0)),
    }


def _coerce_schema(obj: Mapping[str, Any]) -> Dict[str, str]:
    required = {"date", "id", "target"}
    missing = [key for key in required if key not in obj]
    if missing:
        raise ValueError(
            "Metadata artifact schema section missing keys: "
            + ", ".join(sorted(missing))
        )
    return {key: str(obj[key]) for key in required}


def _normalise_time_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    normalised = {
        "enabled": bool(config.get("enabled", False)),
        "features": [str(f) for f in config.get("features", [])],
        "encoding": str(config.get("encoding", "cyclical")),
        "normalize": bool(config.get("normalize", True)),
    }
    if config.get("freq") is not None:
        normalised["freq"] = str(config.get("freq"))
    if config.get("feature_dim") is not None:
        normalised["feature_dim"] = int(config.get("feature_dim"))
    return normalised


def _coerce_time_features(obj: Mapping[str, Any]) -> Dict[str, Any]:
    config_raw = obj.get("config") if isinstance(obj.get("config"), Mapping) else {}
    config = _normalise_time_config(config_raw)
    enabled = bool(obj.get("enabled", config.get("enabled", False)))
    feature_dim = int(obj.get("feature_dim", config.get("feature_dim", 0)) or 0)
    freq = obj.get("freq", config.get("freq"))
    payload: Dict[str, Any] = {
        "config": config,
        "enabled": enabled,
        "feature_dim": feature_dim,
    }
    if freq is not None:
        payload["freq"] = str(freq)
    return payload


def _coerce_static_features(obj: Mapping[str, Any] | None) -> Dict[str, Any]:
    if obj is None:
        return {"feature_names": [], "feature_dim": 0}
    names = obj.get("feature_names")
    if isinstance(names, Iterable) and not isinstance(names, str):
        feature_names = [str(name) for name in names]
    else:
        feature_names = []
    feature_dim = obj.get("feature_dim")
    if feature_dim is None and feature_names:
        feature_dim = len(feature_names)
    return {
        "feature_names": feature_names,
        "feature_dim": int(feature_dim or 0),
    }


@dataclass
class MetadataArtifact:
    meta_version: str
    window: Dict[str, Any]
    schema: Dict[str, str]
    time_features: Dict[str, Any]
    static_features: Dict[str, Any]

    @classmethod
    def from_training(
        cls,
        *,
        window: Any,
        schema: DataSchema,
        time_features: Mapping[str, Any],
        static_features: Mapping[str, Any] | None,
    ) -> "MetadataArtifact":
        window_dict = window.to_dict() if hasattr(window, "to_dict") else dict(window)
        return cls(
            meta_version=METADATA_ARTIFACT_VERSION,
            window=_coerce_window(window_dict),
            schema=_coerce_schema(schema.as_dict()),
            time_features=_coerce_time_features(time_features),
            static_features=_coerce_static_features(static_features),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "MetadataArtifact":
        meta_version = str(payload.get("meta_version", ""))
        window = _coerce_window(_ensure_mapping(payload.get("window"), "window"))
        schema = _coerce_schema(_ensure_mapping(payload.get("schema"), "schema"))
        time_features = _coerce_time_features(
            _ensure_mapping(payload.get("time_features"), "time_features")
        )
        static_features = _coerce_static_features(
            _ensure_mapping(payload.get("static_features"), "static_features")
        )
        return cls(
            meta_version=meta_version,
            window=window,
            schema=schema,
            time_features=time_features,
            static_features=static_features,
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "meta_version": self.meta_version,
            "window": dict(self.window),
            "schema": dict(self.schema),
            "time_features": dict(self.time_features),
            "static_features": dict(self.static_features),
        }

    def validate_config(self, cfg: PipelineConfig) -> None:
        errors: list[str] = []
        if cfg.window.input_len != int(self.window["input_len"]):
            errors.append(
                "window.input_len={0} differs from metadata value {1}".format(
                    cfg.window.input_len, self.window["input_len"]
                )
            )
        if cfg.window.pred_len != int(self.window["pred_len"]):
            errors.append(
                "window.pred_len={0} differs from metadata value {1}".format(
                    cfg.window.pred_len, self.window["pred_len"]
                )
            )
        if cfg.window.stride != int(self.window.get("stride", cfg.window.stride)):
            errors.append(
                "window.stride={0} differs from metadata value {1}".format(
                    cfg.window.stride, self.window.get("stride")
                )
            )
        strategy_expected = str(self.window.get("short_series_strategy", "error"))
        if cfg.window.short_series_strategy != strategy_expected:
            errors.append(
                "window.short_series_strategy='{0}' differs from metadata value '{1}'".format(
                    cfg.window.short_series_strategy, strategy_expected
                )
            )
        schema_dict = self.schema
        if cfg.data.date_col != schema_dict["date"]:
            errors.append(
                "data.date_col='{0}' differs from metadata value '{1}'".format(
                    cfg.data.date_col, schema_dict["date"]
                )
            )
        if cfg.data.id_col != schema_dict["id"]:
            errors.append(
                "data.id_col='{0}' differs from metadata value '{1}'".format(
                    cfg.data.id_col, schema_dict["id"]
                )
            )
        if cfg.data.target_col != schema_dict["target"]:
            errors.append(
                "data.target_col='{0}' differs from metadata value '{1}'".format(
                    cfg.data.target_col, schema_dict["target"]
                )
            )
        cfg_time = _normalise_time_config(cfg.data.time_features.to_dict())
        meta_time = self.time_features
        meta_cfg = _normalise_time_config(meta_time.get("config", {}))
        if bool(cfg_time["enabled"]) != bool(meta_time.get("enabled")):
            errors.append(
                "data.time_features.enabled={0} differs from metadata value {1}".format(
                    cfg_time["enabled"], meta_time.get("enabled")
                )
            )
        if cfg_time["features"] != meta_cfg["features"]:
            errors.append(
                "data.time_features.features={0} differs from metadata value {1}".format(
                    cfg_time["features"], meta_cfg["features"]
                )
            )
        if cfg_time["encoding"] != meta_cfg["encoding"]:
            errors.append(
                "data.time_features.encoding='{0}' differs from metadata value '{1}'".format(
                    cfg_time["encoding"], meta_cfg["encoding"]
                )
            )
        if cfg_time["normalize"] != meta_cfg["normalize"]:
            errors.append(
                "data.time_features.normalize={0} differs from metadata value {1}".format(
                    cfg_time["normalize"], meta_cfg["normalize"]
                )
            )
        freq_expected = meta_time.get("freq")
        if freq_expected is not None and cfg_time.get("freq") not in {None, freq_expected}:
            errors.append(
                "data.time_features.freq='{0}' differs from metadata value '{1}'".format(
                    cfg_time.get("freq"), freq_expected
                )
            )
        cfg_dim = cfg.data.time_features.feature_dim
        if cfg_dim is not None:
            meta_dim = int(meta_time.get("feature_dim", cfg_dim))
            if int(cfg_dim) != meta_dim:
                errors.append(
                    "data.time_features.feature_dim={0} differs from metadata value {1}".format(
                        cfg_dim, meta_dim
                    )
                )
        if errors:
            raise ValueError(
                "Configuration incompatible with metadata artifact:\n"
                + "\n".join(f"- {msg}" for msg in errors)
            )

    def validate_artifacts(
        self,
        *,
        schema: DataSchema,
        scaler_meta: Mapping[str, Any],
        num_series: int | None = None,
    ) -> None:
        errors: list[str] = []
        schema_dict = schema.as_dict()
        for key, expected in self.schema.items():
            actual = schema_dict.get(key)
            if actual != expected:
                errors.append(
                    "Schema column '{0}' stored as '{1}' but metadata expects '{2}'".format(
                        key, actual, expected
                    )
                )
        static_expected = self.static_features
        expected_dim = int(static_expected.get("feature_dim", 0))
        expected_names = list(static_expected.get("feature_names", []))
        scaler_names = scaler_meta.get("feature_names")
        if expected_names:
            if scaler_names is None:
                errors.append(
                    "Static feature names missing from scaler metadata; expected {0}".format(
                        expected_names
                    )
                )
            elif list(scaler_names) != expected_names:
                errors.append(
                    "Static feature names {0} differ from metadata value {1}".format(
                        list(scaler_names), expected_names
                    )
                )
        static_arr = scaler_meta.get("static_features")
        static_dim = None
        if static_arr is not None:
            array = np.asarray(static_arr)
            if array.ndim == 1:
                static_dim = 1
            elif array.ndim >= 2:
                static_dim = int(array.shape[1])
        if expected_dim and static_dim is not None and static_dim != expected_dim:
            errors.append(
                "Static feature dimension {0} differs from metadata value {1}".format(
                    static_dim, expected_dim
                )
            )
        if expected_dim and static_arr is None:
            errors.append(
                "Static feature matrix missing from scaler metadata; expected dimension {0}".format(
                    expected_dim
                )
            )
        if num_series is not None and static_arr is not None:
            array = np.asarray(static_arr)
            if array.ndim >= 2 and array.shape[0] not in {num_series, 0}:
                errors.append(
                    "Static feature row count {0} does not match number of series {1}".format(
                        array.shape[0], num_series
                    )
                )
        tf_meta = scaler_meta.get("time_features") or {}
        scaler_enabled = bool(
            tf_meta.get("enabled", tf_meta.get("config", {}).get("enabled", False))
        )
        scaler_dim = int(tf_meta.get("feature_dim", tf_meta.get("config", {}).get("feature_dim", 0)) or 0)
        scaler_freq = tf_meta.get("freq")
        if bool(self.time_features.get("enabled")) != scaler_enabled:
            errors.append(
                "Scaler metadata time feature enablement {0} differs from metadata value {1}".format(
                    scaler_enabled, self.time_features.get("enabled")
                )
            )
        meta_dim = int(self.time_features.get("feature_dim", scaler_dim))
        if scaler_dim and meta_dim and scaler_dim != meta_dim:
            errors.append(
                "Scaler time feature dimension {0} differs from metadata value {1}".format(
                    scaler_dim, meta_dim
                )
            )
        meta_freq = self.time_features.get("freq")
        if (meta_freq is not None) and (scaler_freq is not None) and str(meta_freq) != str(scaler_freq):
            errors.append(
                "Scaler time feature frequency '{0}' differs from metadata value '{1}'".format(
                    scaler_freq, meta_freq
                )
            )
        if errors:
            raise ValueError(
                "Stored artifacts incompatible with metadata artifact:\n"
                + "\n".join(f"- {msg}" for msg in errors)
            )


def save_metadata_artifact(artifact: MetadataArtifact, path: str) -> None:
    save_json(artifact.to_payload(), path)


def load_metadata_artifact(path: str) -> MetadataArtifact:
    payload = load_json(path)
    if not isinstance(payload, MutableMapping):
        raise ValueError("Metadata artifact must be a JSON object")
    meta_version = str(payload.get("meta_version", "0"))
    visited = set()
    while meta_version not in SUPPORTED_METADATA_VERSIONS:
        if meta_version in visited:
            raise ValueError(
                f"Metadata artifact migration loop detected for version '{meta_version}'"
            )
        migration = METADATA_MIGRATIONS.get(meta_version)
        if migration is None:
            supported = ", ".join(sorted(SUPPORTED_METADATA_VERSIONS))
            raise ValueError(
                f"Metadata artifact version '{meta_version}' is not supported. "
                f"Supported versions: {supported}"
            )
        payload = migration(payload)
        meta_version = str(payload.get("meta_version", "0"))
        visited.add(meta_version)
    artifact = MetadataArtifact.from_payload(payload)
    return artifact
