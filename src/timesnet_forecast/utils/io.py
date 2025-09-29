from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Literal, Optional, Mapping, Iterable, Any
import os
import json
import pickle
import re
import yaml
import numpy as np
import pandas as pd
import logging


logger = logging.getLogger(__name__)


SCHEMA_ARTIFACT_VERSION = "1.0"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _sample_series(series: pd.Series, sample_size: int = 128) -> pd.Series:
    if series.empty:
        return series
    if len(series) <= sample_size:
        return series
    return series.iloc[:sample_size]


def _looks_datetime(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    sample = _sample_series(series)
    if sample.dtype == object or pd.api.types.is_string_dtype(sample):
        parsed = pd.to_datetime(sample, errors="coerce", utc=False)
        non_null = parsed.notna().sum()
        return non_null >= max(1, int(0.6 * len(sample)))
    return False


def _looks_identifier(series: pd.Series) -> bool:
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_string_dtype(dtype):
        return True
    if dtype == object:
        return True
    return False


def _looks_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _find_first(df: pd.DataFrame, candidates: Iterable[str], predicate) -> Tuple[Optional[str], Optional[str]]:
    for name in candidates:
        if name in df.columns and predicate(df[name]):
            return name, "name_match"
    return None, None


def _detect_schema(df: pd.DataFrame, preferred: Mapping[str, str] | None = None) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    preferred = preferred or {}
    result: Dict[str, str] = {}
    details: Dict[str, Dict[str, Any]] = {}
    used: set[str] = set()

    def assign(role: str, column: str, reason: str, extra: Optional[Dict[str, Any]] = None) -> None:
        result[role] = column
        used.add(column)
        payload = {"reason": reason}
        if extra:
            payload.update(extra)
        details[role] = payload

    for role in ("date", "id", "target"):
        pref = preferred.get(role)
        if pref is not None and pref in df.columns:
            assign(role, pref, "override_match")

    date_candidates = [
        "date",
        "datetime",
        "timestamp",
        "ds",
        "time",
        "영업일자",
    ]
    id_candidates = [
        "id",
        "series",
        "series_id",
        "store_id",
        "store",
        "menu",
        "item",
        "영업장명_메뉴명",
        "영업장명",
    ]
    target_candidates = [
        "target",
        "value",
        "sales",
        "demand",
        "y",
        "매출수량",
        "qty",
    ]

    if "date" not in result:
        name, reason = _find_first(
            df, [c for c in date_candidates if c not in used], _looks_datetime
        )
        if name:
            assign("date", name, reason or "name_match")
    if "date" not in result:
        for column in df.columns:
            if column in used:
                continue
            if _looks_datetime(df[column]):
                assign("date", column, "datetime_dtype")
                break

    if "id" not in result:
        name, reason = _find_first(
            df, [c for c in id_candidates if c not in used], _looks_identifier
        )
        if name:
            assign("id", name, reason or "name_match")
    if "id" not in result:
        for column in df.columns:
            if column in used or column == result.get("date"):
                continue
            if _looks_identifier(df[column]):
                assign("id", column, "identifier_like")
                break

    if "target" not in result:
        name, reason = _find_first(
            df, [c for c in target_candidates if c not in used], _looks_numeric
        )
        if name:
            assign("target", name, reason or "name_match")
    if "target" not in result:
        for column in df.columns:
            if column in used:
                continue
            if column == result.get("date") or column == result.get("id"):
                continue
            if _looks_numeric(df[column]):
                assign("target", column, "numeric_like")
                break

    return result, details


def _extract_schema_overrides(data_cfg: Mapping[str, Any]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    schema_cfg = data_cfg.get("schema", {}) if isinstance(data_cfg, Mapping) else {}
    for key in ("date", "id", "target"):
        explicit = schema_cfg.get(key)
        alt = data_cfg.get(f"{key}_col") if isinstance(data_cfg, Mapping) else None
        value = explicit if not _is_missing(explicit) else alt
        if not _is_missing(value):
            overrides[key] = str(value)
    return overrides


@dataclass
class DataSchema:
    date_col: str
    id_col: str
    target_col: str
    sources: Dict[str, str] = field(default_factory=dict)
    detection: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        data_cfg: Mapping[str, Any],
        sample_df: pd.DataFrame | None = None,
        *,
        allow_auto: bool = True,
    ) -> "DataSchema":
        overrides = _extract_schema_overrides(data_cfg)
        preferred = dict(overrides)

        if sample_df is None and (allow_auto and len(preferred) < 3):
            raise ValueError(
                "DataSchema requires a sample dataframe to infer missing fields"
            )

        detected: Dict[str, str] = {}
        details: Dict[str, Dict[str, Any]] = {}
        if sample_df is not None and allow_auto:
            detected, details = _detect_schema(sample_df, preferred)

        fields: Dict[str, str] = {}
        sources: Dict[str, str] = {}
        for key in ("date", "id", "target"):
            if key in overrides:
                candidate = overrides[key]
                if sample_df is not None and candidate not in sample_df.columns:
                    raise KeyError(
                        f"Configured {key}_col '{candidate}' not present in data columns"
                    )
                fields[key] = candidate
                sources[key] = "override"
            elif key in detected:
                fields[key] = detected[key]
                sources[key] = details.get(key, {}).get("reason", "detected")
            else:
                raise ValueError(
                    f"Unable to determine column for '{key}'. Provide an override via data.{key}_col"
                )

        schema = cls(
            date_col=str(fields["date"]),
            id_col=str(fields["id"]),
            target_col=str(fields["target"]),
            sources=sources,
            detection=details,
        )
        if sample_df is not None:
            schema.require_columns(sample_df.columns)
        schema._log_resolution(overrides)
        return schema

    @classmethod
    def from_fields(
        cls,
        fields: Mapping[str, Any],
        *,
        sources: Mapping[str, str] | None = None,
        detection: Mapping[str, Dict[str, Any]] | None = None,
    ) -> "DataSchema":
        missing = [k for k in ("date", "id", "target") if k not in fields]
        if missing:
            raise ValueError(
                f"Schema artifact missing required fields: {', '.join(missing)}"
            )
        return cls(
            date_col=str(fields["date"]),
            id_col=str(fields["id"]),
            target_col=str(fields["target"]),
            sources=dict(sources or {}),
            detection=dict(detection or {}),
        )

    def __getitem__(self, key: str) -> str:
        if key == "date":
            return self.date_col
        if key == "id":
            return self.id_col
        if key == "target":
            return self.target_col
        raise KeyError(key)

    def as_dict(self) -> Dict[str, str]:
        return {"date": self.date_col, "id": self.id_col, "target": self.target_col}

    def require_columns(self, columns: Iterable[str], *, context: str | None = None) -> None:
        missing = [col for col in self.as_dict().values() if col not in columns]
        if missing:
            location = f" in {context}" if context else ""
            raise KeyError(f"Missing required columns{location}: {', '.join(missing)}")

    def validate_overrides(self, data_cfg: Mapping[str, Any]) -> None:
        overrides = _extract_schema_overrides(data_cfg)
        mismatched: Dict[str, Tuple[str, str]] = {}
        for key in ("date", "id", "target"):
            cfg_val = overrides.get(key)
            if cfg_val is None:
                continue
            resolved = self[key]
            if cfg_val != resolved:
                mismatched[key] = (cfg_val, resolved)
        if mismatched:
            msgs = [
                f"{k}: configured='{cfg}' stored='{resolved}'"
                for k, (cfg, resolved) in mismatched.items()
            ]
            raise ValueError(
                "Configured schema columns do not match stored artifact: "
                + "; ".join(msgs)
            )

    def _log_resolution(self, overrides: Mapping[str, str]) -> None:
        parts = []
        for key, col in self.as_dict().items():
            src = overrides.get(key)
            if src == col:
                origin = "override"
            else:
                origin = self.sources.get(key, "detected")
            parts.append(f"{key}='{col}' ({origin})")
        logger.info("Resolved data schema: %s", ", ".join(parts))


def resolve_schema(cfg: dict, sample_df: pd.DataFrame | None = None) -> DataSchema:
    data_cfg = cfg.get("data", {}) if isinstance(cfg, Mapping) else {}
    return DataSchema.from_config(data_cfg, sample_df=sample_df)


def normalize_id(s: str) -> str:
    # Collapse spaces -> "_" and strip; keep Korean/Unicode as-is.
    s2 = " ".join(str(s).split())  # collapse inner spaces
    s2 = s2.strip().replace(" ", "_")
    return s2


def normalize_series_name(name: str) -> str:
    """Normalize a series/menu name using :func:`normalize_id`.

    This helper mirrors :func:`normalize_id` but is provided for clarity when
    dealing specifically with series (menu) names. It collapses multiple spaces
    into a single underscore and strips leading/trailing whitespace while
    preserving any non-ASCII characters.
    """
    return normalize_id(name)


def load_yaml(path: str) -> dict:
    """Load YAML file into a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_id_col(df: pd.DataFrame, id_col: str) -> pd.Series:
    # Here id is single column already; just normalize
    return df[id_col].astype(str).map(normalize_id)


def pivot_long_to_wide(
    df: pd.DataFrame,
    date_col: str,
    id_col: str,
    target_col: str,
    fill_missing_dates: bool = True,
    fillna0: bool = True,
) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df[id_col] = build_id_col(df, id_col)
    df = df[[date_col, id_col, target_col]].sort_values([date_col, id_col])
    wide = df.pivot(index=date_col, columns=id_col, values=target_col)
    if fill_missing_dates:
        full_idx = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
        wide = wide.reindex(full_idx)
    if fillna0:
        wide = wide.fillna(0.0)
    # Ensure sorted columns (ids)
    wide = wide.sort_index(axis=1)
    wide.index.name = None
    wide.columns.name = None
    return wide.astype(float)


def fit_series_scaler(
    wide_df: pd.DataFrame,
    method: Literal["zscore", "minmax", "none"] = "zscore",
    per_series: bool = True,
    eps: float = 1e-8,
) -> Tuple[Optional[Dict[str, Tuple[float, float]]], pd.DataFrame]:
    """
    Return scaler dict and normalized DataFrame.
    - zscore: (mean, std)
    - minmax: (min, max)
    - none: identity (0,1)
    """
    ids = list(wide_df.columns)
    scaler: Dict[str, Tuple[float, float]] = {}
    if method == "none":
        return None, wide_df.copy()

    X = wide_df.values.astype(np.float32)
    if per_series:
        if method == "zscore":
            mu = np.mean(X, axis=0)
            sd = np.std(X, axis=0)
            sd = np.where(sd < eps, 1.0, sd)
            Xn = (X - mu) / sd
            for i, c in enumerate(ids):
                scaler[c] = (float(mu[i]), float(sd[i]))
        else:  # minmax
            mn = np.min(X, axis=0)
            mx = np.max(X, axis=0)
            rng = np.where((mx - mn) < eps, 1.0, (mx - mn))
            Xn = (X - mn) / rng
            for i, c in enumerate(ids):
                scaler[c] = (float(mn[i]), float(mx[i]))
        return scaler, pd.DataFrame(Xn, index=wide_df.index, columns=ids)
    else:
        if method == "zscore":
            mu = float(np.mean(X))
            sd = float(np.std(X))
            sd = sd if sd >= eps else 1.0
            Xn = (X - mu) / sd
            for c in ids:
                scaler[c] = (mu, sd)
        else:
            mn = float(np.min(X))
            mx = float(np.max(X))
            rng = (mx - mn) if (mx - mn) >= eps else 1.0
            Xn = (X - mn) / rng
            for c in ids:
                scaler[c] = (mn, mx)
        return scaler, pd.DataFrame(Xn, index=wide_df.index, columns=ids)


def inverse_transform(
    arr: np.ndarray,
    ids: List[str],
    scaler: Optional[Dict[str, Tuple[float, float]]],
    method: str,
) -> np.ndarray:
    """
    arr: [T_or_H, N]
    """
    out = np.zeros_like(arr, dtype=np.float32)
    for j, c in enumerate(ids):
        a = arr[:, j]
        if method == "zscore" and scaler is not None:
            mu, sd = scaler[c]
            out[:, j] = a * sd + mu
        elif method == "minmax" and scaler is not None:
            mn, mx = scaler[c]
            rng = (mx - mn) if (mx - mn) != 0 else 1.0
            out[:, j] = a * rng + mn
        else:
            out[:, j] = a
    return out


def save_pickle(obj: object, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> object:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_schema_artifact(
    path: str,
    schema: DataSchema,
    *,
    normalization: Mapping[str, Any] | None = None,
    extras: Mapping[str, Any] | None = None,
    version: str = SCHEMA_ARTIFACT_VERSION,
) -> None:
    payload: Dict[str, Any] = {
        "version": str(version),
        "fields": schema.as_dict(),
        "sources": dict(schema.sources),
        "detection": dict(schema.detection),
    }
    if normalization is not None:
        payload["normalization"] = dict(normalization)
    if extras is not None:
        payload["extras"] = dict(extras)
    save_json(payload, path)


def load_schema_artifact(path: str) -> Tuple[DataSchema, Dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError("Schema artifact must be a JSON object")

    if "fields" in payload:
        fields = payload["fields"]
    else:
        # Backwards compatibility: legacy format stored fields at top level.
        fields = {k: payload.get(k) for k in ("date", "id", "target")}
    schema = DataSchema.from_fields(
        fields,
        sources=payload.get("sources"),
        detection=payload.get("detection"),
    )
    meta = {
        "version": payload.get("version", "0"),
        "normalization": payload.get("normalization"),
        "extras": payload.get("extras"),
        "raw": payload,
    }
    return schema, meta


def validate_normalization_config(
    preprocess_cfg: Mapping[str, Any], normalization_meta: Mapping[str, Any] | None
) -> None:
    if normalization_meta is None:
        return
    expected_method = normalization_meta.get("method")
    expected_per_series = normalization_meta.get("per_series")
    expected_eps = normalization_meta.get("eps")

    mismatches: List[str] = []

    if expected_method is not None:
        configured = preprocess_cfg.get("normalize")
        if configured is None:
            preprocess_cfg["normalize"] = expected_method
        elif str(configured) != str(expected_method):
            mismatches.append(
                f"normalize configured='{configured}' stored='{expected_method}'"
            )
    if expected_per_series is not None:
        configured_bool = preprocess_cfg.get("normalize_per_series")
        if configured_bool is None:
            preprocess_cfg["normalize_per_series"] = bool(expected_per_series)
        elif bool(configured_bool) != bool(expected_per_series):
            mismatches.append(
                "normalize_per_series configured='{}' stored='{}'".format(
                    configured_bool, expected_per_series
                )
            )
    if expected_eps is not None:
        configured_eps = preprocess_cfg.get("eps")
        if configured_eps is None:
            preprocess_cfg["eps"] = expected_eps
        else:
            try:
                configured_eps_f = float(configured_eps)
            except (TypeError, ValueError):
                mismatches.append(
                    f"eps configured='{configured_eps}' stored='{expected_eps}'"
                )
            else:
                if not np.isclose(configured_eps_f, float(expected_eps)):
                    mismatches.append(
                        f"eps configured='{configured_eps}' stored='{expected_eps}'"
                    )

    if mismatches:
        raise ValueError(
            "Preprocess normalization settings do not match training artifacts: "
            + "; ".join(mismatches)
        )


def merge_forecasts(pred_list: List[pd.DataFrame]) -> pd.DataFrame:
    """Merge a list of forecast DataFrames.

    Parameters
    ----------
    pred_list : List[pd.DataFrame]
        Each DataFrame should contain predictions for a set of ``row_key``
        values. Menu columns may have arbitrary spacing or capitalization.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame whose column names are normalized via
        :func:`normalize_series_name` and indexed by ``row_key``.
    """

    normed: List[pd.DataFrame] = []
    for df in pred_list:
        df2 = df.copy()
        if "row_key" in df2.columns:
            df2 = df2.set_index("row_key")
        df2.columns = [normalize_series_name(c) for c in df2.columns]
        normed.append(df2)
    return pd.concat(normed, ignore_index=False)


def parse_row_key(row_key: str) -> Tuple[str, int]:
    """Parse a submission row key into its test part and day number.

    Supports patterns like:

    - ``"TEST_00+Day 1"``
    - ``"TEST_00+1일"``

    Parameters
    ----------
    row_key : str
        Row identifier of the form ``<part>+Day <n>`` or variants such as
        ``<part>+<n>일``.

    Returns
    -------
    Tuple[str, int]
        ``(part, day_number)``

    Raises
    ------
    ValueError
        If the row key does not match the supported pattern.
    """

    pattern = r"^(.*)\+(?:D(?:ay)?\s*)?(\d+)\D*$"
    m = re.match(pattern, row_key.strip(), flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Unsupported row key format: {row_key}")

    part = m.group(1).strip()
    day_num = int(m.group(2))
    return part, day_num


def format_submission(sample_df: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """
    Populate a sample submission DataFrame with predictions.

    Parameters
    ----------
    sample_df : pd.DataFrame
        The sample submission template. The first column is assumed to contain
        the row key (e.g. ``TEST_00+1일``) while the remaining columns are menu
        names.
    preds : pd.DataFrame
        DataFrame containing predictions indexed by ``row_key``. Its columns are
        expected to be normalized menu names.

    Returns
    -------
    pd.DataFrame
        Filled submission DataFrame with the same column order and names as
        ``sample_df``.
    """

    out = sample_df.copy()
    row_col = out.columns[0]
    menu_cols = [c for c in out.columns if c != row_col]
    norm_cols = [normalize_series_name(c) for c in menu_cols]

    for i, row in out.iterrows():
        rk = row[row_col]
        try:
            test_part, day_num = parse_row_key(rk)
        except ValueError:
            logging.warning(f"Invalid row key encountered: {rk}")
            out.loc[i, menu_cols] = 0.0
            continue

        lookup_key = f"{test_part}+D{day_num}"
        if lookup_key not in preds.index:
            out.loc[i, menu_cols] = 0.0
            continue

        vals = preds.loc[lookup_key, norm_cols]
        vals = vals.reindex(norm_cols).fillna(0.0).values
        out.loc[i, menu_cols] = vals

    assert list(out.columns) == list(sample_df.columns)
    return out
