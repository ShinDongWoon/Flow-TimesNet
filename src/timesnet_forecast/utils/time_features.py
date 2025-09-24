from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Union

import numpy as np
import pandas as pd

EncodingType = Union[str, Mapping[str, str]]

DEFAULT_FEATURES: List[str] = [
    "day_of_week",
    "day_of_month",
    "month",
    "day_of_year",
]


def _extract_day_of_week(index: pd.DatetimeIndex) -> Tuple[np.ndarray, int]:
    values = index.dayofweek.to_numpy()
    return values.astype(np.int64, copy=False), 7


def _extract_day_of_month(index: pd.DatetimeIndex) -> Tuple[np.ndarray, int]:
    values = index.day.to_numpy() - 1
    return values.astype(np.int64, copy=False), 31


def _extract_month(index: pd.DatetimeIndex) -> Tuple[np.ndarray, int]:
    values = index.month.to_numpy() - 1
    return values.astype(np.int64, copy=False), 12


def _extract_hour(index: pd.DatetimeIndex) -> Tuple[np.ndarray, int]:
    values = index.hour.to_numpy()
    return values.astype(np.int64, copy=False), 24


def _extract_minute(index: pd.DatetimeIndex) -> Tuple[np.ndarray, int]:
    values = index.minute.to_numpy()
    return values.astype(np.int64, copy=False), 60


def _extract_day_of_year(index: pd.DatetimeIndex) -> Tuple[np.ndarray, int]:
    values = index.dayofyear.to_numpy() - 1
    return values.astype(np.int64, copy=False), 366


def _extract_week_of_year(index: pd.DatetimeIndex) -> Tuple[np.ndarray, int]:
    iso = index.isocalendar()
    week = getattr(iso, "week", None)
    if week is None:
        week = iso["week"]
    values = week.to_numpy() - 1
    return values.astype(np.int64, copy=False), 53


FEATURE_EXTRACTORS = {
    "day_of_week": _extract_day_of_week,
    "day_of_month": _extract_day_of_month,
    "month": _extract_month,
    "hour": _extract_hour,
    "minute": _extract_minute,
    "day_of_year": _extract_day_of_year,
    "week_of_year": _extract_week_of_year,
}


def _resolve_encoding(feature: str, encoding: EncodingType) -> str:
    if isinstance(encoding, Mapping):
        enc_val = encoding.get(feature, encoding.get("default", "cyclical"))
    else:
        enc_val = encoding
    enc_str = str(enc_val).lower()
    if enc_str not in {"cyclical", "onehot", "numeric"}:
        raise ValueError(
            f"Unsupported encoding '{enc_val}' for feature '{feature}'. Expected 'cyclical', 'onehot', or 'numeric'."
        )
    return enc_str


def _encode_component(
    values: np.ndarray, period: int, encoding: str, normalize: bool
) -> np.ndarray:
    if values.ndim != 1:
        values = values.reshape(-1)
    if period <= 0:
        period = int(values.max(initial=0) - values.min(initial=0) + 1)
        period = max(period, 1)
    mod_values = np.mod(values, period)
    if encoding == "cyclical":
        if period == 0:
            period = 1
        angles = 2.0 * np.pi * (mod_values.astype(np.float32) / float(period))
        sin_term = np.sin(angles)
        cos_term = np.cos(angles)
        return np.stack([sin_term, cos_term], axis=1).astype(np.float32, copy=False)
    if encoding == "onehot":
        onehot = np.zeros((values.size, int(period)), dtype=np.float32)
        idx = mod_values.astype(np.int64, copy=False)
        if idx.size > 0:
            onehot[np.arange(idx.size), idx] = 1.0
        return onehot
    numeric = mod_values.astype(np.float32, copy=False)
    if normalize and period > 1:
        denom = float(period - 1)
        if denom <= 0:
            denom = 1.0
        numeric = numeric / denom
    return numeric.reshape(-1, 1)


def _as_datetime_index(index: Union[pd.DatetimeIndex, Sequence]) -> pd.DatetimeIndex:
    if isinstance(index, pd.DatetimeIndex):
        return index
    return pd.to_datetime(np.asarray(index))


def build_time_features(
    index: Union[pd.DatetimeIndex, Sequence],
    config: Mapping[str, object] | None,
    *,
    return_names: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
    """Construct a feature matrix of temporal covariates.

    Args:
        index: Datetime index describing each timestamp.
        config: Dictionary containing keys ``enabled``, ``features``,
            ``encoding``, and ``normalize``. Unknown keys are ignored.
        return_names: When ``True`` the feature names are returned alongside the
            matrix for debugging or logging purposes.

    Returns:
        ``np.ndarray`` with shape ``[len(index), feature_dim]`` in ``float32``
        precision. When ``return_names`` is ``True`` a tuple containing the
        matrix and a list of column names is returned.
    """

    cfg = dict(config or {})
    enabled = bool(cfg.get("enabled", False))
    idx = _as_datetime_index(index)
    if not enabled:
        empty = np.zeros((len(idx), 0), dtype=np.float32)
        if return_names:
            return empty, []
        return empty

    features: Iterable[str] = cfg.get("features") or DEFAULT_FEATURES
    encoding_cfg: EncodingType = cfg.get("encoding", "cyclical")
    normalize = bool(cfg.get("normalize", True))

    matrices: List[np.ndarray] = []
    names: List[str] = []
    for feature in features:
        extractor = FEATURE_EXTRACTORS.get(feature)
        if extractor is None:
            raise ValueError(f"Unsupported time feature '{feature}'.")
        values, period = extractor(idx)
        encoding = _resolve_encoding(feature, encoding_cfg)
        encoded = _encode_component(values, period, encoding, normalize)
        if encoded.size == 0:
            continue
        matrices.append(encoded.astype(np.float32, copy=False))
        if encoding == "cyclical":
            names.extend([f"{feature}_sin", f"{feature}_cos"])
        elif encoding == "onehot":
            names.extend([f"{feature}_{i}" for i in range(encoded.shape[1])])
        else:
            names.append(feature)

    if not matrices:
        empty = np.zeros((len(idx), 0), dtype=np.float32)
        if return_names:
            return empty, []
        return empty

    matrix = np.hstack(matrices).astype(np.float32, copy=False)
    if return_names:
        return matrix, names
    return matrix


__all__ = ["build_time_features"]
