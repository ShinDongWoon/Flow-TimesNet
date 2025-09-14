from __future__ import annotations

from typing import Dict, List, Tuple, Literal
import os
import json
import pickle
import yaml
import numpy as np
import pandas as pd


def resolve_schema(cfg: dict) -> Dict[str, str]:
    return {
        "date": cfg["data"]["date_col"],
        "target": cfg["data"]["target_col"],
        "id": cfg["data"]["id_col"],
    }


def normalize_id(s: str) -> str:
    # Collapse spaces -> "_" and strip; keep Korean/Unicode as-is.
    s2 = " ".join(str(s).split())  # collapse inner spaces
    s2 = s2.strip().replace(" ", "_")
    return s2


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
) -> Tuple[Dict[str, Tuple[float, float]], pd.DataFrame]:
    """
    Return scaler dict and normalized DataFrame.
    - zscore: (mean, std)
    - minmax: (min, max)
    - none: identity (0,1)
    """
    ids = list(wide_df.columns)
    scaler: Dict[str, Tuple[float, float]] = {}
    if method == "none":
        for c in ids:
            scaler[c] = (0.0, 1.0)
        return scaler, wide_df.copy()

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
    arr: np.ndarray, ids: List[str], scaler: Dict[str, Tuple[float, float]], method: str
) -> np.ndarray:
    """
    arr: [T_or_H, N]
    """
    out = np.zeros_like(arr, dtype=np.float32)
    for j, c in enumerate(ids):
        a = arr[:, j]
        if method == "zscore":
            mu, sd = scaler[c]
            out[:, j] = a * sd + mu
        elif method == "minmax":
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


def parse_row_key(row_key: str) -> Tuple[str, int]:
    """
    "TEST_00+Day 1" -> ("TEST_00", 1)
    """
    part, day = row_key.split("+Day")
    part = part.strip()
    day_num = int(day.strip())
    return part, day_num


def format_submission(
    sample_df: pd.DataFrame,
    preds_by_date,
    date_col: str | None = None,
) -> pd.DataFrame:
    out = sample_df.copy()
    if date_col:
        menu_cols = [c for c in out.columns if c != date_col]
        out[date_col] = pd.to_datetime(out[date_col])
        for i, row in out.iterrows():
            current_date = row[date_col]
            if current_date in preds_by_date.index:
                vals = preds_by_date.loc[current_date]
                out.loc[i, menu_cols] = vals.reindex(menu_cols).fillna(0.0).values
            else:
                out.loc[i, menu_cols] = 0.0
        return out

    # Fallback to legacy row_key based lookup
    preds_by_test = preds_by_date
    menu_cols = [c for c in out.columns if c != "row_key"]
    for i, row in out.iterrows():
        rk = row["row_key"]
        test_part, day_num = parse_row_key(rk)
        P = preds_by_test.get(test_part)
        if P is None:
            out.loc[i, menu_cols] = 0.0
            continue
        drow = f"D{day_num}"
        if drow not in P.index:
            out.loc[i, menu_cols] = 0.0
            continue
        vals = P.loc[drow]
        out.loc[i, menu_cols] = vals.reindex(menu_cols).fillna(0.0).values
    return out
