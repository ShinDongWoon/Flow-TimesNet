from __future__ import annotations

from typing import List, Tuple
import pandas as pd


def make_holdout_slices(wide_df: pd.DataFrame, holdout_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    assert holdout_days > 0
    val = wide_df.iloc[-holdout_days:].copy()
    trn = wide_df.iloc[:-holdout_days].copy()
    return trn, val


def make_rolling_slices(
    wide_df: pd.DataFrame, folds: int, step_days: int, val_len: int
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Build rolling folds from the tail. Each fold shifts back by step_days.
    """
    out: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    end = len(wide_df)
    for k in range(folds):
        val_end = end - k * step_days
        val_start = max(0, val_end - val_len)
        trn = wide_df.iloc[:val_start].copy()
        val = wide_df.iloc[val_start:val_end].copy()
        if len(val) == 0 or len(trn) == 0:
            break
        out.append((trn, val))
    return out
