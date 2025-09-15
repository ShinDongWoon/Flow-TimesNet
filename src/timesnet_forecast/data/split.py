from __future__ import annotations

from typing import Iterator, Tuple
import pandas as pd


def make_holdout_slices(wide_df: pd.DataFrame, holdout_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    assert holdout_days > 0
    val = wide_df.iloc[-holdout_days:].copy()
    trn = wide_df.iloc[:-holdout_days].copy()
    return trn, val


def make_rolling_slices(
    wide_df: pd.DataFrame, folds: int, step_days: int, val_len: int
) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield rolling train/validation slices from the tail.

    Each fold shifts the validation window back by ``step_days``. Slices are
    returned as views into ``wide_df`` without copying so that downstream code
    can process one fold at a time without holding multiple copies of the data
    in memory.
    """

    end = len(wide_df)
    for k in range(folds):
        val_end = end - k * step_days
        val_start = max(0, val_end - val_len)
        trn = wide_df.iloc[:val_start]
        val = wide_df.iloc[val_start:val_end]
        if len(val) == 0 or len(trn) == 0:
            break
        yield trn, val
