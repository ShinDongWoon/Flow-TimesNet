from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np


def wsmape_grouped(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ids: List[str],
    weights: Optional[Dict[str, float]] = None,
    eps: float = 1e-8,
) -> float:
    """Weighted SMAPE grouped by store.

    Each id is "store_menu", store = split('_',1)[0].
    Only dates with A_{t,i} != 0 included for item-level mean.
    If denom |A|+|P| == 0 at a timepoint, that point is skipped.
    If an item has no valid points, its SMAPE is 0 by definition here.
    """
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have same shape"
    assert y_true.ndim == 2, "expected [T_or_H, N]"
    N = y_true.shape[1]
    stores = [s.split("_", 1)[0] for s in ids]
    store_to_idx: Dict[str, List[int]] = {}
    for j, st in enumerate(stores):
        store_to_idx.setdefault(st, []).append(j)

    if weights is None:
        weights = {st: 1.0 for st in store_to_idx}
    Z = sum(weights.values()) if weights else 1.0
    wnorm = {st: (weights.get(st, 0.0) / Z) for st in store_to_idx}

    def smape_item(a: np.ndarray, p: np.ndarray) -> float:
        mask = (np.abs(a) > eps)  # valid actual
        a, p = a[mask], p[mask]
        if a.size == 0:
            return 0.0
        denom = np.abs(a) + np.abs(p)
        mask2 = denom > eps
        if not np.any(mask2):
            return 0.0
        return float(np.mean(2.0 * np.abs(a[mask2] - p[mask2]) / denom[mask2]))

    score = 0.0
    for st, idxs in store_to_idx.items():
        if len(idxs) == 0:
            continue
        item_scores = [smape_item(y_true[:, j], y_pred[:, j]) for j in idxs]
        score += wnorm.get(st, 0.0) * (float(np.mean(item_scores)) if item_scores else 0.0)
    return float(score)


def smape_mean(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean symmetric MAPE across all series.

    Uses the same masking strategy as :func:`_smape_np` in the unit tests:
    only points where the actual value magnitude exceeds ``eps`` contribute
    to the final mean.
    """
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have same shape"
    mask = np.abs(y_true) > eps
    if not np.any(mask):
        return 0.0
    denom = np.abs(y_true) + np.abs(y_pred)
    smape = 2.0 * np.abs(y_pred - y_true)[mask] / denom[mask]
    return float(np.mean(smape))
