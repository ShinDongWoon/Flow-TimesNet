from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset):
    """
    Produce (X, Y) windows from wide-format [T, N] series.
    direct mode: Y length = ``pred_len``
    recursive mode: Y length = 1 by default, but can be overridden.
    """
    def __init__(
        self,
        wide_values: np.ndarray,  # [T, N]
        input_len: int,
        pred_len: int,
        mode: str,  # "direct"|"recursive"
        recursive_pred_len: int | None = None,
        augment: Dict | None = None,
        pmax_global: int | None = None,
        min_history_for_training: int | None = None,
    ) -> None:
        super().__init__()
        assert mode in ("direct", "recursive")
        self.X = wide_values.astype(np.float32)
        self.T, self.N = self.X.shape
        self.L = int(input_len)
        if pmax_global is None:
            raise ValueError("pmax_global must be provided")
        self.P = int(pmax_global)
        if self.P <= 0:
            raise ValueError("pmax_global must be positive")
        if mode == "direct":
            self.H = int(pred_len)
        else:
            # In recursive mode training windows normally have one-step targets.
            # ``recursive_pred_len`` can override this to produce multi-step
            # targets (e.g. for validation).
            self.H = int(recursive_pred_len if recursive_pred_len is not None else 1)
        self.mode = mode
        augment = augment or {}
        self.add_noise_std = float(augment.get("add_noise_std", 0.0))
        self.time_shift = int(augment.get("time_shift", 0))
        raw_min_history = (
            self.L if min_history_for_training is None else int(min_history_for_training)
        )
        if raw_min_history < 0:
            raise ValueError("min_history_for_training must be non-negative")
        self.min_history = int(min(self.P, raw_min_history))
        # Cache prefix sums of non-zero observations to quickly count history length
        # for any [start, end) interval. We pad with a leading zero row so that
        # ``cumsum[e] - cumsum[s]`` yields the number of non-zero points in
        # ``[s, e)``.
        nz = (self.X != 0).astype(np.int64)
        self._nz_cumsum = np.zeros((self.T + 1, self.N), dtype=np.int64)
        if self.T > 0:
            self._nz_cumsum[1:] = np.cumsum(nz, axis=0)
        self.max_start = max(0, self.T - self.L - self.H)
        valid_pairs: List[Tuple[int, int]] = []
        max_history_observed = 0
        if self.T - self.L - self.H >= 0:
            for s in range(self.max_start + 1):
                e = s + self.L
                start_hist = max(0, e - self.P)
                hist = self._nz_cumsum[e, :] - self._nz_cumsum[start_hist, :]
                if hist.size:
                    hist_max = int(hist.max())
                    if hist_max > max_history_observed:
                        max_history_observed = hist_max
                valid_js = np.flatnonzero(hist >= self.min_history)
                for j in valid_js.tolist():
                    valid_pairs.append((int(j), int(s)))
            if not valid_pairs and max_history_observed > 0 and self.min_history > max_history_observed:
                self.min_history = int(max_history_observed)
                for s in range(self.max_start + 1):
                    e = s + self.L
                    start_hist = max(0, e - self.P)
                    hist = self._nz_cumsum[e, :] - self._nz_cumsum[start_hist, :]
                    valid_js = np.flatnonzero(hist >= self.min_history)
                    for j in valid_js.tolist():
                        valid_pairs.append((int(j), int(s)))
            if not valid_pairs:
                for s in range(self.max_start + 1):
                    for j in range(self.N):
                        valid_pairs.append((int(j), int(s)))
        self.valid_indices = (
            np.asarray(valid_pairs, dtype=np.int64)
            if valid_pairs
            else np.zeros((0, 2), dtype=np.int64)
        )
        # In recursive mode ``self.H`` may be 1 (training) or >1 (validation).

    def __len__(self) -> int:
        return int(self.valid_indices.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if idx < 0 or idx >= len(self):
            raise IndexError("SlidingWindowDataset index out of range")
        series_idx, start_idx = self.valid_indices[idx]
        s = int(start_idx)
        if self.time_shift > 0 and self.max_start > 0:
            delta = np.random.randint(-self.time_shift, self.time_shift + 1)
            cand = int(np.clip(s + delta, 0, self.max_start))
            e_cand = cand + self.L
            start_hist_cand = max(0, e_cand - self.P)
            hist_cand = (
                self._nz_cumsum[e_cand, series_idx]
                - self._nz_cumsum[start_hist_cand, series_idx]
            )
            if hist_cand >= self.min_history:
                s = cand
        e = s + self.L
        start_hist = int(max(0, e - self.P))
        x = self.X[start_hist:e, series_idx : series_idx + 1]  # [<=P, 1]
        y = self.X[e : e + self.H, series_idx : series_idx + 1]  # [H, 1]
        if self.add_noise_std > 0:
            x = x + np.random.normal(scale=self.add_noise_std, size=x.shape)
        if x.shape[0] < self.P:
            pad_len = self.P - x.shape[0]
            pad = np.zeros((pad_len, 1), dtype=np.float32)
            x = np.concatenate([pad, x.astype(np.float32, copy=False)], axis=0)
        x = x.astype(np.float32, copy=False)
        y = y.astype(np.float32, copy=False)
        return torch.from_numpy(x), torch.from_numpy(y), int(series_idx)
