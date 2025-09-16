from __future__ import annotations

from typing import Tuple, Dict
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
        # Indices where a full (L,H) window fits. ``self.H`` is the mode-specific
        # output length, so the last valid start index is ``T - L - H``.
        self.idxs = np.arange(self.T - self.L - self.H + 1)
        # In recursive mode ``self.H`` may be 1 (training) or >1 (validation).

    def __len__(self) -> int:
        return int(len(self.idxs))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = int(self.idxs[idx])
        if self.time_shift > 0:
            delta = np.random.randint(-self.time_shift, self.time_shift + 1)
            s = int(np.clip(s + delta, 0, self.T - self.L - self.H))
        e = s + self.L
        start = int(max(0, e - self.P))
        x = self.X[start:e, :]  # [<=P, N]
        y = self.X[e : e + self.H, :]  # [H, N] or [1, N]
        if self.add_noise_std > 0:
            x = x + np.random.normal(scale=self.add_noise_std, size=x.shape)
        if x.shape[0] < self.P:
            pad_len = self.P - x.shape[0]
            pad = np.zeros((pad_len, self.N), dtype=np.float32)
            x = np.concatenate([pad, x.astype(np.float32, copy=False)], axis=0)
        x = x.astype(np.float32, copy=False)
        y = y.astype(np.float32, copy=False)
        return torch.from_numpy(x), torch.from_numpy(y)
