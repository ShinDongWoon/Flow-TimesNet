from __future__ import annotations

from typing import Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset):
    """
    Produce (X, Y) windows from wide-format [T, N] series.
    direct mode: Y length = H; recursive mode: Y length = 1
    """
    def __init__(
        self,
        wide_values: np.ndarray,  # [T, N]
        input_len: int,
        pred_len: int,
        mode: str,  # "direct"|"recursive"
    ) -> None:
        super().__init__()
        assert mode in ("direct", "recursive")
        self.X = wide_values.astype(np.float32)
        self.T, self.N = self.X.shape
        self.L = int(input_len)
        self.H = int(pred_len if mode == "direct" else 1)
        self.mode = mode
        # Indices where a full (L,H) window fits
        self.idxs = np.arange(self.T - self.L - (pred_len - 1))  # last valid start inclusive
        # When pred_len=H, last usable start is T-L-H

    def __len__(self) -> int:
        return int(len(self.idxs))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = int(self.idxs[idx])
        e = s + self.L
        x = self.X[s:e, :]  # [L, N]
        y = self.X[e : e + self.H, :]  # [H, N] or [1, N]
        return torch.from_numpy(x), torch.from_numpy(y)
