from __future__ import annotations

from typing import Dict, Sequence
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
        valid_mask: np.ndarray | None = None,  # [T, N]
        series_static: np.ndarray | None = None,
        series_ids: Sequence[int] | np.ndarray | None = None,
    ) -> None:
        super().__init__()
        assert mode in ("direct", "recursive")
        self.X = wide_values.astype(np.float32)
        if valid_mask is not None and valid_mask.shape != self.X.shape:
            raise ValueError(
                "valid_mask must match wide_values shape"
            )
        if valid_mask is None:
            self.M = np.ones_like(self.X, dtype=np.float32)
        else:
            self.M = valid_mask.astype(np.float32)
        self.T, self.N = self.X.shape
        self.L = int(input_len)
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
        self._X_tensor = torch.from_numpy(self.X)
        self._M_tensor = torch.from_numpy(self.M)

        if series_static is not None:
            static_arr = np.asarray(series_static, dtype=np.float32)
            if static_arr.ndim == 1:
                static_arr = static_arr.reshape(-1, 1)
            if static_arr.shape[0] != self.N:
                raise ValueError(
                    "series_static must have shape [num_series, num_features]"
                )
            self.series_static = torch.from_numpy(static_arr)
        else:
            self.series_static = None

        if series_ids is not None:
            ids_arr = np.asarray(series_ids)
            if ids_arr.ndim != 1:
                raise ValueError("series_ids must be a 1D sequence")
            if ids_arr.shape[0] != self.N:
                raise ValueError("series_ids length must match number of series")
            self.series_ids = torch.from_numpy(ids_arr.astype(np.int64, copy=False))
        else:
            self.series_ids = None

    def __len__(self) -> int:
        return int(len(self.idxs))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        s = int(self.idxs[idx])
        if self.time_shift > 0:
            delta = np.random.randint(-self.time_shift, self.time_shift + 1)
            s = int(np.clip(s + delta, 0, self.T - self.L - self.H))
        e = s + self.L
        x_tensor = self._X_tensor[s:e, :].clone()
        if self.add_noise_std > 0:
            noise = torch.randn_like(x_tensor) * self.add_noise_std
            x_tensor = x_tensor + noise
        y_tensor = self._X_tensor[e : e + self.H, :].clone()
        mask_tensor = self._M_tensor[e : e + self.H, :].clone()
        items: list[torch.Tensor] = [x_tensor, y_tensor, mask_tensor]
        if self.series_static is not None:
            items.append(self.series_static)
        if self.series_ids is not None:
            items.append(self.series_ids)
        return tuple(items)
