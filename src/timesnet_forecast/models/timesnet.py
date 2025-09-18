from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class PeriodGroup:
    """Container describing a batch of folded tiles sharing the same shape."""

    values: torch.Tensor
    batch_indices: torch.Tensor
    frequency_indices: torch.Tensor
    cycles: int
    period: int


class PeriodicityTransform(nn.Module):
    """Approximate period folding via FFT."""

    def __init__(
        self, k_periods: int, pmax: int, min_period_threshold: int = 1
    ) -> None:
        super().__init__()
        self.k = int(k_periods)
        self.pmax = int(max(1, pmax))
        min_thresh = int(max(1, min_period_threshold))
        self.min_period_threshold = int(min(self.pmax, min_thresh))

    @staticmethod
    def _topk_freq(x: torch.Tensor, k: int) -> torch.Tensor:
        """Return indices of top-k nonzero frequencies.

        Args:
            x: [..., T] 실수 시퀀스
            k: 선택할 빈도 수

        Returns:
            torch.Tensor: [..., k] 형태의 주파수 인덱스
        """
        T = x.size(-1)
        spec = torch.fft.rfft(x, dim=-1)
        mag2 = spec.real**2 + spec.imag**2
        mag2[..., 0] = 0.0  # DC 성분 제외
        k = min(k, mag2.size(-1) - 1) if mag2.size(-1) > 1 else 0
        if k <= 0:
            shape = list(x.shape[:-1]) + [1]
            return torch.ones(shape, dtype=torch.long, device=x.device)
        topk = torch.topk(mag2, k=k, dim=-1).indices
        topk = torch.clamp(topk, min=1)
        return topk

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> List[PeriodGroup]:
        """Approximate period folding without dense padding.

        Args:
            x: Tensor shaped ``[B, T, N]`` containing the history.
            mask: Optional tensor with the same shape as ``x`` marking valid
                timesteps with values greater than zero. When provided, leading
                padded entries are ignored when estimating dominant frequencies.

        Returns:
            List of :class:`PeriodGroup` instances. Each group contains a batch of
            folded tiles sharing the same ``(cycles, period)`` shape alongside the
            indices of the originating series and their frequency ranks.
        """

        B, T, N = x.shape
        BN = B * N
        seqs = x.permute(0, 2, 1).reshape(BN, T)

        if self.k <= 0:
            return []

        if mask is not None:
            if mask.shape != x.shape:
                raise ValueError("mask must match input shape")
            mask_flat = mask.permute(0, 2, 1).reshape(BN, T)
            mask_bool = mask_flat > 0
            valid_lengths = mask_bool.to(torch.long).sum(dim=-1)
        else:
            valid_lengths = torch.full(
                (BN,), T, dtype=torch.long, device=x.device
            )

        if mask is None:
            kidx = self._topk_freq(seqs, self.k)
        else:
            cols = max(self.k, 1)
            kidx = torch.ones((BN, cols), dtype=torch.long, device=x.device)
            has_data = valid_lengths > 0
            if has_data.any():
                uniq = torch.unique(valid_lengths[has_data])
                for length_val in uniq.tolist():
                    length_int = int(length_val)
                    if length_int <= 0:
                        continue
                    idxs = torch.nonzero(valid_lengths == length_int, as_tuple=False).squeeze(-1)
                    if idxs.numel() == 0:
                        continue
                    seq_subset = seqs.index_select(0, idxs)
                    seq_trim = seq_subset[:, T - length_int :]
                    subset_kidx = self._topk_freq(seq_trim, self.k)
                    if subset_kidx.size(1) < cols:
                        pad = torch.ones(
                            (subset_kidx.size(0), cols - subset_kidx.size(1)),
                            dtype=torch.long,
                            device=x.device,
                        )
                        subset_kidx = torch.cat([subset_kidx, pad], dim=1)
                    kidx.index_copy_(0, idxs, subset_kidx[:, :cols])

        K = kidx.size(1)
        if K == 0 or kidx.numel() == 0:
            return []

        effective_lengths = torch.clamp(valid_lengths, min=1)
        lengths_exp = effective_lengths.view(BN, 1)
        kidx_clamped = torch.clamp(kidx, min=1)
        periods = lengths_exp // kidx_clamped
        periods = torch.clamp(periods, max=self.pmax)
        min_period = torch.full_like(periods, self.min_period_threshold)
        min_period = torch.minimum(min_period, lengths_exp)
        periods = torch.maximum(periods, min_period)
        periods = torch.clamp(periods, min=1)

        has_observations = (valid_lengths > 0).view(BN, 1)
        cycles = torch.where(
            has_observations,
            torch.clamp(lengths_exp // torch.clamp(periods, min=1), min=1),
            torch.zeros_like(periods),
        )

        groups: dict[tuple[int, int], dict[str, List[torch.Tensor] | List[int]]] = {}
        for bn in range(BN):
            seq = seqs[bn]
            for freq_rank in range(K):
                cycle_val = int(cycles[bn, freq_rank].item())
                period_val = int(periods[bn, freq_rank].item())
                if cycle_val <= 0 or period_val <= 0:
                    continue
                take = cycle_val * period_val
                if take <= 0:
                    continue
                start = max(T - take, 0)
                idx_range = torch.arange(take, device=x.device)
                gather_idx = (start + idx_range).clamp(min=0, max=max(T - 1, 0))
                tile = seq.index_select(0, gather_idx).view(cycle_val, period_val).contiguous()
                key = (cycle_val, period_val)
                payload = groups.setdefault(
                    key,
                    {"tiles": [], "batch": [], "freq": []},
                )
                payload["tiles"].append(tile)
                payload["batch"].append(bn)
                payload["freq"].append(freq_rank)

        out: List[PeriodGroup] = []
        for (cycle_val, period_val), payload in groups.items():
            tiles = torch.stack(payload["tiles"], dim=0)
            batch_idx = torch.as_tensor(
                payload["batch"], dtype=torch.long, device=x.device
            )
            freq_idx = torch.as_tensor(
                payload["freq"], dtype=torch.long, device=x.device
            )
            out.append(
                PeriodGroup(
                    values=tiles,
                    batch_indices=batch_idx,
                    frequency_indices=freq_idx,
                    cycles=cycle_val,
                    period=period_val,
                )
            )

        return out


class TimesBlock(nn.Module):
    """Multi-scale 2-D convolutional block inspired by TimesNet."""

    def __init__(
        self,
        channels: int,
        kernel_set: Sequence[int],
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        kernels = list(kernel_set) if len(kernel_set) > 0 else [1]
        self.paths = nn.ModuleList()
        for k in kernels:
            k_int = max(int(k), 1)
            pad = k_int // 2
            self.paths.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=(k_int, 1),
                        padding=(pad, 0),
                    ),
                    self._build_activation(activation),
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=(1, k_int),
                        padding=(0, pad),
                    ),
                    self._build_activation(activation),
                )
            )
        self.proj = nn.Conv2d(channels * len(self.paths), channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.out_act = self._build_activation(activation)

    @staticmethod
    def _build_activation(name: str) -> nn.Module:
        name_norm = name.lower()
        if name_norm == "relu":
            return nn.ReLU()
        return nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        feats = [path(x) for path in self.paths]
        z = torch.cat(feats, dim=1)
        z = self.proj(z)
        z = self.out_act(z)
        z = self.dropout(z)
        return z + res


class TimesNet(nn.Module):
    """TimesNet forecaster with grouped period folding and 2-D convolutions."""
    def __init__(
        self,
        input_len: int,
        pred_len: int,
        d_model: int,
        n_layers: int,
        k_periods: int,
        pmax: int,
        kernel_set: List[int],
        dropout: float,
        activation: str,
        mode: str,
        min_period_threshold: int = 1,
        channels_last: bool = False,
        use_checkpoint: bool = True,
        min_sigma: float = 1e-3,
        min_sigma_vector: torch.Tensor | Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        assert mode in ("direct", "recursive")
        self.mode = mode
        self.pred_len = int(pred_len)
        self._out_steps = self.pred_len if self.mode == "direct" else 1
        self.period = PeriodicityTransform(
            k_periods=k_periods,
            pmax=pmax,
            min_period_threshold=min_period_threshold,
        )
        self.k = int(k_periods)
        self.input_len = int(input_len)
        self.act = activation
        # Blocks are instantiated lazily once we observe the input device/dtype.
        self.frontend: nn.Module = nn.Identity()
        self.raw_proj: nn.Module = nn.Identity()
        self.blocks: nn.ModuleList = nn.ModuleList()
        self.pool: nn.Module = nn.Identity()
        self.head: nn.Module = nn.Identity()
        self._lazy_built = False
        self.kernel_set = kernel_set
        self.dropout = float(dropout)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.channels_last = bool(channels_last)
        self.use_checkpoint = bool(use_checkpoint)
        self.min_sigma = float(min_sigma)
        self.register_buffer("min_sigma_vector", None)
        if min_sigma_vector is not None:
            min_sigma_tensor = torch.as_tensor(min_sigma_vector, dtype=torch.float32)
            self.min_sigma_vector = min_sigma_tensor.reshape(1, 1, -1)

    # ------------------------------------------------------------------
    # Backwards compatibility hooks
    # ------------------------------------------------------------------
    def _resize_frontend(self, *args, **kwargs) -> None:
        """Kept for compatibility with legacy patching in tests."""

        return None

    def _build_lazy(self, x: torch.Tensor) -> None:
        """Instantiate convolutional blocks on first use."""

        device = x.device
        dtype = x.dtype
        self.frontend = nn.Conv2d(1, self.d_model, kernel_size=1).to(
            device=device, dtype=dtype
        )
        self.raw_proj = nn.Conv1d(1, self.d_model, kernel_size=1).to(
            device=device, dtype=dtype
        )
        blocks: List[nn.Module] = []
        for _ in range(self.n_layers):
            blocks.append(
                TimesBlock(
                    channels=self.d_model,
                    kernel_set=self.kernel_set,
                    dropout=self.dropout,
                    activation=self.act,
                ).to(device=device, dtype=dtype)
            )
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool1d(self._out_steps)
        self.head = nn.Conv1d(self.d_model, 2, kernel_size=1).to(
            device=device, dtype=dtype
        )
        self._lazy_built = True

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T, N]
        returns:
          Tuple(mu, sigma) where each tensor has shape:
            direct:    [B, pred_len, N]
            recursive: [B, 1, N]
        """
        B, T, N = x.shape
        sigma_floor = None
        if isinstance(self.min_sigma_vector, torch.Tensor) and self.min_sigma_vector.numel() > 0:
            if self.min_sigma_vector.shape[-1] != N:
                raise ValueError(
                    "min_sigma_vector length does not match number of series"
                )
            sigma_floor = self.min_sigma_vector
        groups = self.period(x, mask=mask)
        BN = B * N
        if len(groups) == 0:
            out_steps = self._out_steps
            mu = x.new_zeros(B, out_steps, N)
            if sigma_floor is not None:
                sigma = (
                    sigma_floor.to(dtype=x.dtype, device=x.device)
                    .expand(B, out_steps, -1)
                    .clone()
                )
            else:
                sigma = x.new_full((B, out_steps, N), self.min_sigma)
            return mu, sigma

        if not self._lazy_built:
            self._build_lazy(x=x)

        features = x.new_zeros(BN, self.d_model, T)
        coverage = x.new_zeros(BN, 1, T)
        for group in groups:
            if group.values.numel() == 0:
                continue
            tiles = group.values.unsqueeze(1)  # [M, 1, C, P]
            tiles = self.frontend(tiles)
            for blk in self.blocks:
                tiles = (
                    checkpoint(blk, tiles, use_reentrant=False)
                    if self.use_checkpoint
                    else blk(tiles)
                )
            flat = tiles.reshape(tiles.size(0), self.d_model, -1)
            if flat.size(-1) == 0:
                continue
            if flat.size(-1) > T:
                flat = flat[..., -T:]
            length = flat.size(-1)
            start = max(T - length, 0)
            pad_left = start
            pad_right = T - start - length
            flat_padded = F.pad(flat, (pad_left, pad_right))
            features.index_add_(0, group.batch_indices, flat_padded)
            coverage_update = flat.new_ones((flat.size(0), 1, length))
            coverage_update = F.pad(coverage_update, (pad_left, pad_right))
            coverage.index_add_(0, group.batch_indices, coverage_update)

        coverage_mask = coverage > 0
        coverage_safe = torch.where(coverage_mask, coverage, torch.ones_like(coverage))
        features = features / coverage_safe
        step_mask = coverage_mask.to(features.dtype)
        hist_mask_flat = None
        if mask is not None:
            hist_mask_flat = mask.permute(0, 2, 1).reshape(BN, T)
            step_mask = torch.maximum(
                step_mask,
                (hist_mask_flat > 0).unsqueeze(1).to(features.dtype),
            )

        raw_input = x.permute(0, 2, 1).reshape(BN, T).unsqueeze(1)
        if hist_mask_flat is not None:
            raw_input = raw_input * hist_mask_flat.unsqueeze(1)
        raw_features = self.raw_proj(raw_input)
        features = features + raw_features
        features = features * step_mask

        if hist_mask_flat is not None:
            valid_lengths = hist_mask_flat.to(torch.long).sum(dim=-1)
        else:
            valid_lengths = torch.full((BN,), T, dtype=torch.long, device=x.device)

        pooled_feats: List[torch.Tensor] = []
        pooled_masks: List[torch.Tensor] = []
        for idx in range(BN):
            length = int(valid_lengths[idx].item())
            length = max(min(length, features.size(-1)), 1)
            start_idx = features.size(-1) - length
            feat_slice = features[idx : idx + 1, :, start_idx:]
            mask_slice = step_mask[idx : idx + 1, :, start_idx:]
            pooled_feats.append(F.adaptive_avg_pool1d(feat_slice, self.input_len))
            pooled_masks.append(F.adaptive_avg_pool1d(mask_slice, self.input_len))
        features = torch.cat(pooled_feats, dim=0)
        pooled_mask = torch.cat(pooled_masks, dim=0)

        eps = torch.finfo(features.dtype).eps
        features = features / (pooled_mask + eps)
        mask_seq = (pooled_mask > 0).to(features.dtype)
        features = features * mask_seq

        z = self.pool(features)
        y_all = self.head(z)
        out_steps = self._out_steps
        params = (
            y_all.reshape(B, N, 2, out_steps).permute(0, 1, 3, 2).contiguous()
        )
        mu = params[..., 0].permute(0, 2, 1).contiguous()
        log_sigma = params[..., 1]
        sigma = F.softplus(log_sigma).permute(0, 2, 1).contiguous()
        if sigma_floor is not None:
            sigma = sigma + sigma_floor.to(dtype=sigma.dtype, device=sigma.device)
        else:
            sigma = sigma + self.min_sigma
        return mu, sigma
