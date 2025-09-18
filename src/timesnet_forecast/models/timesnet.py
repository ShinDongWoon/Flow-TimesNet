from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
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


def _adaptive_pool_valid_lengths(
    features: torch.Tensor,
    mask: torch.Tensor,
    valid_lengths: torch.Tensor,
    target_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pool the trailing valid prefix for a batch of sequences.

    This vectorized variant reproduces the behaviour of slicing each
    ``features[idx]`` / ``mask[idx]`` pair to the final ``valid_lengths[idx]``
    timesteps and then applying :func:`torch.nn.functional.adaptive_avg_pool1d`.
    """

    if features.ndim != 3 or mask.ndim != 3:
        raise ValueError("features and mask must be 3D tensors")

    if features.shape[0] != mask.shape[0] or features.shape[-1] != mask.shape[-1]:
        raise ValueError("features and mask must share batch and time dimensions")

    if features.device != mask.device:
        raise ValueError("features and mask must reside on the same device")

    BN, _, T = features.shape
    if T == 0:
        raise ValueError("features must have a non-empty time dimension")

    target_steps = max(int(target_len), 1)

    lengths = valid_lengths.clamp(min=1, max=T)
    if not torch.all(lengths == T):
        time_idx = torch.arange(T, device=features.device, dtype=valid_lengths.dtype)
        time_idx = time_idx.view(1, 1, T)
        start = (T - lengths).view(BN, 1, 1)
        range_mask = (time_idx >= start).to(features.dtype)
        features = features * range_mask
        mask = mask * range_mask

    offsets = (T - lengths).view(BN, 1)
    steps = torch.arange(target_steps + 1, device=features.device, dtype=valid_lengths.dtype)
    scaled = lengths.view(BN, 1) * steps.view(1, -1)
    start_rel = scaled[:, :-1] // target_steps
    end_rel = (scaled[:, 1:] + target_steps - 1) // target_steps

    start_idx = (start_rel + offsets).clamp(min=0, max=T)
    end_idx = (end_rel + offsets).clamp(min=0, max=T)
    counts = (end_idx - start_idx).clamp(min=1)

    feat_cumsum = torch.cumsum(features.to(torch.float64), dim=-1)
    feat_cumsum = torch.cat(
        [feat_cumsum.new_zeros(BN, features.size(1), 1), feat_cumsum], dim=-1
    )
    mask_cumsum = torch.cumsum(mask.to(torch.float64), dim=-1)
    mask_cumsum = torch.cat(
        [mask_cumsum.new_zeros(BN, mask.size(1), 1), mask_cumsum], dim=-1
    )

    start_idx_feat = start_idx.view(BN, 1, target_steps).expand(-1, features.size(1), -1)
    end_idx_feat = end_idx.view(BN, 1, target_steps).expand(-1, features.size(1), -1)
    feat_start = torch.gather(feat_cumsum, dim=-1, index=start_idx_feat)
    feat_end = torch.gather(feat_cumsum, dim=-1, index=end_idx_feat)
    feat_sum = feat_end - feat_start

    start_idx_mask = start_idx.view(BN, 1, target_steps).expand(-1, mask.size(1), -1)
    end_idx_mask = end_idx.view(BN, 1, target_steps).expand(-1, mask.size(1), -1)
    mask_start = torch.gather(mask_cumsum, dim=-1, index=start_idx_mask)
    mask_end = torch.gather(mask_cumsum, dim=-1, index=end_idx_mask)
    mask_sum = mask_end - mask_start

    denom = counts.view(BN, 1, target_steps).to(torch.float64)
    pooled_feats = (feat_sum / denom).to(features.dtype)
    pooled_mask = (mask_sum / denom).to(mask.dtype)
    return pooled_feats, pooled_mask


def _pool_trailing_sums(
    values: torch.Tensor,
    valid_lengths: torch.Tensor,
    target_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aggregate trailing prefixes into ``target_len`` bins via summation.

    Args:
        values: Tensor shaped ``[B, C, T]``.
        valid_lengths: Tensor shaped ``[B]`` describing the number of valid
            trailing timesteps per sequence. When a length is zero the
            corresponding output sums and counts are also zeroed.
        target_len: Number of output bins.

    Returns:
        Tuple ``(sum_values, counts)`` where ``sum_values`` has shape
        ``[B, C, target_len]`` and contains the summed values for each output
        bin, and ``counts`` has shape ``[B, target_len]`` storing the number of
        contributing timesteps per bin.
    """

    if values.ndim != 3:
        raise ValueError("values must be a 3D tensor")

    if valid_lengths.ndim != 1 or valid_lengths.size(0) != values.size(0):
        raise ValueError("valid_lengths must be 1D with the same batch size")

    B, C, T = values.shape
    if T <= 0:
        raise ValueError("values must have a non-empty time dimension")

    target_steps = max(int(target_len), 1)
    device = values.device
    lengths = valid_lengths.to(device=device, dtype=torch.long)
    lengths = lengths.clamp(min=0, max=T)
    has_valid = lengths > 0
    lengths_safe = torch.where(has_valid, lengths, torch.ones_like(lengths))

    if not torch.all(lengths_safe == T):
        time_idx = torch.arange(T, device=device, dtype=lengths_safe.dtype)
        time_idx = time_idx.view(1, 1, T)
        start = (T - lengths_safe).view(B, 1, 1)
        range_mask = (time_idx >= start).to(values.dtype)
        values = values * range_mask

    offsets = (T - lengths_safe).view(B, 1)
    steps = torch.arange(target_steps + 1, device=device, dtype=lengths_safe.dtype)
    scaled = lengths_safe.view(B, 1) * steps.view(1, -1)
    start_rel = scaled[:, :-1] // target_steps
    end_rel = (scaled[:, 1:] + target_steps - 1) // target_steps

    start_idx = (start_rel + offsets).clamp(min=0, max=T)
    end_idx = (end_rel + offsets).clamp(min=0, max=T)
    counts = (end_idx - start_idx).clamp(min=1)

    values_cumsum = torch.cumsum(values.to(torch.float64), dim=-1)
    values_cumsum = torch.cat(
        [values_cumsum.new_zeros(B, C, 1), values_cumsum], dim=-1
    )
    start_idx_vals = start_idx.view(B, 1, target_steps).expand(-1, C, -1)
    end_idx_vals = end_idx.view(B, 1, target_steps).expand(-1, C, -1)
    start_vals = torch.gather(values_cumsum, dim=-1, index=start_idx_vals)
    end_vals = torch.gather(values_cumsum, dim=-1, index=end_idx_vals)
    sum_vals = (end_vals - start_vals).to(values.dtype)

    if not torch.all(has_valid):
        keep_mask = has_valid.view(B, 1, 1).to(sum_vals.dtype)
        sum_vals = sum_vals * keep_mask
        counts = counts * has_valid.view(B, 1).expand_as(counts).to(counts.dtype)

    return sum_vals, counts


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

        valid_mask = (cycles > 0) & (periods > 0)
        if not torch.any(valid_mask):
            return []

        cycle_vals = torch.masked_select(cycles, valid_mask)
        period_vals = torch.masked_select(periods, valid_mask)

        batch_grid = (
            torch.arange(BN, device=x.device, dtype=torch.long)
            .view(BN, 1)
            .expand(BN, K)
        )
        freq_grid = (
            torch.arange(K, device=x.device, dtype=torch.long)
            .view(1, K)
            .expand(BN, K)
        )
        batch_vals = torch.masked_select(batch_grid, valid_mask)
        freq_vals = torch.masked_select(freq_grid, valid_mask)

        pair_keys = torch.stack([cycle_vals, period_vals], dim=1)
        unique_pairs, inverse = torch.unique(pair_keys, dim=0, return_inverse=True)

        out: List[PeriodGroup] = []
        max_time_idx = max(T - 1, 0)
        for group_idx in range(unique_pairs.size(0)):
            selector = inverse == group_idx
            if not torch.any(selector):
                continue
            select_idx = torch.nonzero(selector, as_tuple=False).squeeze(-1)

            bn_sel = torch.index_select(batch_vals, 0, select_idx)
            freq_sel = torch.index_select(freq_vals, 0, select_idx)
            if bn_sel.numel() == 0:
                continue

            cycles_int = int(unique_pairs[group_idx, 0].item())
            period_int = int(unique_pairs[group_idx, 1].item())
            take = cycles_int * period_int
            if take <= 0:
                continue

            start = max(T - take, 0)
            gather_idx = torch.arange(take, device=x.device, dtype=torch.long)
            if gather_idx.numel() == 0:
                continue
            gather_idx = (gather_idx + start).clamp_(min=0, max=max_time_idx)

            seq_subset = seqs.index_select(0, bn_sel)
            tiles = seq_subset.index_select(1, gather_idx)
            tiles = tiles.reshape(-1, cycles_int, period_int).contiguous()

            out.append(
                PeriodGroup(
                    values=tiles,
                    batch_indices=bn_sel,
                    frequency_indices=freq_sel,
                    cycles=cycles_int,
                    period=period_int,
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
        period_group_chunk: int | None = None,
        period_group_memory_ratio: float | None = None,
        period_group_max_chunk_bytes: int | None = None,
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
        # Maximum number of tiles processed from a period group at once. ``None``
        # preserves the historical behaviour of processing the full group.
        if period_group_chunk is not None:
            try:
                chunk_override = int(period_group_chunk)
            except (TypeError, ValueError, OverflowError):
                chunk_override = None
            else:
                if chunk_override <= 0:
                    chunk_override = None
        else:
            chunk_override = None
        self.period_group_chunk: Optional[int] = chunk_override
        if period_group_memory_ratio is not None:
            try:
                ratio_override = float(period_group_memory_ratio)
            except (TypeError, ValueError):
                ratio_override = None
            else:
                if not math.isfinite(ratio_override):
                    ratio_override = None
        else:
            ratio_override = None
        if period_group_max_chunk_bytes is not None:
            try:
                max_bytes_override = int(period_group_max_chunk_bytes)
            except (TypeError, ValueError, OverflowError):
                max_bytes_override = None
            else:
                if max_bytes_override <= 0:
                    max_bytes_override = None
        else:
            max_bytes_override = None
        self.period_group_memory_ratio: Optional[float] = ratio_override
        self.period_group_max_chunk_bytes: Optional[int] = max_bytes_override

    def _resolve_period_group_chunk(
        self, group: PeriodGroup, dtype: torch.dtype | None
    ) -> int:
        total_tiles = int(group.values.size(0))
        if total_tiles <= 0:
            return 0

        if self.period_group_chunk is not None:
            chunk = max(1, min(total_tiles, int(self.period_group_chunk)))
            return chunk

        dtype_obj = dtype or group.values.dtype
        try:
            element_size = torch.tensor(0, dtype=dtype_obj).element_size()
        except TypeError:
            element_size = torch.tensor(0, dtype=group.values.dtype).element_size()
        element_size = max(int(element_size), 1)

        cycles = max(int(group.cycles), 1)
        period = max(int(group.period), 1)
        base_elements = float(self.d_model) * float(cycles) * float(period)
        kernel_paths = max(len(self.kernel_set), 1)
        block_multiplier = max(self.n_layers, 1) * kernel_paths
        estimated_elements = base_elements * (1.0 + float(block_multiplier))
        safety_factor = 4.0
        bytes_per_tile = int(
            max(math.ceil(estimated_elements * element_size * safety_factor), 1)
        )

        ratio = self.period_group_memory_ratio
        if ratio is None or not math.isfinite(ratio) or ratio <= 0.0:
            ratio = 0.5
        ratio = min(max(ratio, 1e-4), 1.0)

        budget_bytes: int | None = None
        if group.values.is_cuda and torch.cuda.is_available():
            try:
                free_bytes, _ = torch.cuda.mem_get_info(group.values.device)
            except Exception:
                free_bytes = None
            if free_bytes is not None:
                budget_bytes = int(max(free_bytes * ratio, 0))

        if budget_bytes is None or budget_bytes <= 0:
            default_cpu_budget = 256 * 1024 * 1024  # 256 MiB
            budget_bytes = int(default_cpu_budget * ratio)

        max_bytes = self.period_group_max_chunk_bytes
        if max_bytes is not None:
            budget_bytes = max(1, min(budget_bytes, max_bytes))

        if bytes_per_tile <= 0:
            return max(total_tiles, 1)

        chunk = budget_bytes // bytes_per_tile
        if chunk <= 0:
            chunk = 1
        return max(1, min(total_tiles, chunk))

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

        input_window = int(max(1, min(T, self.input_len)))
        features = x.new_zeros(BN, self.d_model, input_window)
        coverage = x.new_zeros(BN, 1, input_window)
        for group in groups:
            total_tiles = group.values.size(0)
            if total_tiles == 0:
                continue
            chunk_dtype = (
                group.values.dtype
                if isinstance(group.values, torch.Tensor)
                else x.dtype
            )
            chunk_size = self._resolve_period_group_chunk(
                group=group, dtype=chunk_dtype
            )
            if chunk_size <= 0:
                chunk_size = int(total_tiles)
            chunk_size = max(1, min(int(chunk_size), int(total_tiles)))
            for start_idx in range(0, total_tiles, chunk_size):
                end_idx = min(start_idx + chunk_size, total_tiles)
                values_chunk = group.values[start_idx:end_idx]
                if values_chunk.numel() == 0:
                    continue
                tiles = values_chunk.unsqueeze(1)  # [M, 1, C, P]
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
                tile_len = flat.size(-1)
                tile_lengths = torch.full(
                    (flat.size(0),),
                    tile_len,
                    dtype=torch.long,
                    device=flat.device,
                )
                flat_sum, tile_counts = _pool_trailing_sums(
                    values=flat,
                    valid_lengths=tile_lengths,
                    target_len=input_window,
                )
                coverage_update = tile_counts.to(dtype=flat_sum.dtype).unsqueeze(1)
                if features.dtype != flat_sum.dtype:
                    # Mixed precision autocast can yield lower precision tiles; align the
                    # accumulation buffers with the tile dtype to avoid redundant casts.
                    features = features.to(dtype=flat_sum.dtype)
                if coverage.dtype != flat_sum.dtype:
                    coverage = coverage.to(dtype=flat_sum.dtype)
                if flat_sum.device != features.device:
                    raise RuntimeError(
                        "flat_sum and features must reside on the same device"
                    )
                batch_chunk = group.batch_indices[start_idx:end_idx]
                features.index_add_(0, batch_chunk, flat_sum)
                if coverage_update.device != coverage.device:
                    raise RuntimeError(
                        "coverage_update and coverage must reside on the same device"
                    )
                coverage.index_add_(0, batch_chunk, coverage_update)

        coverage_mask = coverage > 0
        coverage_safe = torch.where(coverage_mask, coverage, torch.ones_like(coverage))
        features = features / coverage_safe
        step_mask = coverage_mask.to(features.dtype)
        hist_mask_flat = None
        mask_lengths = None
        if mask is not None:
            hist_mask_flat = mask.permute(0, 2, 1).reshape(BN, T)
            mask_lengths = hist_mask_flat.to(torch.long).sum(dim=-1)
            mask_vals = hist_mask_flat.unsqueeze(1).to(features.dtype)
            mask_sums, _ = _pool_trailing_sums(
                values=mask_vals,
                valid_lengths=mask_lengths,
                target_len=input_window,
            )
            step_mask = torch.maximum(
                step_mask,
                (mask_sums > 0).to(features.dtype),
            )

        raw_input = x.permute(0, 2, 1).reshape(BN, T).unsqueeze(1)
        if hist_mask_flat is not None:
            raw_input = raw_input * hist_mask_flat.unsqueeze(1)
            raw_lengths = mask_lengths
        else:
            raw_lengths = torch.full((BN,), T, dtype=torch.long, device=x.device)
        raw_sums, raw_counts = _pool_trailing_sums(
            values=raw_input,
            valid_lengths=raw_lengths,
            target_len=input_window,
        )
        raw_counts = raw_counts.to(raw_sums.dtype).unsqueeze(1)
        raw_counts_safe = torch.where(
            raw_counts > 0,
            raw_counts,
            raw_counts.new_ones(raw_counts.shape),
        )
        raw_avg = raw_sums / raw_counts_safe
        raw_avg = torch.where(raw_counts > 0, raw_avg, torch.zeros_like(raw_avg))
        if raw_avg.dtype != features.dtype:
            raw_avg = raw_avg.to(dtype=features.dtype)
        raw_features = self.raw_proj(raw_avg)
        if raw_features.dtype != features.dtype:
            raw_features = raw_features.to(dtype=features.dtype)
        features = features + raw_features
        features = features * step_mask

        if hist_mask_flat is not None:
            valid_lengths = torch.minimum(
                mask_lengths,
                torch.full_like(mask_lengths, input_window),
            )
        else:
            valid_lengths = torch.full(
                (BN,), input_window, dtype=torch.long, device=x.device
            )

        needs_pooling = input_window != self.input_len
        if not needs_pooling:
            needs_pooling = bool(torch.any(valid_lengths < input_window).item())

        if needs_pooling:
            features, pooled_mask = _adaptive_pool_valid_lengths(
                features=features,
                mask=step_mask,
                valid_lengths=valid_lengths,
                target_len=self.input_len,
            )
        else:
            pooled_mask = step_mask

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
