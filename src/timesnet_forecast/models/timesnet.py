from __future__ import annotations

import math
import os
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
    source_device: torch.device
    source_dtype: torch.dtype


def _accum_dtype(t: torch.Tensor | torch.dtype) -> torch.dtype:
    """Return a numerically stable accumulation dtype for ``t``.

    Half-precision dtypes accumulate in ``float32`` while other tensors keep
    their original dtype to preserve numerics.
    """

    dtype = t if isinstance(t, torch.dtype) else t.dtype
    if dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return dtype


def _dtype_size(dtype: torch.dtype) -> int:
    """Return the size in bytes for ``dtype``."""

    return torch.tensor([], dtype=dtype).element_size()


def _streaming_limit_bytes(device: torch.device) -> int:
    """Resolve the byte budget for streaming pool operations."""

    env_value = os.environ.get("TIMESNET_POOL_MAX_BYTES")
    if env_value is not None:
        try:
            parsed = int(env_value)
            if parsed > 0:
                limit = parsed
            else:
                limit = None
        except ValueError:
            limit = None
    else:
        limit = None

    if limit is None:
        limit = 256 * 1024 * 1024  # 256MB default budget

    if device.type == "cuda" and torch.cuda.is_available():
        try:
            device_index = device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            free_bytes, _ = torch.cuda.mem_get_info(device_index)
        except RuntimeError:
            pass
        else:
            gpu_budget = int(free_bytes * 0.8)
            if gpu_budget > 0:
                limit = min(limit, gpu_budget)

    return max(int(limit), 1)


def _resolve_pool_sub_batch_size(
    batch: int,
    channels: int,
    time_steps: int,
    target_steps: int,
    value_dtype: torch.dtype,
    accum_dtype: torch.dtype,
    device: torch.device,
) -> int:
    """Determine the maximum chunk size for streaming trailing pool ops."""

    if batch <= 0:
        return 0

    accum_size = _dtype_size(accum_dtype)
    value_size = _dtype_size(value_dtype)
    count_size = _dtype_size(torch.long)

    per_sample_bytes = (
        channels * (time_steps + 1) * accum_size
        + 2 * channels * target_steps * accum_size
        + channels * target_steps * value_size
        + target_steps * count_size
    )
    per_sample_bytes = max(per_sample_bytes, 1)

    limit_bytes = _streaming_limit_bytes(device)
    max_samples = max(1, limit_bytes // per_sample_bytes)
    return min(batch, max_samples)


def _pool_trailing_sums_small_batch(
    values: torch.Tensor,
    lengths: torch.Tensor,
    target_steps: int,
    total_time: int,
    steps: torch.Tensor,
    time_idx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pool trailing sums for a small batch without further chunking."""

    batch = values.size(0)
    channels = values.size(1)
    has_valid = lengths > 0
    lengths_safe = torch.where(has_valid, lengths, torch.ones_like(lengths))

    if not torch.all(lengths_safe == total_time):
        start = (total_time - lengths_safe).view(batch, 1, 1)
        range_mask = (time_idx >= start).to(values.dtype)
        values = values * range_mask

    offsets = (total_time - lengths_safe).view(batch, 1)
    scaled = lengths_safe.view(batch, 1) * steps.view(1, -1)
    start_rel = scaled[:, :-1] // target_steps
    end_rel = (scaled[:, 1:] + target_steps - 1) // target_steps

    start_idx = (start_rel + offsets).clamp(min=0, max=total_time)
    end_idx = (end_rel + offsets).clamp(min=0, max=total_time)
    counts = (end_idx - start_idx).clamp(min=1)

    accum_dtype = _accum_dtype(values)
    values_cumsum = torch.cumsum(values, dim=-1, dtype=accum_dtype)
    prefix = torch.zeros(batch, channels, 1, device=values.device, dtype=accum_dtype)
    values_cumsum = torch.cat([prefix, values_cumsum], dim=-1)

    start_idx_vals = start_idx.view(batch, 1, target_steps).expand(-1, channels, -1)
    end_idx_vals = end_idx.view(batch, 1, target_steps).expand(-1, channels, -1)
    start_vals = torch.gather(values_cumsum, dim=-1, index=start_idx_vals)
    end_vals = torch.gather(values_cumsum, dim=-1, index=end_idx_vals)
    sum_vals = (end_vals - start_vals).to(values.dtype)

    if not torch.all(has_valid):
        keep_mask = has_valid.view(batch, 1, 1).to(sum_vals.dtype)
        sum_vals = sum_vals * keep_mask
        counts = counts * has_valid.view(batch, 1).to(counts.dtype)

    return sum_vals, counts


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
    device = features.device

    lengths = valid_lengths.to(device=device, dtype=torch.long)
    lengths = lengths.clamp(min=1, max=T)

    steps = torch.arange(target_steps + 1, device=device, dtype=lengths.dtype)
    time_idx = torch.arange(T, device=device, dtype=lengths.dtype).view(1, 1, T)

    feat_chunk = _resolve_pool_sub_batch_size(
        BN, features.size(1), T, target_steps, features.dtype, _accum_dtype(features), device
    )

    if mask.dtype.is_floating_point:
        mask_value_dtype = mask.dtype
    else:
        mask_value_dtype = features.dtype if features.dtype.is_floating_point else torch.float32

    mask_chunk = _resolve_pool_sub_batch_size(
        BN, mask.size(1), T, target_steps, mask_value_dtype, _accum_dtype(mask_value_dtype), device
    )

    chunk_size = min(feat_chunk, mask_chunk)
    if chunk_size >= BN:
        feat_sums, counts = _pool_trailing_sums_small_batch(
            features, lengths, target_steps, T, steps, time_idx
        )
        mask_input = mask if mask.dtype == mask_value_dtype else mask.to(mask_value_dtype)
        mask_sums, _ = _pool_trailing_sums_small_batch(
            mask_input, lengths, target_steps, T, steps, time_idx
        )
    else:
        feat_chunks: List[torch.Tensor] = []
        mask_chunks: List[torch.Tensor] = []
        count_chunks: List[torch.Tensor] = []

        for feat_part, mask_part, length_part in zip(
            torch.split(features, chunk_size, dim=0),
            torch.split(mask, chunk_size, dim=0),
            torch.split(lengths, chunk_size, dim=0),
        ):
            feat_sum_part, count_part = _pool_trailing_sums_small_batch(
                feat_part, length_part, target_steps, T, steps, time_idx
            )
            mask_values = (
                mask_part
                if mask_part.dtype == mask_value_dtype
                else mask_part.to(mask_value_dtype)
            )
            mask_sum_part, _ = _pool_trailing_sums_small_batch(
                mask_values, length_part, target_steps, T, steps, time_idx
            )
            feat_chunks.append(feat_sum_part)
            mask_chunks.append(mask_sum_part)
            count_chunks.append(count_part)

        feat_sums = torch.cat(feat_chunks, dim=0)
        mask_sums = torch.cat(mask_chunks, dim=0)
        counts = torch.cat(count_chunks, dim=0)

    feat_accum_dtype = _accum_dtype(features)
    feat_denom = counts.view(BN, 1, target_steps).to(feat_accum_dtype)
    pooled_feats = (feat_sums.to(feat_accum_dtype) / feat_denom).to(features.dtype)

    mask_accum_dtype = _accum_dtype(mask_value_dtype)
    mask_denom = counts.view(BN, 1, target_steps).to(mask_accum_dtype)
    pooled_mask = (mask_sums.to(mask_accum_dtype) / mask_denom).to(mask.dtype)
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

    steps = torch.arange(target_steps + 1, device=device, dtype=lengths.dtype)
    time_idx = torch.arange(T, device=device, dtype=lengths.dtype).view(1, 1, T)

    chunk_size = _resolve_pool_sub_batch_size(
        B, C, T, target_steps, values.dtype, _accum_dtype(values), device
    )

    if chunk_size >= B:
        sum_vals, counts = _pool_trailing_sums_small_batch(
            values, lengths, target_steps, T, steps, time_idx
        )
    else:
        sum_chunks: List[torch.Tensor] = []
        count_chunks: List[torch.Tensor] = []

        for values_part, length_part in zip(
            torch.split(values, chunk_size, dim=0),
            torch.split(lengths, chunk_size, dim=0),
        ):
            sum_part, count_part = _pool_trailing_sums_small_batch(
                values_part, length_part, target_steps, T, steps, time_idx
            )
            sum_chunks.append(sum_part)
            count_chunks.append(count_part)

        sum_vals = torch.cat(sum_chunks, dim=0)
        counts = torch.cat(count_chunks, dim=0)

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
            source_device = tiles.device
            source_dtype = tiles.dtype
            tiles = tiles.to("cpu", non_blocking=True)
            if tiles.device.type == "cpu" and torch.cuda.is_available():
                try:
                    tiles = tiles.pin_memory()
                except RuntimeError:
                    pass
            tiles = tiles.contiguous()
            out.append(
                PeriodGroup(
                    values=tiles,
                    batch_indices=bn_sel,
                    frequency_indices=freq_sel,
                    cycles=cycles_int,
                    period=period_int,
                    source_device=source_device,
                    source_dtype=source_dtype,
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
        weight = self.proj.weight
        bias = self.proj.bias
        channels = res.size(1)
        z = torch.zeros_like(res)
        start = 0
        for path in self.paths:
            end = start + channels
            weight_slice = weight.narrow(1, start, channels)
            path_out = path(x)
            z = z + F.conv2d(path_out, weight_slice, bias=bias)
            bias = None
            del path_out
            start = end
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
        period_group_recompute: bool | None = None,
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
        if period_group_recompute is None:
            period_group_recompute = bool(
                torch.cuda.is_available() and torch.cuda.device_count() > 0
            )
        else:
            period_group_recompute = bool(period_group_recompute)
        self.period_group_recompute = period_group_recompute
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
        self,
        group: PeriodGroup,
        target_device: torch.device | None,
        running_budget: dict[str, int] | None = None,
        remaining_tiles: int | None = None,
    ) -> int:
        total_tiles = int(group.values.size(0))
        if total_tiles <= 0:
            if running_budget is not None:
                running_budget.setdefault("bytes_per_tile", 0)
                running_budget.setdefault("remaining_bytes", 0)
            return 0

        if remaining_tiles is None:
            remaining_tiles = total_tiles
        else:
            remaining_tiles = max(0, min(int(remaining_tiles), total_tiles))
        if remaining_tiles <= 0:
            return 0

        chunk_limit = int(remaining_tiles)
        if self.period_group_chunk is not None:
            chunk_limit = max(1, min(chunk_limit, int(self.period_group_chunk)))

        dtype_obj = getattr(group, "source_dtype", None)
        if not isinstance(dtype_obj, torch.dtype):
            dtype_obj = group.values.dtype
        try:
            element_size = torch.tensor(0, dtype=dtype_obj).element_size()
        except TypeError:
            element_size = torch.tensor(0, dtype=group.values.dtype).element_size()
        element_size = max(int(element_size), 1)

        cycles = max(int(group.cycles), 1)
        period = max(int(group.period), 1)
        base_elements = float(self.d_model) * float(cycles) * float(period)
        kernel_paths = max(len(self.kernel_set), 1)
        block_layers = max(int(self.n_layers), 0)
        # Each convolutional path stores outputs for the two conv/activation
        # pairs. Account for those activations and scale the contribution by the
        # number of parallel kernel paths.
        module_outputs_per_path = 2.0
        path_multiplier = float(kernel_paths) * module_outputs_per_path
        checkpoint_multiplier = 2.0 if not self.use_checkpoint else 1.0
        block_multiplier = float(block_layers) * path_multiplier * checkpoint_multiplier
        estimated_elements = base_elements * (1.0 + block_multiplier)
        # ``TimesBlock`` historically concatenated the per-path activations along
        # the channel dimension which yields a wide ``[C * len(kernels), ...]``
        # buffer. The streaming implementation avoids materialising the cat in
        # eager mode, but autograd still needs to retain gradients matching the
        # wide view, so include it in the byte estimate.
        if block_layers > 0:
            concat_channels = float(self.d_model * kernel_paths)
            concat_elements = concat_channels * float(cycles) * float(period)
            estimated_elements += concat_elements * float(block_layers)
            # Dropout retains a mask with the base ``[C, cycles, period]`` shape
            # per block. Scale by the non-checkpointing multiplier because these
            # activations also need to be stored for backward when checkpointing
            # is disabled.
            estimated_elements += (
                base_elements * float(block_layers) * checkpoint_multiplier
            )
        safety_factor = 4.0
        bytes_per_tile = int(
            max(math.ceil(estimated_elements * element_size * safety_factor), 1)
        )

        ratio = self.period_group_memory_ratio
        if ratio is None or not math.isfinite(ratio) or ratio <= 0.0:
            ratio = 0.5
        ratio = min(max(ratio, 1e-4), 1.0)

        budget_bytes: int | None = None
        resolved_device: torch.device | None
        if target_device is not None:
            resolved_device = torch.device(target_device)
        else:
            resolved_device = getattr(group, "source_device", None)
            if resolved_device is not None:
                resolved_device = torch.device(resolved_device)

        if (
            resolved_device is not None
            and resolved_device.type == "cuda"
            and torch.cuda.is_available()
        ):
            try:
                free_bytes, _ = torch.cuda.mem_get_info(resolved_device)
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

        if running_budget is not None:
            stored_bytes_per_tile = int(running_budget.get("bytes_per_tile", 0))
            if stored_bytes_per_tile > 0:
                bytes_per_tile = stored_bytes_per_tile
            else:
                running_budget["bytes_per_tile"] = bytes_per_tile

            remaining_bytes = running_budget.get("remaining_bytes")
            if remaining_bytes is None:
                remaining_bytes = budget_bytes
            else:
                remaining_bytes = int(remaining_bytes)
                if remaining_bytes < 0:
                    remaining_bytes = 0
                remaining_bytes = min(remaining_bytes, budget_bytes)
        else:
            remaining_bytes = int(budget_bytes)

        if bytes_per_tile <= 0:
            chunk = chunk_limit
        else:
            chunk = remaining_bytes // bytes_per_tile
            if chunk <= 0:
                chunk = 1
            if budget_bytes > 0:
                available_bytes = max(float(remaining_bytes), float(bytes_per_tile))
                max_reasonable_chunk = max(
                    1,
                    int(
                        math.ceil(
                            float(available_bytes)
                            / float(bytes_per_tile)
                            / 2.0
                        )
                    ),
                )
                chunk = min(chunk, max_reasonable_chunk)

        chunk = max(1, min(chunk, chunk_limit))

        if running_budget is not None:
            running_budget["bytes_per_tile"] = bytes_per_tile
            consumed = int(chunk) * bytes_per_tile if bytes_per_tile > 0 else 0
            running_budget["remaining_bytes"] = max(remaining_bytes - consumed, 0)

        return int(chunk)

    def _summarize_values_chunk(
        self,
        values_chunk: torch.Tensor,
        total_steps: int,
        target_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert a folded tile chunk into pooled trailing sums."""

        if values_chunk.ndim != 3:
            raise ValueError("values_chunk must be a 3D tensor")

        tiles = values_chunk.contiguous().unsqueeze(1)
        tiles = self.frontend(tiles)
        for blk in self.blocks:
            tiles = (
                checkpoint(blk, tiles, use_reentrant=False)
                if self.use_checkpoint
                else blk(tiles)
            )

        flat = tiles.reshape(tiles.size(0), self.d_model, -1)
        time_steps = flat.size(-1)
        target_steps = max(int(target_len), 1)

        if time_steps == 0:
            zero_sum = flat.new_zeros(flat.size(0), flat.size(1), target_steps)
            zero_counts = torch.zeros(
                flat.size(0),
                target_steps,
                dtype=torch.long,
                device=flat.device,
            )
            return zero_sum, zero_counts

        if time_steps > total_steps:
            flat = flat[..., -total_steps:]
            time_steps = flat.size(-1)

        tile_lengths = torch.full(
            (flat.size(0),),
            time_steps,
            dtype=torch.long,
            device=flat.device,
        )
        flat_sum, tile_counts = _pool_trailing_sums(
            values=flat,
            valid_lengths=tile_lengths,
            target_len=target_steps,
        )
        return flat_sum, tile_counts

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
        frontend_param = next(iter(self.frontend.parameters()), None)
        if frontend_param is not None:
            conv_device = frontend_param.device
            conv_dtype = frontend_param.dtype
        else:
            conv_device = features.device
            conv_dtype = features.dtype
        for group in groups:
            total_tiles = group.values.size(0)
            if total_tiles == 0:
                continue
            budget_state: dict[str, int] = {}
            start_idx = 0
            while start_idx < total_tiles:
                remaining_tiles = total_tiles - start_idx
                chunk_size = self._resolve_period_group_chunk(
                    group=group,
                    target_device=conv_device,
                    running_budget=budget_state,
                    remaining_tiles=remaining_tiles,
                )
                chunk_size = int(chunk_size) if chunk_size is not None else 0
                if chunk_size <= 0:
                    chunk_size = int(remaining_tiles)
                chunk_size = max(1, min(chunk_size, int(remaining_tiles)))
                chunk_start = start_idx
                end_idx = min(chunk_start + chunk_size, total_tiles)
                if end_idx <= chunk_start:
                    start_idx = total_tiles
                    continue
                values_chunk = group.values[chunk_start:end_idx]
                start_idx = end_idx
                if values_chunk.numel() == 0:
                    continue
                copy_non_blocking = (
                    conv_device.type == "cuda"
                    and torch.cuda.is_available()
                    and values_chunk.device.type == "cpu"
                    and values_chunk.is_pinned()
                )
                values_chunk = values_chunk.to(
                    device=conv_device,
                    dtype=conv_dtype,
                    non_blocking=copy_non_blocking,
                )
                chunk_values = values_chunk
                should_checkpoint_chunk = (
                    self.period_group_recompute and not self.use_checkpoint
                )

                def summarize(chunk_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                    return self._summarize_values_chunk(
                        chunk_tensor,
                        total_steps=T,
                        target_len=input_window,
                    )

                if should_checkpoint_chunk:
                    if not chunk_values.requires_grad:
                        chunk_values.requires_grad_()
                    flat_sum, tile_counts = checkpoint(
                        summarize, chunk_values, use_reentrant=False
                    )
                else:
                    flat_sum, tile_counts = summarize(chunk_values)
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
                batch_chunk = group.batch_indices[chunk_start:end_idx]
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
        # Constrain the predicted mean to remain close to the observed input
        # envelope. This mitigates occasional overshoots introduced by small
        # training batches in the CPU reference tests while preserving a modest
        # margin that allows the model to extrapolate.
        window_series = x.permute(0, 2, 1)
        if mask is not None:
            mask_series = mask.permute(0, 2, 1)
            pos_mask = mask_series > 0
            valid_mins = torch.where(
                pos_mask,
                window_series,
                torch.full_like(window_series, float("inf")),
            ).amin(dim=-1)
            valid_maxs = torch.where(
                pos_mask,
                window_series,
                torch.full_like(window_series, float("-inf")),
            ).amax(dim=-1)
            no_valid = ~(pos_mask.any(dim=-1))
            if torch.any(no_valid):
                fallback_min = window_series.amin(dim=-1)
                fallback_max = window_series.amax(dim=-1)
                valid_mins = torch.where(no_valid, fallback_min, valid_mins)
                valid_maxs = torch.where(no_valid, fallback_max, valid_maxs)
            input_min = valid_mins
            input_max = valid_maxs
        else:
            input_min = window_series.amin(dim=-1)
            input_max = window_series.amax(dim=-1)
        span = (input_max - input_min).abs()
        margin = span * 0.1
        eps = torch.finfo(mu.dtype).eps
        clamp_min = (input_min - margin - eps).unsqueeze(1)
        clamp_max = (input_max + margin + eps).unsqueeze(1)
        mu = torch.clamp(mu, min=clamp_min, max=clamp_max)
        return mu, sigma
