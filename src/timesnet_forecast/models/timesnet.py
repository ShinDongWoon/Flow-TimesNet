from __future__ import annotations

import math
import os
from contextlib import nullcontext
from typing import Sequence, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _module_dtype_from_dtype(dtype: torch.dtype) -> torch.dtype:
    """Return a numerically stable dtype based on ``dtype``."""

    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _module_dtype_from_tensor(tensor: torch.Tensor | None) -> torch.dtype:
    """Return a numerically stable dtype for module parameters."""

    if tensor is None:
        return torch.get_default_dtype()
    return _module_dtype_from_dtype(tensor.dtype)


def _module_to_reference(module: nn.Module, reference: torch.Tensor) -> nn.Module:
    """Move ``module`` to ``reference``'s device using a safe dtype."""

    target_dtype = _module_dtype_from_tensor(reference)
    return module.to(device=reference.device, dtype=target_dtype)


class FFTPeriodSelector(nn.Module):
    """Shared dominant period selector based on FFT magnitude spectra."""

    def __init__(
        self, k_periods: int, pmax: int, min_period_threshold: int = 1
    ) -> None:
        super().__init__()
        self.k = int(max(0, k_periods))
        self.pmax = int(max(1, pmax))
        min_thresh = int(max(1, min_period_threshold))
        self.min_period_threshold = int(min(self.pmax, min_thresh))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select shared dominant periods for ``x``.

        Args:
            x: Input tensor shaped ``[B, L, C]`` where ``L`` is the temporal
                dimension and ``C`` enumerates feature channels.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Selected period lengths ``[K]`` as ``torch.long`` values.
                - Corresponding per-sample amplitudes ``[B, K]`` for weighting.
        """

        if x.ndim != 3:
            raise ValueError("FFTPeriodSelector expects input shaped [B, L, C]")

        B, L, C = x.shape
        device = x.device
        dtype = x.dtype

        if self.k <= 0 or L <= 1 or C <= 0 or B <= 0:
            empty_idx = torch.zeros(0, dtype=torch.long, device=device)
            empty_amp = torch.zeros(B, 0, dtype=dtype, device=device)
            return empty_idx, empty_amp

        fft_dtype = x.dtype
        if fft_dtype in (torch.float16, torch.bfloat16):
            fft_dtype = torch.float32

        autocast_enabled = (
            x.is_cuda
            and torch.cuda.is_available()
            and torch.is_autocast_enabled()
        )
        autocast_context = (
            torch.amp.autocast("cuda", enabled=False)
            if autocast_enabled
            else nullcontext()
        )

        with autocast_context:
            fft_input = x.to(fft_dtype) if x.dtype != fft_dtype else x
            spec = torch.fft.rfft(fft_input, dim=1)
            amp = torch.abs(spec)
        amp_channel_median = amp.median(dim=2).values
        amp_mean = amp_channel_median.mean(dim=0)
        amp_samples = amp_channel_median

        if amp_mean.numel() <= 1:
            empty_idx = torch.zeros(0, dtype=torch.long, device=device)
            empty_amp = torch.zeros(B, 0, dtype=amp_samples.dtype, device=device)
            return empty_idx, empty_amp.to(dtype)

        amp_mean = amp_mean.to(dtype)
        amp_mean[0] = amp_mean.new_tensor(float("-inf"))  # Remove DC component

        available = amp_mean.numel() - 1
        k = min(self.k, available)
        if k <= 0:
            empty_idx = torch.zeros(0, dtype=torch.long, device=device)
            empty_amp = torch.zeros(B, 0, dtype=amp_samples.dtype, device=device)
            return empty_idx, empty_amp.to(dtype)

        freq_indices = torch.arange(amp_mean.numel(), device=device, dtype=torch.long)
        log_indices = torch.log1p(freq_indices.to(torch.float32))
        scores = amp_mean - 1e-8 * log_indices.to(dtype)
        _, indices = torch.topk(scores, k=k, largest=True)
        safe_indices = indices.to(device=device, dtype=torch.long).clamp_min(1)
        sample_values = amp_samples.gather(
            1, safe_indices.view(1, -1).expand(B, -1)
        )

        L_t = torch.tensor(L, dtype=torch.long, device=device)
        upper_bound = min(self.pmax, max(L - 1, self.min_period_threshold))
        if upper_bound < self.min_period_threshold:
            empty_idx = torch.zeros(0, dtype=torch.long, device=device)
            empty_amp = torch.zeros(B, 0, dtype=amp_samples.dtype, device=device)
            return empty_idx, empty_amp.to(dtype)

        periods = (L_t + safe_indices - 1) // safe_indices
        periods = torch.clamp(periods, min=self.min_period_threshold, max=upper_bound)

        cycles = (L_t + periods - 1) // periods
        valid_mask = cycles >= 2
        if not torch.any(valid_mask):
            empty_idx = torch.zeros(0, dtype=torch.long, device=device)
            empty_amp = torch.zeros(B, 0, dtype=amp_samples.dtype, device=device)
            return empty_idx, empty_amp.to(dtype)

        periods = periods[valid_mask]
        sample_values = sample_values[:, valid_mask]

        return periods, sample_values.to(dtype)


class InceptionBranch(nn.Module):
    """Single inception branch with bottlenecked convolutions."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: tuple[int, int],
        bottleneck_ratio: float,
    ) -> None:
        super().__init__()
        if bottleneck_ratio <= 0:
            raise ValueError("bottleneck_ratio must be a positive value")
        kh, kw = kernel_size
        pad = (max(kh // 2, 0), max(kw // 2, 0))
        if math.isclose(bottleneck_ratio, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            # Preserve the legacy single convolution behaviour when no
            # bottlenecking is requested.
            self.branch = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=(kh, kw), padding=pad)
            )
        else:
            base = min(in_ch, out_ch)
            mid_ch = max(
                1, int(math.ceil(base / float(bottleneck_ratio)))
            )
            self.branch = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, kernel_size=1),
                nn.Conv2d(mid_ch, mid_ch, kernel_size=(kh, kw), padding=pad),
                nn.Conv2d(mid_ch, out_ch, kernel_size=1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(x)


class InceptionBlock(nn.Module):
    """2D inception block that preserves the cycle/period grid."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_set: Sequence[Sequence[int] | int | Tuple[int, int]],
        dropout: float,
        act: str,
        bottleneck_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        parsed_kernels: list[tuple[int, int]] = []
        for k in kernel_set:
            if isinstance(k, tuple):
                kh, kw = k
            elif isinstance(k, Sequence):
                if len(k) != 2:
                    raise ValueError("kernel_set entries must be (kh, kw) pairs")
                kh, kw = k
            else:
                kh = kw = int(k)
            parsed_kernels.append((int(kh), int(kw)))
        if not parsed_kernels:
            raise ValueError("kernel_set must contain at least one kernel size")
        self.paths = nn.ModuleList(
            [
                InceptionBranch(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    kernel_size=(kh, kw),
                    bottleneck_ratio=bottleneck_ratio,
                )
                for kh, kw in parsed_kernels
            ]
        )
        self.proj = nn.Conv2d(out_ch * len(parsed_kernels), out_ch, kernel_size=1)
        if in_ch != out_ch:
            self.res_proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.res_proj = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        act_name = act.lower()
        if act_name == "relu":
            self.act = nn.ReLU()
        else:
            self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inception-style convolutions on 2D inputs."""

        res = self.res_proj(x)
        feats = [path(x) for path in self.paths]
        z = torch.cat(feats, dim=1)
        z = self.proj(z)
        z = self.act(z)
        z = self.dropout(z)
        return z + res


class TimesBlock(nn.Module):
    """TimesNet block operating on ``[B, L, C]`` features."""

    def __init__(
        self,
        d_model: int | None,
        kernel_set: Sequence[Sequence[int] | int | Tuple[int, int]],
        dropout: float,
        activation: str,
        d_ff: int | None = None,
        bottleneck_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        self._configured_d_model = int(d_model) if d_model is not None else None
        if d_ff is None:
            self._configured_d_ff: int | None = None
        else:
            self._configured_d_ff = int(d_ff)
            if self._configured_d_ff <= 0:
                raise ValueError("d_ff must be a positive integer")
        self.d_model: int | None = None
        self.d_ff: int | None = None
        self.bottleneck_ratio = float(bottleneck_ratio)
        if self.bottleneck_ratio <= 0:
            raise ValueError("bottleneck_ratio must be a positive value")

        act_name = activation.lower()
        if act_name == "relu":
            self._activation_name = "relu"
        else:
            self._activation_name = "gelu"
        kernel_spec: list[tuple[int, int]] = []
        for k in kernel_set:
            if isinstance(k, tuple):
                kh, kw = k
            elif isinstance(k, Sequence):
                if len(k) != 2:
                    raise ValueError("kernel_set entries must be (kh, kw) pairs")
                kh, kw = k
            else:
                kh = kw = int(k)
            kernel_spec.append((int(kh), int(kw)))
        if not kernel_spec:
            raise ValueError("kernel_set must contain at least one kernel size")
        self._kernel_spec = kernel_spec
        self._dropout = float(dropout)
        self.inception: nn.Sequential | None = None
        if self._configured_d_model is not None:
            self._build_layers(
                self._configured_d_model,
                device=torch.device("cpu"),
                dtype=torch.get_default_dtype(),
            )
        # ``period_selector`` is injected from ``TimesNet`` after instantiation to
        # avoid registering the shared selector multiple times.
        self.period_selector: FFTPeriodSelector | None = None
        self._period_calls: int = 0
        self._vec_calls: int = 0

    def _build_layers(self, channels: int, device: torch.device, dtype: torch.dtype) -> None:
        if channels <= 0:
            raise ValueError("TimesBlock requires a positive channel count")
        self.d_model = int(channels)
        hidden = self._configured_d_ff if self._configured_d_ff is not None else self.d_model
        if hidden <= 0:
            raise ValueError("Derived hidden dimension must be positive")
        self.d_ff = int(hidden)
        target_dtype = _module_dtype_from_dtype(dtype)
        if self._activation_name == "relu":
            mid_activation: nn.Module = nn.ReLU()
        else:
            mid_activation = nn.GELU()
        self.inception = nn.Sequential(
            InceptionBlock(
                in_ch=self.d_model,
                out_ch=self.d_ff,
                kernel_set=self._kernel_spec,
                dropout=self._dropout,
                act=self._activation_name,
                bottleneck_ratio=self.bottleneck_ratio,
            ),
            mid_activation,
            InceptionBlock(
                in_ch=self.d_ff,
                out_ch=self.d_model,
                kernel_set=self._kernel_spec,
                dropout=self._dropout,
                act=self._activation_name,
                bottleneck_ratio=self.bottleneck_ratio,
            ),
        ).to(device=device, dtype=target_dtype)
        self.inception = self.inception.to(memory_format=torch.channels_last)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply weighted period residuals to ``x``."""

        if x.ndim != 3:
            raise ValueError("TimesBlock expects input shaped [B, L, d_model]")
        if self.period_selector is None:
            raise RuntimeError("TimesBlock.period_selector has not been set")

        self._period_calls = getattr(self, "_period_calls", 0) + 1
        if self.inception is None:
            if self._configured_d_model is not None and x.size(-1) != self._configured_d_model:
                raise ValueError(
                    "Configured d_model does not match the incoming channel dimension"
                )
            self._build_layers(x.size(-1), device=x.device, dtype=x.dtype)
        else:
            self.inception = _module_to_reference(self.inception, x)
            self.inception = self.inception.to(memory_format=torch.channels_last)
            if self.d_model is not None and x.size(-1) != self.d_model:
                raise ValueError("Number of channels changed between calls")

        periods, amplitudes = self.period_selector(x)
        valid = int((periods > 0).sum().item())
        total = int(periods.numel())
        self._valid_period_calls = getattr(self, "_valid_period_calls", 0) + valid
        self._zero_period_calls = getattr(self, "_zero_period_calls", 0) + (total - valid)
        if periods.numel() == 0:
            return x

        B, L, C = x.shape
        device = x.device
        dtype = x.dtype

        amplitudes = amplitudes.to(device=device, dtype=dtype)
        periods = periods.to(device=device, dtype=torch.long)

        disable_flag = os.getenv("TIMESBLOCK_VEC_DISABLE")
        use_vectorized = True
        if disable_flag and disable_flag.strip().lower() not in {"0", "false", "off"}:
            use_vectorized = False

        if use_vectorized:
            combined = self._period_conv_vectorized(x, periods, amplitudes)
        else:
            combined = self._period_conv_loop(x, periods, amplitudes)

        if combined is None:
            return x
        return x + combined

    def _combine_period_residuals(
        self,
        residuals: list[torch.Tensor],
        amplitudes: torch.Tensor,
        valid_indices: list[int],
        batch_size: int,
        periods: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not residuals:
            return None

        stacked = torch.stack(residuals, dim=-1)
        amp = amplitudes
        if amp.dim() == 1:
            amp = amp.view(1, -1).expand(batch_size, -1)
        amp = amp[:, valid_indices] if amp.numel() > 0 else amp
        if amp.numel() > 0:
            softmax_dtype = amp.dtype
            if softmax_dtype in (torch.float16, torch.bfloat16):
                amp_for_softmax = amp.to(dtype=torch.float32)
                weights_float = F.softmax(amp_for_softmax, dim=1)
            else:
                weights_float = F.softmax(amp, dim=1)
            should_verify = False
            if periods is not None and valid_indices:
                idx_tensor = torch.as_tensor(
                    valid_indices, device=periods.device, dtype=torch.long
                )
                period_vals = torch.index_select(periods, 0, idx_tensor)
                period_hash = int(period_vals.sum().item()) if period_vals.numel() > 0 else 0
                should_verify = (period_hash % 97) == 1
            if should_verify:
                eps = torch.finfo(weights_float.dtype).eps
                weight_sum = weights_float.sum(dim=1, keepdim=True)
                zero_mask = weight_sum <= eps
                if zero_mask.any():
                    uniform = torch.full_like(
                        weights_float, 1.0 / max(weights_float.size(1), 1)
                    )
                    weights_float = torch.where(zero_mask, uniform, weights_float)
                    weight_sum = torch.where(
                        zero_mask, torch.ones_like(weight_sum), weight_sum
                    )
                weights_float = weights_float / weight_sum.clamp_min(eps)
            weights_flat = weights_float.to(dtype=amp.dtype)
        else:
            weights_flat = amp
        weights = weights_flat.view(batch_size, 1, 1, -1)
        return (stacked * weights).sum(dim=-1)

    def _period_conv_loop(
        self, x: torch.Tensor, periods: torch.Tensor, amplitudes: torch.Tensor
    ) -> torch.Tensor | None:
        B, L, C = x.shape
        x_perm = x.permute(0, 2, 1).contiguous()
        residuals: list[torch.Tensor] = []
        valid_indices: list[int] = []

        for idx in range(periods.numel()):
            period = int(periods[idx].item())
            if period <= 0:
                continue
            pad_len = (-L) % period
            if pad_len > 0:
                x_pad = F.pad(x_perm, (0, pad_len))
            else:
                x_pad = x_perm
            total_len = x_pad.size(-1)
            cycles = total_len // period
            if cycles < 2:
                continue
            grid = x_pad.view(B, C, cycles, period)
            conv_out = self.inception(grid)
            delta = conv_out - grid
            delta = delta.view(B, C, total_len)
            delta = delta.permute(0, 2, 1)
            if pad_len > 0:
                delta = delta[:, :-pad_len, :]
            residuals.append(delta)
            valid_indices.append(idx)

        return self._combine_period_residuals(
            residuals, amplitudes, valid_indices, B, periods
        )

    def _period_conv_vectorized(
        self, x: torch.Tensor, periods: torch.Tensor, amplitudes: torch.Tensor
    ) -> torch.Tensor | None:
        B, L, C = x.shape
        device = x.device
        dtype = x.dtype

        periods_flat = periods.view(-1)
        if periods_flat.numel() == 0:
            return None

        if not isinstance(self.inception, nn.Sequential):
            return self._period_conv_loop(x, periods, amplitudes)

        x_perm = x.permute(0, 2, 1).contiguous()
        valid_info: list[tuple[int, int, int, int]] = []
        padded_grids: list[torch.Tensor] = []
        H_max = 0
        W_max = 0

        for idx in range(periods_flat.numel()):
            period_val = int(periods_flat[idx].item())
            if period_val <= 0:
                continue
            pad_len = (-L) % period_val
            total_len = L + pad_len
            cycles = total_len // period_val if period_val > 0 else 0
            if cycles < 2:
                continue
            if pad_len > 0:
                x_pad = F.pad(x_perm, (0, pad_len))
            else:
                x_pad = x_perm
            grid = x_pad.view(B, C, cycles, period_val)
            H_max = max(H_max, cycles)
            W_max = max(W_max, period_val)
            valid_info.append((idx, period_val, pad_len, cycles))
            padded_grids.append(grid)

        if not padded_grids:
            return None

        final_grids: list[torch.Tensor] = []
        final_masks: list[torch.Tensor] = []
        for grid, (_, period_val, pad_len, cycles) in zip(padded_grids, valid_info):
            pad_h = H_max - cycles
            pad_w = W_max - period_val
            if pad_h or pad_w:
                grid_padded = F.pad(grid, (0, pad_w, 0, pad_h))
            else:
                grid_padded = grid
            final_grids.append(grid_padded)
            mask = torch.ones(
                (B, 1, cycles, period_val), device=device, dtype=dtype
            )
            if pad_h or pad_w:
                mask = F.pad(mask, (0, pad_w, 0, pad_h))
            final_masks.append(mask)

        grid_stack = torch.stack(final_grids, dim=1)
        mask_stack = torch.stack(final_masks, dim=1)

        B_eff, K_valid = grid_stack.shape[:2]
        grid_batch = grid_stack.permute(1, 0, 2, 3, 4).reshape(
            K_valid * B_eff, C, H_max, W_max
        )
        mask_batch = mask_stack.permute(1, 0, 2, 3, 4).reshape(
            K_valid * B_eff, 1, H_max, W_max
        )

        grid_batch = grid_batch.contiguous(memory_format=torch.channels_last)
        mask_batch = mask_batch.contiguous()

        chunk_env = os.getenv("TIMESBLOCK_K_CHUNK")
        if chunk_env:
            try:
                chunk_limit = max(1, int(chunk_env))
            except ValueError:
                chunk_limit = K_valid
        else:
            chunk_limit = K_valid
        chunk_limit = max(1, min(chunk_limit, K_valid))

        if isinstance(self.inception, nn.Sequential):
            layered_inception: list[nn.Module] | None = list(self.inception)
        else:
            layered_inception = None

        residuals: list[torch.Tensor] = []
        start = 0
        info_index = 0
        self._vec_calls += 1
        while start < K_valid:
            end = min(start + chunk_limit, K_valid)
            batch_start = start * B_eff
            batch_end = end * B_eff
            batch = grid_batch[batch_start:batch_end]
            batch_mask = mask_batch[batch_start:batch_end]
            batch = batch.contiguous(memory_format=torch.channels_last)
            if layered_inception is not None:
                z = batch * batch_mask
                for layer in layered_inception:
                    z = layer(z)
                    z = z * batch_mask
                conv_out = z
            else:
                conv_out = self.inception(batch)
                conv_out = conv_out * batch_mask
            delta = (conv_out - batch) * batch_mask
            chunk_len = end - start
            delta = delta.view(chunk_len, B_eff, C, H_max, W_max)
            delta = delta.permute(1, 0, 2, 3, 4)
            for local in range(chunk_len):
                orig_idx, period_val, pad_len, cycles = valid_info[info_index]
                info_index += 1
                delta_hw = delta[:, local, :, :cycles, :period_val]
                delta_flat = delta_hw.reshape(B_eff, C, cycles * period_val)
                if pad_len > 0:
                    delta_flat = delta_flat[:, :, :-pad_len]
                delta_perm = delta_flat.permute(0, 2, 1).contiguous()
                residuals.append(delta_perm)
            start = end

        valid_indices = [info[0] for info in valid_info]
        return self._combine_period_residuals(
            residuals, amplitudes, valid_indices, B_eff, periods
        )


class PositionalEmbedding(nn.Module):
    """Deterministic sinusoidal positional encoding."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = int(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("PositionalEmbedding expects input shaped [B, L, C]")
        B, L, _ = x.shape
        device = x.device
        orig_dtype = x.dtype
        calc_dtype = torch.float32
        position = torch.arange(L, device=device, dtype=calc_dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=calc_dtype)
            * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(L, self.d_model, device=device, dtype=calc_dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        cos_term = div_term
        if pe[:, 1::2].shape[1] != cos_term.shape[0]:
            cos_term = cos_term[: pe[:, 1::2].shape[1]]
        pe[:, 1::2] = torch.cos(position * cos_term)
        return pe.to(dtype=orig_dtype).unsqueeze(0).expand(B, -1, -1)


class RMSNorm(nn.Module):
    """Root-mean-square normalization with affine parameters."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("RMSNorm expects a positive embedding dimension")
        self.eps = float(eps)
        dim = int(d_model)
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.weight.numel():
            raise ValueError("RMSNorm dimension mismatch")
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        scale = torch.rsqrt(variance + self.eps)
        normed = x * scale
        return normed * self.weight + self.bias


class DataEmbedding(nn.Module):
    """Value + positional (+ optional temporal) embedding."""

    _VALID_NORM_MODES = {"none", "layer", "rms", "decoupled"}

    def __init__(
        self,
        c_in: int,
        d_model: int,
        dropout: float,
        time_features: int | None = None,
        use_norm: bool = True,
        embed_norm_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.value_embedding = nn.Linear(int(c_in), int(d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        if time_features is not None and time_features > 0:
            self.temporal_embedding: nn.Module | None = nn.Linear(
                int(time_features), int(d_model)
            )
        else:
            self.temporal_embedding = None
        use_norm_flag = bool(use_norm)
        if embed_norm_mode is None:
            embed_norm_mode = "decoupled" if use_norm_flag else "none"
        mode = embed_norm_mode.lower()
        if mode not in self._VALID_NORM_MODES:
            raise ValueError(
                f"embed_norm_mode must be one of {sorted(self._VALID_NORM_MODES)}, got {embed_norm_mode!r}"
            )
        self.embed_norm_mode = mode
        if not use_norm_flag and mode != "none":
            # Preserve backward compatibility: explicit mode selection overrides use_norm flag.
            use_norm_flag = True
        self.use_norm = mode != "none"
        self.norm: nn.Module | None
        self.aux_norm: nn.Module | None
        if mode == "layer":
            self.norm = nn.LayerNorm(int(d_model))
            self.aux_norm = None
            self.register_parameter("gate", None)
        elif mode == "rms":
            self.norm = RMSNorm(int(d_model))
            self.aux_norm = None
            self.register_parameter("gate", None)
        elif mode == "decoupled":
            self.norm = None
            self.aux_norm = nn.LayerNorm(int(d_model))
            gate = torch.full((1, 1, int(d_model)), 0.1, dtype=torch.float32)
            self.gate = nn.Parameter(gate)
        else:  # none
            self.norm = None
            self.aux_norm = None
            self.register_parameter("gate", None)
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self, x: torch.Tensor, x_mark: torch.Tensor | None = None
    ) -> torch.Tensor:
        if x.ndim not in (3, 4):
            raise ValueError(
                "DataEmbedding expects input shaped [B, L, C] or [B, L, N, C]"
            )

        mark: torch.Tensor | None
        if x.ndim == 4:
            B, L, N, C = x.shape
            x_flat = x.reshape(B * N, L, C)
            if x_mark is None:
                mark = None
            else:
                if x_mark.ndim == 3:
                    if x_mark.shape[0] != B or x_mark.shape[1] != L:
                        raise ValueError(
                            "x_mark must match batch/time dimensions of x"
                        )
                    mark_expanded = x_mark.unsqueeze(2).expand(-1, -1, N, -1)
                elif x_mark.ndim == 4:
                    if x_mark.shape[:3] != (B, L, N):
                        raise ValueError(
                            "x_mark must align with [B, L, N] dimensions of x"
                        )
                    mark_expanded = x_mark
                else:
                    raise ValueError(
                        "x_mark must have shape [B, L, T] or [B, L, N, T]"
                    )
                mark = mark_expanded.reshape(B * N, L, mark_expanded.size(-1))
        else:
            x_flat = x
            if x_mark is not None and x_mark.ndim != 3:
                raise ValueError("x_mark must share dimensions [B, L, T]")
            mark = x_mark

        value = self.value_embedding(x_flat)
        pos = self.position_embedding(x_flat)
        temporal = (
            self.temporal_embedding(mark)
            if self.temporal_embedding is not None and mark is not None
            else None
        )
        aux = pos + temporal if isinstance(temporal, torch.Tensor) else pos
        if aux.ndim == 4 or value.ndim == 4:
            raise AssertionError("DataEmbedding should not operate on 4D tensors")

        if self.embed_norm_mode == "decoupled":
            assert self.aux_norm is not None and isinstance(self.gate, torch.Tensor)
            aux_normed = self.aux_norm(aux)
            gate = self.gate
            if gate.dtype != value.dtype:
                gate = gate.to(value.dtype)
            out = value + gate * aux_normed
        else:
            out = value + aux
            if self.embed_norm_mode == "layer":
                assert self.norm is not None
                out = self.norm(out)
            elif self.embed_norm_mode == "rms":
                assert self.norm is not None
                out = self.norm(out)
        out = self.dropout(out)

        if x.ndim == 4:
            return out.view(B, L, N, out.size(-1))
        return out


class LowRankTemporalContext(nn.Module):
    """Synthesize zero-mean temporal context signals without 4D tensors."""

    def __init__(self, rank: int, init_scale: float = 1e-2) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LowRankTemporalContext requires a positive rank")
        self.rank = int(rank)
        self.scale = nn.Parameter(torch.as_tensor(float(init_scale), dtype=torch.float32))
        self.register_buffer("_cached_basis", torch.empty(0), persistent=False)
        self._cached_length: int = 0

    def _compute_basis(
        self, length: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        calc_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        steps = torch.arange(length, device=device, dtype=calc_dtype).unsqueeze(1)
        freqs = torch.arange(1, self.rank + 1, device=device, dtype=calc_dtype).unsqueeze(0)
        basis = torch.cos(math.pi / float(length) * (steps + 0.5) * freqs)
        basis = basis - basis.mean(dim=0, keepdim=True)
        norm = torch.linalg.norm(basis, dim=0, keepdim=True)
        eps = torch.finfo(basis.dtype).eps
        basis = basis / norm.clamp_min(eps)
        return basis.to(dtype=dtype)

    def _basis(self, length: int, reference: torch.Tensor) -> torch.Tensor:
        device = reference.device
        dtype = reference.dtype
        if self._cached_basis.numel() == 0 or self._cached_length != length:
            basis = self._compute_basis(length, device, dtype)
            self._cached_basis = basis.detach()
            self._cached_length = length
        return self._cached_basis.to(device=device, dtype=dtype)

    def forward(self, coeff: torch.Tensor, length: int) -> torch.Tensor:
        if coeff.ndim != 3:
            raise ValueError("LowRankTemporalContext expects coeff shaped [B, N, R]")
        if coeff.size(-1) != self.rank:
            raise ValueError("Coefficient dimension mismatch with configured rank")
        basis = self._basis(length, coeff)
        context = torch.einsum("lr,bnr->bln", basis, coeff)
        context = context - context.mean(dim=1, keepdim=True)
        scale = self.scale.to(device=coeff.device, dtype=coeff.dtype)
        return context * scale


class TimesNet(nn.Module):
    """TimesNet with official embedding + linear forecasting head."""

    def __init__(
        self,
        input_len: int,
        pred_len: int,
        d_model: int,
        n_layers: int,
        k_periods: int,
        kernel_set: Sequence[Sequence[int] | int | Tuple[int, int]],
        dropout: float,
        activation: str,
        mode: str,
        d_ff: int | None = None,
        bottleneck_ratio: float = 1.0,
        min_period_threshold: int = 1,
        channels_last: bool = False,
        use_checkpoint: bool = True,
        use_embedding_norm: bool = True,
        embed_norm_mode: str | None = None,
        min_sigma: float = 1e-3,
        min_sigma_vector: torch.Tensor | Sequence[float] | None = None,
        id_embed_dim: int = 32,
        static_proj_dim: int | None = None,
        static_layernorm: bool = True,
        use_zero_mean_context: bool = False,
        context_rank: int = 0,
        context_scale: float = 1e-2,
        use_constant_context_bias: bool = False,
        use_late_bias_head: bool = True,
    ) -> None:
        super().__init__()
        del channels_last  # retained for backward compatibility
        assert mode in ("direct", "recursive")
        self.mode = mode
        self.input_len = int(input_len)
        self.pred_len = int(pred_len)
        self.requested_d_model = int(d_model)
        if d_ff is None:
            self.requested_d_ff: int | None = None
        else:
            requested_ff = int(d_ff)
            if requested_ff <= 0:
                raise ValueError("d_ff must be a positive integer")
            self.requested_d_ff = requested_ff
        self.d_model: int | None = None
        self.d_ff: int | None = self.requested_d_ff
        self.bottleneck_ratio = float(bottleneck_ratio)
        if self.bottleneck_ratio <= 0:
            raise ValueError("bottleneck_ratio must be a positive value")
        self.n_layers = int(n_layers)
        self.dropout = float(dropout)
        self.use_checkpoint = bool(use_checkpoint)
        self.use_embedding_norm = bool(use_embedding_norm)
        if embed_norm_mode is None:
            resolved_mode = "decoupled" if self.use_embedding_norm else "none"
        else:
            resolved_mode = embed_norm_mode
        self.embed_norm_mode = resolved_mode
        self.min_sigma = float(min_sigma)
        self.k_periods = int(k_periods)
        self.kernel_set = list(kernel_set)
        self.period_selector = FFTPeriodSelector(
            k_periods=self.k_periods,
            pmax=self.input_len,
            min_period_threshold=min_period_threshold,
        )
        self.blocks = nn.ModuleList(
            [
                TimesBlock(
                    d_model=None,
                    d_ff=self.requested_d_ff,
                    kernel_set=self.kernel_set,
                    dropout=self.dropout,
                    activation=activation,
                    bottleneck_ratio=self.bottleneck_ratio,
                )
                for _ in range(self.n_layers)
            ]
        )
        for block in self.blocks:
            object.__setattr__(block, "period_selector", self.period_selector)
        self.residual_dropout = nn.Dropout(self.dropout)
        self.layer_norm: nn.LayerNorm | None = None
        self.forecast_time_proj = nn.Linear(self.input_len, self.pred_len)
        with torch.no_grad():
            self.forecast_time_proj.weight.zero_()
            if self.pred_len > 0:
                self.forecast_time_proj.weight[:, -1] = 1.0
            if self.forecast_time_proj.bias is not None:
                self.forecast_time_proj.bias.zero_()
        self.embedding: DataEmbedding | None = None
        self.embedding_time_features: int | None = None
        self.output_proj: nn.Conv1d | None = None
        self.sigma_proj: nn.Conv1d | None = None
        self.mu_head: nn.Linear | None = None
        self.sigma_head: nn.Linear | None = None
        self.output_dim: int | None = None
        self.input_channels: int | None = None
        self._out_steps = self.pred_len if self.mode == "direct" else 1
        self.register_buffer("min_sigma_vector", None)
        if min_sigma_vector is not None:
            min_sigma_tensor = torch.as_tensor(min_sigma_vector, dtype=torch.float32)
            self.min_sigma_vector = min_sigma_tensor.reshape(1, 1, -1)
        self.id_embed_dim = int(id_embed_dim)
        if self.id_embed_dim < 0:
            raise ValueError("id_embed_dim must be non-negative")
        if static_proj_dim is None:
            self.static_proj_dim: int | None = None
        else:
            proj_val = int(static_proj_dim)
            if proj_val <= 0:
                raise ValueError("static_proj_dim must be a positive integer when provided")
            self.static_proj_dim = proj_val
        self.static_layernorm = bool(static_layernorm)
        self.series_embedding: nn.Embedding | None = None
        self.static_proj: nn.Linear | None = None
        self.static_norm: nn.Module | None = None
        self.context_norm: nn.LayerNorm | None = None
        self.context_proj: nn.Linear | None = None
        self.context_coeff: nn.Linear | None = None
        self.temporal_context: LowRankTemporalContext | None = None
        self.late_bias_norm: nn.LayerNorm | None = None
        self.late_bias_head: nn.Linear | None = None
        self.register_parameter("late_bias_gate", None)
        self.pre_embedding_norm: nn.Module | None = None
        self.pre_embedding_dropout = nn.Dropout(self.dropout)
        self._static_in_features: int | None = None
        self._static_out_dim: int = 0
        self._series_id_vocab: int | None = None
        self._series_id_reference: torch.Tensor | None = None
        self.debug_memory: bool = False
        self.use_zero_mean_context = bool(use_zero_mean_context)
        self.use_constant_context_bias = bool(use_constant_context_bias)
        self.use_late_bias_head = bool(use_late_bias_head)
        self.context_rank = int(context_rank)
        if self.context_rank < 0:
            raise ValueError("context_rank must be non-negative")
        self.context_scale_default = float(context_scale)

    def _ensure_embedding(
        self,
        x: torch.Tensor,
        x_mark: torch.Tensor | None = None,
        series_static: torch.Tensor | None = None,
        series_ids: torch.Tensor | None = None,
    ) -> None:
        """Lazily instantiate embedding/output projection when dimensions are known."""

        c_in = int(x.size(-1))
        time_dim = int(x_mark.size(-1)) if x_mark is not None else 0
        if self.input_channels is None:
            self.input_channels = c_in
        elif self.input_channels != c_in:
            raise ValueError("Number of series changed between calls")

        if self.requested_d_model is None:
            raise ValueError("TimesNet requires a configured d_model")
        if self.d_model is None:
            self.d_model = int(self.requested_d_model)
        elif self.d_model != int(self.requested_d_model):
            raise ValueError("d_model changed between calls")

        if self.requested_d_ff is None:
            self.d_ff = self.d_model
        else:
            self.d_ff = self.requested_d_ff

        static_out_dim = 0
        if series_static is not None:
            if series_static.ndim == 2:
                static_ref = series_static
            elif series_static.ndim == 3:
                static_ref = series_static[0]
            else:
                raise ValueError("series_static must have shape [N, F] or [B, N, F]")
            if static_ref.size(0) != c_in:
                raise ValueError(
                    "series_static must align with the number of input series"
                )
            static_in = int(static_ref.size(-1))
            if static_in <= 0:
                raise ValueError("series_static must have at least one feature")
            if self.static_proj is None:
                proj_dim = (
                    self.static_proj_dim if self.static_proj_dim is not None else static_in
                )
                self.static_proj = _module_to_reference(
                    nn.Linear(static_in, proj_dim), x
                )
                if self.static_layernorm:
                    self.static_norm = _module_to_reference(
                        nn.LayerNorm(proj_dim), x
                    )
                else:
                    self.static_norm = nn.Identity()
                self._static_in_features = static_in
                self._static_out_dim = proj_dim
            else:
                if self.static_proj.in_features != static_in:
                    raise ValueError(
                        "series_static feature dimension changed between calls"
                    )
                self.static_proj = _module_to_reference(self.static_proj, x)
                if self.static_norm is not None:
                    self.static_norm = _module_to_reference(self.static_norm, x)
                self._static_out_dim = int(self.static_proj.out_features)
            static_out_dim = self._static_out_dim
        elif self.static_proj is not None:
            self.static_proj = _module_to_reference(self.static_proj, x)
            if self.static_norm is not None:
                self.static_norm = _module_to_reference(self.static_norm, x)
            static_out_dim = int(self.static_proj.out_features)
            self._static_out_dim = static_out_dim
        else:
            self._static_out_dim = 0

        ids_reference: torch.Tensor | None = None
        id_feature_dim = 0
        if self.id_embed_dim > 0:
            if series_ids is not None:
                if series_ids.ndim == 1:
                    ids_reference = series_ids.to(torch.long)
                elif series_ids.ndim == 2:
                    ids_reference = series_ids[0].to(torch.long)
                else:
                    raise ValueError(
                        "series_ids must have shape [N] or [B, N]"
                    )
                if ids_reference.numel() != c_in:
                    raise ValueError(
                        "series_ids length must match number of series"
                    )
            if self.series_embedding is None:
                if ids_reference is None:
                    ids_reference = torch.arange(
                        c_in, device=x.device, dtype=torch.long
                    )
                vocab = int(ids_reference.max().item()) + 1 if ids_reference.numel() > 0 else c_in
                self.series_embedding = _module_to_reference(
                    nn.Embedding(vocab, self.id_embed_dim), x
                )
                self._series_id_vocab = vocab
                self._series_id_reference = ids_reference.to(device=x.device)
            else:
                self.series_embedding = _module_to_reference(
                    self.series_embedding, x
                )
                if ids_reference is not None:
                    vocab = int(ids_reference.max().item()) + 1 if ids_reference.numel() > 0 else c_in
                    if vocab > int(self.series_embedding.num_embeddings):
                        raise ValueError(
                            "series_ids vocabulary expanded between calls"
                        )
                    self._series_id_reference = ids_reference.to(device=x.device)
                elif self._series_id_reference is None:
                    self._series_id_reference = torch.arange(
                        c_in, device=x.device, dtype=torch.long
                    )
                self._series_id_vocab = int(self.series_embedding.num_embeddings)
            if (
                self._series_id_reference is not None
                and self._series_id_reference.numel() != c_in
            ):
                raise ValueError("series identifier count changed between calls")
            id_feature_dim = int(self.series_embedding.embedding_dim)

        context_dim = static_out_dim + id_feature_dim
        if context_dim > 0:
            if (
                self.context_norm is None
                or tuple(self.context_norm.normalized_shape) != (context_dim,)
            ):
                self.context_norm = _module_to_reference(
                    nn.LayerNorm(context_dim), x
                )
            else:
                self.context_norm = _module_to_reference(self.context_norm, x)
            if self.use_zero_mean_context and self.context_rank > 0:
                if (
                    self.context_coeff is None
                    or self.context_coeff.in_features != context_dim
                    or self.context_coeff.out_features != self.context_rank
                ):
                    coeff = nn.Linear(context_dim, self.context_rank)
                    with torch.no_grad():
                        coeff.weight.zero_()
                        if coeff.bias is not None:
                            coeff.bias.zero_()
                    self.context_coeff = _module_to_reference(coeff, x)
                else:
                    self.context_coeff = _module_to_reference(self.context_coeff, x)
                if (
                    self.temporal_context is None
                    or self.temporal_context.rank != self.context_rank
                ):
                    self.temporal_context = _module_to_reference(
                        LowRankTemporalContext(
                            rank=self.context_rank,
                            init_scale=self.context_scale_default,
                        ),
                        x,
                    )
                else:
                    self.temporal_context = _module_to_reference(
                        self.temporal_context, x
                    )
            else:
                self.context_coeff = None
                self.temporal_context = None
            if self.use_constant_context_bias and context_dim > 0:
                if (
                    self.context_proj is None
                    or self.context_proj.in_features != context_dim
                ):
                    self.context_proj = _module_to_reference(
                        nn.Linear(context_dim, 1), x
                    )
                    with torch.no_grad():
                        self.context_proj.weight.zero_()
                        self.context_proj.bias.zero_()
                else:
                    self.context_proj = _module_to_reference(self.context_proj, x)
            else:
                self.context_proj = None
            if self.use_late_bias_head:
                if (
                    self.late_bias_norm is None
                    or tuple(self.late_bias_norm.normalized_shape)
                    != (context_dim,)
                ):
                    self.late_bias_norm = _module_to_reference(
                        nn.LayerNorm(context_dim), x
                    )
                else:
                    self.late_bias_norm = _module_to_reference(
                        self.late_bias_norm, x
                    )
                if (
                    self.late_bias_head is None
                    or self.late_bias_head.in_features != context_dim
                    or self.late_bias_head.out_features != self._out_steps
                ):
                    head = nn.Linear(context_dim, self._out_steps)
                    with torch.no_grad():
                        head.weight.zero_()
                        head.bias.zero_()
                    self.late_bias_head = _module_to_reference(head, x)
                else:
                    self.late_bias_head = _module_to_reference(
                        self.late_bias_head, x
                    )
                gate_shape = (1, self._out_steps, 1)
                if (
                    not isinstance(self.late_bias_gate, nn.Parameter)
                    or tuple(self.late_bias_gate.shape) != gate_shape
                ):
                    gate_param = torch.full(
                        gate_shape, 0.05, dtype=torch.float32, device=x.device
                    )
                    self.late_bias_gate = nn.Parameter(gate_param)
                else:
                    self.late_bias_gate.data = self.late_bias_gate.data.to(
                        device=x.device, dtype=torch.float32
                    )
            else:
                self.late_bias_norm = None
                self.late_bias_head = None
                if isinstance(self.late_bias_gate, nn.Parameter):
                    self.late_bias_gate = None
        else:
            self.context_norm = None
            self.context_proj = None
            self.context_coeff = None
            self.temporal_context = None
            self.late_bias_norm = None
            self.late_bias_head = None
            if isinstance(self.late_bias_gate, nn.Parameter):
                self.late_bias_gate = None

        total_per_series = 1 + context_dim
        if total_per_series <= 1:
            if not isinstance(self.pre_embedding_norm, nn.Identity):
                self.pre_embedding_norm = nn.Identity()
            self.pre_embedding_norm = self.pre_embedding_norm.to(device=x.device)
        else:
            needs_new = not isinstance(self.pre_embedding_norm, nn.LayerNorm)
            if not needs_new:
                assert isinstance(self.pre_embedding_norm, nn.LayerNorm)
                needs_new = (
                    tuple(self.pre_embedding_norm.normalized_shape)
                    != (total_per_series,)
                )
            if needs_new:
                self.pre_embedding_norm = _module_to_reference(
                    nn.LayerNorm(total_per_series), x
                )
            else:
                self.pre_embedding_norm = _module_to_reference(
                    self.pre_embedding_norm, x
                )
        self.pre_embedding_dropout = self.pre_embedding_dropout.to(device=x.device)

        if isinstance(self.min_sigma_vector, torch.Tensor) and self.min_sigma_vector.numel() > 0:
            current = int(self.min_sigma_vector.shape[-1])
            if current < c_in:
                raise ValueError(
                    "min_sigma_vector length does not match number of series"
                )
            if current != c_in:
                self.min_sigma_vector = self.min_sigma_vector[..., :c_in]

        if self.embedding_time_features is not None and self.embedding_time_features != time_dim:
            raise ValueError("Temporal feature dimension changed between calls")

        if self.embedding is None:
            embed = DataEmbedding(
                c_in=c_in,
                d_model=self.d_model,
                dropout=self.dropout,
                time_features=time_dim if time_dim > 0 else None,
                use_norm=self.use_embedding_norm,
                embed_norm_mode=self.embed_norm_mode,
            )
            self.embedding = _module_to_reference(embed, x)
        else:
            self.embedding = _module_to_reference(self.embedding, x)
        self.embedding_time_features = time_dim

        self.forecast_time_proj = _module_to_reference(
            self.forecast_time_proj, x
        )

        if self.layer_norm is None or (
            isinstance(self.layer_norm, nn.LayerNorm)
            and tuple(self.layer_norm.normalized_shape) != (self.d_model,)
        ):
            self.layer_norm = _module_to_reference(
                nn.LayerNorm(self.d_model), x
            )
        else:
            self.layer_norm = _module_to_reference(self.layer_norm, x)

        if self.output_proj is None or self.output_proj.in_channels != self.d_model:
            self.output_proj = _module_to_reference(
                nn.Conv1d(self.d_model, self.d_model, kernel_size=1), x
            )
            with torch.no_grad():
                self.output_proj.weight.zero_()
                self.output_proj.bias.zero_()
        else:
            self.output_proj = _module_to_reference(self.output_proj, x)
        if self.sigma_proj is None or self.sigma_proj.in_channels != self.d_model:
            self.sigma_proj = _module_to_reference(
                nn.Conv1d(self.d_model, self.d_model, kernel_size=1), x
            )
            with torch.no_grad():
                self.sigma_proj.weight.zero_()
                self.sigma_proj.bias.zero_()
        else:
            self.sigma_proj = _module_to_reference(self.sigma_proj, x)

        if self.mu_head is None or (
            self.mu_head.in_features != self.d_model
            or self.mu_head.out_features != c_in
        ):
            self.mu_head = _module_to_reference(
                nn.Linear(self.d_model, c_in), x
            )
            with torch.no_grad():
                self.mu_head.weight.zero_()
                self.mu_head.bias.zero_()
        else:
            self.mu_head = _module_to_reference(self.mu_head, x)

        if self.sigma_head is None or (
            self.sigma_head.in_features != self.d_model
            or self.sigma_head.out_features != c_in
        ):
            self.sigma_head = _module_to_reference(
                nn.Linear(self.d_model, c_in), x
            )
            with torch.no_grad():
                self.sigma_head.weight.zero_()
                self.sigma_head.bias.zero_()
        else:
            self.sigma_head = _module_to_reference(self.sigma_head, x)

        self.output_dim = self.input_channels

    def _dispersion_floor_from_ref(self, ref: torch.Tensor) -> torch.Tensor:
        if isinstance(self.min_sigma_vector, torch.Tensor) and self.min_sigma_vector.numel() > 0:
            floor = self.min_sigma_vector.to(device=ref.device, dtype=ref.dtype)
            return floor.expand_as(ref).clone()
        return ref.new_full(ref.shape, self.min_sigma)

    def forward(
        self,
        x: torch.Tensor,
        x_mark: torch.Tensor | None = None,
        series_static: torch.Tensor | None = None,
        series_ids: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError("TimesNet expects input shaped [B, T, N]")
        B, T, N = x.shape
        if T < self.input_len:
            raise ValueError(
                f"Input sequence length {T} is shorter than required input_len {self.input_len}"
            )
        if x_mark is not None:
            if x_mark.shape[:2] != x.shape[:2]:
                raise ValueError("x_mark must share batch/time dimensions with x")
            mark_slice = x_mark[:, -self.input_len :, :]
        else:
            mark_slice = None
        enc_x_value = x[:, -self.input_len :, :]
        enc_x_features = enc_x_value.clone()
        self._ensure_embedding(enc_x_features, mark_slice, series_static, series_ids)
        target_steps = self.pred_len if self.mode == "direct" else self._out_steps
        time_len = enc_x_value.size(1)

        context_components: list[torch.Tensor] = []
        context_concat: torch.Tensor | None = None

        if self.static_proj is not None and series_static is not None:
            if series_static.ndim == 2:
                static_input = series_static.unsqueeze(0).expand(B, -1, -1)
            elif series_static.ndim == 3:
                if series_static.size(0) != B:
                    raise ValueError(
                        "series_static batch dimension must match input batch size"
                    )
                static_input = series_static
            else:
                raise ValueError(
                    "series_static must have shape [N, F] or [B, N, F]"
                )
            static_proj = self.static_proj(
                static_input.to(
                    device=enc_x_features.device,
                    dtype=enc_x_features.dtype,
                    non_blocking=enc_x_features.is_cuda,
                )
            )
            if self.static_norm is not None:
                static_proj = self.static_norm(static_proj)
            context_components.append(static_proj)

        if self.series_embedding is not None and self.id_embed_dim > 0:
            if series_ids is None:
                if self._series_id_reference is None:
                    ids_tensor = torch.arange(
                        N, device=enc_x_features.device, dtype=torch.long
                    ).unsqueeze(0)
                else:
                    ids_tensor = self._series_id_reference.view(1, -1).to(
                        enc_x_features.device
                    )
                    if ids_tensor.size(1) != N:
                        raise ValueError(
                            "Stored series identifiers do not match input dimension"
                        )
                if ids_tensor.size(0) == 1 and B > 1:
                    ids_tensor = ids_tensor.expand(B, -1)
            else:
                ids_tensor = series_ids
                if ids_tensor.ndim == 1:
                    ids_tensor = ids_tensor.unsqueeze(0)
                if ids_tensor.ndim != 2:
                    raise ValueError(
                        "series_ids must have shape [N] or [B, N]"
                    )
                if ids_tensor.size(0) == 1 and B > 1:
                    ids_tensor = ids_tensor.expand(B, -1)
                if ids_tensor.size(0) != B:
                    raise ValueError(
                        "series_ids batch dimension does not match input"
                    )
                if ids_tensor.size(1) != N:
                    raise ValueError(
                        "series_ids length must match number of series"
                    )
                ids_tensor = ids_tensor.to(
                    device=enc_x_features.device, dtype=torch.long
                )
                self._series_id_reference = ids_tensor[0].detach().clone()
            ids_tensor = ids_tensor.to(
                device=enc_x_features.device, dtype=torch.long
            )
            id_embed = self.series_embedding(ids_tensor)
            context_components.append(id_embed)

        if context_components:
            context_concat = torch.cat(context_components, dim=-1)
            if self.context_norm is not None:
                context_concat = self.context_norm(context_concat)
            if (
                self.use_zero_mean_context
                and self.context_coeff is not None
                and self.temporal_context is not None
            ):
                coeff_input = context_concat.to(
                    dtype=self.context_coeff.weight.dtype
                )
                coeff = self.context_coeff(coeff_input)
                context_signal = self.temporal_context(coeff, enc_x_features.size(1))
                # Enforce that the temporal context remains 3D and avoids [B, L, N, C] tensors.
                if context_signal.ndim == 4:
                    raise RuntimeError(
                        "Temporal context must remain 3D to avoid 4D tensors"
                    )
                if context_signal.shape[:2] != enc_x_features.shape[:2]:
                    raise RuntimeError(
                        "Temporal context must align with [B, L] dimensions"
                    )
                if context_signal.size(-1) != enc_x_features.size(-1):
                    raise RuntimeError(
                        "Temporal context must align with input series dimension"
                    )
                enc_x_features = enc_x_features + context_signal.to(
                    dtype=enc_x_features.dtype
                )
            if self.use_constant_context_bias and self.context_proj is not None:
                bias_input = context_concat.to(
                    dtype=self.context_proj.weight.dtype
                )
                bias = self.context_proj(bias_input).squeeze(-1)
                enc_x_features = enc_x_features + bias.to(
                    dtype=enc_x_features.dtype
                ).unsqueeze(1)

        if enc_x_features.ndim == 4:
            raise RuntimeError("Embedding input must remain 3D; found 4D tensor")

        features = self.embedding(enc_x_features, mark_slice)
        if features.ndim != 3:
            raise RuntimeError(
                "Embedding output must have shape [B, L, d_model]"
            )
        if features.size(1) != self.input_len:
            raise RuntimeError("Embedded sequence length mismatch with input_len")
        if features.size(-1) != self.d_model:
            raise RuntimeError(
                "Embedding output dimension mismatch with configured d_model"
            )
        seq_features = features
        hist_steps = min(target_steps, time_len)
        history_tail = enc_x_value[:, -hist_steps:, :]
        if hist_steps < target_steps:
            pad = history_tail[:, -1:, :].expand(-1, target_steps - hist_steps, -1)
            history_tail = torch.cat([history_tail, pad], dim=1)

        history_tail = history_tail.to(enc_x_value.dtype)

        if self.debug_memory and seq_features.is_cuda and torch.cuda.is_available():
            mem_bytes = torch.cuda.memory_allocated(seq_features.device)
            print(
                f"[TimesNet] CUDA memory allocated after embedding: {mem_bytes / (1024 ** 2):.2f} MiB"
            )

        self.period_selector = self.period_selector.to(
            device=seq_features.device, dtype=seq_features.dtype
        )
        for block in self.blocks:
            object.__setattr__(block, "period_selector", self.period_selector)

        def _apply_late_bias(pre_activation: torch.Tensor) -> torch.Tensor:
            if (
                context_concat is not None
                and self.late_bias_head is not None
                and self.late_bias_norm is not None
                and isinstance(self.late_bias_gate, nn.Parameter)
            ):
                c = context_concat.to(
                    dtype=self.late_bias_head.weight.dtype,
                    device=self.late_bias_head.weight.device,
                )
                c = self.late_bias_norm(c)
                bias = self.late_bias_head(c)
                bias = bias.permute(0, 2, 1).contiguous()
                gate = self.late_bias_gate.to(
                    dtype=pre_activation.dtype, device=pre_activation.device
                )
                pre_activation = pre_activation + gate * bias.to(
                    dtype=pre_activation.dtype, device=pre_activation.device
                )
            return pre_activation

        for block in self.blocks:
            if seq_features.shape[-1] != self.d_model:
                raise RuntimeError(
                    "Residual input to TimesBlock must maintain d_model channels"
                )
            if self.use_checkpoint:
                updated = checkpoint(block, seq_features, use_reentrant=False)
            else:
                updated = block(seq_features)
            delta = updated - seq_features
            seq_features = seq_features + self.residual_dropout(delta)
            seq_features = self.layer_norm(seq_features)

        features_final = seq_features
        features_bn = features_final.permute(0, 2, 1).contiguous()
        assert (
            self.forecast_time_proj.in_features == self.input_len
        ), "forecast_time_proj input size mismatch"
        assert (
            self.forecast_time_proj.out_features == self.pred_len
        ), "forecast_time_proj output size mismatch"
        baseline_bn_full = self.forecast_time_proj(features_bn)
        if target_steps != self.pred_len:
            baseline_bn = baseline_bn_full[:, :, -target_steps:].contiguous()
        else:
            baseline_bn = baseline_bn_full.contiguous()
        assert self.output_proj is not None  # for type checkers
        assert self.mu_head is not None  # for type checkers
        mu_delta_bn = self.output_proj(baseline_bn)
        mu_hidden = (baseline_bn + mu_delta_bn).permute(0, 2, 1).contiguous()
        rate_preact = self.mu_head(mu_hidden) + history_tail
        rate_preact = _apply_late_bias(rate_preact)
        rate = F.softplus(rate_preact) + 1e-6
        assert self.sigma_proj is not None  # for type checkers
        sigma_bn = self.sigma_proj(baseline_bn)
        sigma_hidden = sigma_bn.permute(0, 2, 1).contiguous()
        assert self.sigma_head is not None  # for type checkers
        sigma_head = self.sigma_head(sigma_hidden)
        floor = self._dispersion_floor_from_ref(rate)
        dispersion = F.softplus(sigma_head) + floor + 1e-6
        if torch.any(~torch.isfinite(rate)) or torch.any(rate <= 0):
            raise RuntimeError("Predicted rate must be finite and strictly positive")
        if torch.any(~torch.isfinite(dispersion)) or torch.any(dispersion <= 0):
            raise RuntimeError("Predicted dispersion must be finite and strictly positive")
        if rate.shape != (B, target_steps, N):
            raise RuntimeError("Predicted rate has incorrect shape")
        if dispersion.shape != (B, target_steps, N):
            raise RuntimeError("Predicted dispersion has incorrect shape")
        return rate, dispersion
