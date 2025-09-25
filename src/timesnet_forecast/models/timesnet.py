from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Tuple, Sequence
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
            torch.cuda.amp.autocast(enabled=False)
            if autocast_enabled
            else nullcontext()
        )

        with autocast_context:
            fft_input = x.to(fft_dtype) if x.dtype != fft_dtype else x
            spec = torch.fft.rfft(fft_input, dim=1)
            amp = torch.abs(spec)
        amp_mean = amp.mean(dim=(0, 2))
        amp_samples = amp.mean(dim=2)

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

        freq_indices = torch.arange(amp_mean.numel(), device=device, dtype=dtype)
        tie_break = freq_indices * torch.finfo(dtype).eps
        scores = amp_mean - tie_break
        _, indices = torch.topk(scores, k=k, largest=True)
        safe_indices = indices.to(device=device, dtype=torch.long).clamp_min(1)
        sample_values = amp_samples.gather(
            1, safe_indices.view(1, -1).expand(B, -1)
        )

        L_t = torch.tensor(L, dtype=torch.long, device=device)
        periods = (L_t + safe_indices - 1) // safe_indices
        periods = torch.clamp(
            periods,
            min=self.min_period_threshold,
            max=self.pmax,
        )

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply weighted period residuals to ``x``."""

        if x.ndim != 3:
            raise ValueError("TimesBlock expects input shaped [B, L, d_model]")
        if self.period_selector is None:
            raise RuntimeError("TimesBlock.period_selector has not been set")

        if self.inception is None:
            if self._configured_d_model is not None and x.size(-1) != self._configured_d_model:
                raise ValueError(
                    "Configured d_model does not match the incoming channel dimension"
                )
            self._build_layers(x.size(-1), device=x.device, dtype=x.dtype)
        else:
            self.inception = _module_to_reference(self.inception, x)
            if self.d_model is not None and x.size(-1) != self.d_model:
                raise ValueError("Number of channels changed between calls")

        periods, amplitudes = self.period_selector(x)
        if periods.numel() == 0:
            return x

        B, L, C = x.shape
        device = x.device
        dtype = x.dtype

        amplitudes = amplitudes.to(device=device, dtype=dtype)
        periods = periods.to(device=device, dtype=torch.long)

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
            grid = x_pad.view(B, C, cycles, period)
            conv_out = self.inception(grid)
            delta = conv_out - grid
            delta = delta.view(B, C, total_len)
            delta = delta.permute(0, 2, 1)
            if pad_len > 0:
                delta = delta[:, :-pad_len, :]
            residuals.append(delta)
            valid_indices.append(idx)

        if not residuals:
            return x

        stacked = torch.stack(residuals, dim=-1)  # [B, L, C, K_valid]

        if amplitudes.dim() == 1:
            amplitudes = amplitudes.view(1, -1).expand(B, -1)
        amp = amplitudes[:, valid_indices] if amplitudes.numel() > 0 else amplitudes
        weights = F.softmax(amp, dim=1) if amp.numel() > 0 else amp
        weights = weights.view(B, 1, 1, -1)
        combined = (stacked * weights).sum(dim=-1)
        return x + combined


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
        dtype = x.dtype
        position = torch.arange(L, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(L, self.d_model, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        cos_term = div_term
        if pe[:, 1::2].shape[1] != cos_term.shape[0]:
            cos_term = cos_term[: pe[:, 1::2].shape[1]]
        pe[:, 1::2] = torch.cos(position * cos_term)
        return pe.unsqueeze(0).expand(B, -1, -1)


class DataEmbedding(nn.Module):
    """Value + positional (+ optional temporal) embedding."""

    def __init__(
        self,
        c_in: int,
        d_model: int,
        dropout: float,
        time_features: int | None = None,
        use_norm: bool = True,
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
        self.use_norm = bool(use_norm)
        self.norm = nn.LayerNorm(int(d_model))
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
        if isinstance(temporal, torch.Tensor):
            out = value + pos + temporal
        else:
            out = value + pos
        if self.use_norm:
            out = self.norm(out)
        out = self.dropout(out)

        if x.ndim == 4:
            return out.view(B, L, N, out.size(-1))
        return out


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
        min_sigma: float = 1e-3,
        min_sigma_vector: torch.Tensor | Sequence[float] | None = None,
        id_embed_dim: int = 32,
        static_proj_dim: int | None = None,
        static_layernorm: bool = True,
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
        self.min_sigma = float(min_sigma)
        self.k_periods = int(k_periods)
        self.kernel_set = list(kernel_set)
        self.period_selector = FFTPeriodSelector(
            k_periods=self.k_periods,
            pmax=self.input_len + self.pred_len,
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
        self.predict_linear = nn.Linear(self.input_len, self.input_len + self.pred_len)
        with torch.no_grad():
            self.predict_linear.weight.zero_()
            identity = torch.eye(self.input_len)
            self.predict_linear.weight[: self.input_len, :] = identity
            if self.pred_len > 0:
                self.predict_linear.weight[self.input_len :, -1] = 1.0
            self.predict_linear.bias.zero_()
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
        self.pre_embedding_norm: nn.Module | None = None
        self.pre_embedding_dropout = nn.Dropout(self.dropout)
        self._static_in_features: int | None = None
        self._static_out_dim: int = 0
        self._series_id_vocab: int | None = None
        self._series_id_reference: torch.Tensor | None = None
        self.debug_memory: bool = False

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
            if self.context_proj is None or self.context_proj.in_features != context_dim:
                self.context_proj = _module_to_reference(
                    nn.Linear(context_dim, 1), x
                )
                with torch.no_grad():
                    self.context_proj.weight.zero_()
                    self.context_proj.bias.zero_()
            else:
                self.context_proj = _module_to_reference(self.context_proj, x)
        else:
            self.context_norm = None
            self.context_proj = None

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

        if (
            isinstance(self.min_sigma_vector, torch.Tensor)
            and self.min_sigma_vector.numel() > 0
            and self.min_sigma_vector.shape[-1] != c_in
        ):
            raise ValueError(
                "min_sigma_vector length does not match number of series"
            )

        if self.embedding_time_features is not None and self.embedding_time_features != time_dim:
            raise ValueError("Temporal feature dimension changed between calls")

        if self.embedding is None:
            embed = DataEmbedding(
                c_in=c_in,
                d_model=self.d_model,
                dropout=self.dropout,
                time_features=time_dim if time_dim > 0 else None,
                use_norm=self.use_embedding_norm,
            )
            self.embedding = _module_to_reference(embed, x)
        else:
            self.embedding = _module_to_reference(self.embedding, x)
        self.embedding_time_features = time_dim

        self.predict_linear = self.predict_linear.to(
            device=x.device, dtype=_module_dtype_from_tensor(x)
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

    def _sigma_from_ref(self, ref: torch.Tensor) -> torch.Tensor:
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
        enc_x = x[:, -self.input_len :, :]
        self._ensure_embedding(enc_x, mark_slice, series_static, series_ids)
        target_steps = self.pred_len if self.mode == "direct" else self._out_steps
        time_len = enc_x.size(1)

        context_components: list[torch.Tensor] = []

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
                    device=enc_x.device,
                    dtype=enc_x.dtype,
                    non_blocking=enc_x.is_cuda,
                )
            )
            if self.static_norm is not None:
                static_proj = self.static_norm(static_proj)
            context_components.append(static_proj)

        if self.series_embedding is not None and self.id_embed_dim > 0:
            if series_ids is None:
                if self._series_id_reference is None:
                    ids_tensor = torch.arange(
                        N, device=enc_x.device, dtype=torch.long
                    ).unsqueeze(0)
                else:
                    ids_tensor = self._series_id_reference.view(1, -1).to(enc_x.device)
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
                ids_tensor = ids_tensor.to(device=enc_x.device, dtype=torch.long)
                self._series_id_reference = ids_tensor[0].detach().clone()
            ids_tensor = ids_tensor.to(device=enc_x.device, dtype=torch.long)
            id_embed = self.series_embedding(ids_tensor)
            context_components.append(id_embed)

        if context_components and self.context_proj is not None:
            context_concat = torch.cat(context_components, dim=-1)
            if self.context_norm is not None:
                context_concat = self.context_norm(context_concat)
            bias = self.context_proj(context_concat).squeeze(-1)
            enc_x = enc_x + bias.unsqueeze(1)

        features = self.embedding(enc_x, mark_slice)
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
        feat_bn = features.permute(0, 2, 1).contiguous()
        feat_flat = feat_bn.reshape(B * self.d_model, self.input_len)
        extended = self.predict_linear(feat_flat)
        extended = extended.view(B, self.d_model, self.input_len + self.pred_len)
        features = extended.permute(0, 2, 1).contiguous()
        total_len = features.size(1)
        baseline = features[:, -target_steps:, :].contiguous()
        hist_steps = min(target_steps, time_len)
        history_tail = enc_x[:, -hist_steps:, :]
        if hist_steps < target_steps:
            pad = history_tail[:, -1:, :].expand(-1, target_steps - hist_steps, -1)
            history_tail = torch.cat([history_tail, pad], dim=1)

        history_tail = history_tail.to(enc_x.dtype)

        if self.debug_memory and features.is_cuda and torch.cuda.is_available():
            mem_bytes = torch.cuda.memory_allocated(features.device)
            print(
                f"[TimesNet] CUDA memory allocated after predict_linear: {mem_bytes / (1024 ** 2):.2f} MiB"
            )

        self.period_selector = self.period_selector.to(
            device=features.device, dtype=features.dtype
        )
        for block in self.blocks:
            object.__setattr__(block, "period_selector", self.period_selector)

        seq_features = features

        preview_periods, _ = self.period_selector(seq_features)
        if preview_periods.numel() == 0:
            mu_hidden = baseline
            mu = self.mu_head(mu_hidden) + history_tail
            sigma_hidden = self.sigma_head(baseline)
            floor = self._sigma_from_ref(mu)
            sigma = F.softplus(sigma_hidden) + floor
            return mu, sigma

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
        features = seq_features
        target_features = features[:, -target_steps:, :].contiguous()
        target_bn = target_features.permute(0, 2, 1).contiguous()
        mu_delta_bn = self.output_proj(target_bn)
        mu_delta = mu_delta_bn.permute(0, 2, 1).contiguous()
        mu_hidden = baseline + mu_delta
        mu = self.mu_head(mu_hidden) + history_tail
        assert self.sigma_proj is not None  # for type checkers
        sigma_bn = self.sigma_proj(target_bn)
        sigma_hidden = sigma_bn.permute(0, 2, 1).contiguous()
        sigma_head = self.sigma_head(sigma_hidden)
        floor = self._sigma_from_ref(mu)
        sigma = F.softplus(sigma_head) + floor
        return mu, sigma
