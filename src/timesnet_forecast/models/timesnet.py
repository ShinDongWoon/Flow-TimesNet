from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Tuple, Sequence
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


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
    """TimesNet block operating on ``[B, L, d_model]`` features."""

    def __init__(
        self,
        d_model: int,
        kernel_set: Sequence[Sequence[int] | int | Tuple[int, int]],
        dropout: float,
        activation: str,
        d_ff: int | None = None,
        bottleneck_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        if d_ff is None:
            d_ff = self.d_model
        self.d_ff = int(d_ff)
        if self.d_ff <= 0:
            raise ValueError("d_ff must be a positive integer")
        self.bottleneck_ratio = float(bottleneck_ratio)
        if self.bottleneck_ratio <= 0:
            raise ValueError("bottleneck_ratio must be a positive value")

        act_name = activation.lower()
        if act_name == "relu":
            mid_activation: nn.Module = nn.ReLU()
        else:
            mid_activation = nn.GELU()

        kernel_spec = list(kernel_set)
        self.inception = nn.Sequential(
            InceptionBlock(
                in_ch=self.d_model,
                out_ch=self.d_ff,
                kernel_set=kernel_spec,
                dropout=dropout,
                act=activation,
                bottleneck_ratio=self.bottleneck_ratio,
            ),
            mid_activation,
            InceptionBlock(
                in_ch=self.d_ff,
                out_ch=self.d_model,
                kernel_set=kernel_spec,
                dropout=dropout,
                act=activation,
                bottleneck_ratio=self.bottleneck_ratio,
            ),
        )
        # ``period_selector`` is injected from ``TimesNet`` after instantiation to
        # avoid registering the shared selector multiple times.
        self.period_selector: FFTPeriodSelector | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply weighted period residuals to ``x``."""

        if x.ndim != 3:
            raise ValueError("TimesBlock expects input shaped [B, L, d_model]")
        if self.period_selector is None:
            raise RuntimeError("TimesBlock.period_selector has not been set")

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
        value = self.value_embedding(x)
        pos = self.position_embedding(x)
        temporal = (
            self.temporal_embedding(x_mark)
            if self.temporal_embedding is not None and x_mark is not None
            else None
        )
        if isinstance(temporal, torch.Tensor):
            out = value + pos + temporal
        else:
            out = value + pos
        if self.use_norm:
            out = self.norm(out)
        return self.dropout(out)


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
    ) -> None:
        super().__init__()
        del channels_last  # retained for backward compatibility
        assert mode in ("direct", "recursive")
        self.mode = mode
        self.input_len = int(input_len)
        self.pred_len = int(pred_len)
        self.d_model = int(d_model)
        if d_ff is None:
            d_ff = 4 * self.d_model
        self.d_ff = int(d_ff)
        if self.d_ff <= 0:
            raise ValueError("d_ff must be a positive integer")
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
                    d_model=self.d_model,
                    d_ff=self.d_ff,
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
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.predict_linear = nn.Linear(self.input_len, self.input_len + self.pred_len)
        self.embedding: DataEmbedding | None = None
        self.embedding_time_features: int | None = None
        self.output_proj: nn.Linear | None = None
        self.sigma_proj: nn.Linear | None = None
        self.output_dim: int | None = None
        self._out_steps = self.pred_len if self.mode == "direct" else 1
        self.register_buffer("min_sigma_vector", None)
        if min_sigma_vector is not None:
            min_sigma_tensor = torch.as_tensor(min_sigma_vector, dtype=torch.float32)
            self.min_sigma_vector = min_sigma_tensor.reshape(1, 1, -1)

    def _ensure_embedding(
        self, x: torch.Tensor, x_mark: torch.Tensor | None = None
    ) -> None:
        """Lazily instantiate embedding/output projection when dimensions are known."""

        c_in = int(x.size(-1))
        time_dim = int(x_mark.size(-1)) if x_mark is not None else 0
        if self.embedding is None or self.output_proj is None or self.sigma_proj is None:
            if (
                isinstance(self.min_sigma_vector, torch.Tensor)
                and self.min_sigma_vector.numel() > 0
                and self.min_sigma_vector.shape[-1] != c_in
            ):
                raise ValueError(
                    "min_sigma_vector length does not match number of series"
                )
            if time_dim == 0:
                time_arg = None
            else:
                time_arg = time_dim
            self.embedding = DataEmbedding(
                c_in=c_in,
                d_model=self.d_model,
                dropout=self.dropout,
                time_features=time_arg,
                use_norm=self.use_embedding_norm,
            ).to(device=x.device, dtype=x.dtype)
            self.embedding_time_features = time_dim
            self.output_proj = nn.Linear(self.d_model, c_in).to(
                device=x.device, dtype=x.dtype
            )
            self.sigma_proj = nn.Linear(self.d_model, c_in).to(
                device=x.device, dtype=x.dtype
            )
            self.output_dim = c_in
        else:
            if self.output_dim != c_in:
                raise ValueError("Number of series changed between calls")
            expected_time = self.embedding_time_features or 0
            if time_dim != expected_time:
                raise ValueError("Temporal feature dimension changed between calls")

    def _sigma_from_ref(self, ref: torch.Tensor) -> torch.Tensor:
        if isinstance(self.min_sigma_vector, torch.Tensor) and self.min_sigma_vector.numel() > 0:
            floor = self.min_sigma_vector.to(device=ref.device, dtype=ref.dtype)
            return floor.expand_as(ref).clone()
        return ref.new_full(ref.shape, self.min_sigma)

    def forward(
        self, x: torch.Tensor, x_mark: torch.Tensor | None = None
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
        self._ensure_embedding(enc_x, mark_slice)
        target_steps = self.pred_len if self.mode == "direct" else self._out_steps
        features = self.embedding(enc_x, mark_slice)  # type: ignore[arg-type]
        d_model = features.size(-1)
        feat_t = features.permute(0, 2, 1).contiguous()
        feat_flat = feat_t.reshape(B * d_model, self.input_len)
        extended = self.predict_linear(feat_flat)
        extended = extended.view(B, d_model, self.input_len + self.pred_len)
        features = extended.permute(0, 2, 1).contiguous()

        self.period_selector = self.period_selector.to(
            device=features.device, dtype=features.dtype
        )
        for block in self.blocks:
            object.__setattr__(block, "period_selector", self.period_selector)

        preview_periods, _ = self.period_selector(features)
        if preview_periods.numel() == 0:
            mu = enc_x.new_zeros(B, self._out_steps, N)
            sigma = self._sigma_from_ref(mu)
            return mu, sigma

        for block in self.blocks:
            if self.use_checkpoint:
                updated = checkpoint(block, features, use_reentrant=False)
            else:
                updated = block(features)
            delta = updated - features
            features = features + self.residual_dropout(delta)
            features = self.layer_norm(features)
        target_features = features[:, -target_steps:, :].contiguous()
        mu = self.output_proj(target_features)  # type: ignore[operator]
        assert self.sigma_proj is not None  # for type checkers
        floor = self._sigma_from_ref(mu)
        sigma_head = self.sigma_proj(target_features)
        sigma = F.softplus(sigma_head) + floor
        return mu, sigma
