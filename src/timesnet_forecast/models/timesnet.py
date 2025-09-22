from __future__ import annotations

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

        spec = torch.fft.rfft(x, dim=1)
        amp = torch.abs(spec)
        amp_mean = amp.mean(dim=(0, 2))
        amp_samples = amp.mean(dim=2)

        if amp_mean.numel() <= 1:
            empty_idx = torch.zeros(0, dtype=torch.long, device=device)
            empty_amp = torch.zeros(B, 0, dtype=amp_samples.dtype, device=device)
            return empty_idx, empty_amp.to(dtype)

        amp_mean = amp_mean.to(dtype)
        amp_mean[0] = 0.0  # Remove DC component

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
        sample_values = amp_samples.gather(
            1, indices.view(1, -1).expand(B, -1)
        )
        indices = torch.clamp(indices, min=1)

        periods = torch.div(L, indices, rounding_mode="floor")
        periods = periods.to(torch.long)
        periods = torch.clamp(
            periods,
            min=self.min_period_threshold,
            max=self.pmax,
        )

        return periods, sample_values.to(dtype)


class InceptionBlock(nn.Module):
    """2D inception block that preserves the cycle/period grid."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_set: Sequence[Sequence[int] | int | Tuple[int, int]],
        dropout: float,
        act: str,
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
        self.paths = nn.ModuleList()
        for kh, kw in parsed_kernels:
            pad = (max(kh // 2, 0), max(kw // 2, 0))
            self.paths.append(
                nn.Conv2d(in_ch, out_ch, kernel_size=(kh, kw), padding=pad)
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
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.inception = InceptionBlock(
            in_ch=self.d_model,
            out_ch=self.d_model,
            kernel_set=kernel_set,
            dropout=dropout,
            act=activation,
        )
        # ``period_selector`` is injected from ``TimesNet`` after instantiation to
        # avoid registering the shared selector multiple times.
        self.period_selector: FFTPeriodSelector | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute residual updates for ``x`` using dominant periods."""

        if x.ndim != 3:
            raise ValueError("TimesBlock expects input shaped [B, L, d_model]")
        if self.period_selector is None:
            raise RuntimeError("TimesBlock.period_selector has not been set")

        periods, amplitudes = self.period_selector(x)
        if periods.numel() == 0:
            return torch.zeros_like(x)

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
            return torch.zeros_like(x)

        stacked = torch.stack(residuals, dim=-1)  # [B, L, C, K_valid]

        if amplitudes.dim() == 1:
            amplitudes = amplitudes.view(1, -1).expand(B, -1)
        amp = amplitudes[:, valid_indices] if amplitudes.numel() > 0 else amplitudes
        weights = F.softmax(amp, dim=1) if amp.numel() > 0 else amp
        weights = weights.view(B, 1, 1, -1)
        combined = (stacked * weights).sum(dim=-1)
        return combined


class TimesNet(nn.Module):
    """
    입력 [B, T, N]에서 FFT 기반으로 공유 주기를 선택하고,
    2D InceptionBlock이 [채널(K), 주기(C), 길이(P)] 격자에서 패턴을 추출한 뒤
    주기 축 풀링과 residual 스택을 통해 예측을 생성합니다.
    """
    def __init__(
        self,
        input_len: int,
        pred_len: int,
        d_model: int,
        n_layers: int,
        k_periods: int,
        pmax: int,
        kernel_set: Sequence[Sequence[int] | int | Tuple[int, int]],
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
        self.period_selector = FFTPeriodSelector(
            k_periods=k_periods,
            pmax=pmax,
            min_period_threshold=min_period_threshold,
        )
        self.k_periods = int(k_periods)
        self.input_len = int(input_len)
        self.act = activation
        # We don't know the period-length ``P`` at build time, so layers are built lazily
        # during the first forward pass once the flattened length ``K*P`` is known.
        self.blocks: nn.ModuleList = nn.ModuleList()
        self.input_proj: nn.Module = nn.Identity()
        self.residual_dropout: nn.Module = nn.Identity()
        self.pool: nn.Module = nn.Identity()
        self.head: nn.Module = nn.Identity()
        self._lazy_built = False
        self.kernel_set = list(kernel_set)
        self.dropout = float(dropout)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.use_checkpoint = bool(use_checkpoint)
        self.min_sigma = float(min_sigma)
        self.register_buffer("min_sigma_vector", None)
        if min_sigma_vector is not None:
            min_sigma_tensor = torch.as_tensor(min_sigma_vector, dtype=torch.float32)
            self.min_sigma_vector = min_sigma_tensor.reshape(1, 1, -1)

    def _build_lazy(self, x: torch.Tensor) -> None:
        """Instantiate convolutional blocks on first use.

        Args:
            x: reference tensor for device/dtype placement
        """
        self.input_proj = nn.Linear(1, self.d_model).to(device=x.device, dtype=x.dtype)
        with torch.no_grad():
            self.input_proj.weight.zero_()
            self.input_proj.bias.zero_()
            if self.input_proj.weight.size(0) > 0:
                self.input_proj.weight[0, 0] = 1.0
        blocks: list[TimesBlock] = []
        for _ in range(self.n_layers):
            block = TimesBlock(
                d_model=self.d_model,
                kernel_set=self.kernel_set,
                dropout=self.dropout,
                activation=self.act,
            ).to(device=x.device, dtype=x.dtype)
            object.__setattr__(block, "period_selector", self.period_selector)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.residual_dropout = nn.Dropout(self.dropout)
        # Pool only over the temporal dimension; series dimension is preserved.
        self.pool = nn.AdaptiveAvgPool1d(self._out_steps)
        # Lightweight 1x1 conv head emits (mu, log_sigma) per horizon step.
        self.head = nn.Conv1d(self.d_model, 2, kernel_size=1).to(
            device=x.device, dtype=x.dtype
        )
        self._lazy_built = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        if not self._lazy_built:
            self._build_lazy(x=x)

        BN = B * N
        series = x.permute(0, 2, 1).reshape(BN, T, 1)
        features = self.input_proj(series)

        self.period_selector = self.period_selector.to(
            device=features.device, dtype=features.dtype
        )
        for block in self.blocks:
            object.__setattr__(block, "period_selector", self.period_selector)

        preview_periods, _ = self.period_selector(features)
        if preview_periods.numel() == 0:
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

        for block in self.blocks:
            if self.use_checkpoint:
                delta = checkpoint(block, features, use_reentrant=False)
            else:
                delta = block(features)
            delta = self.residual_dropout(delta)
            features = features + delta

        z = features.permute(0, 2, 1)
        z = self.pool(z)
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
