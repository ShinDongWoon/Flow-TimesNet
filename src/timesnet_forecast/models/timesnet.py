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
                - Corresponding averaged amplitudes ``[K]`` for weighting.
        """

        if x.ndim != 3:
            raise ValueError("FFTPeriodSelector expects input shaped [B, L, C]")

        B, L, C = x.shape
        device = x.device
        dtype = x.dtype

        if self.k <= 0 or L <= 1 or C <= 0 or B <= 0:
            empty_idx = torch.zeros(0, dtype=torch.long, device=device)
            empty_amp = torch.zeros(0, dtype=dtype, device=device)
            return empty_idx, empty_amp

        spec = torch.fft.rfft(x, dim=1)
        amp = torch.abs(spec)
        amp_mean = amp.mean(dim=(0, 2))

        if amp_mean.numel() <= 1:
            empty_idx = torch.zeros(0, dtype=torch.long, device=device)
            empty_amp = torch.zeros(0, dtype=amp_mean.dtype, device=device)
            return empty_idx, empty_amp.to(dtype)

        amp_mean = amp_mean.to(dtype)
        amp_mean[0] = 0.0  # Remove DC component

        available = amp_mean.numel() - 1
        k = min(self.k, available)
        if k <= 0:
            empty_idx = torch.zeros(0, dtype=torch.long, device=device)
            empty_amp = torch.zeros(0, dtype=dtype, device=device)
            return empty_idx, empty_amp

        freq_indices = torch.arange(amp_mean.numel(), device=device, dtype=dtype)
        tie_break = freq_indices * torch.finfo(dtype).eps
        scores = amp_mean - tie_break
        _, indices = torch.topk(scores, k=k, largest=True)
        values = amp_mean.gather(0, indices)
        indices = torch.clamp(indices, min=1)

        periods = torch.div(L, indices, rounding_mode="floor")
        periods = periods.to(torch.long)
        periods = torch.clamp(
            periods,
            min=self.min_period_threshold,
            max=self.pmax,
        )

        return periods, values


def fold_by_periods(
    seqs: torch.Tensor, periods: torch.Tensor, pmax: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fold 1D sequences according to ``periods`` producing a period grid.

    Args:
        seqs: Tensor shaped ``[B, L]`` containing the sequences to fold.
        periods: Integer tensor ``[K]`` with period lengths to extract.
        pmax: Maximum length of the period axis in the folded representation.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Folded tensor ``[B, K, C_max, pmax]``.
            - Mask tensor with the same shape marking valid elements.
    """

    if periods.numel() == 0:
        empty = seqs.new_zeros(seqs.size(0), 0, 1, pmax)
        return empty, empty

    device = seqs.device
    L = seqs.size(1)
    periods = periods.to(device=device, dtype=torch.long)
    periods = torch.clamp(periods, min=1, max=int(pmax))

    cycles = torch.div(L, torch.clamp(periods, min=1), rounding_mode="floor")
    cycles = torch.clamp(cycles, min=1)
    take = cycles * periods

    Cmax_t = torch.clamp(cycles.max(), min=1)
    idx_c = torch.arange(Cmax_t, device=device)
    idx_p = torch.arange(int(pmax), device=device)

    base = torch.clamp(L - take, min=0)
    base = base.view(1, -1, 1, 1)
    periods_exp = periods.view(1, -1, 1, 1)

    indices = base + idx_c.view(1, 1, -1, 1) * periods_exp + idx_p.view(1, 1, 1, -1)
    indices = indices.clamp(min=0, max=max(L - 1, 0))
    indices = indices.expand(seqs.size(0), -1, -1, -1)

    seqs_exp = seqs.unsqueeze(1).unsqueeze(2).expand(-1, periods.size(0), idx_c.numel(), -1)
    gathered = torch.gather(seqs_exp, dim=-1, index=indices)

    mask_c = idx_c.view(1, 1, -1, 1) < cycles.view(1, -1, 1, 1)
    mask_p = idx_p.view(1, 1, 1, -1) < periods.view(1, -1, 1, 1)
    mask = (mask_c & mask_p).to(gathered.dtype)
    mask = mask.expand(seqs.size(0), -1, -1, -1).contiguous()

    gathered = gathered * mask
    return gathered, mask


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
    """TimesNet block that consumes folded 2D representations and emits 1D updates."""

    def __init__(
        self,
        period_channels: int,
        d_model: int,
        kernel_set: Sequence[Sequence[int] | int | Tuple[int, int]],
        dropout: float,
        activation: str,
        pmax: int,
        min_period_threshold: int,
    ) -> None:
        super().__init__()
        self.period_channels = int(period_channels)
        self.d_model = int(d_model)
        self.inception = InceptionBlock(
            in_ch=self.period_channels,
            out_ch=1,
            kernel_set=kernel_set,
            dropout=dropout,
            act=activation,
        )
        self.fuse = nn.Conv1d(self.d_model * 3, self.d_model, kernel_size=1)
        act_name = activation.lower()
        if act_name == "relu":
            self.act_layer = nn.ReLU()
        else:
            self.act_layer = nn.GELU()
        self.inner_dropout = nn.Dropout(dropout)
        self.pmax = int(pmax)
        self.min_period_threshold = int(max(1, min_period_threshold))

    def forward(
        self,
        features: torch.Tensor,
        step_mask: torch.Tensor | None,
        periods: torch.Tensor,
        amplitudes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fold the 1D feature map using shared periods and compute an update."""

        B_eff, channel_count, seq_len = features.shape
        device = features.device
        dtype = features.dtype

        periods = periods.to(device=device, dtype=torch.long)
        periods = periods[periods > 0]
        periods = torch.clamp(
            periods,
            min=self.min_period_threshold,
            max=self.pmax,
        )

        if periods.numel() == 0:
            zero_update = torch.zeros_like(features)
            if step_mask is not None:
                new_mask = step_mask.to(dtype=dtype)
            else:
                new_mask = torch.ones(B_eff, 1, seq_len, device=device, dtype=dtype)
            return zero_update, new_mask

        if step_mask is not None:
            features_in = features * step_mask.to(dtype=dtype)
        else:
            features_in = features

        seq = features_in.reshape(B_eff * channel_count, seq_len)
        folded, mask = fold_by_periods(seq, periods, self.pmax)
        if folded.numel() == 0:
            zero_update = torch.zeros_like(features)
            if step_mask is not None:
                new_mask = step_mask.to(dtype=dtype)
            else:
                new_mask = torch.ones(B_eff, 1, seq_len, device=device, dtype=dtype)
            return zero_update, new_mask

        Cmax = folded.size(2)
        K = folded.size(1)
        if K > self.period_channels:
            folded = folded[:, : self.period_channels]
            mask = mask[:, : self.period_channels]
            K = self.period_channels
        if K < self.period_channels:
            pad_shape = (folded.size(0), self.period_channels - K, Cmax, self.pmax)
            folded = torch.cat([folded, folded.new_zeros(pad_shape)], dim=1)
            mask = torch.cat([mask, mask.new_zeros(pad_shape)], dim=1)
        folded = folded.reshape(B_eff, channel_count, self.period_channels, Cmax, self.pmax)
        mask = mask.reshape(B_eff, channel_count, self.period_channels, Cmax, self.pmax)
        mask = mask.to(dtype=folded.dtype)

        conv_in = folded * mask
        conv_in_flat = conv_in.reshape(
            B_eff * channel_count, self.period_channels, Cmax, self.pmax
        )
        z2d_flat = self.inception(conv_in_flat)
        z2d = z2d_flat.reshape(
            B_eff, channel_count, -1, Cmax, self.pmax
        ).squeeze(2)

        cycle_mask = mask.amax(dim=2)
        weights = cycle_mask.to(dtype=z2d.dtype)
        eps = torch.finfo(z2d.dtype).eps
        weighted = z2d * weights
        avg_pool = weighted.sum(dim=2) / (weights.sum(dim=2) + eps)
        max_mask = cycle_mask > 0
        neg_fill = torch.finfo(z2d.dtype).min
        masked = z2d.masked_fill(~max_mask, neg_fill)
        max_pool = masked.max(dim=2).values
        valid_steps = max_mask.any(dim=2).to(dtype=max_pool.dtype)
        max_pool = torch.where(valid_steps > 0, max_pool, torch.zeros_like(max_pool))
        pooled = torch.cat([avg_pool, max_pool], dim=1)

        branch_sum = conv_in.sum(dim=3)
        branch_weight = mask.sum(dim=3)
        branch_recon = branch_sum / (branch_weight + eps)
        branch_recon = branch_recon * (branch_weight > 0).to(dtype=branch_recon.dtype)

        branch_valid = mask.amax(dim=(3, 4)) > 0
        freq_amp: torch.Tensor | None
        if amplitudes is not None and amplitudes.numel() > 0:
            amp = amplitudes.to(device=device, dtype=branch_recon.dtype)
            amp = amp[: self.period_channels]
            if amp.size(0) < self.period_channels:
                pad = torch.zeros(
                    self.period_channels - amp.size(0),
                    device=device,
                    dtype=branch_recon.dtype,
                )
                amp = torch.cat([amp, pad], dim=0)
            freq_amp = amp.view(1, 1, -1).expand(B_eff, channel_count, -1)
        else:
            freq_amp = None

        if freq_amp is not None and freq_amp.numel() > 0:
            invalid_fill = torch.finfo(branch_recon.dtype).min
            amp = torch.where(
                branch_valid,
                freq_amp,
                torch.full_like(freq_amp, invalid_fill),
            )
            weight_logits = amp
            branch_weights = F.softmax(weight_logits, dim=2)
        else:
            branch_weights = branch_valid.to(dtype=branch_recon.dtype)

        branch_weights = branch_weights * branch_valid.to(dtype=branch_recon.dtype)
        weight_denom = branch_weights.sum(dim=2, keepdim=True)
        weight_safe = torch.where(
            weight_denom > 0,
            branch_weights / (weight_denom + eps),
            torch.zeros_like(branch_weights),
        )
        weighted_recon = (branch_recon * weight_safe.unsqueeze(-1)).sum(dim=2)

        cycle_weight = mask.sum(dim=3)
        step_valid = (cycle_weight > 0).any(dim=2)
        new_step_mask = step_valid.any(dim=1, keepdim=True).to(dtype=dtype)
        if step_mask is not None:
            new_step_mask = new_step_mask * (step_mask > 0).to(dtype=dtype)

        mask_to_apply = new_step_mask
        weighted_recon = weighted_recon * mask_to_apply
        features_mod = (features + weighted_recon) * mask_to_apply

        pooled = pooled * mask_to_apply
        combined = torch.cat([pooled, features_mod], dim=1)
        z = self.fuse(combined)
        z = self.act_layer(z)
        z = self.inner_dropout(z)
        z = z * mask_to_apply
        return z, new_step_mask


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
        self.selector_embedding: nn.Module = nn.Identity()
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
        self.selector_embedding = nn.Conv1d(1, self.d_model, kernel_size=1).to(
            device=x.device, dtype=x.dtype
        )
        self.input_proj = nn.Conv1d(self.k_periods, self.d_model, kernel_size=1).to(
            device=x.device, dtype=x.dtype
        )
        blocks = []
        for _ in range(self.n_layers):
            block = TimesBlock(
                period_channels=self.k_periods,
                d_model=self.d_model,
                kernel_set=self.kernel_set,
                dropout=self.dropout,
                activation=self.act,
                pmax=self.period_selector.pmax,
                min_period_threshold=self.period_selector.min_period_threshold,
            ).to(device=x.device, dtype=x.dtype)
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
        seqs = x.permute(0, 2, 1).reshape(BN, T)
        selector_in = seqs.unsqueeze(1)
        selector_embed = self.selector_embedding(selector_in)
        selector_embed = selector_embed.permute(0, 2, 1)
        self.period_selector = self.period_selector.to(
            device=selector_embed.device, dtype=selector_embed.dtype
        )
        periods, amplitudes = self.period_selector(selector_embed)

        if periods.numel() == 0:
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

        folded, mask = fold_by_periods(seqs, periods, self.period_selector.pmax)
        if folded.numel() == 0:
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

        mask = mask.to(dtype=folded.dtype)
        cycle_weight = mask.sum(dim=2)
        eps = torch.finfo(folded.dtype).eps
        aggregated = (folded * mask).sum(dim=2) / (cycle_weight + eps)
        aggregated = aggregated * (cycle_weight > 0).to(dtype=aggregated.dtype)

        K = aggregated.size(1)
        if K > self.k_periods:
            aggregated = aggregated[:, : self.k_periods]
            cycle_weight = cycle_weight[:, : self.k_periods]
        elif K < self.k_periods:
            pad_k = self.k_periods - K
            pad_agg = aggregated.new_zeros(aggregated.size(0), pad_k, aggregated.size(2))
            aggregated = torch.cat([aggregated, pad_agg], dim=1)
            pad_cycle = cycle_weight.new_zeros(cycle_weight.size(0), pad_k, cycle_weight.size(2))
            cycle_weight = torch.cat([cycle_weight, pad_cycle], dim=1)

        step_mask = (cycle_weight > 0).any(dim=1, keepdim=True).to(dtype=aggregated.dtype)
        features = self.input_proj(aggregated)
        features = features * step_mask
        for blk in self.blocks:
            if self.use_checkpoint:
                delta, step_mask = checkpoint(
                    blk,
                    features,
                    step_mask,
                    periods,
                    amplitudes,
                    use_reentrant=False,
                )
            else:
                delta, step_mask = blk(features, step_mask, periods, amplitudes)
            delta = self.residual_dropout(delta)
            features = (features + delta) * step_mask
        z = features * step_mask
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
