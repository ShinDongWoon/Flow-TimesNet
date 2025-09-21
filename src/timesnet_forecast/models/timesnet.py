from __future__ import annotations

from typing import Tuple, Sequence
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


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
    def _topk_freq(x: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return indices of top-k nonzero frequencies alongside amplitudes.

        Args:
            x: [..., T] 실수 시퀀스
            k: 선택할 빈도 수

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - [..., k] 형태의 주파수 인덱스
                - [..., k] 형태의 주파수 진폭
        """
        T = x.size(-1)
        spec = torch.fft.rfft(x, dim=-1)
        amp = torch.abs(spec)
        if amp.size(-1) > 0:
            amp[..., 0] = 0.0  # DC 성분 제외
        avail = amp.size(-1) - 1 if amp.size(-1) > 1 else 0
        k = min(k, avail) if avail > 0 else 0
        if k <= 0:
            idx_shape = list(x.shape[:-1]) + [0]
            empty_idx = torch.zeros(idx_shape, dtype=torch.long, device=x.device)
            empty_amp = torch.zeros(idx_shape, dtype=amp.dtype, device=x.device)
            return empty_idx, empty_amp
        values, indices = torch.topk(amp, k=k, dim=-1)
        indices = torch.clamp(indices, min=1)
        return indices, values

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorised period folding.

        Args:
            x: [B, T, N]

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                folded tensor [B, N, K, C_max, P_max]
                mask tensor   [B, N, K, C_max, P_max]
                amplitude      [B, N, K]
        """
        B, T, N = x.shape
        # Flatten batch & channel for joint processing
        seqs = x.permute(0, 2, 1).reshape(B * N, T)  # [BN, T]

        if self.k <= 0:
            empty = x.new_zeros(B, N, 0, 1, self.pmax)
            empty_amp = x.new_zeros(B, N, 0)
            return empty, empty, empty_amp

        kidx, kamp = self._topk_freq(seqs, self.k)  # [BN, K]
        K = kidx.size(1)
        if K == 0 or kidx.numel() == 0:
            Kpad = max(int(self.k), 0)
            empty = x.new_zeros(B, N, Kpad, 1, self.pmax)
            empty_amp = x.new_zeros(B, N, Kpad)
            return empty, empty, empty_amp

        # Compute period lengths and cycles
        P = T // torch.clamp(kidx, min=1)
        P = torch.clamp(
            P, min=self.min_period_threshold, max=self.pmax
        )  # [BN, K]
        cycles = torch.clamp(T // P, min=1)  # [BN, K]
        take = cycles * P

        Pmax = self.pmax
        # Keep ``Cmax`` as a tensor to avoid ``.item()`` which breaks GPU capture
        Cmax_t = torch.clamp(cycles.max(), min=1)
        idx_c = torch.arange(Cmax_t, device=x.device)
        idx_p = torch.arange(Pmax, device=x.device)

        BN = B * N
        base = torch.clamp(T - take, min=0)[..., None, None]
        P_exp = P[..., None, None]
        indices = base + idx_c.view(1, 1, -1, 1) * P_exp + idx_p.view(1, 1, 1, -1)
        indices = indices.clamp(min=0, max=T - 1)

        seqs_exp = (
            seqs.unsqueeze(1).unsqueeze(2).expand(BN, K, idx_c.size(0), -1)
        )
        gathered = torch.gather(seqs_exp, dim=-1, index=indices)

        mask_c = idx_c.view(1, 1, -1, 1) < cycles[..., None, None]
        mask_p = idx_p.view(1, 1, 1, -1) < P[..., None, None]
        mask = (mask_c & mask_p).to(gathered.dtype)
        gathered = gathered * mask

        Cmax = gathered.size(2)
        gathered = gathered.reshape(B, N, K, Cmax, Pmax)
        flat_mask = mask.reshape(B, N, K, Cmax, Pmax)
        kamp = kamp.reshape(B, N, K)
        if K < self.k:
            pad_shape = (B, N, self.k - K, Cmax, Pmax)
            gathered_pad = gathered.new_zeros(pad_shape)
            mask_pad = flat_mask.new_zeros(pad_shape)
            gathered = torch.cat([gathered, gathered_pad], dim=2)
            flat_mask = torch.cat([flat_mask, mask_pad], dim=2)
            amp_pad = kamp.new_zeros(B, N, self.k - K)
            kamp = torch.cat([kamp, amp_pad], dim=2)
        return gathered, flat_mask, kamp


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
        self.period_transform = PeriodicityTransform(
            k_periods=self.period_channels,
            pmax=int(pmax),
            min_period_threshold=int(min_period_threshold),
        )

    def forward(
        self,
        features: torch.Tensor,
        step_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fold the 1D feature map and compute a residual update."""

        B_eff, channel_count, seq_len = features.shape
        device = features.device
        dtype = features.dtype
        self.period_transform = self.period_transform.to(device=device, dtype=dtype)
        if step_mask is not None:
            features_in = features * step_mask.to(dtype=dtype)
        else:
            features_in = features

        seq = features_in.reshape(B_eff * channel_count, seq_len)
        seq = seq.unsqueeze(-1)
        folded, mask, freq_amp = self.period_transform(seq)
        if folded.numel() == 0:
            zero_update = torch.zeros_like(features)
            if step_mask is not None:
                new_mask = step_mask.to(dtype=dtype)
            else:
                new_mask = torch.zeros(B_eff, 1, seq_len, device=device, dtype=dtype)
            return zero_update, new_mask

        K = folded.size(2)
        cycles = folded.size(3)
        periods = folded.size(4)
        folded = folded.reshape(B_eff, channel_count, K, cycles, periods)
        mask = mask.reshape(B_eff, channel_count, K, cycles, periods).to(dtype=folded.dtype)
        freq_amp = freq_amp.reshape(B_eff, channel_count, K)

        conv_in = folded * mask
        conv_in_flat = conv_in.reshape(B_eff * channel_count, K, cycles, periods)
        z2d_flat = self.inception(conv_in_flat)
        z2d = z2d_flat.reshape(B_eff, channel_count, -1, cycles, periods).squeeze(2)

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
        if freq_amp is not None and freq_amp.numel() > 0:
            amp = freq_amp.to(dtype=branch_recon.dtype)
            invalid_fill = torch.finfo(branch_recon.dtype).min
            amp = torch.where(
                branch_valid,
                amp,
                torch.full_like(amp, invalid_fill),
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
    입력 [B, T, N] -> PeriodicityTransform -> [B, N, K, C, P]
    2D InceptionBlock이 [채널(K), 주기(C), 길이(P)] 격자에서 패턴을 추출하고
    주기 축 풀링을 통해 1D 시퀀스로 환원한 뒤 residual 스택을 구성합니다.
    전역 풀링 후 선형 헤드가 pred_len 생성 (direct) 또는 1스텝 (recursive).
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
        self.period = PeriodicityTransform(
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
                pmax=self.period.pmax,
                min_period_threshold=self.period.min_period_threshold,
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
        z_all, mask_all, _ = self.period(x)
        BN = B * N
        K = z_all.size(2)
        if K == 0:
            out_steps = self._out_steps
            mu = x.new_zeros(B, out_steps, N)
            if sigma_floor is not None:
                sigma = sigma_floor.to(dtype=x.dtype, device=x.device).expand(B, out_steps, -1).clone()
            else:
                sigma = x.new_full((B, out_steps, N), self.min_sigma)
            return mu, sigma
        folded = z_all.reshape(BN, K, z_all.size(3), z_all.size(4))
        mask = mask_all.reshape(BN, K, mask_all.size(3), mask_all.size(4))
        mask = mask.to(dtype=folded.dtype)
        cycle_weight = mask.sum(dim=2)
        eps = torch.finfo(folded.dtype).eps
        aggregated = (folded * mask).sum(dim=2) / (cycle_weight + eps)
        aggregated = aggregated * (cycle_weight > 0).to(dtype=aggregated.dtype)
        step_mask = (cycle_weight > 0).any(dim=1, keepdim=True).to(dtype=aggregated.dtype)
        if not self._lazy_built:
            self._build_lazy(x=aggregated)
        features = self.input_proj(aggregated)
        features = features * step_mask
        for blk in self.blocks:
            if self.use_checkpoint:
                delta, step_mask = checkpoint(
                    lambda inp, mask_in: blk(inp, mask_in),
                    features,
                    step_mask,
                    use_reentrant=False,
                )
            else:
                delta, step_mask = blk(features, step_mask)
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
