from __future__ import annotations

import math
from typing import List, Tuple, Sequence
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn import init


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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorised period folding.

        Args:
            x: [B, T, N]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                folded tensor [B, N, K * C_max, P_max]
                mask tensor   [B, N, K * C_max, P_max]
        """
        B, T, N = x.shape
        # Flatten batch & channel for joint processing
        seqs = x.permute(0, 2, 1).reshape(B * N, T)  # [BN, T]

        if self.k <= 0:
            empty = x.new_zeros(B, N, 0, self.pmax)
            return empty, empty

        kidx = self._topk_freq(seqs, self.k)  # [BN, K]
        K = kidx.size(1)
        if K == 0 or kidx.numel() == 0:
            empty = x.new_zeros(B, N, 0, self.pmax)
            return empty, empty

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

        gathered = gathered.reshape(B, N, K * gathered.size(2), Pmax)
        flat_mask = mask.reshape(B, N, K * mask.size(2), Pmax)
        return gathered, flat_mask


class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_set: List[int],
        dropout: float,
        act: str,
        channels_last: bool = False,
    ) -> None:
        super().__init__()
        self.channels_last = bool(channels_last)
        self.paths = nn.ModuleList()
        for k in kernel_set:
            pad = (k - 1) // 2 if self.channels_last else k // 2
            if self.channels_last:
                self.paths.append(
                    nn.Conv2d(in_ch, out_ch, kernel_size=(k, 1), padding=(pad, 0))
                )
            else:
                self.paths.append(nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad))
        if self.channels_last:
            self.proj = nn.Conv2d(out_ch * len(kernel_set), out_ch, kernel_size=1)
        else:
            self.proj = nn.Conv1d(out_ch * len(kernel_set), out_ch, kernel_size=1)
        # Residual projection if channel dims differ
        if in_ch != out_ch:
            if self.channels_last:
                self.res_proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)
            else:
                self.res_proj = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        else:
            self.res_proj = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        if act.lower() == "gelu":
            self.act = nn.GELU()
        elif act.lower() == "relu":
            self.act = nn.ReLU()
        else:
            self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inception-style convolutions.

        Args:
            x: [B, C, L] when ``channels_last`` is ``False``
               or [B, C, L, 1] when ``channels_last`` is ``True``
        """
        res = self.res_proj(x)
        feats = [p(x) for p in self.paths]
        z = torch.cat(feats, dim=1)
        z = self.proj(z)
        z = self.act(z)
        z = self.dropout(z)
        return z + res


class TimesBlock(nn.Module):
    """TimesNet block that consumes folded 2D representations and emits 1D updates."""

    def __init__(
        self,
        d_model: int,
        kernel_set: Sequence[int],
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.kernel_set = list(kernel_set)
        self.dropout_p = float(dropout)
        act_name = activation.lower()
        if act_name == "relu":
            self.act_layer = nn.ReLU()
        else:
            self.act_layer = nn.GELU()
        self._built = False

    def _build(self, device: torch.device, dtype: torch.dtype) -> None:
        self.paths = nn.ModuleList()
        for k in self.kernel_set:
            pad_h = max((k - 1) // 2, 0)
            conv = nn.Conv2d(
                1,
                self.d_model,
                kernel_size=(k, 1),
                padding=(pad_h, 0),
            ).to(device=device, dtype=dtype)
            self.paths.append(conv)
        proj_in = self.d_model * (len(self.kernel_set) + 1)
        self.proj = nn.Conv1d(proj_in, self.d_model, kernel_size=1).to(
            device=device, dtype=dtype
        )
        self.inner_dropout = nn.Dropout(self.dropout_p)
        self._built = True

    def forward(
        self,
        features: torch.Tensor,
        folded: torch.Tensor,
        mask: torch.Tensor,
        step_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project folded 2D inputs into a 1D residual update."""

        if not self._built:
            self._build(device=features.device, dtype=features.dtype)
        base = folded.unsqueeze(1)  # [B, 1, KC, L]
        mask = mask.to(dtype=folded.dtype)
        mask2d = mask.unsqueeze(1)
        feats = []
        for path in self.paths:
            feats.append(path(base))  # [B, d_model, KC, L]
        if feats:
            z_paths = torch.cat(feats, dim=1)
            mask_exp = mask2d.expand(-1, z_paths.size(1), -1, -1)
            z_paths = z_paths * mask_exp
            eps = torch.finfo(z_paths.dtype).eps
            denom = mask.sum(dim=1, keepdim=True)  # [B, 1, L]
            z_paths = z_paths.sum(dim=2) / (denom + eps)
        else:
            steps = folded.size(-1)
            z_paths = folded.new_zeros(folded.size(0), 0, steps)
        combined = torch.cat([z_paths, features], dim=1)
        z = self.proj(combined)
        z = self.act_layer(z)
        z = self.inner_dropout(z)
        if step_mask is not None:
            z = z * step_mask.to(dtype=z.dtype)
        return z


class TimesNet(nn.Module):
    """
    입력 [B, T, N] -> PeriodicityTransform -> [B, K, P, N]
    채널/주기 축을 conv1d가 처리하기 쉽게 [B, N, K*P]로 변환 후 InceptionBlock 스택.
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
        self.kernel_set = kernel_set
        self.dropout = float(dropout)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.channels_last = bool(channels_last)
        self.use_checkpoint = bool(use_checkpoint)
        self.min_sigma = float(min_sigma)
        self.front_channels = 0
        self.register_buffer("min_sigma_vector", None)
        if min_sigma_vector is not None:
            min_sigma_tensor = torch.as_tensor(min_sigma_vector, dtype=torch.float32)
            self.min_sigma_vector = min_sigma_tensor.reshape(1, 1, -1)

    def _build_lazy(self, x: torch.Tensor) -> None:
        """Instantiate convolutional blocks on first use.

        Args:
            x: reference tensor for device/dtype placement
        """
        self.input_proj = nn.Conv1d(self.front_channels, self.d_model, kernel_size=1).to(
            device=x.device, dtype=x.dtype
        )
        blocks = []
        for _ in range(self.n_layers):
            block = TimesBlock(
                d_model=self.d_model,
                kernel_set=self.kernel_set,
                dropout=self.dropout,
                activation=self.act,
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

    def _resize_frontend(
        self, new_in_channels: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Expand the input projection if additional cycles appear.

        Args:
            new_in_channels: desired number of input channels
            device: target device for the resized layer
            dtype: target dtype for the resized layer
        """
        if new_in_channels <= self.front_channels:
            return
        conv = self.input_proj
        old_weight = conv.weight.data
        old_in = old_weight.size(1)
        if new_in_channels <= old_in:
            self.front_channels = new_in_channels
            return
        new_shape = (old_weight.size(0), new_in_channels, *old_weight.shape[2:])
        with torch.no_grad():
            new_weight = torch.zeros(new_shape, device=device, dtype=dtype)
            new_weight[:, :old_in, ...] = old_weight
            if new_in_channels > old_in:
                init.kaiming_uniform_(new_weight[:, old_in:, ...], a=math.sqrt(5))
            conv.weight.data = new_weight
        conv.in_channels = new_in_channels
        self.front_channels = new_in_channels

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
        z_all, mask_all = self.period(x)
        BN = B * N
        KC = z_all.size(2)
        z = z_all.reshape(BN, KC, z_all.size(-1))
        mask = mask_all.reshape(BN, KC, mask_all.size(-1))
        pooled_mask = F.adaptive_avg_pool1d(mask, self.input_len)
        z = F.adaptive_avg_pool1d(z, self.input_len)
        eps = torch.finfo(z.dtype).eps
        z = z / (pooled_mask + eps)
        mask = (pooled_mask > 0).to(z.dtype)
        KC = z.size(1)
        steps = z.size(-1)
        if KC == 0:
            out_steps = self._out_steps
            mu = x.new_zeros(B, out_steps, N)
            if sigma_floor is not None:
                sigma = sigma_floor.to(dtype=x.dtype, device=x.device).expand(B, out_steps, -1).clone()
            else:
                sigma = x.new_full((B, out_steps, N), self.min_sigma)
            return mu, sigma
        if not self._lazy_built:
            self.front_channels = KC
            self._build_lazy(x=z)
        else:
            if KC > self.front_channels:
                self._resize_frontend(KC, device=z.device, dtype=z.dtype)
            elif KC < self.front_channels:
                pad = self.front_channels - KC
                if pad > 0:
                    pad_shape = (z.size(0), pad, steps)
                    z = torch.cat([z, z.new_zeros(pad_shape)], dim=1)
                    mask = torch.cat([mask, mask.new_zeros(pad_shape)], dim=1)
                    KC = self.front_channels
        z = z * mask
        folded = z
        step_mask = mask.amax(dim=1, keepdim=True).to(dtype=z.dtype)
        features = self.input_proj(z)
        features = features * step_mask
        for blk in self.blocks:
            if self.use_checkpoint:
                delta = checkpoint(
                    lambda inp, fold, m, s: blk(inp, fold, m, s),
                    features,
                    folded,
                    mask,
                    step_mask,
                    use_reentrant=False,
                )
            else:
                delta = blk(features, folded, mask, step_mask)
            delta = self.residual_dropout(delta)
            features = features + delta
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
