from __future__ import annotations

from typing import List, Tuple
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
        # We don't know the period-length ``P`` at build time, so layers are built lazily
        # during the first forward pass once the flattened length ``K*P`` is known.
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

    def _build_lazy(self, x: torch.Tensor) -> None:
        """Instantiate convolutional blocks on first use.

        Args:
            x: reference tensor for device/dtype placement
        """
        if self.channels_last:
            first = nn.Conv2d(self.k, self.d_model, kernel_size=(1, 1)).to(
                device=x.device, dtype=x.dtype
            )
        else:
            first = nn.Conv1d(self.k, self.d_model, kernel_size=1).to(
                device=x.device, dtype=x.dtype
            )
        blocks = [first]
        for _ in range(self.n_layers):
            blocks.append(
                InceptionBlock(
                    self.d_model,
                    self.d_model,
                    self.kernel_set,
                    self.dropout,
                    self.act,
                    channels_last=self.channels_last,
                ).to(device=x.device, dtype=x.dtype)
            )
        self.blocks = nn.ModuleList(blocks)
        # Pool only over the temporal dimension; series dimension is preserved.
        self.pool = nn.AdaptiveAvgPool1d(1)
        out_steps = self._out_steps
        # Linear head operates independently for each series on the feature dimension.
        self.head = nn.Linear(self.d_model, out_steps * 2).to(device=x.device, dtype=x.dtype)
        self._lazy_built = True

    def _resize_frontend(self, new_in_channels: int, device: torch.device, dtype: torch.dtype) -> None:
        """Expand the input projection if additional cycles appear.

        Args:
            new_in_channels: desired number of input channels
            device: target device for the resized layer
            dtype: target dtype for the resized layer
        """
        if new_in_channels == self.k:
            return
        if self.channels_last:
            old = self.blocks[0]
            conv = nn.Conv2d(new_in_channels, self.d_model, kernel_size=(1, 1)).to(
                device=device, dtype=dtype
            )
            with torch.no_grad():
                copy = min(old.weight.size(1), conv.weight.size(1))
                conv.weight[:, :copy].copy_(old.weight[:, :copy])
                if old.bias is not None and conv.bias is not None:
                    conv.bias.copy_(old.bias)
        else:
            old = self.blocks[0]
            conv = nn.Conv1d(new_in_channels, self.d_model, kernel_size=1).to(
                device=device, dtype=dtype
            )
            with torch.no_grad():
                copy = min(old.weight.size(1), conv.weight.size(1))
                conv.weight[:, :copy].copy_(old.weight[:, :copy])
                if old.bias is not None and conv.bias is not None:
                    conv.bias.copy_(old.bias)
        self.blocks[0] = conv
        self.k = new_in_channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T, N]
        returns:
          Tuple(mu, sigma) where each tensor has shape:
            direct:    [B, pred_len, N]
            recursive: [B, 1, N]
        """
        B, T, N = x.shape
        z_all, mask_all = self.period(x)
        if z_all.size(2) == 0:
            out_steps = self._out_steps
            mu = x.new_zeros(B, out_steps, N)
            sigma = x.new_full((B, out_steps, N), self.min_sigma)
            return mu, sigma

        steps = min(self.input_len, z_all.size(-1))
        z = z_all[..., :steps]
        mask = mask_all[..., :steps]
        KC = z.size(2)
        z = z.reshape(B * N, KC, steps)
        mask = mask.reshape(B * N, KC, steps)
        if not self._lazy_built:
            self.k = KC
            self._build_lazy(x=z)
        else:
            if KC > self.k:
                self._resize_frontend(KC, device=z.device, dtype=z.dtype)
            elif KC < self.k:
                pad = self.k - KC
                if pad > 0:
                    pad_shape = (z.size(0), pad, steps)
                    z = torch.cat([z, z.new_zeros(pad_shape)], dim=1)
                    mask = torch.cat([mask, mask.new_zeros(pad_shape)], dim=1)
                    KC = self.k
        z = z * mask
        step_mask = mask.amax(dim=1, keepdim=True).to(dtype=z.dtype)
        if self.channels_last:
            z = z.unsqueeze(-1).contiguous(memory_format=torch.channels_last)
            step_mask = step_mask.unsqueeze(-1)
        for blk in self.blocks:
            z = checkpoint(blk, z, use_reentrant=False) if self.use_checkpoint else blk(z)
        if self.channels_last:
            z = z.squeeze(-1)
            step_mask = step_mask.squeeze(-1)
        z = z * step_mask
        z = self.pool(z).squeeze(-1)
        z = z.view(B, N, self.d_model)
        y_all = self.head(z)
        out_steps = self._out_steps
        params = y_all.view(B, N, out_steps, 2)
        mu = params[..., 0].permute(0, 2, 1).contiguous()
        log_sigma = params[..., 1]
        sigma = F.softplus(log_sigma).permute(0, 2, 1).contiguous() + self.min_sigma
        return mu, sigma
