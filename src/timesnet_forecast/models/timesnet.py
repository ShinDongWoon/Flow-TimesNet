from __future__ import annotations

from typing import List
import torch
from torch import nn
import torch.nn.functional as F


class PeriodicityTransform(nn.Module):
    """
    FFT 기반으로 상위 k 주파수를 골라 2D로 접는 근사 구현.
    아이디어:
      - 배치/채널을 한 번에 처리하여 rFFT 스펙트럼에서 에너지가 큰 순서대로 k개 빈도를 선택
      - 각 선택된 빈도 f에 대해 period_len = max(1, T // f)
      - 마지막 (cycles * period_len) 구간을 [cycles, period_len]로 접어 평균(축=0)해 [period_len] 시퀀스를 얻음
      - 모든 채널에서 가장 긴 period_len을 찾아 전 채널이 공유하는 P_max까지 pad하여 [B, K, P_max, N] 텐서 구성
    """

    def __init__(self, k_periods: int, series_chunk: int = 0) -> None:
        super().__init__()
        self.k = int(k_periods)
        # Process at most ``series_chunk`` sequences at a time in ``forward`` to
        # avoid constructing large [BN, K, Cmax, Pmax] tensors.  A value of 0
        # disables chunking and preserves the previous fully-vectorised
        # behaviour.
        self.series_chunk = int(series_chunk)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Vectorised period folding.

        Args:
            x: [B, T, N]

        Returns:
            torch.Tensor: [B, K, P_max, N]
        """
        B, T, N = x.shape
        # Flatten batch & channel for joint processing
        seqs = x.permute(0, 2, 1).reshape(B * N, T)  # [BN, T]

        if self.k <= 0:
            return x.new_zeros(B, 0, 1, N)

        kidx = self._topk_freq(seqs, self.k)  # [BN, K]
        K = kidx.size(1)
        if K == 0 or kidx.numel() == 0:
            return x.new_zeros(B, 0, 1, N)

        # Compute period lengths and cycles
        P = torch.clamp(T // torch.clamp(kidx, min=1), min=1)  # [BN, K]
        cycles = torch.clamp(T // P, min=1)  # [BN, K]
        take = cycles * P

        Pmax = max(1, int(P.max().item()))
        Cmax = max(1, int(cycles.max().item()))

        idx_c = torch.arange(Cmax, device=x.device)
        idx_p = torch.arange(Pmax, device=x.device)

        BN = B * N
        chunk = self.series_chunk if self.series_chunk > 0 else BN
        seg_all = x.new_zeros(BN, K, Pmax)

        for s in range(0, BN, chunk):
            e = min(s + chunk, BN)
            seq_chunk = seqs[s:e]
            P_chunk = P[s:e]
            cycles_chunk = cycles[s:e]
            take_chunk = take[s:e]

            base = torch.clamp(T - take_chunk, min=0)[..., None, None]
            P_exp = P_chunk[..., None, None]
            indices = base + idx_c.view(1, 1, -1, 1) * P_exp + idx_p.view(1, 1, 1, -1)
            indices = indices.clamp(min=0, max=T - 1)

            seqs_exp = seq_chunk.unsqueeze(1).unsqueeze(2).expand(-1, K, Cmax, -1)
            gathered = torch.gather(seqs_exp, dim=-1, index=indices)

            mask_c = idx_c.view(1, 1, -1, 1) < cycles_chunk[..., None, None]
            mask_p = idx_p.view(1, 1, 1, -1) < P_chunk[..., None, None]
            mask = mask_c & mask_p
            gathered = gathered * mask

            seg = gathered.sum(dim=2) / torch.clamp(cycles_chunk[..., None], min=1)
            seg_all[s:e] = seg

        seg_all = seg_all.view(B, N, K, Pmax).permute(0, 2, 3, 1)  # [B, K, Pmax, N]
        return seg_all


class InceptionBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_set: List[int], dropout: float, act: str) -> None:
        super().__init__()
        self.paths = nn.ModuleList()
        for k in kernel_set:
            pad = k // 2
            self.paths.append(nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad))
        self.proj = nn.Conv1d(out_ch * len(kernel_set), out_ch, kernel_size=1)
        # Residual projection if channel dims differ
        if in_ch != out_ch:
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
        """
        x: [B, C, L]
        """
        res = self.res_proj(x)
        feats = [p(x) for p in self.paths]  # list of [B, out_ch, L]
        z = torch.cat(feats, dim=1)         # [B, out_ch*P, L]
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
        kernel_set: List[int],
        dropout: float,
        activation: str,
        mode: str,
        series_chunk: int = 128,
    ) -> None:
        super().__init__()
        assert mode in ("direct", "recursive")
        self.mode = mode
        self.pred_len = int(pred_len)
        # Share the ``series_chunk`` value with the periodicity transform so that
        # both stages limit memory usage when processing many series.
        self.period = PeriodicityTransform(k_periods=k_periods, series_chunk=series_chunk)
        self.k = int(k_periods)
        self.act = activation
        # Conv will process each series independently along the temporal dimension.
        # We don't know the period-length ``P`` at build time, so layers are built lazily
        # during the first forward pass once the flattened length ``K*P`` is known.
        self.stem: nn.Module = nn.Identity()
        self.blocks: nn.Module = nn.Identity()
        self.pool: nn.Module = nn.Identity()
        self.head: nn.Module = nn.Identity()
        self._lazy_built = False
        self.kernel_set = kernel_set
        self.dropout = float(dropout)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.series_chunk = int(series_chunk)

    def _build_lazy(self, x: torch.Tensor) -> None:
        """Instantiate convolutional blocks on first use.

        Args:
            x: reference tensor for device/dtype placement
        """
        # Stem that maps each series (treated as a separate "batch" item) from 1 → d_model channels.
        self.stem = nn.Conv1d(in_channels=1, out_channels=self.d_model, kernel_size=1).to(
            device=x.device, dtype=x.dtype
        )
        blocks = []
        for _ in range(self.n_layers):
            blocks.append(
                InceptionBlock(
                    self.d_model, self.d_model, self.kernel_set, self.dropout, self.act
                ).to(device=x.device, dtype=x.dtype)
            )
        self.blocks = nn.Sequential(*blocks)
        # Pool only over the temporal dimension; series dimension is preserved.
        self.pool = nn.AdaptiveAvgPool1d(1)
        out_steps = self.pred_len if self.mode == "direct" else 1
        # Linear head operates independently for each series on the feature dimension.
        self.head = nn.Linear(self.d_model, out_steps).to(device=x.device, dtype=x.dtype)
        self._lazy_built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, N]
        returns:
          direct:    [B, pred_len, N]
          recursive: [B, 1, N]
        """
        B, T, N = x.shape
        # Periodicity folding -> [B, K, P, N]
        z_all = self.period(x)
        _, K, P, _ = z_all.shape

        outs = []
        chunk = int(self.series_chunk) if self.series_chunk > 0 else N
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            z = z_all[..., s:e]  # [B, K, P, n_chunk]
            n = e - s
            z = z.permute(0, 3, 1, 2).contiguous().view(B * n, 1, K * P)
            if not self._lazy_built:
                self._build_lazy(x=z)
            z = self.stem(z)  # [B*n, d_model, K*P]
            z = self.blocks(z)  # [B*n, d_model, K*P]
            z = self.pool(z).squeeze(-1)  # [B*n, d_model]
            z = z.view(B, n, self.d_model)  # [B, n, d_model]
            y = self.head(z)  # [B, n, out_steps]
            outs.append(y)
        y_all = torch.cat(outs, dim=1)  # [B, N, out_steps]
        return y_all.permute(0, 2, 1)  # [B, out_steps, N]
