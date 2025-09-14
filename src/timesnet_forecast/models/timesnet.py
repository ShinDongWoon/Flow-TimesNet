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

    def __init__(self, k_periods: int) -> None:
        super().__init__()
        self.k = int(k_periods)

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

        kidx = self._topk_freq(seqs, self.k)  # [BN, K]
        K = kidx.size(1)

        # Compute period lengths and cycles
        P = torch.clamp(T // torch.clamp(kidx, min=1), min=1)  # [BN, K]
        cycles = torch.clamp(T // P, min=1)  # [BN, K]
        take = cycles * P

        Pmax = int(P.max().item())
        Cmax = int(cycles.max().item())

        idx_c = torch.arange(Cmax, device=x.device)
        idx_p = torch.arange(Pmax, device=x.device)

        base = torch.clamp(T - take, min=0)[..., None, None]
        P_exp = P[..., None, None]
        indices = base + idx_c.view(1, 1, -1, 1) * P_exp + idx_p.view(1, 1, 1, -1)
        indices = indices.clamp(min=0, max=T - 1)

        # Gather segments for all sequences
        seqs_exp = seqs.unsqueeze(1).unsqueeze(2).expand(-1, K, Cmax, -1)
        gathered = torch.gather(seqs_exp, dim=-1, index=indices)

        mask_c = idx_c.view(1, 1, -1, 1) < cycles[..., None, None]
        mask_p = idx_p.view(1, 1, 1, -1) < P[..., None, None]
        mask = mask_c & mask_p
        gathered = gathered * mask

        seg = gathered.sum(dim=2) / torch.clamp(cycles[..., None], min=1)
        seg = seg.view(B, N, K, Pmax).permute(0, 2, 3, 1)  # [B, K, Pmax, N]
        return seg


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
    ) -> None:
        super().__init__()
        assert mode in ("direct", "recursive")
        self.mode = mode
        self.pred_len = int(pred_len)
        self.period = PeriodicityTransform(k_periods=k_periods)
        self.k = int(k_periods)
        self.incept: List[InceptionBlock] = []
        self.act = activation
        # Conv will see channels = N (series), length = K*P
        # We don't know P at build time; use lightweight identity modules that will
        # be replaced lazily with proper layers on the first forward pass.
        self.stem: nn.Module = nn.Identity()
        self.blocks: nn.Module = nn.Identity()
        self.pool: nn.Module = nn.Identity()
        self.head: nn.Module = nn.Identity()
        # We'll lazily init convs on first forward when N and P are known
        self._lazy_built = False
        self.kernel_set = kernel_set
        self.dropout = float(dropout)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)

    def _build_lazy(self, N: int, L: int, x: torch.Tensor) -> None:
        # N known (channels), L known (length=K*P)
        self.stem = nn.Conv1d(in_channels=N, out_channels=self.d_model, kernel_size=1).to(
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
        # Global pooling over length
        self.pool = nn.AdaptiveAvgPool1d(1)
        # Head: map d_model -> pred_len or 1 per channel N, but we pooled length only.
        # We'll predict for each series independently by a linear over features then expand to N.
        out_steps = self.pred_len if self.mode == "direct" else 1
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
        z = self.period(x)
        _, K, P, _ = z.shape
        # Rearrange to conv-friendly: [B, N, K*P]
        z = z.permute(0, 3, 1, 2).contiguous().view(B, N, K * P)
        if not self._lazy_built:
            self._build_lazy(N=N, L=K * P, x=z)
        # Shared conv along "length" axis (time-like)
        z = self.stem(z)            # [B, d_model, K*P]
        z = self.blocks(z)          # [B, d_model, K*P]
        z = self.pool(z).squeeze(-1)  # [B, d_model]
        y = self.head(z)            # [B, H]
        # expand to all N channels (shared head per series)
        y = y.unsqueeze(-1).expand(B, y.size(1), N)  # [B, H, N]
        return y
