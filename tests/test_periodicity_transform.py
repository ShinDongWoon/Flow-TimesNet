import math
from pathlib import Path
import sys
from typing import Tuple

import torch

# Ensure the project src is on the path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.models.timesnet import PeriodicityTransform


def _periodicity_transform_naive(
    x: torch.Tensor, k: int, pmax: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation using explicit loops for verification."""
    B, T, N = x.shape
    if k <= 0:
        empty = x.new_zeros(B, N, 0, 1, pmax)
        return empty, empty

    seqs = x.permute(0, 2, 1).reshape(B * N, T)
    kidx = PeriodicityTransform._topk_freq(seqs, k)
    K = kidx.size(1)
    if K == 0 or kidx.numel() == 0:
        empty = x.new_zeros(B, N, max(k, 0), 1, pmax)
        return empty, empty

    kidx = kidx.view(B, N, K)
    P = torch.clamp(T // torch.clamp(kidx, min=1), min=1)
    P = torch.clamp(P, min=1, max=pmax)
    cycles = torch.clamp(T // P, min=1)
    Cmax = int(torch.clamp(cycles.max(), min=1).item())

    folded = x.new_zeros(B, N, K, Cmax, pmax)
    mask = torch.zeros_like(folded)
    for b in range(B):
        for n in range(N):
            for ki in range(K):
                period = int(P[b, n, ki].item())
                cycle = int(cycles[b, n, ki].item())
                if period <= 0 or cycle <= 0:
                    continue
                take = cycle * period
                base = max(T - take, 0)
                for c in range(Cmax):
                    for p in range(pmax):
                        if c < cycle and p < period:
                            idx = base + c * period + p
                            idx = min(max(idx, 0), T - 1)
                            folded[b, n, ki, c, p] = x[b, idx, n]
                            mask[b, n, ki, c, p] = 1.0
    if K < k:
        pad_shape = (B, N, k - K, Cmax, pmax)
        folded = torch.cat([folded, folded.new_zeros(pad_shape)], dim=2)
        mask = torch.cat([mask, mask.new_zeros(pad_shape)], dim=2)
    return folded, mask


def test_periodicity_transform_period_length_consistency():
    """PeriodicityTransform should output a shared period length across channels."""
    B, T, N = 2, 200, 3
    t = torch.arange(T, dtype=torch.float32)
    # Each channel has a different dominant frequency
    freqs = [2, 5, 8]
    x = torch.stack([torch.sin(2 * math.pi * f * t / T) for f in freqs], dim=-1)
    x = x.unsqueeze(0).repeat(B, 1, 1)
    # small noise to make signal more realistic
    x += 0.01 * torch.randn_like(x)

    k = 1
    transform = PeriodicityTransform(k, pmax=T)
    folded, mask = transform(x)

    assert folded.shape == mask.shape
    assert folded.shape[:2] == (B, N)
    assert folded.shape[2] == k
    assert folded.shape[-1] == T
    assert torch.all((mask > 0).any(dim=(2, 3, 4)))


def test_periodicity_transform_matches_naive():
    """Vectorised implementation should match naive loop-based version."""
    torch.manual_seed(0)
    B, T, N, k = 3, 64, 4, 3
    x = torch.randn(B, T, N)
    pmax = T
    out_ref, mask_ref = _periodicity_transform_naive(x, k, pmax)
    transform = PeriodicityTransform(k, pmax=pmax)
    out_vec, mask_vec = transform(x)
    assert torch.allclose(out_vec, out_ref, atol=1e-6)
    assert torch.allclose(mask_vec, mask_ref, atol=1e-6)


def test_periodicity_transform_min_period_threshold_expands_period():
    class FixedFreqTransform(PeriodicityTransform):
        def _topk_freq(self, x: torch.Tensor, k: int) -> torch.Tensor:  # type: ignore[override]
            BN = x.shape[0]
            return torch.full((BN, k), 20, dtype=torch.long, device=x.device)

    x = torch.arange(1, 51, dtype=torch.float32).view(1, 50, 1)

    low = FixedFreqTransform(k_periods=1, pmax=10, min_period_threshold=1)
    high = FixedFreqTransform(k_periods=1, pmax=10, min_period_threshold=6)

    out_low, mask_low = low(x)
    out_high, mask_high = high(x)

    # With the higher minimum period, more slots along P dimension remain populated.
    low_active = (mask_low > 0).any(dim=(2, 3)).squeeze(0).squeeze(0)
    high_active = (mask_high > 0).any(dim=(2, 3)).squeeze(0).squeeze(0)
    nz_low = torch.count_nonzero(low_active)
    nz_high = torch.count_nonzero(high_active)

    assert nz_low.item() == 2  # T // 20 = 2 before applying threshold
    assert nz_high.item() == 6  # Clamped up to the requested minimum


def test_periodicity_transform_take_gt_T_with_compile():
    """Regression test for take > T with torch.compile."""
    torch.manual_seed(0)
    B, T, N = 1, 5, 1
    x = torch.randn(B, T, N)

    class DegenerateTransform(PeriodicityTransform):
        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
            B, T, N = x.shape
            seqs = x.permute(0, 2, 1).reshape(B * N, T)
            kidx = torch.ones(B * N, self.k, dtype=torch.long, device=x.device)
            K = kidx.size(1)
            # Force P larger than T so that take > T
            P = torch.full_like(kidx, self.pmax)
            cycles = torch.ones_like(kidx)
            take = cycles * P

            Pmax = self.pmax
            # Keep ``Cmax`` tensor-based to avoid `.item()` for GPU capture
            Cmax_t = torch.clamp(cycles.max(), min=1)
            idx_c = torch.arange(Cmax_t, device=x.device)
            idx_p = torch.arange(Pmax, device=x.device)

            base = torch.clamp(T - take, min=0)[..., None, None]
            P_exp = P[..., None, None]
            indices = base + idx_c.view(1, 1, -1, 1) * P_exp + idx_p.view(1, 1, 1, -1)
            indices = indices.clamp(min=0, max=T - 1)

            seqs_exp = (
                seqs.unsqueeze(1).unsqueeze(2).expand(-1, K, idx_c.size(0), -1)
            )
            gathered = torch.gather(seqs_exp, dim=-1, index=indices)

            mask_c = idx_c.view(1, 1, -1, 1) < cycles[..., None, None]
            mask_p = idx_p.view(1, 1, 1, -1) < P[..., None, None]
            mask = (mask_c & mask_p).to(gathered.dtype)
            gathered = gathered * mask

            gathered = gathered.view(B, N, K, gathered.size(2), Pmax)
            flat_mask = mask.view(B, N, K, mask.size(2), Pmax)
            if K < self.k:
                pad_shape = (B, N, self.k - K, gathered.size(3), Pmax)
                gathered = torch.cat([gathered, gathered.new_zeros(pad_shape)], dim=2)
                flat_mask = torch.cat([flat_mask, flat_mask.new_zeros(pad_shape)], dim=2)
            return gathered, flat_mask

    transform = DegenerateTransform(k_periods=1, pmax=T + 2)
    compiled = torch.compile(transform)
    folded, mask = compiled(x)
    assert folded.shape == mask.shape
    assert folded.shape[-1] == T + 2

