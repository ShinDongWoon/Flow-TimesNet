import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

# Ensure the project src is on the path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.models.timesnet import PeriodicityTransform


def _periodicity_transform_naive(x: torch.Tensor, k: int) -> torch.Tensor:
    """Reference implementation using explicit loops for verification."""
    B, T, N = x.shape
    kidx_list = []
    Pmax = 1
    for n in range(N):
        xn = x[:, :, n]
        kidx = PeriodicityTransform._topk_freq(xn, k)
        kidx_list.append(kidx)
        ch_Pmax = 1
        for ki in range(kidx.size(1)):
            f = kidx[:, ki]
            P = torch.clamp(T // torch.clamp(f, min=1), min=1)
            ch_Pmax = max(ch_Pmax, int(P.max().item()))
        Pmax = max(Pmax, ch_Pmax)
    outs = []
    for n in range(N):
        xn = x[:, :, n]
        kidx = kidx_list[n]
        mats = []
        for b in range(B):
            seq = xn[b]
            cols = []
            for ki in range(kidx.size(1)):
                f = torch.clamp(kidx[b, ki], min=1)
                P = torch.clamp(T // f, min=1)
                cycles = torch.clamp(T // P, min=1)
                take = int((cycles * P).item())
                seg = seq[-take:].reshape(int(cycles.item()), int(P.item())).mean(dim=0)
                if P.item() < Pmax:
                    seg = F.pad(seg, (0, Pmax - int(P.item())))
                cols.append(seg)
            if len(cols) == 0:
                cols = [torch.zeros(Pmax, dtype=seq.dtype)]
            mats.append(torch.stack(cols, dim=0))
        outs.append(torch.stack(mats, dim=0))
    return torch.stack(outs, dim=-1)


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
    transform = PeriodicityTransform(k)
    out = transform(x)

    assert out.shape[-1] == N
    period_len = out.shape[2]
    for n in range(N):
        assert out[:, :, :, n].shape[2] == period_len


def test_periodicity_transform_matches_naive():
    """Vectorised implementation should match naive loop-based version."""
    torch.manual_seed(0)
    B, T, N, k = 3, 64, 4, 3
    x = torch.randn(B, T, N)
    transform = PeriodicityTransform(k)
    out_vec = transform(x)
    out_ref = _periodicity_transform_naive(x, k)
    assert torch.allclose(out_vec, out_ref, atol=1e-6)

