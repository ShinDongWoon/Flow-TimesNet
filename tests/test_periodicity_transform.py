import math
from pathlib import Path
import sys

import torch

# Ensure the project src is on the path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.models.timesnet import PeriodicityTransform


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

