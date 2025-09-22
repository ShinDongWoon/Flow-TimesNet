import math
from pathlib import Path
import sys

import torch

# Ensure the project src is on the path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.models.timesnet import FFTPeriodSelector


def test_fft_period_selector_shared_periods_and_ordering():
    torch.manual_seed(0)
    B, L, C = 2, 256, 3
    dominant_freqs = [4, 8]
    dominant_amps = [3.0, 1.5]
    t = torch.arange(L, dtype=torch.float32)

    batches = []
    for _ in range(B):
        cols = []
        for _ in range(C):
            signal = sum(
                amp * torch.sin(2 * math.pi * freq * t / L)
                for amp, freq in zip(dominant_amps, dominant_freqs)
            )
            signal += 0.01 * torch.randn_like(t)
            cols.append(signal)
        batches.append(torch.stack(cols, dim=1))
    x = torch.stack(batches, dim=0)  # [B, L, C]

    selector = FFTPeriodSelector(k_periods=2, pmax=L)
    periods, amplitudes = selector(x)

    expected_periods = [L // dominant_freqs[0], L // dominant_freqs[1]]
    assert periods.tolist() == expected_periods
    assert amplitudes.shape == (2,)
    assert amplitudes[0] >= amplitudes[1]


def test_fft_period_selector_respects_bounds():
    L = 64
    freq_low = 2   # raw period 32 -> clamp to pmax
    freq_high = 20  # raw period 3 -> clamp to min threshold
    t = torch.arange(L, dtype=torch.float32)
    signal = 2.0 * torch.sin(2 * math.pi * freq_low * t / L)
    signal += 1.0 * torch.sin(2 * math.pi * freq_high * t / L)
    x = signal.view(1, L, 1)

    selector = FFTPeriodSelector(k_periods=2, pmax=16, min_period_threshold=5)
    periods, amplitudes = selector(x)

    assert periods.tolist() == [16, 5]
    assert torch.all(amplitudes > 0)


def test_fft_period_selector_handles_zero_k():
    L = 32
    t = torch.arange(L, dtype=torch.float32)
    signal = torch.sin(2 * math.pi * 3 * t / L)
    x = signal.view(1, L, 1)

    selector = FFTPeriodSelector(k_periods=0, pmax=L)
    periods, amplitudes = selector(x)

    assert periods.numel() == 0
    assert amplitudes.numel() == 0
