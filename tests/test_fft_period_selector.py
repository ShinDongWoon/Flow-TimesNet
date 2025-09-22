import math
from pathlib import Path
import sys

import pytest
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
    assert amplitudes.shape == (B, 2)
    assert torch.all(amplitudes[:, 0] >= amplitudes[:, 1])


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
    assert amplitudes.shape == (1, 2)
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


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for AMP coverage"
)
def test_fft_period_selector_amp_non_power_of_two_sequence():
    torch.manual_seed(0)
    device = torch.device("cuda")
    B, L, C = 2, 150, 4  # L is intentionally non-power-of-two

    selector = FFTPeriodSelector(k_periods=3, pmax=L).to(device)

    x_ref = torch.randn(B, L, C, device=device, dtype=torch.float32, requires_grad=True)
    periods_ref, amplitudes_ref = selector(x_ref)
    loss_ref = amplitudes_ref.sum()
    loss_ref.backward()
    grad_ref = x_ref.grad.detach().clone()

    x_amp = x_ref.detach().clone().requires_grad_(True)
    with torch.cuda.amp.autocast():
        periods_amp, amplitudes_amp = selector(x_amp)
        loss_amp = amplitudes_amp.sum()

    assert periods_amp.dtype == torch.long
    assert amplitudes_amp.dtype == x_amp.dtype

    loss_amp.backward()

    assert torch.equal(periods_amp, periods_ref)
    assert torch.allclose(amplitudes_amp, amplitudes_ref, atol=1e-3, rtol=1e-3)
    assert x_amp.grad is not None
    assert torch.allclose(x_amp.grad, grad_ref, atol=1e-3, rtol=1e-3)
