import os
import sys
from pathlib import Path

import torch
import pytest
from torch import nn

# Ensure project src is on path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.models.timesnet import (
    FFTPeriodSelector,
    RMSNorm,
    TimesBlock,
    TimesNet,
    _apply_layer_norm,
    _apply_rms_norm,
)


class FixedSelector(nn.Module):
    def __init__(self, periods, amplitudes) -> None:
        super().__init__()
        self._periods = torch.as_tensor(periods, dtype=torch.long)
        self._amplitudes = torch.as_tensor(amplitudes, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        dtype = x.dtype
        periods = self._periods.to(device=device)
        amplitudes = self._amplitudes.to(device=device, dtype=dtype)
        if amplitudes.dim() == 1:
            amplitudes = amplitudes.unsqueeze(0)
        batch = x.size(0)
        if amplitudes.size(0) == 1 and batch > 1:
            amplitudes = amplitudes.expand(batch, -1)
        return periods, amplitudes


def _build_block(d_model: int = 4) -> TimesBlock:
    block = TimesBlock(
        d_model=d_model,
        kernel_set=[(3, 3)],
        dropout=0.0,
        activation="gelu",
    )
    block.eval()
    return block


def test_cpu_period_conv_env_equivalence(monkeypatch):
    torch.manual_seed(0)
    block = _build_block(d_model=3)
    selector = FixedSelector(periods=[3, 5], amplitudes=[[0.1, -0.2]])
    object.__setattr__(block, "period_selector", selector)

    x = torch.randn(2, 21, 3, dtype=torch.bfloat16)

    monkeypatch.delenv("TIMES_MP_CONV", raising=False)
    baseline = block(x)

    monkeypatch.setenv("TIMES_MP_CONV", "1")
    toggled = block(x)

    assert baseline.shape == toggled.shape == x.shape
    assert baseline.dtype == toggled.dtype == x.dtype
    torch.testing.assert_close(
        baseline.to(torch.float32), toggled.to(torch.float32), atol=1e-6, rtol=1e-4
    )


@pytest.mark.parametrize("env_flag", [None, "1"])
def test_cpu_precision_sensitive_ops(monkeypatch, env_flag):
    if env_flag is None:
        monkeypatch.delenv("TIMES_MP_CONV", raising=False)
    else:
        monkeypatch.setenv("TIMES_MP_CONV", env_flag)

    torch.manual_seed(1)

    selector = FFTPeriodSelector(k_periods=2, pmax=16)
    fft_input = torch.randn(2, 18, 3)
    periods, amplitudes = selector(fft_input)
    if amplitudes.numel() > 0:
        assert torch.isfinite(amplitudes).all()
        assert torch.all(amplitudes >= 0)

    layer_norm = nn.LayerNorm(4)
    ln_input = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    ln_out = _apply_layer_norm(layer_norm, ln_input)
    assert ln_out.dtype == ln_input.dtype
    assert torch.isfinite(ln_out.to(torch.float32)).all()

    rms_norm = RMSNorm(4)
    rms_input = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    rms_out = _apply_rms_norm(rms_norm, rms_input)
    assert rms_out.dtype == rms_input.dtype
    assert torch.isfinite(rms_out.to(torch.float32)).all()

    block = _build_block(d_model=2)
    selector = FixedSelector(periods=[3, 4], amplitudes=[[0.3, -0.5]])
    object.__setattr__(block, "period_selector", selector)
    residuals = [
        torch.full((1, 5, 2), 0.5, dtype=torch.float32),
        torch.full((1, 5, 2), 0.25, dtype=torch.float32),
    ]
    amplitudes = torch.tensor([[0.1, -0.2]], dtype=torch.bfloat16)
    combined = block._combine_period_residuals(
        residuals, amplitudes, [0, 1], batch_size=1, periods=torch.tensor([3, 4])
    )
    assert combined is not None
    assert torch.isfinite(combined.to(torch.float32)).all()
    assert torch.all(combined > 0)

    model = TimesNet(
        input_len=12,
        pred_len=4,
        d_model=8,
        d_ff=16,
        n_layers=1,
        k_periods=1,
        kernel_set=[(3, 3)],
        dropout=0.0,
        activation="gelu",
        mode="direct",
    )
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 12, 2))
    x = torch.randn(2, 12, 2)
    with torch.no_grad():
        rate, dispersion = model(x)
    assert rate.shape == (2, 4, 2)
    assert dispersion.shape == (2, 4, 2)
    assert torch.isfinite(rate).all()
    assert torch.isfinite(dispersion).all()
    assert torch.all(rate > 0)
    assert torch.all(dispersion > 0)
