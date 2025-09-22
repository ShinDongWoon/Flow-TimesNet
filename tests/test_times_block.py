import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure the project src is on the path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.models.timesnet import FFTPeriodSelector, TimesBlock


class StaticSelector(nn.Module):
    def __init__(self, periods, amplitudes) -> None:
        super().__init__()
        self.periods = torch.as_tensor(periods, dtype=torch.long)
        self.amplitudes = torch.as_tensor(amplitudes, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = x.size(0)
        device = x.device
        dtype = x.dtype
        periods = self.periods.to(device=device)
        amps = self.amplitudes.to(device=device, dtype=dtype)
        if amps.dim() == 1:
            amps = amps.unsqueeze(0)
        if amps.size(0) == 1 and B > 1:
            amps = amps.expand(B, -1)
        return periods, amps


class AddPeriodScale(nn.Module):
    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        period = grid.size(-1)
        return grid + period * torch.ones_like(grid)


def _reference_times_block(
    x: torch.Tensor,
    block: TimesBlock,
    periods: torch.Tensor,
    amplitudes: torch.Tensor,
) -> torch.Tensor:
    if periods.numel() == 0:
        return x

    B, L, C = x.shape
    x_perm = x.permute(0, 2, 1).contiguous()
    residuals = []
    valid_indices = []

    for idx in range(periods.numel()):
        period = int(periods[idx].item())
        if period <= 0:
            continue
        pad_len = (-L) % period
        if pad_len > 0:
            x_pad = F.pad(x_perm, (0, pad_len))
        else:
            x_pad = x_perm
        total_len = x_pad.size(-1)
        cycles = total_len // period
        grid = x_pad.view(B, C, cycles, period)
        conv_out = block.inception(grid)
        delta = conv_out - grid
        delta = delta.view(B, C, total_len).permute(0, 2, 1)
        if pad_len > 0:
            delta = delta[:, :-pad_len, :]
        residuals.append(delta)
        valid_indices.append(idx)

    if not residuals:
        return x

    stacked = torch.stack(residuals, dim=-1)
    amps = amplitudes[:, valid_indices]
    weights = F.softmax(amps, dim=1)
    weights = weights.view(B, 1, 1, -1)
    combined = (stacked * weights).sum(dim=-1)
    return x + combined


def test_times_block_preserves_shape():
    torch.manual_seed(0)
    block = TimesBlock(
        d_model=4,
        kernel_set=[(3, 3)],
        dropout=0.0,
        activation="gelu",
    )
    selector = StaticSelector(periods=[2, 4], amplitudes=[[1.0, 0.5]])
    object.__setattr__(block, "period_selector", selector)

    x = torch.randn(3, 10, 4)
    out = block(x)
    assert out.shape == x.shape
    assert (out - x).shape == x.shape


def test_times_block_softmax_weighting():
    block = TimesBlock(
        d_model=1,
        kernel_set=[(3, 3)],
        dropout=0.0,
        activation="gelu",
    )
    block.inception = AddPeriodScale()
    selector = StaticSelector(periods=[2, 4], amplitudes=[2.0, 0.0])
    object.__setattr__(block, "period_selector", selector)

    x = torch.zeros(1, 8, 1)
    out = block(x)

    logits = torch.tensor([[2.0, 0.0]])
    weights = torch.softmax(logits, dim=1)
    expected_value = (weights * torch.tensor([[2.0, 4.0]])).sum(dim=1)
    expected = expected_value.view(1, 1, 1).expand_as(out)
    residual = out - x
    assert torch.allclose(residual, expected)


def test_times_block_returns_input_when_no_valid_periods():
    block = TimesBlock(
        d_model=2,
        kernel_set=[(3, 3)],
        dropout=0.0,
        activation="gelu",
    )
    selector = StaticSelector(periods=[0, -1], amplitudes=[1.0, 1.0])
    object.__setattr__(block, "period_selector", selector)

    x = torch.randn(2, 5, 2)
    out = block(x)
    assert torch.allclose(out, x)
    assert torch.allclose(out - x, torch.zeros_like(x))


def test_times_block_matches_reference_impl():
    torch.manual_seed(1)
    block = TimesBlock(
        d_model=2,
        kernel_set=[(3, 3)],
        dropout=0.0,
        activation="gelu",
    )
    selector = FFTPeriodSelector(k_periods=2, pmax=8)
    object.__setattr__(block, "period_selector", selector)

    x = torch.randn(1, 8, 2)
    periods, amplitudes = block.period_selector(x)
    expected = _reference_times_block(x, block, periods, amplitudes)
    out = block(x)
    assert torch.allclose(out, expected, atol=1e-6)
