import sys
from pathlib import Path

import torch
from torch import nn


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.models.timesnet import TimesBlock


class FixedSelector(nn.Module):
    def __init__(self, periods, amplitudes) -> None:
        super().__init__()
        self._periods = torch.as_tensor(periods, dtype=torch.long)
        self._amplitudes = torch.as_tensor(amplitudes, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = x.size(0)
        device = x.device
        dtype = x.dtype
        periods = self._periods.to(device=device)
        amplitudes = self._amplitudes.to(device=device, dtype=dtype)
        if amplitudes.dim() == 1:
            amplitudes = amplitudes.unsqueeze(0)
        if amplitudes.size(0) == 1 and B > 1:
            amplitudes = amplitudes.expand(B, -1)
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


def test_vectorized_matches_loop(monkeypatch):
    torch.manual_seed(0)
    block = _build_block(d_model=4)
    selector = FixedSelector(periods=[3, 5], amplitudes=[[0.1, -0.4], [1.2, 0.7]])
    object.__setattr__(block, "period_selector", selector)

    x = torch.randn(2, 28, 4)

    monkeypatch.setenv("TIMESBLOCK_VEC_DISABLE", "1")
    block._period_calls = 0
    loop_out = block(x)
    monkeypatch.delenv("TIMESBLOCK_VEC_DISABLE", raising=False)

    block._period_calls = 0
    block._vec_calls = 0
    vec_out = block(x)

    assert block._vec_calls >= 1
    diff = (loop_out - vec_out).abs().max().item()
    assert diff < 1e-5


def test_kchunk_equivalence(monkeypatch):
    torch.manual_seed(1)
    block = _build_block(d_model=4)
    selector = FixedSelector(periods=[3, 4, 6], amplitudes=[[0.2, -0.3, 0.5]])
    object.__setattr__(block, "period_selector", selector)

    x = torch.randn(1, 30, 4)

    monkeypatch.setenv("TIMESBLOCK_K_CHUNK", "1")
    block._period_calls = 0
    block._vec_calls = 0
    chunked_out = block(x)
    monkeypatch.delenv("TIMESBLOCK_K_CHUNK", raising=False)

    block._period_calls = 0
    block._vec_calls = 0
    default_out = block(x)

    assert block._vec_calls >= 1
    max_diff = (chunked_out - default_out).abs().max().item()
    assert max_diff < 1e-5


def test_cycles_ge_2(monkeypatch):
    torch.manual_seed(2)
    block = _build_block(d_model=2)
    selector = FixedSelector(periods=[64, 4], amplitudes=[[3.0, -1.0]])
    object.__setattr__(block, "period_selector", selector)

    x = torch.randn(1, 16, 2)

    block._period_calls = 0
    block._vec_calls = 0
    out = block(x)

    assert out.shape == x.shape
    assert block._vec_calls >= 1

    monkeypatch.setenv("TIMESBLOCK_VEC_DISABLE", "1")
    block._period_calls = 0
    loop_out = block(x)
    monkeypatch.delenv("TIMESBLOCK_VEC_DISABLE", raising=False)

    assert torch.allclose(out, loop_out, atol=1e-5, rtol=1e-5)


def test_duplicate_periods_aggregate_once(monkeypatch):
    torch.manual_seed(3)
    block = _build_block(d_model=3)
    selector = FixedSelector(
        periods=[4, 4, 8, 4], amplitudes=[[0.5, -1.2, 0.3, 0.7]]
    )
    object.__setattr__(block, "period_selector", selector)

    x = torch.randn(2, 25, 3)

    monkeypatch.setenv("TIMESBLOCK_VEC_DISABLE", "1")
    loop_out = block(x)
    monkeypatch.delenv("TIMESBLOCK_VEC_DISABLE", raising=False)

    block._vec_calls = 0
    vec_out = block(x)

    assert block._vec_calls >= 1
    assert torch.allclose(loop_out, vec_out, atol=1e-5, rtol=1e-5)
