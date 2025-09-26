import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from timesnet_forecast.models.timesnet import TimesBlock


class DummySelector(torch.nn.Module):
    def __init__(self, periods: torch.Tensor, amplitudes: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("_periods", periods.to(dtype=torch.long))
        self.register_buffer("_amplitudes", amplitudes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        periods = self._periods.to(device=x.device)
        amp = self._amplitudes.to(device=x.device, dtype=x.dtype)
        if amp.dim() == 2 and amp.size(0) != x.size(0):
            amp = amp.expand(x.size(0), -1)
        return periods, amp


def _make_block(d_model: int = 8) -> TimesBlock:
    block = TimesBlock(d_model=d_model, kernel_set=[3], dropout=0.0, activation="gelu")
    return block


def test_bucketed_equals_slow() -> None:
    torch.manual_seed(0)
    B, L, C = 2, 32, 8
    x = torch.randn(B, L, C)
    periods = torch.tensor([7, 9, 14], dtype=torch.long)
    amplitudes = torch.randn(B, periods.numel(), dtype=x.dtype)
    block = _make_block(C)
    block.period_selector = DummySelector(periods, amplitudes)

    periods_dev = periods.to(x.device)
    amplitudes_dev = amplitudes.to(x.device)

    slow = block._period_conv_loop(x, periods_dev, amplitudes_dev)
    fast = block._period_conv_bucketed_slicing(x, periods_dev, amplitudes_dev)
    assert slow is not None
    assert fast is not None
    torch.testing.assert_close(fast, slow, rtol=1e-5, atol=1e-5)


def test_mixed_padding_shapes() -> None:
    torch.manual_seed(1)
    B, L, C = 1, 37, 4
    x = torch.randn(B, L, C)
    periods = torch.tensor([4, 5, 6, 11, 13], dtype=torch.long)
    amplitudes = torch.randn(1, periods.numel(), dtype=x.dtype)
    block = _make_block(C)
    block.period_selector = DummySelector(periods, amplitudes)

    out = block._period_conv_bucketed_slicing(x, periods, amplitudes)
    assert out is not None
    assert out.shape == x.shape


def test_no_mask_in_vectorized_path() -> None:
    module_path = Path(__file__).resolve().parents[1] / "src" / "timesnet_forecast" / "models" / "timesnet.py"
    source = module_path.read_text()
    marker = "def _period_conv_bucketed_slicing"
    start = source.find(marker)
    assert start != -1
    remainder = source[start:]
    end = remainder.find("def ", len(marker))
    if end != -1:
        snippet = remainder[:end]
    else:
        snippet = remainder
    forbidden = ["mask_batch", "* mask", ".mul(mask"]
    for token in forbidden:
        assert token not in snippet
