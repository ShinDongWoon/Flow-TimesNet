import sys
from pathlib import Path

import pytest
import torch

# Ensure the project src is on the path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.models.timesnet import TimesNet


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for AMP autocast tests"
)
def test_timesnet_amp_index_add_dtype_alignment(monkeypatch):
    device = torch.device("cuda")
    B, T, N = 2, 24, 4
    input_len = 16
    pred_len = 8

    model = TimesNet(
        input_len=input_len,
        pred_len=pred_len,
        d_model=8,
        n_layers=1,
        k_periods=2,
        pmax=T,
        kernel_set=[3],
        dropout=0.0,
        activation="gelu",
        mode="direct",
        use_checkpoint=False,
    ).to(device)

    # Build lazy modules on the correct device before running under autocast.
    with torch.no_grad():
        warmup = torch.randn(1, T, N, device=device)
        model(warmup)

    recorded_dtypes = []
    original_index_add_ = torch.Tensor.index_add_

    def _capture_index_add(self, dim, index, other, *args, **kwargs):
        recorded_dtypes.append((self.dtype, other.dtype))
        return original_index_add_(self, dim, index, other, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "index_add_", _capture_index_add)

    x = torch.randn(B, T, N, device=device, dtype=torch.float32)

    with torch.cuda.amp.autocast(device_type="cuda", dtype=torch.float16):
        mu, sigma = model(x)

    assert mu.shape == (B, pred_len, N)
    assert sigma.shape == (B, pred_len, N)
    assert recorded_dtypes, "index_add_ should have been invoked during aggregation"

    for accumulator_dtype, update_dtype in recorded_dtypes:
        assert accumulator_dtype == torch.float16
        assert update_dtype == torch.float16
