import sys
from pathlib import Path

import torch

# Ensure the project src is on the path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.models.timesnet import TimesNet


def test_forward_shape_and_tail_processing():
    B, L, H, N = 2, 16, 4, 3
    torch.manual_seed(0)

    model = TimesNet(
        input_len=L,
        pred_len=H,
        d_model=8,
        n_layers=1,
        k_periods=1,
        pmax=L,
        kernel_set=[3],
        dropout=0.0,
        activation="gelu",
        mode="direct",
    )
    with torch.no_grad():
        model(torch.zeros(1, L, N))  # build lazy layers

    x = torch.randn(B, L, N)
    model.train()
    out_train = model(x)
    model.eval()
    out_eval = model(x)
    assert out_train.shape == out_eval.shape == (B, H, N)

    long_x = torch.randn(B, L + 5, N)
    out_long = model(long_x)
    out_tail = model(long_x[:, -L:, :])
    assert out_long.shape == out_tail.shape == (B, H, N)
