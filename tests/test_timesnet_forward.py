import sys
from pathlib import Path

import torch

# Ensure the project src is on the path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.models.timesnet import TimesNet


def test_forward_shape_and_head_processing():
    B, L, H, N = 2, 16, 4, 3
    torch.manual_seed(0)

    model = TimesNet(
        input_len=L,
        pred_len=H,
        d_model=8,
        n_layers=1,
        k_periods=1,
        pmax=L,
        kernel_set=[(3, 3)],
        dropout=0.0,
        activation="gelu",
        mode="direct",
    )
    with torch.no_grad():
        model(torch.zeros(1, L, N))  # build lazy layers

    x = torch.randn(B, L, N)
    model.train()
    mu_train, sigma_train = model(x)
    model.eval()
    mu_eval, sigma_eval = model(x)
    assert mu_train.shape == mu_eval.shape == (B, H, N)
    assert sigma_train.shape == sigma_eval.shape == (B, H, N)
    assert torch.all(sigma_eval > 0)

    long_x = torch.randn(B, L + 5, N)
    mu_long, sigma_long = model(long_x)
    mu_head, sigma_head = model(long_x[:, :L, :])
    assert mu_long.shape == mu_head.shape == (B, H, N)
    assert sigma_long.shape == sigma_head.shape == (B, H, N)


def test_timesnet_applies_per_series_floor():
    B, L, H, N = 1, 12, 3, 4
    floor = torch.tensor([0.05, 0.1, 0.2, 0.3], dtype=torch.float32)

    model = TimesNet(
        input_len=L,
        pred_len=H,
        d_model=8,
        n_layers=1,
        k_periods=1,
        pmax=L,
        kernel_set=[(3, 3)],
        dropout=0.0,
        activation="gelu",
        mode="direct",
        min_sigma_vector=floor,
    )
    with torch.no_grad():
        model(torch.zeros(1, L, N))

    x = torch.randn(B, L, N)
    _, sigma = model(x)
    expected_floor = floor.view(1, 1, N)
    assert torch.all(sigma >= expected_floor)
