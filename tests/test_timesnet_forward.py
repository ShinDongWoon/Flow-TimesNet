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
        k_periods=3,
        pmax=L,
        kernel_set=[3, 5],
        dropout=0.0,
        activation="gelu",
        mode="direct",
    )
    with torch.no_grad():
        model(torch.randn(1, L, N))  # build lazy layers with non-zero data

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
        kernel_set=[3],
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


def test_timesnet_handles_zero_periods():
    B, L, H, N = 2, 10, 2, 2
    model = TimesNet(
        input_len=L,
        pred_len=H,
        d_model=4,
        n_layers=0,
        k_periods=0,
        pmax=L,
        kernel_set=[3],
        dropout=0.0,
        activation="gelu",
        mode="direct",
        min_sigma=0.05,
    )
    x = torch.randn(B, L, N)
    mu, sigma = model(x)
    assert mu.shape == (B, H, N)
    assert sigma.shape == (B, H, N)
    assert torch.allclose(mu, torch.zeros_like(mu))
    assert torch.allclose(sigma, torch.full_like(mu, 0.05))


def test_timesnet_respects_history_mask():
    B, L, H, N = 1, 12, 2, 2
    total_len = L + 6
    torch.manual_seed(1)
    model = TimesNet(
        input_len=L,
        pred_len=H,
        d_model=8,
        n_layers=1,
        k_periods=2,
        pmax=total_len,
        kernel_set=[3],
        dropout=0.0,
        activation="gelu",
        mode="direct",
    )
    with torch.no_grad():
        model(torch.randn(1, L, N))

    padded = torch.randn(B, total_len, N)
    mask = torch.zeros_like(padded)
    mask[:, -L:, :] = 1.0
    trimmed = padded[:, -L:, :]

    model.eval()
    mu_mask, sigma_mask = model(padded, mask=mask)
    mu_trim, sigma_trim = model(trimmed)

    assert torch.allclose(mu_mask, mu_trim, atol=1e-5)
    assert torch.allclose(sigma_mask, sigma_trim, atol=1e-5)
