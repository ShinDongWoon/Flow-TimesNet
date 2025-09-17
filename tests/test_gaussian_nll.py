import math
import sys
from pathlib import Path

import torch
from contextlib import nullcontext

# Ensure the project src is on the path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.train import gaussian_nll_loss


def test_gaussian_nll_matches_manual_computation():
    mu = torch.tensor([[[0.5, -1.0], [1.5, 0.0]]], dtype=torch.float32)
    sigma = torch.tensor([[[1.0, 0.5], [2.0, 1.5]]], dtype=torch.float32)
    target = torch.tensor([[[1.0, -0.5], [2.0, -0.5]]], dtype=torch.float32)

    loss = gaussian_nll_loss(mu, sigma, target)
    manual = 0.5 * (
        ((target - mu) / sigma) ** 2
        + 2.0 * torch.log(sigma)
        + math.log(2.0 * math.pi)
    )
    assert torch.allclose(loss, manual)


def test_gaussian_nll_respects_min_sigma():
    mu = torch.zeros((1, 1, 1), dtype=torch.float32)
    sigma = torch.full((1, 1, 1), 1e-8, dtype=torch.float32)
    target = torch.zeros((1, 1, 1), dtype=torch.float32)

    loss_clamped = gaussian_nll_loss(mu, sigma, target, min_sigma=1e-3)
    loss_reference = gaussian_nll_loss(mu, torch.full_like(sigma, 1e-3), target)
    assert torch.allclose(loss_clamped, loss_reference)


def test_gaussian_nll_autocast_stability():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
    else:
        if not hasattr(torch, "autocast"):
            autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(device_type="cpu", dtype=torch.bfloat16)

    mu = torch.zeros((1, 1, 1), dtype=torch.float32, device=device)
    sigma = torch.full((1, 1, 1), 1e-6, dtype=torch.float32, device=device)
    target = torch.full((1, 1, 1), 1e6, dtype=torch.float32, device=device)

    with autocast_ctx:
        loss = gaussian_nll_loss(mu, sigma, target, min_sigma=1e-6)

    assert torch.isfinite(loss).all()
