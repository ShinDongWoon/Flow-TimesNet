from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.losses import negative_binomial_mask, negative_binomial_nll


def test_negative_binomial_nll_matches_manual():
    y = torch.tensor([[[0.0, 5.0], [2.0, 1.0]]], dtype=torch.float32)
    rate = torch.tensor([[[1.5, 2.0], [3.0, 0.5]]], dtype=torch.float32)
    dispersion = torch.tensor([[[0.25, 0.5], [0.75, 0.25]]], dtype=torch.float32)

    loss = negative_binomial_nll(y, rate, dispersion)

    alpha = torch.clamp(dispersion, min=1e-8)
    mu = torch.clamp(rate, min=1e-8)
    log1p_alpha_mu = torch.log1p(alpha * mu)
    log_alpha = torch.log(alpha)
    log_mu = torch.log(mu)
    inv_alpha = 1.0 / alpha
    manual_ll = (
        torch.lgamma(y + inv_alpha)
        - torch.lgamma(inv_alpha)
        - torch.lgamma(y + 1.0)
        + inv_alpha * (-log1p_alpha_mu)
        + y * (log_alpha + log_mu - log1p_alpha_mu)
    )
    assert torch.allclose(loss, -manual_ll.mean())


def test_negative_binomial_nll_respects_mask():
    y = torch.tensor([[[1.0, 0.0], [4.0, 2.0]]], dtype=torch.float32)
    rate = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
    dispersion = torch.full_like(rate, 0.5)
    mask = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], dtype=torch.float32)

    full_loss = negative_binomial_nll(y, rate, dispersion)
    masked_loss = negative_binomial_nll(y, rate, dispersion, mask=mask)

    # Manual computation using only the unmasked elements
    alpha = torch.clamp(dispersion, min=1e-8)
    mu = torch.clamp(rate, min=1e-8)
    inv_alpha = 1.0 / alpha
    log1p_alpha_mu = torch.log1p(alpha * mu)
    log_alpha = torch.log(alpha)
    log_mu = torch.log(mu)
    log_prob = (
        torch.lgamma(y + inv_alpha)
        - torch.lgamma(inv_alpha)
        - torch.lgamma(y + 1.0)
        + inv_alpha * (-log1p_alpha_mu)
        + y * (log_alpha + log_mu - log1p_alpha_mu)
    )
    manual_masked = -(log_prob * mask).sum() / mask.sum()

    assert torch.allclose(masked_loss, manual_masked)


def test_negative_binomial_mask_ignores_zeros_but_masks_nans():
    y = torch.tensor([[[0.0, float("nan")]]])
    rate = torch.tensor([[[1.0, 2.0]]])
    dispersion = torch.tensor([[[0.5, 0.5]]])
    base_mask = torch.ones_like(y)

    valid_mask = negative_binomial_mask(y, rate, dispersion, base_mask)
    assert valid_mask.dtype == torch.bool
    assert valid_mask.shape == y.shape
    assert valid_mask[0, 0, 0]
    assert not valid_mask[0, 0, 1]


def test_negative_binomial_nll_autocast_stability():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
    elif hasattr(torch, "autocast"):
        autocast_ctx = torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    else:
        autocast_ctx = nullcontext()

    y = torch.full((2, 3, 1), 10.0, dtype=torch.float32, device=device)
    rate = torch.full_like(y, 5.0)
    dispersion = torch.full_like(y, 0.2)

    with autocast_ctx:
        loss = negative_binomial_nll(y, rate, dispersion)

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
