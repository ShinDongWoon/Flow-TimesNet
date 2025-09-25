from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.losses import negative_binomial_nll


def test_negative_binomial_nll_matches_manual():
    y = torch.tensor([[[0.0, 5.0], [2.0, 1.0]]], dtype=torch.float32)
    rate = torch.tensor([[[1.5, 2.0], [3.0, 0.5]]], dtype=torch.float32)
    dispersion = torch.tensor([[[0.25, 0.5], [0.75, 0.25]]], dtype=torch.float32)

    loss = negative_binomial_nll(y, rate, dispersion)

    alpha = torch.clamp(dispersion, min=1e-8)
    mu = torch.clamp(rate, min=1e-8)
    r = 1.0 / alpha
    log_p = torch.log(r) - torch.log(r + mu)
    log1m_p = torch.log(mu) - torch.log(r + mu)
    manual = -(
        torch.lgamma(y + r)
        - torch.lgamma(r)
        - torch.lgamma(y + 1.0)
        + r * log_p
        + y * log1m_p
    )
    assert torch.allclose(loss, manual.mean())


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
    r = 1.0 / alpha
    log_p = torch.log(r) - torch.log(r + mu)
    log1m_p = torch.log(mu) - torch.log(r + mu)
    log_prob = (
        torch.lgamma(y + r)
        - torch.lgamma(r)
        - torch.lgamma(y + 1.0)
        + r * log_p
        + y * log1m_p
    )
    manual_masked = -(log_prob * mask).sum() / mask.sum()

    assert torch.allclose(masked_loss, manual_masked)


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
