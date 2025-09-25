from __future__ import annotations

import torch


def negative_binomial_mask(
    y: torch.Tensor,
    rate: torch.Tensor,
    dispersion: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute a boolean mask for valid NB likelihood elements."""

    finite_mask = torch.isfinite(y) & torch.isfinite(rate) & torch.isfinite(dispersion)
    if mask is not None:
        mask_bool = mask.to(dtype=torch.bool)
        if mask_bool.shape != finite_mask.shape:
            mask_bool = mask_bool.expand_as(finite_mask)
        finite_mask = finite_mask & mask_bool
    return finite_mask


def negative_binomial_nll(
    y: torch.Tensor,
    rate: torch.Tensor,
    dispersion: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Negative binomial negative log-likelihood averaged over valid elements."""

    dtype = torch.float32
    y = torch.clamp(y.to(dtype), min=0.0)
    rate = rate.to(dtype)
    dispersion = dispersion.to(dtype)

    alpha = torch.clamp(dispersion, min=eps)
    mu = torch.clamp(rate, min=eps)
    log1p_alpha_mu = torch.log1p(alpha * mu)
    log_alpha = torch.log(alpha)
    log_mu = torch.log(mu)
    inv_alpha = torch.reciprocal(alpha)
    ll = (
        torch.lgamma(y + inv_alpha)
        - torch.lgamma(inv_alpha)
        - torch.lgamma(y + 1.0)
        + inv_alpha * (-log1p_alpha_mu)
        + y * (log_alpha + log_mu - log1p_alpha_mu)
    )

    valid_mask = negative_binomial_mask(y, mu, alpha, mask)
    weight = valid_mask.to(dtype)
    denom = torch.clamp(weight.sum(), min=1.0)
    return -(ll * weight).sum() / denom
