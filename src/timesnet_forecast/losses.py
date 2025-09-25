from __future__ import annotations

import torch


def negative_binomial_nll(
    y: torch.Tensor,
    rate: torch.Tensor,
    dispersion: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Negative binomial negative log-likelihood averaged over valid elements."""

    y = y.to(rate.dtype)
    alpha = torch.clamp(dispersion, min=eps)
    mu = torch.clamp(rate, min=eps)
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
    if mask is not None:
        log_prob = log_prob * mask.to(log_prob.dtype)
        denom = torch.clamp(mask.sum(), min=1.0)
    else:
        denom = log_prob.numel()
    return -(log_prob.sum() / denom)
