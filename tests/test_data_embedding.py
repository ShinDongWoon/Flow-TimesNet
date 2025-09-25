from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.models.timesnet import DataEmbedding


def test_data_embedding_preserves_temporal_variance() -> None:
    torch.manual_seed(0)
    batch = 8
    length = 64
    features = 3
    d_model = 16

    t = torch.linspace(0, 2 * math.pi, steps=length)
    base = torch.stack(
        (
            torch.sin(t),
            torch.cos(t),
            torch.sin(2 * t + 0.5),
        ),
        dim=-1,
    )  # [L, F]
    x = base.unsqueeze(0).repeat(batch, 1, 1)
    x = x + 0.05 * torch.randn_like(x)

    embed = DataEmbedding(
        c_in=features,
        d_model=d_model,
        dropout=0.0,
        time_features=None,
        use_norm=True,
        embed_norm_mode="decoupled",
    )

    embed.eval()
    with torch.no_grad():
        value_branch = embed.value_embedding(x)
        out = embed(x)

    value_time_var = value_branch.var(dim=1, unbiased=False)
    out_time_var = out.var(dim=1, unbiased=False)

    ratio = (out_time_var.mean(dim=1) / value_time_var.mean(dim=1)).min()
    assert torch.isfinite(ratio)
    assert float(ratio) > 0.1


def test_data_embedding_default_mode_decoupled() -> None:
    embed = DataEmbedding(
        c_in=2,
        d_model=4,
        dropout=0.0,
        time_features=None,
        use_norm=True,
    )
    assert embed.embed_norm_mode == "decoupled"
