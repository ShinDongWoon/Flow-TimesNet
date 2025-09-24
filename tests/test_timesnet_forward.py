import sys
from pathlib import Path

import torch
import torch.nn as nn

# Ensure the project src is on the path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.models.timesnet import TimesNet
from timesnet_forecast.train import gaussian_nll_loss


def test_forward_shape_and_head_processing():
    B, L, H, N = 2, 16, 4, 3
    torch.manual_seed(0)

    model = TimesNet(
        input_len=L,
        pred_len=H,
        d_model=8,
        d_ff=16,
        n_layers=1,
        k_periods=1,
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


def test_timesnet_pre_embedding_norm_adapts_to_feature_count():
    torch.manual_seed(0)
    B, L, H, N = 2, 12, 4, 1
    model = TimesNet(
        input_len=L,
        pred_len=H,
        d_model=8,
        d_ff=16,
        n_layers=1,
        k_periods=1,
        kernel_set=[(3, 3)],
        dropout=0.0,
        activation="gelu",
        mode="direct",
        id_embed_dim=0,
        static_proj_dim=2,
    )

    xb = torch.randn(B, L, N)
    with torch.no_grad():
        model(xb)
    assert isinstance(model.pre_embedding_norm, nn.Identity)

    static_features = torch.randn(N, 5)
    with torch.no_grad():
        model(xb, series_static=static_features)
    assert isinstance(model.pre_embedding_norm, nn.LayerNorm)
    assert tuple(model.pre_embedding_norm.normalized_shape) == (3,)


def test_timesnet_applies_per_series_floor():
    B, L, H, N = 1, 12, 3, 4
    floor = torch.tensor([0.05, 0.1, 0.2, 0.3], dtype=torch.float32)

    model = TimesNet(
        input_len=L,
        pred_len=H,
        d_model=8,
        d_ff=16,
        n_layers=1,
        k_periods=1,
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


def test_timesnet_sigma_head_produces_finite_loss():
    torch.manual_seed(0)
    B, L, H, N = 4, 24, 6, 2

    model = TimesNet(
        input_len=L,
        pred_len=H,
        d_model=16,
        d_ff=32,
        n_layers=2,
        k_periods=2,
        kernel_set=[(3, 3)],
        dropout=0.1,
        activation="gelu",
        mode="direct",
    )
    model.train()
    with torch.no_grad():
        _ = model(torch.zeros(1, L, N))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x = torch.randn(B, L, N)
    y = torch.randn(B, H, N)

    optimizer.zero_grad()
    mu, sigma = model(x)
    loss = gaussian_nll_loss(mu, sigma, y).mean()
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        mu_eval, sigma_eval = model(x)
        post_update_loss = gaussian_nll_loss(mu_eval, sigma_eval, y).mean()
        assert torch.isfinite(post_update_loss)
        assert torch.all(sigma_eval > 0)


def test_timesnet_static_and_id_features_pipeline():
    torch.manual_seed(0)
    B, L, H, N, static_dim = 3, 18, 5, 4, 6
    model = TimesNet(
        input_len=L,
        pred_len=H,
        d_model=16,
        d_ff=32,
        n_layers=2,
        k_periods=2,
        kernel_set=[(3, 3)],
        dropout=0.1,
        activation="gelu",
        mode="direct",
        id_embed_dim=8,
        static_proj_dim=10,
    )

    # Build lazy layers with full feature inputs
    static_ref = torch.randn(N, static_dim)
    ids_ref = torch.arange(N, dtype=torch.long)
    with torch.no_grad():
        model(torch.zeros(1, L, N), series_static=static_ref, series_ids=ids_ref)

    assert model.series_embedding is not None
    assert model.series_embedding.embedding_dim == model.id_embed_dim == 8
    assert model.static_proj is not None
    assert model.static_proj.out_features == 10

    xb = torch.randn(B, L, N)
    mu, sigma = model(xb, series_static=static_ref, series_ids=ids_ref)
    assert mu.shape == (B, H, N)
    assert sigma.shape == (B, H, N)
    assert torch.all(sigma > 0)

    # Batched static/id tensors should also be accepted
    static_batched = static_ref.unsqueeze(0)
    ids_batched = ids_ref.unsqueeze(0)
    mu_batched, sigma_batched = model(
        xb[:1], series_static=static_batched, series_ids=ids_batched
    )
    assert mu_batched.shape == (1, H, N)
    assert sigma_batched.shape == (1, H, N)


def test_timesnet_static_id_normalization_preserves_scale():
    torch.manual_seed(0)
    B, L, H, N, static_dim = 2, 12, 3, 3, 5
    model = TimesNet(
        input_len=L,
        pred_len=H,
        d_model=8,
        d_ff=16,
        n_layers=1,
        k_periods=1,
        kernel_set=[(3, 3)],
        dropout=0.0,
        activation="gelu",
        mode="direct",
        id_embed_dim=6,
        static_proj_dim=7,
    )

    static_ref = torch.randn(N, static_dim)
    ids_ref = torch.arange(N, dtype=torch.long)

    with torch.no_grad():
        model(torch.zeros(1, L, N), series_static=static_ref, series_ids=ids_ref)

    model.eval()
    xb = torch.randn(B, L, N)
    mu_ref, sigma_ref = model(xb, series_static=static_ref, series_ids=ids_ref)

    scaled_static = static_ref * 250.0
    mu_scaled, sigma_scaled = model(
        xb, series_static=scaled_static, series_ids=ids_ref
    )

    assert torch.all(torch.isfinite(mu_scaled))
    assert torch.all(torch.isfinite(sigma_scaled))

    mu_ref_mean = mu_ref.abs().mean()
    mu_scaled_mean = mu_scaled.abs().mean()
    sigma_ref_mean = sigma_ref.abs().mean()
    sigma_scaled_mean = sigma_scaled.abs().mean()

    mu_rel_change = (mu_ref_mean - mu_scaled_mean).abs() / mu_ref_mean.clamp(min=1e-6)
    sigma_rel_change = (sigma_ref_mean - sigma_scaled_mean).abs() / sigma_ref_mean.clamp(
        min=1e-6
    )

    assert mu_rel_change < 0.2
    assert sigma_rel_change < 0.2
