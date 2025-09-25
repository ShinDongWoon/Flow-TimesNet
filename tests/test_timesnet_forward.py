import sys
from pathlib import Path

import torch
import torch.nn as nn

# Ensure the project src is on the path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.models.timesnet import TimesNet
from timesnet_forecast.losses import negative_binomial_nll


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
    rate_train, dispersion_train = model(x)
    model.eval()
    rate_eval, dispersion_eval = model(x)
    assert rate_train.shape == rate_eval.shape == (B, H, N)
    assert dispersion_train.shape == dispersion_eval.shape == (B, H, N)
    assert torch.all(rate_eval > 0)
    assert torch.all(dispersion_eval > 0)

    long_x = torch.randn(B, L + 5, N)
    rate_long, dispersion_long = model(long_x)
    rate_head, dispersion_head = model(long_x[:, :L, :])
    assert rate_long.shape == rate_head.shape == (B, H, N)
    assert dispersion_long.shape == dispersion_head.shape == (B, H, N)


def test_timesnet_blocks_track_period_calls():
    torch.manual_seed(0)
    B, L, H, N = 1, 12, 3, 1
    model = TimesNet(
        input_len=L,
        pred_len=H,
        d_model=8,
        d_ff=16,
        n_layers=2,
        k_periods=2,
        kernel_set=[(3, 3)],
        dropout=0.0,
        activation="gelu",
        mode="direct",
    )
    x = torch.randn(B, L, N)
    with torch.no_grad():
        model(x)
    period_counts = [getattr(block, "_period_calls", 0) for block in model.blocks]
    assert len(period_counts) == 2
    assert all(count > 0 for count in period_counts)


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
    _, dispersion = model(x)
    expected_floor = floor.view(1, 1, N)
    assert torch.all(dispersion >= expected_floor)


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
    rate, dispersion = model(x)
    loss = negative_binomial_nll(y, rate, dispersion)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        rate_eval, dispersion_eval = model(x)
        post_update_loss = negative_binomial_nll(y, rate_eval, dispersion_eval)
        assert torch.isfinite(post_update_loss)
        assert torch.all(rate_eval > 0)
        assert torch.all(dispersion_eval > 0)


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
    rate, dispersion = model(xb, series_static=static_ref, series_ids=ids_ref)
    assert rate.shape == (B, H, N)
    assert dispersion.shape == (B, H, N)
    assert torch.all(rate > 0)
    assert torch.all(dispersion > 0)

    # Batched static/id tensors should also be accepted
    static_batched = static_ref.unsqueeze(0)
    ids_batched = ids_ref.unsqueeze(0)
    rate_batched, dispersion_batched = model(
        xb[:1], series_static=static_batched, series_ids=ids_batched
    )
    assert rate_batched.shape == (1, H, N)
    assert dispersion_batched.shape == (1, H, N)


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
    rate_ref, dispersion_ref = model(xb, series_static=static_ref, series_ids=ids_ref)

    scaled_static = static_ref * 250.0
    rate_scaled, dispersion_scaled = model(
        xb, series_static=scaled_static, series_ids=ids_ref
    )

    assert torch.all(torch.isfinite(rate_scaled))
    assert torch.all(torch.isfinite(dispersion_scaled))

    rate_ref_mean = rate_ref.abs().mean()
    rate_scaled_mean = rate_scaled.abs().mean()
    dispersion_ref_mean = dispersion_ref.abs().mean()
    dispersion_scaled_mean = dispersion_scaled.abs().mean()

    rate_rel_change = (rate_ref_mean - rate_scaled_mean).abs() / rate_ref_mean.clamp(
        min=1e-6
    )
    dispersion_rel_change = (
        (dispersion_ref_mean - dispersion_scaled_mean).abs()
        / dispersion_ref_mean.clamp(min=1e-6)
    )

    assert rate_rel_change < 0.2
    assert dispersion_rel_change < 0.2
