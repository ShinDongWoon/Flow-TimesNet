import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

# Ensure the project src is on the path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.models.timesnet import (
    PeriodGroup,
    TimesNet,
    _adaptive_pool_valid_lengths,
    _pool_trailing_sums,
)


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


def test_timesnet_pools_when_mask_shortens_valid_history(monkeypatch):
    import timesnet_forecast.models.timesnet as timesnet_mod

    B, L, H, N = 1, 12, 2, 2
    valid = 3
    torch.manual_seed(2)
    model = TimesNet(
        input_len=L,
        pred_len=H,
        d_model=8,
        n_layers=1,
        k_periods=2,
        pmax=L,
        kernel_set=[3],
        dropout=0.0,
        activation="gelu",
        mode="direct",
    )
    with torch.no_grad():
        model(torch.randn(1, L, N))

    padded = torch.randn(B, L, N)
    mask = torch.zeros_like(padded)
    mask[:, -valid:, :] = 1.0

    calls = {}
    original_pool = timesnet_mod._adaptive_pool_valid_lengths

    def capture_pooling(**kwargs):
        calls["valid_lengths"] = kwargs["valid_lengths"].clone()
        calls["target_len"] = kwargs["target_len"]
        return original_pool(**kwargs)
    monkeypatch.setattr(timesnet_mod, "_adaptive_pool_valid_lengths", capture_pooling)

    model.eval()
    model(padded, mask=mask)

    assert calls, "Expected adaptive pooling to run when mask shortens history"
    assert calls["target_len"] == model.input_len
    assert torch.all(calls["valid_lengths"] == valid)


def test_pool_trailing_sums_matches_adaptive_pool():
    torch.manual_seed(123)
    B, C, T = 4, 3, 19
    target = 7
    values = torch.randn(B, C, T)
    valid_lengths = torch.randint(1, T + 1, (B,))

    sums, counts = _pool_trailing_sums(values, valid_lengths, target)
    counts_f = counts.to(values.dtype).unsqueeze(1)
    averages = sums / counts_f

    pooled, _ = _adaptive_pool_valid_lengths(
        features=values,
        mask=torch.ones(B, 1, T, dtype=values.dtype),
        valid_lengths=valid_lengths,
        target_len=target,
    )

    torch.testing.assert_close(averages, pooled, atol=1e-6, rtol=1e-5)


def test_pool_trailing_sums_streaming_matches_reference_cpu(monkeypatch):
    torch.manual_seed(7)
    B, C, T = 8, 4, 31
    target = 9
    values = torch.randn(B, C, T)
    valid_lengths = torch.randint(0, T + 1, (B,), dtype=torch.long)

    monkeypatch.setenv("TIMESNET_POOL_MAX_BYTES", str(1_000_000_000))
    ref_sums, ref_counts = _pool_trailing_sums(values, valid_lengths, target)

    monkeypatch.setenv("TIMESNET_POOL_MAX_BYTES", "512")
    stream_sums, stream_counts = _pool_trailing_sums(values, valid_lengths, target)

    torch.testing.assert_close(stream_sums, ref_sums)
    torch.testing.assert_close(stream_counts, ref_counts)


def test_adaptive_pool_streaming_matches_reference_cpu(monkeypatch):
    torch.manual_seed(11)
    BN, C, T = 6, 3, 27
    target = 5
    features = torch.randn(BN, C, T)
    mask = torch.randint(0, 2, (BN, 2, T), dtype=torch.int32).to(torch.bool)
    valid_lengths = torch.randint(1, T + 1, (BN,), dtype=torch.long)

    monkeypatch.setenv("TIMESNET_POOL_MAX_BYTES", str(1_000_000_000))
    ref_feats, ref_masks = _adaptive_pool_valid_lengths(
        features, mask, valid_lengths, target
    )

    monkeypatch.setenv("TIMESNET_POOL_MAX_BYTES", "512")
    stream_feats, stream_masks = _adaptive_pool_valid_lengths(
        features, mask, valid_lengths, target
    )

    torch.testing.assert_close(stream_feats, ref_feats, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(stream_masks, ref_masks)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pool_trailing_sums_streaming_matches_reference_cuda(monkeypatch):
    torch.manual_seed(21)
    device = torch.device("cuda")
    B, C, T = 5, 4, 28
    target = 6
    values = torch.randn(B, C, T, device=device, dtype=torch.float32)
    valid_lengths = torch.randint(0, T + 1, (B,), dtype=torch.long)

    monkeypatch.setenv("TIMESNET_POOL_MAX_BYTES", str(1_000_000_000))
    ref_sums, ref_counts = _pool_trailing_sums(values, valid_lengths, target)

    monkeypatch.setenv("TIMESNET_POOL_MAX_BYTES", "512")
    stream_sums, stream_counts = _pool_trailing_sums(values, valid_lengths, target)

    torch.testing.assert_close(stream_sums, ref_sums)
    torch.testing.assert_close(stream_counts, ref_counts)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_adaptive_pool_streaming_matches_reference_cuda(monkeypatch):
    torch.manual_seed(22)
    device = torch.device("cuda")
    BN, C, T = 7, 2, 30
    target = 8
    features = torch.randn(BN, C, T, device=device, dtype=torch.float32)
    mask = torch.randint(0, 2, (BN, 2, T), device=device, dtype=torch.int32).to(torch.bool)
    valid_lengths = torch.randint(1, T + 1, (BN,), dtype=torch.long)

    monkeypatch.setenv("TIMESNET_POOL_MAX_BYTES", str(1_000_000_000))
    ref_feats, ref_masks = _adaptive_pool_valid_lengths(
        features, mask, valid_lengths, target
    )

    monkeypatch.setenv("TIMESNET_POOL_MAX_BYTES", "512")
    stream_feats, stream_masks = _adaptive_pool_valid_lengths(
        features, mask, valid_lengths, target
    )

    torch.testing.assert_close(stream_feats, ref_feats, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(stream_masks, ref_masks)


def test_period_group_chunk_preserves_outputs():
    torch.manual_seed(42)
    B, L, H, N = 2, 24, 5, 3
    base = TimesNet(
        input_len=L,
        pred_len=H,
        d_model=16,
        n_layers=1,
        k_periods=3,
        pmax=L,
        kernel_set=[3, 5],
        dropout=0.0,
        activation="gelu",
        mode="direct",
        use_checkpoint=False,
    )
    chunked = TimesNet(
        input_len=L,
        pred_len=H,
        d_model=16,
        n_layers=1,
        k_periods=3,
        pmax=L,
        kernel_set=[3, 5],
        dropout=0.0,
        activation="gelu",
        mode="direct",
        use_checkpoint=False,
    )
    with torch.no_grad():
        base(torch.randn(1, L, N))
        chunked(torch.randn(1, L, N))
    chunked.load_state_dict(base.state_dict())

    x = torch.randn(B, L, N)
    base.eval()
    chunked.eval()
    chunked.period_group_chunk = 1

    mu_full, sigma_full = base(x)
    mu_chunk, sigma_chunk = chunked(x)

    torch.testing.assert_close(mu_chunk, mu_full, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(sigma_chunk, sigma_full, atol=1e-6, rtol=1e-6)


def test_period_group_chunk_helper_caps_size(monkeypatch):
    torch.manual_seed(0)
    B, L, H, N = 1, 128, 4, 1
    model = TimesNet(
        input_len=L,
        pred_len=H,
        d_model=32,
        n_layers=2,
        k_periods=1,
        pmax=L,
        kernel_set=[3, 5, 7],
        dropout=0.0,
        activation="gelu",
        mode="direct",
        use_checkpoint=False,
    )
    model.eval()

    tile_count = 256
    cycles = 64
    period = 64
    dtype = torch.float32
    values = torch.randn(tile_count, cycles, period, dtype=dtype)
    batch_indices = torch.zeros(tile_count, dtype=torch.long)
    freq_indices = torch.zeros(tile_count, dtype=torch.long)
    fake_group = PeriodGroup(
        values=values,
        batch_indices=batch_indices,
        frequency_indices=freq_indices,
        cycles=cycles,
        period=period,
        source_device=values.device,
        source_dtype=values.dtype,
    )

    monkeypatch.setattr(model.period, "forward", lambda *args, **kwargs: [fake_group])

    x = torch.randn(B, L, N, dtype=dtype)
    max_bytes_backup = model.period_group_max_chunk_bytes
    chunk_backup = model.period_group_chunk
    model.period_group_max_chunk_bytes = 1 << 20  # 1 MiB budget

    try:
        with torch.no_grad():
            model.period_group_chunk = tile_count
            mu_full, sigma_full = model(x)

            model.period_group_chunk = None
            frontend_param = next(iter(model.frontend.parameters()), None)
            target_device = frontend_param.device if frontend_param is not None else x.device
            chunk_size = model._resolve_period_group_chunk(
                fake_group,
                target_device=target_device,
            )
            assert chunk_size < tile_count

            mu_chunk, sigma_chunk = model(x)

        torch.testing.assert_close(mu_chunk, mu_full, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(sigma_chunk, sigma_full, atol=1e-6, rtol=1e-6)
    finally:
        model.period_group_max_chunk_bytes = max_bytes_backup
        model.period_group_chunk = chunk_backup


def test_period_group_chunks_shrink_with_budget(monkeypatch):
    torch.manual_seed(0)
    B, L, H, N = 1, 32, 4, 1
    dtype = torch.float32
    model = TimesNet(
        input_len=L,
        pred_len=H,
        d_model=4,
        n_layers=0,
        k_periods=1,
        pmax=L,
        kernel_set=[3],
        dropout=0.0,
        activation="gelu",
        mode="direct",
        use_checkpoint=False,
    )
    with torch.no_grad():
        model(torch.randn(1, L, N, dtype=dtype))

    tile_count = 12
    cycles = 2
    period = 3
    values = torch.randn(tile_count, cycles, period, dtype=dtype)
    batch_indices = torch.zeros(tile_count, dtype=torch.long)
    freq_indices = torch.zeros(tile_count, dtype=torch.long)
    fake_group = PeriodGroup(
        values=values,
        batch_indices=batch_indices,
        frequency_indices=freq_indices,
        cycles=cycles,
        period=period,
        source_device=values.device,
        source_dtype=values.dtype,
    )

    monkeypatch.setattr(model.period, "forward", lambda *args, **kwargs: [fake_group])

    element_size = torch.tensor(0, dtype=dtype).element_size()
    base_elements = float(model.d_model) * float(cycles) * float(period)
    kernel_paths = max(len(model.kernel_set), 1)
    block_multiplier = max(model.n_layers, 1) * kernel_paths
    estimated_elements = base_elements * (1.0 + float(block_multiplier))
    safety_factor = 4.0
    bytes_per_tile = int(
        max(math.ceil(estimated_elements * element_size * safety_factor), 1)
    )

    model.period_group_max_chunk_bytes = bytes_per_tile * 5
    model.period_group_chunk = None
    model.period_group_memory_ratio = 1.0

    chunk_sizes: list[int] = []

    def capture_frontend(module, inputs, output):
        chunk_sizes.append(int(inputs[0].shape[0]))

    hook = model.frontend.register_forward_hook(capture_frontend)
    try:
        x = torch.randn(B, L, N, dtype=dtype)
        model.eval()
        with torch.no_grad():
            model(x)
    finally:
        hook.remove()

    assert len(chunk_sizes) >= 3
    assert chunk_sizes[0] > 1
    assert chunk_sizes[1] < chunk_sizes[0]
    assert all(size <= chunk_sizes[1] for size in chunk_sizes[1:])


def test_adaptive_pool_matches_reference_for_variable_lengths():
    torch.manual_seed(0)
    BN, C, T = 7, 5, 23
    output_len = 11
    features = torch.randn(BN, C, T)
    valid_lengths = torch.randint(1, T + 1, (BN,), dtype=torch.long)

    step_mask = torch.zeros(BN, 1, T, dtype=features.dtype)
    for idx, length in enumerate(valid_lengths.tolist()):
        start = T - length
        step_mask[idx, :, start:] = 1.0

    # Reference implementation using the original per-index slicing
    ref_feats = []
    ref_masks = []
    for idx in range(BN):
        length = int(valid_lengths[idx].item())
        length = max(min(length, T), 1)
        start_idx = T - length
        feat_slice = features[idx : idx + 1, :, start_idx:]
        mask_slice = step_mask[idx : idx + 1, :, start_idx:]
        ref_feats.append(F.adaptive_avg_pool1d(feat_slice, output_len))
        ref_masks.append(F.adaptive_avg_pool1d(mask_slice, output_len))
    expected_feats = torch.cat(ref_feats, dim=0)
    expected_masks = torch.cat(ref_masks, dim=0)

    pooled_feats, pooled_masks = _adaptive_pool_valid_lengths(
        features, step_mask, valid_lengths, output_len
    )

    torch.testing.assert_close(pooled_feats, expected_feats)
    torch.testing.assert_close(pooled_masks, expected_masks)
