import math
from pathlib import Path
import sys
from typing import List, Tuple

import torch

# Ensure the project src is on the path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.models.timesnet import PeriodicityTransform, PeriodGroup


def _periodicity_transform_naive(
    x: torch.Tensor,
    k: int,
    pmax: int,
    mask: torch.Tensor | None = None,
    min_period_threshold: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation using explicit loops for verification."""
    B, T, N = x.shape
    if k <= 0:
        empty = x.new_zeros(B, N, 0, pmax)
        return empty, empty

    BN = B * N
    seqs = x.permute(0, 2, 1).reshape(BN, T)

    if mask is not None:
        if mask.shape != x.shape:
            raise ValueError("mask must match the input shape")
        mask_flat = mask.permute(0, 2, 1).reshape(BN, T)
        valid_lengths = (mask_flat > 0).to(torch.long).sum(dim=-1)
    else:
        valid_lengths = torch.full((BN,), T, dtype=torch.long, device=x.device)

    if mask is None:
        kidx = PeriodicityTransform._topk_freq(seqs, k)
    else:
        cols = max(k, 1)
        kidx = torch.ones((BN, cols), dtype=torch.long, device=x.device)
        has_data = valid_lengths > 0
        if has_data.any():
            uniq = torch.unique(valid_lengths[has_data])
            for length_val in uniq.tolist():
                length_int = int(length_val)
                if length_int <= 0:
                    continue
                idxs = torch.nonzero(valid_lengths == length_int, as_tuple=False).squeeze(-1)
                if idxs.numel() == 0:
                    continue
                seq_subset = seqs.index_select(0, idxs)
                seq_trim = seq_subset[:, T - length_int :]
                subset_kidx = PeriodicityTransform._topk_freq(seq_trim, k)
                if subset_kidx.size(1) < cols:
                    pad = torch.ones(
                        (subset_kidx.size(0), cols - subset_kidx.size(1)),
                        dtype=torch.long,
                        device=x.device,
                    )
                    subset_kidx = torch.cat([subset_kidx, pad], dim=1)
                kidx.index_copy_(0, idxs, subset_kidx[:, :cols])

    K = kidx.size(1)
    if K == 0 or kidx.numel() == 0:
        empty = x.new_zeros(B, N, 0, pmax)
        return empty, empty

    kidx = kidx.view(BN, K)
    effective_lengths = torch.clamp(valid_lengths, min=1)
    lengths_exp = effective_lengths.view(BN, 1)
    kidx_clamped = torch.clamp(kidx, min=1)
    periods = lengths_exp // kidx_clamped
    periods = torch.clamp(periods, max=pmax)
    min_period = torch.full_like(periods, min_period_threshold)
    min_period = torch.minimum(min_period, lengths_exp)
    periods = torch.maximum(periods, min_period)
    periods = torch.clamp(periods, min=1)

    has_observations = (valid_lengths > 0).view(BN, 1)
    cycles = torch.where(
        has_observations,
        torch.clamp(lengths_exp // torch.clamp(periods, min=1), min=1),
        torch.zeros_like(periods),
    )

    if cycles.numel() == 0:
        empty = x.new_zeros(B, N, 0, pmax)
        return empty, empty

    max_cycles = int(cycles.max().item()) if cycles.numel() > 0 else 0
    if max_cycles <= 0:
        empty = x.new_zeros(B, N, 0, pmax)
        return empty, empty

    folded = x.new_zeros(B, N, K * max_cycles, pmax)
    mask = torch.zeros_like(folded)
    for bn in range(BN):
        seq = seqs[bn]
        b = bn // N
        n = bn % N
        for freq_rank in range(K):
            period_val = int(periods[bn, freq_rank].item())
            cycle_val = int(cycles[bn, freq_rank].item())
            if period_val <= 0 or cycle_val <= 0:
                continue
            take = cycle_val * period_val
            if take <= 0:
                continue
            start = max(T - take, 0)
            idx_range = torch.arange(take, device=x.device, dtype=torch.long)
            if idx_range.numel() == 0:
                continue
            gather_idx = (start + idx_range).clamp(min=0, max=max(T - 1, 0))
            tile = seq.index_select(0, gather_idx).view(cycle_val, period_val)
            row_base = freq_rank * max_cycles
            row_slice = slice(row_base, row_base + cycle_val)
            col_slice = slice(0, period_val)
            folded[b, n, row_slice, col_slice] = tile
            mask[b, n, row_slice, col_slice] = 1.0
    return folded, mask


def _groups_to_dense(
    groups: List[PeriodGroup],
    x: torch.Tensor,
    k: int,
    pmax: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct dense representations from grouped period tiles."""

    B, _, N = x.shape
    if k <= 0 or len(groups) == 0:
        empty = x.new_zeros(B, N, 0, pmax)
        return empty, empty

    max_cycles = max((g.cycles for g in groups), default=0)
    if max_cycles <= 0:
        empty = x.new_zeros(B, N, 0, pmax)
        return empty, empty

    folded = x.new_zeros(B, N, k * max_cycles, pmax)
    mask = torch.zeros_like(folded)
    for group in groups:
        cycles = group.cycles
        period = group.period
        if cycles <= 0 or period <= 0:
            continue
        for tile, bn_idx, freq_idx in zip(
            group.values, group.batch_indices, group.frequency_indices
        ):
            bn = int(bn_idx.item())
            freq = int(freq_idx.item())
            b = bn // N
            n = bn % N
            row_base = freq * max_cycles
            row_slice = slice(row_base, row_base + cycles)
            col_slice = slice(0, period)
            folded[b, n, row_slice, col_slice] = tile
            mask[b, n, row_slice, col_slice] = 1.0
    return folded, mask


def test_periodicity_transform_period_length_consistency():
    """PeriodicityTransform should output a shared period length across channels."""
    B, T, N = 2, 200, 3
    t = torch.arange(T, dtype=torch.float32)
    # Each channel has a different dominant frequency
    freqs = [2, 5, 8]
    x = torch.stack([torch.sin(2 * math.pi * f * t / T) for f in freqs], dim=-1)
    x = x.unsqueeze(0).repeat(B, 1, 1)
    # small noise to make signal more realistic
    x += 0.01 * torch.randn_like(x)

    k = 1
    transform = PeriodicityTransform(k, pmax=T)
    groups = transform(x)
    folded, mask = _groups_to_dense(groups, x, k, transform.pmax)

    assert folded.shape == mask.shape
    assert folded.shape[:2] == (B, N)
    assert folded.shape[-1] == T
    assert folded.shape[2] % k == 0
    assert torch.all((mask > 0).any(dim=(2, 3)))


def test_periodicity_transform_matches_naive():
    """Vectorised implementation should match naive loop-based version."""
    torch.manual_seed(0)
    B, T, N, k = 3, 64, 4, 3
    x = torch.randn(B, T, N)
    pmax = T
    out_ref, mask_ref = _periodicity_transform_naive(x, k, pmax)
    transform = PeriodicityTransform(k, pmax=pmax)
    groups = transform(x)
    out_vec, mask_vec = _groups_to_dense(groups, x, k, pmax)
    assert torch.allclose(out_vec, out_ref, atol=1e-6)
    assert torch.allclose(mask_vec, mask_ref, atol=1e-6)


def test_periodicity_transform_matches_naive_with_mask():
    """Vectorised implementation should match naive reference when masked."""

    torch.manual_seed(1)
    B, T, N, k = 2, 48, 3, 2
    x = torch.randn(B, T, N)
    pmax = T

    valid_lengths = torch.randint(0, T + 1, (B, N))
    if not torch.any(valid_lengths > 0):
        valid_lengths[0, 0] = T

    hist_mask = torch.zeros(B, T, N, dtype=torch.float32)
    for b in range(B):
        for n in range(N):
            length = int(valid_lengths[b, n].item())
            if length > 0:
                hist_mask[b, -length:, n] = 1.0

    out_ref, mask_ref = _periodicity_transform_naive(x, k, pmax, mask=hist_mask)
    transform = PeriodicityTransform(k, pmax=pmax)
    groups = transform(x, mask=hist_mask)
    out_vec, mask_vec = _groups_to_dense(groups, x, k, pmax)

    assert torch.allclose(out_vec, out_ref, atol=1e-6)
    assert torch.allclose(mask_vec, mask_ref, atol=1e-6)


def test_periodicity_transform_respects_history_mask():
    period = 4
    valid_len = 16
    total_len = 32
    t_valid = torch.arange(valid_len, dtype=torch.float32)
    signal = torch.sin(2 * math.pi * t_valid / period)
    trimmed = signal.view(1, valid_len, 1)
    padded_prefix = torch.zeros((1, total_len - valid_len, 1), dtype=torch.float32)
    padded = torch.cat([padded_prefix, trimmed], dim=1)
    hist_mask = torch.zeros_like(padded)
    hist_mask[:, -valid_len:, :] = 1.0

    transform = PeriodicityTransform(k_periods=1, pmax=total_len)
    groups_masked = transform(padded, mask=hist_mask)
    folded_masked, mask_masked = _groups_to_dense(
        groups_masked, padded, transform.k, transform.pmax
    )
    groups_trim = transform(trimmed)
    folded_trim, mask_trim = _groups_to_dense(
        groups_trim, trimmed, transform.k, transform.pmax
    )

    assert torch.allclose(folded_masked, folded_trim, atol=1e-6)
    assert torch.allclose(mask_masked, mask_trim, atol=1e-6)


def test_periodicity_transform_min_period_threshold_expands_period():
    class FixedFreqTransform(PeriodicityTransform):
        def _topk_freq(self, x: torch.Tensor, k: int) -> torch.Tensor:  # type: ignore[override]
            BN = x.shape[0]
            return torch.full((BN, k), 20, dtype=torch.long, device=x.device)

    x = torch.arange(1, 51, dtype=torch.float32).view(1, 50, 1)

    low = FixedFreqTransform(k_periods=1, pmax=10, min_period_threshold=1)
    high = FixedFreqTransform(k_periods=1, pmax=10, min_period_threshold=6)

    groups_low = low(x)
    _, mask_low = _groups_to_dense(groups_low, x, low.k, low.pmax)
    groups_high = high(x)
    _, mask_high = _groups_to_dense(groups_high, x, high.k, high.pmax)

    # With the higher minimum period, more slots along P dimension remain populated.
    low_active = (mask_low > 0).any(dim=2).squeeze(0).squeeze(0)
    high_active = (mask_high > 0).any(dim=2).squeeze(0).squeeze(0)
    nz_low = torch.count_nonzero(low_active)
    nz_high = torch.count_nonzero(high_active)

    assert nz_low.item() == 2  # T // 20 = 2 before applying threshold
    assert nz_high.item() == 6  # Clamped up to the requested minimum


def test_periodicity_transform_take_gt_T_with_compile():
    """Regression test for take > T with torch.compile."""
    torch.manual_seed(0)
    B, T, N = 1, 5, 1
    x = torch.randn(B, T, N)

    class DegenerateTransform(PeriodicityTransform):
        def forward(self, x: torch.Tensor, mask=None):  # type: ignore[override]
            B, T, N = x.shape
            BN = B * N
            if self.k <= 0:
                return []
            seqs = x.permute(0, 2, 1).reshape(BN, T)
            tiles = []
            batch = []
            freq = []
            for bn in range(BN):
                seq = seqs[bn]
                tile = seq.new_empty(self.pmax)
                if T > 0:
                    tile[:T] = seq
                    tile[T:] = seq[-1]
                else:
                    tile.fill_(0.0)
                tiles.append(tile.view(1, self.pmax))
                batch.append(bn)
                freq.append(0)
            if not tiles:
                return []
            values = torch.stack(tiles, dim=0)
            batch_idx = torch.tensor(batch, dtype=torch.long, device=x.device)
            freq_idx = torch.tensor(freq, dtype=torch.long, device=x.device)
            group = PeriodGroup(
                values=values,
                batch_indices=batch_idx,
                frequency_indices=freq_idx,
                cycles=1,
                period=self.pmax,
            )
            return [group]

    transform = DegenerateTransform(k_periods=1, pmax=T + 2)
    compiled = torch.compile(transform)
    groups = compiled(x)
    assert isinstance(groups, list)
    assert len(groups) == 1
    assert groups[0].period == transform.pmax

