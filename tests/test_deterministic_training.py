import math
from pathlib import Path
import sys
from typing import Dict

import torch

# Ensure the project src is on the path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.models.timesnet import TimesNet
from timesnet_forecast.losses import negative_binomial_nll
from timesnet_forecast.utils.seed import seed_everything


def _run_short_training(seed: int) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    seed_everything(seed, deterministic=True)

    T = 48
    input_len, pred_len = 12, 4
    t = torch.arange(T, dtype=torch.float32)
    data = torch.stack(
        [
            0.5 + torch.sin(2 * math.pi * t / 12.0),
            -0.25 + torch.cos(2 * math.pi * t / 8.0),
        ],
        dim=-1,
    )

    train_series = data[:32]
    X_windows, Y_windows = [], []
    for start in range(0, train_series.size(0) - input_len - pred_len + 1):
        window = train_series[start : start + input_len + pred_len]
        X_windows.append(window[:input_len])
        Y_windows.append(window[input_len:])
    X = torch.stack(X_windows)
    Y = torch.stack(Y_windows)

    model = TimesNet(
        input_len=input_len,
        pred_len=pred_len,
        d_model=16,
        d_ff=32,
        n_layers=2,
        k_periods=2,
        kernel_set=[(3, 3)],
        dropout=0.0,
        activation="gelu",
        mode="direct",
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    history = []
    batch_size = 8
    for _ in range(3):
        perm = torch.randperm(X.size(0))
        total_loss = 0.0
        count = 0
        for j in range(0, len(perm), batch_size):
            idx = perm[j : j + batch_size]
            xb = X[idx]
            yb = Y[idx]
            optimizer.zero_grad()
            rate, dispersion = model(xb)
            loss = negative_binomial_nll(yb, rate, dispersion)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            count += 1
        history.append(total_loss / max(count, 1))

    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return torch.tensor(history, dtype=torch.float64), state


def test_deterministic_training_reproducible():
    prev_deterministic = torch.backends.cudnn.deterministic
    prev_benchmark = torch.backends.cudnn.benchmark
    prev_algorithms = torch.are_deterministic_algorithms_enabled()
    prev_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

    try:
        losses_a, state_a = _run_short_training(2024)
        losses_b, state_b = _run_short_training(2024)
    finally:
        if prev_algorithms:
            torch.use_deterministic_algorithms(True, warn_only=prev_warn_only)
        else:
            torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = prev_deterministic
        torch.backends.cudnn.benchmark = prev_benchmark

    torch.testing.assert_close(losses_a, losses_b)
    assert state_a.keys() == state_b.keys()
    for key in state_a:
        torch.testing.assert_close(state_a[key], state_b[key])
