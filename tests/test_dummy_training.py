import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

# Ensure the project src is on the path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.models.timesnet import TimesNet
from timesnet_forecast.train import gaussian_nll_loss, _eval_metrics
from timesnet_forecast.utils.metrics import wsmape_grouped, smape_mean


def test_dummy_training_smape_wsmape():
    torch.manual_seed(0)
    np.random.seed(0)

    T, N, input_len, pred_len = 80, 2, 16, 4
    t = torch.arange(T, dtype=torch.float32)
    freqs = [2, 4]
    data = torch.stack([10 + torch.sin(2 * math.pi * f * t / T) for f in freqs], dim=-1)

    static_features = torch.tensor(
        [[1.0, -0.5, 0.25], [0.5, 1.0, -0.75]], dtype=torch.float32
    )
    series_ids = torch.arange(N, dtype=torch.long)

    train_series = data[:60]
    Xs, Ys = [], []
    for i in range(len(train_series) - input_len - pred_len + 1):
        Xs.append(train_series[i : i + input_len])
        Ys.append(train_series[i + input_len : i + input_len + pred_len])
    X = torch.stack(Xs)
    Y = torch.stack(Ys)

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
        id_embed_dim=4,
        static_proj_dim=3,
    )
    # Lazily build model parameters
    with torch.no_grad():
        _ = model(X[:1], series_static=static_features, series_ids=series_ids)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(30):
        idx = torch.randperm(X.size(0))
        for j in range(0, len(idx), 4):
            xb = X[idx[j : j + 4]]
            yb = Y[idx[j : j + 4]]
            optimizer.zero_grad()
            mu, sigma = model(xb, series_static=static_features, series_ids=series_ids)
            loss_tensor = gaussian_nll_loss(mu, sigma, yb)
            loss = loss_tensor.mean()
            loss.backward()
            optimizer.step()

    input_seq = data[60 - input_len : 60]
    actual = data[60 : 60 + pred_len]
    with torch.no_grad():
        pred_mu, pred_sigma = model(
            input_seq.unsqueeze(0),
            series_static=static_features,
            series_ids=series_ids,
        )
        pred = pred_mu.squeeze(0)
        assert torch.all(pred_sigma > 0)

    y_true = actual.numpy()
    y_pred = pred.numpy()

    smape = smape_mean(y_true, y_pred)
    wsmape = wsmape_grouped(y_true, y_pred, ids=["A_1", "A_2"])

    assert smape < 0.1
    assert wsmape < 0.1


def test_eval_metrics_returns_masked_nll():
    mu = torch.tensor([[[1.5, 2.0], [2.0, 4.0]]], dtype=torch.float32)
    sigma = torch.full_like(mu, 0.5)
    target = torch.tensor([[[1.0, 2.5], [3.0, 1.0]]], dtype=torch.float32)
    mask = torch.tensor([[[1.0, 0.0], [1.0, 1.0]]], dtype=torch.float32)

    class DummyModel(torch.nn.Module):
        def __init__(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
            super().__init__()
            self.register_buffer("mu_buf", mu)
            self.register_buffer("sigma_buf", sigma)
            self.input_len = 1

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            batch = x.shape[0]
            mu = self.mu_buf.expand(batch, -1, -1)
            sigma = self.sigma_buf.expand(batch, -1, -1)
            return mu, sigma

    model = DummyModel(mu, sigma)
    xb = torch.zeros((1, 3, 2), dtype=torch.float32)
    loader = [(xb, target, mask)]
    metrics = _eval_metrics(
        model,
        loader,
        torch.device("cpu"),
        mode="direct",
        ids=["A_1", "A_2"],
        pred_len=2,
        channels_last=False,
        use_loss_mask=True,
        min_sigma=0.0,
    )

    expected_loss = gaussian_nll_loss(mu, sigma, target)
    expected_nll = float((expected_loss * mask).sum().item() / mask.sum().item())
    expected_smape = smape_mean(
        (target * mask).numpy().reshape(-1, 2),
        (mu * mask).numpy().reshape(-1, 2),
    )

    assert metrics["nll"] == pytest.approx(expected_nll)
    assert metrics["smape"] == pytest.approx(expected_smape)
