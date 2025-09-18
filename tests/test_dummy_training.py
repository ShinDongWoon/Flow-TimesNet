import math
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest
import torch

# Ensure the project src is on the path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.config import Config
from timesnet_forecast import train as train_module
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
        n_layers=2,
        k_periods=2,
        pmax=input_len,
        kernel_set=[3],
        dropout=0.0,
        activation="gelu",
        mode="direct",
    )
    # Lazily build model parameters
    with torch.no_grad():
        _ = model(X[:1])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(30):
        idx = torch.randperm(X.size(0))
        for j in range(0, len(idx), 4):
            xb = X[idx[j : j + 4]]
            yb = Y[idx[j : j + 4]]
            optimizer.zero_grad()
            mu, sigma = model(xb)
            loss_tensor = gaussian_nll_loss(mu, sigma, yb)
            loss = loss_tensor.mean()
            loss.backward()
            optimizer.step()

    input_seq = data[60 - input_len : 60]
    actual = data[60 : 60 + pred_len]
    with torch.no_grad():
        pred_mu, pred_sigma = model(input_seq.unsqueeze(0))
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
            self.period = SimpleNamespace(pmax=1)

        def forward(
            self, x: torch.Tensor, mask: torch.Tensor | None = None
        ) -> tuple[torch.Tensor, torch.Tensor]:
            batch = x.shape[0]
            mu = self.mu_buf.expand(batch, -1, -1)
            sigma = self.sigma_buf.expand(batch, -1, -1)
            return mu, sigma

    model = DummyModel(mu, sigma)
    xb = torch.zeros((1, 3, 2), dtype=torch.float32)
    hist_mask = torch.ones_like(xb)
    loader = [(xb, target, mask, hist_mask)]
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


def test_training_warmup_uses_pmax_no_resize(tmp_path, monkeypatch):
    periods = 96
    dates = pd.date_range("2023-01-01", periods=periods, freq="D")
    t = np.arange(periods, dtype=np.float32)
    s1 = np.sin(2 * math.pi * t / 32) + 10.0
    s2 = np.cos(2 * math.pi * t / 24) + 5.0
    rows = []
    for i, d in enumerate(dates):
        rows.append({"date": d, "id": "S1", "target": float(s1[i])})
        rows.append({"date": d, "id": "S2", "target": float(s2[i])})
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "train.csv"
    df.to_csv(csv_path, index=False)

    overrides = [
        f"data.train_csv={csv_path}",
        "data.date_col=date",
        "data.id_col=id",
        "data.target_col=target",
        "data.encoding=utf-8",
        "data.fill_missing_dates=False",
        "train.device=cpu",
        "train.epochs=1",
        "train.batch_size=8",
        "train.num_workers=1",
        "train.pin_memory=False",
        "train.persistent_workers=False",
        "train.prefetch_factor=2",
        "train.amp=False",
        "train.compile=False",
        "train.cuda_graphs=False",
        "train.channels_last=True",
        "train.use_loss_masking=False",
        "train.val.strategy=holdout",
        "train.val.holdout_days=24",
        "preprocess.normalize=none",
        "preprocess.normalize_per_series=True",
        "preprocess.clip_negative=False",
        "preprocess.eps=1e-8",
        "model.mode=direct",
        "model.input_len=16",
        "model.pred_len=4",
        "model.d_model=8",
        "model.n_layers=1",
        "model.dropout=0.0",
        "model.k_periods=2",
        "model.kernel_set=[3]",
        "model.pmax_cap=64",
        "model.min_period_threshold=1",
        "train.lr=1e-3",
        "train.weight_decay=0.0",
        "train.grad_clip_norm=0.0",
        f"artifacts.dir={tmp_path / 'artifacts'}",
        "artifacts.model_file=model.pth",
        "artifacts.scaler_file=scaler.pkl",
        "artifacts.schema_file=schema.json",
        "artifacts.config_file=config.yaml",
    ]
    cfg = Config.from_files("configs/default.yaml", overrides=overrides).to_dict()

    resize_called = False

    original_resize = TimesNet._resize_frontend

    def spy_resize(self, new_in_channels, device, dtype):
        nonlocal resize_called
        resize_called = True
        return original_resize(self, new_in_channels, device, dtype)

    monkeypatch.setattr(TimesNet, "_resize_frontend", spy_resize)

    train_module.train_once(cfg)

    assert cfg["model"]["pmax"] > cfg["model"]["input_len"]
    assert not resize_called
