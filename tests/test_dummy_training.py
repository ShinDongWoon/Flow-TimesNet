import math
from pathlib import Path
import sys

import numpy as np
import torch

# Ensure the project src is on the path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.models.timesnet import TimesNet
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
    loss_fn = torch.nn.MSELoss()

    for _ in range(30):
        idx = torch.randperm(X.size(0))
        for j in range(0, len(idx), 4):
            xb = X[idx[j : j + 4]]
            yb = Y[idx[j : j + 4]]
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()

    input_seq = data[60 - input_len : 60]
    actual = data[60 : 60 + pred_len]
    with torch.no_grad():
        pred = model(input_seq.unsqueeze(0)).squeeze(0)

    y_true = actual.numpy()
    y_pred = pred.numpy()

    smape = smape_mean(y_true, y_pred)
    wsmape = wsmape_grouped(y_true, y_pred, ids=["A_1", "A_2"])

    assert smape < 0.1
    assert wsmape < 0.1
