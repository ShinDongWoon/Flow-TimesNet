import math
from pathlib import Path
import sys

import numpy as np

# Ensure the project src is on the path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.train import _compute_pmax_global


def test_compute_pmax_global():
    T = 100
    t = np.arange(T, dtype=np.float32)
    s1 = np.sin(2 * math.pi * 5 * t / T)  # period 20
    s2 = np.sin(2 * math.pi * 2 * t / T)  # period 50
    arr = np.stack([s1, s2], axis=1)
    pmax = _compute_pmax_global([arr], k=1)
    assert pmax == 50

