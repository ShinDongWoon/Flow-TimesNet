import sys
from pathlib import Path

import torch

# Ensure the project src is on the path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast import train


def _cuda_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    class _CudaDevice:
        type = "cuda"

    return _CudaDevice()


def test_cuda_graphs_disabled_when_compile_enabled():
    train_cfg = {
        "cuda_graphs": True,
        "compile": True,
    }
    device = _cuda_device()

    assert not train._should_use_cuda_graphs(train_cfg, device)


def test_cuda_graphs_enabled_without_compile():
    train_cfg = {
        "cuda_graphs": True,
        "compile": False,
    }
    device = _cuda_device()

    assert train._should_use_cuda_graphs(train_cfg, device)
