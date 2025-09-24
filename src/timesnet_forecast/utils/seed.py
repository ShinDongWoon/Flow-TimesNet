from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            # ``torch.use_deterministic_algorithms`` enforces that all kernels
            # selected by autograd are deterministic. GEMM operations backed by
            # cuBLAS require a workspace configuration hint to guarantee
            # determinism when CUDA >= 10.2 is used. Without setting the
            # ``CUBLAS_WORKSPACE_CONFIG`` environment variable PyTorch raises a
            # runtime error the first time such an operation is encountered
            # (e.g. the temporal embedding ``nn.Linear`` used when time feature
            # embeddings are enabled). Set a safe default if the user has not
            # configured it explicitly so deterministic training just works.
            workspace_env = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            if workspace_env not in {":16:8", ":4096:8"}:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=False)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)
