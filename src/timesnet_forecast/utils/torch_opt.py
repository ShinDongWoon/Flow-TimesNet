from __future__ import annotations

from typing import Optional
import contextlib
import torch
from torch import nn


def amp_autocast(enabled: bool):
    """
    autocast context for float16 mixed precision on CUDA.
    """
    if enabled and torch.cuda.is_available():
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def maybe_channels_last(module: nn.Module, enabled: bool) -> nn.Module:
    if enabled:
        for p in module.parameters():
            if p.is_floating_point() and p.dim() == 4:
                p.data = p.data.contiguous(memory_format=torch.channels_last)
        return module.to(memory_format=torch.channels_last)
    return module


def maybe_compile(module: nn.Module, enabled: bool) -> nn.Module:
    if not enabled:
        return module
    try:
        module = torch.compile(module, fullgraph=False)
    except Exception as e:
        # graceful fallback
        print(f"[WARN] torch.compile failed: {e}. Fallback to eager.")
    return module


def move_to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device, non_blocking=True)
