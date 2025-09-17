from __future__ import annotations

import contextlib
import importlib
import types
from typing import Dict, Iterable, Optional, Sequence

import torch
from torch import nn, Tensor


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


def _load_error_classes(paths: Sequence[str]) -> Sequence[type]:
    classes = []
    for path in paths:
        module_name, _, attr = path.rpartition(".")
        if not module_name:
            continue
        try:
            mod = importlib.import_module(module_name)
            cls = getattr(mod, attr)
        except Exception:
            continue
        else:
            if isinstance(cls, type):
                classes.append(cls)
    return classes


_COMPILE_ERROR_PREFIXES: Sequence[str] = ("torch._dynamo", "torch._inductor")
_COMPILE_ERROR_TYPES: Sequence[type] = _load_error_classes(
    [
        "torch._dynamo.exc.BackendCompilerFailed",
        "torch._dynamo.exc.TorchRuntimeError",
        "torch._dynamo.exc.InternalTorchDynamoError",
        "torch._dynamo.exc.Unsupported",
        "torch._inductor.exc.BackendCompilerFailed",
    ]
)


def _iter_error_chain(exc: BaseException) -> Iterable[BaseException]:
    seen: set[int] = set()
    stack = [exc]
    while stack:
        current = stack.pop()
        ident = id(current)
        if ident in seen:
            continue
        seen.add(ident)
        yield current
        if current.__cause__ is not None:
            stack.append(current.__cause__)
        if current.__context__ is not None:
            stack.append(current.__context__)


def _is_compile_error(exc: BaseException) -> bool:
    for err in _iter_error_chain(exc):
        if any(isinstance(err, cls) for cls in _COMPILE_ERROR_TYPES):
            return True
        module_name = getattr(err.__class__, "__module__", "")
        if any(module_name.startswith(prefix) for prefix in _COMPILE_ERROR_PREFIXES):
            return True
    return False


def _wrap_compiled_module(compiled: nn.Module, eager: nn.Module) -> nn.Module:
    state = {"use_compiled": True, "warned": False}
    eager_module = getattr(compiled, "_orig_mod", eager)
    orig_call = compiled.__call__
    orig_forward = compiled.forward

    def _handle_error(err: BaseException) -> bool:
        if state["use_compiled"] and _is_compile_error(err):
            state["use_compiled"] = False
            if not state["warned"]:
                print(
                    f"[WARN] torch.compile execution failed: {err}. Fallback to eager.")
                state["warned"] = True
            return True
        return False

    def call_with_fallback(self: nn.Module, *args, **kwargs):
        if not state["use_compiled"]:
            return eager_module(*args, **kwargs)
        try:
            return orig_call(*args, **kwargs)
        except Exception as err:  # pragma: no cover - runtime path
            if _handle_error(err):
                return eager_module(*args, **kwargs)
            raise

    def forward_with_fallback(self: nn.Module, *args, **kwargs):
        if not state["use_compiled"]:
            return eager_module(*args, **kwargs)
        try:
            return orig_forward(*args, **kwargs)
        except Exception as err:  # pragma: no cover - runtime path
            if _handle_error(err):
                return eager_module(*args, **kwargs)
            raise

    compiled.__call__ = types.MethodType(call_with_fallback, compiled)
    compiled.forward = types.MethodType(forward_with_fallback, compiled)
    return compiled


def maybe_compile(
    module: nn.Module,
    enabled: bool,
    warmup_args: Optional[Sequence] = None,
    warmup_kwargs: Optional[Dict] = None,
) -> nn.Module:
    if not enabled:
        return module
    try:
        torch._dynamo.config.capture_scalar_outputs = True
        compiled_module = torch.compile(module, fullgraph=False)
    except Exception as e:
        # graceful fallback
        print(f"[WARN] torch.compile failed: {e}. Fallback to eager.")
        return module

    wrapped = _wrap_compiled_module(compiled_module, module)

    if warmup_args or warmup_kwargs:
        args = tuple(warmup_args or ())
        kwargs = dict(warmup_kwargs or {})
        try:
            with torch.no_grad():
                wrapped(*args, **kwargs)
        except Exception as err:  # pragma: no cover - runtime path
            if _is_compile_error(err):
                # ``wrapped`` already falls back to eager; suppress the error.
                pass
            else:
                raise

    return wrapped


def clean_state_dict(state: Dict[str, Tensor]) -> Dict[str, Tensor]:
    return {
        k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v
        for k, v in state.items()
    }


def move_to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device, non_blocking=True)
