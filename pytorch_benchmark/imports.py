# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/00_imports.ipynb (unless otherwise specified).

__all__ = ['is_tpu_available', 'is_bf16_available', 'is_multigpu_available']

# Cell
import sys, operator, torch
from packaging.version import Version, parse

if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

# Cell
try:
    import torch_xla.core.xla_model as xm  # noqa: F401

    _tpu_available = True
except ImportError:
    _tpu_available = False

# Cell
_torch_version = parse(importlib_metadata.version("torch"))

# Cell
def is_tpu_available(check_device=True) -> bool:
    "Checks if `torch_xla` is installed and potentially if a TPU is in the environment"
    if _tpu_available and check_device:
        try:
            # Will raise a RuntimeError if no XLA configuration is found
            _ = xm.xla_device()
            return True
        except RuntimeError:
            return False
    return _tpu_available

# Cell
def is_bf16_available(ignore_tpu=False):
    "Checks if bf16 is supported, optionally ignoring the TPU"
    if is_tpu_available(): return not ignore_tpu
    if operator.ge(_torch_version, Version("1.10")):
        if torch.cuda.is_available():
            return torch.cuda.is_bf16_supported()
        return True
    return False

# Cell
def is_multigpu_available() -> bool:
    "Checks if number of cuda devices available > 1"
    return torch.cuda.is_available() and torch.cuda.device_count() > 1