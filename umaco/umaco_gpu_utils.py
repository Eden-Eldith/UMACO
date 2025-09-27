"""Centralized GPU utility helpers for UMACO modules.

This module provides a single point of truth for:
- Selecting the GPU/CPU array backend (CuPy vs NumPy)
- Guarding execution against incomplete CUDA installations
- Converting arrays/scalars between CuPy and NumPy when needed

All scripts should import from here instead of duplicating `_resolve_gpu_backend`
implementations. This keeps GPU fallbacks consistent and makes the "GPU-first"
policy explicit.
"""
from __future__ import annotations

import logging
import os
from types import ModuleType
from typing import Tuple

import numpy as np

logger = logging.getLogger("UMACO-GPU")

try:  # Prefer CuPy
    import cupy as _cupy  # type: ignore
    _cupy_available = True
except ImportError:  # pragma: no cover - environment dependent
    _cupy = None
    _cupy_available = False

cp: ModuleType | np.ndarray = _cupy if _cupy_available else np  # type: ignore[assignment]
GPU_AVAILABLE: bool = bool(_cupy_available)


def ensure_cupy_runtime_ready(module_name: str = "umaco_gpu_utils") -> None:
    """Validate the CUDA runtime and keep global state in sync."""
    global cp, GPU_AVAILABLE

    allow_cpu = os.getenv("UMACO_ALLOW_CPU", "0") == "1"
    module_logger = logging.getLogger(module_name)

    if not GPU_AVAILABLE:
        if allow_cpu:
            module_logger.warning(
                "CuPy is unavailable; running in CPU compatibility mode because UMACO_ALLOW_CPU=1."
            )
            cp = np
            return
        raise RuntimeError(
            "UMACO requires CuPy for GPU execution. Install cupy-cudaXX (matching your CUDA "
            "version) or set UMACO_ALLOW_CPU=1 to acknowledge CPU fallback."
        )

    try:
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count < 1:
            raise RuntimeError("No CUDA-capable device detected")
        cp.cuda.nvrtc.getVersion()
    except Exception as exc:  # pragma: no cover - environment dependent
        if not allow_cpu:
            raise RuntimeError(
                "UMACO detected a CUDA runtime issue (likely missing nvrtc). Install the matching "
                "CUDA toolkit or add its bin directory to PATH, or set UMACO_ALLOW_CPU=1 to "
                "explicitly allow CPU execution."
            ) from exc
        module_logger.warning(
            "CuPy runtime issue detected (%s); falling back to NumPy compatibility mode because "
            "UMACO_ALLOW_CPU=1.",
            exc,
        )
        cp = np
        GPU_AVAILABLE = False


def resolve_gpu_backend(module_name: str = "umaco_gpu_utils") -> Tuple[ModuleType, bool]:
    """Return the preferred array module (CuPy or NumPy) plus availability flag."""
    global cp, GPU_AVAILABLE

    allow_cpu = os.getenv("UMACO_ALLOW_CPU", "0") == "1"
    module_logger = logging.getLogger(module_name)

    if not GPU_AVAILABLE:
        if allow_cpu:
            module_logger.warning(
                "CuPy is not installed; running in NumPy compatibility mode because UMACO_ALLOW_CPU=1."
            )
            cp = np
            return np, False
        raise RuntimeError(
            f"{module_name} requires CuPy for GPU execution. Install cupy-cudaXX (matching your CUDA "
            "version) or set UMACO_ALLOW_CPU=1 to acknowledge CPU fallback."
        )

    try:
        cp.cuda.runtime.getDeviceCount()
        cp.cuda.nvrtc.getVersion()
    except Exception as exc:  # pragma: no cover - environment dependent
        if not allow_cpu:
            raise RuntimeError(
                "CuPy is installed but the CUDA runtime is unhealthy (missing nvrtc or device). "
                "Install the matching CUDA toolkit or set UMACO_ALLOW_CPU=1 to override."
            ) from exc
        module_logger.warning(
            "CUDA runtime issue detected (%s); running in NumPy compatibility mode because "
            "UMACO_ALLOW_CPU=1.",
            exc,
        )
        cp = np
        GPU_AVAILABLE = False
        return np, False

    return cp, True


def asnumpy(arr):
    """Convert a CuPy array to NumPy without touching GPU data otherwise."""
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def to_numpy_scalar(val):
    """Convert a CuPy scalar to a Python/NumPy float."""
    if hasattr(val, "item"):
        return float(val.item())
    if hasattr(val, "get"):
        return float(val.get())
    return float(val)
