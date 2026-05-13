"""Kernel loader — GPU compute capability detection and binary dispatch.

Detects the current GPU architecture at import time.  Tries precompiled
cubin first (if available), then Triton/JIT compilation, then CUDA JIT
(requires nvcc), then falls back to pure PyTorch.
"""

import os
import torch


def _detect_arch():
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"


def _kernels_dir():
    return os.path.join(os.path.dirname(__file__), "kernels")


# ── Triton decode kernel loading ──

def _try_load_triton_decode_cubin():
    """Load a precompiled Triton decode cubin and return a callable.

    Returns None if no matching cubin exists or loading fails.
    """
    try:
        import triton
        from triton.runtime import driver
    except ImportError:
        return None

    arch = _detect_arch()
    if arch is None:
        return None

    cubin_path = os.path.join(_kernels_dir(), arch, "decode_packed.cubin")
    if not os.path.exists(cubin_path):
        return None

    try:
        mod, func, _ = driver.active.utils.load_binary(
            "decode_packed", cubin_path,
            driver.active.get_current_stream(),
        )
    except Exception:
        return None

    # func is a compiled kernel handle — wrap in a callable
    BLOCK = 256

    def _launch(W_p, lut):
        N = W_p.numel()
        out = torch.empty(N, dtype=torch.float16, device=W_p.device)
        grid = ((N + BLOCK - 1) // BLOCK, 1, 1)
        func(grid[0], grid[1], grid[2], BLOCK, W_p, lut, out, N)
        return out.view(W_p.shape)

    return _launch


def _try_triton_decode_jit():
    """Try the Triton JIT path — just call the kernel once to warm up.

    Returns a callable (W_p, lut) -> decoded fp16 tensor, or None.
    """
    try:
        from .kernel import _decode_packed_triton
    except ImportError:
        return None

    if _decode_packed_triton is None:
        return None

    BLOCK = 256

    def _call(W_p, lut):
        N = W_p.numel()
        out = torch.empty(N, dtype=torch.float16, device=W_p.device)
        grid = ((N + BLOCK - 1) // BLOCK,)
        _decode_packed_triton[grid](W_p, lut, out, N, BLOCK=BLOCK)
        return out.view(W_p.shape)

    return _call


def _try_cuda_decode_jit():
    """CUDA JIT via load_inline.  Requires nvcc on PATH.

    Returns a callable (W_p, lut) -> decoded fp16 tensor, or None.
    """
    try:
        from torch.utils.cpp_extension import load_inline
        from .kernel import _cuda_source, _cpp_source
    except ImportError:
        return None

    try:
        _ext = load_inline(
            name="packr_decode_jit",
            cpp_sources=_cpp_source,
            cuda_sources=_cuda_source,
            functions=["decode_packed_cuda"],
            with_cuda=True,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
    except Exception:
        return None

    def _call(W_p, lut):
        return _ext.decode_packed_cuda(W_p, lut.half())

    return _call


def _fallback_decode(W_p, lut):
    """Pure-PyTorch LUT lookup — correct but materializes full tensor."""
    return lut[W_p.long()].to(torch.float16)


def load_decode_fn():
    """Return the best available decode function.

    Priority:
      1. Precompiled Triton cubin
      2. Triton JIT (warm-up)
      3. CUDA JIT (requires nvcc)
      4. Pure-PyTorch fallback
    """
    for attempt in (_try_load_triton_decode_cubin,
                    _try_triton_decode_jit,
                    _try_cuda_decode_jit):
        try:
            fn = attempt()
            if fn is not None:
                return fn
        except Exception:
            continue
    return _fallback_decode


# ── Optimizer kernel (fused AdamW) ──

def has_precompiled_optimizer():
    arch = _detect_arch()
    if arch is None:
        return False
    path = os.path.join(_kernels_dir(), arch, "fused_adam_8bit.cubin")
    return os.path.exists(path)


def load_optimizer_cubin():
    """Load a precompiled Triton cubin for the fused AdamW kernel.

    Returns (module, function) or (None, None) on failure.
    """
    try:
        import triton
        from triton.runtime import driver
    except ImportError:
        return None, None

    arch = _detect_arch()
    if arch is None:
        return None, None

    cubin_path = os.path.join(_kernels_dir(), arch, "fused_adam_8bit.cubin")
    if not os.path.exists(cubin_path):
        return None, None

    try:
        mod, func, _ = driver.active.utils.load_binary(
            "fused_adam_8bit", cubin_path,
            driver.active.get_current_stream(),
        )
        return mod, func
    except Exception:
        return None, None


# ── Query ──

_current_arch = _detect_arch()


def kernel_status():
    return {
        "arch": _current_arch,
        "decode_precompiled": os.path.exists(
            os.path.join(_kernels_dir(), _current_arch or "none", "decode_packed.cubin")
        ),
        "optimizer_precompiled": has_precompiled_optimizer(),
    }
