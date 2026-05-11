"""Kernel loader — GPU compute capability detection and binary dispatch.

On import, detects the current GPU architecture.  Tries precompiled cubin
first, then JIT compilation (requires nvcc), then falls back to pure
PyTorch operations that trade VRAM for portability.
"""

import os
import torch


# ── Architecture detection ──

def _detect_arch() -> str | None:
    """Return sm_XX string for the current CUDA device, or None if no GPU."""
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"


# ── Decode kernel (LUT lookup) ──

def _has_precompiled_decode(arch: str) -> bool:
    """Check if a precompiled decode kernel exists for the given arch."""
    base = os.path.dirname(__file__)
    cubin = os.path.join(base, "..", "kernels", arch, "decode_packed.cubin")
    return os.path.exists(cubin)


def _load_precompiled_decode(arch: str):
    """Load a precompiled decode_packed cubin via load_inline."""
    from torch.utils.cpp_extension import load_inline

    base = os.path.dirname(__file__)
    cubin_path = os.path.join(base, "..", "kernels", arch, "decode_packed.cubin")

    if not os.path.exists(cubin_path):
        raise FileNotFoundError(f"No precompiled kernel for {arch}")

    with open(cubin_path, "rb") as f:
        cubin_data = f.read()

    return load_inline(
        name=f"packr_decode_{arch}",
        cpp_sources="torch::Tensor decode_packed_cuda(torch::Tensor W_p, torch::Tensor lut);",
        functions=["decode_packed_cuda"],
        extra_ldflags=[f"--cubin={cubin_path}"],
        with_cuda=True,
    )


def _jit_compile_decode():
    """JIT-compile the decode kernel from source (requires nvcc on PATH)."""
    from kernel import _cuda_source, _cpp_source
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="packr_decode_jit",
        cpp_sources=_cpp_source,
        cuda_sources=_cuda_source,
        functions=["decode_packed_cuda"],
        with_cuda=True,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )


# ── Optimizer kernel (fused AdamW) ──

def _has_precompiled_optimizer(arch: str) -> bool:
    """Check if a precompiled optimizer cubin exists for the given arch."""
    base = os.path.dirname(__file__)
    cubin = os.path.join(base, "..", "kernels", arch, "fused_adam_8bit.cubin")
    return os.path.exists(cubin)


def _load_precompiled_optimizer(arch: str):
    """Load a precompiled Triton cubin for the fused AdamW kernel."""
    import triton
    from triton.runtime import driver

    base = os.path.dirname(__file__)
    cubin_path = os.path.join(base, "..", "kernels", arch, "fused_adam_8bit.cubin")

    if not os.path.exists(cubin_path):
        raise FileNotFoundError(f"No precompiled optimizer kernel for {arch}")

    # Load cubin into Triton's runtime
    _, mod, func, _ = driver.active.load_binary("fused_adam_8bit", cubin_path, driver.active.get_current_stream())
    return mod, func


# ── Module-level state ──

_current_arch = _detect_arch()
_decode_ext = None
_adam_cubin = None

HAS_PRECOMPILED = _current_arch is not None and _has_precompiled_decode(_current_arch)
HAS_NVCC = False  # set True if JIT succeeds


# ── Initialization ──

def _init_decode():
    """Lazy-initialize the decode kernel extension.  Called on first use."""
    global _decode_ext, HAS_NVCC

    if _decode_ext is not None:
        return

    arch = _current_arch

    # 1. Try precompiled cubin
    if arch is not None and _has_precompiled_decode(arch):
        try:
            _decode_ext = _load_precompiled_decode(arch)
            return
        except Exception:
            pass

    # 2. Try JIT compilation (needs nvcc)
    try:
        _decode_ext = _jit_compile_decode()
        HAS_NVCC = True
        return
    except Exception:
        pass

    # 3. Fallback — will use _decode_fallback at call time
    _decode_ext = None


def _decode_fallback(W_p, lut):
    """Pure-PyTorch LUT lookup — correct but materializes full weight tensor.

    Used when no precompiled binary matches the GPU and nvcc is unavailable.
    """
    return lut[W_p.long()]


def get_decode_fn():
    """Return the best available decode function.

    Returns:
        callable: (W_p, lut) -> decoded float tensor
    """
    _init_decode()

    if _decode_ext is not None:
        def _precompiled(W_p, lut):
            return _decode_ext.decode_packed_cuda(W_p, lut.half())
        return _precompiled

    return _decode_fallback


# ── Query ──

def kernel_status() -> dict:
    """Return a human-readable dict describing the kernel loading state."""
    return {
        "arch": _current_arch,
        "precompiled_available": HAS_PRECOMPILED,
        "nvcc_available": HAS_NVCC,
        "decode_loaded": _decode_ext is not None or HAS_PRECOMPILED,
    }
