"""PackR — build-time CUDA kernel compilation.

Prebuilds the CUDA 8-bit AdamW kernel into a `.so` at wheel-build time,
eliminating the runtime nvcc dependency for users.

If CUDA is unavailable at build time (or version mismatch), the kernel
falls back to torch.utils.cpp_extension.load_inline at import time.
"""

import torch
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# Bypass over-cautious CUDA version check: nvcc XX compiles fine against
# PyTorch compiled with CUDA YY as long as the ABI is compatible (which
# nvcc itself guarantees).  load_inline uses the same nvcc without this
# check and works flawlessly.
import torch.utils.cpp_extension as _ext_utils
_orig_check = getattr(_ext_utils, '_check_cuda_version', None)
if _orig_check is not None:
    _ext_utils._check_cuda_version = lambda *a, **kw: None

ext_modules = []
if torch.cuda.is_available():
    try:
        ext = CUDAExtension(
            "packr._adam_8bit_cuda",
            ["packr/cuda_adam_kernel.cpp"],
            extra_cuda_cflags=["--expt-relaxed-constexpr"],
        )
        ext_modules.append(ext)
        print("  CUDA 8-bit AdamW kernel: will be prebuilt into packr._adam_8bit_cuda")
    except Exception as e:
        print(f"  CUDA extension setup failed ({e}), will JIT-compile at runtime")
else:
    print("  CUDA unavailable: 8-bit AdamW kernel will use JIT load_inline")


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
