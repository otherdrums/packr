"""Build precompiled kernels for all target GPU architectures.

Run locally when kernel source changes.  Requires triton, torch with CUDA,
and nvcc on PATH.

Output:
  packr/kernels/sm_XX/*.cubin              Triton AOT cubins
  packr/_adam_8bit_cuda*.so                CUDA 8-bit AdamW (CUDAExtension)

Eliminates nvcc dependency for PyPI users — kernels are AOT-compiled here
and shipped as part of the wheel.
"""

import os
import subprocess
import sys


TARGET_ARCHES = [75, 80, 86, 89, 90]
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
KERNELS_DIR = os.path.join(PROJECT_ROOT, "packr", "kernels")

os.environ.setdefault("TRITON_ALLOW_AOT", "1")


def _ensure_imports():
    """Ensure the project root is on sys.path for clean imports."""
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)


def build_decode_kernel():
    """AOT-compile the Triton decode kernel for each target arch."""
    _ensure_imports()

    from packr.kernel import _decode_packed_triton
    from triton.compiler import ASTSource, compile as triton_compile
    from triton.backends.compiler import GPUTarget

    signature = {
        "W_p_ptr": "*i8",
        "lut_ptr": "*fp16",
        "out_ptr": "*fp16",
        "N": "i32",
    }
    constexprs = {"BLOCK": 256}

    src = ASTSource(_decode_packed_triton, signature, constexprs=constexprs)
    print(f"  Decode kernel source: {src.name}")

    for arch in TARGET_ARCHES:
        out_dir = os.path.join(KERNELS_DIR, f"sm_{arch}")
        os.makedirs(out_dir, exist_ok=True)
        cubin_path = os.path.join(out_dir, "decode_packed.cubin")

        target = GPUTarget("cuda", arch, 32)
        result = triton_compile(src, target=target)
        cubin_data = result.asm.get("cubin", b"")

        with open(cubin_path, "wb") as f:
            f.write(cubin_data)

        size = len(cubin_data)
        print(f"    sm_{arch}: {cubin_path} ({size} bytes)")


def build_optimizer_kernel():
    """AOT-compile the fused AdamW Triton kernel for each target arch."""
    _ensure_imports()

    from packr.optim import _fused_adam_8bit_kernel
    from triton.compiler import ASTSource, compile as triton_compile
    from triton.backends.compiler import GPUTarget

    signature = {
        "p_ptr": "*fp32",
        "g_ptr": "*fp32",
        "m_ptr": "*i8",
        "v_ptr": "*i8",
        "m_scale_ptr": "*fp32",
        "v_scale_ptr": "*fp32",
        "lr": "fp32",
        "beta1": "fp32",
        "beta2": "fp32",
        "eps": "fp32",
        "bias_correction1": "fp32",
        "bias_correction2": "fp32",
        "weight_decay": "fp32",
        "N": "i32",
    }
    constexprs = {"BLOCK": 256}

    src = ASTSource(_fused_adam_8bit_kernel, signature, constexprs=constexprs)
    print(f"  Optimizer kernel source: {src.name}")

    for arch in TARGET_ARCHES:
        out_dir = os.path.join(KERNELS_DIR, f"sm_{arch}")
        os.makedirs(out_dir, exist_ok=True)
        cubin_path = os.path.join(out_dir, "fused_adam_8bit.cubin")

        target = GPUTarget("cuda", arch, 32)
        result = triton_compile(src, target=target)
        cubin_data = result.asm.get("cubin", b"")

        with open(cubin_path, "wb") as f:
            f.write(cubin_data)

        size = len(cubin_data)
        print(f"    sm_{arch}: {cubin_path} ({size} bytes)")


def build_cuda_adam_extension():
    """Build the CUDA 8-bit AdamW .so via setup.py (CUDAExtension).

    If the CUDA version mismatch check fails (system nvcc != torch's CUDA),
    the kernel falls back to load_inline JIT at import time.

    The Triton AOT cubins above (decode, optimizer) always work regardless.
    """
    print("  CUDA 8-bit AdamW extension:")
    result = subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    if result.returncode != 0:
        err = result.stderr[-300:] if result.stderr else result.stdout[-300:]
        print(f"    setup.py build_ext failed (falls back to JIT): {err[:200]}")
        return

    # Find the built .so
    for f in os.listdir(os.path.join(PROJECT_ROOT, "packr")):
        if f.startswith("_adam_8bit_cuda") and f.endswith(".so"):
            size = os.path.getsize(os.path.join(PROJECT_ROOT, "packr", f))
            print(f"    {f} ({size} bytes)")
            break
    print("    Done.")


def main():
    print("Building PackR kernels (Triton AOT + CUDA)...")
    print(f"  Target arches: {TARGET_ARCHES}")
    print()

    print("  Decode kernel:")
    build_decode_kernel()

    print()
    print("  Optimizer kernel:")
    build_optimizer_kernel()

    print()
    build_cuda_adam_extension()

    print()
    print("  Done.  Commit kernels/ and *.so to version control.")


if __name__ == "__main__":
    main()
