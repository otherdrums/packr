"""Build precompiled Triton kernel cubins for all target GPU architectures.

Run locally when kernel source changes.  Requires triton and torch with CUDA.
Output: kernels/sm_XX/decode_packed.cubin and kernels/sm_XX/fused_adam_8bit.cubin

Eliminates nvcc dependency for PyPI users — kernels are AOT-compiled here
and shipped as part of the wheel.
"""

import os
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


def main():
    print("Building PackR kernels (Triton AOT)...")
    print(f"  Target arches: {TARGET_ARCHES}")
    print()

    print("  Decode kernel:")
    build_decode_kernel()

    print()
    print("  Optimizer kernel:")
    build_optimizer_kernel()

    print()
    print("  Done.  Commit kernels/ to version control.")


if __name__ == "__main__":
    main()
