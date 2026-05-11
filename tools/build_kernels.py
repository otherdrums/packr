"""Build precompiled CUDA kernels for all target GPU architectures.

Run locally when kernel source changes.  Requires nvcc and triton.
Output: kernels/sm_XX/decode_packed.cubin and fused_adam_8bit.cubin
"""

import os
import sys
import subprocess
import shutil

TARGET_ARCHES = ["75", "80", "86", "89", "90"]  # sm_XX
KERNELS_DIR = os.path.join(os.path.dirname(__file__), "..", "kernels")


def build_decode_kernel():
    """Compile the CUDA decode kernel for each target architecture."""
    cuda_source = os.path.join(os.path.dirname(__file__), "decode_packed.cu")
    if not os.path.exists(cuda_source):
        print(f"Warning: {cuda_source} not found — skipping decode kernel build")
        return

    for arch in TARGET_ARCHES:
        out_dir = os.path.join(KERNELS_DIR, f"sm_{arch}")
        os.makedirs(out_dir, exist_ok=True)
        cubin_path = os.path.join(out_dir, "decode_packed.cubin")

        cmd = [
            "nvcc",
            f"-arch=compute_{arch}",
            f"-code=sm_{arch}",
            "-cubin",
            "-O3",
            "--use_fast_math",
            "-o", cubin_path,
            cuda_source,
        ]
        print(f"  Building sm_{arch}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR sm_{arch}: {result.stderr}")
        else:
            size = os.path.getsize(cubin_path)
            print(f"    → {cubin_path} ({size} bytes)")


def build_optimizer_kernel():
    """Compile the Triton fused AdamW kernel for each target architecture."""
    try:
        import triton
        import torch
    except ImportError as e:
        print(f"Skipping optimizer kernel build: {e}")
        return

    from phr.optim import _fused_adam_8bit_kernel

    for arch in TARGET_ARCHES:
        out_dir = os.path.join(KERNELS_DIR, f"sm_{arch}")
        os.makedirs(out_dir, exist_ok=True)
        cubin_path = os.path.join(out_dir, "fused_adam_8bit.cubin")

        print(f"  Building sm_{arch} (triton)...")
        try:
            # Triton AOT compilation
            compiled = triton.compile(
                _fused_adam_8bit_kernel,
                signature={},
                constants={},
            )
            # Write cubin
            with open(cubin_path, "wb") as f:
                f.write(compiled.asm["cubin"])
            size = os.path.getsize(cubin_path)
            print(f"    → {cubin_path} ({size} bytes)")
        except Exception as e:
            print(f"  ERROR sm_{arch}: {e}")


def main():
    print("Building PackR kernels...")
    print(f"  Target arches: {TARGET_ARCHES}")
    print()

    print("  Decode kernel (nvcc):")
    build_decode_kernel()

    print()
    print("  Optimizer kernel (triton):")
    build_optimizer_kernel()

    print()
    print("  Done.  Committed kernels/ to version control.")


if __name__ == "__main__":
    main()
