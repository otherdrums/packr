"""
PackR fused decode kernel + matmul (cuBLAS via torch).

Two decode backends:
  - Triton (preferred): AOT-compiled cubin loaded at init time.  No nvcc needed.
  - CUDA JIT (fallback):  load_inline compiles at first use.  Requires nvcc.
  - Pure PyTorch (final):  lut[W_p.long()].  Correct but materializes full tensor.
"""

import torch

# ---------------------------------------------------------------------------
# Triton decode kernel — AOT-compiled to cubin for release builds.
# Lives alongside the CUDA source for JIT fallback.
# ---------------------------------------------------------------------------

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _decode_packed_triton(
        W_p_ptr,       # uint8* device pointer
        lut_ptr,       # float16* device pointer
        out_ptr,       # float16* device pointer
        N,             # int32 num_elements
        BLOCK: tl.constexpr,
    ):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = idx < N
        w_p = tl.load(W_p_ptr + idx, mask=mask).to(tl.int32)
        decoded = tl.load(lut_ptr + w_p, mask=mask)
        tl.store(out_ptr + idx, decoded, mask=mask)

    _HAS_TRITON = True
except ImportError:
    _decode_packed_triton = None
    _HAS_TRITON = False

# ---------------------------------------------------------------------------
# CUDA JIT source — kept for fallback when precompiled cubins unavailable.
# ---------------------------------------------------------------------------

_cuda_source = """
#include <torch/extension.h>
#include <cuda_fp16.h>

__global__ void decode_packed_kernel(
    const uint8_t* __restrict__ W_p,
    const half* __restrict__ lut,
    half* __restrict__ decoded,
    int num_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        decoded[idx] = lut[W_p[idx]];
    }
}

torch::Tensor decode_packed_cuda(torch::Tensor W_p, torch::Tensor lut) {
    TORCH_CHECK(W_p.is_contiguous(), "W_p must be contiguous");
    auto decoded = torch::empty(W_p.sizes(), W_p.options().dtype(torch::kHalf));
    int num_elements = W_p.numel();
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    decode_packed_kernel<<<blocks, threads>>>(
        W_p.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(lut.data_ptr<at::Half>()),
        reinterpret_cast<half*>(decoded.data_ptr<at::Half>()),
        num_elements
    );
    return decoded;
}
"""

_cpp_source = "torch::Tensor decode_packed_cuda(torch::Tensor W_p, torch::Tensor lut);"

# ---------------------------------------------------------------------------
# Lazy-init decode function — tries precompiled cubins first.
# ---------------------------------------------------------------------------

_decode_fn = None


def _init_decode():
    global _decode_fn
    if _decode_fn is not None:
        return
    from ._kernel_loader import load_decode_fn
    _decode_fn = load_decode_fn()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def packr_matmul(x, W_p, W_f, lut, bias=None):
    """
    PackR matmul:  out = x @ (W_f + lut[W_p]) + bias

    Strategy:
      1. decode: LUT lookup     → decoded [K,N] fp16
      2. combine: W_f + decoded → w_full  [K,N]
      3. matmul: x @ w_full     → out     [M,N]

    Args:
        x:    [M, K] fp32 or bf16 activations
        W_p:  [K, N] uint8 byte indices (contiguous)
        W_f:  [K, N] bf16 residual weights
        lut:  [256]  fp32 lookup table
        bias: [N] optional bias

    Returns:
        [M, N] float32
    """
    assert x.is_cuda and W_p.is_cuda and W_f.is_cuda and lut.is_cuda
    assert W_p.dtype == torch.uint8, f"W_p must be uint8, got {W_p.dtype}"
    assert W_p.is_contiguous(), "W_p must be contiguous"

    M, K = x.shape
    K2, N = W_f.shape
    assert K2 == K, f"W_f rows ({K2}) != x cols ({K})"

    _init_decode()

    decoded = _decode_fn(W_p, lut)

    # 2. W_f (bf16) + decoded (fp16) → fp32 (PyTorch type promotion)
    w_full = W_f + decoded

    # 3. cuBLAS matmul — promote both to fp32 for accuracy
    if x.dtype != w_full.dtype:
        out = x.float() @ w_full
    else:
        out = x @ w_full

    if bias is not None:
        out = out + bias

    return out
