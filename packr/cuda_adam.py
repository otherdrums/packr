"""CUDA 8-bit AdamW — dtype-agnostic, int8 m/v, per-param launch.

Handles bf16 and fp32 params/grads via an is_bf16 flag.

Two loading paths:
1. Prebuilt: ``packr._adam_8bit_cuda`` — built by ``setup.py`` (CUDAExtension)
   at wheel-build time.  No nvcc required at runtime.
2. JIT fallback: ``torch.utils.cpp_extension.load_inline`` — compiles the
   embedded CUDA source on first import.  Requires nvcc on PATH.
"""

import logging
import torch

_cuda_mod = None

def _get_cuda_mod():
    global _cuda_mod
    if _cuda_mod is not None:
        return _cuda_mod

    # 1. Try prebuilt .so (from CUDAExtension in setup.py)
    try:
        from packr._adam_8bit_cuda import launch_adam_8bit
        _cuda_mod = type("CudaMod", (), {"launch_adam_8bit": staticmethod(launch_adam_8bit)})()
        return _cuda_mod
    except (ImportError, OSError):
        pass

    # 2. Fall back to JIT inline compilation (requires nvcc)
    from torch.utils.cpp_extension import load_inline

    CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

static __global__ void adam_8bit_kernel(
    void* p_raw, void* g_raw,
    signed char* m, signed char* v,
    float* ms, float* vs,
    int N, int is_bf16,
    float lr, float b1, float b2, float eps,
    float bc1, float bc2, float wd
) {
    int bid = blockIdx.x;
    int idx = blockIdx.x * 256 + threadIdx.x;

    float pv = 0, gv = 0, mf = 0, vf = 0;
    if (idx < N) {
        if (is_bf16) {
            unsigned short pu = ((unsigned short*)p_raw)[idx];
            unsigned short gu = ((unsigned short*)g_raw)[idx];
            pv = __uint_as_float((unsigned int)pu << 16);
            gv = __uint_as_float((unsigned int)gu << 16);
        } else {
            pv = ((float*)p_raw)[idx];
            gv = ((float*)g_raw)[idx];
        }
        mf = (float)m[idx] * ms[bid];
        vf = (float)v[idx] * vs[bid];
    }

    if (wd > 0) pv -= lr * wd * pv;

    float mn = b1 * mf + (1 - b1) * gv;
    float vn = b2 * vf + (1 - b2) * gv * gv;

    __shared__ float sm[32], sv[32];
    float ma = fabsf(mn), va = fabsf(vn);
    for (int o = 16; o > 0; o >>= 1) { ma = fmaxf(ma, __shfl_xor_sync(-1, ma, o)); va = fmaxf(va, __shfl_xor_sync(-1, va, o)); }
    int w = threadIdx.x / 32, l = threadIdx.x % 32;
    if (l == 0) { sm[w] = ma; sv[w] = va; }
    __syncthreads();
    if (w == 0) {
        ma = threadIdx.x < 8 ? sm[l] : 0; va = threadIdx.x < 8 ? sv[l] : 0;
        for (int o = 16; o > 0; o >>= 1) { ma = fmaxf(ma, __shfl_xor_sync(-1, ma, o)); va = fmaxf(va, __shfl_xor_sync(-1, va, o)); }
        if (threadIdx.x == 0) { ms[bid] = fmaxf(ma / 127.0f, 1e-14f); vs[bid] = fmaxf(va / 127.0f, 1e-14f); }
    }
    __syncthreads();

    float nms = ms[bid], nvs = vs[bid];
    signed char mi = (signed char)fminf(fmaxf((mn >= 0 ? mn / nms + 0.5f : mn / nms - 0.5f), -127.0f), 127.0f);
    signed char vi = (signed char)fminf(fmaxf((vn / nvs + 0.5f), 1.0f), 127.0f);
    float pn = pv - lr * (mn / bc1) / (sqrtf(vn / bc2) + eps);

    if (idx < N) {
        if (is_bf16)
            ((unsigned short*)p_raw)[idx] = (unsigned short)(__float_as_uint(pn) >> 16);
        else
            ((float*)p_raw)[idx] = pn;
        m[idx] = mi;
        v[idx] = vi;
    }
}

void launch_adam_8bit(
    torch::Tensor p, torch::Tensor g,
    torch::Tensor m, torch::Tensor v,
    torch::Tensor ms, torch::Tensor vs,
    int is_bf16,
    float lr, float b1, float b2, float eps,
    float bc1, float bc2, float wd
) {
    int N = p.numel();
    adam_8bit_kernel<<<(N + 255) / 256, 256>>>(
        p.data_ptr(), g.data_ptr(),
        (signed char*)m.data_ptr<int8_t>(),
        (signed char*)v.data_ptr<int8_t>(),
        ms.data_ptr<float>(), vs.data_ptr<float>(),
        N, is_bf16,
        lr, b1, b2, eps, bc1, bc2, wd);
}
'''
    CPP_SOURCE = r'''
#include <torch/extension.h>
void launch_adam_8bit(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, float, float, float, float, float, float, float);
'''

    _cuda_mod = load_inline(
        name='adam_8bit',
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        functions=['launch_adam_8bit'],
        verbose=False,
        extra_cuda_cflags=['--expt-relaxed-constexpr'],
    )
    return _cuda_mod


def _init_state(state, p):
    """Initialize int8 m/v + per-block float32 scales for one param."""
    N = p.numel()
    num_blocks = (N + 255) // 256
    state["m"] = torch.zeros(N, dtype=torch.int8, device=p.device)
    state["v"] = torch.zeros(N, dtype=torch.int8, device=p.device)
    state["m_scale"] = torch.ones(num_blocks, dtype=torch.float32, device=p.device)
    state["v_scale"] = torch.ones(num_blocks, dtype=torch.float32, device=p.device)


class CUDA8BitAdam(torch.optim.Optimizer):
    """8-bit AdamW with per-block int8 quantization.

    Dtype-agnostic: handles bf16 and fp32 params/grads automatically.

    Args:
        params:      iterable of parameters
        lr:          learning rate
        betas:       (beta1, beta2)
        eps:         epsilon
        weight_decay: AdamW-style weight decay
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._step_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()

        self._step_count += 1
        step = self._step_count
        cuda_mod = _get_cuda_mod()

        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            bc1 = 1.0 - b1 ** step
            bc2 = 1.0 - b2 ** step

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "m" not in state:
                    _init_state(state, p)

                if p.dtype == torch.bfloat16:
                    # sm_75 (Turing) mis-handles the kernel's is_bf16=1 path
                    # (aliased unsigned short -> float reads).  Workaround:
                    # create ephemeral fp32 copies, use the correct fp32 path,
                    # copy result back.  Temporaries freed each step.
                    p_f32 = p.data.float()
                    g_f32 = p.grad.data.float()
                    cuda_mod.launch_adam_8bit(
                        p_f32, g_f32,
                        state["m"], state["v"],
                        state["m_scale"], state["v_scale"],
                        0,
                        lr, b1, b2, eps, bc1, bc2, wd,
                    )
                    p.data.copy_(p_f32.to(torch.bfloat16))
                else:
                    cuda_mod.launch_adam_8bit(
                        p, p.grad,
                        state["m"], state["v"],
                        state["m_scale"], state["v_scale"],
                        0,
                        lr, b1, b2, eps, bc1, bc2, wd,
                    )

        return None
