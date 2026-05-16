// CUDA 8-bit AdamW kernel — dtype-agnostic (bf16 + fp32).
// Prebuilt into packr._adam_8bit_cuda via setup.py (CUDAExtension).
// Falls back to load_inline JIT if .so is unavailable.

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_adam_8bit", &launch_adam_8bit,
          "8-bit AdamW optimizer step (dtype-agnostic: bf16 or fp32)");
}
