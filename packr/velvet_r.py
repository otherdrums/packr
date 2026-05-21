"""VelvetR — per-row LSH dual-signal attenuation controller.

Uses TWO LSH signals per row:
  - delta hash:  direction stability of the delta weight (position convergence)
  - gradient hash: direction stability of the gradient (learning signal SNR)

Attenuation is a geometric mix of both:
  atten = delta_sim ** (1 - mix) * (1 - grad_sim) ** mix

Delta alone can't distinguish "converged" from "stuck" (both have stable
positions).  Gradient alone can't distinguish "learning" from "converged"
(converged gradient is noise = unstable).  Together they form a complete
signal: attenuation rises only when BOTH agree the row is done.
"""

import torch
import torch.nn as nn
import numpy as np

import triton
import triton.language as tl

BLOCK_SIZE = 256

# Multi-scale comparison offsets (in steps) — logarithmic, 3x spacing
LSH_OFFSETS = (1, 3, 10, 30, 100, 300, 1000)

# Exponential weights for far-offset dominance.
# Near offsets (1, 3) are always ~1.0 since consecutive hashes barely change;
# they dominate the mean and mask the far-offset signal.  These weights
# suppress near offsets and amplify far ones so attenuation reflects
# true long-term convergence, not short-term direction consistency.
# Weights grow as 2^{idx}, aligned with log-spaced offsets.
LSH_WEIGHTS = (1, 1, 2, 4, 8, 16, 32)


@triton.jit
def _lsh_hash_fused_kernel(
    delta_ptr,     # [in_features, out_features] bf16 row-major
    proj_ptr,      # [K, out_features] bf16 row-major
    hash_ptr,      # [in_features, K] uint8 output
    in_features,
    out_features,
    K: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_OUT)
    acc = tl.zeros([K], dtype=tl.float32)

    for start in range(0, out_features, BLOCK_OUT):
        o = start + offs
        mask = o < out_features
        d = tl.load(delta_ptr + row * out_features + o, mask=mask).to(tl.float32)
        for k in range(K):
            p = tl.load(proj_ptr + k * out_features + o, mask=mask).to(tl.float32)
            dot = tl.sum(d * p)
            acc = tl.where(tl.arange(0, K) == k, acc + dot, acc)

    tl.store(hash_ptr + row * K + tl.arange(0, K), (acc > 0).to(tl.uint8))


class DeltaSignatureDB:
    """Sliding window of LSH hashes for per-row convergence detection.

    Each step, all delta (or gradient) rows are LSH-hashed via Triton kernel and
    appended to a ring buffer stored in pinned CPU memory.  Multi-scale comparison
    against past hashes produces the weighted mean of cosine similarities.

    Used TWICE per VelvetR instance: one for delta hashes (position stability)
    and one for gradient hashes (learning signal SNR).  The two signals are mixed
    geometrically into the final attenuation.

    The window lives on CPU (pinned for async GPU<->CPU transfers) to save
    ~369MB of GPU VRAM.  Only the needed offsets (~644KB) are transferred
    to GPU during compute_attenuation.
    """

    _projection_cache: dict = {}

    def __init__(self, num_rows: int, K: int = 16, window_size: int = 4200, seed: int = 42):
        self.K = K
        self.bytes_per_hash = K // 8
        self.num_rows = num_rows
        self._window_size = window_size
        self._window_cpu = torch.zeros(window_size, num_rows, self.bytes_per_hash,
                                       dtype=torch.uint8, pin_memory=True)
        self._cursor = 0
        self._count = 0

    def prefill(self, target_similarity: torch.Tensor):
        assert target_similarity.shape == (self.num_rows,)
        S = target_similarity.float().numpy()
        S = np.clip(S, 0, 1)
        sum_target = 255 * (1.0 - S)
        byte0_target = sum_target / 2.0
        rng = np.random.default_rng(42)
        jitter = rng.uniform(-5, 5, size=(self._window_size, len(S)))

        for pos in range(self._window_size):
            b0 = np.clip(np.round(byte0_target + jitter[pos]), 0, 255).astype(np.uint8)
            b1 = np.clip(np.round(sum_target - b0.astype(np.float64)
                                  + rng.uniform(-5, 5, size=len(S))), 0, 255).astype(np.uint8)
            self._window_cpu[pos, :, 0] = torch.from_numpy(b0)
            self._window_cpu[pos, :, 1] = torch.from_numpy(b1)
        self._cursor = 0
        self._count = self._window_size

    @classmethod
    def get_gpu_projections(cls, out_features: int, K: int = 64, seed: int = 42) -> torch.Tensor:
        key = (K, out_features)
        if key not in cls._projection_cache:
            gen = torch.Generator().manual_seed(seed)
            proj = torch.randn(K, out_features, generator=gen)
            proj = proj / proj.norm(dim=1, keepdim=True)
            cls._projection_cache[key] = proj.to(torch.float32).cuda()
        return cls._projection_cache[key]

    def hash_rows(self, delta: torch.Tensor) -> torch.Tensor:
        in_f, out_f = delta.shape
        proj = self.get_gpu_projections(out_f, self.K)
        if delta.device.type == 'cpu':
            proj = proj.cpu()
            result = delta.float() @ proj.t()
            bits = (result > 0).to(torch.uint8).cuda()
        else:
            bits = torch.empty(in_f, self.K, dtype=torch.uint8, device='cuda')
            grid = (in_f,)
            _lsh_hash_fused_kernel[grid](
                delta, proj, bits, in_f, out_f,
                K=self.K, BLOCK_OUT=BLOCK_SIZE,
            )
        bits_view = bits.view(in_f, self.bytes_per_hash, 8)
        weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device='cuda', dtype=torch.uint8)
        return (bits_view * weights).sum(dim=2).to(torch.uint8)

    def push(self, hashes: torch.Tensor):
        self._window_cpu[self._cursor].copy_(hashes, non_blocking=True)
        self._cursor = (self._cursor + 1) % self._window_size
        self._count = min(self._count + 1, self._window_size)

    def compute_attenuation(self, current_hashes: torch.Tensor) -> torch.Tensor:
        count = self._count
        indices, wl = [], []
        for i, off in enumerate(LSH_OFFSETS):
            if off > count:
                break
            indices.append((self._cursor - off) % self._window_size)
            wl.append(LSH_WEIGHTS[i])
        if len(indices) == 0:
            return torch.zeros(self.num_rows, device='cuda')
        stored_slices = [self._window_cpu[i].cuda(non_blocking=True) for i in indices]
        stored = torch.stack(stored_slices).float()
        current = current_hashes.unsqueeze(0).float()
        diff = (current - stored).abs()
        byte_sim = 1.0 - diff / 255.0
        matching = byte_sim.mean(dim=2)
        cos_sim = 2 * matching - 1
        weights_t = torch.tensor(wl, device='cuda', dtype=torch.float32)
        attenuation = (cos_sim * weights_t.unsqueeze(1)).sum(dim=0) / weights_t.sum()
        return torch.clamp(attenuation, 0.0, 1.0)


class VelvetRController:
    """Per-row LSH dual-signal attenuation controller.

    Tracks two signals per row over a sliding window:
      - delta hash: weight position stability
      - gradient hash: gradient direction consistency

    Attenuation = geometric mix of both.  Rises only when BOTH signals
    agree the row is converged (stable position + stable gradient direction).

    Usage:
        controller = VelvetRController(num_rows=768)
        # After backward(), before optimizer.step():
        atten = controller.update(delta_weight, gradient)
    """

    def __init__(self, num_rows: int, K: int = 16, window_size: int = 4200,
                 gradient_mix: float = 0.5, grad_ema_beta: float = 0.9967):
        self.num_rows = num_rows
        self._gradient_mix = gradient_mix
        self._grad_ema_beta = grad_ema_beta
        self._hash_counter = 0
        self._hash_interval = 1

        self._sig_db = DeltaSignatureDB(num_rows=num_rows, K=K, window_size=window_size)
        self._grad_sig_db = DeltaSignatureDB(num_rows=num_rows, K=K, window_size=window_size)

        self.register_buffer('_atten_byte',
            torch.zeros(num_rows, dtype=torch.uint8))
        self.register_buffer('_grad_sim',
            torch.zeros(num_rows, dtype=torch.float32))
        self.register_buffer('_gradient_avg',
            torch.zeros(num_rows, dtype=torch.bfloat16))

    def register_buffer(self, name, tensor):
        self.__dict__[name] = tensor

    @torch.no_grad()
    def update_gradient_hash(self, gradient: torch.Tensor):
        """Update gradient EMA, hash the EMA, update grad_sim.

        Call after backward() and before optimizer.step()/zero_grad().
        """
        if gradient is None:
            self._grad_sim.zero_()
            return
        beta = self._grad_ema_beta
        self._gradient_avg.copy_(beta * self._gradient_avg + (1.0 - beta) * gradient)
        current_hashes = self._grad_sig_db.hash_rows(self._gradient_avg)
        grad_sim = self._grad_sig_db.compute_attenuation(current_hashes)
        self._grad_sim.copy_(grad_sim)
        self._grad_sig_db.push(current_hashes)

    @torch.no_grad()
    def update_delta_hash(self, delta_weight: torch.Tensor):
        """Hash delta weights and update _atten_byte from both signals.

        Call after optimizer.step().
        """
        self._hash_counter += 1
        if self._hash_counter < self._hash_interval:
            return
        self._hash_counter = 0

        current_hashes = self._sig_db.hash_rows(delta_weight)
        delta_sim = self._sig_db.compute_attenuation(current_hashes)
        self._sig_db.push(current_hashes)

        mix = self._gradient_mix
        attenuation = delta_sim.pow(1.0 - mix) * (1.0 - self._grad_sim).pow(mix)
        self._atten_byte.copy_((attenuation * 255).to(dtype=torch.uint8))

    @torch.no_grad()
    def prefill(self, target_similarity: torch.Tensor):
        """Pre-fill both LSH windows with synthetic hashes."""
        self._sig_db.prefill(target_similarity)
        self._grad_sig_db.prefill(target_similarity)
        self._hash_counter = 0
        self._atten_byte.copy_((target_similarity * 255).to(dtype=torch.uint8))

    def get_attenuation(self) -> torch.Tensor:
        return self._atten_byte.float() / 255.0
