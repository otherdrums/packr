"""PackRLinearDelta — frozen base + trainable delta + VelvetR per-row attenuation.

Forward:  output = x @ (base_W + delta * (1 - attenuation))
             └─ frozen ─┘   └─ trainable ─┘

The base weights are frozen at pretrained values.  The delta weights are
trainable and adapt to the downstream task.  VelvetR controls per-row
attenuation, ramping up attenuation on converged rows and keeping active
rows at full capacity.

Drop-in replacement for nn.Linear.  Accepts the same input/output shapes.
"""

ATTENUATION_SKIP_THRESHOLD = 1.0
"""Gate threshold: if ALL rows across ALL layers have atten >= this,
backward can be skipped for the current step."""

import torch
import torch.nn as nn
from .velvet_r import VelvetRController


class PackRLinearDelta(nn.Module):
    """Linear layer with frozen base + VelvetR-attenuated trainable delta.

    GPU/VRAM:
        base_W:   [in, out] bf16    frozen pretrained weight
        delta:    [in, out] bf16    trainable delta
        velvet_r: VelvetRController  per-row LSH attenuation

    Args:
        in_features:  Input dimension.
        out_features: Output dimension.
        bias:         Include bias term (default True).
        gradient_mix: Blend between delta LSH and gradient LSH signals
                      (0=delta only, 1=gradient only, default 0.5).
        lsh_K:        Number of random LSH projections (default 16).
        lsh_window:   LSH sliding window length (default 4200).
        hash_interval: Steps between delta hash computations (default 1).
        grad_ema_beta: EMA smoothing for gradient hashing (default 0.9967).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 gradient_mix: float = 0.5, lsh_K: int = 16,
                 lsh_window: int = 4200, hash_interval: int = 1,
                 grad_ema_beta: float = 0.9967):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._hash_interval = hash_interval
        self._hash_counter = 0

        self.base_W = nn.Parameter(
            torch.zeros(in_features, out_features, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.delta = nn.Parameter(
            torch.zeros(in_features, out_features, dtype=torch.bfloat16),
            requires_grad=True,
        )
        self.bias = nn.Parameter(
            torch.zeros(out_features, dtype=torch.bfloat16)
        ) if bias else None

        self.velvet_r = VelvetRController(
            num_rows=in_features,
            K=lsh_K,
            window_size=lsh_window,
            gradient_mix=gradient_mix,
            grad_ema_beta=grad_ema_beta,
        )

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs) -> 'PackRLinearDelta':
        """Create a PackRLinearDelta from an existing nn.Linear."""
        mod = cls(linear.in_features, linear.out_features, bias=linear.bias is not None, **kwargs)
        mod.base_W.data.copy_(linear.weight.data.to(torch.bfloat16))
        if linear.bias is not None:
            mod.bias.data.copy_(linear.bias.data.to(torch.bfloat16))
        return mod

    @torch.no_grad()
    def compute_grad_hash(self):
        """Call after backward(), before optimizer.step()."""
        grad = self.delta.grad
        if grad is not None:
            self.velvet_r.update_gradient_hash(grad)

    @torch.no_grad()
    def compute_delta_hash(self):
        """Call after optimizer.step()."""
        self.velvet_r.update_delta_hash(self.delta)

    @torch.no_grad()
    def post_step(self):
        """Convenience: update delta hash + ratio cache."""
        self.compute_delta_hash()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        orig_dtype = x.dtype
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])

        if x.dtype != torch.bfloat16:
            x_bf16 = x.to(torch.bfloat16)
            del x
        else:
            x_bf16 = x

        nv = self.velvet_r.get_attenuation().to(torch.bfloat16).unsqueeze(1)
        W = self.base_W.to(x_bf16.device) + self.delta * (1.0 - nv)
        out = x_bf16 @ W
        if self.bias is not None:
            out = out + self.bias

        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)
        if len(orig_shape) == 3:
            out = out.reshape(orig_shape[0], orig_shape[1], -1)
        return out
