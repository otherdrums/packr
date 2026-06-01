"""PackRLinearDelta — frozen base + trainable delta.

Forward:  output = x @ (base_W + delta)
             └─ frozen ─┘   └─ trainable ─┘

The base weights are frozen at pretrained values.  The delta weights are
trainable and adapt to the downstream task.

Drop-in replacement for nn.Linear.  Accepts the same input/output shapes.
"""

import torch
import torch.nn as nn


class PackRLinearDelta(nn.Module):
    """Linear layer with frozen base + trainable delta.

    GPU/VRAM:
        base_W:   [in, out] bf16    frozen pretrained weight
        delta:    [in, out] bf16    trainable delta

    Args:
        in_features:  Input dimension.
        out_features: Output dimension.
        bias:         Include bias term (default True).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

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

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs) -> 'PackRLinearDelta':
        """Create a PackRLinearDelta from an existing nn.Linear."""
        mod = cls(linear.in_features, linear.out_features, bias=linear.bias is not None)
        mod.base_W.data.copy_(linear.weight.data.T.to(torch.bfloat16))
        if linear.bias is not None:
            mod.bias.data.copy_(linear.bias.data.to(torch.bfloat16))
        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        orig_dtype = x.dtype
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])

        dev = x.device
        if x.dtype != torch.bfloat16:
            x_bf16 = x.to(torch.bfloat16)
            del x
        else:
            x_bf16 = x

        W = self.base_W.to(dev) + self.delta.to(dev)
        out = x_bf16 @ W
        if self.bias is not None:
            out = out + self.bias.to(dev)

        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)
        if len(orig_shape) == 3:
            out = out.reshape(orig_shape[0], orig_shape[1], -1)
        return out
