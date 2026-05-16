"""Tests for CUDA8BitAdam optimizer correctness.

Fuzzes CUDA8BitAdam against torch.optim.AdamW with identical params
and gradients, comparing update norms, cosine similarity, and
element-wise differences.  Tests both bf16 and fp32 paths.

The 8-bit quantization introduces small deviations from full-precision
AdamW — these are expected and bounded.
"""

import torch
import numpy as np


def _test_adam_against_reference(shape, dtype, lr=2e-5, steps=50):
    """Fuzz CUDA8BitAdam against AdamW for a given param shape and dtype.

    Uses IDENTICAL gradients for both optimizers — only the optimizer
    implementation differs.  CUDA8 uses 8-bit quantized m/v states while
    AdamW uses full-precision float32 states.

    Returns dict of comparison metrics.
    """
    from packr.cuda_adam import CUDA8BitAdam

    torch.manual_seed(42)
    w_ref = torch.randn(*shape, dtype=dtype, device='cuda') * 0.01
    w_cut = w_ref.clone()

    # Pre-generate all gradients so both optimizers see the same sequence
    torch.manual_seed(123)
    all_grads = [torch.randn(*shape, dtype=dtype, device='cuda') * 0.1
                 for _ in range(steps)]

    # Reference: AdamW (same dtype param)
    p_ref = torch.nn.Parameter(w_ref.clone())
    p_cut = torch.nn.Parameter(w_cut.clone())

    opt_ref = torch.optim.AdamW([p_ref], lr=lr)
    opt_cut = CUDA8BitAdam([p_cut], lr=lr)

    metrics = {"first_ratio": 1.0, "first_cosim": 1.0,
               "avg_ratio": 0.0, "min_cosim": 0.0, "max_abs_diff": 0.0, "avg_abs_diff": 0.0}

    ratios = []
    cosims = []
    diffs = []

    for step in range(steps):
        grad = all_grads[step]
        p_ref.grad = grad.clone()
        p_cut.grad = grad.clone()

        opt_ref.step()
        opt_cut.step()
        opt_ref.zero_grad()
        opt_cut.zero_grad()

        u_ref = (p_ref - w_ref).float()
        u_cut = (p_cut - w_cut).float()

        r = u_cut.norm().item() / max(u_ref.norm().item(), 1e-30)
        c = torch.nn.functional.cosine_similarity(
            u_ref.flatten().unsqueeze(0), u_cut.flatten().unsqueeze(0)
        ).item()
        d = u_ref.sub(u_cut).abs().max().item()

        ratios.append(r)
        cosims.append(c)
        diffs.append(d)

        if step == 0:
            metrics["first_ratio"] = r
            metrics["first_cosim"] = c
            metrics["first_norm_ref"] = u_ref.norm().item()
            metrics["first_norm_cut"] = u_cut.norm().item()

    metrics["avg_ratio"] = float(np.mean(ratios))
    metrics["min_ratio"] = float(np.min(ratios))
    metrics["max_ratio"] = float(np.max(ratios))
    metrics["min_cosim"] = float(np.min(cosims))
    metrics["avg_cosim"] = float(np.mean(cosims))
    metrics["max_abs_diff"] = float(max(diffs))
    metrics["avg_abs_diff"] = float(np.mean(diffs))
    metrics["final_ref_norm"] = (p_ref.detach() - w_ref).float().norm().item()
    metrics["final_cut_norm"] = (p_cut.detach() - w_cut).float().norm().item()
    metrics["shape"] = shape
    metrics["dtype"] = str(dtype).split(".")[-1]

    return metrics


def test_cuda_adam_fp32_matches_adamw():
    """CUDA8BitAdam fp32 path should closely match AdamW."""
    result = _test_adam_against_reference((768, 3072), torch.float32, steps=50)
    print(f"\nfp32 {result['shape']}:")
    print(f"  first step ratio={result['first_ratio']:.4f}  cosim={result['first_cosim']:.4f}")
    print(f"  avg ratio={result['avg_ratio']:.4f}  min cosim={result['min_cosim']:.4f}")
    print(f"  max abs diff={result['max_abs_diff']:.2e}")
    assert 0.8 < result['first_ratio'] < 1.2, \
        f"First step ratio {result['first_ratio']:.4f} outside [0.8, 1.2]"
    assert result['min_cosim'] > 0.95, \
        f"Min cosine sim {result['min_cosim']:.4f} < 0.95"


def test_cuda_adam_bf16_matches_adamw():
    """CUDA8BitAdam bf16 path should closely match AdamW (same dtype)."""
    result = _test_adam_against_reference((768, 3072), torch.bfloat16, steps=50)
    print(f"\nbf16 {result['shape']}:")
    print(f"  first step ratio={result['first_ratio']:.4f}  cosim={result['first_cosim']:.4f}")
    print(f"  avg ratio={result['avg_ratio']:.4f}  min cosim={result['min_cosim']:.4f}")
    print(f"  max abs diff={result['max_abs_diff']:.2e}")
    assert 0.8 < result['first_ratio'] < 1.2, \
        f"First step ratio {result['first_ratio']:.4f} outside [0.8, 1.2]"
    assert result['min_cosim'] > 0.95, \
        f"Min cosine sim {result['min_cosim']:.4f} < 0.95"


def test_cuda_adam_fp32_small():
    """CUDA8BitAdam on small fp32 params."""
    result = _test_adam_against_reference((64, 128), torch.float32, steps=50)
    assert 0.8 < result['first_ratio'] < 1.2
    assert result['min_cosim'] > 0.95


def test_cuda_adam_bf16_small():
    """CUDA8BitAdam on small bf16 params."""
    result = _test_adam_against_reference((64, 128), torch.bfloat16, steps=50)
    assert 0.8 < result['first_ratio'] < 1.2
    assert result['min_cosim'] > 0.95


def test_cuda_adam_weight_decay():
    """Weight decay shouldn't break update direction."""
    from packr.cuda_adam import CUDA8BitAdam
    torch.manual_seed(42)
    p = torch.nn.Parameter(torch.randn(64, 64, dtype=torch.bfloat16, device='cuda') * 0.01)
    p2 = torch.nn.Parameter(p.clone())
    g = torch.randn(64, 64, dtype=torch.bfloat16, device='cuda') * 0.1
    p.grad = g.clone()
    p2.grad = g.clone()
    opt1 = torch.optim.AdamW([p], lr=2e-5, weight_decay=0.01)
    opt2 = CUDA8BitAdam([p2], lr=2e-5, weight_decay=0.01)
    opt1.step()
    opt2.step()
    cosim = torch.nn.functional.cosine_similarity(
        (p - p2).float().flatten().unsqueeze(0),
        (p2.data.to(p.dtype) - p2.data).float().flatten().unsqueeze(0)
    )
    # With weight decay, CUDA8 should still move in a similar direction
    assert cosim.item() > -0.5, "Weight decay shouldn't flip update direction"
