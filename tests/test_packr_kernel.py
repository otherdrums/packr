"""Kernel correctness tests — packr_matmul and decode backend."""

import torch
import pytest
from packr.kernel import packr_matmul


class TestPackRMatmul:
    def test_output_shape(self, device):
        x = torch.randn(4, 64, device=device)
        W_p = torch.randint(0, 256, (64, 32), dtype=torch.uint8, device=device)
        W_f = torch.randn(64, 32, dtype=torch.bfloat16, device=device)
        lut = torch.randn(256, device=device)
        out = packr_matmul(x, W_p, W_f, lut)
        assert out.shape == (4, 32)
        assert out.dtype == torch.float32

    def test_with_bias(self, device):
        x = torch.randn(4, 64, device=device)
        W_p = torch.randint(0, 256, (64, 32), dtype=torch.uint8, device=device)
        W_f = torch.randn(64, 32, dtype=torch.bfloat16, device=device)
        lut = torch.randn(256, device=device)
        bias = torch.randn(32, device=device)
        out = packr_matmul(x, W_p, W_f, lut, bias=bias)
        assert out.shape == (4, 32)

    def test_matches_nn_linear(self, device):
        """packr_matmul output approximately matches nn.Linear with same weights."""
        K, N = 64, 32
        M = 4
        W = torch.randn(K, N, device=device)

        # Build a packed representation from W
        W_abs = W.abs()
        q99 = torch.quantile(W_abs.flatten().float(), 0.99).item()
        codebook_vals = torch.linspace(-q99, q99, 256, device=device)
        diff = W_abs.unsqueeze(-1) - codebook_vals.abs().unsqueeze(0).unsqueeze(0)
        W_p = diff.abs().argmin(dim=-1).to(torch.uint8)
        W_f = (W - codebook_vals[W_p.long()]).to(torch.bfloat16)

        x = torch.randn(M, K, device=device)
        out = packr_matmul(x, W_p, W_f, codebook_vals)

        expected = (x.float() @ W.float())
        rel_err = (out - expected).abs().max() / expected.abs().max()
        assert rel_err < 0.05, f"Relative error {rel_err:.4f} exceeds 5%"

    def test_batch_input(self, device):
        """packr_matmul handles varying batch sizes."""
        for M in [1, 4, 16, 64]:
            x = torch.randn(M, 64, device=device)
            W_p = torch.randint(0, 256, (64, 32), dtype=torch.uint8, device=device)
            W_f = torch.randn(64, 32, dtype=torch.bfloat16, device=device)
            lut = torch.randn(256, device=device)
            out = packr_matmul(x, W_p, W_f, lut)
            assert out.shape == (M, 32)

    def test_contiguity_assertion(self, device):
        """Non-contiguous W_p should raise."""
        x = torch.randn(4, 64, device=device)
        W_f = torch.randn(64, 32, dtype=torch.bfloat16, device=device)
        lut = torch.randn(256, device=device)
        # Create non-contiguous W_p by slicing a larger tensor
        W_p_big = torch.randint(0, 256, (128, 64), dtype=torch.uint8, device=device)
        W_p_noncontig = W_p_big[:64, :32]  # slice produces non-contiguous view
        assert not W_p_noncontig.is_contiguous()
        with pytest.raises(AssertionError):
            packr_matmul(x, W_p_noncontig, W_f, lut)
