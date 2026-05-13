"""PackR core tests — forward, backward, gradient flow, and kernel correctness."""

import torch
import pytest
from packr import PackRConfig, compress_model


def _needs_cuda(fn):
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required"
    )(fn)


class TestPackRForward:
    def test_forward_shape(self, device):
        config = PackRConfig(layer_scope="all", gradient_checkpointing=False)
        model = torch.nn.Sequential(torch.nn.Linear(64, 32, bias=False))
        model = compress_model(model, config)
        model = model.to(device)
        x = torch.randn(8, 64, device=device)
        out = model(x)
        assert out.shape == (8, 32)

    def test_forward_matches_nn_linear(self, device):
        config = PackRConfig(layer_scope="all", gradient_checkpointing=False)
        # Create fresh nn.Linear before compression
        original = torch.nn.Linear(64, 32, bias=False)
        w_original = original.weight.data.clone()
        model = compress_model(original, config)
        model = model.to(device)

        # Reference: same weight, standard nn.Linear
        nn_ref = torch.nn.Linear(64, 32, bias=False).to(device)
        nn_ref.weight.data.copy_(w_original.to(device))

        x = torch.randn(8, 64, device=device)
        out_packr = model(x)
        out_nn = nn_ref(x)

        rel_err = (out_packr - out_nn).abs().max() / out_nn.abs().max()
        assert rel_err < 0.05, f"Relative error {rel_err:.4f} exceeds 5%"

    def test_forward_3d_input(self, device):
        config = PackRConfig(layer_scope="all", gradient_checkpointing=False)
        model = torch.nn.Sequential(torch.nn.Linear(64, 32, bias=False))
        model = compress_model(model, config)
        model = model.to(device)
        x = torch.randn(2, 4, 64, device=device)
        out = model(x)
        assert out.shape == (2, 4, 32)

    def test_forward_with_bias(self, device):
        config = PackRConfig(layer_scope="all", gradient_checkpointing=False)
        model = torch.nn.Sequential(torch.nn.Linear(64, 32, bias=True))
        model = compress_model(model, config)
        model = model.to(device)
        x = torch.randn(8, 64, device=device)
        out = model(x)
        assert out.shape == (8, 32)

    @_needs_cuda
    def test_forward_all_cuda(self):
        device = torch.device("cuda")
        self.test_forward_shape(device)


class TestPackRBackward:
    def test_gradient_flow(self, device):
        config = PackRConfig(layer_scope="all", gradient_checkpointing=False,
                             learnable_lut=True)
        model = torch.nn.Sequential(torch.nn.Linear(64, 32, bias=False))
        model = compress_model(model, config)
        model = model.to(device)
        x = torch.randn(8, 64, device=device)
        out = model(x)
        loss = out.sum()
        loss.backward()

        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"{name} has no grad"
                assert not torch.isnan(p.grad).any(), f"{name} grad has NaN"

    def test_lut_gradient_exists(self, device):
        config = PackRConfig(layer_scope="all", gradient_checkpointing=False,
                             learnable_lut=True)
        model = torch.nn.Sequential(torch.nn.Linear(64, 32, bias=False))
        model = compress_model(model, config)
        model = model.to(device)
        x = torch.randn(8, 64, device=device)
        out = model(x)
        loss = out.sum()
        loss.backward()

        # lut should receive gradient
        lut_grad = model[0].lut.grad
        assert lut_grad is not None
        assert lut_grad.shape == (256,)
        assert not torch.isnan(lut_grad).any()

    def test_gradient_consistency(self, device):
        """Verify that two forward+backward passes produce non-identical gradients."""
        config = PackRConfig(layer_scope="all", gradient_checkpointing=False,
                             learnable_lut=True)
        model = torch.nn.Sequential(torch.nn.Linear(64, 32, bias=False))
        model = compress_model(model, config)
        model = model.to(device)

        x1 = torch.randn(8, 64, device=device)
        out1 = model(x1)
        out1.sum().backward()
        grad1 = model[0].lut.grad.clone()

        model.zero_grad()
        x2 = torch.randn(8, 64, device=device)
        out2 = model(x2)
        out2.sum().backward()
        grad2 = model[0].lut.grad.clone()

        # Different inputs should produce different LUT gradients
        assert not torch.allclose(grad1, grad2), "Different inputs produced identical LUT gradients"

    @_needs_cuda
    def test_gradient_flow_cuda(self):
        device = torch.device("cuda")
        self.test_gradient_flow(device)


class TestPackRMultiLayer:
    def test_two_layer_forward(self, device):
        config = PackRConfig(layer_scope="all", gradient_checkpointing=False)
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 32, bias=False),
            torch.nn.Linear(32, 16, bias=False),
        )
        model = compress_model(model, config)
        model = model.to(device)
        x = torch.randn(8, 64, device=device)
        out = model(x)
        assert out.shape == (8, 16)

    def test_two_layer_backward(self, device):
        config = PackRConfig(layer_scope="all", gradient_checkpointing=False,
                             learnable_lut=True)
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 32, bias=False),
            torch.nn.Linear(32, 16, bias=False),
        )
        model = compress_model(model, config)
        model = model.to(device)
        x = torch.randn(8, 64, device=device)
        out = model(x)
        loss = out.sum()
        loss.backward()

        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"{name} has no grad"
