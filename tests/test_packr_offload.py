"""Offload tests — W_p streaming, eviction, re-fetch."""

import torch
import pytest
from packr import PackRConfig, compress_model


class TestPackROffload:
    def test_offload_forward_shape(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        config = PackRConfig(layer_scope="all", offload=True,
                             gradient_checkpointing=False)
        model = torch.nn.Sequential(torch.nn.Linear(64, 32, bias=False))
        model = compress_model(model, config)
        model = model.cuda()

        x = torch.randn(8, 64, device="cuda")
        out = model(x)
        assert out.shape == (8, 32)

    def test_offload_forward_backward(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        config = PackRConfig(layer_scope="all", offload=True,
                             gradient_checkpointing=False, learnable_lut=True)
        model = torch.nn.Sequential(torch.nn.Linear(64, 32, bias=False))
        model = compress_model(model, config)
        model = model.cuda()

        for _ in range(5):
            x = torch.randn(8, 64, device="cuda")
            out = model(x)
            loss = out.sum()
            loss.backward()

        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"{name} has no grad"

    def test_offload_wp_evicted(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        config = PackRConfig(layer_scope="all", offload=True,
                             gradient_checkpointing=False)
        model = torch.nn.Sequential(torch.nn.Linear(64, 32, bias=False))
        model = compress_model(model, config)
        model = model.cuda()

        x = torch.randn(8, 64, device="cuda")
        out = model(x)
        out.sum().backward()

        # After forward+backward with offload, W_p should be on CPU
        for name, m in model.named_modules():
            if hasattr(m, 'W_p'):
                assert m.W_p.device.type == "cpu", \
                    f"{name}.W_p should be on CPU after eviction, got {m.W_p.device}"

    def test_offload_multi_iteration(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        config = PackRConfig(layer_scope="all", offload=True,
                             gradient_checkpointing=False, learnable_lut=True)
        model = torch.nn.Sequential(torch.nn.Linear(128, 64, bias=False))
        model = compress_model(model, config)
        model = model.cuda()

        for i in range(20):
            x = torch.randn(8, 128, device="cuda")
            out = model(x)
            loss = out.sum()
            loss.backward()

        # After many iterations, W_p should still be on CPU (evicted)
        for name, m in model.named_modules():
            if hasattr(m, 'W_p'):
                assert m.W_p.device.type == "cpu", \
                    f"{name}.W_p leaked to GPU after {i+1} iterations"
