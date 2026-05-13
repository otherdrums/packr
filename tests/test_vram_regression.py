"""VRAM regression tests — PackR mode stays within budget."""

import torch
import pytest
from packr import PackRConfig, compress_model

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for VRAM tests"
)


class TestPackRVRAM:
    def test_forward_peak_vram(self):
        config = PackRConfig(layer_scope="all", gradient_checkpointing=False)
        model = torch.nn.Sequential(
            torch.nn.Linear(768, 3072, bias=False),
            torch.nn.Linear(3072, 768, bias=False),
        )
        model = compress_model(model, config)
        model = model.cuda()

        x = torch.randn(4, 768, device="cuda")
        for _ in range(3):
            _ = model(x)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        for _ in range(10):
            _ = model(x)
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        assert peak_mb < 60, f"Forward VRAM {peak_mb:.1f} MB exceeds 60 MB budget"

    def test_full_step_peak_vram(self):
        config = PackRConfig(layer_scope="all", gradient_checkpointing=False)
        model = torch.nn.Sequential(
            torch.nn.Linear(768, 3072, bias=False),
            torch.nn.Linear(3072, 768, bias=False),
        )
        model = compress_model(model, config)
        model = model.cuda()

        x = torch.randn(4, 768, device="cuda")
        for _ in range(3):
            out = model(x)
            out.sum().backward()
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        for _ in range(10):
            out = model(x)
            out.sum().backward()
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        assert peak_mb < 120, f"Full-step VRAM {peak_mb:.1f} MB exceeds 120 MB budget"

    def test_single_layer_step_vram(self):
        """Single 768x3072 layer should fit well under budget."""
        config = PackRConfig(layer_scope="all", gradient_checkpointing=False)
        model = torch.nn.Sequential(torch.nn.Linear(768, 3072, bias=False))
        model = compress_model(model, config)
        model = model.cuda()

        x = torch.randn(4, 768, device="cuda")
        for _ in range(3):
            out = model(x)
            out.sum().backward()
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        out = model(x)
        out.sum().backward()
        torch.cuda.synchronize()

        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        assert peak_mb < 80, f"Single-step VRAM {peak_mb:.1f} MB exceeds 80 MB budget"


class TestPackRVRAMOffload:
    def test_offload_forward_peak_vram(self):
        config = PackRConfig(layer_scope="all", offload=True,
                             gradient_checkpointing=False)
        model = torch.nn.Sequential(
            torch.nn.Linear(768, 3072, bias=False),
            torch.nn.Linear(3072, 768, bias=False),
        )
        model = compress_model(model, config)
        model = model.cuda()

        x = torch.randn(4, 768, device="cuda")
        for _ in range(3):
            _ = model(x)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        for _ in range(10):
            _ = model(x)
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        assert peak_mb < 60, f"Offload forward VRAM {peak_mb:.1f} MB exceeds 60 MB budget"
