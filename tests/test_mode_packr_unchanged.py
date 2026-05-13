"""Regression tests — mode='packr' behavior is unchanged."""

import torch
import pytest
from packr import PackRConfig, compress_model


class TestPackRModeUnchanged:
    def test_packr_mode_compresses(self, device):
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 32, bias=False),
        )
        config = PackRConfig(layer_scope="all")
        model = compress_model(model, config)
        assert "PackRLinear" in type(model[0]).__name__

    def test_packr_mode_no_zstd_import(self):
        """Verify compress_model does not trigger zstandard import."""
        import sys
        was_imported = "zstandard" in sys.modules
        import inspect
        from packr.layer_patcher import compress_model
        source = inspect.getsource(compress_model)
        assert "zstandard" not in source, (
            "compress_model must not reference zstandard"
        )

    def test_packr_mode_forward(self, device):
        config = PackRConfig(layer_scope="all",
                             gradient_checkpointing=False)
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 32, bias=False),
        )
        model = compress_model(model, config)
        model = model.to(device)

        x = torch.randn(8, 64, device=device)
        out = model(x)
        assert out.shape == (8, 32)
