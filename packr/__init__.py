"""
PackR — Packed Residual for memory-efficient neural network training.

Usage:
    from packr import compress_model, PackRConfig

    config = PackRConfig()
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    model = compress_model(model, config)
    # train normally with standard PyTorch / HuggingFace loop
"""

from .kernel import packr_matmul
from .autograd import PackRMatmulFunction
from .layer import PackRLinear
from .linear_delta import PackRLinearDelta
from .layer_patcher import compress_model
from .config import PackRConfig
from .optim import FusedQuantizedAdam

__all__ = [
    "PackRConfig",
    "PackRLinear",
    "PackRLinearDelta",
    "PackRMatmulFunction",
    "packr_matmul",
    "compress_model",
    "FusedQuantizedAdam",
]
