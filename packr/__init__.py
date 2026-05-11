"""
PackR — Packed Residual for memory-efficient neural network training.

Usage:
    from packr import compress_model, PackRConfig

    config = PackRConfig(scheme="phr", learnable_lut=True, offload=True)
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    model = compress_model(model, config)
    # train normally with standard PyTorch / HuggingFace loop
"""

from .kernel import packr_matmul
from .autograd import PackRMatmulFunction
from .layer import PackRLinear
from .layer_patcher import compress_model
from .config import PackRConfig, SchemeType
from .optim import FusedQuantizedAdam
from .offload import OffloadManager
from .velvet import VelvetController

# Legacy aliases for backward compatibility with phr-era code
PHRConfig = PackRConfig
PHRLinear = PackRLinear
PHRMatmulFunction = PackRMatmulFunction
phr_matmul = packr_matmul
CV2LRTController = VelvetController

__all__ = [
    "PackRConfig",
    "PackRLinear",
    "PackRMatmulFunction",
    "packr_matmul",
    "compress_model",
    "SchemeType",
    "FusedQuantizedAdam",
    "OffloadManager",
    "VelvetController",
    # Legacy
    "PHRConfig",
    "PHRLinear",
    "PHRMatmulFunction",
    "phr_matmul",
    "CV2LRTController",
]
