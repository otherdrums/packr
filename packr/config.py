"""Configuration for PackR memory-efficient training."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class PackRConfig:
    """Configuration for PackR-based memory-efficient fine-tuning.

    Args:
        layer_scope:            Which linear layers to replace
        gradient_checkpointing: Enable gradient checkpointing on the backbone
        block_size:             Quantization block size for 8-bit optimizer
        bf16:                   Convert model to bfloat16 before training
                                (saves ~100MB VRAM for BERT-base, no quality loss)
        optimizer_type:         Which optimizer to use
    """

    layer_scope: Literal["ffn", "attention", "all"] = "ffn"
    gradient_checkpointing: bool = True
    block_size: int = 256
    bf16: bool = False
    optimizer_type: Literal["triton8", "cuda8", "adamw"] = "cuda8"
