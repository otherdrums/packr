"""Configuration for PackR memory-efficient training."""

from dataclasses import dataclass, field
from typing import Optional, Literal


SchemeType = Literal["phr"]
ModeType = Literal["packr", "zpackr"]


@dataclass
class PackRConfig:
    """Configuration for PackR-based memory-efficient fine-tuning.

    Args:
        mode:                   "packr" or "zpackr"
        scheme:                 Compression scheme (currently only "phr")
        learnable_lut:          Whether the LUT codebook is trainable
        layer_scope:            Which linear layers to replace
        gradient_checkpointing: Enable gradient checkpointing on the backbone
        use_8bit_optimizer:     Use FusedQuantizedAdam (Triton 8-bit Adam)
        offload:                Enable CPU/system RAM offloading
        block_size:             Quantization block size for 8-bit optimizer
        bf16:                   Convert model to bfloat16 before training
                                (saves ~100MB VRAM for BERT-base, no quality loss)
        hash_interval:          Compute LSH hash every N steps (1 = every step)
    """

    mode: ModeType = "packr"
    scheme: SchemeType = "phr"
    learnable_lut: bool = True
    layer_scope: Literal["ffn", "attention", "all"] = "ffn"
    gradient_checkpointing: bool = True
    use_8bit_optimizer: bool = True
    offload: bool = False
    block_size: int = 256
    bf16: bool = False
    hash_interval: int = 1
    optimizer_type: Literal["triton8", "cuda8", "adamw"] = "triton8"


# Legacy alias
PHRConfig = PackRConfig
