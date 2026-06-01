"""Layer patcher — replaces nn.Linear layers with PackRLinearDelta."""

import torch.nn as nn
from .linear_delta import PackRLinearDelta
from .config import PackRConfig


def compress_model(model: nn.Module, config: PackRConfig = None):
    """Replace nn.Linear layers with PackRLinearDelta (frozen base + delta)."""
    if config is None:
        config = PackRConfig()

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not _matches_scope(name, config.layer_scope):
            continue

        delta_lin = PackRLinearDelta.from_linear(module)
        _replace_module(model, name, delta_lin)

    if config.gradient_checkpointing:
        _enable_gradient_checkpointing(model)

    return model


def _replace_module(model, name, new_module):
    parent = model
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _matches_scope(name: str, scope: str) -> bool:
    """Check if a module path matches the target scope."""
    if scope == "all":
        return True
    if scope == "ffn":
        return _is_ffn(name)
    if scope == "attention":
        return _is_attention(name) and not _is_ffn(name)
    return False


def _is_ffn(name: str) -> bool:
    """Check if a layer is part of a feed-forward network."""
    name_lower = name.lower()

    # ALBERT: standalone .ffn intermediate, .ffn_output output
    if name_lower.endswith(".ffn") and not name_lower.endswith(".ffn_output"):
        return True
    if name_lower.endswith(".ffn_output"):
        if "attention" not in name_lower:
            return True

    ffn_intermediate = ["intermediate", "fc1", "mlp.up", "ffn.up", "dense_h_to_4h", "ffn.lin1"]
    ffn_output = ["output.dense", "fc2", "mlp.down", "ffn.down", "dense_4h_to_h", "ffn.lin2"]

    if any(m in name_lower for m in ffn_intermediate):
        return True
    if any(m in name_lower for m in ffn_output):
        if "attention" not in name_lower:
            return True
    return False


def _is_attention(name: str) -> bool:
    """Check if a layer name corresponds to attention projection."""
    name_lower = name.lower()
    if ".attention." in name_lower:
        return True
    attn_markers = ["query", "key", "value", "q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
                    "q_lin", "k_lin", "v_lin", "out_lin"]
    return any(m in name_lower for m in attn_markers)


def _enable_gradient_checkpointing(model: nn.Module):
    """Enable gradient checkpointing on the backbone, if supported."""
    try:
        model.gradient_checkpointing_enable()
    except (AttributeError, ValueError, RuntimeError):
        pass
