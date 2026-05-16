"""Layer patcher — replaces nn.Linear layers with PackRLinear or ZPackRLinear."""

import torch.nn as nn
from .layer import PackRLinear
from .config import PackRConfig
from .offload import OffloadManager


def compress_model(model: nn.Module, config: PackRConfig = None):
    """
    Replace nn.Linear layers in a model with PackR or ZPackR compressed equivalents.

    Returns:
        model: nn.Module with PackRLinear or ZPackRLinear layers.
    """
    if config is None:
        config = PackRConfig()

    if config.mode == "zpackr":
        return _compress_zpackr(model, config)

    # ── PackR mode (default) ──
    packr_layers = []  # ordered (name, PackRLinear) for offload sequencing

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        if not _matches_scope(name, config.layer_scope):
            continue

        packr = PackRLinear.from_linear(module)
        packr.lut.requires_grad_(config.learnable_lut)

        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], packr)

        packr_layers.append((name, packr))

    if config.gradient_checkpointing:
        _enable_gradient_checkpointing(model)

    if config.offload and packr_layers:
        if next(model.parameters()).is_cpu:
            model.cuda()

        mgr = OffloadManager(prefetch_depth=1)
        layer_names = []
        for name, packr in packr_layers:
            mgr.register_wp(name, packr.W_p)
            packr.attach_offload(mgr, name)
            layer_names.append(name)
        mgr.set_layer_sequence(layer_names)
        model._offload_manager = mgr

    return model


def _compress_zpackr(model: nn.Module, config: PackRConfig):
    """Replace nn.Linear layers with ZPackRLinear (frozen base + LZ4 delta)."""
    from .zpackr_layer import ZPackRLinear

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not _matches_scope(name, config.layer_scope):
            continue

        zpackr = ZPackRLinear.from_linear(module, hash_interval=config.hash_interval,
                                          gradient_mix=config.gradient_mix,
                                          grad_ema_beta=config.grad_ema_beta)
        _replace_module(model, name, zpackr)

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
    ffn_intermediate = ["intermediate", "fc1", "mlp.up", "ffn.up", "dense_h_to_4h"]
    ffn_output = ["output.dense", "fc2", "mlp.down", "ffn.down", "dense_4h_to_h"]

    name_lower = name.lower()
    # FFN intermediate (e.g. encoder.layer.X.intermediate.dense)
    if any(m in name_lower for m in ffn_intermediate):
        return True
    # FFN output (e.g. encoder.layer.X.output.dense) — NOT attention output
    if any(m in name_lower for m in ffn_output):
        if "attention" not in name_lower:
            return True
    return False


def _is_attention(name: str) -> bool:
    """Check if a layer name corresponds to attention projection."""
    attn_markers = ["query", "key", "value", "q_proj", "k_proj", "v_proj", "o_proj", "out_proj"]
    name_lower = name.lower()
    return any(m in name_lower for m in attn_markers)


def _enable_gradient_checkpointing(model: nn.Module):
    """Enable gradient checkpointing on the backbone, if supported."""
    try:
        model.gradient_checkpointing_enable()
    except AttributeError:
        pass
