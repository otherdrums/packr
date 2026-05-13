# PackR — Compressed neural network weights with learnable codebooks

> **Warning — Early development.**  PackR is under active development and not
> yet ready for production use.  APIs and training dynamics are subject to
> change without notice.  Expect breakage, improvement, and iteration.

Drop-in `nn.Linear` replacement that stores weights as uint8 bytecode indices
into a trainable 256-entry lookup table plus bfloat16 residual deltas —
3 bytes/weight, 37% less GPU memory with accuracy matching or exceeding full fine-tune.

```bash
pip install packr
```

## Features

- **3 bytes/weight** storage (uint8 indices + bf16 residual + 256-entry LUT)
- **FusedQuantizedAdam** — 8-bit AdamW via Triton, 2 bytes/param optimizer states
  vs 8 bytes for standard AdamW
- **Fused CUDA decode kernel** — no persistent full-precision weight matrix
- **CPU/system RAM offloading** — stream frozen indices and optimizer states
  from pinned RAM to GPU on demand
- **Velvet adaptive scheduler** — closed-loop per-layer LR control; reads `exp_avg_sq`
  velocity every optimizer step, EMA-filters out micro-batch noise, dynamically
  throttles saturated layers while keeping hungry layers at full learning rate
- **Drop-in replacement** — `compress_model(model)` converts any HuggingFace model

## How It Works

PackR decomposes a weight matrix into three components:

- **W_p** (uint8, 1 byte/weight, frozen): Byte indices into a 256-entry codebook
- **W_f** (bfloat16, 2 bytes/weight, trainable): Floating-point residual
- **lut** (float32, 256 entries, trainable): Learnable codebook

Forward pass: `out = x @ (W_f + lut[W_p]) + bias`

| Representation | Persistent VRAM per weight |
|---------------|:-------------------------:|
| Standard fp32 | 4 bytes |
| Standard fp16 | 2 bytes |
| PackR | 3 bytes |

## Quick Start

```python
from transformers import AutoModelForSequenceClassification
from packr import compress_model, PackRConfig, FusedQuantizedAdam, VelvetController

# Compress FFN layers
config = PackRConfig(scheme="phr", learnable_lut=True, offload=False)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model = compress_model(model, config)
model.cuda()

# 8-bit AdamW optimizer (full beta1=0.9 momentum, 6 bytes/param saved)
optimizer = FusedQuantizedAdam(model.parameters(), lr=2e-5, betas=(0.9, 0.999))

# Velvet: adaptive per-layer LR from gradient velocity (optional)
velvet = VelvetController(optimizer, beta=0.97, min_multiplier=0.175)

# Standard PyTorch training loop — no changes
for batch in loader:
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    velvet.step()     # reads exp_avg_sq velocity, adjusts LRs
    optimizer.zero_grad()
```

## Velvet — Adaptive Per-Layer Learning Rates

Velvet (Velocity to Learning Rate Translation) replaces hand-tuned LR schedules
with real-time closed-loop adaptation.  Every optimizer step, it reads each
layer's `exp_avg_sq` (the AdamW second-moment buffer), computes the filtered
velocity of gradient variance, and translates that velocity to a per-layer LR
multiplier.

### How It Works

1. **Read**: After `optimizer.step()`, Velvet reads `exp_avg_sq` for every
   parameter.  Int8 block-quantized states (FusedQuantizedAdam) are
   dequantized automatically.

2. **Velocity**: `Δv = v_mean_current − v_mean_previous` captures whether the
   layer's gradients are still climbing (active learning) or have flattened
   (saturation).

3. **EMA filter**: Raw step-to-step velocity is noisy (SGD is stochastic).
   An exponential moving average with β=0.97 (half-life ~23 steps) separates
   signal from micro-batch jitter.

4. **Normalize**: Divide by current `v_mean` to get the relative rate of
   change — comparable across layers with different weight magnitudes.

5. **Translate**: `multiplier = clamp(min, max, |EMA_vel| / v_mean × scale)`.
   High velocity → layer is hungry → multiplier stays at 1.0 (full LR).
   Velocity → 0 → layer is saturated → multiplier decays to `min_multiplier`.

### Usage

```python
from packr import VelvetController

velvet = VelvetController(optimizer, beta=0.97, min_multiplier=0.175)

# In training loop:
for step, batch in enumerate(loader):
    loss = model(**batch).loss
    loss.backward()
    if step < warmup_steps:
        velvet.warmup_step(step, warmup_steps)
    optimizer.step()
    velvet.step()
    optimizer.zero_grad()
```

### Parameters

| Parameter | Default | Role |
|-----------|:------:|------|
| `beta` | 0.97 | EMA smoothing (higher = slower to react) |
| `min_multiplier` | 0.175 | LR floor when velocity flatlines |
| `max_multiplier` | 1.0 | LR ceiling when actively learning |
| `velocity_scale` | 10.0 | Sensitivity of velocity → multiplier mapping |

## Offloading

Stream frozen `W_p` indices and optimizer states from pinned system RAM:

```python
config = PackRConfig(offload=True)
model = compress_model(model, config)
# Training loop unchanged — offloading is transparent
```

### How It Works

Three mechanisms coordinate transparently:

- **W_p streaming** — A small GPU buffer pool reuses tensors for the current
  layer's forward pass.  Pinned CPU memory holds canonical uint8 indices.
  Synchronous default-stream copies avoid races with cuBLAS.

- **Chunked optimizer state streaming** — m/v/scales are stored as pinned CPU
  tensors grouped into ~100 MB chunks.  During `step()`, each chunk's states
  are copied to GPU via non-blocking transfers, used by the Triton kernel,
  then evicted via double-buffered offload-stream DMA overlapping with the
  next chunk's compute.

- **Automatic wiring** — `compress_model()` creates the OffloadManager and
  attaches it to every PackRLinear layer and the optimizer.  The training
  loop needs zero changes — offloading is invisible at the Python level.

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- Triton 2.1+
- nvcc 12.x (for JIT compilation on unsupported GPUs)

Precompiled kernel binaries for sm_75, sm_80, sm_86, sm_89, and sm_90 are
shipped in the wheel.  If your GPU architecture isn't covered, PackR falls
back to pure PyTorch operations (correct results, higher VRAM).

## License

MIT
