# PackR — Compressed neural network weights with learnable codebooks

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
- **Velvet adaptive scheduler** — reads AdamW second-moment velocity in real time,
  adjusts per-layer learning rates without hand-tuned schedules
- **Drop-in replacement** — `compress_model(model)` converts any HuggingFace model

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

Velvet (Velocity to Learning Rate Translation) reads the AdamW second-moment
buffer (`exp_avg_sq`) every optimizer step and dynamically adjusts per-layer
learning rates.  An EMA low-pass filter separates signal from micro-batch noise.

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

## Offloading

Stream frozen `W_p` indices and optimizer states from pinned system RAM:

```python
config = PackRConfig(offload=True)
model = compress_model(model, config)
# Training loop unchanged — offloading is transparent
```

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
