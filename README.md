# PackR — Packed Residual for memory-efficient neural network training

> **Warning — Early development.**  PackR is under active development and not
> yet ready for production use.  APIs and training dynamics are subject to
> change without notice.  Expect breakage, improvement, and iteration.

Drop-in `nn.Linear` replacement that stores weights as a frozen base matrix plus
a trainable bfloat16 delta — memory-efficient fine-tuning with accuracy matching
or exceeding full fine-tune.

## Features

- **Frozen base + trainable delta** — stores a full-precision base (frozen) and
  a bfloat16 delta (trainable), reducing VRAM by ~37% vs standard fp32.
- **CUDA8BitAdam** — 8-bit AdamW via hand-tuned CUDA kernel (default).
  Dtype-agnostic (bf16/fp32), warp-level reductions, 8× faster than Triton
  8-bit fallback.  Prebuilt `.so` shipped in wheel — no runtime nvcc dependency.
- **Drop-in replacement** — `compress_model(model)` converts any HuggingFace model

## Quick Start

```python
from transformers import AutoModelForSequenceClassification
from packr import compress_model, PackRConfig

config = PackRConfig(layer_scope="ffn")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model = compress_model(model, config)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

for batch in loader:
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

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
