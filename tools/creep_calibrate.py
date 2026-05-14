"""LZ4 ratio creep calibration — measure how ratio evolves with repeated training.

Feeds the same batch repeatedly, computes LZ4 ratios per block at every step
(no variance gating), and records the per-step % change to determine the
natural creep rate of delta compressibility.

Output: creep_calibration.jsonl with per-step per-block ratio data.
"""

import os
import sys
import json
import time
import torch
import lz4.block
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from packr.config import PackRConfig
from packr.layer_patcher import compress_model
from packr.optim import FusedQuantizedAdam
from packr.zpackr_layer import ZPackRLinear

# ── Setup ──
torch.manual_seed(42)
device = torch.device("cuda")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

config = PackRConfig(mode="zpackr", layer_scope="ffn")
model = compress_model(model, config)
model = model.to(device)

# Cache ZPackRLinear layers
zpl_layers = [
    (name.replace("bert.encoder.", "enc."), m)
    for name, m in model.named_modules()
    if isinstance(m, ZPackRLinear)
]
print(f"  ZPackRLinear layers: {len(zpl_layers)}")

# Load one batch from SST-2
print("Loading SST-2 batch...")
dataset = load_dataset("glue", "sst2", split="train")
dataset = dataset.map(
    lambda ex: tokenizer(ex["sentence"], truncation=True, padding="max_length", max_length=128),
    batched=True,
)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])

optimizer = FusedQuantizedAdam(model.parameters(), lr=2e-5, betas=(0.9, 0.999), block_size=256)

# Grab one fixed batch
batch = dataset[:16]
labels = batch["label"].to(device)
batch_gpu = {k: v.to(device) for k, v in batch.items() if k != "label"}

# ── Repeated training loop ──
STEPS = 200
output_path = "runs/creep_calibration.jsonl"
os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

with open(output_path, "w") as f_out:
    for step in range(STEPS):
        model.train()
        outputs = model(**batch_gpu, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # post_step every step, disable variance gating
        for _, module in zpl_layers:
            module._sync_full_delta()

            delta_np = module._full_delta.view(torch.uint8).contiguous().view(-1).numpy()
            block_el_bytes = module.block_size * module.out_features * 2
            block_ratios = []

            for blk in range(module.num_blocks):
                byte_start = blk * block_el_bytes
                byte_end = min(byte_start + block_el_bytes, delta_np.nbytes)
                if byte_end <= byte_start:
                    block_ratios.append(1.0)
                    continue
                blk_bytes = delta_np[byte_start:byte_end].tobytes()
                compressed = lz4.block.compress(blk_bytes, store_size=False)
                ratio = len(blk_bytes) / max(len(compressed), 1)
                block_ratios.append(ratio)

            # Store ratios for this layer
            module._block_gaps = block_ratios

        # Log per-step per-layer ratios
        log = {"step": step + 1, "layers": {}}
        for short_name, module in zpl_layers:
            ratios = module._block_gaps
            log["layers"][short_name] = {
                "ratio_max": max(ratios),
                "ratio_min": min(ratios),
                "ratio_mean": sum(ratios) / len(ratios),
                "blocks": [{"blk": i, "ratio": ratios[i]} for i in range(len(ratios))],
            }
        f_out.write(json.dumps(log) + "\n")
        f_out.flush()

        if step == 0 or (step + 1) % 20 == 0:
            print(f"  Step {step + 1}: mean_ratio={log['layers'][zpl_layers[0][0]]['ratio_mean']:.4f}")

elapsed = time.perf_counter()
print(f"\nDone. {STEPS} steps in {elapsed:.1f}s. Data: {output_path}")
