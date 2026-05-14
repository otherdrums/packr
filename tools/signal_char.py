"""Signal characterization — measure zstd ratio creep rate, noise floor, persistence.

Tracks per-block zstd ratios over 500 steps on a fixed batch.
Measures:
  - Creep rate (slope of ratio vs step)
  - Noise floor (std dev of ratio between steps)
  - Persistence (does creep continue or plateau?)
  - Layer variance (early vs late layers)
  - Gradient vs no-gradient (is creep from delta change or compressor?)

Output: signal_char.jsonl in runs/
"""

import os, sys, json, time
import torch, numpy as np
import zstandard as zstd
from collections import defaultdict

sys.path.insert(0, "/home/otherdrums/packr")

from packr.config import PackRConfig
from packr.layer_patcher import compress_model
from packr.optim import FusedQuantizedAdam
from packr.zpackr_layer import ZPackRLinear

torch.manual_seed(42)
device = torch.device("cuda")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

# Setup
print("Loading...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
config = PackRConfig(mode="zpackr", layer_scope="ffn")
model = compress_model(model, config).to(device)

layers = [(n.replace("bert.encoder.", "enc."), m) for n, m in model.named_modules() if isinstance(m, ZPackRLinear)]
print(f"  {len(layers)} layers")

dataset = load_dataset("glue", "sst2", split="train")
dataset = dataset.map(lambda ex: tokenizer(ex["sentence"], truncation=True, padding="max_length", max_length=128), batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])

optimizer = FusedQuantizedAdam(model.parameters(), lr=2e-5, block_size=256)
cctx = zstd.ZstdCompressor(level=1)

batch = dataset[:16]
labels = batch["label"].to(device)
bg = {k: v.to(device) for k, v in batch.items() if k != "label"}

# === Phase 1: Training with gradients (300 steps) ===
print("\n=== Phase 1: Training (gradients on, 300 steps) ===")
history = defaultdict(lambda: defaultdict(list))  # {layer_name: {blk: [ratios]}}

for step in range(300):
    model.train()
    loss = model(**bg, labels=labels).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Measure every 5 steps to see fine-grained creep
    if step % 5 == 0 or step == 0:
        for name, module in layers:
            module._sync_full_delta()
            dn = module._full_delta.view(torch.uint8).contiguous().view(-1).numpy()
            beb = module.block_size * module.out_features * 2
            for blk in range(module.num_blocks):
                bs = blk * beb; be = min(bs + beb, dn.nbytes)
                if be <= bs: continue
                r = len(dn[bs:be]) / max(len(cctx.compress(dn[bs:be].tobytes())), 1)
                history[name][blk].append((step, r))

        # Show first layer + last layer
        f_name, f_mod = layers[0]
        l_name, l_mod = layers[-1]
        f_rs = [f"{history[f_name][b][-1][1]:.6f}" for b in range(f_mod.num_blocks)]
        l_rs = [f"{history[l_name][b][-1][1]:.6f}" for b in range(l_mod.num_blocks)]
        if step <= 20 or step % 50 == 0:
            print(f"  Step {step:>4}: L0={f_rs}  L11={l_rs}")

# === Phase 2: No gradients (100 steps) — measure compressor-only noise ===
print("\n=== Phase 2: No gradients (compressor noise floor, 100 steps) ===")
no_grad_history = defaultdict(lambda: defaultdict(list))

for step in range(100):
    model.eval()
    with torch.no_grad():
        _ = model(**bg, labels=labels)
    # No optimizer step — delta stays frozen
    # Measure ratio (should be pure compressor noise)
    if step % 5 == 0:
        for name, module in layers:
            module._sync_full_delta()
            dn = module._full_delta.view(torch.uint8).contiguous().view(-1).numpy()
            beb = module.block_size * module.out_features * 2
            for blk in range(module.num_blocks):
                bs = blk * beb; be = min(bs + beb, dn.nbytes)
                if be <= bs: continue
                r = len(dn[bs:be]) / max(len(cctx.compress(dn[bs:be].tobytes())), 1)
                no_grad_history[name][blk].append((step, r))

# === Analysis ===
print("\n=== Analysis ===")

for name, mod in layers:
    for blk in range(mod.num_blocks):
        train_pts = history[name][blk]
        if not train_pts: continue

        # Creep rate: slope of ratio vs step (linear regression on last 50% of points)
        n = len(train_pts)
        mid = n // 2
        steps = np.array([p[0] for p in train_pts[mid:]])
        ratios = np.array([p[1] for p in train_pts[mid:]])
        if len(steps) < 2: continue

        # Linear fit
        A = np.vstack([steps, np.ones(len(steps))]).T
        slope, intercept = np.linalg.lstsq(A, ratios, rcond=None)[0]
        creep_pct_per_step = slope / ratios.mean() * 100

        # Noise floor from phase 2
        ng_pts = no_grad_history[name].get(blk, [])
        if ng_pts:
            ng_ratios = np.array([p[1] for p in ng_pts])
            noise_std = ng_ratios.std()
        else:
            noise_std = 0

        # Only log if creep is detectable
        if abs(creep_pct_per_step) > 1e-8:
            # Signal-to-noise
            snr = abs(slope) / max(noise_std, 1e-12) if noise_std > 0 else float('inf')

            # Print per-block stats
            print(f"  {name:40s} blk{blk}: ratio={ratios[-1]:.6f} "
                  f"creep={creep_pct_per_step:+.6f}%/step "
                  f"noise_std={noise_std:.8f} "
                  f"SNR={snr:.1f} "
                  f"total_creep={(ratios[-1]-ratios[0])/ratios[0]*100:+.4f}%")

# Overall stats
all_slopes = []
for name, mod in layers:
    for blk in range(mod.num_blocks):
        pts = history[name][blk]
        if len(pts) < 2: continue
        mid = len(pts) // 2
        steps = np.array([p[0] for p in pts[mid:]])
        ratios = np.array([p[1] for p in pts[mid:]])
        A = np.vstack([steps, np.ones(len(steps))]).T
        slope = np.linalg.lstsq(A, ratios, rcond=None)[0][0]
        all_slopes.append(slope / ratios.mean() * 100)

if all_slopes:
    arr = np.array(all_slopes)
    print(f"\n  Overall creep: median={np.median(arr):+.6f}%/step "
          f"mean={arr.mean():+.6f}%/step "
          f"std={arr.std():.6f}%/step")

print("\nDone.")
