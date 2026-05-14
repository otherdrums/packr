"""zstd ratio creep — track per-block zstd ratios on same batch over 200 steps."""
import os, sys, json
import torch, numpy as np
import zstandard as zstd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from packr.config import PackRConfig
from packr.layer_patcher import compress_model
from packr.optim import FusedQuantizedAdam
from packr.zpackr_layer import ZPackRLinear

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

zpl_layers = [(n.replace("bert.encoder.", "enc."), m) for n, m in model.named_modules() if isinstance(m, ZPackRLinear)]
print(f"  Layers: {len(zpl_layers)}")

dataset = load_dataset("glue", "sst2", split="train")
dataset = dataset.map(lambda ex: tokenizer(ex["sentence"], truncation=True, padding="max_length", max_length=128), batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])

optimizer = FusedQuantizedAdam(model.parameters(), lr=2e-5, betas=(0.9, 0.999), block_size=256)
cctx = zstd.ZstdCompressor(level=1)

batch = dataset[:16]
labels = batch["label"].to(device)
batch_gpu = {k: v.to(device) for k, v in batch.items() if k != "label"}

track = {}  # {layer_name: {blk: [ratio_per_step]}}
for name, _ in zpl_layers:
    track[name] = {}

STEPS = 200
for step in range(STEPS):
    model.train()
    loss = model(**batch_gpu, labels=labels).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if step % 10 == 0:
        for name, module in zpl_layers:
            module._sync_full_delta()
            dn = module._full_delta.view(torch.uint8).contiguous().view(-1).numpy()
            beb = module.block_size * module.out_features * 2
            for blk in range(module.num_blocks):
                bs = blk * beb; be = min(bs + beb, dn.nbytes)
                if be <= bs: continue
                r = len(dn[bs:be]) / max(len(cctx.compress(dn[bs:be].tobytes())), 1)
                if blk not in track[name]:
                    track[name][blk] = []
                track[name][blk].append(r)

        # Print first layer
        first = zpl_layers[0]
        rs = [track[first[0]].get(b, [0])[-1] for b in range(first[1].num_blocks)]
        print(f"  Step {step+1:>4}: {[f'{r:.6f}' for r in rs]}")

print("\n=== Creep analysis ===")
for name in sorted(track.keys()):
    for blk in sorted(track[name].keys()):
        r = track[name][blk]
        if len(r) >= 2:
            creep = (r[-1] - r[0]) / r[0] * 100
            if abs(creep) > 0.01:
                print(f"  {name} blk{blk}: {r[0]:.6f} -> {r[-1]:.6f} ({creep:+.4f}%)")

print("\nDone.")
