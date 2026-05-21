"""Precompute per-row LSH initial novelty and pre-fill hash windows.

For each unique token in the training data, measures novelty from normalized
inverse frequency.  Runs a single epoch through frozen BERT, capturing hidden
state magnitudes at each FFN layer input.  Aggregates per-dim novelty:
dims that carry large magnitudes for frequent tokens get high initial similarity
(low novelty).  Dims that carry large magnitudes for rare tokens get low initial
similarity (high novelty).

Output: vocab/row_novelty.pt — dict {layer_name: [in_features] float32 target similarity}
"""

import sys, os, torch
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Token novelty from frequency ──
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    raw = load_dataset("glue", "sst2")
    cols = [c for c in raw["train"].column_names if c != "label"]

    print("Counting token frequencies...")
    freq = defaultdict(int)
    train = raw["train"].map(
        lambda x: tokenizer(x["sentence"], truncation=True, padding="max_length", max_length=128),
        batched=True, remove_columns=cols,
    )
    train.set_format("torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])
    for ex in train:
        for tid in ex["input_ids"]:
            freq[tid] += 1

    max_freq = max(freq.values()) if freq else 1
    token_novelty = {tid: 1.0 - min(1.0, count / max_freq)
                     for tid, count in freq.items()}
    print(f"  Unique tokens in data: {len(freq)}")

    # ── Hook frozen BERT hidden states ──
    print("Loading frozen BERT...")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",
                                                               num_labels=2,
                                                               output_hidden_states=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Per-layer accumulator
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    layer_novelty = [torch.zeros(hidden_size, dtype=torch.float32, device=device)
                     for _ in range(num_layers)]
    layer_counts = [torch.zeros(hidden_size, dtype=torch.float32, device=device)
                    for _ in range(num_layers)]

    print(f"  Processing training data ({len(train)} examples)...")
    batch_size = 32
    loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True, drop_last=True
    )

    import time
    t0 = time.time()
    processed = 0

    for batch in loader:
        ids = batch["input_ids"].to(device)
        attn = batch.get("attention_mask", torch.ones_like(ids)).to(device)
        tt = batch.get("token_type_ids", torch.zeros_like(ids)).to(device)

        with torch.no_grad():
            outputs = model(input_ids=ids, attention_mask=attn, token_type_ids=tt)

        for layer_idx in range(num_layers):
            h = outputs.hidden_states[layer_idx]
            h_abs = h.abs()

            for b in range(ids.shape[0]):
                for s in range(ids.shape[1]):
                    tid = ids[b, s].item()
                    nv = token_novelty.get(tid, 1.0)
                    mag = h_abs[b, s]
                    layer_novelty[layer_idx] += mag * (1.0 - nv)
                    layer_counts[layer_idx] += mag

        processed += ids.shape[0]
        if processed % 500 == 0:
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            print(f"    {processed} / {len(train)} examples ({rate:.0f}/s)")

    # ── Compute per-dim target similarity ──
    row_novelty = {}
    for layer_idx in range(num_layers):
        avg_novelty = layer_novelty[layer_idx] / (layer_counts[layer_idx] + 1e-8)
        target_sim = 1.0 - avg_novelty.clamp(0, 1)
        row_novelty[f"layer.{layer_idx}.intermediate"] = target_sim.cpu()

    print(f"\nPer-layer stats:")
    for layer_idx in range(num_layers):
        t = row_novelty[f"layer.{layer_idx}.intermediate"]
        print(f"  layer {layer_idx:>2d}: mean_sim={t.mean():.4f}  std={t.std():.4f}  "
              f"min={t.min():.4f}  max={t.max():.4f}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "vocab")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "row_novelty.pt")
    torch.save(row_novelty, out_path)
    print(f"\nSaved to {out_path}")
    elapsed = time.time() - t0
    print(f"Elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
