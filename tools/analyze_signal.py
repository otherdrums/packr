"""Analyze LSH signal dynamic range during ZPackR training.

Runs 500 steps of SST-2, captures per-row attenuation and delta L2
into histograms at every step.  Outputs a pickle for plotting.

Usage:
    python tools/analyze_signal.py
    python tools/analyze_signal.py --output runs/sig.pkl --no-bf16
"""

import os
import sys
import pickle
import time
import argparse

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from packr.config import PackRConfig
from packr.layer_patcher import compress_model
from packr.cuda_adam import CUDA8BitAdam
from packr.zpackr_layer import ZPackRLinear, LSH_OFFSETS

NUM_STEPS = 500
ATTEN_BINS = 32
DELTA_BINS = 32


def main():
    parser = argparse.ArgumentParser(description="Analyze LSH signal dynamic range")
    parser.add_argument("--output", default="runs/signal_analysis.pkl",
                        help="Output pickle path")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Convert model to bfloat16")
    parser.add_argument("--no-bf16", action="store_false", dest="bf16")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("ERROR: CUDA required for Triton hash kernel")
        sys.exit(1)

    # ── Setup ──
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )

    pcfg = PackRConfig(mode="zpackr", bf16=args.bf16,
                       hash_interval=1, optimizer_type="cuda8")
    model = compress_model(model, pcfg)

    if args.bf16:
        model = model.to(torch.bfloat16)
        if not getattr(nn.LayerNorm, '_zpackr_bf16_patched', False):
            orig_ln = nn.LayerNorm.forward
            def bf16_ln(mod, inp):
                if inp.dtype == torch.bfloat16:
                    w = mod.weight.float() if mod.weight is not None else None
                    b = mod.bias.float() if mod.bias is not None else None
                    return nn.functional.layer_norm(
                        inp.float(), mod.normalized_shape, w, b, mod.eps
                    ).bfloat16()
                return orig_ln(mod, inp)
            nn.LayerNorm.forward = bf16_ln
            nn.LayerNorm._zpackr_bf16_patched = True
        print("  Converted to bfloat16")

    model = model.to(device)

    # Cache ZPackRLinear layers
    zpl_layers = [
        (name.replace("bert.encoder.", "enc."), m)
        for name, m in model.named_modules()
        if isinstance(m, ZPackRLinear)
    ]
    print(f"  {len(zpl_layers)} ZPackRLinear layers")

    optimizer = CUDA8BitAdam(model.parameters(), lr=2e-5)

    # Dataset
    from datasets import load_dataset
    raw = load_dataset("glue", "sst2")
    cols = [c for c in raw["train"].column_names if c != "label"]
    train = raw["train"].map(
        lambda x: tokenizer(x["sentence"], truncation=True,
                            padding="max_length", max_length=128),
        batched=True, remove_columns=cols,
    )
    train.set_format("torch",
                     columns=["input_ids", "attention_mask",
                              "token_type_ids", "label"])
    loader = torch.utils.data.DataLoader(
        train, batch_size=16, shuffle=True, drop_last=True
    )

    # ── Accumulators ──
    atten_hist = np.zeros((NUM_STEPS, ATTEN_BINS), dtype=np.int64)
    delta_hist = np.zeros((NUM_STEPS, DELTA_BINS), dtype=np.int64)
    offset_sims = np.full((NUM_STEPS, len(LSH_OFFSETS)), np.nan, dtype=np.float32)
    layer_atten_means = np.zeros((NUM_STEPS, len(zpl_layers)), dtype=np.float32)
    losses = np.zeros(NUM_STEPS, dtype=np.float32)
    step_ms = np.zeros(NUM_STEPS, dtype=np.float32)
    effective_lr = np.zeros(NUM_STEPS, dtype=np.float32)

    atten_edges = np.linspace(0, 256, ATTEN_BINS + 1, dtype=np.int64)
    delta_edges = np.logspace(-6, 1, DELTA_BINS + 1)

    model.train()
    train_iter = iter(loader)

    for step in range(NUM_STEPS):
        step_start = time.perf_counter()

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(loader)
            batch = next(train_iter)

        labels = batch.pop("label", None).to(device)
        batch_gpu = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch_gpu, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Hash + update attenuation
        for _, module in zpl_layers:
            module.compute_hash_gpu()

        step_t = (time.perf_counter() - step_start) * 1000

        # ── Collect per-row data ──
        all_atten = []
        all_delta = []

        for li, (_, module) in enumerate(zpl_layers):
            atten = module._atten_byte.float().cpu().numpy()
            delta_l2 = module.delta_salient.float().norm(dim=1).detach().cpu().numpy()

            all_atten.append(atten)
            all_delta.append(delta_l2)
            layer_atten_means[step, li] = atten.mean()

        all_atten = np.concatenate(all_atten)
        all_delta = np.concatenate(all_delta)

        atten_hist[step], _ = np.histogram(all_atten, bins=atten_edges)
        delta_hist[step], _ = np.histogram(all_delta, bins=delta_edges)

        # Per-offset similarities from first layer (representative)
        db = zpl_layers[0][1]._sig_db
        cursor = (db._cursor - 1) % db._window_size
        cur = db._window_cpu[cursor:cursor + 1].cuda().float()
        for oi, off in enumerate(LSH_OFFSETS):
            if off > db._count:
                break
            idx = (db._cursor - off) % db._window_size
            stored = db._window_cpu[idx:idx + 1].cuda().float()
            byte_sim = 1.0 - (cur - stored).abs() / 255.0
            matching = byte_sim.mean(dim=2)
            cos_sim = (2 * matching - 1).mean().item()
            offset_sims[step, oi] = cos_sim

        losses[step] = loss.item()
        step_ms[step] = step_t
        effective_lr[step] = (1.0 - all_atten.mean() / 255.0) * 2e-5

        if (step + 1) % 100 == 0:
            p1 = np.percentile(all_atten, 10)
            p50 = np.percentile(all_atten, 50)
            p90 = np.percentile(all_atten, 90)
            print(f"  step {step + 1:>4d}  loss={loss.item():.4f}  "
                  f"atten p10/p50/p90={p1:.0f}/{p50:.0f}/{p90:.0f}  "
                  f"eff_lr={effective_lr[step]:.2e}  {step_t:.0f}ms")

    # ── Save ──
    data = {
        "config": {
            "num_steps": NUM_STEPS,
            "atten_bins": ATTEN_BINS,
            "delta_bins": DELTA_BINS,
            "atten_edges": atten_edges.tolist(),
            "delta_edges": delta_edges.tolist(),
            "lsh_offsets": list(LSH_OFFSETS),
            "layer_names": [n for n, _ in zpl_layers],
        },
        "atten_hist": atten_hist,
        "delta_hist": delta_hist,
        "offset_sims": offset_sims,
        "layer_atten_means": layer_atten_means,
        "losses": losses,
        "step_ms": step_ms,
        "effective_lr": effective_lr,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(data, f)

    # ── Print summary ──
    print()
    print("=" * 60)
    print("SIGNAL ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"  Steps:             {NUM_STEPS}")
    print(f"  Loss:              {losses[0]:.4f} → {losses[-1]:.4f}")
    print(f"  Effective LR[0]:   {effective_lr[0]:.2e}")
    print(f"  Effective LR[-1]:  {effective_lr[-1]:.2e}")

    # Print offset similarity trajectory
    print(f"\n  Per-offset cos_sim (first layer):")
    print(f"  {'step':>5s}", end="")
    for off in LSH_OFFSETS:
        print(f"  {f'off={off}':>8s}", end="")
    print()
    for s in [0, min(49, NUM_STEPS - 1), min(99, NUM_STEPS - 1),
              min(199, NUM_STEPS - 1), min(499, NUM_STEPS - 1)]:
        if s >= NUM_STEPS:
            break
        print(f"  {s + 1:>5d}", end="")
        for oi in range(len(LSH_OFFSETS)):
            v = offset_sims[s, oi]
            if np.isnan(v):
                print(f"  {'---':>8s}", end="")
            else:
                print(f"  {v:>8.4f}", end="")
        print()

    # Print atten histogram at sampled steps
    print(f"\n  Attenuation histogram (each row is 32 bins over 0-255):")
    for s in [0, 49, 99, 199, NUM_STEPS - 1]:
        if s >= NUM_STEPS:
            break
        total = atten_hist[s].sum()
        frac = atten_hist[s] / total * 100
        # Collapse to a compact representation
        bars = ""
        for b in range(ATTEN_BINS):
            if frac[b] > 1:
                bars += "█"
            elif frac[b] > 0.1:
                bars += "▌"
            else:
                bars += "·"
        lo = np.percentile(
            np.repeat(atten_edges[:-1], atten_hist[s].astype(int)), 10
        ) if total > 0 else 0
        hi = np.percentile(
            np.repeat(atten_edges[:-1], atten_hist[s].astype(int)), 90
        ) if total > 0 else 0
        mid = np.percentile(
            np.repeat(atten_edges[:-1], atten_hist[s].astype(int)), 50
        ) if total > 0 else 0
        print(f"  step={s + 1:>4d}  p10={lo:5.0f} p50={mid:5.0f} p90={hi:5.0f}  {bars}")

    print(f"\n  Saved to {args.output}")


if __name__ == "__main__":
    main()
