"""Ablation: measure how fixed at attenuation ceilings affect SST-2 accuracy.

For each max_atten level (0.0 to 1.0), clamps ALL rows to that attenuation
(no hash computation), trains BERT-base for 500 steps, reports eval accuracy
at every 100 steps.  Isolates the attenuation → accuracy relationship
from any hash artifacts.

Usage:
    python tools/ablate_atten_ceiling.py
    python tools/ablate_atten_ceiling.py --levels 0.0 0.3 0.5 0.7 1.0
    python tools/ablate_atten_ceiling.py --steps 300

Output: runs/ablate_<timestamp>.pkl  +  console table
"""

import os
import sys
import time
import pickle
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from packr.config import PackRConfig
from packr.layer_patcher import compress_model
from packr.cuda_adam import CUDA8BitAdam
from packr.zpackr_layer import ZPackRLinear


def main():
    parser = argparse.ArgumentParser(
        description="Ablation: measure max_atten ceiling vs SST-2 accuracy"
    )
    parser.add_argument("--levels", type=float, nargs="*",
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument("--steps", type=int, default=500,
                        help="Training steps per level")
    parser.add_argument("--eval-interval", type=int, default=100,
                        help="Steps between eval checkpoints")
    parser.add_argument("--eval-steps", type=int, default=10,
                        help="Batches per eval (fewer = faster)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("ERROR: CUDA required")
        sys.exit(1)

    # ── Shared setup (reused across levels) ──
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
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

    eval_ds = raw["validation"].map(
        lambda x: tokenizer(x["sentence"], truncation=True,
                            padding="max_length", max_length=128),
        batched=True, remove_columns=cols,
    )
    eval_ds.set_format("torch",
                       columns=["input_ids", "attention_mask",
                                "token_type_ids", "label"])
    eval_loader = torch.utils.data.DataLoader(
        eval_ds, batch_size=32, shuffle=False
    )

    pcfg = PackRConfig(mode="zpackr", bf16=True,
                       hash_interval=1, optimizer_type="cuda8",
                       gradient_mix=0.0)

    # BN + LN patch for bf16
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

    # ── Run ablation for each level ──
    results = {}  # max_atten -> list of (step, accuracy)
    total_est = len(args.levels) * args.steps

    print(f"\nAblation: {len(args.levels)} levels × {args.steps} steps = {total_est} total steps")
    print(f"Levels: {args.levels}")
    print()

    for li, max_atten in enumerate(args.levels):
        torch.manual_seed(42)

        # --- Fresh model ---
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2,
        )
        print(f"[{li+1}/{len(args.levels)}] max_atten={max_atten:.1f}  loading+compressing...", end=" ")

        model = compress_model(model, pcfg)
        model = model.to(torch.bfloat16)
        model = model.to(device)

        # Collect ZPackRLinear layers
        zpl_layers = [
            (n, m) for n, m in model.named_modules()
            if isinstance(m, ZPackRLinear)
        ]

        # Set fixed attenuation for ALL rows
        fixed_byte = int(max_atten * 255)
        for _, module in zpl_layers:
            module._atten_byte.fill_(fixed_byte)
        print(f"atten_byte={fixed_byte}")

        # Optimizer
        optim = CUDA8BitAdam(model.parameters(), lr=2e-5)

        model.train()
        train_iter = iter(loader)
        step_accs = []

        step_start_ts = time.time()

        for step in range(args.steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(loader)
                batch = next(train_iter)

            labels = batch.pop("label", None).to(device)
            bg = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**bg, labels=labels)
            loss = outputs.loss
            loss.backward()
            optim.step()
            optim.zero_grad()

            # ── Eval ──
            if (step + 1) % args.eval_interval == 0:
                model.eval()
                all_preds = []
                all_labels = []
                for ei, ebatch in enumerate(eval_loader):
                    if ei >= args.eval_steps:
                        break
                    elabels = ebatch.pop("label", None)
                    ebatch_gpu = {k: v.to(device) for k, v in ebatch.items()}
                    with torch.no_grad():
                        eout = model(**ebatch_gpu)
                    preds = eout.logits.argmax(dim=-1).cpu().numpy()
                    all_preds.extend(preds)
                    if elabels is not None:
                        all_labels.extend(elabels.cpu().numpy())
                acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
                step_accs.append((step + 1, acc))
                model.train()

                # Estimate remaining time
                steps_done = li * args.steps + step + 1
                elapsed = time.time() - step_start_ts
                total_elapsed = elapsed / steps_done * total_est
                remaining = total_elapsed * (1 - steps_done / total_est)
                print(f"    step {step+1:>3d}  acc={acc:.4f}  "
                      f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remain)")

        results[max_atten] = step_accs

    # ── Save ──
    out = {
        "config": {
            "levels": args.levels,
            "steps": args.steps,
            "eval_interval": args.eval_interval,
            "eval_steps": args.eval_steps,
        },
        "results": {str(k): v for k, v in results.items()},
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"runs/ablate_{ts}.pkl"
    os.makedirs("runs", exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out, f)

    # ── Print table ──
    print()
    print("=" * 80)
    print("ABLATION RESULTS: max_atten ceeling vs SST-2 eval accuracy")
    print("=" * 80)
    header = f"{'max_atten':>10s}"
    for s in range(args.eval_interval, args.steps + 1, args.eval_interval):
        header += f"  step={s:>4d}"
    print(header)
    print("-" * len(header))
    for at in args.levels:
        row = f"{at:>10.1f}"
        acc_map = dict(results[at])
        for s in range(args.eval_interval, args.steps + 1, args.eval_interval):
            acc = acc_map.get(s, -1)
            row += f"  {acc:>8.4f}" if acc >= 0 else f"  {'---':>8s}"
        print(row)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
