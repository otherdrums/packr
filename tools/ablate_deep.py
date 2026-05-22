"""Deep ablation: run selected max_atten levels to 8000 steps on SST-2.

Supports any HuggingFace classification model via --model.
Picks N levels from the ceiling sweep and trains each for --steps steps,
eval every --eval-interval steps.  Saves per-step loss to metrics.jsonl
per level and a combined pickle at the end.

Usage:
    python tools/ablate_deep.py --levels 0.0 0.5 0.7
    python tools/ablate_deep.py --model distilbert-base-uncased --levels 0.0 0.5 0.7 --steps 4000
"""

import os, sys, time, pickle, argparse, json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from packr.config import PackRConfig
from packr.layer_patcher import compress_model
from packr.cuda_adam import CUDA8BitAdam
from packr.linear_delta import PackRLinearDelta


def main():
    parser = argparse.ArgumentParser(description="Deep ablation: long training at selected max_atten")
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                        help="HuggingFace model name (default: bert-base-uncased)")
    parser.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7],
                        help="Attenuation ceilings to test")
    parser.add_argument("--steps", type=int, default=5000, help="Training steps per level")
    parser.add_argument("--eval-interval", type=int, default=500, help="Steps between eval")
    parser.add_argument("--eval-steps", type=int, default=50, help="Batches per eval")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", type=str, default="sst2",
                        help="GLUE task name: sst2 (default) or rte")
    parser.add_argument("--prefill", type=str, default=None,
                        help="Path to row_novelty.pt for LSH window prefill")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("ERROR: CUDA required"); sys.exit(1)

    # ── Output directory ──
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = args.model.replace("/", "_")
    root_dir = f"runs/ablate_deep_{model_slug}_{ts}"
    os.makedirs(root_dir, exist_ok=True)

    # Save config
    cfg = vars(args)
    cfg["cuda_device"] = torch.cuda.get_device_name(0)
    with open(os.path.join(root_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # ── Data ──
    from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_cfg = AutoConfig.from_pretrained(args.model)
    model_type = model_cfg.model_type
    needs_type_ids = model_type in ("bert", "bertweet", "camembert", "roberta")
    num_labels = model_cfg.num_labels if hasattr(model_cfg, "num_labels") else 2

    raw = load_dataset("glue", args.task)
    cols = [c for c in raw["train"].column_names if c != "label"]

    # Tokenization: SST-2 uses single sentence, RTE uses sentence pair
    if args.task == "rte":
        def _tokenize(batch):
            return tokenizer(batch["sentence1"], batch["sentence2"],
                             truncation=True, padding="max_length", max_length=128)
        num_labels = 2
    else:  # sst2 (default)
        def _tokenize(batch):
            return tokenizer(batch["sentence"],
                             truncation=True, padding="max_length", max_length=128)
        num_labels = 2

    torch_cols = ["input_ids", "attention_mask", "label"]
    if needs_type_ids:
        torch_cols = ["input_ids", "attention_mask", "token_type_ids", "label"]
    train = raw["train"].map(_tokenize, batched=True, remove_columns=cols)
    train.set_format("torch", columns=torch_cols)
    loader = torch.utils.data.DataLoader(train, batch_size=16, shuffle=True, drop_last=True)

    eval_ds = raw["validation"].map(_tokenize, batched=True, remove_columns=cols)
    eval_ds.set_format("torch", columns=torch_cols)
    eval_loader = torch.utils.data.DataLoader(eval_ds, batch_size=32, shuffle=False)

    pcfg = PackRConfig(mode="zpackr", bf16=True, hash_interval=1,
                       optimizer_type="cuda8", gradient_mix=0.0)

    # Optional LSH prefill
    prefill_data = None
    if args.prefill is not None:
        prefill_data = torch.load(args.prefill)

    # BN + LN bf16 patch
    if not getattr(nn.LayerNorm, '_zpackr_bf16_patched', False):
        orig_layernorm = nn.LayerNorm.forward
        def bf16_ln(mod, inp):
            if inp.dtype == torch.bfloat16:
                w = mod.weight.float() if mod.weight is not None else None
                b = mod.bias.float() if mod.bias is not None else None
                return nn.functional.layer_norm(
                    inp.float(), mod.normalized_shape, w, b, mod.eps).bfloat16()
            return orig_layernorm(mod, inp)
        nn.LayerNorm.forward = bf16_ln
        nn.LayerNorm._zpackr_bf16_patched = True

    results = {}
    total = len(args.levels) * args.steps
    run_start = time.time()

    print(f"\nDeep ablation: {len(args.levels)} levels × {args.steps} steps = {total} total steps")
    print(f"Levels: {args.levels}")
    print(f"Output: {root_dir}/")

    for li, max_atten in enumerate(args.levels):
        torch.manual_seed(args.seed)

        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)

        print(f"\n[{li+1}/{len(args.levels)}] max_atten={max_atten:.1f}  loading+compressing...", end=" ")

        model = compress_model(model, pcfg)
        model = model.to(torch.bfloat16)
        model = model.to(device)

        zpl_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, PackRLinearDelta)]
        fixed_byte = int(max_atten * 255)
        for _, module in zpl_layers:
            module.velvet_r._atten_byte.fill_(fixed_byte)

        # Apply LSH prefill if provided (overrides atten_byte fill per row)
        if prefill_data is not None:
            for name, module in zpl_layers:
                # Match layer name to key in prefill data
                key = None
                for k in prefill_data:
                    if k in name:
                        key = k
                        break
                if key is not None:
                    target = prefill_data[key].to(module.velvet_r._atten_byte.device)
                    target = target * (1.0 - max_atten)
                    module.velvet_r.prefill(target)
                else:
                    module.velvet_r.prefill(torch.ones(module.in_features))
        print(f"atten_byte={fixed_byte}")

        optim = CUDA8BitAdam(model.parameters(), lr=args.lr)

        # Per-level output
        level_label = f"atten_{max_atten:.1f}".replace(".", "p")
        level_dir = os.path.join(root_dir, level_label)
        os.makedirs(level_dir, exist_ok=True)
        metrics_file = open(os.path.join(level_dir, "metrics.jsonl"), "w")

        model.train()
        train_iter = iter(loader)
        step_accs = []
        level_start = time.time()
        best_acc = 0.0

        for step in range(args.steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(loader)
                batch = next(train_iter)

            labels = batch.pop("label", None).to(device)
            bg = {k: v.to(device) for k, v in batch.items()}
            if not needs_type_ids:
                bg.pop("token_type_ids", None)

            step_t0 = time.time()
            outputs = model(**bg, labels=labels)
            loss = outputs.loss
            loss.backward()
            optim.step()
            optim.zero_grad()
            step_time = time.time() - step_t0

            # Per-step loss to metrics.jsonl
            record = {"step": step + 1, "loss": loss.item(), "step_time_s": step_time}
            acc = None

            if (step + 1) % args.eval_interval == 0:
                model.eval()
                all_preds, all_labels = [], []
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
                if acc > best_acc:
                    best_acc = acc
                step_accs.append((step + 1, acc))
                model.train()

                steps_done = li * args.steps + step + 1
                elapsed = time.time() - run_start
                total_elapsed = elapsed / steps_done * total
                remaining = total_elapsed * (1 - steps_done / total)
                print(f"    step {step+1:>4d}  acc={acc:.4f}  best={best_acc:.4f}  loss={loss.item():.4f}  "
                      f"({time.time()-level_start:.0f}s this level, ~{remaining:.0f}s remain)")

            record["acc"] = acc
            metrics_file.write(json.dumps(record) + "\n")
            if step % 100 == 0:
                metrics_file.flush()

        metrics_file.close()

        # Level summary.json
        summary = {
            "max_atten": max_atten, "atten_byte": fixed_byte,
            "steps": args.steps, "best_acc": best_acc,
            "eval_accs": step_accs,
            "level_time_s": time.time() - level_start,
        }
        with open(os.path.join(level_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        results[max_atten] = step_accs

    # ── Final combined pickle ──
    out_path = os.path.join(root_dir, "results.pkl")
    out = {
        "config": cfg,
        "results": {str(k): v for k, v in results.items()},
        "final_table": _format_table(args.levels, args.steps, args.eval_interval, results),
    }
    with open(out_path, "wb") as f:
        pickle.dump(out, f)

    # ── Print ──
    print("\n" + "=" * 80)
    print("DEEP ABLATION RESULTS")
    print("=" * 80)
    print(out["final_table"])
    print(f"\nSaved to {root_dir}/")


def _format_table(levels, max_steps, eval_interval, results):
    header = f"{'max_atten':>10s}"
    for s in range(eval_interval, max_steps + 1, eval_interval):
        header += f"  step={s:>4d}"
    lines = [header, "-" * len(header)]
    for at in levels:
        row = f"{at:>10.1f}"
        acc_map = dict(results[at])
        for s in range(eval_interval, max_steps + 1, eval_interval):
            a = acc_map.get(s, -1)
            row += f"  {a:>8.4f}" if a >= 0 else f"  {'---':>8s}"
        lines.append(row)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
