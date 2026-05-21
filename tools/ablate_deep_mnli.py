"""Deep ablation on MNLI: run selected max_atten levels to 8000 steps.

Tests the TST cross-task prediction: higher info gap (~0.20) should
produce a stricter (lower) attenuation ceiling than SST-2 (~0.09).

Reports both matched and mismatched validation accuracy.

Usage:
    python tools/ablate_deep_mnli.py --levels 0.0 0.3 0.5
    python tools/ablate_deep_mnli.py --levels 0.0 0.3 0.5 0.7 --steps 4000 --eval-interval 500
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
    parser = argparse.ArgumentParser(description="Deep ablation on MNLI")
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                        help="HuggingFace model name (default: bert-base-uncased)")
    parser.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7],
                        help="Attenuation ceilings to test")
    parser.add_argument("--steps", type=int, default=5000, help="Training steps per level")
    parser.add_argument("--eval-interval", type=int, default=500, help="Steps between eval")
    parser.add_argument("--eval-steps", type=int, default=100, help="Batches per eval (matched+mismatched)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("ERROR: CUDA required"); sys.exit(1)

    # ── Output directory ──
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = args.model.replace("/", "_")
    root_dir = f"runs/ablate_deep_mnli_{model_slug}_{ts}"
    os.makedirs(root_dir, exist_ok=True)

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
    num_labels = model_cfg.num_labels if hasattr(model_cfg, "num_labels") else 3
    torch_cols = ["input_ids", "attention_mask", "label"]
    if needs_type_ids:
        torch_cols = ["input_ids", "attention_mask", "token_type_ids", "label"]

    raw = load_dataset("glue", "mnli")

    # Tokenize train: premise + hypothesis
    cols = [c for c in raw["train"].column_names if c != "label"]
    train = raw["train"].map(
        lambda x: tokenizer(x["premise"], x["hypothesis"],
                            truncation=True, padding="max_length", max_length=128),
        batched=True, remove_columns=cols,
    )
    train.set_format("torch", columns=torch_cols)
    loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True, drop_last=True)

    # Eval: both matched and mismatched
    eval_cols = [c for c in raw["validation_matched"].column_names if c != "label"]
    eval_matched = raw["validation_matched"].map(
        lambda x: tokenizer(x["premise"], x["hypothesis"],
                            truncation=True, padding="max_length", max_length=128),
        batched=True, remove_columns=eval_cols,
    )
    eval_matched.set_format("torch", columns=torch_cols)

    eval_mismatched = raw["validation_mismatched"].map(
        lambda x: tokenizer(x["premise"], x["hypothesis"],
                            truncation=True, padding="max_length", max_length=128),
        batched=True, remove_columns=eval_cols,
    )
    eval_mismatched.set_format("torch", columns=torch_cols)

    _eval_loader = lambda ds: torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)

    pcfg = PackRConfig(mode="zpackr", bf16=True, hash_interval=1,
                       optimizer_type="cuda8", gradient_mix=0.0)

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

    print(f"\nDeep ablation MNLI: {len(args.levels)} levels × {args.steps} steps = {total} total steps")
    print(f"Levels: {args.levels}")
    print(f"Output: {root_dir}/")

    for li, max_atten in enumerate(args.levels):
        torch.manual_seed(args.seed)

        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)
        print(f"\n[{li+1}/{len(args.levels)}] max_atten={max_atten:.1f}  loading+compressing...", end=" ")

        model = compress_model(model, pcfg)
        model = model.to(torch.bfloat16)
        model = model.to(device)

        zpl_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, PackRLinearDelta)]
        fixed_byte = int(max_atten * 255)
        for _, module in zpl_layers:
            module.velvet_r._atten_byte.fill_(fixed_byte)
        print(f"atten_byte={fixed_byte}")

        optim = CUDA8BitAdam(model.parameters(), lr=args.lr)

        level_label = f"atten_{max_atten:.1f}".replace(".", "p")
        level_dir = os.path.join(root_dir, level_label)
        os.makedirs(level_dir, exist_ok=True)
        metrics_file = open(os.path.join(level_dir, "metrics.jsonl"), "w")

        model.train()
        train_iter = iter(loader)
        step_accs = []
        level_start = time.time()
        best_avg = 0.0

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

            record = {"step": step + 1, "loss": loss.item(), "step_time_s": step_time}
            acc_matched = None
            acc_mismatched = None

            if (step + 1) % args.eval_interval == 0:
                model.eval()

                def _eval(ds_loader):
                    all_preds, all_labels = [], []
                    for ei, ebatch in enumerate(ds_loader):
                        if ei >= args.eval_steps:
                            break
                        elabels = ebatch.pop("label", None)
                        ebatch_gpu = {k: v.to(device) for k, v in ebatch.items()}
                        if not needs_type_ids:
                            ebatch_gpu.pop("token_type_ids", None)
                        with torch.no_grad():
                            eout = model(**ebatch_gpu)
                        preds = eout.logits.argmax(dim=-1).cpu().numpy()
                        all_preds.extend(preds)
                        if elabels is not None:
                            all_labels.extend(elabels.cpu().numpy())
                    return float(np.mean(np.array(all_preds) == np.array(all_labels)))

                acc_matched = _eval(_eval_loader(eval_matched))
                acc_mismatched = _eval(_eval_loader(eval_mismatched))
                acc_avg = (acc_matched + acc_mismatched) / 2

                if acc_avg > best_avg:
                    best_avg = acc_avg
                step_accs.append((step + 1, acc_matched, acc_mismatched, acc_avg))
                model.train()

                steps_done = li * args.steps + step + 1
                elapsed = time.time() - run_start
                total_elapsed = elapsed / steps_done * total
                remaining = total_elapsed * (1 - steps_done / total)
                print(f"    step {step+1:>4d}  m={acc_matched:.4f}  mm={acc_mismatched:.4f}  "
                      f"avg={acc_avg:.4f}  best_avg={best_avg:.4f}  loss={loss.item():.4f}  "
                      f"({time.time()-level_start:.0f}s, ~{remaining:.0f}s remain)")

            record["acc_matched"] = acc_matched
            record["acc_mismatched"] = acc_mismatched
            metrics_file.write(json.dumps(record) + "\n")
            if step % 100 == 0:
                metrics_file.flush()

        metrics_file.close()

        summary = {
            "max_atten": max_atten, "atten_byte": fixed_byte,
            "steps": args.steps, "best_avg": best_avg,
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

    print("\n" + "=" * 80)
    print("DEEP ABLATION MNLI RESULTS")
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
        acc_map = {}
        for step, matched, mismatched, avg in results[at]:
            acc_map[step] = avg
        for s in range(eval_interval, max_steps + 1, eval_interval):
            a = acc_map.get(s, -1)
            row += f"  {a:>8.4f}" if a >= 0 else f"  {'---':>8s}"
        lines.append(row)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
