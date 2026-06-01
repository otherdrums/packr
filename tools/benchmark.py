"""PackR Benchmark — apples-to-apples comparison against standard fine-tune.

Runs four fine-tune scenarios with identical hyperparameters (seed, LR, dtype,
batch size, max steps) so the only variables are layer architecture (standard
nn.Linear vs PackRLinearDelta), compression, and optimizer offloading.

Usage:
    python tools/benchmark.py --model bert-base-uncased --task sst2 --max-steps 2000

Output:
    runs/benchmark_<task>_<ts>_<commit>/
        baseline_adamw/   — no compression, torch.optim.AdamW
        baseline_optmatch/ — no compression, same 8-bit optimizer as PackR
        packr/             — PackRLinearDelta compression
        packr_offload/     — PackR + CPU offload of optimizer states
        comparison.json    — aggregated results with deltas
"""

import os
import sys
import json
import time
import argparse
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from tools.train_harness import Trainer, TrainerConfig, PackRConfig, _git_commit_short, _timestamp


BENCHMARK_LABEL = "benchmark"


def _make_benchmark_dir(base: str, task: str):
    commit = _git_commit_short()
    ts = _timestamp()
    dirname = f"{BENCHMARK_LABEL}_{task}_{ts}_{commit}"
    path = os.path.join(base, dirname)
    os.makedirs(path, exist_ok=True)
    return path


def _run_cleanup():
    """Free GPU memory between benchmark runs for clean VRAM measurements."""
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_max_memory_allocated()


def _count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _count_total_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _compute_elapsed_seconds(summary: dict) -> float:
    return summary.get("elapsed_seconds", 0.0)


def _load_summary(run_dir: str) -> dict:
    path = os.path.join(run_dir, "summary.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


Scenario = dict  # type alias


def build_scenarios(args) -> list[Scenario]:
    """Build four benchmark scenarios with identical base config."""
    common = dict(
        model_name=args.model,
        task_name=args.task,
        num_labels=args.num_labels,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        grad_accum_steps=args.grad_accum_steps,
        max_seq_length=args.max_seq_length,
        warmup_steps=args.warmup_steps,
        eval_interval=args.eval_interval,
        eval_steps=args.eval_steps,
        checkpoint_interval=args.max_steps + 1,  # no checkpoints during bench
        output_dir=args.output_dir,
        seed=args.seed,
    )

    # bf16 is shared across all runs for apples-to-apples comparison.
    # Baseline AdamW gets the same dtype as PackR — the only difference
    # is the layer architecture (standard nn.Linear vs PackRLinearDelta).
    bf16 = args.bf16
    opt = args.optimizer

    return [
        {
            "label": "baseline_adamw",
            "desc": "Standard AdamW (no compression)",
            "packr_enabled": False,
            "offload_enabled": False,
            "packr_config": PackRConfig(bf16=bf16, optimizer_type="adamw"),
            **common,
        },
        {
            "label": "baseline_optmatch",
            "desc": f"No compression, optimizer={opt}",
            "packr_enabled": False,
            "offload_enabled": False,
            "packr_config": PackRConfig(bf16=bf16, optimizer_type=opt),
            **common,
        },
        {
            "label": "packr",
            "desc": f"PackRLinearDelta, optimizer={opt}",
            "packr_enabled": True,
            "offload_enabled": False,
            "packr_config": PackRConfig(bf16=bf16, optimizer_type=opt),
            **common,
        },
        {
            "label": "packr_offload",
            "desc": "PackR + CPU optimizer offload",
            "packr_enabled": True,
            "offload_enabled": True,
            "packr_config": PackRConfig(bf16=bf16, optimizer_type="triton8"),
            **common,
        },
    ]


def run_scenario(scenario: Scenario, bench_dir: str) -> dict:
    """Run a single benchmark scenario and return its results."""
    label = scenario["label"]
    run_dir = os.path.join(bench_dir, label)
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  [{label}] {scenario['desc']}")
    print(f"{'='*60}")

    cfg = TrainerConfig(
        packr_enabled=scenario["packr_enabled"],
        packr_config=scenario["packr_config"],
        offload_enabled=scenario["offload_enabled"],
        model_name=scenario["model_name"],
        task_name=scenario["task_name"],
        num_labels=scenario["num_labels"],
        lr=scenario["lr"],
        betas=scenario["betas"],
        weight_decay=scenario["weight_decay"],
        batch_size=scenario["batch_size"],
        max_steps=scenario["max_steps"],
        grad_accum_steps=scenario["grad_accum_steps"],
        max_seq_length=scenario["max_seq_length"],
        warmup_steps=scenario["warmup_steps"],
        eval_interval=scenario["eval_interval"],
        eval_steps=scenario["eval_steps"],
        checkpoint_interval=scenario["checkpoint_interval"],
        output_dir=run_dir,
        run_label=label,
        seed=scenario["seed"],
    )
    trainer = Trainer(cfg)
    result = trainer.run()

    _run_cleanup()
    return result


def _fmt_pct(diff: float) -> str:
    s = f"{diff:+.1f}%"
    if abs(diff) < 0.05:
        return "  ~0.0%"
    return s


def _fmt_val(v):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def print_comparison(results: list[dict], run_labels: list[str]):
    """Print a formatted comparison table."""
    col_w = 28
    sep = "  "

    headers = ["", "Accuracy", "VRAM (MB)", "Steps/s", "Elapsed (s)", "Trainable Params"]
    print(f"\n{'=' * (col_w + len(sep) * len(headers))}")
    print(f"  Comparison — apples-to-apples (same seed, dtype, LR, batch size)")
    print(f"{'=' * (col_w + len(sep) * len(headers))}")

    # Header row
    cells = [h.rjust(14) for h in headers]
    print(f"{'Scenario'.ljust(col_w)}{sep}{sep.join(cells)}")
    print("-" * (col_w + len(sep) * len(headers)))

    # Data rows
    baselines = {}
    for i, (label, r) in enumerate(zip(run_labels, results)):
        acc = r.get("ephemeral", {}).get("eval_metric")
        vram = r.get("peak_vram_mb")
        elapsed = r.get("elapsed_seconds", 0)
        steps = r.get("total_steps", 1)
        steps_per_s = steps / elapsed if elapsed > 0 else 0
        trainable = r.get("trainable_params", 0)

        if trainable >= 1e6:
            param_str = f"{trainable/1e6:.1f}M"
        else:
            param_str = f"{trainable:,}"

        row = [
            f"{acc:.4f}" if acc is not None else "N/A",
            f"{vram:.0f}" if vram else "N/A",
            f"{steps_per_s:.1f}" if steps_per_s else "N/A",
            f"{elapsed:.0f}" if elapsed else "N/A",
            param_str,
        ]
        cells = [c.rjust(14) for c in row]
        print(f"{label.ljust(col_w)}{sep}{sep.join(cells)}")
        baselines[label] = {"acc": acc, "vram": vram, "steps_per_s": steps_per_s, "trainable": trainable}

    # Delta rows
    packr = baselines.get("packr", {})
    for ref_label, ref_name in [("baseline_adamw", "vs AdamW"), ("baseline_optmatch", "vs OptMatch")]:
        ref = baselines.get(ref_label, {})
        if not ref or not packr:
            continue
        acc_diff = ((packr["acc"] - ref["acc"]) / ref["acc"] * 100) if ref["acc"] and packr["acc"] else None
        vram_diff = ((packr["vram"] - ref["vram"]) / ref["vram"] * 100) if ref["vram"] and packr["vram"] else None
        speed_diff = ((packr["steps_per_s"] - ref["steps_per_s"]) / ref["steps_per_s"] * 100) if ref["steps_per_s"] and packr["steps_per_s"] else None
        param_diff = ((packr["trainable"] - ref["trainable"]) / ref["trainable"] * 100) if ref["trainable"] and packr["trainable"] else None

        row = [
            _fmt_pct(acc_diff) if acc_diff is not None else "   N/A",
            _fmt_pct(vram_diff) if vram_diff is not None else "   N/A",
            _fmt_pct(speed_diff) if speed_diff is not None else "   N/A",
            "  N/A",
            _fmt_pct(param_diff) if param_diff is not None else "   N/A",
        ]
        cells = [c.rjust(14) for c in row]
        print(f"PackR {ref_name}".ljust(col_w) + sep + sep.join(cells))

    # PackR+Offload vs PackR
    offload = baselines.get("packr_offload", {})
    if offload and packr:
        vram_diff = ((offload["vram"] - packr["vram"]) / packr["vram"] * 100) if packr["vram"] and offload["vram"] else None
        speed_diff = ((offload["steps_per_s"] - packr["steps_per_s"]) / packr["steps_per_s"] * 100) if packr["steps_per_s"] and offload["steps_per_s"] else None
        row = [
            "  N/A",
            _fmt_pct(vram_diff) if vram_diff is not None else "   N/A",
            _fmt_pct(speed_diff) if speed_diff is not None else "   N/A",
            "  N/A",
            "  N/A",
        ]
        cells = [c.rjust(14) for c in row]
        print(f"PackR+Offload vs PackR".ljust(col_w) + sep + sep.join(cells))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="PackR Benchmark — apples-to-apples comparison"
    )
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--task", default="sst2")
    parser.add_argument("--num-labels", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=128,
                        help="Max sequence length (increase for MNLI etc.)")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--bf16", action="store_true", default=False,
                        help="Convert model to bfloat16 (applied uniformly to all runs)")
    parser.add_argument("--optimizer", choices=["triton8", "cuda8"], default="cuda8",
                        help="Optimizer for PackR and baseline_optmatch runs")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Build scenarios — all share the same hyperparams for apples-to-apples
    scenarios = build_scenarios(args)
    run_labels = [s["label"] for s in scenarios]

    bench_dir = _make_benchmark_dir(args.output_dir, args.task)
    print(f"Benchmark output: {bench_dir}")
    print(f"Model: {args.model}  Task: {args.task}  Steps: {args.max_steps}")
    print(f"Seed: {args.seed}  bf16: {args.bf16}  Optimizer: {args.optimizer}")
    print(f"\nAll runs share identical hyperparams (seed, dtype, LR, batch size).")
    print(f"The only differences are layer architecture and compression.\n")

    results = []
    for scenario in scenarios:
        result = run_scenario(scenario, bench_dir)
        label = scenario["label"]
        run_dir = os.path.join(bench_dir, label)
        summary = _load_summary(run_dir)
        results.append({
            "label": label,
            "desc": scenario["desc"],
            "run_dir": run_dir,
            "ephemeral": result,
            "peak_vram_mb": summary.get("peak_vram_mb"),
            "total_steps": summary.get("total_steps", args.max_steps),
            "elapsed_seconds": summary.get("elapsed_seconds"),
            "final_eval_metric": summary.get("final_eval_metric"),
            "trainable_params": None,  # populated below
        })

    # Compute param counts by loading configs from saved summaries
    for r in results:
        summary = _load_summary(r["run_dir"])
        cfg = summary.get("config", {})
        # Estimate: for PackR, trainable = delta params only (base_W frozen)
        # For baseline, all params are trainable.
        # We don't have a model loaded here, so estimate from config.
        if "packr" in r["label"] and cfg.get("packr_enabled", True):
            # Rough estimate: for BERT-base, ~110M total params, delta is ~half (bf16)
            # This is a simplification; exact count would need a model instance
            r["trainable_params"] = None  # mark as estimated
        else:
            r["trainable_params"] = None

    # Write comparison.json
    comparison = {
        "benchmark_dir": bench_dir,
        "model": args.model,
        "task": args.task,
        "seed": args.seed,
        "bf16": args.bf16,
        "runs": [
            {
                "label": r["label"],
                "desc": r["desc"],
                "final_eval_metric": r["final_eval_metric"],
                "peak_vram_mb": r["peak_vram_mb"],
                "elapsed_seconds": r["elapsed_seconds"],
                "total_steps": r["total_steps"],
            }
            for r in results
        ],
    }
    # Add deltas
    packr_data = next((r for r in comparison["runs"] if r["label"] == "packr"), None)
    if packr_data:
        for ref_label in ("baseline_adamw", "baseline_optmatch"):
            ref = next((r for r in comparison["runs"] if r["label"] == ref_label), None)
            if ref and packr_data["final_eval_metric"] and ref["final_eval_metric"]:
                key = f"packr_vs_{ref_label}"
                comparison[key] = {
                    "acc_delta_pct": round(
                        (packr_data["final_eval_metric"] - ref["final_eval_metric"])
                        / ref["final_eval_metric"] * 100, 2
                    ),
                }
                if packr_data["peak_vram_mb"] and ref["peak_vram_mb"]:
                    comparison[key]["vram_delta_pct"] = round(
                        (packr_data["peak_vram_mb"] - ref["peak_vram_mb"])
                        / ref["peak_vram_mb"] * 100, 2
                    )

    comp_path = os.path.join(bench_dir, "comparison.json")
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"Comparison saved: {comp_path}")

    # Print comparison table
    print_comparison(results, run_labels)

    # Summary verdict
    packr_res = next((r for r in results if r["label"] == "packr"), None)
    baseline_res = next((r for r in results if r["label"] == "baseline_adamw"), None)
    if packr_res and baseline_res:
        p_acc = packr_res.get("final_eval_metric")
        b_acc = baseline_res.get("final_eval_metric")
        p_vram = packr_res.get("peak_vram_mb")
        b_vram = baseline_res.get("peak_vram_mb")
        if p_acc is not None and b_acc is not None:
            acc_str = f"{abs(p_acc - b_acc):.4f} {'better' if p_acc > b_acc else 'worse'}"
            print(f"Accuracy delta:  PackR is {acc_str} than AdamW baseline")
        if p_vram and b_vram:
            vram_save = (1 - p_vram / b_vram) * 100
            print(f"VRAM savings:    PackR uses {vram_save:.0f}% less memory ({p_vram:.0f} vs {b_vram:.0f} MB)")


if __name__ == "__main__":
    main()
