"""ZPackR Training Harness — drop-in trainer for GLUE tasks with full instrumentation.

Records per-step metrics (loss, super ratio, salience, weight ratios, VRAM,
Velvet multipliers, gate stats) to JSON Lines for analysis and ablation.

Usage:
    from tools.train_harness import ZPackRTrainer, TrainerConfig

    config = TrainerConfig(
        model_name="bert-base-uncased",
        task_name="sst2",
        packr_config=PackRConfig(mode="zpackr"),
        max_steps=2000,
        output_dir="runs/sst2_zpackr",
    )
    trainer = ZPackRTrainer(config)
    results = trainer.run()
"""

import os
import sys
import json
import time
import hashlib
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Literal

import torch
import torch.nn as nn
import numpy as np
import threading

# Ensure packr is importable from tools/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from packr.config import PackRConfig
from packr.layer_patcher import compress_model
from packr.optim import FusedQuantizedAdam
from packr.velvet import VelvetController
from packr.prompt_gate import should_skip_backward
from packr.zpackr_layer import ZPackRLinear, ATTENUATION_SKIP_THRESHOLD


def _git_commit_short():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
            cwd=os.path.join(os.path.dirname(__file__), ".."),
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _make_output_dir(base: str, label: str):
    commit = _git_commit_short()
    ts = _timestamp()
    dirname = f"{ts}_{commit}"
    if label:
        dirname = f"{label}_{dirname}"
    path = os.path.join(base, dirname)
    os.makedirs(path, exist_ok=True)
    return path


# ── GLUE task metadata ──

GLUE_TASKS = {
    "sst2": {"num_labels": 2, "keys": ("sentence", None), "metric": "accuracy"},
    "mnli": {"num_labels": 3, "keys": ("premise", "hypothesis"), "metric": "accuracy"},
    "qnli": {"num_labels": 2, "keys": ("question", "sentence"), "metric": "accuracy"},
    "qqp":  {"num_labels": 2, "keys": ("question1", "question2"), "metric": "accuracy"},
    "rte":  {"num_labels": 2, "keys": ("sentence1", "sentence2"), "metric": "accuracy"},
    "mrpc": {"num_labels": 2, "keys": ("sentence1", "sentence2"), "metric": "accuracy"},
    "cola": {"num_labels": 2, "keys": ("sentence", None), "metric": "matthews_correlation"},
    "stsb": {"num_labels": 1, "keys": ("sentence1", "sentence2"), "metric": "pearson"},
}


# ── Configuration ──

@dataclass
class TrainerConfig:
    """Full training configuration with all tunables exposed."""

    # Task
    model_name: str = "bert-base-uncased"
    task_name: str = "sst2"
    num_labels: Optional[int] = None

    # PackR
    packr_config: PackRConfig = field(default_factory=PackRConfig)

    # Optimization
    lr: float = 2e-5
    betas: tuple = (0.9, 0.999)
    weight_decay: float = 0.0
    batch_size: int = 16
    max_steps: int = 10000
    grad_accum_steps: int = 1
    max_seq_length: int = 128

    # Velvet
    velvet_enabled: bool = True
    velvet_beta: float = 0.97
    velvet_min_multiplier: float = 0.175
    velvet_max_multiplier: float = 1.0
    velvet_velocity_scale: float = 10.0
    warmup_steps: int = 0

    # Gate (convergence-driven: skip backward when all blocks fully attenuated)
    attenuation_skip_enabled: bool = True
    attenuation_skip_threshold: float = ATTENUATION_SKIP_THRESHOLD

    # ZPackR
    attenuation_skip_enabled: bool = True
    attenuation_skip_threshold: float = ATTENUATION_SKIP_THRESHOLD

    # Evaluation
    eval_interval: int = 500
    eval_steps: int = 20

    # Checkpoint
    checkpoint_interval: int = 2000

    # Output
    output_dir: str = "runs"
    run_label: str = ""
    seed: int = 42

    def __post_init__(self):
        if self.num_labels is None and self.task_name in GLUE_TASKS:
            self.num_labels = GLUE_TASKS[self.task_name]["num_labels"]


# ── Trainer ──

class ZPackRTrainer:
    """Drop-in trainer for GLUE tasks with full ZPackR instrumentation.

    Records per-step metrics to metrics.jsonl in the output directory.
    Supports both packr and zpackr modes, Velvet, prompt gating,
    checkpointing, and structured ablation runs.
    """

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.output_dir = _make_output_dir(config.output_dir, config.run_label)
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self._metrics_file = open(os.path.join(self.output_dir, "metrics.jsonl"), "w")
        self._step = 0
        self._global_step = 0
        self._start_time = None
        self._model = None
        self._optimizer = None
        self._velvet = None
        self._tokenizer = None
        self._train_loader = None
        self._eval_dataset = None
        self._metric = None
        self._scaler = None  # for amp
        self._ephemeral = {}  # per-run metrics accumulator
        self._gate_skipped_total = 0
        self._gate_total = 0
        self._zpl_layers = None  # cached list of ZPackRLinear instances
        self._zstd_thread = None  # background zstd compression thread
        self._peak_vram = 0      # max VRAM seen during run
        self._last_eval_time = None  # for throughput display
        self._metrics_buffer = [] # batched flush every N steps

        self._log_config()

    def _log_config(self):
        cfg = asdict(self.config)
        cfg["packr_config"] = {
            k: str(v) if k == "scheme" else v
            for k, v in asdict(self.config.packr_config).items()
        }
        cfg["git_commit"] = _git_commit_short()
        cfg["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            cfg["gpu_name"] = torch.cuda.get_device_name(0)
            cfg["gpu_arch"] = f"sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}"
        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2, default=str)

    # ── Setup ──

    def setup(self):
        self._log("Setting up model, tokenizer, dataset ...")
        torch.manual_seed(self.config.seed)

        # Suppress expected startup warnings
        import os as _os
        _os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

        # Tokenizer & model
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name, num_labels=self.config.num_labels,
        )

        # Compress
        self._log(f"Compressing model (mode={self.config.packr_config.mode}) ...")
        self._model = compress_model(self._model, self.config.packr_config)
        self._model = self._model.to(self.device)

        # Cache ZPackRLinear layers to avoid walking named_modules every step
        if self.config.packr_config.mode == "zpackr":
            self._zpl_layers = [
                (name.replace("bert.encoder.", "enc."), m)
                for name, m in self._model.named_modules()
                if isinstance(m, ZPackRLinear)
            ]

        # Dataset
        from datasets import load_dataset
        from datasets import logging as ds_logging
        ds_logging.set_verbosity_error()
        task_info = GLUE_TASKS[self.config.task_name]

        raw_dataset = load_dataset("glue", self.config.task_name)
        train_dataset = raw_dataset["train"]

        self._eval_dataset = raw_dataset.get(
            "validation", raw_dataset.get("validation_matched", raw_dataset["train"])
        )

        self._tokenize = self._make_tokenize_fn(task_info["keys"])

        train_dataset = train_dataset.map(
            self._tokenize, batched=True,
            remove_columns=[c for c in train_dataset.column_names if c not in ("label",)]
        )
        self._eval_dataset = self._eval_dataset.map(
            self._tokenize, batched=True,
            remove_columns=[c for c in self._eval_dataset.column_names if c not in ("label",)]
        )

        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])
        self._eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])

        self._train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True,
            drop_last=True,
        )

        # Optimizer
        self._optimizer = FusedQuantizedAdam(
            self._model.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
            weight_decay=self.config.weight_decay,
            block_size=self.config.packr_config.block_size,
        )

        # Velvet
        if self.config.velvet_enabled:
            self._velvet = VelvetController(
                self._optimizer,
                beta=self.config.velvet_beta,
                min_multiplier=self.config.velvet_min_multiplier,
                max_multiplier=self.config.velvet_max_multiplier,
                velocity_scale=self.config.velvet_velocity_scale,
            )

        # Metric
        import evaluate
        self._metric = evaluate.load("glue", self.config.task_name)

        self._log(f"Setup complete.  Output: {self.output_dir}")

    def _make_tokenize_fn(self, keys):
        tokenizer = self._tokenizer
        max_length = self.config.max_seq_length

        def tokenize(examples):
            key1, key2 = keys
            if key2:
                return tokenizer(
                    examples[key1], examples[key2],
                    truncation=True, padding="max_length",
                    max_length=max_length,
                )
            else:
                return tokenizer(
                    examples[key1],
                    truncation=True, padding="max_length",
                    max_length=max_length,
                )
        return tokenize

    # ── Run ──

    def run(self) -> dict:
        self.setup()
        self._start_time = time.perf_counter()
        self._log(f"Starting training ({self.config.max_steps} steps) ...")

        self._model.train()
        train_iter = iter(self._train_loader)

        while self._global_step < self.config.max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self._train_loader)
                batch = next(train_iter)

            step_start = time.perf_counter()

            # Swap in attenuation from background thread (computed during last forward)
            if self._zpl_layers is not None:
                for _, module in self._zpl_layers:
                    module.swap_attenuation()

            # ── Forward ──
            labels = batch.pop("label", None)
            batch_gpu = {k: v.to(self.device) for k, v in batch.items()}
            if labels is not None:
                labels = labels.to(self.device)

            outputs = self._model(**batch_gpu, labels=labels)
            loss = outputs.loss / self.config.grad_accum_steps

            # ── Convergence gate: skip backward if all blocks fully attenuated ──
            gate_skipped = False
            if self.config.attenuation_skip_enabled and self._zpl_layers is not None:
                gate_skipped = should_skip_backward(
                    self._zpl_layers, self.config.attenuation_skip_threshold
                )
                if gate_skipped:
                    self._gate_skipped_total += 1
                self._gate_total += 1

            if not gate_skipped:
                loss.backward()

                if (self._global_step + 1) % self.config.grad_accum_steps == 0:
                    self._optimizer.step()

                    # Warmup
                    if self.config.warmup_steps > 0 and self._velvet is not None:
                        if self._global_step < self.config.warmup_steps:
                            self._velvet.warmup_step(self._global_step, self.config.warmup_steps)

                    # Velvet
                    if self._velvet is not None:
                        self._velvet.step()

                    self._optimizer.zero_grad()

                    # Stage delta GPU→CPU, then apply to _full_delta
                    if self._zpl_layers is not None:
                        for _, module in self._zpl_layers:
                            module.stage_delta_async(None)
                        for _, module in self._zpl_layers:
                            module.apply_staged_delta()

                    # Launch single background thread to compress all layers
                    if self._zpl_layers is not None:
                        if self._zstd_thread is not None:
                            self._zstd_thread.join()

                        def compress_all():
                            for _, module in self._zpl_layers:
                                module._compress_async()
                        self._zstd_thread = threading.Thread(target=compress_all, daemon=True)
                        self._zstd_thread.start()

            # ── Record step ──
            step_ms = (time.perf_counter() - step_start) * 1000
            self._record_step(self._gather_metrics(loss.item() * self.config.grad_accum_steps, step_ms, gate_skipped))

            # ── Eval ──
            if (self._global_step + 1) % self.config.eval_interval == 0:
                self._run_eval()

            # ── Checkpoint ──
            if (self._global_step + 1) % self.config.checkpoint_interval == 0:
                self._save_checkpoint()

            self._global_step += 1

        # Final eval
        self._run_eval()
        self._save_summary()
        self._metrics_file.close()

        elapsed = time.perf_counter() - self._start_time
        self._log(f"Training complete in {elapsed:.1f}s.  Results: {self.output_dir}")
        return self._ephemeral

    # ── Metrics ──

    def _gather_metrics(self, loss: float, step_ms: float, gate_skipped: bool) -> dict:
        metrics = {
            "step": self._global_step + 1,
            "loss": loss,
            "step_ms": step_ms,
            "gate_skipped": gate_skipped,
        }

        # Velvet multipliers
        if self._velvet is not None:
            try:
                stats = self._velvet.get_stats()
                multipliers = {}
                for gname, ginfo in stats.get("per_group", {}).items():
                    multipliers[gname] = round(ginfo.get("multiplier", 1.0), 4)
                metrics["velvet_multipliers"] = multipliers
                metrics["velvet_max_mult"] = max(multipliers.values()) if multipliers else 0
                metrics["velvet_min_mult"] = min(multipliers.values()) if multipliers else 0
            except Exception:
                pass

        # ZPackR salience + weight ratios
        if self._zpl_layers is not None:
            salience = {}
            total_salient_kb = 0
            total_capacity_kb = 0
            thresholds = {}
            for short_name, module in self._zpl_layers:
                kept = module.salient_count
                total = module.num_blocks
                salience[short_name] = {"kept": kept, "total": total, "fraction": round(kept / max(total, 1), 3)}
                total_salient_kb += kept * module.block_size * module.out_features * 2 / 1024
                total_capacity_kb += total * module.block_size * module.out_features * 2 / 1024
                t = module.salience_threshold
                if t is not None:
                    thresholds[short_name] = round(t, 4)
            if salience:
                metrics["salience"] = salience
                metrics["salient_vram_kb"] = round(total_salient_kb, 0)
                metrics["salient_vram_fraction"] = round(total_salient_kb / max(total_capacity_kb, 1), 3)

        # VRAM
        if self.device.type == "cuda":
            metrics["vram_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 1)
            metrics["vram_peak_mb"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
            torch.cuda.reset_peak_memory_stats()

        return metrics

    def _record_step(self, data: dict):
        data["type"] = "step"
        self._metrics_buffer.append(json.dumps(data))
        # Flush every 10 steps (events flush immediately)
        if len(self._metrics_buffer) >= 10:
            self._flush_metrics()
        # Track peak VRAM across the run
        peak = data.get("vram_peak_mb", 0)
        if peak > self._peak_vram:
            self._peak_vram = peak

    def _flush_metrics(self):
        if self._metrics_buffer:
            self._metrics_file.write("\n".join(self._metrics_buffer) + "\n")
            self._metrics_file.flush()
            self._metrics_buffer.clear()

    def _record_event(self, event_type: str, data: dict):
        self._flush_metrics()  # drain buffer before event
        data["type"] = event_type
        data["step"] = self._global_step + 1
        self._metrics_file.write(json.dumps(data) + "\n")
        self._metrics_file.flush()

    # ── Evaluation ──

    @torch.no_grad()
    def _run_eval(self):
        self._model.eval()
        all_preds = []
        all_labels = []

        eval_loader = torch.utils.data.DataLoader(
            self._eval_dataset, batch_size=self.config.batch_size * 2,
            shuffle=False,
        )

        for i, batch in enumerate(eval_loader):
            if i >= self.config.eval_steps:
                break
            labels = batch.pop("label", None)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            if labels is not None:
                labels = labels.to(self.device)
            outputs = self._model(**batch)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            if labels is not None:
                all_labels.extend(labels.cpu().numpy())

        eval_loss = None
        if all_labels:
            try:
                result = self._metric.compute(predictions=all_preds, references=all_labels)
                eval_loss = result.get(self._metric_name(), 0.0)
            except Exception:
                eval_loss = float(np.mean(np.array(all_preds) == np.array(all_labels)))

        self._record_event("eval", {
            "eval_metric": eval_loss,
            "num_eval_samples": len(all_preds),
        })

        self._ephemeral["eval_metric"] = eval_loss
        if self._last_eval_time is not None:
            elapsed = time.perf_counter() - self._last_eval_time
            ms_per = elapsed * 1000 / self.config.eval_interval
            self._log(f"  Eval at step {self._global_step + 1}: {eval_loss:.4f}  ({ms_per:.0f}ms/step)")
        else:
            self._log(f"  Eval at step {self._global_step + 1}: {eval_loss:.4f}")
        self._last_eval_time = time.perf_counter()
        self._model.train()

    def _metric_name(self):
        return GLUE_TASKS.get(self.config.task_name, {}).get("metric", "accuracy")

    # ── Checkpoint ──

    def _save_checkpoint(self):
        step_dir = os.path.join(self.checkpoint_dir, f"step_{self._global_step + 1}")
        os.makedirs(step_dir, exist_ok=True)

        # ZPackR layer checkpoints
        if self.config.packr_config.mode == "zpackr":
            from zpackr.checkpoint import save_zpackr_checkpoint
            save_zpackr_checkpoint(self._model, step_dir)

        # Optimizer + Velvet state
        state = {
            "step": self._global_step + 1,
            "optimizer": self._optimizer.state_dict(),
        }
        if self._velvet is not None:
            state["velvet_stats"] = self._velvet.get_stats()
        torch.save(state, os.path.join(step_dir, "trainer_state.pt"))

        self._record_event("checkpoint", {"path": step_dir})

    # ── Summary ──

    def _save_summary(self):
        summary = {
            "total_steps": self._global_step,
            "elapsed_seconds": time.perf_counter() - self._start_time,
            "final_eval_metric": self._ephemeral.get("eval_metric"),
            "peak_vram_mb": self._peak_vram,
            "gate_skipped": self._gate_skipped_total,
            "gate_total": self._gate_total,
            "gate_skip_rate": round(self._gate_skipped_total / max(self._gate_total, 1), 3),
            "output_dir": self.output_dir,
            "config": asdict(self.config),
        }
        with open(os.path.join(self.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
        self._record_event("summary", summary)

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}")


# ── CLI ──

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ZPackR Training Harness")
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--task", default="sst2")
    parser.add_argument("--mode", default="zpackr", choices=["packr", "zpackr"])
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--velvet", action="store_true", default=True)
    parser.add_argument("--no-velvet", action="store_false", dest="velvet")
    parser.add_argument("--attenuation-skip", action="store_true", default=True)
    parser.add_argument("--no-attenuation-skip", action="store_false", dest="attenuation_skip")
    parser.add_argument("--attenuation-skip-threshold", type=float, default=ATTENUATION_SKIP_THRESHOLD)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--label", default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainerConfig(
        model_name=args.model,
        task_name=args.task,
        packr_config=PackRConfig(mode=args.mode),
        lr=args.lr,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        eval_steps=args.eval_steps,
        warmup_steps=args.warmup_steps,
        velvet_enabled=args.velvet,
        attenuation_skip_enabled=args.attenuation_skip,
        attenuation_skip_threshold=args.attenuation_skip_threshold,
        output_dir=args.output_dir,
        run_label=args.label,
        seed=args.seed,
    )
    trainer = ZPackRTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
