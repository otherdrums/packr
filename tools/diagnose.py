"""ZPackR Diagnostic Trainer — ratio-logging for signal calibration.

Thin wrapper around ZPackRTrainer that adds per-block compression ratio
tracking at each post_step boundary.  Produces a ratio_log.jsonl file
with per-step and per-block signals.

Usage:
    python tools/diagnose.py --task sst2 --max-steps 500 --post-step-interval 1

Output:
    runs/<label>_<ts>_<commit>/
        metrics.jsonl          # standard harness metrics
        ratio_log.jsonl        # per-step ratios + per-block snapshots
        config.json            # full config snapshot
        summary.json           # final summary
"""

import os
import sys
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from tools.train_harness import ZPackRTrainer, TrainerConfig, GLUE_TASKS
from packr.config import PackRConfig
from packr.prompt_gate import should_skip_backward
from packr.zpackr_layer import ZPackRLinear, ATTENUATION_SKIP_THRESHOLD


def _timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _git_commit_short():
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
            cwd=os.path.join(os.path.dirname(__file__), ".."),
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


class DiagnosticTrainer(ZPackRTrainer):
    """ZPackRTrainer with per-block ratio logging at each post_step."""

    def __init__(self, config: TrainerConfig):
        super().__init__(config)
        self._ratio_file = None

    def run(self) -> dict:
        self.setup()

        ratio_path = os.path.join(self.output_dir, "ratio_log.jsonl")
        self._ratio_file = open(ratio_path, "w")

        self._start_time = time.perf_counter()
        self._log(f"Starting diagnostic training ({self.config.max_steps} steps) ...")

        self._model.train()
        train_iter = iter(self._train_loader)

        while self._global_step < self.config.max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self._train_loader)
                batch = next(train_iter)

            step_start = time.perf_counter()

            # ── Forward ──
            labels = batch.pop("label", None)
            batch_gpu = {k: v.to(self.device) for k, v in batch.items()}
            if labels is not None:
                labels = labels.to(self.device)

            outputs = self._model(**batch_gpu, labels=labels)
            loss = outputs.loss / self.config.grad_accum_steps

            # ── Convergence gate ──
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

                    if self.config.warmup_steps > 0 and self._velvet is not None:
                        if self._global_step < self.config.warmup_steps:
                            self._velvet.warmup_step(
                                self._global_step, self.config.warmup_steps
                            )

                    if self._velvet is not None:
                        self._velvet.step()

                    if self._zpl_layers is not None:
                        for _, module in self._zpl_layers:
                            module.post_step()

                    self._optimizer.zero_grad()

                    if self._zpl_layers is not None:
                        for _, module in self._zpl_layers:
                            module.stage_delta_async(None)

            # ── Log ratios every step (cached between post_steps) ──
            if self._zpl_layers is not None:
                self._log_ratios(loss.item(), gate_skipped)

            step_ms = (time.perf_counter() - step_start) * 1000
            self._record_step(self._gather_metrics(
                loss.item() * self.config.grad_accum_steps, step_ms, gate_skipped
            ))

            if (self._global_step + 1) % self.config.eval_interval == 0:
                self._run_eval()

            if (self._global_step + 1) % self.config.checkpoint_interval == 0:
                self._save_checkpoint()

            self._global_step += 1

        self._run_eval()
        self._save_summary()
        self._metrics_file.close()
        self._ratio_file.close()

        elapsed = time.perf_counter() - self._start_time
        self._log(f"Diagnostic training complete in {elapsed:.1f}s. Output: {self.output_dir}")
        return self._ephemeral

    # ── Ratio logging ──

    def _log_ratios(self, loss: float, gate_skipped: bool):
        """Extract and log per-block compression ratios from all ZPackRLinear layers."""
        if self._zpl_layers is None:
            return

        log = {
            "step": self._global_step + 1,
            "loss": loss,
            "gate_skipped": gate_skipped,
            "layers": {},
        }

        all_ratios = []
        all_kept_ratios = []

        for short_name, module in self._zpl_layers:
            data = module.get_block_ratios()
            ratios = data["ratios"]

            layer_info = {
                "ratio_max": max(ratios),
                "ratio_min": min(ratios),
                "ratio_mean": sum(ratios) / len(ratios),
                "salient_count": data["salient_count"],
                "num_blocks": data["num_blocks"],
            }

            gaps = data.get("block_gaps", ratios)
            attenuations = data.get("attenuation_scores")
            if attenuations is None:
                span = 7.0  # RATIO_CEILING - RATIO_FLOOR
                attenuations = [max(0.0, min(1.0, (r - 1.0) / span)) for r in ratios]

            layer_info["blocks"] = [
                {
                    "blk": i,
                    "ratio": ratios[i],
                    "attenuation": attenuations[i] if i < len(attenuations) else 1.0,
                    "delta_l2": round(data["delta_l2"][i], 8),
                }
                for i in range(len(ratios))
            ]

            log["layers"][short_name] = layer_info

            all_ratios.extend(ratios)
            salient_mask = module.block_mask
            for i in range(len(ratios)):
                if salient_mask[i]:
                    all_kept_ratios.append(ratios[i])

        if all_ratios:
            log["weight_ratio_max"] = max(all_ratios)
            log["weight_ratio_min"] = min(all_ratios)
            log["weight_ratio_mean"] = sum(all_ratios) / len(all_ratios)
        if all_kept_ratios:
            log["weight_ratio_kept_mean"] = sum(all_kept_ratios) / len(all_kept_ratios)

        self._ratio_file.write(json.dumps(log) + "\n")
        self._ratio_file.flush()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="ZPackR Diagnostic Trainer — ratio logging for signal calibration"
    )
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--task", default="sst2")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--velvet", action="store_true", default=True)
    parser.add_argument("--no-velvet", action="store_false", dest="velvet")
    parser.add_argument("--attenuation-skip", action="store_true", default=True)
    parser.add_argument("--no-attenuation-skip", action="store_false", dest="attenuation_skip")
    parser.add_argument("--attenuation-skip-threshold", type=float, default=ATTENUATION_SKIP_THRESHOLD)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--label", default="", help="Prefix for output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainerConfig(
        model_name=args.model,
        task_name=args.task,
        packr_config=PackRConfig(mode="zpackr"),
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
        run_label=args.label or "diagnostic",
        seed=args.seed,
    )
    trainer = DiagnosticTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
