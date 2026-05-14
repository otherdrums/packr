"""Velvet — Velocity to Learning Rate Translation.

Closed-loop adaptive learning rate controller that reads AdamW second-moment
statistics (exp_avg_sq / v) every optimizer step and adjusts per-param-group
LRs in real time based on the filtered velocity of gradient variance.

Core insight: when a layer is actively learning, its exp_avg_sq climbs
(positive velocity → LR stays hot).  When a layer saturates, exp_avg_sq
flattens (velocity → 0 → LR decays).  The Exponential Moving Average (EMA)
acts as a low-pass filter separating signal from micro-batch noise.

Works with both standard torch.optim.AdamW (float32 state) and
FusedQuantizedAdam (int8 block-quantized state) via auto-detection.
"""

from __future__ import annotations

import torch
from typing import Dict, List, Optional


class VelvetController:
    """Velocity to Learning Rate Translation controller.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer whose param_groups will be adapted in real time.
    beta : float
        EMA smoothing coefficient for velocity (0 < beta < 1).
        Higher = more smoothing, slower to react.  Default 0.97.
        Set to None to auto-tune from total_opt_steps.
    min_multiplier : float
        Minimum LR multiplier applied when velocity is near zero
        (layer is saturating).  Default 0.175.
    max_multiplier : float
        Maximum LR multiplier applied when velocity is high
        (layer is actively learning).  Default 1.0.
    velocity_scale : float
        Scaling factor that maps normalized velocity to the [0,1] range
        before clamping to [min_multiplier, max_multiplier].
        Higher = more aggressive LR reduction.  Default 10.0.
        Set to None to auto-tune from total_opt_steps.
    train_samples : int, optional
        Number of training examples in the dataset.  When provided and
        beta/min_multiplier/velocity_scale are None, all four are
        auto-tuned from a single scale factor derived purely from
        dataset size — invariant to batch size and accumulation steps.
        Normalization point: 32,000 examples (≈ B⊂8 × acc⊂4 × 1,000 opt steps).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        beta: float = 0.97,
        min_multiplier: float = 0.175,
        max_multiplier: float = 1.0,
        velocity_scale: float = 10.0,
        train_samples: Optional[int] = None,
        total_opt_steps: Optional[int] = None,
        opt_steps_per_epoch: Optional[int] = None,
    ):
        velocity_scale_default = 10.0  # used as reference for auto-tuning

        tuning_steps = train_samples
        if tuning_steps is None:
            tuning_steps = opt_steps_per_epoch
        if tuning_steps is None:
            tuning_steps = total_opt_steps

        if tuning_steps is not None and tuning_steps > 0:
            if train_samples is not None:
                scale = min(1.0, (tuning_steps / 32000.0) ** 0.33)
            else:
                scale = min(1.0, (tuning_steps / 1000.0) ** 0.33)
            auto_half_life = max(5, int(25.0 * scale))
            if beta is None:
                beta = 0.5 ** (1.0 / auto_half_life)
                min_multiplier = 0.175 + 0.325 * (1.0 - scale)
                v_ref_half_life = max(3, int(23.0 * scale))
                self._v_ref_beta = 0.5 ** (1.0 / v_ref_half_life)
                self._auto_tuned = True
                self._min_observation_steps = 1
            else:
                self._v_ref_beta = 0.97
                self._auto_tuned = False
                self._min_observation_steps = 1
            if velocity_scale is None:
                velocity_scale = 1.5 + (velocity_scale_default - 1.5) * scale
        else:
            self._auto_tuned = False
            self._min_observation_steps = 1
        if velocity_scale is None:
            velocity_scale = velocity_scale_default

        if not 0 < beta < 1:
            raise ValueError(f"beta must be in (0, 1), got {beta}")
        if not 0 <= min_multiplier <= max_multiplier:
            raise ValueError(
                f"min_multiplier ({min_multiplier}) must be <= "
                f"max_multiplier ({max_multiplier})"
            )

        self._optimizer = optimizer
        self.beta = beta
        self.min_m = min_multiplier
        self.max_m = max_multiplier
        self.vel_scale = velocity_scale

        # Per-parameter tracking, keyed by id(p)
        self._v_ref: Dict[int, float] = {}           # slow EMA of v_mean (reference level)
        self._ema_velocity: Dict[int, float] = {}   # EMA-filtered velocity
        self._base_lr: Dict[int, float] = {}        # group_idx → base_lr

        # Capture base LRs immediately (before warmup or any scheduling
        # modifies them).  These are immutable reference values.
        for group_idx, group in enumerate(self._optimizer.param_groups):
            self._base_lr[group_idx] = group["lr"]

        self._step_count = 0
        self._stats: Dict[str, list] = {
            "step": [],
            "group": [],
            "v_mean": [],
            "velocity": [],
            "multiplier": [],
            "lr": [],
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def warmup_step(self, global_step: int, warmup_steps: int) -> None:
        """Apply linear warmup LR schedule (called every micro-batch).

        Ramps each param group's LR from 10% to 100% of its base_lr
        over ``warmup_steps`` micro-batches.
        """
        factor = 0.1 + 0.9 * (global_step / max(warmup_steps, 1))
        for group_idx, group in enumerate(self._optimizer.param_groups):
            group["lr"] = self._base_lr[group_idx] * factor

    @torch.no_grad()
    def step(self) -> None:
        """Called **after** ``optimizer.step()``.

        Reads the freshly-updated ``v`` (exp_avg_sq) states, computes
        the filtered velocity, and translates it to per-group LR multipliers.

        During the observation window (auto-tuned from dataset size), EMA
        accumulates velocity estimates but LRs stay at max multiplier.
        After the window closes, the velocity→LR translation engages.
        Base LRs are captured at construction time so warmup cannot
        corrupt them.
        """
        self._step_count += 1
        is_first = self._step_count == 1
        observing = self._step_count <= self._min_observation_steps

        for group_idx, group in enumerate(self._optimizer.param_groups):
            multipliers: List[float] = []

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self._optimizer.state.get(p)
                if state is None or "v" not in state:
                    continue

                pid = id(p)

                # ---- dequantize and compute mean of v ----
                v_mean = self._dequantize_v_mean(state["v"], state.get("v_scale"))

                if is_first or pid not in self._v_ref:
                    # Seed both reference EMA and velocity EMA at v_mean.
                    self._v_ref[pid] = v_mean
                    self._ema_velocity[pid] = 0.0
                    multipliers.append(self.max_m)
                    continue

                # ---- update slow v_mean reference (trend line) ----
                v_ref_old = self._v_ref[pid]
                v_ref_new = self._v_ref_beta * v_ref_old + (1.0 - self._v_ref_beta) * v_mean
                self._v_ref[pid] = v_ref_new

                # ---- velocity = reference's own rate of change, denoised ----
                # v_ref is a slow EMA — its per-step movement is tiny.
                # Normalize by (1−β) to recover the equivalent v_mean shift,
                # so velocity scale is independent of v_ref_new inertia.
                delta = (v_ref_new - v_ref_old) / (1.0 - self._v_ref_beta)

                # ---- EMA filter on velocity ----
                ema = (
                    self.beta * self._ema_velocity[pid]
                    + (1.0 - self.beta) * delta
                )
                self._ema_velocity[pid] = ema

                # ---- observation window: keep max LR, let EMA accumulate ----
                if observing:
                    multipliers.append(self.max_m)
                    continue

                # ---- normalize by reference (not current v_mean) ----
                norm_vel = abs(ema) / (v_ref_new + 1e-12)

                # ---- translate to multiplier ----
                multiplier = self.min_m + (self.max_m - self.min_m) * min(
                    1.0, norm_vel * self.vel_scale
                )
                multipliers.append(multiplier)

            # Apply per-param-group multiplier
            if multipliers:
                group_m = sum(multipliers) / len(multipliers)
                group["lr"] = self._base_lr[group_idx] * group_m

                # Collect stats for the *first* param in group (representative)
                sample_id = id(group["params"][0])
                self._stats["step"].append(self._step_count)
                self._stats["group"].append(group_idx)
                self._stats["v_mean"].append(
                    self._v_ref.get(sample_id, 0.0)
                )
                self._stats["velocity"].append(
                    self._ema_velocity.get(sample_id, 0.0)
                )
                self._stats["multiplier"].append(group_m)
                self._stats["lr"].append(group["lr"])

    def get_stats(self) -> dict:
        """Return a snapshot of internal velocity/multiplier statistics.

        Useful for heartbeat logging and post-hoc analysis.
        """
        # Per-group latest values
        per_group = {}
        for group_idx, group in enumerate(self._optimizer.param_groups):
            gname = group.get("name", f"group_{group_idx}")
            per_group[gname] = {
                "base_lr": self._base_lr.get(group_idx),
                "current_lr": group["lr"],
                "multiplier": group["lr"] / max(self._base_lr.get(group_idx, 1e-12), 1e-12),
            }

        return {
            "step_count": self._step_count,
            "beta": self.beta,
            "v_ref_beta": self._v_ref_beta,
            "vel_scale": self.vel_scale,
            "min_multiplier": self.min_m,
            "max_multiplier": self.max_m,
            "auto_tuned": self._auto_tuned,
            "observation_steps": self._min_observation_steps,
            "per_group": per_group,
            "history": {
                "steps": self._stats["step"][-100:],  # last 100
                "multipliers": self._stats["multiplier"][-100:],
                "velocities": self._stats["velocity"][-100:],
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dequantize_v_mean(
        self, v: torch.Tensor, v_scale: Optional[torch.Tensor]
    ) -> float:
        """Compute the mean of the exp_avg_sq state tensor.

        Handles two formats:
        - Float32 (standard AdamW): ``v`` is the raw floating-point tensor.
        - Int8 block-quantized (FusedQuantizedAdam): ``v`` is int8 with
          per-block float32 ``v_scale``.  Dequantized as:
          ``v_fp = v_i8.float().view(num_blocks, block_size) * v_scale``.
        """
        if v.dtype == torch.float32:
            return v.mean().item()

        # Int8 block-quantized path
        if v_scale is None:
            return v.float().mean().item()

        N = v.numel()
        num_blocks = v_scale.numel()
        block_size = (N + num_blocks - 1) // num_blocks

        # Reshape to [num_blocks, block_size], dequantize, mean
        v_r = v.float()
        if N < num_blocks * block_size:
            # There's padding — handle gracefully
            return v_r.mean().item()

        v_r = v_r.view(num_blocks, block_size)
        block_means = v_r.mean(dim=1) * v_scale
        return block_means.mean().item()
