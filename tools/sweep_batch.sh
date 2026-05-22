#!/bin/bash
# Batch ablation sweep — runs after Distil MNLI 0.9 finishes.
# Chains: RTE ceiling → ALBERT SST-2 ceiling → MNLI knife-edge
# Each writes to its own timestamped run directory.

set -e
cd "$(dirname "$0")/.."

echo "============================================"
echo "Batch ablation sweep starting at $(date)"
echo "============================================"

# ── 1. RTE ceiling sweep on BERT (30 min) ──
echo ""
echo "[1/3] RTE ceiling sweep: BERT atten=0.0,0.5,0.7 × 500 steps"
echo ""
python tools/ablate_deep.py \
  --task rte \
  --levels 0.0 0.5 0.7 \
  --steps 500 \
  --eval-interval 100

# ── 2. ALBERT SST-2 ceiling sweep (45 min) ──
echo ""
echo "[2/3] ALBERT SST-2 ceiling sweep: atten=0.0,0.5,0.7 × 500 steps"
echo ""
python tools/ablate_deep.py \
  --model albert-base-v2 \
  --levels 0.0 0.5 0.7 \
  --steps 500 \
  --eval-interval 100

# ── 3. MNLI knife-edge on BERT (60 min) ──
echo ""
echo "[3/3] MNLI knife-edge: BERT atten=0.6,0.8 × 250 steps"
echo ""
python tools/ablate_deep_mnli.py \
  --levels 0.6 0.8 \
  --steps 250 \
  --eval-interval 50 \
  --eval-steps 100

echo ""
echo "============================================"
echo "Batch sweep done at $(date)"
echo "============================================"
