#!/bin/bash
# S²VOPD + contrastive target — Qwen3.5-4B. One epoch (65 steps) on Vision-OPD-6K, 8 GPUs.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
export EXPERIMENT_SUFFIX="-contrastive"
exec bash "${PROJECT_ROOT}/scripts/run_s2vopd.sh" \
  data.train_files="[\"${PROJECT_ROOT}/data/train.parquet\"]" \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.self_distillation.contrastive_target=True \
  actor_rollout_ref.actor.self_distillation.contrastive_alpha=0.5 \
  actor_rollout_ref.actor.self_distillation.contrastive_beta=0.1 \
  actor_rollout_ref.actor.self_distillation.contrastive_anchor=0.5 \
  trainer.total_training_steps=65 \
  trainer.save_freq=13 \
  "$@"
