#!/usr/bin/env bash
set -euo pipefail

# =========================
# Non-Mutualistic V2
# 3 AGVs : 6 Pickers
# Seed 2, one repeat only
# =========================

ENV_ID="tarware-medium-3agvs-6pickers-partialobs-v1"

NUM_SEEDS=1
REPEATS_PER_SEED=1
SEED=2
MAX_STEPS=1000

OUT_DIR="outputs/debug_v2_3_6_seed2_once_$(date +%Y%m%d_%H%M%S)"

TOPK_REQUESTS=10
TOPK_EMPTY=10
TOPK_GOALS=10
PICKERS_TO_AGVS=1
BLOCK_CONFLICTING_ACTIONS=1
CARE_FOR_AGENTS_IN_COST=0

STAGE1_POOL_K=8
STAGE2_PICKER_OPTIONS_PER_RACK=3
MAX_REQUESTS_PER_BATCH=2
IDLE_PROBE_GAP_STEPS=25
UNIQUE_PICKER=1
UNIQUE_RACK=1

mkdir -p "${OUT_DIR}"

echo "========================================"
echo "Running Non-Mutualistic V2 seed 2 once"
echo "Environment: ${ENV_ID}"
echo "Output directory: ${OUT_DIR}"
echo "Seed: ${SEED}"
echo "Repeats: ${REPEATS_PER_SEED}"
echo "Max steps: ${MAX_STEPS}"
echo "========================================"

python3 scripts/run_non_mutualistic_comm_llm_v2.py \
  --env_id "${ENV_ID}" \
  --num_seeds "${NUM_SEEDS}" \
  --repeats_per_seed "${REPEATS_PER_SEED}" \
  --seed "${SEED}" \
  --max_steps "${MAX_STEPS}" \
  --out_dir "${OUT_DIR}" \
  --topk_requests "${TOPK_REQUESTS}" \
  --topk_empty "${TOPK_EMPTY}" \
  --topk_goals "${TOPK_GOALS}" \
  --pickers_to_agvs "${PICKERS_TO_AGVS}" \
  --block_conflicting_actions "${BLOCK_CONFLICTING_ACTIONS}" \
  --care_for_agents_in_cost "${CARE_FOR_AGENTS_IN_COST}" \
  --stage1_pool_k "${STAGE1_POOL_K}" \
  --stage2_picker_options_per_rack "${STAGE2_PICKER_OPTIONS_PER_RACK}" \
  --max_requests_per_batch "${MAX_REQUESTS_PER_BATCH}" \
  --idle_probe_gap_steps "${IDLE_PROBE_GAP_STEPS}" \
  --unique_picker "${UNIQUE_PICKER}" \
  --unique_rack "${UNIQUE_RACK}"

echo
echo "========================================"
echo "Finished V2 seed 2 run"
echo "Results are in: ${OUT_DIR}"
echo "========================================"