#!/usr/bin/env bash
set -euo pipefail

# =========================
# Fixed experiment settings
# =========================

ENV_IDS=(
  "tarware-medium-6agvs-3pickers-partialobs-v1"
  "tarware-medium-6agvs-6pickers-partialobs-v1"
  "tarware-medium-3agvs-6pickers-partialobs-v1"
)

NUM_SEEDS=5          # 不同随机种子的个数（原 EPISODES）
REPEATS_PER_SEED=2   # 每个种子重复次数
SEED=0               # 起始种子（将生成 SEED, SEED+1, ..., SEED+NUM_SEEDS-1）
MAX_STEPS=1000

OUT_DIR="outputs/night_run_20260510"

TOPK_REQUESTS=10
TOPK_EMPTY=10
TOPK_GOALS=10
PICKERS_TO_AGVS=1
BLOCK_CONFLICTING_ACTIONS=1
CARE_FOR_AGENTS_IN_COST=0

STAGE1_POOL_K=8
STAGE2_PICKER_OPTIONS_PER_RACK=3

MIN_RECOMMUNICATION_GAP_STEPS=8
UNIQUE_PICKER=1
UNIQUE_RACK=1

# Symbiotic-only
STAGE1_BACKUPS=2
STAGE2_MAX_OPTIONS_PER_REQUEST=2

# Non-mutualistic-only
MAX_REQUESTS_PER_BATCH=2
IDLE_PROBE_GAP_STEPS=25

mkdir -p "${OUT_DIR}"

echo "========================================"
echo "Night run started"
echo "Output directory: ${OUT_DIR}"
echo "Number of seeds: ${NUM_SEEDS}"
echo "Repeats per seed: ${REPEATS_PER_SEED}"
echo "Base seed: ${SEED}"
echo "Max steps: ${MAX_STEPS}"
echo "========================================"

run_symbiotic() {
  local env_id="$1"
  echo
  echo ">>> Running SYMBIOTIC on ${env_id}"
  python scripts/run_symbiotic_comm_llm.py \
    --env_id "${env_id}" \
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
    --stage1_backups "${STAGE1_BACKUPS}" \
    --stage2_picker_options_per_rack "${STAGE2_PICKER_OPTIONS_PER_RACK}" \
    --stage2_max_options_per_request "${STAGE2_MAX_OPTIONS_PER_REQUEST}" \
    --idle_probe_gap_steps "${IDLE_PROBE_GAP_STEPS}" \
    --unique_picker "${UNIQUE_PICKER}" \
    --unique_rack "${UNIQUE_RACK}"
}

run_non_mutualistic_v2() {
  local env_id="$1"
  echo
  echo ">>> Running NON-MUTUALISTIC V2 on ${env_id}"
  python scripts/run_non_mutualistic_comm_llm_v2.py \
    --env_id "${env_id}" \
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
}

for env_id in "${ENV_IDS[@]}"; do
  echo
  echo "########################################"
  echo "Environment: ${env_id}"
  echo "########################################"

  run_symbiotic "${env_id}"
  run_non_mutualistic_v2 "${env_id}"
done

echo
echo "========================================"
echo "Night run finished"
echo "Results are in: ${OUT_DIR}"
echo "========================================"