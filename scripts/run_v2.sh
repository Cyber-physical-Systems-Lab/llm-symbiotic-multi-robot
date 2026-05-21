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

EPISODES=4
SEED=1
MAX_STEPS=1500

OUT_DIR="outputs/run_v2_20260419"

TOPK_REQUESTS=10
TOPK_EMPTY=10
TOPK_GOALS=10
PICKERS_TO_AGVS=1
BLOCK_CONFLICTING_ACTIONS=1
CARE_FOR_AGENTS_IN_COST=0

STAGE1_POOL_K=8
STAGE2_PICKER_OPTIONS_PER_RACK=3
WAIT_TIMEOUT_STEPS=40
MIN_RECOMMUNICATION_GAP_STEPS=8
UNIQUE_PICKER=1
UNIQUE_RACK=1

# Non-mutualistic-only
MAX_REQUESTS_PER_BATCH=2
IDLE_PROBE_GAP_STEPS=25

mkdir -p "${OUT_DIR}"

echo "========================================"
echo "Running NonMutualisticCommLLMPlannerV2 only"
echo "Output directory: ${OUT_DIR}"
echo "Episodes: ${EPISODES}"
echo "Seed: ${SEED}"
echo "Max steps: ${MAX_STEPS}"
echo "========================================"

run_non_mutualistic_v2() {
  local env_id="$1"
  echo
  echo ">>> Running NON-MUTUALISTIC V2 on ${env_id}"
  python scripts/run_non_mutualistic_llm.py \
    --env_id "${env_id}" \
    --episodes "${EPISODES}" \
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
    --wait_timeout_steps "${WAIT_TIMEOUT_STEPS}" \
    --min_recommunication_gap_steps "${MIN_RECOMMUNICATION_GAP_STEPS}" \
    --idle_probe_gap_steps "${IDLE_PROBE_GAP_STEPS}" \
    --unique_picker "${UNIQUE_PICKER}" \
    --unique_rack "${UNIQUE_RACK}"
}

for env_id in "${ENV_IDS[@]}"; do
  echo
  echo "########################################"
  echo "Environment: ${env_id}"
  echo "########################################"

  run_non_mutualistic_v2 "${env_id}"
done

echo
echo "========================================"
echo "Run finished"
echo "Results are in: ${OUT_DIR}"
echo "========================================"
