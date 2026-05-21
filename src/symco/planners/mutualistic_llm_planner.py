

"""
Symbiotic communication + LLM planner for TA-RWARE (batch, 3-stage, limited-bandwidth).

This version includes P0/P1/P2 fixes:

P0 (must-have stability):
- Stage1/Stage2 LLM calls wrapped in try/except -> whole-plan fallback on any exception.
- Stage1 candidate construction filters by valid_action_masks (no illegal racks sent to LLM).
- Stage2 sanitize de-duplicates options per request.

P1 (trigger stability after fallback):
- After whole-plan fallback, this planner clears its own active_assignments and updates last_communication_step
  to avoid immediate re-trigger loops and stale internal state.

P2 (academic + analysis):
- Computes and stores objective scores for Stage3 output:
    num_assignments, sum_sync_cost, sum_eta_gap, max_sync_cost
  (Recorded into last_final_plan["objective_scores"] when available; runner can log it via comm_final_plan.)

Strictness policy:
- Only semantic-preserving sanitization. No decision-making repairs.
- If LLM output is invalid (structure or outside communicated options), we fallback WHOLE plan to rule-based symbiotic.

Stages:
  Stage 1 (AGV LLM): propose per-request primary + backups from per-request candidates (eta_agv only).
  Stage 2 (Picker LLM): output per-option picker-side support feedback (<=3 options), not binding assignments.
  Stage 3 (Commitment LLM): revise or retain AGV proposals under picker feedback.

Environment assumptions:
- `state` is produced by `StateBuilder` and contains:
  state["agents"] list of {id,type,busy,carrying,has_delivered,target,...}
  state["goal_ids"], state["requests_rack_ids_topk"], state["empty_rack_ids_topk"]
  state["valid_action_masks"], state["cost_table"] with keys "agv"/"picker"
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, List, Optional, Set, Tuple

from symco.llm.vllm_client import VLLMChatClient, load_vllm_config_from_env
from symco.planners.prompts import (
    stage1_system_prompt_v0,
    stage1_system_prompt_v1,
    stage2_system_prompt_v0,
    stage2_system_prompt_v1,
    stage3_system_prompt_v0,
    stage3_system_prompt_v1,
)
from symco.planners.symbiotic_planner import SymbioticPlanner as RuleSymbioticPlanner


# ----------------------------
# Config
# ----------------------------

class _DictMessage:
    def __init__(self, payload: dict):
        self._payload = payload

    def to_dict(self) -> dict:
        return self._payload




@dataclass(slots=True)
class SymbioticCommLLMPlannerConfig:
    # Candidate budget
    stage1_pool_k: int = 8                   # candidates per request into Stage1 prompt
    stage1_backups: int = 2                  # backups per request from the model (sanitizer caps)
    stage2_picker_options_per_rack: int = 3  # top P pickers per rack candidate (by eta_picker)
    stage2_max_options_per_request: int = 3  # keep up to 3 joint options per request (align with Stage1 candidates)
    max_requests_per_batch: int = 3         # mini-batch size within one communication round

    # Communication triggering
    idle_probe_gap_steps: int = 20   # 当没有可用 picker 时，允许低频探测

    # Output constraints
    unique_picker: bool = True
    unique_rack: bool = True
    enable_rationale: bool = False

    # Debug
    debug: bool = False


# ----------------------------
# Planner
# ----------------------------

class SymbioticCommLLMPlanner:
    """
    Symbiotic comm + LLM planner.

    Runner-compatible debug fields:
      - last_request: Stage1 sanitized message dict (or None)
      - last_response: Stage2 sanitized message dict (or None)
      - last_final_plan: Stage3 sanitized message dict (or fallback marker)
      - last_communication_triggered: bool
      - last_used_fallback: bool
    """

    def __init__(self, config: SymbioticCommLLMPlannerConfig | None = None):
        self.config = config or SymbioticCommLLMPlannerConfig()
        self.enable_rationale = getattr(self.config, "enable_rationale", False)

        vcfg = load_vllm_config_from_env()
        self.agv_client = VLLMChatClient(vcfg)
        self.picker_client = VLLMChatClient(vcfg)
        self.final_client = VLLMChatClient(vcfg)

        self._reset_episode_state()

    def _reset_episode_state(self) -> None:
        """Reset per-episode mutable planner state without recreating LLM clients."""
        # Communication-trigger state
        self.step_counter: int = 0
        self.last_communication_step: int = -10**9
        self.last_idle_probe_step: int = -10**9
        self.last_available_picker_count: int = 0
        self.active_assignments: Dict[int, Dict[str, Any]] = {}  # agv_id -> {picker_id,rack_id,purpose,start_step}
        self._last_registered_assignment_by_agv: Dict[int, Dict[str, Any]] = {}
        self._last_dropped_assignment_by_agv: Dict[int, Dict[str, Any]] = {}

        # Region mapping for load balancing
        self.rack_to_region = None   # 将在 plan 中从 state 读取
        self.region_to_racks = None

        # Runner logging
        self.last_request: Optional[dict] = None
        self.last_response: Optional[dict] = None
        self.last_final_plan: Optional[dict] = None
        self.last_communication_triggered: bool = False
        self.last_used_fallback: bool = False
        self.last_trigger_reasons: list[str] = []
        self.trigger_reason_counts: Dict[str, int] = {}
        self.trigger_reason_steps: list[dict[str, Any]] = []
        self.last_no_communication_reason_trace: dict[str, Any] = {}

        # Whole-plan fallback (rule-based symbiotic). We'll instantiate per-fallback to avoid internal state coupling.
        # self._rule_fallback = RuleSymbioticPlanner()

    def reset(self) -> None:
        """Reset per-episode planner state without recreating LLM clients."""
        self._reset_episode_state()

    # ----------------------------
    # Public API
    # ----------------------------

    def plan(self, state: dict[str, Any]) -> list[int]:
        """
        Return macro actions ordered by agent id.

        When no communication is triggered, returns all zeros (keep current missions).
        """
        self.step_counter += 1

        # 首次调用时从 state 读取区域映射
        if self.rack_to_region is None:
            self.rack_to_region = state.get("rack_to_region", {})
            self.region_to_racks = state.get("region_to_racks", {})

        self._update_active_assignments_from_state(state)

        if not self._should_trigger_communication(state):
            self.last_communication_triggered = False
            self.last_used_fallback = False
            self.last_request = None
            self.last_response = None
            self.last_final_plan = None
            fixed_only_actions = self._assemble_actions_from_assignments(state, assignments=[])
            self._debug_print_delivery_flow(
                state=state,
                batch_requests=[],
                returned_actions=fixed_only_actions,
                branch="NO_COMMUNICATION",
            )
            return fixed_only_actions
        

        self.last_communication_triggered = True
        self.last_used_fallback = False
        if self.config.debug:
            print(
                "COMM_TRIGGER_REASONS\n",
                json.dumps(self.last_trigger_reasons, ensure_ascii=False, indent=2),
            )

        # Build batch request contexts (deterministic)
        batch_requests = self._build_batch_requests(state)

        # If nothing to negotiate, just do deterministic fixed actions (goal moves etc.)
        if not batch_requests:
            self.last_request = {"requests": []}
            self.last_response = {"responses": []}
            self.last_final_plan = _DictMessage({
                "assignments": [],
                "skipped": [],
                "explanation": "No cooperative requests.",
                "objective_scores": {
                    "num_assignments": 0,
                    "sum_sync_cost": 0,
                    "sum_eta_gap": 0,
                    "max_sync_cost": 0
                },
            })
            actions = self._assemble_actions_from_assignments(state, assignments=[])
            if self._has_nonzero_actions(actions):
                self._mark_communication_effective()
            self._debug_print_delivery_flow(
                state=state,
                batch_requests=batch_requests,
                returned_actions=actions,
                branch="COMM_WITHOUT_BATCH_REQUESTS",
            )
            return actions

        request_batches = self._chunk_requests(
            batch_requests,
            max(1, int(self.config.max_requests_per_batch)),
        )
        if self.config.debug:
            print(
                "COMMUNICATION_MINI_BATCHES\n",
                json.dumps(
                    [
                        [str(req.get("request_id", "")) for req in batch]
                        for batch in request_batches
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
            )

        all_stage1_requests: list[dict[str, Any]] = []
        all_stage2_responses: list[dict[str, Any]] = []
        aggregate_stage3_requests: list[dict[str, Any]] = []
        aggregate_skipped: list[str] = []
        all_assignments: list[dict[str, Any]] = []
        batch_failure_reasons: list[str] = []

        reserved_agv_ids_this_round: set[int] = set()
        reserved_picker_ids_this_round: set[int] = set()
        reserved_rack_ids_this_round: set[int] = set()
        protected_picker_bindings = self._protected_picker_bindings_for_waiting_agvs(state)
        protected_picker_ids = {int(picker_id) for picker_id in protected_picker_bindings.values()}
        active_reserved_picker_ids = {
            int(assignment.get("picker_id", -1))
            for assignment in self.active_assignments.values()
            if isinstance(assignment, dict) and self._safe_int(assignment.get("picker_id")) > 0
        }
        available_pickers_for_round = [
            agent
            for agent in self._sorted_agents(state)
            if agent.get("type") == "PICKER"
            and not bool(agent.get("busy", False))
            and int(agent["id"]) not in active_reserved_picker_ids
            and int(agent["id"]) not in protected_picker_ids
        ]
        remaining_assignment_budget = int(len(available_pickers_for_round))

        for batch_index, request_batch in enumerate(request_batches, start=1):
            batch_request_ids = [str(req.get("request_id", "")) for req in request_batch]
            if self.config.debug:
                print(
                    f"BATCH_{batch_index}_REQUESTS\n",
                    json.dumps(batch_request_ids, ensure_ascii=False, indent=2),
                )

            stage1_payload = self._build_stage1_payload(
                state,
                request_batch,
                reserved_agv_ids=reserved_agv_ids_this_round,
                reserved_picker_ids=reserved_picker_ids_this_round,
            )
            if self.config.debug:
                print(f"BATCH_{batch_index}_STAGE1_PAYLOAD\n", json.dumps(stage1_payload, ensure_ascii=False, indent=2))

            stage1_raw: dict[str, Any] = {"requests": []}
            try:
                stage1_raw = self.agv_client.chat_json(
                    self._stage1_system_prompt(),
                    self._stage1_user_prompt(stage1_payload),
                )
            except Exception as exc:
                self.last_used_fallback = True
                batch_failure_reasons.append(f"batch{batch_index}: Stage1 LLM exception: {exc}")
                if self.config.debug:
                    print(f"BATCH_{batch_index}_FAILURE: Stage1 LLM exception: {exc}")
                continue

            if self.config.debug:
                print(f"BATCH_{batch_index}_STAGE1_RAW\n", json.dumps(stage1_raw, ensure_ascii=False, indent=2))

            stage1_bundle = self._sanitize_stage1_output(stage1_payload, stage1_raw)
            all_stage1_requests.extend(stage1_bundle)

            if not stage1_bundle:
                self.last_used_fallback = True
                batch_failure_reasons.append(f"batch{batch_index}: Stage1 produced no valid proposals.")
                if self.config.debug:
                    print(f"BATCH_{batch_index}_FAILURE: Stage1 produced no valid proposals.")
                continue

            stage2_payload = self._build_stage2_payload(
                state,
                stage1_bundle,
                reserved_picker_ids=reserved_picker_ids_this_round,
                reserved_rack_ids=reserved_rack_ids_this_round,
            )
            if self.config.debug:
                print(f"BATCH_{batch_index}_STAGE2_PAYLOAD\n", json.dumps(stage2_payload, ensure_ascii=False, indent=2))

            stage2_raw: dict[str, Any] = {"responses": []}
            try:
                stage2_raw = self.picker_client.chat_json(
                    self._stage2_system_prompt(),
                    self._stage2_user_prompt(stage2_payload),
                )
            except Exception as exc:
                self.last_used_fallback = True
                batch_failure_reasons.append(f"batch{batch_index}: Stage2 LLM exception: {exc}")
                if self.config.debug:
                    print(f"BATCH_{batch_index}_FAILURE: Stage2 LLM exception: {exc}")
                continue

            if self.config.debug:
                print(f"BATCH_{batch_index}_STAGE2_RAW\n", json.dumps(stage2_raw, ensure_ascii=False, indent=2))

            stage2_responses = self._sanitize_stage2_output(stage2_payload, stage2_raw)
            all_stage2_responses.extend(stage2_responses)
            supportable_requests_in_this_batch = sum(
                1
                for resp in stage2_responses
                if isinstance(resp, dict) and str(resp.get("overall_support", "")).upper() == "SUPPORT"
            )
            remaining_assignment_budget_before_stage3 = int(remaining_assignment_budget)
            max_assignments_this_batch = min(
                int(supportable_requests_in_this_batch),
                int(remaining_assignment_budget),
            )

            if self._all_requests_unsupported(stage2_responses):
                stage3_final = self._build_no_assignment_batch_plan(stage2_responses)
                aggregate_skipped.extend(
                    [
                        rid
                        for rid in stage3_final.get("skipped", [])
                        if isinstance(rid, str)
                    ]
                )
                if self.config.debug:
                    print(
                        f"BATCH_{batch_index}_STAGE3_SHORT_CIRCUIT\n",
                        json.dumps(stage3_final, ensure_ascii=False, indent=2),
                    )
                    print(
                        f"BATCH_{batch_index}_SELECTION_BUDGET "
                        f"{json.dumps({'supportable_requests_in_this_batch': int(supportable_requests_in_this_batch), 'remaining_assignment_budget_before_stage3': int(remaining_assignment_budget_before_stage3), 'max_assignments_this_batch': int(max_assignments_this_batch), 'kept_assignments_in_this_batch': 0, 'remaining_assignment_budget_after_batch': int(remaining_assignment_budget)}, ensure_ascii=False)}"
                    )
                continue

            if max_assignments_this_batch <= 0:
                stage3_final = {
                    "assignments": [],
                    "skipped": [
                        rid
                        for rid in (
                            resp.get("request_id")
                            for resp in stage2_responses
                            if isinstance(resp, dict)
                        )
                        if isinstance(rid, str)
                    ],
                    "explanation": "No remaining assignment budget for this batch.",
                }
                aggregate_skipped.extend(
                    [
                        rid
                        for rid in stage3_final.get("skipped", [])
                        if isinstance(rid, str)
                    ]
                )
                if self.config.debug:
                    print(
                        f"BATCH_{batch_index}_STAGE3_SHORT_CIRCUIT\n",
                        json.dumps(stage3_final, ensure_ascii=False, indent=2),
                    )
                    print(
                        f"BATCH_{batch_index}_SELECTION_BUDGET "
                        f"{json.dumps({'supportable_requests_in_this_batch': int(supportable_requests_in_this_batch), 'remaining_assignment_budget_before_stage3': int(remaining_assignment_budget_before_stage3), 'max_assignments_this_batch': int(max_assignments_this_batch), 'kept_assignments_in_this_batch': 0, 'remaining_assignment_budget_after_batch': int(remaining_assignment_budget)}, ensure_ascii=False)}"
                    )
                continue

            stage3_payload = self._build_stage3_payload(
                state,
                stage1_bundle,
                stage2_payload,
                stage2_responses,
                max_assignments_this_batch=max_assignments_this_batch,
            )
            if self.config.debug:
                print(f"BATCH_{batch_index}_STAGE3_PAYLOAD\n", json.dumps(stage3_payload, ensure_ascii=False, indent=2))

            stage3_raw: dict[str, Any] = {"assignments": [], "skipped": []}
            try:
                stage3_raw = self.final_client.chat_json(
                    self._stage3_system_prompt(),
                    self._stage3_user_prompt(stage3_payload),
                )
            except Exception as exc:
                self.last_used_fallback = True
                batch_failure_reasons.append(f"batch{batch_index}: Stage3 LLM exception: {exc}")
                if self.config.debug:
                    print(f"BATCH_{batch_index}_FAILURE: Stage3 LLM exception: {exc}")
                continue

            if self.config.debug:
                print(f"BATCH_{batch_index}_STAGE3_RAW\n", json.dumps(stage3_raw, ensure_ascii=False, indent=2))

            stage3_final = self._sanitize_stage3_output(stage3_payload, stage3_raw)
            if stage3_final is None:
                self.last_used_fallback = True
                batch_failure_reasons.append(f"batch{batch_index}: Stage3 invalid output.")
                if self.config.debug:
                    print(f"BATCH_{batch_index}_FAILURE: Stage3 invalid output.")
                continue

            aggregate_stage3_requests.extend(stage3_payload.get("requests", []))
            aggregate_skipped.extend(
                [
                    rid
                    for rid in stage3_final.get("skipped", [])
                    if isinstance(rid, str)
                ]
            )

            batch_assignments = self._dedupe_assignments(
                stage3_final.get("assignments", []),
                reserved_picker_ids=reserved_picker_ids_this_round,
                reserved_rack_ids=reserved_rack_ids_this_round,
            )
            all_assignments.extend(batch_assignments)
            remaining_assignment_budget -= len(batch_assignments)
            if remaining_assignment_budget < 0:
                remaining_assignment_budget = 0
            self._reserve_resources_from_assignments(
                batch_assignments,
                reserved_picker_ids_this_round,
                reserved_rack_ids_this_round,
            )
            for assignment in batch_assignments:
                if not isinstance(assignment, dict):
                    continue
                agv_id = self._safe_int(assignment.get("agv_id"))
                if agv_id > 0:
                    reserved_agv_ids_this_round.add(int(agv_id))

            if self.config.debug:
                print(
                    f"BATCH_{batch_index}_ASSIGNMENTS\n",
                    json.dumps(batch_assignments, ensure_ascii=False, indent=2),
                )
                print(
                    f"BATCH_{batch_index}_SELECTION_BUDGET "
                    f"{json.dumps({'supportable_requests_in_this_batch': int(supportable_requests_in_this_batch), 'remaining_assignment_budget_before_stage3': int(remaining_assignment_budget_before_stage3), 'max_assignments_this_batch': int(max_assignments_this_batch), 'kept_assignments_in_this_batch': int(len(batch_assignments)), 'remaining_assignment_budget_after_batch': int(remaining_assignment_budget)}, ensure_ascii=False)}"
                )
                print(
                    f"BATCH_{batch_index}_ALL_ASSIGNMENTS_AFTER_EXTEND\n",
                    json.dumps(all_assignments, ensure_ascii=False, indent=2),
                )

        self.last_request = _DictMessage({"requests": all_stage1_requests})
        self.last_response = _DictMessage({"responses": all_stage2_responses})

        aggregate_final_plan = {
            "assignments": all_assignments,
            "skipped": self._dedupe_strings(aggregate_skipped),
            "explanation": "; ".join(batch_failure_reasons) if batch_failure_reasons else "",
        }
        aggregate_stage3_payload = {
            "requests": aggregate_stage3_requests,
        }
        aggregate_final_plan["objective_scores"] = self._compute_objective_scores(
            aggregate_stage3_payload,
            aggregate_final_plan,
        )
        self.last_final_plan = _DictMessage(aggregate_final_plan)

        if self.config.debug:
            print("ALL_ASSIGNMENTS_BEFORE_ASSEMBLE\n", json.dumps(all_assignments, ensure_ascii=False, indent=2))
            print(
                "LAST_FINAL_PLAN_ASSIGNMENTS\n",
                json.dumps(aggregate_final_plan.get("assignments", []), ensure_ascii=False, indent=2),
            )

        tentative_actions = self._assemble_actions_from_assignments(state, assignments=all_assignments)
        if batch_requests and not self._has_nonzero_actions(tentative_actions):
            if self.config.debug:
                print("ROUND_LEVEL_FALLBACK_TRIGGERED: no executable actions after communication round")
                print("ROUND_LEVEL_BATCH_REQUESTS\n", json.dumps(batch_requests, ensure_ascii=False, indent=2))
                print("ROUND_LEVEL_ALL_ASSIGNMENTS\n", json.dumps(all_assignments, ensure_ascii=False, indent=2))
                print("ROUND_LEVEL_TENTATIVE_ACTIONS\n", json.dumps(tentative_actions, ensure_ascii=False, indent=2))
            fallback_actions = self._fallback_whole_plan(
                state,
                reason="Communication produced no executable actions.",
            )
            self._mark_communication_effective()
            self._debug_print_delivery_flow(
                state=state,
                batch_requests=batch_requests,
                returned_actions=fallback_actions,
                branch="COMMUNICATION_ROUND_FALLBACK",
            )
            return fallback_actions

        actions = tentative_actions
        if self._has_nonzero_actions(actions):
            self._mark_communication_effective()
        self._debug_print_delivery_flow(
            state=state,
            batch_requests=batch_requests,
            returned_actions=actions,
            branch="COMMUNICATION",
        )

        # Update short-term cooperative assignment tracking (for triggering)
        self._register_active_assignments_from_assignments(state, all_assignments)

        return actions

    # ----------------------------
    # Trigger logic
    # ----------------------------

    def _should_trigger_communication(self, state: dict[str, Any]) -> bool:
        agents = self._sorted_agents(state)
        request_racks = state.get("requests_rack_ids_topk", [])
        empty_racks = state.get("empty_rack_ids_topk", [])
        trigger_reasons: list[str] = []

        def preview_no_batch_requests_possible() -> bool:
            preview_requests, _ = self._build_batch_request_trace(state)
            return len(preview_requests) == 0

        def record_no_communication_trace(reason: str, **extra: Any) -> None:
            trace = {
                "step": int(self.step_counter),
                "reason": str(reason),
                "available_pickers": int(available_picker_count),
                "idle_need": bool(idle_need),
                "eligible_idle_agv_ids": sorted(int(x) for x in eligible_idle_agv_ids),
                "no_picker_idle_probe_gap_not_reached": bool(
                    extra.get("no_picker_idle_probe_gap_not_reached", False)
                ),
                "idle_probe_gap_remaining": (
                    int(extra["idle_probe_gap_remaining"])
                    if extra.get("idle_probe_gap_remaining") is not None
                    else None
                ),
                "no_batch_requests_possible": bool(extra.get("no_batch_requests_possible", False)),
            }
            self.last_no_communication_reason_trace = trace

        def finish(result: bool) -> bool:
            self.last_available_picker_count = int(available_picker_count)
            if result:
                self.last_no_communication_reason_trace = {}
                deduped_trigger_reasons: list[str] = []
                seen_reasons: set[str] = set()
                for reason in trigger_reasons:
                    if not isinstance(reason, str) or reason in seen_reasons:
                        continue
                    seen_reasons.add(reason)
                    deduped_trigger_reasons.append(reason)
                self.last_trigger_reasons = deduped_trigger_reasons
                for reason in self.last_trigger_reasons:
                    self.trigger_reason_counts[reason] = int(self.trigger_reason_counts.get(reason, 0)) + 1
                self.trigger_reason_steps.append(
                    {
                        "step": int(self.step_counter),
                        "reasons": list(self.last_trigger_reasons),
                        "available_pickers": int(available_picker_count),
                    }
                )
            else:
                self.last_trigger_reasons = []
            return bool(result)

        # ----------------------------
        # Compute currently available pickers for NEW cooperation
        # Keep this consistent with Stage2 availability semantics:
        #   picker must be not busy, not already reserved by an active assignment,
        #   and not protected for an already-waiting AGV.
        # ----------------------------
        protected_picker_bindings = self._protected_picker_bindings_for_waiting_agvs(state)
        protected_picker_ids = {int(picker_id) for picker_id in protected_picker_bindings.values()}
        reserved_picker_ids = {
            int(assignment.get("picker_id", -1))
            for assignment in self.active_assignments.values()
            if isinstance(assignment, dict) and self._safe_int(assignment.get("picker_id")) > 0
        }
        available_picker_count = sum(
            1
            for a in agents
            if a.get("type") == "PICKER"
            and not bool(a.get("busy", False))
            and int(a["id"]) not in reserved_picker_ids
            and int(a["id"]) not in protected_picker_ids
        )
        picker_became_available = (
            int(self.last_available_picker_count) == 0 and int(available_picker_count) > 0
        )

        # ----------------------------
        # Legal idle AGV need:
        # only idle, targetless AGVs that need a new cooperative LOAD/UNLOAD task.
        # ----------------------------
        eligible_idle_agv_ids: list[int] = []
        for a in agents:
            if a.get("type") != "AGV":
                continue
            if bool(a.get("busy", False)):
                continue

            target = int(a.get("target", 0) or 0)
            if target != 0:
                continue

            carrying = bool(a.get("carrying", False))
            has_delivered = bool(a.get("has_delivered", False))
            needs_new_task = (
                ((not carrying) and bool(request_racks))
                or (carrying and has_delivered and bool(empty_racks))
            )
            if needs_new_task:
                eligible_idle_agv_ids.append(int(a["id"]))

        idle_need = bool(eligible_idle_agv_ids)
        candidate_trigger_reason: Optional[str] = None

        if not idle_need:
            record_no_communication_trace(
                "no_idle_need",
                no_batch_requests_possible=preview_no_batch_requests_possible(),
            )
            return finish(False)

        if available_picker_count > 0:
            if picker_became_available:
                candidate_trigger_reason = "picker_became_available"
            else:
                candidate_trigger_reason = "ordinary_idle_need"

        # ----------------------------
        # Low-frequency probing when no picker is currently available:
        # when idle cooperative need exists but no picker is available for a new
        # commitment, probe only every `idle_probe_gap_steps` to avoid
        # communication every step under picker scarcity.
        # ----------------------------
        if candidate_trigger_reason is None:
            idle_probe_gap_elapsed = self.step_counter - self.last_idle_probe_step
            if idle_probe_gap_elapsed < int(self.config.idle_probe_gap_steps):
                idle_probe_gap_remaining = int(self.config.idle_probe_gap_steps) - int(idle_probe_gap_elapsed)
                if idle_probe_gap_remaining < 0:
                    idle_probe_gap_remaining = 0
                record_no_communication_trace(
                    "no_picker_idle_probe_gap_not_reached",
                    no_picker_idle_probe_gap_not_reached=True,
                    idle_probe_gap_remaining=idle_probe_gap_remaining,
                    no_batch_requests_possible=preview_no_batch_requests_possible(),
                )
                return finish(False)
            candidate_trigger_reason = "idle_probe_no_picker"

        if candidate_trigger_reason == "idle_probe_no_picker":
            self.last_idle_probe_step = self.step_counter
        trigger_reasons.append(candidate_trigger_reason)
        return finish(True)

    def _coords_tuple(self, coords: Any) -> Optional[Tuple[int, int]]:
        if not isinstance(coords, (list, tuple)) or len(coords) != 2:
            return None
        try:
            return int(coords[0]), int(coords[1])
        except (TypeError, ValueError):
            return None

    def _normalize_coords_yx(self, coords: Any) -> Optional[Tuple[int, int]]:
        return self._coords_tuple(coords)
    
    def _normalize_coords_xy(self, coords: Any) -> Optional[Tuple[int, int]]:
        """Normalize coordinates in (x, y) order into an integer tuple."""
        if not isinstance(coords, (list, tuple)) or len(coords) != 2:
            return None
        try:
            return int(coords[0]), int(coords[1])
        except (TypeError, ValueError):
            return None
        
    def _compute_region_load(self) -> dict[int, int]:
        """统计每个区域当前已分配的 AGV 数量（来自 active_assignments）"""
        if self.rack_to_region is None or not self.region_to_racks:
            return {}
        region_load = {region_idx: 0 for region_idx in self.region_to_racks.keys()}
        for assignment in self.active_assignments.values():
            rack_id = assignment.get("rack_id")
            if rack_id is not None and rack_id in self.rack_to_region:
                region = self.rack_to_region[rack_id]
                region_load[region] = region_load.get(region, 0) + 1
        return region_load

    def _count_nearby_idle_pickers(
        self,
        state: dict[str, Any],
        rack_id: int,
        eta_threshold: int = 15,
    ) -> int:
        """Count idle pickers whose path-based eta to the rack is within threshold.

        This uses picker cost_table entries, which are derived from env.find_path(...)
        in StateBuilder, so it reflects path-based reachability rather than
        Manhattan-distance proximity.
        """
        rack_id = self._safe_int(rack_id)
        if rack_id <= 0:
            return 0

        protected_picker_bindings = self._protected_picker_bindings_for_waiting_agvs(state)
        protected_picker_ids = {int(picker_id) for picker_id in protected_picker_bindings.values()}
        count = 0
        for agent in state.get("agents", []):
            if not isinstance(agent, dict):
                continue
            if agent.get("type") != "PICKER":
                continue
            if bool(agent.get("busy", False)):
                continue
            if int(agent.get("id", 0) or 0) in protected_picker_ids:
                continue

            picker_id = self._safe_int(agent.get("id"))
            if picker_id <= 0:
                continue

            eta_picker = self._safe_cost(
                self._agent_cost_map(state, "picker", picker_id),
                rack_id,
            )
            if eta_picker is None:
                continue

            if int(eta_picker) <= int(eta_threshold):
                count += 1

        return count
    

    def _debug_print_delivery_flow(
        self,
        state: dict[str, Any],
        batch_requests: list[dict[str, Any]],
        returned_actions: list[int],
        branch: str,
    ) -> None:
        """Print focused diagnostics around delivered shelves and empty-rack flow."""
        if not self.config.debug:
            return

        agents = self._sorted_agents(state)
        agvs = [agent for agent in agents if agent.get("type") == "AGV"]
        empty_racks = [int(x) for x in state.get("empty_rack_ids_topk", [])]
        has_delivered_agv = any(bool(agent.get("has_delivered", False)) for agent in agvs)
        if not has_delivered_agv and empty_racks:
            return

        print(f"\n===== DELIVERY FLOW DEBUG STEP {self.step_counter} [{branch}] =====")
        for agent in agvs:
            print(
                {
                    "id": agent.get("id"),
                    "busy": agent.get("busy"),
                    "carrying": agent.get("carrying"),
                    "has_delivered": agent.get("has_delivered"),
                    "target": agent.get("target"),
                    "target_coords_yx": agent.get("target_coords_yx"),
                    "coords_yx": agent.get("coords_yx"),
                }
            )
        print("empty_rack_ids_topk:", empty_racks)
        if not empty_racks:
            print("empty_rack_ids_topk is EMPTY")
        print("batch_requests:", batch_requests)
        fixed_direct_actions = self._build_fixed_direct_actions(state)
        print("fixed_direct_actions:", fixed_direct_actions)
        print("returned_actions:", returned_actions)
        mismatch_trace = self._build_coop_wait_mismatch_trace(state)
        if mismatch_trace:
            print("COOP_WAIT_MISMATCH_TRACE\n", json.dumps(mismatch_trace, ensure_ascii=False, indent=2))
        empty_rack_trace = self._build_empty_rack_disappearance_trace(state)
        if empty_rack_trace is not None:
            print("EMPTY_RACK_DISAPPEARANCE_TRACE\n", json.dumps(empty_rack_trace, ensure_ascii=False, indent=2))

    def _build_coop_wait_mismatch_trace(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        agents = self._sorted_agents(state)
        pickers_by_id = {
            int(agent["id"]): agent
            for agent in agents
            if isinstance(agent, dict) and agent.get("type") == "PICKER"
        }
        request_frontier = {int(x) for x in self._fresh_request_rack_frontier(state)}
        active_by_picker: dict[int, dict[str, Any]] = {}
        for agv_id, assignment in self.active_assignments.items():
            if not isinstance(assignment, dict):
                continue
            picker_id = self._safe_int(assignment.get("picker_id"))
            if picker_id > 0:
                active_by_picker[int(picker_id)] = {
                    "agv_id": int(agv_id),
                    **dict(assignment),
                }

        trace: list[dict[str, Any]] = []
        for agent in agents:
            if not isinstance(agent, dict) or agent.get("type") != "AGV":
                continue
            agv_id = int(agent["id"])
            busy = bool(agent.get("busy", False))
            carrying = bool(agent.get("carrying", False))
            has_delivered = bool(agent.get("has_delivered", False))
            target = int(agent.get("target", 0) or 0)
            pos = self._coords_tuple(agent.get("coords_yx"))
            target_coords = self._coords_tuple(agent.get("target_coords_yx"))
            if not (busy and (not carrying) and (not has_delivered) and target > 0 and pos is not None and target_coords is not None and pos == target_coords):
                continue

            active_assignment = self.active_assignments.get(int(agv_id))
            last_assignment = self._last_registered_assignment_by_agv.get(int(agv_id))
            dropped_assignment = self._last_dropped_assignment_by_agv.get(int(agv_id))
            assigned_picker_id = -1
            if isinstance(active_assignment, dict):
                assigned_picker_id = self._safe_int(active_assignment.get("picker_id"))
            elif isinstance(last_assignment, dict) and self._safe_int(last_assignment.get("rack_id")) == target:
                assigned_picker_id = self._safe_int(last_assignment.get("picker_id"))

            picker_state = pickers_by_id.get(int(assigned_picker_id)) if assigned_picker_id > 0 else None
            picker_target = self._safe_int(picker_state.get("target")) if isinstance(picker_state, dict) else 0
            picker_same_cell = (
                isinstance(picker_state, dict)
                and self._coords_tuple(picker_state.get("coords_yx")) == pos
            )
            diagnosis = "waiting_for_picker_unknown"
            if isinstance(picker_state, dict) and bool(picker_state.get("busy", False)) and picker_target == target:
                diagnosis = "picker_inbound_to_same_rack"
            elif isinstance(picker_state, dict) and bool(picker_state.get("busy", False)) and picker_target > 0 and picker_target != target:
                diagnosis = "suspected_picker_reassigned_elsewhere"
            elif not isinstance(active_assignment, dict):
                diagnosis = "suspected_assignment_dropped_no_active_binding"
            elif picker_same_cell:
                diagnosis = "picker_on_same_cell_closure_should_happen"

            trace.append(
                {
                    "agv_id": int(agv_id),
                    "rack_id": int(target),
                    "active_assignment": active_assignment,
                    "last_registered_assignment": last_assignment,
                    "last_dropped_assignment": dropped_assignment,
                    "rack_in_request_frontier": bool(int(target) in request_frontier),
                    "assigned_picker_id": int(assigned_picker_id) if assigned_picker_id > 0 else None,
                    "assigned_picker_state": (
                        {
                            "busy": bool(picker_state.get("busy", False)),
                            "target": int(picker_target),
                            "coords_yx": picker_state.get("coords_yx"),
                            "target_coords_yx": picker_state.get("target_coords_yx"),
                        }
                        if isinstance(picker_state, dict)
                        else None
                    ),
                    "assigned_picker_active_assignment": (
                        active_by_picker.get(int(assigned_picker_id))
                        if assigned_picker_id > 0
                        else None
                    ),
                    "diagnosis": diagnosis,
                }
            )
        return trace

    def _build_empty_rack_disappearance_trace(self, state: dict[str, Any]) -> Optional[dict[str, Any]]:
        empty_racks_all = [int(x) for x in state.get("empty_rack_ids_all", state.get("empty_rack_ids_topk", []))]
        if empty_racks_all:
            return None

        semantic = state.get("empty_rack_semantic_trace")
        if not isinstance(semantic, dict):
            return None

        location_evaluations = semantic.get("location_evaluations", [])
        blocked_by_reason: dict[str, int] = {}
        blocked_hole_samples: list[dict[str, Any]] = []
        for item in location_evaluations:
            if not isinstance(item, dict):
                continue
            reasons = item.get("excluded_reasons", [])
            if not isinstance(reasons, list):
                reasons = []
            for reason in reasons:
                blocked_by_reason[str(reason)] = int(blocked_by_reason.get(str(reason), 0)) + 1
            if "agv_on_cell_is_loading_or_noop_with_carried_shelf" in reasons and len(blocked_hole_samples) < 8:
                blocked_hole_samples.append(
                    {
                        "location_id": self._safe_int(item.get("location_id")),
                        "coords_yx": item.get("coords_yx"),
                        "agv_id": self._safe_int(item.get("agv_layer_id")),
                        "agv_req_action": item.get("agv_req_action"),
                        "carried_shelf_layer_id": self._safe_int(item.get("carried_shelf_layer_id")),
                    }
                )

        coords_to_loc_id: dict[tuple[int, int], int] = {}
        for loc_id_str, coords_xy in state.get("location_coords_xy", {}).items():
            if not isinstance(coords_xy, list) or len(coords_xy) != 2:
                continue
            try:
                loc_id = int(loc_id_str)
                x = int(coords_xy[0])
                y = int(coords_xy[1])
            except (TypeError, ValueError):
                continue
            coords_to_loc_id[(y, x)] = int(loc_id)

        carrying_agv_contexts: list[dict[str, Any]] = []
        for agent in self._sorted_agents(state):
            if not isinstance(agent, dict) or agent.get("type") != "AGV":
                continue
            if not bool(agent.get("carrying", False)):
                continue
            coords_yx = self._coords_tuple(agent.get("coords_yx"))
            current_loc_id = coords_to_loc_id.get(coords_yx) if coords_yx is not None else None
            carrying_agv_contexts.append(
                {
                    "agv_id": self._safe_int(agent.get("id")),
                    "busy": bool(agent.get("busy", False)),
                    "has_delivered": bool(agent.get("has_delivered", False)),
                    "target": self._safe_int(agent.get("target")),
                    "coords_yx": agent.get("coords_yx"),
                    "target_coords_yx": agent.get("target_coords_yx"),
                    "current_loc_id": int(current_loc_id) if current_loc_id is not None else None,
                    "at_target": (
                        coords_yx is not None
                        and coords_yx == self._coords_tuple(agent.get("target_coords_yx"))
                    ),
                }
            )

        return {
            "step": int(self.step_counter),
            "total_empty_racks_in_environment": 0,
            "request_queue_shelf_count": len(semantic.get("request_queue_shelf_ids", [])),
            "carried_shelf_count": len(semantic.get("carried_shelf_ids", [])),
            "blocked_reason_counts": blocked_by_reason,
            "blocked_hole_samples": blocked_hole_samples,
            "carrying_agv_contexts": carrying_agv_contexts,
        }

    def _is_assignment_completed(self, agv_state: dict[str, Any], assignment: dict[str, Any]) -> bool:
        purpose = str(assignment.get("purpose", ""))
        carrying = bool(agv_state.get("carrying", False))
        if purpose == "LOAD":
            return carrying
        if purpose == "UNLOAD":
            return not carrying
        return True

    def _matching_assignment_for_context(
        self,
        agv_state: dict[str, Any],
        assignment: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        """Return assignment only when it still matches the AGV's current cooperative context."""
        if not isinstance(assignment, dict):
            return None
        purpose = self._cooperative_purpose_for_agv(agv_state)
        target = self._safe_int(agv_state.get("target"))
        if purpose is None or target <= 0:
            return None
        if str(assignment.get("purpose", "")).upper() != str(purpose).upper():
            return None
        if self._safe_int(assignment.get("rack_id")) != int(target):
            return None
        if self._safe_int(assignment.get("picker_id")) <= 0:
            return None
        return assignment

    def _preferred_picker_for_context(
        self,
        state: dict[str, Any],
        agv_id: int,
        rack_id: int,
        purpose: str,
    ) -> int:
        """Return the most trusted picker bound to the current AGV/rack/purpose context."""
        agents = self._sorted_agents(state)
        agv_state = next(
            (
                agent
                for agent in agents
                if isinstance(agent, dict)
                and agent.get("type") == "AGV"
                and self._safe_int(agent.get("id")) == int(agv_id)
            ),
            None,
        )
        if not isinstance(agv_state, dict):
            return -1

        candidates = [
            self.active_assignments.get(int(agv_id)),
            self._last_registered_assignment_by_agv.get(int(agv_id)),
        ]
        for assignment in candidates:
            if not isinstance(assignment, dict):
                continue
            if str(assignment.get("purpose", "")).upper() != str(purpose).upper():
                continue
            if self._safe_int(assignment.get("rack_id")) != int(rack_id):
                continue
            picker_id = self._safe_int(assignment.get("picker_id"))
            if picker_id > 0:
                return int(picker_id)
        return -1

    def _update_active_assignments_from_state(self, state: dict[str, Any]) -> None:
        """Keep only assignments that still match the AGV state."""
        agents = self._sorted_agents(state)
        agvs_by_id = {int(a["id"]): a for a in agents if a.get("type") == "AGV"}
        new_active: Dict[int, Dict[str, Any]] = {}
        for agv_id, assignment in self.active_assignments.items():
            agv_state = agvs_by_id.get(int(agv_id))
            if agv_state is None:
                self._last_dropped_assignment_by_agv[int(agv_id)] = {
                    **dict(assignment),
                    "drop_reason": "agv_missing",
                    "drop_step": int(self.step_counter),
                }
                continue
            if self._is_assignment_completed(agv_state, assignment):
                self._last_dropped_assignment_by_agv[int(agv_id)] = {
                    **dict(assignment),
                    "drop_reason": "completed",
                    "drop_step": int(self.step_counter),
                }
                continue
            if self._matching_assignment_for_context(agv_state, assignment) is not None:
                new_active[int(agv_id)] = dict(assignment)
                continue
            current_target = int(agv_state.get("target", 0) or 0)
            rack_id = int(assignment.get("rack_id", 0))
            if current_target not in {0, rack_id}:
                self._last_dropped_assignment_by_agv[int(agv_id)] = {
                    **dict(assignment),
                    "drop_reason": "target_mismatch",
                    "drop_step": int(self.step_counter),
                    "current_target": int(current_target),
                }
                continue
            new_active[int(agv_id)] = assignment
        self.active_assignments = new_active

    def _register_active_assignments_from_assignments(self, state: dict[str, Any], assignments: list[dict[str, Any]]) -> None:
        """Register assignments as short-term cooperative commitments."""
        agents = self._sorted_agents(state)
        agents_by_id = {int(a["id"]): a for a in agents}
        new_active = dict(self.active_assignments)

        for item in assignments:
            if not isinstance(item, dict):
                continue
            agv_id = self._safe_int(item.get("agv_id"))
            picker_id = self._safe_int(item.get("picker_id"))
            rack_id = self._safe_int(item.get("rack_id"))
            if agv_id <= 0 or picker_id <= 0 or rack_id <= 0:
                continue
            agv_state = agents_by_id.get(int(agv_id))
            if agv_state is None or agv_state.get("type") != "AGV":
                continue
            purpose = self._cooperative_purpose_for_agv(agv_state)
            if purpose is None:
                continue
            current_target = int(agv_state.get("target", 0) or 0)
            if bool(agv_state.get("busy", False)) and current_target > 0 and int(rack_id) != int(current_target):
                if self.config.debug:
                    print(
                        "ASSIGNMENT_REGISTRATION_SKIPPED_TARGET_MISMATCH\n",
                        json.dumps(
                            {
                                "step": int(self.step_counter),
                                "agv_id": int(agv_id),
                                "picker_id": int(picker_id),
                                "assignment_rack_id": int(rack_id),
                                "current_target": int(current_target),
                                "purpose": str(purpose),
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                    )
                continue

            existing = new_active.get(int(agv_id))
            if (
                existing is not None
                and int(existing.get("picker_id", -1)) == int(picker_id)
                and int(existing.get("rack_id", -1)) == int(rack_id)
                and str(existing.get("purpose", "")) == purpose
            ):
                continue

            new_active[int(agv_id)] = {
                "picker_id": int(picker_id),
                "rack_id": int(rack_id),
                "purpose": purpose,
                "start_step": self.step_counter,
            }
            self._last_registered_assignment_by_agv[int(agv_id)] = {
                "picker_id": int(picker_id),
                "rack_id": int(rack_id),
                "purpose": str(purpose),
                "start_step": int(self.step_counter),
            }

        self.active_assignments = new_active

    def _cooperative_purpose_for_agv(self, agv_state: dict[str, Any]) -> Optional[str]:
        carrying = bool(agv_state.get("carrying", False))
        has_delivered = bool(agv_state.get("has_delivered", False))
        if not carrying:
            return "LOAD"
        if carrying and has_delivered:
            return "UNLOAD"
        return None

    def _protected_picker_bindings_for_waiting_agvs(self, state: dict[str, Any]) -> dict[int, int]:
        """Return AGV->picker bindings that should be preserved for ongoing cooperative execution."""
        agents = self._sorted_agents(state)
        bindings: dict[int, int] = {}
        for agent in agents:
            if not isinstance(agent, dict) or agent.get("type") != "AGV":
                continue
            agv_id = int(agent.get("id", 0) or 0)
            if agv_id <= 0:
                continue
            busy = bool(agent.get("busy", False))
            carrying = bool(agent.get("carrying", False))
            has_delivered = bool(agent.get("has_delivered", False))
            target = int(agent.get("target", 0) or 0)
            if not busy or target <= 0:
                continue

            expected_purpose: Optional[str] = None
            if (not carrying) and (not has_delivered):
                expected_purpose = "LOAD"
            elif carrying and has_delivered:
                expected_purpose = "UNLOAD"
            if expected_purpose is None:
                continue

            picker_id = -1
            active_assignment = self.active_assignments.get(int(agv_id))
            if (
                isinstance(active_assignment, dict)
                and str(active_assignment.get("purpose", "")).upper() == expected_purpose
                and self._safe_int(active_assignment.get("rack_id")) == target
            ):
                picker_id = self._safe_int(active_assignment.get("picker_id"))
            else:
                last_assignment = self._last_registered_assignment_by_agv.get(int(agv_id))
                if (
                    isinstance(last_assignment, dict)
                    and str(last_assignment.get("purpose", "")).upper() == expected_purpose
                    and self._safe_int(last_assignment.get("rack_id")) == target
                ):
                    picker_id = self._safe_int(last_assignment.get("picker_id"))

            if picker_id > 0:
                bindings[int(agv_id)] = int(picker_id)
        return bindings

    # ----------------------------
    # Batch request building
    # ----------------------------

    def _active_claimed_rack_ids(self) -> set[int]:
        """Rack ids already claimed by active cooperative assignments."""
        return {
            int(assignment.get("rack_id", -1))
            for assignment in self.active_assignments.values()
            if isinstance(assignment, dict) and self._safe_int(assignment.get("rack_id")) > 0
        }

    def _fresh_request_rack_frontier(self, state: dict[str, Any]) -> list[int]:
        """Recompute the current open-request top-k frontier after excluding claimed racks."""
        request_racks_all = [
            int(x)
            for x in state.get("requests_rack_ids_all", state.get("requests_rack_ids_topk", []))
            if self._safe_int(x) > 0
        ]
        claimed_rack_ids = self._active_claimed_rack_ids()
        open_request_racks = [
            int(rack_id)
            for rack_id in request_racks_all
            if int(rack_id) not in claimed_rack_ids
        ]
        request_limit = len([int(x) for x in state.get("requests_rack_ids_topk", []) if self._safe_int(x) > 0])
        if request_limit <= 0 or len(open_request_racks) <= request_limit:
            return open_request_racks

        agents = self._sorted_agents(state)
        agv_coords = [
            self._normalize_coords_yx(agent.get("coords_yx"))
            for agent in agents
            if isinstance(agent, dict) and agent.get("type") == "AGV"
        ]
        agv_coords = [coords for coords in agv_coords if coords is not None]
        if not agv_coords:
            return open_request_racks[:request_limit]

        location_coords_xy = state.get("location_coords_xy", {})

        def sort_key(rack_id: int) -> tuple[int, int]:
            coords_xy = location_coords_xy.get(str(int(rack_id)))
            rack_coords = self._normalize_coords_xy(coords_xy)
            if rack_coords is None:
                return 10**9, int(rack_id)
            min_distance = min(
                abs(int(coords[0]) - int(rack_coords[0])) + abs(int(coords[1]) - int(rack_coords[1]))
                for coords in agv_coords
            )
            return int(min_distance), int(rack_id)

        return sorted(open_request_racks, key=sort_key)[:request_limit]

    def _build_batch_request_trace(
        self,
        state: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Build batch requests plus per-AGV exclusion diagnostics."""
        agents = self._sorted_agents(state)
        req_racks = self._fresh_request_rack_frontier(state)
        empty_racks = [int(x) for x in state.get("empty_rack_ids_topk", [])]

        requests: list[dict[str, Any]] = []
        exclusion_trace: list[dict[str, Any]] = []
        for a in agents:
            if a.get("type") != "AGV":
                continue

            agv_id = int(a["id"])
            trace_entry: dict[str, Any] = {
                "agv_id": int(agv_id),
                "busy": bool(a.get("busy", False)),
                "carrying": bool(a.get("carrying", False)),
                "has_delivered": bool(a.get("has_delivered", False)),
                "target": int(a.get("target", 0) or 0),
                "active_assignment_present": isinstance(self.active_assignments.get(int(agv_id)), dict),
                "excluded_reasons": [],
            }
            if bool(a.get("busy", False)):
                trace_entry["excluded_reasons"].append("excluded_because_busy")
                exclusion_trace.append(trace_entry)
                continue
            target = int(a.get("target", 0) or 0)
            if target != 0:
                trace_entry["excluded_reasons"].append("excluded_because_target_nonzero")
                exclusion_trace.append(trace_entry)
                continue

            carrying = bool(a.get("carrying", False))
            has_delivered = bool(a.get("has_delivered", False))
            if carrying and has_delivered:
                purpose = "UNLOAD"
            elif carrying and (not has_delivered):
                # carrying and not delivered -> fixed goal move, not cooperative request
                trace_entry["excluded_reasons"].append("excluded_because_carrying_state_purpose_logic_blocked")
                exclusion_trace.append(trace_entry)
                continue
            else:
                purpose = "LOAD"
            trace_entry["purpose"] = str(purpose)

            if purpose == "UNLOAD":
                pool = empty_racks
                if not pool:
                    trace_entry["excluded_reasons"].append("excluded_because_no_empty_rack_candidates")
            else:
                pool = req_racks

            if not pool:
                if "excluded_because_no_empty_rack_candidates" not in trace_entry["excluded_reasons"]:
                    trace_entry["excluded_reasons"].append("excluded_because_no_valid_candidate_pool")
                exclusion_trace.append(trace_entry)
                continue

            request_id = f"agv-{agv_id}-{purpose.lower()}"
            requests.append(
                {
                    "request_id": request_id,
                    "agv_id": agv_id,
                    "purpose": purpose,
                    "candidate_pool": pool,
                }
            )
            trace_entry["request_id"] = str(request_id)
            trace_entry["candidate_pool_size"] = int(len(pool))
            exclusion_trace.append(trace_entry)

        return requests, exclusion_trace

    def _build_batch_requests(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        """Deterministically build cooperative AGV requests for this step."""
        requests, exclusion_trace = self._build_batch_request_trace(state)
        if (
            self.config.debug
            and not requests
            and any(
                isinstance(agent, dict) and agent.get("type") == "AGV" and bool(agent.get("busy", False))
                for agent in self._sorted_agents(state)
            )
        ):
            pass
        return requests

    def _chunk_requests(self, requests: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
        """Split requests into deterministic mini-batches."""
        if chunk_size <= 0:
            chunk_size = 1
        return [requests[idx : idx + chunk_size] for idx in range(0, len(requests), chunk_size)]

    def _reserve_resources_from_assignments(
        self,
        assignments: list[dict[str, Any]],
        reserved_picker_ids: set[int],
        reserved_rack_ids: set[int],
    ) -> None:
        """Update per-round reserved resources from committed assignments."""
        for item in assignments:
            if not isinstance(item, dict):
                continue
            picker_id = self._safe_int(item.get("picker_id"))
            rack_id = self._safe_int(item.get("rack_id"))
            if picker_id > 0:
                reserved_picker_ids.add(int(picker_id))
            if rack_id > 0:
                reserved_rack_ids.add(int(rack_id))

    def _dedupe_assignments(
        self,
        assignments: list[dict[str, Any]],
        reserved_picker_ids: set[int] | None = None,
        reserved_rack_ids: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Drop assignments that would reuse already-reserved picker/rack resources."""
        reserved_picker_ids = reserved_picker_ids or set()
        reserved_rack_ids = reserved_rack_ids or set()
        seen_request_ids: set[str] = set()
        local_pickers = set(reserved_picker_ids)
        local_racks = set(reserved_rack_ids)
        deduped: list[dict[str, Any]] = []

        for item in assignments:
            if not isinstance(item, dict):
                continue
            request_id = item.get("request_id")
            if not isinstance(request_id, str) or request_id in seen_request_ids:
                continue
            picker_id = self._safe_int(item.get("picker_id"))
            rack_id = self._safe_int(item.get("rack_id"))
            agv_id = self._safe_int(item.get("agv_id"))
            if agv_id <= 0 or picker_id <= 0 or rack_id <= 0:
                continue
            if picker_id in local_pickers or rack_id in local_racks:
                continue
            deduped.append(
                {
                    "request_id": request_id,
                    "agv_id": int(agv_id),
                    "picker_id": int(picker_id),
                    "rack_id": int(rack_id),
                }
            )
            seen_request_ids.add(request_id)
            local_pickers.add(int(picker_id))
            local_racks.add(int(rack_id))

        return deduped

    def _dedupe_strings(self, values: list[str]) -> list[str]:
        """Preserve order while removing duplicate strings."""
        seen: set[str] = set()
        deduped: list[str] = []
        for value in values:
            if not isinstance(value, str) or value in seen:
                continue
            seen.add(value)
            deduped.append(value)
        return deduped

    def _mark_communication_effective(self) -> None:
        """Record the most recent step that produced an effective communication outcome."""
        self.last_communication_step = self.step_counter

    def _has_nonzero_actions(self, actions: list[int]) -> bool:
        """Return whether at least one macro-action is executable and non-zero.

        A communication round now falls back at the round level if it produces
        no executable nonzero actions after aggregation. This prevents idle
        deadlock caused by structurally non-empty but actionless results, such
        as all-zero fixed actions or fully skipped mini-batches.
        """
        return any(self._safe_int(action) > 0 for action in actions)

    def _all_requests_unsupported(self, stage2_responses: list[dict[str, Any]]) -> bool:
        """Return True when no request in this batch has any commit-worthy option.

        When Stage 2 determines that no request in the current mini-batch has any
        picker-supportable option, Stage 3 is skipped and a deterministic
        no-assignment outcome is produced. This is a normal control-flow outcome,
        not a fallback.
        """
        if not stage2_responses:
            return True

        for response in stage2_responses:
            if not isinstance(response, dict):
                continue
            options = response.get("options", [])
            if not isinstance(options, list):
                options = []
            for option in options:
                if not isinstance(option, dict):
                    continue
                support_level = str(option.get("support_level", "")).upper()
                if support_level in {"STRONG", "WEAK"}:
                    return False

            overall_support = str(response.get("overall_support", "")).upper()
            options_count = self._safe_int(response.get("options_count", 0))
            if overall_support == "SUPPORT" and options_count > 0:
                return False

        return True

    def _build_no_assignment_batch_plan(self, stage2_responses: list[dict[str, Any]]) -> dict[str, Any]:
        """Build a deterministic no-assignment result for an unsupported mini-batch."""
        skipped = [
            str(response.get("request_id"))
            for response in stage2_responses
            if isinstance(response, dict) and isinstance(response.get("request_id"), str)
        ]
        return {
            "assignments": [],
            "skipped": skipped,
            "explanation": "No picker-supportable options remained in this batch after Stage 2.",
        }

    # ----------------------------
    # Stage 1 payload + sanitize
    # ----------------------------

    def _build_stage1_payload(
        self,
        state: dict[str, Any],
        batch_requests: list[dict[str, Any]],
        reserved_agv_ids: set[int] | None = None,
        reserved_picker_ids: set[int] | None = None,
    ) -> dict[str, Any]:
        reserved_agv_ids = set() if reserved_agv_ids is None else {int(x) for x in reserved_agv_ids}
        reserved_picker_ids = set() if reserved_picker_ids is None else {int(x) for x in reserved_picker_ids}
        agents = self._sorted_agents(state)
        protected_picker_bindings = self._protected_picker_bindings_for_waiting_agvs(state)
        protected_picker_ids = {int(picker_id) for picker_id in protected_picker_bindings.values()}
        idle_agvs = sum(
            1
            for a in agents
            if a.get("type") == "AGV"
            and not bool(a.get("busy", False))
            and int(a["id"]) not in reserved_agv_ids
        )
        active_reserved_picker_ids = {
            int(assignment.get("picker_id", -1))
            for assignment in self.active_assignments.values()
            if isinstance(assignment, dict) and self._safe_int(assignment.get("picker_id")) > 0
        }
        idle_pickers = sum(
            1
            for a in agents
            if a.get("type") == "PICKER"
            and not bool(a.get("busy", False))
            and int(a["id"]) not in active_reserved_picker_ids
            and int(a["id"]) not in protected_picker_ids
            and int(a["id"]) not in reserved_picker_ids
        )
        picker_scarcity = "low" if idle_pickers >= 2 else "high"

        # 计算区域负载
        region_load = self._compute_region_load()
        

        valid_masks = state.get("valid_action_masks", [])
        agv_index_by_id: dict[int, int] = {}
        for idx, a in enumerate(agents):
            if a.get("type") == "AGV":
                agv_index_by_id[int(a["id"])] = idx

        req_payloads: list[dict[str, Any]] = []
        for req in batch_requests:
            agv_id = int(req["agv_id"])
            pool = [int(x) for x in req["candidate_pool"]]
            cost_map = self._agent_cost_map(state, "agv", agv_id)
            agent_index = agv_index_by_id.get(agv_id, -1)

            scored: list[tuple[int, int]] = []
            for rack_id in pool:
                if agent_index >= 0 and not self._is_valid_action(valid_masks, agent_index, rack_id):
                    continue
                eta = self._safe_cost(cost_map, rack_id)
                if eta is None:
                    continue
                scored.append((rack_id, int(eta)))

            scored.sort(key=lambda t: (t[1], t[0]))
            scored = scored[: max(0, int(self.config.stage1_pool_k))]

            candidates = []
            for rid, eta in scored:
                nearby_idle_pickers = self._count_nearby_idle_pickers(
                    state,
                    int(rid),
                    eta_threshold=15,
                )

                candidates.append(
                    {
                        "rack_id": rid,
                        "eta_agv": eta,
                        "region_id": self.rack_to_region.get(rid, -1) if self.rack_to_region else -1,
                        "nearby_idle_pickers": int(nearby_idle_pickers),
                    }
                )
            
            if not candidates:
                continue

            req_payloads.append(
                {
                    "request_id": req["request_id"],
                    "agv_id": agv_id,
                    "purpose": req["purpose"],
                    "candidates": candidates,
                }
            )

        return {
            "system_pressure": {
                "idle_agvs": int(idle_agvs),
                "idle_pickers": int(idle_pickers),
                "picker_scarcity": picker_scarcity,
                "active_cooperative_assignments": int(len(self.active_assignments)),
                "region_load": region_load,
            },
            "requests": req_payloads,
        }

    def _sanitize_stage1_output(self, payload: dict[str, Any], raw: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Strict semantic-preserving sanitization.

        Stage 1 model output is intentionally minimal:
        - request_id
        - primary_rack_id
        - backup_rack_ids
        - reason (V1 only)

        Request metadata and candidate details are restored from the payload
        during sanitization to reduce output tokens and improve schema stability.

        - Must output list in raw["requests"]
        - Each request_id must exist
        - primary/backup must come from communicated candidates
        - No decision fallback if missing/invalid (drop that request)
        """
        if not isinstance(raw, dict):
            return []

        raw_reqs = raw.get("requests")
        if not isinstance(raw_reqs, list):
            return []

        # Build allowed rack sets
        allowed_by_request: dict[str, Set[int]] = {}
        request_meta: dict[str, dict[str, Any]] = {}
        for item in payload.get("requests", []):
            if not isinstance(item, dict):
                continue
            rid = item.get("request_id")
            if not isinstance(rid, str):
                continue
            cands = item.get("candidates", [])
            allowed: set[int] = set()
            for c in cands:
                if isinstance(c, dict):
                    x = self._safe_int(c.get("rack_id"))
                    if x > 0:
                        allowed.add(int(x))
            if allowed:
                allowed_by_request[rid] = allowed
                request_meta[rid] = {
                    "agv_id": int(item.get("agv_id", 0)),
                    "purpose": str(item.get("purpose", "")),
                    "candidates": cands,
                }

        seen: set[str] = set()
        sanitized: list[dict[str, Any]] = []

        for item in raw_reqs:
            if not isinstance(item, dict):
                continue
            request_id = item.get("request_id")
            if not isinstance(request_id, str) or request_id in seen:
                continue
            if request_id not in allowed_by_request:
                continue

            primary = self._safe_int(item.get("primary_rack_id"))
            if primary <= 0 or primary not in allowed_by_request[request_id]:
                continue

            backups_in = item.get("backup_rack_ids", [])
            if not isinstance(backups_in, list):
                backups_in = []

            backups: list[int] = []
            for b in backups_in:
                bid = self._safe_int(b)
                if bid <= 0:
                    continue
                if bid == primary:
                    continue
                if bid not in allowed_by_request[request_id]:
                    continue
                if bid in backups:
                    continue
                backups.append(int(bid))
                if len(backups) >= int(self.config.stage1_backups):
                    break

            reason = item.get("reason", "")
            if reason is None:
                reason = ""
            reason = str(reason)

            meta = request_meta[request_id]
            sanitized.append(
                {
                    "request_id": request_id,
                    "agv_id": int(meta["agv_id"]),
                    "purpose": str(meta["purpose"]),
                    "primary_rack_id": int(primary),
                    "backup_rack_ids": backups,
                    "candidates": meta["candidates"],  # keep minimal for later building
                }
            )
            if self.enable_rationale:
                sanitized[-1]["reason"] = reason
            seen.add(request_id)

        return sanitized

    # ----------------------------
    # Stage 2 payload + sanitize
    # ----------------------------

    def _build_stage2_payload(
        self,
        state: dict[str, Any],
        stage1_bundle: list[dict[str, Any]],
        reserved_picker_ids: set[int] | None = None,
        reserved_rack_ids: set[int] | None = None,
    ) -> dict[str, Any]:
        """Build Stage 2 payload from the bounded rack set proposed by Stage 1.

        Stage 2 no longer re-searches over the full candidate pool. It only
        evaluates the primary + backups proposed by Stage 1, preserving the
        intended proposal -> feedback -> revision semantics.

        Stage 2 option construction is coverage-first: it preserves the bounded
        rack alternatives proposed by Stage 1 before adding extra picker
        variants. This keeps Stage 2 aligned with the intended
        proposal -> feedback -> revision semantics.
        """
        agents = self._sorted_agents(state)
        idle_agvs = sum(1 for a in agents if a.get("type") == "AGV" and not bool(a.get("busy", False)))
        reserved_picker_ids = set() if reserved_picker_ids is None else {int(x) for x in reserved_picker_ids}
        reserved_rack_ids = set() if reserved_rack_ids is None else {int(x) for x in reserved_rack_ids}
        protected_picker_bindings = self._protected_picker_bindings_for_waiting_agvs(state)
        protected_picker_ids = {int(picker_id) for picker_id in protected_picker_bindings.values()}
        # Reserve pickers already committed to active cooperative assignments so
        # an unfinished unload/load does not lose its paired picker mid-execution.
        active_reserved_picker_ids = {
            int(assignment.get("picker_id", -1))
            for assignment in self.active_assignments.values()
            if isinstance(assignment, dict) and self._safe_int(assignment.get("picker_id")) > 0
        }
        available_pickers = [
            a
            for a in agents
            if a.get("type") == "PICKER"
            and not bool(a.get("busy", False))
            and int(a["id"]) not in active_reserved_picker_ids
            and int(a["id"]) not in protected_picker_ids
            and int(a["id"]) not in reserved_picker_ids
        ]
        idle_pickers = len(available_pickers)
        picker_scarcity = "high" if idle_pickers < idle_agvs else "low"

        requests_payload: list[dict[str, Any]] = []
        for req in stage1_bundle:
            request_id = str(req["request_id"])
            agv_id = int(req["agv_id"])
            purpose = str(req["purpose"])
            protected_picker_id_for_request = int(protected_picker_bindings.get(int(agv_id), -1))
            selected_rack_ids: list[int] = []
            primary_rack_id = self._safe_int(req.get("primary_rack_id"))
            if primary_rack_id > 0:
                selected_rack_ids.append(int(primary_rack_id))
            for rack_id in req.get("backup_rack_ids", []):
                rack_id_int = self._safe_int(rack_id)
                if rack_id_int <= 0 or rack_id_int in selected_rack_ids:
                    continue
                selected_rack_ids.append(int(rack_id_int))

            eta_agv_by_rack: dict[int, int] = {}
            for candidate in req.get("candidates", []):
                if not isinstance(candidate, dict):
                    continue
                rack_id = self._safe_int(candidate.get("rack_id"))
                eta_agv = self._safe_int(candidate.get("eta_agv"))
                if rack_id > 0 and eta_agv >= 0:
                    eta_agv_by_rack[int(rack_id)] = int(eta_agv)

            pairs_by_rack: dict[int, list[dict[str, Any]]] = {}
            for rack_id in selected_rack_ids:
                eta_agv = eta_agv_by_rack.get(int(rack_id), -1)
                if rack_id <= 0 or eta_agv < 0:
                    continue
                if int(rack_id) in reserved_rack_ids:
                    continue

                rack_pairs: list[dict[str, Any]] = []
                candidate_pickers = list(available_pickers)
                if protected_picker_id_for_request > 0:
                    protected_picker_state = next(
                        (
                            agent
                            for agent in agents
                            if agent.get("type") == "PICKER"
                            and int(agent.get("id", 0) or 0) == int(protected_picker_id_for_request)
                            and not bool(agent.get("busy", False))
                            and int(agent.get("id", 0) or 0) not in reserved_picker_ids
                        ),
                        None,
                    )
                    if protected_picker_state is not None and all(
                        int(existing.get("id", 0) or 0) != int(protected_picker_id_for_request)
                        for existing in candidate_pickers
                    ):
                        candidate_pickers.append(protected_picker_state)

                for p in candidate_pickers:
                    pid = int(p["id"])
                    eta_picker = self._safe_cost(self._agent_cost_map(state, "picker", pid), rack_id)
                    if eta_picker is None:
                        continue
                    eta_picker = int(eta_picker)
                    rack_pairs.append(
                        {
                            "rack_id": int(rack_id),
                            "picker_id": int(pid),
                            "eta_agv": int(eta_agv),
                            "eta_picker": eta_picker,
                        }
                    )
                rack_pairs.sort(
                    key=lambda item: (
                        int(item["eta_picker"]),
                        int(item["picker_id"]),
                        int(item["rack_id"]),
                    )
                )
                if rack_pairs:
                    pairs_by_rack[int(rack_id)] = rack_pairs

            max_options = max(0, int(self.config.stage2_max_options_per_request))
            request_options: list[dict[str, Any]] = []
            selected_pairs: set[tuple[int, int]] = set()

            # Phase 1: preserve rack coverage with deterministic priority:
            # primary first, then backups in Stage1 order.
            for rack_id in selected_rack_ids:
                if len(request_options) >= max_options:
                    break
                rack_pairs = pairs_by_rack.get(int(rack_id), [])
                if not rack_pairs:
                    continue
                best_pair = rack_pairs[0]
                pair_key = (int(best_pair["rack_id"]), int(best_pair["picker_id"]))
                if pair_key in selected_pairs:
                    continue
                request_options.append(dict(best_pair))
                selected_pairs.add(pair_key)

            # Phase 2: fill remaining slots with extra picker variants.
            if len(request_options) < max_options:
                remaining_pairs: list[dict[str, Any]] = []
                for rack_id in selected_rack_ids:
                    rack_pairs = pairs_by_rack.get(int(rack_id), [])
                    if not rack_pairs:
                        continue
                    for pair in rack_pairs[1:]:
                        pair_key = (int(pair["rack_id"]), int(pair["picker_id"]))
                        if pair_key in selected_pairs:
                            continue
                        remaining_pairs.append(dict(pair))

                remaining_pairs.sort(
                    key=lambda item: (
                        int(item["eta_picker"]),
                        int(item["rack_id"]),
                        int(item["picker_id"]),
                    )
                )
                for pair in remaining_pairs:
                    if len(request_options) >= max_options:
                        break
                    pair_key = (int(pair["rack_id"]), int(pair["picker_id"]))
                    if pair_key in selected_pairs:
                        continue
                    request_options.append(pair)
                    selected_pairs.add(pair_key)

            for idx, option in enumerate(request_options):
                eta_agv = int(option.get("eta_agv", -1))
                eta_picker = int(option.get("eta_picker", -1))
                option["option_id"] = f"OPT_{idx}"
                option["sync_cost"] = max(eta_agv, eta_picker) if eta_agv >= 0 and eta_picker >= 0 else -1
                option["eta_gap"] = abs(eta_agv - eta_picker) if eta_agv >= 0 and eta_picker >= 0 else -1

            requests_payload.append(
                {
                    "request_id": request_id,
                    "stage1_proposal": {
                        "primary_rack_id": int(primary_rack_id) if primary_rack_id > 0 else 0,
                        "backup_rack_ids": [
                            int(rack_id)
                            for rack_id in selected_rack_ids
                            if int(rack_id) != int(primary_rack_id)
                        ],
                    },
                    "options": request_options,
                    "options_count": int(len(request_options)),
                }
            )
            if self.enable_rationale:
                requests_payload[-1]["agv_reason"] = str(req.get("reason", ""))

        return {
            "system_pressure": {
                "idle_pickers": int(idle_pickers),
                "picker_scarcity": picker_scarcity,
            },
            "_stage3_system_pressure": {
                "idle_agvs": int(idle_agvs),
                "idle_pickers": int(idle_pickers),
                "active_cooperative_assignments": int(len(self.active_assignments)),
            },
            "requests": requests_payload,
        }

    def _sanitize_stage2_output(self, payload: dict[str, Any], raw: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Strict Stage 2 sanitization for per-option picker-side feedback:
        Stage 2 model output is intentionally minimal:
        - request_id
        - overall_support
        - option_feedback
        - reason

        Option metadata is restored from the payload during sanitization to
        reduce output tokens and improve schema stability.

        - raw["responses"] must be a list (or wrapper {"response": {...}})
        - each response.request_id must exist
        - option_feedback entries must reference only communicated option_ids
        - support_level must be STRONG / WEAK / REJECT
        - missing options are filled as REJECT
        - if no legal option feedback exists for a request, all options become REJECT
        - overall_support is recomputed conservatively from the sanitized support levels
        - ``options`` is retained only as a temporary downstream compatibility layer
        """
        if not isinstance(raw, dict):
            return []

        raw_resps = raw.get("responses")
        if not isinstance(raw_resps, list):
            one = raw.get("response")
            if isinstance(one, dict):
                raw_resps = [one]
            else:
                return []

        # Build allowed maps
        req_map: dict[str, dict[str, Any]] = {}
        allowed_options_by_req: dict[str, dict[str, dict[str, int]]] = {}
        ordered_option_ids_by_req: dict[str, list[str]] = {}
        options_count_by_req: dict[str, int] = {}

        for req in payload.get("requests", []):
            if not isinstance(req, dict):
                continue
            rid = req.get("request_id")
            if not isinstance(rid, str):
                continue
            req_map[rid] = req
            option_id_map: dict[str, dict[str, int]] = {}
            ordered_option_ids: list[str] = []
            for option in req.get("options", []):
                if not isinstance(option, dict):
                    continue
                option_id = option.get("option_id")
                rack_id = self._safe_int(option.get("rack_id"))
                picker_id = self._safe_int(option.get("picker_id"))
                eta_agv = self._safe_int(option.get("eta_agv"))
                eta_picker = self._safe_int(option.get("eta_picker"))
                if not isinstance(option_id, str):
                    continue
                if rack_id <= 0 or picker_id <= 0 or eta_agv < 0 or eta_picker < 0:
                    continue
                data = {
                    "rack_id": int(rack_id),
                    "picker_id": int(picker_id),
                    "eta_agv": int(eta_agv),
                    "eta_picker": int(eta_picker),
                }
                option_id_map[option_id] = data
                ordered_option_ids.append(option_id)
            allowed_options_by_req[rid] = option_id_map
            ordered_option_ids_by_req[rid] = ordered_option_ids
            options_count_by_req[rid] = int(len(option_id_map))

        # index raw by request_id (first occurrence)
        raw_by_id: dict[str, dict[str, Any]] = {}
        for item in raw_resps:
            if not isinstance(item, dict):
                continue
            rid = item.get("request_id")
            if isinstance(rid, str) and rid not in raw_by_id:
                raw_by_id[rid] = item

        sanitized: list[dict[str, Any]] = []
        for rid in req_map:
            raw_item = raw_by_id.get(rid, {})
            allowed_by_option_id = allowed_options_by_req.get(rid, {})
            ordered_option_ids = ordered_option_ids_by_req.get(rid, [])
            options_count = int(options_count_by_req.get(rid, 0))
            reason = str(raw_item.get("reason", "")) if raw_item.get("reason") is not None else ""

            if options_count == 0:
                sanitized.append(
                    {
                        "request_id": rid,
                        "overall_support": "DO_NOT_SUPPORT",
                        "option_feedback": [],
                        "reason": reason,
                        "options": [],
                        "options_count": 0,
                    }
                )
                continue

            raw_feedback = raw_item.get("option_feedback", [])
            if not isinstance(raw_feedback, list):
                raw_feedback = []

            valid_feedback_by_option_id: dict[str, str] = {}
            for item in raw_feedback:
                if not isinstance(item, dict):
                    continue
                option_id = item.get("option_id")
                support_level = item.get("support_level")
                if not isinstance(option_id, str):
                    continue
                if option_id not in allowed_by_option_id:
                    continue
                if support_level is None:
                    continue
                support_level = str(support_level).upper()
                if support_level not in {"STRONG", "WEAK", "REJECT"}:
                    continue
                valid_feedback_by_option_id[option_id] = support_level

            option_feedback: list[dict[str, Any]] = []
            compatible_options: list[dict[str, Any]] = []

            if not valid_feedback_by_option_id:
                for option_id in ordered_option_ids:
                    option_feedback.append(
                        {
                            "option_id": option_id,
                            "support_level": "REJECT",
                        }
                    )
                sanitized.append(
                    {
                        "request_id": rid,
                        "overall_support": "DO_NOT_SUPPORT",
                        "option_feedback": option_feedback,
                        "reason": reason,
                        # Temporary compatibility layer for unchanged Stage 3.
                        "options": [],
                        "options_count": options_count,
                    }
                )
                continue

            has_positive_support = False
            for option_id in ordered_option_ids:
                support_level = valid_feedback_by_option_id.get(option_id, "REJECT")
                option_feedback.append(
                    {
                        "option_id": option_id,
                        "support_level": support_level,
                    }
                )
                if support_level not in {"STRONG", "WEAK"}:
                    continue
                has_positive_support = True
                option_data = allowed_by_option_id[option_id]
                eta_agv = int(option_data["eta_agv"])
                eta_picker = int(option_data["eta_picker"])
                compatible_options.append(
                    {
                        "option_id": option_id,
                        "rack_id": int(option_data["rack_id"]),
                        "picker_id": int(option_data["picker_id"]),
                        "eta_agv": eta_agv,
                        "eta_picker": eta_picker,
                        "sync_cost": max(eta_agv, eta_picker),
                        "eta_gap": abs(eta_agv - eta_picker),
                        "support_level": support_level,
                    }
                )

            sanitized.append(
                {
                    "request_id": rid,
                    "overall_support": "SUPPORT" if has_positive_support else "DO_NOT_SUPPORT",
                    "option_feedback": option_feedback,
                    "reason": reason,
                    # Temporary compatibility layer for unchanged Stage 3.
                    "options": compatible_options,
                    "options_count": options_count,
                }
            )

        return sanitized

    # ----------------------------
    # Stage 3 payload + sanitize
    # ----------------------------

    def _build_stage3_payload(
        self,
        state: dict[str, Any],
        stage1_bundle: list[dict[str, Any]],
        stage2_payload: dict[str, Any],
        stage2_responses: list[dict[str, Any]],
        max_assignments_this_batch: int,
    ) -> dict[str, Any]:
        """Build Stage 3 revision payload from Stage 1 proposal summary plus Stage 2 feedback."""
        fixed_direct_actions = self._build_fixed_direct_actions(state)

        stage1_by_id: dict[str, dict[str, Any]] = {}
        for req in stage1_bundle:
            if not isinstance(req, dict):
                continue
            rid = req.get("request_id")
            if isinstance(rid, str):
                stage1_by_id[rid] = req

        req_meta: dict[str, dict[str, Any]] = {}
        for req in stage2_payload.get("requests", []):
            if not isinstance(req, dict):
                continue
            rid = req.get("request_id")
            if not isinstance(rid, str):
                continue
            stage1_req = stage1_by_id.get(rid, {})
            req_meta[rid] = {
                "agv_id": int(stage1_req.get("agv_id", 0)),
                "purpose": str(stage1_req.get("purpose", "")),
                "primary_rack_id": int(stage1_req.get("primary_rack_id", 0)),
                "backup_rack_ids": [
                    int(b)
                    for b in stage1_req.get("backup_rack_ids", [])
                    if self._safe_int(b) > 0
                ],
                "stage1_reason": str(stage1_req.get("reason", "")) if stage1_req.get("reason") is not None else "",
                "options": req.get("options", []) if isinstance(req.get("options", []), list) else [],
            }

        responses_by_id = {
            r["request_id"]: r
            for r in stage2_responses
            if isinstance(r, dict) and isinstance(r.get("request_id"), str)
        }

        requests_payload: list[dict[str, Any]] = []
        for rid, meta in req_meta.items():
            resp = responses_by_id.get(
                rid,
                {
                    "overall_support": "DO_NOT_SUPPORT",
                    "option_feedback": [],
                    "reason": "",
                },
            )
            option_feedback = resp.get("option_feedback", [])
            if not isinstance(option_feedback, list):
                option_feedback = []
            support_by_option_id: dict[str, str] = {}
            for item in option_feedback:
                if not isinstance(item, dict):
                    continue
                option_id = item.get("option_id")
                support_level = item.get("support_level")
                if not isinstance(option_id, str):
                    continue
                if support_level is None:
                    continue
                support_by_option_id[option_id] = str(support_level).upper()

            rich_options: list[dict[str, Any]] = []
            for opt in meta.get("options", []):
                if not isinstance(opt, dict):
                    continue
                option_id = opt.get("option_id")
                if not isinstance(option_id, str):
                    continue
                eta_agv = int(self._safe_int(opt.get("eta_agv")))
                eta_picker = int(self._safe_int(opt.get("eta_picker")))
                if eta_agv < 0 or eta_picker < 0:
                    continue
                rich_options.append(
                    {
                        "option_id": option_id,
                        "rack_id": int(opt.get("rack_id", 0)),
                        "picker_id": int(opt.get("picker_id", 0)),
                        "eta_agv": eta_agv,
                        "eta_picker": eta_picker,
                        "sync_cost": max(eta_agv, eta_picker),
                        "eta_gap": abs(eta_agv - eta_picker),
                        "support_level": support_by_option_id.get(option_id, "REJECT"),
                    }
                )

            requests_payload.append(
                {
                    "request_id": rid,
                    "agv_id": int(meta["agv_id"]),
                    "purpose": str(meta["purpose"]),
                    "stage1_proposal": {
                        "primary_rack_id": int(meta.get("primary_rack_id", 0)),
                        "backup_rack_ids": list(meta.get("backup_rack_ids", [])),
                        "reason": str(meta.get("stage1_reason", "")),
                    },
                    "overall_support": str(resp.get("overall_support", "DO_NOT_SUPPORT")),
                    "options": rich_options,
                    "picker_reason": str(resp.get("reason", "")) if resp.get("reason") is not None else "",
                }
            )

        sys_pressure = stage2_payload.get("_stage3_system_pressure", stage2_payload.get("system_pressure", {}))
        if not isinstance(sys_pressure, dict):
            sys_pressure = {}

        return {
            "system_pressure": {
                "idle_agvs": int(sys_pressure.get("idle_agvs", 0)),
                "idle_pickers": int(sys_pressure.get("idle_pickers", 0)),
                "active_cooperative_assignments": int(sys_pressure.get("active_cooperative_assignments", 0)),
            },
            "selection_budget": {
                "max_assignments_this_batch": int(max_assignments_this_batch),
            },
            "constraints": {
                "unique_picker": bool(self.config.unique_picker),
                "unique_rack": bool(self.config.unique_rack),
            },
            "fixed_direct_actions": [{"agent_id": int(k), "location_id": int(v)} for k, v in sorted(fixed_direct_actions.items())],
            "requests": requests_payload,
        }

    def _sanitize_stage3_output(self, payload: dict[str, Any], raw: dict[str, Any]) -> Optional[dict[str, Any]]:
        """
        Stage 3 sanitization is conflict-tolerant at the assignment level:
        conflicting assignments are dropped individually, while valid conflict-free
        assignments are preserved and the remaining requests are marked as skipped.

        Only clearly broken top-level structures return None.
        """
        def invalid(reason: str) -> Optional[dict[str, Any]]:
            if self.config.debug:
                print("STAGE3_SANITIZE_INVALID:", reason)
                print("STAGE3_SANITIZE_RAW\n", json.dumps(raw, ensure_ascii=False, indent=2))
                print("STAGE3_SANITIZE_PAYLOAD\n", json.dumps(payload, ensure_ascii=False, indent=2))
            return None

        if not isinstance(raw, dict):
            return invalid("raw is not a dict")

        assignments_in = raw.get("assignments")
        skipped_in = raw.get("skipped", [])
        if not isinstance(assignments_in, list):
            return invalid('"assignments" is not a list')
        if not isinstance(skipped_in, list):
            skipped_in = []

        # Build allowed options map: request_id -> set((rack,picker)) for STRONG/WEAK only
        req_map: dict[str, dict[str, Any]] = {}
        allowed_pairs: dict[str, Set[Tuple[int, int]]] = {}
        for req in payload.get("requests", []):
            if not isinstance(req, dict):
                continue
            rid = req.get("request_id")
            if not isinstance(rid, str):
                continue
            req_map[rid] = req
            pairs: set[tuple[int, int]] = set()
            for opt in req.get("options", []):
                if not isinstance(opt, dict):
                    continue
                support_level = str(opt.get("support_level", "REJECT")).upper()
                if support_level not in {"STRONG", "WEAK"}:
                    continue
                rack_id = self._safe_int(opt.get("rack_id"))
                picker_id = self._safe_int(opt.get("picker_id"))
                if rack_id > 0 and picker_id > 0:
                    pairs.add((int(rack_id), int(picker_id)))
            allowed_pairs[rid] = pairs

        unique_picker = bool(payload.get("constraints", {}).get("unique_picker", True))
        unique_rack = bool(payload.get("constraints", {}).get("unique_rack", True))

        used_pickers: set[int] = set()
        used_racks: set[int] = set()
        used_requests: set[str] = set()

        sanitized_assignments: list[dict[str, Any]] = []
        dropped_request_ids: list[str] = []
        for item in assignments_in:
            if not isinstance(item, dict):
                continue
            rid = item.get("request_id")
            if not isinstance(rid, str) or rid not in req_map:
                continue
            if rid in used_requests:
                dropped_request_ids.append(rid)
                continue

            agv_id = self._safe_int(item.get("agv_id"))
            rack_id = self._safe_int(item.get("rack_id"))
            picker_id = self._safe_int(item.get("picker_id"))
            if agv_id <= 0 or rack_id <= 0 or picker_id <= 0:
                dropped_request_ids.append(rid)
                continue

            meta_agv = int(req_map[rid].get("agv_id", 0))
            if meta_agv != int(agv_id):
                dropped_request_ids.append(rid)
                continue

            if (int(rack_id), int(picker_id)) not in allowed_pairs.get(rid, set()):
                dropped_request_ids.append(rid)
                continue

            if unique_picker and int(picker_id) in used_pickers:
                dropped_request_ids.append(rid)
                continue
            if unique_rack and int(rack_id) in used_racks:
                dropped_request_ids.append(rid)
                continue

            used_requests.add(rid)
            used_pickers.add(int(picker_id))
            used_racks.add(int(rack_id))

            sanitized_assignments.append(
                {
                    "request_id": rid,
                    "agv_id": int(agv_id),
                    "picker_id": int(picker_id),
                    "rack_id": int(rack_id),
                }
            )

        max_assignments_this_batch = int(
            self._safe_int(payload.get("selection_budget", {}).get("max_assignments_this_batch", -1))
        )
        if max_assignments_this_batch >= 0 and len(sanitized_assignments) > max_assignments_this_batch:
            truncated_assignments = sanitized_assignments[max_assignments_this_batch:]
            sanitized_assignments = sanitized_assignments[:max_assignments_this_batch]
            used_requests = {
                str(item.get("request_id"))
                for item in sanitized_assignments
                if isinstance(item, dict) and isinstance(item.get("request_id"), str)
            }
            for item in truncated_assignments:
                if isinstance(item, dict) and isinstance(item.get("request_id"), str):
                    dropped_request_ids.append(str(item.get("request_id")))

        skipped: list[str] = []
        skipped_seen: set[str] = set()
        for s in skipped_in:
            if isinstance(s, str) and s in req_map and s not in used_requests:
                skipped.append(s)
                skipped_seen.add(s)

        for rid in dropped_request_ids:
            if rid in req_map and rid not in used_requests and rid not in skipped_seen:
                skipped.append(rid)
                skipped_seen.add(rid)

        for rid in req_map:
            if rid not in used_requests and rid not in skipped_seen:
                skipped.append(rid)
                skipped_seen.add(rid)

        explanation = raw.get("explanation", "")
        if explanation is None:
            explanation = ""
        explanation = str(explanation)

        return {"assignments": sanitized_assignments, "skipped": skipped, "explanation": explanation}

    # ----------------------------
    # P2: objective scoring (logging only)
    # ----------------------------

    def _compute_objective_scores(self, stage3_payload: dict[str, Any], final_plan: dict[str, Any]) -> dict[str, int]:
        """
        Compute:
          num_assignments
          sum_sync_cost
          sum_eta_gap
          max_sync_cost
        based strictly on the selected options from stage3_payload.
        """
        # Build (request_id, rack, picker) -> metrics
        metrics: dict[tuple[str, int, int], tuple[int, int]] = {}
        for req in stage3_payload.get("requests", []):
            if not isinstance(req, dict):
                continue
            rid = req.get("request_id")
            if not isinstance(rid, str):
                continue
            for opt in req.get("options", []):
                if not isinstance(opt, dict):
                    continue
                rack_id = self._safe_int(opt.get("rack_id"))
                picker_id = self._safe_int(opt.get("picker_id"))
                sc = self._safe_int(opt.get("sync_cost"))
                eg = self._safe_int(opt.get("eta_gap"))
                if rack_id > 0 and picker_id > 0 and sc >= 0 and eg >= 0:
                    metrics[(rid, int(rack_id), int(picker_id))] = (int(sc), int(eg))

        assignments = final_plan.get("assignments", [])
        if not isinstance(assignments, list):
            assignments = []

        num = 0
        sum_sc = 0
        sum_eg = 0
        max_sc = 0
        for a in assignments:
            if not isinstance(a, dict):
                continue
            rid = a.get("request_id")
            if not isinstance(rid, str):
                continue
            rack_id = self._safe_int(a.get("rack_id"))
            picker_id = self._safe_int(a.get("picker_id"))
            key = (rid, int(rack_id), int(picker_id))
            if key not in metrics:
                continue
            sc, eg = metrics[key]
            num += 1
            sum_sc += int(sc)
            sum_eg += int(eg)
            if sc > max_sc:
                max_sc = int(sc)

        return {
            "num_assignments": int(num),
            "sum_sync_cost": int(sum_sc),
            "sum_eta_gap": int(sum_eg),
            "max_sync_cost": int(max_sc),
        }

    # ----------------------------
    # Deterministic assembly & fixed actions
    # ----------------------------

    def _build_fixed_direct_actions(self, state: dict[str, Any]) -> dict[int, int]:
        """
        Fixed actions that should not be negotiated:
        - busy agents -> 0
        - AGVs carrying and not has_delivered and target==0 -> go to nearest goal
        """
        agents = self._sorted_agents(state)
        fixed: dict[int, int] = {}
        goal_ids = [int(x) for x in state.get("goal_ids", [])]

        for idx, a in enumerate(agents):
            agent_id = int(a["id"])

            if bool(a.get("busy", False)):
                fixed[agent_id] = 0
                continue

            if a.get("type") == "AGV":
                carrying = bool(a.get("carrying", False))
                has_delivered = bool(a.get("has_delivered", False))
                target = int(a.get("target", 0) or 0)

                if carrying and (not has_delivered) and target == 0 and goal_ids:
                    best_goal = self._choose_best_loc(
                        agent_index=idx,
                        candidates=goal_ids,
                        cost_map=self._agent_cost_map(state, "agv", agent_id),
                        valid_action_masks=state.get("valid_action_masks", []),
                    )
                    fixed[agent_id] = int(best_goal)

        return fixed

    def _assemble_actions_from_assignments(self, state: dict[str, Any], assignments: list[dict[str, Any]]) -> list[int]:
        agents = self._sorted_agents(state)
        num_agents = int(state.get("meta", {}).get("num_agents", len(agents)))
        actions = [0] * max(0, num_agents)

        fixed = self._build_fixed_direct_actions(state)
        for agent_id, loc_id in fixed.items():
            if 1 <= int(agent_id) <= num_agents:
                actions[int(agent_id) - 1] = int(loc_id)

        for item in assignments:
            if not isinstance(item, dict):
                continue
            agv_id = self._safe_int(item.get("agv_id"))
            picker_id = self._safe_int(item.get("picker_id"))
            rack_id = self._safe_int(item.get("rack_id"))
            if agv_id <= 0 or picker_id <= 0 or rack_id <= 0:
                continue
            if agv_id in fixed:
                continue
            if picker_id in fixed:
                continue
            if 1 <= agv_id <= num_agents:
                actions[agv_id - 1] = int(rack_id)
            if 1 <= picker_id <= num_agents:
                actions[picker_id - 1] = int(rack_id)

        return actions

    def _zero_actions(self, state: dict[str, Any]) -> list[int]:
        num_agents = int(state.get("meta", {}).get("num_agents", len(state.get("agents", []))))
        return [0] * max(0, num_agents)

    def _fallback_whole_plan(self, state: dict[str, Any], reason: str) -> list[int]:
        """
        Whole-plan fallback only (not decision repairs):
        Use a fresh deterministic rule-based SymbioticPlanner for safety.

        P1 fix:
        - Clear this planner's active_assignments and update communication bookkeeping,
          preventing stale state and immediate re-trigger loops.
        """
        self.last_used_fallback = True
        if self.config.debug:
            print("WHOLE_PLAN_FALLBACK:", reason)

        # Preserve still-valid cooperative contexts; fallback should not silently
        # drop AGV-picker bindings that the environment is still executing.
        self._update_active_assignments_from_state(state)
        self._mark_communication_effective()

        # Use a fresh instance to avoid hidden coupling
        rule_fallback = RuleSymbioticPlanner()
        try:
            actions = rule_fallback.plan(state)
        except Exception as exc:
            if self.config.debug:
                print("RULE_FALLBACK_ERROR:", str(exc))
            actions = self._zero_actions(state)

        self.last_final_plan = _DictMessage({
            "assignments": [],
            "skipped": [],
            "explanation": f"FALLBACK_TO_RULE_SYMBIOTIC: {reason}",
            "objective_scores": {
                "num_assignments": 0,
                "sum_sync_cost": 0,
                "sum_eta_gap": 0,
                "max_sync_cost": 0
            },
        })
        return actions

    # ----------------------------
    # Prompts (short, strict, budget-friendly)
    # ----------------------------

    def _stage1_system_prompt(self) -> str:
        if self.enable_rationale:
            return stage1_system_prompt_v1()
        return stage1_system_prompt_v0()

    def _stage1_user_prompt(self, payload: dict[str, Any]) -> str:
        minimal = {
            "system_pressure": payload.get("system_pressure", {}),
            "requests": [
                {
                    "request_id": r.get("request_id"),
                    "agv_id": r.get("agv_id"),
                    "purpose": r.get("purpose"),
                    "candidates": [
                        {
                            "rack_id": c.get("rack_id"),
                            "eta_agv": c.get("eta_agv"),
                            "region_id": c.get("region_id"),
                            "nearby_idle_pickers": c.get("nearby_idle_pickers"),
                        }
                        for c in r.get("candidates", [])
                        if isinstance(c, dict)
                    ],
                }
                for r in payload.get("requests", [])
                if isinstance(r, dict)
            ],
        }
        return json.dumps(minimal, ensure_ascii=True)

    def _stage2_system_prompt(self) -> str:
        if self.enable_rationale:
            return stage2_system_prompt_v1()
        return stage2_system_prompt_v0()

    def _stage2_user_prompt(self, payload: dict[str, Any]) -> str:
        minimal_reqs = []
        for r in payload.get("requests", []):
            if not isinstance(r, dict):
                continue
            minimal_reqs.append(
                {
                    "request_id": r.get("request_id"),
                    "stage1_proposal": r.get("stage1_proposal", {}),
                    "options": [
                        {
                            "option_id": opt.get("option_id"),
                            "rack_id": opt.get("rack_id"),
                            "picker_id": opt.get("picker_id"),
                            "sync_cost": opt.get("sync_cost"),
                            "eta_gap": opt.get("eta_gap"),
                            "eta_picker": opt.get("eta_picker"),
                        }
                        for opt in r.get("options", [])
                        if isinstance(opt, dict)
                    ],
                }
            )
            if self.enable_rationale:
                minimal_reqs[-1]["agv_reason"] = r.get("agv_reason", "")

        minimal = {
            "system_pressure": payload.get("system_pressure", {}),
            "requests": minimal_reqs,
        }
        return json.dumps(minimal, ensure_ascii=True)

    def _stage3_system_prompt(self) -> str:
        if self.enable_rationale:
            return stage3_system_prompt_v1()
        return stage3_system_prompt_v0()
        

    def _stage3_user_prompt(self, payload: dict[str, Any]) -> str:
        requests = []
        for request in payload.get("requests", []):
            if not isinstance(request, dict):
                continue
            if str(request.get("overall_support", "DO_NOT_SUPPORT")).upper() != "SUPPORT":
                continue
            filtered_options = [
                {
                    "rack_id": option.get("rack_id"),
                    "picker_id": option.get("picker_id"),
                    "sync_cost": option.get("sync_cost"),
                    "eta_gap": option.get("eta_gap"),
                    "support_level": option.get("support_level"),
                }
                for option in request.get("options", [])
                if isinstance(option, dict)
                and str(option.get("support_level", "REJECT")).upper() in {"STRONG", "WEAK"}
            ]
            if not filtered_options:
                continue
            item = {
                "request_id": request.get("request_id"),
                "agv_id": request.get("agv_id"),
                "stage1_proposal": {
                    "primary_rack_id": request.get("stage1_proposal", {}).get("primary_rack_id", 0),
                    "backup_rack_ids": request.get("stage1_proposal", {}).get("backup_rack_ids", []),
                },
                "options": filtered_options,
            }
            requests.append(item)
        minimal = {
            "selection_budget": payload.get("selection_budget", {}),
            "requests": requests,
        }
        return json.dumps(minimal, ensure_ascii=True)

    # ----------------------------
    # Helpers
    # ----------------------------

    def _sorted_agents(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        return sorted(state.get("agents", []), key=lambda a: int(a.get("id", 0)))

    def _agent_cost_map(self, state: dict[str, Any], group: str, agent_id: int) -> dict[str, Any]:
        return state.get("cost_table", {}).get(group, {}).get(str(int(agent_id)), {})

    def _safe_cost(self, cost_map: dict[str, Any], loc_id: int) -> Optional[int]:
        v = cost_map.get(str(int(loc_id)))
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    def _safe_int(self, value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return -1

    def _choose_best_loc(
        self,
        agent_index: int,
        candidates: list[int],
        cost_map: dict[str, Any],
        valid_action_masks: list[list[int]],
    ) -> int:
        best_loc = 0
        best_cost: Optional[int] = None
        for loc_id in candidates:
            loc_id = int(loc_id)
            if not self._is_valid_action(valid_action_masks, agent_index, loc_id):
                continue
            cost = self._safe_cost(cost_map, loc_id)
            if cost is None:
                continue
            if best_cost is None or cost < best_cost or (cost == best_cost and loc_id < best_loc):
                best_loc = loc_id
                best_cost = cost
        return best_loc if best_cost is not None else 0

    def _is_valid_action(self, valid_action_masks: list[list[int]], agent_index: int, loc_id: int) -> bool:
        if agent_index < 0 or agent_index >= len(valid_action_masks):
            return False
        mask = valid_action_masks[agent_index]
        if loc_id < 0 or loc_id >= len(mask):
            return False
        return int(mask[loc_id]) == 1


if __name__ == "__main__":
    # Minimal smoke test with fake_state (will attempt vLLM calls if configured).
    fake_state = {
        "meta": {"num_agents": 4},
        "agents": [
            {"id": 1, "type": "AGV", "coords_yx": [0, 0], "busy": False, "carrying": False, "has_delivered": False, "target": 0},
            {"id": 2, "type": "AGV", "coords_yx": [1, 0], "busy": False, "carrying": False, "has_delivered": False, "target": 0},
            {"id": 3, "type": "PICKER", "coords_yx": [0, 1], "busy": False, "target": 0},
            {"id": 4, "type": "PICKER", "coords_yx": [1, 1], "busy": False, "target": 0},
        ],
        "goal_ids": [1, 2],
        "requests_rack_ids_topk": [37, 42, 55],
        "empty_rack_ids_topk": [51, 52],
        "valid_action_masks": [[1] * 200 for _ in range(4)],
        "cost_table": {
            "agv": {
                "1": {"37": 5, "42": 8, "55": 11, "1": 4, "2": 6},
                "2": {"37": 7, "42": 9, "55": 10, "1": 5, "2": 6},
            },
            "picker": {
                "3": {"37": 4, "42": 6, "55": 10},
                "4": {"37": 7, "42": 3, "55": 8},
            },
        },
    }

    planner = SymbioticCommLLMPlanner(SymbioticCommLLMPlannerConfig(debug=True))
    try:
        actions = planner.plan(fake_state)
        print("ACTIONS:", actions)
        print("last_request:", json.dumps(planner.last_request, ensure_ascii=False, indent=2))
        print("last_response:", json.dumps(planner.last_response, ensure_ascii=False, indent=2))
        print("last_final_plan:", json.dumps(planner.last_final_plan, ensure_ascii=False, indent=2))
        print("triggered:", planner.last_communication_triggered, "fallback:", planner.last_used_fallback)
    except Exception as exc:
        print({"error": str(exc)})
