

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
    wait_timeout_steps: int = 40
    min_recommunication_gap_steps: int = 12
    idle_probe_gap_steps: int = 25   # 新增：当没有可用 picker 时，允许低频探测

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
      - planner_throttled_by_budget: bool
      - last_used_fallback: bool
    """

    def __init__(self, config: SymbioticCommLLMPlannerConfig | None = None):
        self.config = config or SymbioticCommLLMPlannerConfig()
        self.enable_rationale = getattr(self.config, "enable_rationale", False)

        vcfg = load_vllm_config_from_env()
        self.agv_client = VLLMChatClient(vcfg)
        self.picker_client = VLLMChatClient(vcfg)
        self.final_client = VLLMChatClient(vcfg)

        # Event-trigger state
        self.step_counter: int = 0
        self.last_communication_step: int = -10**9
        self.last_idle_probe_step: int = -10**9
        self.active_assignments: Dict[int, Dict[str, Any]] = {}  # agv_id -> {picker_id,rack_id,purpose,start_step}
        self.N_ENROUTE: int = 8
        self._stagnation: Dict[int, Dict[str, Any]] = {}
        self.budget_window_steps: int = 20
        self.budget_max_comm_steps: int = 3
        self._communication_history: List[int] = []

        # Region mapping for load balancing
        self.rack_to_region = None   # 将在 plan 中从 state 读取
        self.region_to_racks = None

        # Runner logging
        self.last_request: Optional[dict] = None
        self.last_response: Optional[dict] = None
        self.last_final_plan: Optional[dict] = None
        self.last_communication_triggered: bool = False
        self.planner_throttled_by_budget: bool = False
        self.last_used_fallback: bool = False

        # Whole-plan fallback (rule-based symbiotic). We'll instantiate per-fallback to avoid internal state coupling.
        # self._rule_fallback = RuleSymbioticPlanner()

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

        self.planner_throttled_by_budget = False
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

        reserved_picker_ids_this_round: set[int] = set()
        reserved_rack_ids_this_round: set[int] = set()
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
        ]
        remaining_assignment_budget = int(len(available_pickers_for_round))

        for batch_index, request_batch in enumerate(request_batches, start=1):
            batch_request_ids = [str(req.get("request_id", "")) for req in request_batch]
            if self.config.debug:
                print(
                    f"BATCH_{batch_index}_REQUESTS\n",
                    json.dumps(batch_request_ids, ensure_ascii=False, indent=2),
                )

            stage1_payload = self._build_stage1_payload(state, request_batch)
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
        enroute_stalled = self._update_stagnation_and_check(state, agents)
        coordination_alert = bool(state.get("coordination_alert", False))

        # ----------------------------
        # Compute currently available pickers for NEW cooperation
        # Keep this consistent with Stage2 availability semantics:
        #   picker must be not busy and not already reserved by an active assignment.
        # ----------------------------
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
        )

        # ----------------------------
        # New-task trigger (idle_need)
        # Only enabled when at least one picker is currently available.
        # If no picker is available, suppress ordinary idle-triggered communication.
        # ----------------------------
        idle_need = False
        if available_picker_count > 0:
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

                if (not carrying) and request_racks:
                    idle_need = True
                    break
                if carrying and has_delivered and empty_racks:
                    idle_need = True
                    break
                if carrying and (not has_delivered):
                    # lost goal target
                    idle_need = True
                    break

        # ----------------------------
        # Recovery triggers
        # These should remain available even under picker scarcity.
        # ----------------------------
        agents_by_id = {int(a["id"]): a for a in agents if isinstance(a, dict) and "id" in a}
        active_assignment_target_lost = False
        active_assignment_timeout = False

        for agv_id, assignment in self.active_assignments.items():
            agv_state = agents_by_id.get(int(agv_id))
            if agv_state is None:
                continue
            if self._is_assignment_completed(agv_state, assignment):
                continue

            elapsed = self.step_counter - int(assignment.get("start_step", self.step_counter))
            current_target = int(agv_state.get("target", 0) or 0)
            expected_rack = int(assignment.get("rack_id", 0))
            busy = bool(agv_state.get("busy", False))

            # still executing the same target -> no recovery trigger yet
            if busy and current_target == expected_rack:
                continue

            if current_target == 0:
                active_assignment_target_lost = True
                break

            if elapsed >= self.config.wait_timeout_steps:
                active_assignment_timeout = True

        # Strong recovery triggers should bypass ordinary min-gap throttling.
        if coordination_alert or active_assignment_target_lost or active_assignment_timeout or enroute_stalled:
            return True

        # ----------------------------
        # Budgeted probing:
        # When no picker is currently available, ordinary idle-triggered comm is suppressed,
        # but we allow a low-frequency probe so the system can discover newly available pickers
        # and avoid starvation of idle AGVs.
        # ----------------------------
        if available_picker_count == 0:
            if self.step_counter - self.last_idle_probe_step >= int(self.config.idle_probe_gap_steps):
                # Only probe if there actually exists at least one idle AGV needing cooperation.
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
                        or (carrying and (not has_delivered))
                    )
                    if needs_new_task:
                        self.last_idle_probe_step = self.step_counter
                        return True

            return False

        # ----------------------------
        # Ordinary idle-triggered communication with available picker(s)
        # Apply min-gap and budget throttling here.
        # ----------------------------
        if not idle_need:
            return False

        min_gap = self._scarcity_aware_min_gap(state, agents)
        recent_comm = (self.step_counter - self.last_communication_step) < min_gap
        if recent_comm:
            return False

        if self._is_budget_throttled():
            self.planner_throttled_by_budget = True
            return False

        return True

    def _scarcity_aware_min_gap(self, state: dict[str, Any], agents: list[dict[str, Any]]) -> int:
        """Increase re-communication gap under picker scarcity."""
        num_agvs = int(state.get("meta", {}).get("num_agvs", 0))
        num_pickers = int(state.get("meta", {}).get("num_pickers", 0))
        if num_agvs <= 0:
            num_agvs = sum(1 for agent in agents if agent.get("type") == "AGV")
        if num_pickers <= 0:
            num_pickers = sum(1 for agent in agents if agent.get("type") == "PICKER")

        idle_pickers = sum(
            1 for agent in agents if agent.get("type") == "PICKER" and not bool(agent.get("busy", False))
        )

        mild_scarcity = (num_agvs > num_pickers) or (idle_pickers <= 1)
        severe_scarcity = (num_agvs > num_pickers) and (idle_pickers <= 1)
        if severe_scarcity:
            return 30
        if mild_scarcity:
            return 20
        return int(self.config.min_recommunication_gap_steps)

    def _is_budget_throttled(self) -> bool:
        """Return whether the rolling communication budget has been exhausted."""
        self._prune_communication_history()
        return len(self._communication_history) >= self.budget_max_comm_steps

    def _record_communication_step(self, step_idx: int) -> None:
        self._communication_history.append(int(step_idx))
        self._prune_communication_history()

    def _prune_communication_history(self) -> None:
        window_start = self.step_counter - self.budget_window_steps + 1
        self._communication_history = [
            int(step_idx) for step_idx in self._communication_history if int(step_idx) >= window_start
        ]

    def _update_stagnation_and_check(
        self,
        state: dict[str, Any],
        agents: list[dict[str, Any]],
    ) -> bool:
        """Track AGV en-route stagnation for communication triggering."""
        seen_agent_ids: set[int] = set()
        enroute_stalled = False

        for agent in agents:
            if not isinstance(agent, dict) or "id" not in agent:
                continue

            agent_id = int(agent["id"])
            seen_agent_ids.add(agent_id)
            pos = self._coords_tuple(agent.get("coords_yx"))
            target = int(agent.get("target", 0) or 0)
            target_coords = self._coords_tuple(agent.get("target_coords_yx"))
            tracker = self._stagnation.setdefault(
                agent_id,
                {
                    "last_pos": pos,
                    "no_move": 0,
                    "last_target": target,
                },
            )

            if int(tracker.get("last_target", 0)) != target:
                tracker["no_move"] = 0
                tracker["last_target"] = target
                tracker["last_pos"] = pos
            else:
                last_pos = self._coords_tuple(tracker.get("last_pos"))
                if pos is not None and last_pos is not None and pos == last_pos:
                    tracker["no_move"] = int(tracker.get("no_move", 0)) + 1
                else:
                    tracker["no_move"] = 0
                tracker["last_pos"] = pos

            busy = bool(agent.get("busy", False))
            agent_type = str(agent.get("type", ""))
            enroute = (
                agent_type == "AGV"
                and busy
                and target != 0
                and pos is not None
                and target_coords is not None
                and pos != target_coords
            )
            if not enroute:
                tracker["no_move"] = 0
            elif int(tracker.get("no_move", 0)) >= self.N_ENROUTE:
                enroute_stalled = True

        stale_agent_ids = set(self._stagnation.keys()) - seen_agent_ids
        for agent_id in stale_agent_ids:
            self._stagnation.pop(agent_id, None)

        return enroute_stalled

    def _coords_tuple(self, coords: Any) -> Optional[Tuple[int, int]]:
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
        rack_coords: tuple[int, int],
        radius: int = 15,
    ) -> int:
        """Count idle pickers within a Manhattan radius of the rack coordinates.

        `rack_coords` must be in `(x, y)` order. If coordinates are unavailable
        or malformed, return 0 conservatively.
        """
        try:
            rack_x, rack_y = int(rack_coords[0]), int(rack_coords[1])
        except (TypeError, ValueError, IndexError):
            return 0

        count = 0
        for agent in state.get("agents", []):
            if not isinstance(agent, dict):
                continue
            if agent.get("type") != "PICKER" or bool(agent.get("busy", False)):
                continue

            picker_x: Optional[int] = None
            picker_y: Optional[int] = None
            coords_yx = agent.get("coords_yx")
            if isinstance(coords_yx, (list, tuple)) and len(coords_yx) == 2:
                try:
                    picker_y = int(coords_yx[0])
                    picker_x = int(coords_yx[1])
                except (TypeError, ValueError):
                    picker_x = None
                    picker_y = None
            elif "x" in agent and "y" in agent:
                try:
                    picker_x = int(agent["x"])
                    picker_y = int(agent["y"])
                except (TypeError, ValueError):
                    picker_x = None
                    picker_y = None

            if picker_x is None or picker_y is None:
                continue

            if abs(picker_x - rack_x) + abs(picker_y - rack_y) <= int(radius):
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
                }
            )
        print("empty_rack_ids_topk:", empty_racks)
        if not empty_racks:
            print("empty_rack_ids_topk is EMPTY")
        print("batch_requests:", batch_requests)
        print("fixed_direct_actions:", self._build_fixed_direct_actions(state))
        print("returned_actions:", returned_actions)

    def _is_assignment_completed(self, agv_state: dict[str, Any], assignment: dict[str, Any]) -> bool:
        purpose = str(assignment.get("purpose", ""))
        carrying = bool(agv_state.get("carrying", False))
        if purpose == "LOAD":
            return carrying
        if purpose == "UNLOAD":
            return not carrying
        return True

    def _update_active_assignments_from_state(self, state: dict[str, Any]) -> None:
        """Keep only assignments that still match the AGV state."""
        agents = self._sorted_agents(state)
        agvs_by_id = {int(a["id"]): a for a in agents if a.get("type") == "AGV"}
        new_active: Dict[int, Dict[str, Any]] = {}
        for agv_id, assignment in self.active_assignments.items():
            agv_state = agvs_by_id.get(int(agv_id))
            if agv_state is None:
                continue
            if self._is_assignment_completed(agv_state, assignment):
                continue
            current_target = int(agv_state.get("target", 0) or 0)
            rack_id = int(assignment.get("rack_id", 0))
            if current_target not in {0, rack_id}:
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

        self.active_assignments = new_active

    def _cooperative_purpose_for_agv(self, agv_state: dict[str, Any]) -> Optional[str]:
        carrying = bool(agv_state.get("carrying", False))
        has_delivered = bool(agv_state.get("has_delivered", False))
        if not carrying:
            return "LOAD"
        if carrying and has_delivered:
            return "UNLOAD"
        return None

    # ----------------------------
    # Batch request building
    # ----------------------------

    def _build_batch_requests(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        """Deterministically build cooperative AGV requests for this step."""
        agents = self._sorted_agents(state)
        req_racks = [int(x) for x in state.get("requests_rack_ids_topk", [])]
        empty_racks = [int(x) for x in state.get("empty_rack_ids_topk", [])]

        requests: list[dict[str, Any]] = []
        for a in agents:
            if a.get("type") != "AGV":
                continue
            if bool(a.get("busy", False)):
                continue
            target = int(a.get("target", 0) or 0)
            if target != 0:
                continue

            agv_id = int(a["id"])
            carrying = bool(a.get("carrying", False))
            has_delivered = bool(a.get("has_delivered", False))

            # carrying and not delivered -> fixed goal move, not cooperative request
            if carrying and (not has_delivered):
                continue

            if carrying and has_delivered:
                purpose = "UNLOAD"
                pool = empty_racks
            else:
                purpose = "LOAD"
                pool = req_racks

            if not pool:
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
        """Record a communication round only when it produced an executable outcome.

        A communication round is only counted as effective if it produces
        executable assignments, executable fixed actions, or explicitly falls
        back to the rule-based planner. Otherwise, the round is not recorded as
        a successful communication event, preventing idle deadlock caused by
        premature throttling.
        """
        self.last_communication_step = self.step_counter
        self._record_communication_step(self.step_counter)

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

    def _build_stage1_payload(self, state: dict[str, Any], batch_requests: list[dict[str, Any]]) -> dict[str, Any]:
        agents = self._sorted_agents(state)
        idle_agvs = sum(1 for a in agents if a.get("type") == "AGV" and not bool(a.get("busy", False)))
        idle_pickers = sum(1 for a in agents if a.get("type") == "PICKER" and not bool(a.get("busy", False)))
        picker_scarcity = "low" if idle_pickers >= 2 else "high"

        # 计算区域负载
        region_load = self._compute_region_load()
        location_coords_xy = state.get("location_coords_xy", {})


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
                coords_xy = location_coords_xy.get(str(int(rid)))
                nearby_idle_pickers = 0
                if isinstance(coords_xy, (list, tuple)) and len(coords_xy) == 2:
                    try:
                        nearby_idle_pickers = self._count_nearby_idle_pickers(
                            state,
                            (int(coords_xy[0]), int(coords_xy[1])),
                        )
                    except (TypeError, ValueError):
                        nearby_idle_pickers = 0

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
            and int(a["id"]) not in reserved_picker_ids
        ]
        idle_pickers = len(available_pickers)
        picker_scarcity = "high" if idle_pickers < idle_agvs else "low"

        requests_payload: list[dict[str, Any]] = []
        for req in stage1_bundle:
            request_id = str(req["request_id"])
            agv_id = int(req["agv_id"])
            purpose = str(req["purpose"])
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
                for p in available_pickers:
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
        - Clear this planner's active_assignments and update last_communication_step,
          preventing stale state and immediate re-trigger loops.
        """
        self.last_used_fallback = True
        if self.config.debug:
            print("WHOLE_PLAN_FALLBACK:", reason)

        # P1: reset internal state related to triggering
        self.active_assignments.clear()
        self.last_communication_step = self.step_counter

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
