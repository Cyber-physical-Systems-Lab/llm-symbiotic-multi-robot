from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Optional, Set

from symco.planners.symbiotic_comm_llm_planner import (
    _DictMessage,
    SymbioticCommLLMPlanner,
    SymbioticCommLLMPlannerConfig,
)


@dataclass(slots=True)
class NonMutualisticCommLLMPlannerV2Config(SymbioticCommLLMPlannerConfig):
    """Config for the non-mutualistic-with-partner-awareness staged baseline."""


class NonMutualisticCommLLMPlannerV2(SymbioticCommLLMPlanner):
    """
    Non-mutualistic communication baseline with partner awareness.

    Definition:
    - Stage 1: AGV-side unilateral commitment with coarse partner awareness.
    - Stage 2: Picker-side unilateral ACK/BUSY on the committed rack.
    - Stage 3: Deterministic integration only. No LLM is used in Stage 3.

    Purity constraints:
    - No rationale generation or propagation.
    - No target revision after Stage 1.
    - No negotiation across stages.
    """

    def __init__(self, config: NonMutualisticCommLLMPlannerV2Config | None = None):
        super().__init__(config or NonMutualisticCommLLMPlannerV2Config())
        # This baseline never uses rationale.
        self.enable_rationale = False
        self._reset_unload_deadlock_debug_state()

    def _reset_unload_deadlock_debug_state(self) -> None:
        """Reset simple UNLOAD/empty-rack deadlock debug counters."""
        self.last_debug_idle_unload_agv_count = 0
        self.last_debug_empty_rack_count = 0
        self.last_debug_actionable_unload_agv_count = 0
        self.last_debug_suspected_unload_deadlock = False
        self.last_debug_idle_unload_on_storage_cell_count = 0
        self.last_debug_idle_unload_on_excluded_cell_count = 0
        self.last_debug_idle_unload_on_current_cell_not_in_empty_topk_count = 0
        self.last_debug_idle_unload_on_current_cell_invalid_action_count = 0
        self.last_debug_idle_unload_on_excluded_cell_examples = []

        self.suspected_unload_deadlock_steps = 0
        self.suspected_unload_deadlock_ge10_events = 0
        self._current_suspected_unload_deadlock_streak = 0
        self.idle_unload_on_excluded_cell_steps = 0
        self.idle_unload_on_excluded_cell_ge10_events = 0
        self._current_idle_unload_on_excluded_cell_streak = 0

    def reset(self) -> None:
        """Reset per-episode planner state without recreating LLM clients."""
        super().reset()
        self._reset_unload_deadlock_debug_state()

    def _location_id_by_coords_yx(self, state: dict[str, Any]) -> dict[tuple[int, int], int]:
        """
        Build a mapping from agent coords_yx to action/location id.

        The state key location_coords_xy is expected to store coordinates as [x, y],
        while agent state uses coords_yx. Convert [x, y] -> (y, x).
        """
        location_coords_xy = state.get("location_coords_xy", {})
        if not isinstance(location_coords_xy, dict):
            return {}

        mapping: dict[tuple[int, int], int] = {}
        for raw_loc_id, raw_coords in location_coords_xy.items():
            loc_id = self._safe_int(raw_loc_id)
            if loc_id <= 0:
                continue
            if not isinstance(raw_coords, (list, tuple)) or len(raw_coords) < 2:
                continue
            x = self._safe_int(raw_coords[0])
            y = self._safe_int(raw_coords[1])
            mapping[(int(y), int(x))] = int(loc_id)

        return mapping

    def _normalize_coords_yx_debug(self, value: Any) -> tuple[int, int] | None:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return int(value[0]), int(value[1])
            except (TypeError, ValueError):
                return None
        return None

    def _update_simple_unload_deadlock_debug(self, state: dict[str, Any]) -> None:
        agents = self._sorted_agents(state)
        valid_masks = state.get("valid_action_masks", [])
        empty_racks = [
            int(x)
            for x in state.get("empty_rack_ids_topk", [])
            if self._safe_int(x) > 0
        ]

        agv_index_by_id = {
            int(agent["id"]): idx
            for idx, agent in enumerate(agents)
            if isinstance(agent, dict) and agent.get("type") == "AGV"
        }
        location_id_by_yx = self._location_id_by_coords_yx(state)
        goal_ids = {
            self._safe_int(x)
            for x in state.get("goal_ids", [])
            if self._safe_int(x) > 0
        }
        empty_rack_set = set(int(x) for x in empty_racks)

        idle_unload_agv_count = 0
        actionable_unload_agv_count = 0
        idle_unload_on_storage_cell_count = 0
        idle_unload_on_excluded_cell_count = 0
        current_cell_not_in_empty_topk_count = 0
        current_cell_invalid_action_count = 0
        excluded_examples: list[dict[str, Any]] = []

        for agent in agents:
            if not isinstance(agent, dict):
                continue
            if agent.get("type") != "AGV":
                continue
            if bool(agent.get("busy", False)):
                continue
            if int(agent.get("target", 0) or 0) != 0:
                continue

            carrying = bool(agent.get("carrying", False))
            has_delivered = bool(agent.get("has_delivered", False))

            if not (carrying and has_delivered):
                continue

            idle_unload_agv_count += 1

            agv_id = int(agent["id"])
            agent_index = agv_index_by_id.get(agv_id, -1)
            cost_map = self._agent_cost_map(state, "agv", agv_id)
            coords_yx = self._normalize_coords_yx_debug(agent.get("coords_yx"))
            current_loc_id = 0
            if coords_yx is not None:
                current_loc_id = int(location_id_by_yx.get(coords_yx, 0))

            if current_loc_id > 0 and current_loc_id not in goal_ids:
                idle_unload_on_storage_cell_count += 1

                current_in_empty_topk = int(current_loc_id) in empty_rack_set

                current_valid_action = False
                if agent_index >= 0:
                    current_valid_action = bool(
                        self._is_valid_action(valid_masks, agent_index, int(current_loc_id))
                    )

                if not current_in_empty_topk:
                    current_cell_not_in_empty_topk_count += 1

                if not current_valid_action:
                    current_cell_invalid_action_count += 1

                current_excluded = (not current_in_empty_topk) or (not current_valid_action)

                if current_excluded:
                    idle_unload_on_excluded_cell_count += 1
                    if len(excluded_examples) < 5:
                        excluded_examples.append(
                            {
                                "agv_id": int(agv_id),
                                "coords_yx": list(coords_yx) if coords_yx is not None else None,
                                "current_loc_id": int(current_loc_id),
                                "current_in_empty_topk": bool(current_in_empty_topk),
                                "current_valid_action": bool(current_valid_action),
                                "empty_rack_count": int(len(empty_racks)),
                            }
                        )

            has_valid_empty = False
            for rack_id in empty_racks:
                rack_id = int(rack_id)

                if agent_index >= 0 and not self._is_valid_action(valid_masks, agent_index, rack_id):
                    continue

                eta_agv = self._safe_cost(cost_map, rack_id)
                if eta_agv is None:
                    continue

                has_valid_empty = True
                break

            if has_valid_empty:
                actionable_unload_agv_count += 1

        suspected = (
            idle_unload_agv_count > 0
            and len(empty_racks) > 0
            and actionable_unload_agv_count == 0
        )

        self.last_debug_idle_unload_agv_count = int(idle_unload_agv_count)
        self.last_debug_empty_rack_count = int(len(empty_racks))
        self.last_debug_actionable_unload_agv_count = int(actionable_unload_agv_count)
        self.last_debug_suspected_unload_deadlock = bool(suspected)
        self.last_debug_idle_unload_on_storage_cell_count = int(idle_unload_on_storage_cell_count)
        self.last_debug_idle_unload_on_excluded_cell_count = int(idle_unload_on_excluded_cell_count)
        self.last_debug_idle_unload_on_current_cell_not_in_empty_topk_count = int(current_cell_not_in_empty_topk_count)
        self.last_debug_idle_unload_on_current_cell_invalid_action_count = int(current_cell_invalid_action_count)
        self.last_debug_idle_unload_on_excluded_cell_examples = excluded_examples

        if idle_unload_on_excluded_cell_count > 0:
            self.idle_unload_on_excluded_cell_steps += 1
            self._current_idle_unload_on_excluded_cell_streak += 1
            if self._current_idle_unload_on_excluded_cell_streak == 10:
                self.idle_unload_on_excluded_cell_ge10_events += 1
        else:
            self._current_idle_unload_on_excluded_cell_streak = 0

        if suspected:
            self.suspected_unload_deadlock_steps += 1
            self._current_suspected_unload_deadlock_streak += 1
            if self._current_suspected_unload_deadlock_streak == 10:
                self.suspected_unload_deadlock_ge10_events += 1
        else:
            self._current_suspected_unload_deadlock_streak = 0

    # ----------------------------
    # Main planning loop
    # ----------------------------

    def plan(self, state: dict[str, Any]) -> list[int]:
        """Run the non-mutualistic mini-batch communication loop."""
        self.step_counter += 1

        if self.rack_to_region is None:
            self.rack_to_region = state.get("rack_to_region", {})
            self.region_to_racks = state.get("region_to_racks", {})

        self.planner_throttled_by_budget = False
        self._update_active_assignments_from_state(state)
        self._update_simple_unload_deadlock_debug(state)

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

        batch_requests = self._build_batch_requests(state)
        if not batch_requests:
            self.last_request = _DictMessage({"requests": []})
            self.last_response = _DictMessage({"responses": []})
            self.last_final_plan = _DictMessage(
                {
                    "assignments": [],
                    "skipped": [],
                    "explanation": "No cooperative requests.",
                    "objective_scores": {
                        "num_assignments": 0,
                        "sum_sync_cost": 0,
                        "sum_eta_gap": 0,
                        "max_sync_cost": 0,
                    },
                    "planner_schema": "non_mutualistic_partner_aware_comm_llm_v2",
                    "stage2_semantics": "ack_busy_committed_target",
                }
            )
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
        all_assignments: list[dict[str, Any]] = []
        aggregate_skipped: list[str] = []
        aggregate_objective_requests: list[dict[str, Any]] = []
        batch_failure_reasons: list[str] = []

        reserved_agv_ids_this_round: set[int] = set()
        reserved_picker_ids_this_round: set[int] = set()
        reserved_rack_ids_this_round: set[int] = set()

        for batch_index, request_batch in enumerate(request_batches, start=1):
            batch_request_ids = [str(req.get("request_id", "")) for req in request_batch]
            if self.config.debug:
                print(
                    f"BATCH_{batch_index}_REQUESTS\n",
                    json.dumps(batch_request_ids, ensure_ascii=False, indent=2),
                )

            # ----------------------------
            # Stage 1: AGV unilateral commitment
            # ----------------------------
            stage1_payload = self._build_stage1_payload(
                state=state,
                batch_requests=request_batch,
                reserved_rack_ids=reserved_rack_ids_this_round,
                reserved_agv_ids=reserved_agv_ids_this_round,
                reserved_picker_ids=reserved_picker_ids_this_round,
            )
            if self.config.debug:
                print(
                    f"BATCH_{batch_index}_STAGE1_PAYLOAD\n",
                    json.dumps(stage1_payload, ensure_ascii=False, indent=2),
                )

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
                print(
                    f"BATCH_{batch_index}_STAGE1_RAW\n",
                    json.dumps(stage1_raw, ensure_ascii=False, indent=2),
                )

            stage1_bundle = self._sanitize_stage1_output(stage1_payload, stage1_raw)
            all_stage1_requests.extend(stage1_bundle)

            if not stage1_bundle:
                self.last_used_fallback = True
                batch_failure_reasons.append(f"batch{batch_index}: Stage1 produced no valid commitments.")
                if self.config.debug:
                    print(f"BATCH_{batch_index}_FAILURE: Stage1 produced no valid commitments.")
                continue

            # ----------------------------
            # Stage 2: Picker unilateral ACK/BUSY
            # ----------------------------
            stage2_payload = self._build_stage2_payload(
                state=state,
                stage1_bundle=stage1_bundle,
                reserved_picker_ids=reserved_picker_ids_this_round,
            )
            if self.config.debug:
                print(
                    f"BATCH_{batch_index}_STAGE2_PAYLOAD\n",
                    json.dumps(stage2_payload, ensure_ascii=False, indent=2),
                )

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
                print(
                    f"BATCH_{batch_index}_STAGE2_RAW\n",
                    json.dumps(stage2_raw, ensure_ascii=False, indent=2),
                )

            stage2_responses = self._sanitize_stage2_output(stage2_payload, stage2_raw)
            all_stage2_responses.extend(stage2_responses)

            # ----------------------------
            # Stage 3: deterministic assembly only
            # ----------------------------
            batch_final_plan, batch_objective_requests = self._build_stage3_deterministic_plan(
                stage1_bundle=stage1_bundle,
                stage2_responses=stage2_responses,
                reserved_picker_ids=reserved_picker_ids_this_round,
                reserved_rack_ids=reserved_rack_ids_this_round,
            )

            batch_assignments = self._dedupe_assignments(
                batch_final_plan.get("assignments", []),
                reserved_picker_ids=reserved_picker_ids_this_round,
                reserved_rack_ids=reserved_rack_ids_this_round,
            )
            all_assignments.extend(batch_assignments)
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

            aggregate_skipped.extend(
                [
                    rid
                    for rid in batch_final_plan.get("skipped", [])
                    if isinstance(rid, str)
                ]
            )
            aggregate_objective_requests.extend(batch_objective_requests)

            if self.config.debug:
                print(
                    f"BATCH_{batch_index}_STAGE3_DETERMINISTIC\n",
                    json.dumps(batch_final_plan, ensure_ascii=False, indent=2),
                )
                print(
                    f"BATCH_{batch_index}_ASSIGNMENTS\n",
                    json.dumps(batch_assignments, ensure_ascii=False, indent=2),
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
            "planner_schema": "non_mutualistic_partner_aware_comm_llm_v2",
            "stage2_semantics": "ack_busy_committed_target",
        }
        aggregate_stage3_payload = {"requests": aggregate_objective_requests}
        aggregate_final_plan["objective_scores"] = self._compute_objective_scores(
            aggregate_stage3_payload,
            aggregate_final_plan,
        )
        self.last_final_plan = _DictMessage(aggregate_final_plan)

        if self.config.debug:
            print(
                "ALL_ASSIGNMENTS_BEFORE_ASSEMBLE\n",
                json.dumps(all_assignments, ensure_ascii=False, indent=2),
            )
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

        self._register_active_assignments_from_assignments(state, all_assignments)
        return actions

    # ----------------------------
    # Stage 1: AGV-side commitment
    # ----------------------------

    def _build_stage1_payload(
        self,
        state: dict[str, Any],
        batch_requests: list[dict[str, Any]],
        reserved_rack_ids: set[int] | None = None,
        reserved_agv_ids: set[int] | None = None,
        reserved_picker_ids: set[int] | None = None,
    ) -> dict[str, Any]:
        """
        Build Stage 1 payload using:
        - AGV feasibility
        - eta_agv
        - region_load
        - coarse partner awareness:
          idle_pickers, picker_scarcity, nearby_idle_pickers
        """
        reserved_rack_ids = set() if reserved_rack_ids is None else {int(x) for x in reserved_rack_ids}
        reserved_agv_ids = set() if reserved_agv_ids is None else {int(x) for x in reserved_agv_ids}
        reserved_picker_ids = set() if reserved_picker_ids is None else {int(x) for x in reserved_picker_ids}

        agents = self._sorted_agents(state)
        protected_picker_bindings = self._protected_picker_bindings_for_waiting_agvs(state)
        protected_picker_ids = {int(picker_id) for picker_id in protected_picker_bindings.values()}
        idle_agvs = sum(
            1
            for agent in agents
            if agent.get("type") == "AGV"
            and not bool(agent.get("busy", False))
            and int(agent["id"]) not in reserved_agv_ids
        )
        active_reserved_picker_ids = {
            int(assignment.get("picker_id", -1))
            for assignment in self.active_assignments.values()
            if isinstance(assignment, dict) and self._safe_int(assignment.get("picker_id")) > 0
        }
        idle_pickers = sum(
            1
            for agent in agents
            if agent.get("type") == "PICKER" and not bool(agent.get("busy", False))
            and int(agent["id"]) not in active_reserved_picker_ids
            and int(agent["id"]) not in protected_picker_ids
            and int(agent["id"]) not in reserved_picker_ids
        )
        picker_scarcity = "high" if idle_pickers <= 1 else "low"

        region_load = self._compute_region_load()
        valid_masks = state.get("valid_action_masks", [])
        

        agv_index_by_id: dict[int, int] = {}
        for idx, agent in enumerate(agents):
            if agent.get("type") == "AGV":
                agv_index_by_id[int(agent["id"])] = idx

        requests_payload: list[dict[str, Any]] = []
        for req in batch_requests:
            agv_id = int(req["agv_id"])
            pool = [int(x) for x in req["candidate_pool"]]
            cost_map = self._agent_cost_map(state, "agv", agv_id)
            agent_index = agv_index_by_id.get(agv_id, -1)

            candidates: list[dict[str, Any]] = []
            for rack_id in pool:
                if rack_id in reserved_rack_ids:
                    continue
                if agent_index >= 0 and not self._is_valid_action(valid_masks, agent_index, rack_id):
                    continue

                eta_agv = self._safe_cost(cost_map, rack_id)
                if eta_agv is None:
                    continue

                nearby_idle_pickers = self._count_nearby_idle_pickers(
                    state,
                    int(rack_id),
                    eta_threshold=15,
                )

                candidates.append(
                    {
                        "rack_id": int(rack_id),
                        "eta_agv": int(eta_agv),
                        "region_id": self.rack_to_region.get(int(rack_id), -1) if self.rack_to_region else -1,
                        "nearby_idle_pickers": int(nearby_idle_pickers),
                    }
                )

            candidates.sort(
                key=lambda c: (
                    int(c["eta_agv"]),
                    -int(c["nearby_idle_pickers"]),
                    int(region_load.get(int(c["region_id"]), 0)) if int(c["region_id"]) >= 0 else 0,
                    int(c["rack_id"]),
                )
            )
            candidates = candidates[: max(0, int(self.config.stage1_pool_k))]

            if not candidates:
                continue

            requests_payload.append(
                {
                    "request_id": req["request_id"],
                    "agv_id": int(agv_id),
                    "purpose": str(req["purpose"]),
                    "candidates": candidates,
                }
            )

        return {
            "system_pressure": {
                "idle_agvs": int(idle_agvs),
                "idle_pickers": int(idle_pickers),
                "picker_scarcity": picker_scarcity,
                "region_load": region_load,
            },
            "requests": requests_payload,
        }

    def _sanitize_stage1_output(self, payload: dict[str, Any], raw: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Sanitize Stage 1 committed rack output.

        Validation-only policy:
        - keep a committed rack only if it is legal
        - keep it only if it does not duplicate an already accepted rack
        - do not repair or replace invalid commitments
        """
        if not isinstance(raw, dict):
            return []

        raw_requests = raw.get("requests")
        if not isinstance(raw_requests, list):
            return []

        request_meta: dict[str, dict[str, Any]] = {}
        allowed_by_request: dict[str, Set[int]] = {}

        for item in payload.get("requests", []):
            if not isinstance(item, dict):
                continue
            request_id = item.get("request_id")
            if not isinstance(request_id, str):
                continue

            candidates = item.get("candidates", [])
            allowed: set[int] = set()
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                rack_id = self._safe_int(candidate.get("rack_id"))
                eta_agv = self._safe_int(candidate.get("eta_agv"))
                if rack_id <= 0 or eta_agv < 0:
                    continue
                allowed.add(int(rack_id))

            if not allowed:
                continue

            allowed_by_request[request_id] = allowed
            request_meta[request_id] = {
                "agv_id": int(item.get("agv_id", 0)),
                "purpose": str(item.get("purpose", "")),
                "candidates": candidates,
            }

        raw_by_request: dict[str, dict[str, Any]] = {}
        for item in raw_requests:
            if not isinstance(item, dict):
                continue
            request_id = item.get("request_id")
            if isinstance(request_id, str) and request_id not in raw_by_request:
                raw_by_request[request_id] = item

        used_racks: set[int] = set()
        sanitized: list[dict[str, Any]] = []

        for request_id, meta in request_meta.items():
            raw_item = raw_by_request.get(request_id, {})
            committed_rack_id = self._safe_int(raw_item.get("committed_rack_id"))

            if committed_rack_id <= 0:
                continue
            if committed_rack_id not in allowed_by_request[request_id]:
                continue
            if committed_rack_id in used_racks:
                continue

            sanitized_item = {
                "request_id": request_id,
                "agv_id": int(meta["agv_id"]),
                "purpose": str(meta["purpose"]),
                "committed_rack_id": int(committed_rack_id),
                "candidates": meta["candidates"],
            }
            sanitized.append(sanitized_item)
            used_racks.add(int(committed_rack_id))

        return sanitized

    def _stage1_system_prompt(self) -> str:
        return (
            "You are the AGV-side commitment module in Stage 1 of a staged non-mutualistic coordination framework.\n"
            "\n"
            "Task:\n"
            "For each request, choose exactly 1 committed rack from the given candidates.\n"
            "\n"
            "Hard constraints:\n"
            "1) Within the same batch, each committed_rack_id must be unique.\n"
            "2) The same rack must never be assigned to two different request_id values in one output.\n"
            "3) If two requests compete for the same rack, assign that rack to only one of them and choose another valid candidate for the other request.\n"
            "4) Batch-level rack uniqueness is a mandatory constraint, not a preference.\n"
            "\n"
            "Priority rules:\n"
            "1) AGV-side accessibility is the main criterion: lower eta_agv is generally better.\n"
            "2) Do not choose a much worse eta_agv option only because it seems more likely to receive picker support.\n"
            "3) Use nearby_idle_pickers only as a weak tie-break when eta_agv values are close.\n"
            "4) If candidates are otherwise similar, prefer candidates in less loaded regions.\n"
            "5) Because this is a single committed choice with no later revision, prefer a stable and conservative rack rather than a risky one.\n"
            "\n"
            "Decision guidance:\n"
            "1) A candidate with nearby_idle_pickers >= 1 is not automatically better.\n"
            "2) If one candidate is clearly better in eta_agv, usually keep that candidate.\n"
            "3) Only use nearby_idle_pickers to break close AGV-side choices, not to override a clearly better eta_agv option.\n"
            "\n"
            "Do NOT do the following:\n"
            "- choose multiple racks for one request\n"
            "- invent racks outside the given candidates\n"
            "\n"
            "Output JSON only.\n"
            'Return exactly: {"requests":[{"request_id":"...","committed_rack_id":37}]}'
        )

    def _stage1_user_prompt(self, payload: dict[str, Any]) -> str:
        minimal = {
            "system_pressure": {
                "idle_pickers": payload.get("system_pressure", {}).get("idle_pickers", 0),
                "picker_scarcity": payload.get("system_pressure", {}).get("picker_scarcity", "low"),
                "region_load": payload.get("system_pressure", {}).get("region_load", {}),
            },
            "requests": [
                {
                    "request_id": req.get("request_id"),
                    "agv_id": req.get("agv_id"),
                    "candidates": [
                        {
                            "rack_id": cand.get("rack_id"),
                            "eta_agv": cand.get("eta_agv"),
                            "region_id": cand.get("region_id"),
                            "nearby_idle_pickers": cand.get("nearby_idle_pickers"),
                        }
                        for cand in req.get("candidates", [])
                        if isinstance(cand, dict)
                    ],
                }
                for req in payload.get("requests", [])
                if isinstance(req, dict)
            ],
        }
        return json.dumps(minimal, ensure_ascii=True)

    # ----------------------------
    # Stage 2: Picker-side ACK/BUSY
    # ----------------------------

    def _build_stage2_payload(
        self,
        state: dict[str, Any],
        stage1_bundle: list[dict[str, Any]],
        reserved_picker_ids: set[int] | None = None,
    ) -> dict[str, Any]:
        """
        Build Stage 2 payload using picker-side information only:
        - committed rack
        - picker candidates with sync_cost / eta_gap / eta_picker
        - idle_pickers as neutral system context

        Note:
        - picker_scarcity is intentionally removed
        - Stage 2 should ACK whenever at least one feasible picker candidate exists
        """
        reserved_picker_ids = set() if reserved_picker_ids is None else {int(x) for x in reserved_picker_ids}

        agents = self._sorted_agents(state)
        protected_picker_bindings = self._protected_picker_bindings_for_waiting_agvs(state)
        protected_picker_ids = {int(picker_id) for picker_id in protected_picker_bindings.values()}

        active_reserved_picker_ids = {
            int(assignment.get("picker_id", -1))
            for assignment in self.active_assignments.values()
            if isinstance(assignment, dict) and self._safe_int(assignment.get("picker_id")) > 0
        }

        available_pickers = [
            agent
            for agent in agents
            if agent.get("type") == "PICKER"
            and not bool(agent.get("busy", False))
            and int(agent["id"]) not in active_reserved_picker_ids
            and int(agent["id"]) not in protected_picker_ids
            and int(agent["id"]) not in reserved_picker_ids
        ]

        idle_pickers = len(available_pickers)

        requests_payload: list[dict[str, Any]] = []
        for req in stage1_bundle:
            request_id = str(req["request_id"])
            committed_rack_id = self._safe_int(req.get("committed_rack_id"))
            if committed_rack_id <= 0:
                continue

            agv_id = self._safe_int(req.get("agv_id"))
            protected_picker_id_for_request = int(protected_picker_bindings.get(int(agv_id), -1))

            eta_agv = self._lookup_stage1_eta_agv(req, committed_rack_id)
            if eta_agv < 0:
                continue

            candidate_pickers = list(available_pickers)

            # Allow the picker already semantically bound to this AGV/rack context
            # to reappear as a candidate for this request only.
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

            picker_candidates: list[dict[str, Any]] = []
            for picker in candidate_pickers:
                picker_id = int(picker["id"])
                eta_picker = self._safe_cost(
                    self._agent_cost_map(state, "picker", picker_id),
                    int(committed_rack_id),
                )
                if eta_picker is None:
                    continue

                eta_picker = int(eta_picker)
                picker_candidates.append(
                    {
                        "picker_id": int(picker_id),
                        "sync_cost": max(int(eta_agv), eta_picker),
                        "eta_gap": abs(int(eta_agv) - eta_picker),
                        "eta_picker": eta_picker,
                    }
                )

            # Keep deterministic ordering; Stage 2 can use these as ranking signals.
            picker_candidates.sort(
                key=lambda item: (
                    int(item["sync_cost"]),
                    int(item["eta_gap"]),
                    int(item["eta_picker"]),
                    int(item["picker_id"]),
                )
            )
            picker_candidates = picker_candidates[: max(0, int(self.config.stage2_picker_options_per_rack))]

            requests_payload.append(
                {
                    "request_id": request_id,
                    "committed_rack_id": int(committed_rack_id),
                    "picker_candidates": picker_candidates,
                }
            )

        return {
            "system_pressure": {
                "idle_pickers": int(idle_pickers),
            },
            "requests": requests_payload,
        }

    def _sanitize_stage2_output(self, payload: dict[str, Any], raw: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Sanitize Stage 2 ACK/BUSY output.

        Deterministic uniqueness repair is allowed:
        if the model ACKs the same picker for multiple requests in one batch,
        later requests are repaired to BUSY.
        """
        if not isinstance(raw, dict):
            return []

        raw_responses = raw.get("responses")
        if not isinstance(raw_responses, list):
            one = raw.get("response")
            if isinstance(one, dict):
                raw_responses = [one]
            else:
                return []

        request_map: dict[str, dict[str, Any]] = {}
        picker_candidates_by_request: dict[str, dict[int, int]] = {}

        for req in payload.get("requests", []):
            if not isinstance(req, dict):
                continue
            request_id = req.get("request_id")
            if not isinstance(request_id, str):
                continue
            request_map[request_id] = req
            picker_candidates_by_request[request_id] = {
                int(candidate["picker_id"]): int(candidate["eta_picker"])
                for candidate in req.get("picker_candidates", [])
                if isinstance(candidate, dict) and self._safe_int(candidate.get("picker_id")) > 0
            }

        raw_by_request: dict[str, dict[str, Any]] = {}
        for item in raw_responses:
            if not isinstance(item, dict):
                continue
            request_id = item.get("request_id")
            if isinstance(request_id, str) and request_id not in raw_by_request:
                raw_by_request[request_id] = item

        used_pickers: set[int] = set()
        sanitized: list[dict[str, Any]] = []

        for request_id, req in request_map.items():
            raw_item = raw_by_request.get(request_id, {})
            picker_candidates = picker_candidates_by_request.get(request_id, {})
            committed_rack_id = int(req.get("committed_rack_id", 0))

            status = str(raw_item.get("status", "BUSY")).upper()
            if status not in {"ACK", "BUSY"}:
                status = "BUSY"

            picker_id = self._safe_int(raw_item.get("picker_id"))
            if status == "ACK" and picker_id in picker_candidates and picker_id not in used_pickers:
                used_pickers.add(int(picker_id))
                sanitized.append(
                    {
                        "request_id": request_id,
                        "status": "ACK",
                        "picker_id": int(picker_id),
                        "committed_rack_id": int(committed_rack_id),
                        "eta_picker": int(picker_candidates[picker_id]),
                        "picker_candidates_count": int(len(picker_candidates)),
                    }
                )
                continue

            sanitized.append(
                {
                    "request_id": request_id,
                    "status": "BUSY",
                    "picker_id": None,
                    "committed_rack_id": int(committed_rack_id),
                    "eta_picker": None,
                    "picker_candidates_count": int(len(picker_candidates)),
                }
            )

        return sanitized

    def _stage2_system_prompt(self) -> str:
        return (
            "You are the picker-side coordination module in Stage 2 of a staged non-mutualistic coordination framework.\n"
            "\n"
            "Task:\n"
            "Each request already has one committed rack chosen by Stage 1.\n"
            "Your job is to decide whether that committed rack should receive picker support now.\n"
            "For each request, either:\n"
            "- return ACK with exactly one picker_id, or\n"
            "- return BUSY.\n"
            "\n"
            "What the input means:\n"
            "- idle_pickers tells you how many picker resources are currently available in this batch.\n"
            "- Each request already has one committed rack. You must not change it.\n"
            "- Each picker candidate is one possible picker for that already committed rack.\n"
            "- sync_cost is the main coordination cost. Lower sync_cost is generally better.\n"
            "- eta_gap measures AGV-picker arrival mismatch. Lower eta_gap is generally better.\n"
            "- eta_picker is the picker-side travel cost. Lower eta_picker is generally better.\n"
            "\n"
            "Decision rules:\n"
            "1) If a request has no picker candidates, return BUSY.\n"
            "2) If a request has picker candidates, it is supportable in principle.\n"
            "3) However, this is a batch-level decision: multiple supportable requests may compete for fewer idle pickers than the number of supportable requests.\n"
            "4) If picker resources are fewer than the number of supportable requests in the batch, assign pickers only to the most worthwhile requests.\n"
            "5) Requests that are supportable in principle but are not selected under the current batch-level competition must return BUSY.\n"
            "6) Do not ACK two different requests with the same picker in the same batch.\n"
            "\n"
            "How to compare supportable requests:\n"
            "1) Lower sync_cost is the main priority.\n"
            "2) If sync_cost is similar, prefer lower eta_gap.\n"
            "3) If still similar, prefer lower eta_picker.\n"
            "4) If still tied, prefer lower picker_id for determinism.\n"
            "\n"
            "How to choose the picker for an ACKed request:\n"
            "- Choose the picker candidate that gives the best support for that request using the same ranking logic.\n"
            "\n"
            "Important limits:\n"
            "- Do not change the committed rack.\n"
            "- Do not suggest another rack.\n"
            "- Use only the provided picker candidates.\n"
            "- Return exactly one status per request.\n"
            "- ACK means one specific picker is assigned now.\n"
            "- BUSY means either no feasible picker candidate exists, or the request is not selected under the current batch-level picker competition.\n"
            "\n"
            "Output JSON only.\n"
            'Return exactly: {"responses":[{"request_id":"...","status":"ACK","picker_id":5}]}'
        )

    def _stage2_user_prompt(self, payload: dict[str, Any]) -> str:
        minimal_requests = []
        for req in payload.get("requests", []):
            if not isinstance(req, dict):
                continue
            minimal_requests.append(
                {
                    "request_id": req.get("request_id"),
                    "committed_rack_id": req.get("committed_rack_id"),
                    "picker_candidates": [
                        {
                            "picker_id": candidate.get("picker_id"),
                            "sync_cost": candidate.get("sync_cost"),
                            "eta_gap": candidate.get("eta_gap"),
                            "eta_picker": candidate.get("eta_picker"),
                        }
                        for candidate in req.get("picker_candidates", [])
                        if isinstance(candidate, dict)
                    ],
                }
            )

        return json.dumps(
            {
                "system_pressure": payload.get("system_pressure", {}),
                "requests": minimal_requests,
            },
            ensure_ascii=True,
        )

    # ----------------------------
    # Stage 3: deterministic integration only
    # ----------------------------

    def _build_stage3_deterministic_plan(
        self,
        stage1_bundle: list[dict[str, Any]],
        stage2_responses: list[dict[str, Any]],
        reserved_picker_ids: set[int] | None = None,
        reserved_rack_ids: set[int] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """
        Deterministically assemble final assignments.

        No LLM is used here.
        No revision is allowed.
        """
        reserved_picker_ids = set() if reserved_picker_ids is None else {int(x) for x in reserved_picker_ids}
        reserved_rack_ids = set() if reserved_rack_ids is None else {int(x) for x in reserved_rack_ids}

        stage1_by_request: dict[str, dict[str, Any]] = {
            str(item["request_id"]): item
            for item in stage1_bundle
            if isinstance(item, dict) and isinstance(item.get("request_id"), str)
        }
        stage2_by_request: dict[str, dict[str, Any]] = {
            str(item["request_id"]): item
            for item in stage2_responses
            if isinstance(item, dict) and isinstance(item.get("request_id"), str)
        }

        assignments: list[dict[str, Any]] = []
        skipped: list[str] = []
        objective_requests: list[dict[str, Any]] = []

        used_pickers = set(reserved_picker_ids)
        used_racks = set(reserved_rack_ids)

        for request_id, stage1_item in stage1_by_request.items():
            stage2_item = stage2_by_request.get(request_id)
            if stage2_item is None:
                skipped.append(request_id)
                continue

            status = str(stage2_item.get("status", "BUSY")).upper()
            if status != "ACK":
                skipped.append(request_id)
                continue

            agv_id = int(stage1_item.get("agv_id", 0))
            rack_id = self._safe_int(stage1_item.get("committed_rack_id"))
            picker_id = self._safe_int(stage2_item.get("picker_id"))
            eta_picker = self._safe_int(stage2_item.get("eta_picker"))

            eta_agv = self._lookup_stage1_eta_agv(stage1_item, rack_id)

            if agv_id <= 0 or rack_id <= 0 or picker_id <= 0 or eta_picker < 0 or eta_agv < 0:
                skipped.append(request_id)
                continue
            if bool(self.config.unique_picker) and picker_id in used_pickers:
                skipped.append(request_id)
                continue
            if bool(self.config.unique_rack) and rack_id in used_racks:
                skipped.append(request_id)
                continue

            assignments.append(
                {
                    "request_id": request_id,
                    "agv_id": int(agv_id),
                    "picker_id": int(picker_id),
                    "rack_id": int(rack_id),
                }
            )
            used_pickers.add(int(picker_id))
            used_racks.add(int(rack_id))

            objective_requests.append(
                {
                    "request_id": request_id,
                    "options": [
                        {
                            "rack_id": int(rack_id),
                            "picker_id": int(picker_id),
                            "sync_cost": max(int(eta_agv), int(eta_picker)),
                            "eta_gap": abs(int(eta_agv) - int(eta_picker)),
                        }
                    ],
                }
            )

        final_plan = {
            "assignments": assignments,
            "skipped": self._dedupe_strings(skipped),
            "explanation": "",
        }
        return final_plan, objective_requests

    def _lookup_stage1_eta_agv(self, stage1_item: dict[str, Any], rack_id: int) -> int:
        """Post-hoc lookup for objective logging only."""
        for candidate in stage1_item.get("candidates", []):
            if not isinstance(candidate, dict):
                continue
            candidate_rack_id = self._safe_int(candidate.get("rack_id"))
            if int(candidate_rack_id) != int(rack_id):
                continue
            eta_agv = self._safe_int(candidate.get("eta_agv"))
            if eta_agv >= 0:
                return int(eta_agv)
        return -1

    # ----------------------------
    # Deterministic fallback
    # ----------------------------

    def _fallback_whole_plan(self, state: dict[str, Any], reason: str) -> list[int]:
        """
        Deterministic non-mutualistic partner-aware fallback.
        This must NOT call a symbiotic rule planner.
        """
        self.last_used_fallback = True
        if self.config.debug:
            print("WHOLE_PLAN_FALLBACK_NON_MUTUALISTIC_PARTNER_AWARE:", reason)

        self._update_active_assignments_from_state(state)
        self._mark_communication_effective()

        batch_requests = self._build_batch_requests(state)
        if not batch_requests:
            actions = self._assemble_actions_from_assignments(state, assignments=[])
            self.last_final_plan = _DictMessage(
                {
                    "assignments": [],
                    "skipped": [],
                    "explanation": f"FALLBACK_TO_RULE_NON_MUTUALISTIC_PARTNER_AWARE: {reason}",
                    "objective_scores": {
                        "num_assignments": 0,
                        "sum_sync_cost": 0,
                        "sum_eta_gap": 0,
                        "max_sync_cost": 0,
                    },
                    "planner_schema": "non_mutualistic_partner_aware_comm_llm_v2",
                    "stage2_semantics": "ack_busy_committed_target",
                }
            )
            return actions

        request_batches = self._chunk_requests(
            batch_requests,
            max(1, int(self.config.max_requests_per_batch)),
        )

        reserved_agv_ids_this_round: set[int] = set()
        reserved_picker_ids_this_round: set[int] = set()
        reserved_rack_ids_this_round: set[int] = set()

        all_stage1_requests: list[dict[str, Any]] = []
        all_stage2_responses: list[dict[str, Any]] = []
        all_assignments: list[dict[str, Any]] = []
        aggregate_skipped: list[str] = []
        aggregate_objective_requests: list[dict[str, Any]] = []

        for request_batch in request_batches:
            stage1_payload = self._build_stage1_payload(
                state=state,
                batch_requests=request_batch,
                reserved_rack_ids=reserved_rack_ids_this_round,
                reserved_agv_ids=reserved_agv_ids_this_round,
                reserved_picker_ids=reserved_picker_ids_this_round,
            )
            stage1_bundle = self._build_stage1_deterministic_from_payload(stage1_payload)
            all_stage1_requests.extend(stage1_bundle)

            if not stage1_bundle:
                continue

            stage2_payload = self._build_stage2_payload(
                state=state,
                stage1_bundle=stage1_bundle,
                reserved_picker_ids=reserved_picker_ids_this_round,
            )
            stage2_responses = self._build_stage2_deterministic_from_payload(stage2_payload)
            all_stage2_responses.extend(stage2_responses)

            batch_final_plan, batch_objective_requests = self._build_stage3_deterministic_plan(
                stage1_bundle=stage1_bundle,
                stage2_responses=stage2_responses,
                reserved_picker_ids=reserved_picker_ids_this_round,
                reserved_rack_ids=reserved_rack_ids_this_round,
            )

            batch_assignments = self._dedupe_assignments(
                batch_final_plan.get("assignments", []),
                reserved_picker_ids=reserved_picker_ids_this_round,
                reserved_rack_ids=reserved_rack_ids_this_round,
            )
            all_assignments.extend(batch_assignments)
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

            aggregate_skipped.extend(
                [
                    rid
                    for rid in batch_final_plan.get("skipped", [])
                    if isinstance(rid, str)
                ]
            )
            aggregate_objective_requests.extend(batch_objective_requests)

        actions = self._assemble_actions_from_assignments(state, assignments=all_assignments)

        final_plan = {
            "assignments": all_assignments,
            "skipped": self._dedupe_strings(aggregate_skipped),
            "explanation": f"FALLBACK_TO_RULE_NON_MUTUALISTIC_PARTNER_AWARE: {reason}",
            "planner_schema": "non_mutualistic_partner_aware_comm_llm_v2",
            "stage2_semantics": "ack_busy_committed_target",
        }
        final_plan["objective_scores"] = self._compute_objective_scores(
            {"requests": aggregate_objective_requests},
            final_plan,
        )

        self.last_request = _DictMessage({"requests": all_stage1_requests})
        self.last_response = _DictMessage({"responses": all_stage2_responses})
        self.last_final_plan = _DictMessage(final_plan)

        return actions

    def _build_stage1_deterministic_from_payload(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Deterministic AGV-side commitment fallback."""
        region_load = payload.get("system_pressure", {}).get("region_load", {})
        if not isinstance(region_load, dict):
            region_load = {}

        used_racks: set[int] = set()
        committed: list[dict[str, Any]] = []

        for req in payload.get("requests", []):
            if not isinstance(req, dict):
                continue
            request_id = req.get("request_id")
            if not isinstance(request_id, str):
                continue

            candidates = []
            for candidate in req.get("candidates", []):
                if not isinstance(candidate, dict):
                    continue
                rack_id = self._safe_int(candidate.get("rack_id"))
                eta_agv = self._safe_int(candidate.get("eta_agv"))
                region_id = self._safe_int(candidate.get("region_id"))
                nearby_idle_pickers = self._safe_int(candidate.get("nearby_idle_pickers"))
                if rack_id <= 0 or eta_agv < 0:
                    continue
                candidates.append(
                    {
                        "rack_id": int(rack_id),
                        "eta_agv": int(eta_agv),
                        "region_id": int(region_id),
                        "nearby_idle_pickers": int(max(0, nearby_idle_pickers)),
                    }
                )

            candidates.sort(
                key=lambda c: (
                    int(c["eta_agv"]),
                    -int(c["nearby_idle_pickers"]),
                    int(region_load.get(str(c["region_id"]), region_load.get(c["region_id"], 0))) if int(c["region_id"]) >= 0 else 0,
                    int(c["rack_id"]),
                )
            )

            chosen = None
            for candidate in candidates:
                rack_id = int(candidate["rack_id"])
                if rack_id in used_racks:
                    continue
                chosen = rack_id
                break

            if chosen is None:
                continue

            committed.append(
                {
                    "request_id": request_id,
                    "agv_id": int(req.get("agv_id", 0)),
                    "purpose": str(req.get("purpose", "")),
                    "committed_rack_id": int(chosen),
                    "candidates": req.get("candidates", []),
                }
            )
            used_racks.add(int(chosen))

        return committed

    def _build_stage2_deterministic_from_payload(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Deterministic picker-side ACK/BUSY fallback.

        Policy:
        - If no feasible picker candidates exist for the committed rack, return BUSY.
        - If at least one feasible picker candidate exists, return ACK with the best picker.
        - Picker scarcity is NOT used as a rejection factor.
        - Candidate ranking is deterministic:
            1) lower sync_cost
            2) lower eta_gap
            3) lower eta_picker
            4) lower picker_id
        """
        used_pickers: set[int] = set()
        responses: list[dict[str, Any]] = []

        for req in payload.get("requests", []):
            if not isinstance(req, dict):
                continue

            request_id = req.get("request_id")
            if not isinstance(request_id, str):
                continue

            candidates: list[dict[str, Any]] = []
            for candidate in req.get("picker_candidates", []):
                if not isinstance(candidate, dict):
                    continue

                picker_id = self._safe_int(candidate.get("picker_id"))
                sync_cost = self._safe_int(candidate.get("sync_cost"))
                eta_gap = self._safe_int(candidate.get("eta_gap"))
                eta_picker = self._safe_int(candidate.get("eta_picker"))

                if picker_id <= 0:
                    continue
                if sync_cost < 0 or eta_gap < 0 or eta_picker < 0:
                    continue

                candidates.append(
                    {
                        "picker_id": int(picker_id),
                        "sync_cost": int(sync_cost),
                        "eta_gap": int(eta_gap),
                        "eta_picker": int(eta_picker),
                    }
                )

            candidates.sort(
                key=lambda c: (
                    int(c["sync_cost"]),
                    int(c["eta_gap"]),
                    int(c["eta_picker"]),
                    int(c["picker_id"]),
                )
            )

            chosen = None
            for candidate in candidates:
                picker_id = int(candidate["picker_id"])
                if picker_id in used_pickers:
                    continue
                chosen = candidate
                break

            if chosen is None:
                responses.append(
                    {
                        "request_id": request_id,
                        "status": "BUSY",
                        "picker_id": None,
                        "committed_rack_id": int(req.get("committed_rack_id", 0)),
                        "eta_picker": None,
                        "picker_candidates_count": int(len(candidates)),
                    }
                )
                continue

            used_pickers.add(int(chosen["picker_id"]))
            responses.append(
                {
                    "request_id": request_id,
                    "status": "ACK",
                    "picker_id": int(chosen["picker_id"]),
                    "committed_rack_id": int(req.get("committed_rack_id", 0)),
                    "eta_picker": int(chosen["eta_picker"]),
                    "picker_candidates_count": int(len(candidates)),
                }
            )

        return responses

if __name__ == "__main__":
    planner = NonMutualisticCommLLMPlannerV2(NonMutualisticCommLLMPlannerV2Config(debug=True))
    print("NonMutualisticCommLLMPlannerV2 ready. Call plan(state) with a configured local vLLM server.")
