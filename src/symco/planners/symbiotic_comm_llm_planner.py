

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
  Stage 2 (Picker LLM): output per-request recommendations (<=2 options), not binding assignments.
  Stage 3 (Finalizer LLM): choose a conflict-free global assignment set from Stage2 options.

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
    stage2_max_options_per_request: int = 2  # model outputs <=2 joint options per request (sanitizer caps)

    # Communication triggering
    wait_timeout_steps: int = 40
    min_recommunication_gap_steps: int = 12
    idle_probe_gap_steps: int = 25   # 新增：当没有可用 picker 时，允许低频探测

    # Output constraints
    unique_picker: bool = True
    unique_rack: bool = True

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
        self.last_communication_step = self.step_counter
        self._record_communication_step(self.step_counter)
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
            self._debug_print_delivery_flow(
                state=state,
                batch_requests=batch_requests,
                returned_actions=actions,
                branch="COMM_WITHOUT_BATCH_REQUESTS",
            )
            return actions

        # -------- Stage 1 (AGV LLM): propose primary+backups --------
        stage1_payload = self._build_stage1_payload(state, batch_requests)
        if self.config.debug:
            print("STAGE1_PAYLOAD\n", json.dumps(stage1_payload, ensure_ascii=False, indent=2))

        stage1_raw: dict[str, Any] = {"requests": []}
        try:
            stage1_raw = self.agv_client.chat_json(
                self._stage1_system_prompt(),
                self._stage1_user_prompt(stage1_payload),
            )
        except Exception as exc:
            return self._fallback_whole_plan(state, reason=f"Stage1 LLM exception: {exc}")

        if self.config.debug:
            print("STAGE1_RAW\n", json.dumps(stage1_raw, ensure_ascii=False, indent=2))

        stage1_bundle = self._sanitize_stage1_output(stage1_payload, stage1_raw)

        # Log Stage1 output (sanitized)
        #self.last_request = {"requests": stage1_bundle}
        self.last_request = _DictMessage({"requests": stage1_bundle})

        # If Stage1 yields nothing, whole-plan fallback
        if not stage1_bundle:
            return self._fallback_whole_plan(state, reason="Stage1 produced no valid proposals.")

        # -------- Stage 2 (Picker LLM): feasibility + options (<=2/request) --------
        stage2_payload = self._build_stage2_payload(state, stage1_bundle)
        if self.config.debug:
            print("STAGE2_PAYLOAD\n", json.dumps(stage2_payload, ensure_ascii=False, indent=2))

        stage2_raw: dict[str, Any] = {"responses": []}
        try:
            stage2_raw = self.picker_client.chat_json(
                self._stage2_system_prompt(),
                self._stage2_user_prompt(stage2_payload),
            )
        except Exception as exc:
            return self._fallback_whole_plan(state, reason=f"Stage2 LLM exception: {exc}")

        if self.config.debug:
            print("STAGE2_RAW\n", json.dumps(stage2_raw, ensure_ascii=False, indent=2))

        stage2_responses = self._sanitize_stage2_output(stage2_payload, stage2_raw)

        #self.last_response = {"responses": stage2_responses}
        self.last_response = _DictMessage({"responses": stage2_responses})

        # -------- Stage 3 (Finalizer LLM): global assignment from options --------
        stage3_payload = self._build_stage3_payload(state, stage2_payload, stage2_responses)
        if self.config.debug:
            print("STAGE3_PAYLOAD\n", json.dumps(stage3_payload, ensure_ascii=False, indent=2))

        stage3_raw: dict[str, Any] = {"assignments": [], "skipped": []}
        try:
            stage3_raw = self.final_client.chat_json(
                self._stage3_system_prompt(),
                self._stage3_user_prompt(stage3_payload),
            )
        except Exception as exc:
            return self._fallback_whole_plan(state, reason=f"Stage3 LLM exception: {exc}")

        if self.config.debug:
            print("STAGE3_RAW\n", json.dumps(stage3_raw, ensure_ascii=False, indent=2))

        stage3_final = self._sanitize_stage3_output(stage3_payload, stage3_raw)

        if stage3_final is None:
            return self._fallback_whole_plan(state, reason="Stage3 invalid output; fallback to rule symbiotic.")

        # P2: compute objective scores for logging/analysis (not used to reject outputs)
        stage3_final["objective_scores"] = self._compute_objective_scores(stage3_payload, stage3_final)

        #self.last_final_plan = stage3_final
        self.last_final_plan = _DictMessage(stage3_final)

        assignments = stage3_final.get("assignments", [])
        actions = self._assemble_actions_from_assignments(state, assignments=assignments)
        self._debug_print_delivery_flow(
            state=state,
            batch_requests=batch_requests,
            returned_actions=actions,
            branch="COMMUNICATION",
        )

        # Update short-term cooperative assignment tracking (for triggering)
        self._register_active_assignments_from_assignments(state, assignments)

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

    # ----------------------------
    # Stage 1 payload + sanitize
    # ----------------------------

    def _build_stage1_payload(self, state: dict[str, Any], batch_requests: list[dict[str, Any]]) -> dict[str, Any]:
        agents = self._sorted_agents(state)
        idle_agvs = sum(1 for a in agents if a.get("type") == "AGV" and not bool(a.get("busy", False)))

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

            candidates = [
                {
                    "rack_id": rid,
                    "eta_agv": eta,
                    "region_id": self.rack_to_region.get(rid, -1) if self.rack_to_region else -1
                }
                for rid, eta in scored
            ]
            
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
                "active_cooperative_assignments": int(len(self.active_assignments)),
                "region_load": region_load,
            },
            "requests": req_payloads,
        }

    def _sanitize_stage1_output(self, payload: dict[str, Any], raw: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Strict semantic-preserving sanitization:
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
                    "reason": reason,
                    "candidates": meta["candidates"],  # keep minimal for later building
                }
            )
            seen.add(request_id)

        return sanitized

    # ----------------------------
    # Stage 2 payload + sanitize
    # ----------------------------

    def _build_stage2_payload(self, state: dict[str, Any], stage1_bundle: list[dict[str, Any]]) -> dict[str, Any]:
        agents = self._sorted_agents(state)
        idle_agvs = sum(1 for a in agents if a.get("type") == "AGV" and not bool(a.get("busy", False)))
        # Reserve pickers already committed to active cooperative assignments so
        # an unfinished unload/load does not lose its paired picker mid-execution.
        reserved_picker_ids = {
            int(assignment.get("picker_id", -1))
            for assignment in self.active_assignments.values()
            if isinstance(assignment, dict) and self._safe_int(assignment.get("picker_id")) > 0
        }
        available_pickers = [
            a
            for a in agents
            if a.get("type") == "PICKER"
            and not bool(a.get("busy", False))
            and int(a["id"]) not in reserved_picker_ids
        ]
        idle_pickers = len(available_pickers)

        requests_payload: list[dict[str, Any]] = []
        for req in stage1_bundle:
            request_id = str(req["request_id"])
            agv_id = int(req["agv_id"])
            purpose = str(req["purpose"])
            raw_candidates = req.get("candidates", [])
            candidate_entries: list[dict[str, Any]] = []
            request_options: list[dict[str, Any]] = []

            for c in raw_candidates:
                if not isinstance(c, dict):
                    continue
                rack_id = self._safe_int(c.get("rack_id"))
                eta_agv = self._safe_int(c.get("eta_agv"))
                if rack_id <= 0 or eta_agv < 0:
                    continue

                picker_opts: list[dict[str, int]] = []
                for p in available_pickers:
                    pid = int(p["id"])
                    eta_picker = self._safe_cost(self._agent_cost_map(state, "picker", pid), rack_id)
                    if eta_picker is None:
                        continue
                    eta_picker = int(eta_picker)
                    picker_opts.append(
                        {
                            "picker_id": int(pid),
                            "eta_picker": eta_picker,
                        }
                    )
                    request_options.append(
                        {
                            "rack_id": int(rack_id),
                            "picker_id": int(pid),
                            "eta_agv": int(eta_agv),
                            "eta_picker": eta_picker,
                        }
                    )

                picker_opts.sort(key=lambda item: (int(item["eta_picker"]), int(item["picker_id"])))
                picker_opts = picker_opts[: max(0, int(self.config.stage2_picker_options_per_rack))]

                candidate_entries.append(
                    {
                        "rack_id": int(rack_id),
                        "eta_agv": int(eta_agv),
                        "picker_options": picker_opts,
                    }
                )

            request_options.sort(
                key=lambda item: (
                    int(item["eta_picker"]),
                    int(item["rack_id"]),
                    int(item["picker_id"]),
                )
            )
            request_options = request_options[: max(0, int(self.config.stage2_max_options_per_request))]
            for idx, option in enumerate(request_options):
                option["option_id"] = f"OPT_{idx}"

            requests_payload.append(
                {
                    "request_id": request_id,
                    "agv_id": int(agv_id),
                    "purpose": purpose,
                    "candidates": candidate_entries,
                    "options": request_options,
                    "options_count": int(len(request_options)),
                }
            )

        return {
            "system_pressure": {
                "idle_agvs": int(idle_agvs),
                "idle_pickers": int(idle_pickers),
                "active_cooperative_assignments": int(len(self.active_assignments)),
            },
            "requests": requests_payload,
        }

    def _sanitize_stage2_output(self, payload: dict[str, Any], raw: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Strict semantic-preserving sanitization:
        - raw["responses"] must be a list (or wrapper {"response": {...}})
        - each response.request_id must exist
        - decision must be DECLINE or RECOMMEND
        - RECOMMEND must match exactly one communicated option by chosen_option_id
          or by (rack_id, picker_id)
        - options_count is carried into the sanitized trace for analysis
        - no default recommendation repair: invalid output becomes DECLINE
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
        allowed_pairs_by_req: dict[str, dict[tuple[int, int], dict[str, int]]] = {}
        options_count_by_req: dict[str, int] = {}

        for req in payload.get("requests", []):
            if not isinstance(req, dict):
                continue
            rid = req.get("request_id")
            if not isinstance(rid, str):
                continue
            req_map[rid] = req
            option_id_map: dict[str, dict[str, int]] = {}
            pair_map: dict[tuple[int, int], dict[str, int]] = {}
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
                pair_map[(int(rack_id), int(picker_id))] = data
            allowed_options_by_req[rid] = option_id_map
            allowed_pairs_by_req[rid] = pair_map
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
        for rid, req in req_map.items():
            raw_item = raw_by_id.get(rid, {})
            decision = raw_item.get("decision", "DECLINE")
            if decision is None:
                decision = "DECLINE"
            decision = str(decision).upper()
            if decision not in {"DECLINE", "RECOMMEND"}:
                decision = "DECLINE"
            allowed_by_option_id = allowed_options_by_req.get(rid, {})
            allowed_by_pair = allowed_pairs_by_req.get(rid, {})
            options_count = int(options_count_by_req.get(rid, 0))

            if options_count == 0:
                reason = str(raw_item.get("reason", "")) if raw_item.get("reason") is not None else ""
                if decision == "RECOMMEND":
                    reason = "Invalid response"
                sanitized.append(
                    {
                        "request_id": rid,
                        "decision": "DECLINE",
                        "options": [],
                        "options_count": 0,
                        "chosen_option_id": None,
                        "chosen_rack_id": None,
                        "chosen_picker_id": None,
                        "reason": reason,
                    }
                )
                continue

            selected: dict[str, int] | None = None
            chosen_option_id = raw_item.get("chosen_option_id")
            if decision == "RECOMMEND" and isinstance(chosen_option_id, str):
                selected = allowed_by_option_id.get(chosen_option_id)

            if decision == "RECOMMEND" and selected is None:
                rack_id = self._safe_int(raw_item.get("rack_id"))
                picker_id = self._safe_int(raw_item.get("picker_id"))
                if rack_id > 0 and picker_id > 0:
                    selected = allowed_by_pair.get((int(rack_id), int(picker_id)))
                    if selected is not None:
                        for option_id, option_data in allowed_by_option_id.items():
                            if option_data == selected:
                                chosen_option_id = option_id
                                break

            if decision != "RECOMMEND" or selected is None:
                sanitized.append(
                    {
                        "request_id": rid,
                        "decision": "DECLINE",
                        "options": [],
                        "options_count": options_count,
                        "chosen_option_id": None,
                        "chosen_rack_id": None,
                        "chosen_picker_id": None,
                        "reason": "Invalid response",
                    }
                )
                continue

            sync_cost = max(int(selected["eta_agv"]), int(selected["eta_picker"]))
            eta_gap = abs(int(selected["eta_agv"]) - int(selected["eta_picker"]))
            sanitized.append(
                {
                    "request_id": rid,
                    "decision": "RECOMMEND",
                    "options": [
                        {
                            "rack_id": int(selected["rack_id"]),
                            "picker_id": int(selected["picker_id"]),
                            "eta_picker": int(selected["eta_picker"]),
                            "sync_cost": int(sync_cost),
                            "eta_gap": int(eta_gap),
                        }
                    ],
                    "options_count": options_count,
                    "chosen_option_id": str(chosen_option_id) if isinstance(chosen_option_id, str) else None,
                    "chosen_rack_id": int(selected["rack_id"]),
                    "chosen_picker_id": int(selected["picker_id"]),
                    "reason": str(raw_item.get("reason", "")) if raw_item.get("reason") is not None else "",
                }
            )

        return sanitized

    # ----------------------------
    # Stage 3 payload + sanitize
    # ----------------------------

    def _build_stage3_payload(
        self,
        state: dict[str, Any],
        stage2_payload: dict[str, Any],
        stage2_responses: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compress to request-level options (<=2 each), include constraints and fixed actions."""
        fixed_direct_actions = self._build_fixed_direct_actions(state)

        req_meta: dict[str, dict[str, Any]] = {}
        for req in stage2_payload.get("requests", []):
            if not isinstance(req, dict):
                continue
            rid = req.get("request_id")
            if not isinstance(rid, str):
                continue
            req_meta[rid] = {
                "agv_id": int(req.get("agv_id", 0)),
                "purpose": str(req.get("purpose", "")),
            }

        responses_by_id = {
            r["request_id"]: r
            for r in stage2_responses
            if isinstance(r, dict) and isinstance(r.get("request_id"), str)
        }

        requests_payload: list[dict[str, Any]] = []
        for rid, meta in req_meta.items():
            resp = responses_by_id.get(rid, {"decision": "DECLINE", "options": []})
            options = resp.get("options", [])
            if not isinstance(options, list):
                options = []

            compact_options: list[dict[str, Any]] = []
            for opt in options:
                if not isinstance(opt, dict):
                    continue
                compact_options.append(
                    {
                        "rack_id": int(opt.get("rack_id", 0)),
                        "picker_id": int(opt.get("picker_id", 0)),
                        "sync_cost": int(opt.get("sync_cost", 0)),
                        "eta_gap": int(opt.get("eta_gap", 0)),
                    }
                )
                if len(compact_options) >= int(self.config.stage2_max_options_per_request):
                    break

            requests_payload.append(
                {
                    "request_id": rid,
                    "agv_id": int(meta["agv_id"]),
                    "purpose": str(meta["purpose"]),
                    "options": compact_options,
                }
            )

        sys_pressure = stage2_payload.get("system_pressure", {})
        if not isinstance(sys_pressure, dict):
            sys_pressure = {}

        return {
            "system_pressure": {
                "idle_agvs": int(sys_pressure.get("idle_agvs", 0)),
                "idle_pickers": int(sys_pressure.get("idle_pickers", 0)),
                "active_cooperative_assignments": int(sys_pressure.get("active_cooperative_assignments", 0)),
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
        Strict semantic-preserving sanitization:
        - require dict with "assignments" list and optional "skipped" list
        - assignments must pick only from provided request.options
        - enforce uniqueness constraints
        - if invalid structure -> None (whole-plan fallback)
        """
        if not isinstance(raw, dict):
            return None

        assignments_in = raw.get("assignments")
        skipped_in = raw.get("skipped", [])
        if not isinstance(assignments_in, list):
            return None
        if not isinstance(skipped_in, list):
            skipped_in = []

        # Build allowed options map: request_id -> set((rack,picker))
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
        for item in assignments_in:
            if not isinstance(item, dict):
                return None
            rid = item.get("request_id")
            if not isinstance(rid, str) or rid not in req_map:
                return None
            if rid in used_requests:
                return None

            agv_id = self._safe_int(item.get("agv_id"))
            rack_id = self._safe_int(item.get("rack_id"))
            picker_id = self._safe_int(item.get("picker_id"))
            if agv_id <= 0 or rack_id <= 0 or picker_id <= 0:
                return None

            meta_agv = int(req_map[rid].get("agv_id", 0))
            if meta_agv != int(agv_id):
                return None

            if (int(rack_id), int(picker_id)) not in allowed_pairs.get(rid, set()):
                return None

            if unique_picker and int(picker_id) in used_pickers:
                return None
            if unique_rack and int(rack_id) in used_racks:
                return None

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

        skipped: list[str] = []
        for s in skipped_in:
            if isinstance(s, str) and s in req_map and s not in used_requests:
                skipped.append(s)

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
        return (
            "You are the AGV-group coordinator (proposal stage). "
            "For EACH request, pick ONE primary rack and up to TWO backup racks from the provided candidates. "
            "Objective: produce a good AGV-side proposal bundle (NOT final global optimum). "
            "Guidelines: "
            "1) Try to minimize the maximum eta_agv among selected primary racks. "
            "2) If it does not worsen the maximum eta_agv, avoid duplicate primary racks across requests. "
            "3) Choose backups that are feasible and near (low eta_agv), different from primary. "
            "4) SPATIAL LOAD BALANCING: You will receive region_load (number of AGVs already assigned to each region). "
            "   Avoid concentrating proposals in the same region, especially if that region already has high load. "
            "   When multiple racks have similar eta_agv, prefer those in regions with lower current load. "
            "Constraints: primary/backup rack_id MUST come from that request's candidates. "
            "Return JSON only, exactly: "
            '{"requests":[{"request_id":"...","primary_rack_id":37,"backup_rack_ids":[52,41],"reason":""}]}'
        )

    def _stage1_user_prompt(self, payload: dict[str, Any]) -> str:
        minimal = {
            "system_pressure": payload.get("system_pressure", {}),
            "requests": [
                {
                    "request_id": r.get("request_id"),
                    "agv_id": r.get("agv_id"),
                    "purpose": r.get("purpose"),
                    "candidates": [
                        {"rack_id": c.get("rack_id"), "eta_agv": c.get("eta_agv"), "region_id": c.get("region_id")}
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
        return (
            "You are the Picker-group coordinator. "
            "For each request, you will receive at most two feasible options. "
            "Choose exactly one action per request: RECOMMEND one provided option, or DECLINE. "
            "Do not invent racks, pickers, or option ids. Use only the provided options. "
            "Return JSON only, exactly: "
            '{"responses":[{"request_id":"...","decision":"RECOMMEND","chosen_option_id":"OPT_0","reason":""}]}'
        )

    def _stage2_user_prompt(self, payload: dict[str, Any]) -> str:
        minimal_reqs = []
        for r in payload.get("requests", []):
            if not isinstance(r, dict):
                continue
            minimal_reqs.append(
                {
                    "request_id": r.get("request_id"),
                    "purpose": r.get("purpose"),
                    "options": [
                        {
                            "option_id": opt.get("option_id"),
                            "rack_id": opt.get("rack_id"),
                            "picker_id": opt.get("picker_id"),
                            "eta_picker": opt.get("eta_picker"),
                            "eta_agv": opt.get("eta_agv"),
                        }
                        for opt in r.get("options", [])
                        if isinstance(opt, dict)
                    ],
                }
            )

        minimal = {
            "system_pressure": payload.get("system_pressure", {}),
            "requests": minimal_reqs,
        }
        return json.dumps(minimal, ensure_ascii=True)

    def _stage3_system_prompt(self) -> str:
        return (
            "You are the final arbitration planner for ONE warehouse step. "
            "You will receive, for each request, up to TWO feasible options (rack_id, picker_id) with metrics. "
            "Choose a conflict-free set of assignments. "
            "Constraints: "
            "1) Use ONLY the provided options for each request (do not invent racks/pickers). "
            "2) Respect unique_picker and unique_rack if true. "
            "Objective (lexicographic): "
            "A) maximize number of assigned requests, "
            "B) minimize sum sync_cost, "
            "C) minimize sum eta_gap. "
            "Keep fixed_direct_actions unchanged (they are executed regardless). "
            "Return JSON only, exactly: "
            '{"assignments":[{"request_id":"...","agv_id":1,"picker_id":3,"rack_id":37}],"skipped":["..."],"explanation":""}'
        )

    def _stage3_user_prompt(self, payload: dict[str, Any]) -> str:
        minimal = {
            "system_pressure": payload.get("system_pressure", {}),
            "constraints": payload.get("constraints", {}),
            "fixed_direct_actions": payload.get("fixed_direct_actions", []),
            "requests": payload.get("requests", []),
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
