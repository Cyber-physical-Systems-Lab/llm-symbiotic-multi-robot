"""
LLM-backed Step-1 symbiotic planner for TA-RWARE coordination (Mode S: STRICT).

Design (ablation / partial LLM):
- Stage 1 (AGVRequest): rule-based, communicates a SMALL candidate set per cooperative task.
- Stage 2 (PickerResponse): LLM-based, but MUST reason ONLY over communicated candidates/options.
- Stage 3 (FinalPlan): rule-based, converts request+response into executable macro actions.

Mode S (STRICT semantic-preserving sanitization):
✅ Allowed repairs (do NOT substitute decisions):
- Packaging repairs:
    {"response": {...}} -> {"responses":[...]}
    {"responses":[...]} already OK
- Field alias repairs:
    assigned_picker_id -> picker_id
    assigned_rack / chosen_rack_id -> chosen_rack_id   (accept both keys as aliases)
- Type coercion:
    "37" -> 37, "3" -> 3
- Ignore unrelated fields
- Defaults:
    reason=""

❌ Not allowed:
- Any fallback that ACCEPTs a candidate when the model output is invalid
- Any selection outside communicated candidate racks / per-rack picker_options
- Any "repair" that changes a DECLINE into ACCEPT/PREFER

If the model output is missing/invalid, we degrade to DECLINE (conservative, no target revision).

Notes:
- This file intentionally avoids any "zone" features.
- This implementation makes ONE LLM call per communicated request (conservative, robust for local models).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from symco.llm.vllm_client import VLLMChatClient, load_vllm_config_from_env
from symco.protocol.messages import (
    AGVRequest,
    AGVRequestItem,
    CandidateRack,
    FinalPlan,
    FinalPlanItem,
    PickerResponse,
    PickerResponseItem,
    final_plan_to_action_list,
)


@dataclass(slots=True)
class SymbioticLLMPlannerConfig:
    """Configuration for the symbiotic LLM planner (Mode S)."""

    candidate_limit: int = 3  # candidates per AGV request
    debug: bool = False
    wait_timeout_steps: int = 40
    min_recommunication_gap_steps: int = 8


class SymbioticLLMPlanner:
    """Symbiotic planner with rule-based requests and final plan, and LLM-based picker responses."""

    def __init__(self, config: SymbioticLLMPlannerConfig | None = None) -> None:
        self.config = config or SymbioticLLMPlannerConfig()
        self.client = VLLMChatClient(load_vllm_config_from_env())

        self.step_counter: int = 0
        self.active_assignments: dict[int, dict[str, Any]] = {}

        self.wait_timeout_steps: int = int(self.config.wait_timeout_steps)
        self.min_recommunication_gap_steps: int = int(self.config.min_recommunication_gap_steps)

        self.last_communication_triggered: bool = False
        self.last_communication_step: int = -10**9

        # Runner reads these for comm_metrics (they have to_dict()).
        self.last_request: AGVRequest | None = None
        self.last_response: PickerResponse | None = None
        self.last_final_plan: FinalPlan | None = None

    # =========================
    # Public planning entrypoint
    # =========================

    def plan(self, state: dict[str, Any]) -> list[int]:
        """Run event-triggered request/response/final-plan pipeline and return macro actions."""
        self.step_counter += 1
        self._update_active_assignments_from_state(state)

        if not self._should_trigger_communication(state):
            self.last_communication_triggered = False
            self.last_request = None
            self.last_response = None
            self.last_final_plan = self._zero_final_plan(
                state,
                explanation="No communication trigger; keep current missions.",
            )
            num_agents = int(state.get("meta", {}).get("num_agents", 0))
            return final_plan_to_action_list(self.last_final_plan, num_agents)

        self.last_communication_triggered = True
        self.last_communication_step = self.step_counter

        request = self._build_agv_request(state)
        response = self._build_picker_response_llm_strict(state, request)
        final_plan = self._build_final_plan(state, request, response)

        self.last_request = request
        self.last_response = response
        self.last_final_plan = final_plan
        self._register_assignments_from_final_plan(state, final_plan)

        num_agents = int(state.get("meta", {}).get("num_agents", 0))
        return final_plan_to_action_list(final_plan, num_agents)

    # =========================
    # Communication trigger policy (same spirit as your rule-based symbiotic planner)
    # =========================

    def _should_trigger_communication(self, state: dict[str, Any]) -> bool:
        """Event-triggered communication policy (conservative)."""
        if bool(state.get("coordination_alert", False)):
            return True

        request_racks = state.get("requests_rack_ids_topk", [])
        empty_racks = state.get("empty_rack_ids_topk", [])
        agents = self._sorted_agents(state)
        agents_by_id = {int(agent["id"]): agent for agent in agents}

        # Trigger when an AGV truly needs a new decision (target == 0) and is not currently busy.
        for agent in agents:
            if agent.get("type") != "AGV":
                continue
            if bool(agent.get("busy", False)):
                continue

            carrying = bool(agent.get("carrying", False))
            has_delivered = bool(agent.get("has_delivered", False))
            target = int(agent.get("target", 0) or 0)

            if not carrying and target == 0 and request_racks:
                return True
            if carrying and has_delivered and target == 0 and empty_racks:
                return True
            if carrying and not has_delivered and target == 0:
                return True

        recent_comm = (self.step_counter - self.last_communication_step) < self.min_recommunication_gap_steps

        # Re-trigger only when an existing cooperative assignment appears stalled / lost target / timed out.
        for agv_id, assignment in self.active_assignments.items():
            agv_state = agents_by_id.get(int(agv_id))
            if agv_state is None:
                continue
            if self._is_assignment_completed(agv_state, assignment):
                continue

            elapsed = self.step_counter - int(assignment["start_step"])
            current_target = int(agv_state.get("target", 0) or 0)
            expected_rack = int(assignment["rack_id"])
            busy = bool(agv_state.get("busy", False))

            if busy and current_target == expected_rack:
                continue

            if recent_comm and current_target == expected_rack:
                continue

            if current_target == 0:
                return True
            if elapsed >= self.wait_timeout_steps:
                return True

        return False

    # =========================
    # Stage 0 helpers
    # =========================

    def _zero_final_plan(self, state: dict[str, Any], explanation: str) -> FinalPlan:
        macro_actions = [
            FinalPlanItem(agent_id=int(agent["id"]), location_id=0)
            for agent in self._sorted_agents(state)
        ]
        return FinalPlan(macro_actions=macro_actions, explanation=explanation)

    def _sorted_agents(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        return sorted(state.get("agents", []), key=lambda agent: int(agent["id"]))

    def _agent_cost_map(self, state: dict[str, Any], group: str, agent_id: int) -> dict[str, Any]:
        return state.get("cost_table", {}).get(group, {}).get(str(int(agent_id)), {})

    def _safe_cost(self, cost_map: dict[str, Any], loc_id: int) -> int | None:
        value = cost_map.get(str(int(loc_id)))
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _safe_int(self, value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return -1

    def _is_valid_action(self, valid_action_masks: list[list[int]], agent_index: int, loc_id: int) -> bool:
        if agent_index < 0 or agent_index >= len(valid_action_masks):
            return False
        mask = valid_action_masks[agent_index]
        if loc_id < 0 or loc_id >= len(mask):
            return False
        return int(mask[loc_id]) == 1

    # =========================
    # Stage 1: AGVRequest (RULE-BASED, communicates candidate set)
    # =========================

    def _build_agv_request(self, state: dict[str, Any]) -> AGVRequest:
        agents = self._sorted_agents(state)
        requests: list[AGVRequestItem] = []
        reserved_primary_racks: set[int] = set()

        for agent_index, agent in enumerate(agents):
            if agent.get("type") != "AGV":
                continue
            if bool(agent.get("busy", False)):
                continue

            agv_id = int(agent["id"])
            carrying = bool(agent.get("carrying", False))
            has_delivered = bool(agent.get("has_delivered", False))

            # Carrying but not delivered -> go to goal (not a cooperative rack request).
            if carrying and not has_delivered:
                continue

            if carrying and has_delivered:
                purpose = "UNLOAD"
                candidate_pool = state.get("empty_rack_ids_topk", [])
            else:
                purpose = "LOAD"
                candidate_pool = state.get("requests_rack_ids_topk", [])

            candidates = self._build_candidate_racks(
                agent_index=agent_index,
                agv_id=agv_id,
                candidate_pool=[int(loc_id) for loc_id in candidate_pool],
                state=state,
                reserved_primary_racks=reserved_primary_racks,
                limit=int(self.config.candidate_limit),
            )
            if not candidates:
                continue

            reserved_primary_racks.add(candidates[0].rack_id)
            requests.append(
                AGVRequestItem(
                    request_id=f"agv-{agv_id}-{purpose.lower()}",
                    agv_id=agv_id,
                    purpose=purpose,
                    candidates=candidates,
                    priority=1,
                )
            )

        return AGVRequest(
            requests=requests,
            notes=["Rule-based AGV intentions over communicated candidates."],
        )

    def _build_candidate_racks(
        self,
        agent_index: int,
        agv_id: int,
        candidate_pool: list[int],
        state: dict[str, Any],
        reserved_primary_racks: set[int],
        limit: int,
    ) -> list[CandidateRack]:
        scored: list[tuple[int, int]] = []
        cost_map = self._agent_cost_map(state, "agv", agv_id)
        valid_masks = state.get("valid_action_masks", [])

        for rack_id in candidate_pool:
            rack_id = int(rack_id)
            if not self._is_valid_action(valid_masks, agent_index, rack_id):
                continue
            eta_agv = self._safe_cost(cost_map, rack_id)
            if eta_agv is None:
                continue
            scored.append((rack_id, eta_agv))

        scored.sort(key=lambda item: (item[1], item[0]))
        if not scored:
            return []

        # Primary candidate: try to avoid duplicates across AGVs.
        ordered: list[tuple[int, int]] = []
        primary_added = False
        for rack_id, eta in scored:
            if rack_id not in reserved_primary_racks:
                ordered.append((rack_id, eta))
                primary_added = True
                break
        if not primary_added:
            ordered.append(scored[0])

        # Fill remaining slots by best eta order.
        used = {ordered[0][0]}
        for rack_id, eta in scored:
            if len(ordered) >= max(1, limit):
                break
            if rack_id in used:
                continue
            ordered.append((rack_id, eta))
            used.add(rack_id)

        return [CandidateRack(rack_id=rack_id, eta_agv=eta) for rack_id, eta in ordered]

    # =========================
    # Stage 2: PickerResponse (LLM, STRICT Mode S)
    # =========================

    def _build_picker_response_llm_strict(self, state: dict[str, Any], request: AGVRequest) -> PickerResponse:
        """LLM-based picker responses over communicated candidates only (Mode S strict)."""
        agents = self._sorted_agents(state)
        responses: list[PickerResponseItem] = []

        for req_item in request.requests:
            picker_payload = self._picker_request_payload(state, agents, req_item)

            if self.config.debug:
                print("PICKER_SINGLE_REQUEST_PAYLOAD:")
                print(json.dumps(picker_payload, ensure_ascii=False, indent=2))

            raw: dict[str, Any] = {"responses": []}
            # One call per request (conservative).
            raw = self.client.chat_json(self._picker_system_prompt(), self._picker_user_prompt(picker_payload))

            if self.config.debug:
                print("LLM_PICKER_OUTPUT_RAW:")
                print(json.dumps(raw, ensure_ascii=False, indent=2))

            response_item = self._sanitize_picker_response_item_strict(
                state=state,
                agents=agents,
                request_item=req_item,
                llm_output=raw,
            )
            responses.append(response_item)

        return PickerResponse(
            responses=responses,
            notes=["LLM picker responses over communicated candidates (Mode S strict)."],
        )

    def _picker_request_payload(self, state: dict[str, Any], agents: list[dict[str, Any]], req_item: AGVRequestItem) -> dict[str, Any]:
        """Build minimal per-request payload: candidates + per-candidate picker options (eta_picker)."""
        available_pickers = [
            int(agent["id"])
            for agent in agents
            if agent.get("type") == "PICKER" and not bool(agent.get("busy", False))
        ]

        candidates_payload: list[dict[str, Any]] = []
        for cand in req_item.candidates:
            rack_id = int(cand.rack_id)
            picker_options: list[dict[str, Any]] = []
            for picker_id in available_pickers:
                eta_picker = self._safe_cost(self._agent_cost_map(state, "picker", picker_id), rack_id)
                if eta_picker is None:
                    continue
                picker_options.append({"picker_id": int(picker_id), "eta_picker": int(eta_picker)})

            candidates_payload.append(
                {
                    "rack_id": rack_id,
                    "eta_agv": int(cand.eta_agv),
                    "picker_options": picker_options,
                }
            )

        return {
            "request_id": req_item.request_id,
            "agv_id": int(req_item.agv_id),
            "purpose": str(req_item.purpose),
            "candidates": candidates_payload,
        }

    def _sanitize_picker_response_item_strict(
        self,
        state: dict[str, Any],
        agents: list[dict[str, Any]],
        request_item: AGVRequestItem,
        llm_output: dict[str, Any],
    ) -> PickerResponseItem:
        """Strict semantic-preserving sanitization for ONE request -> ONE response item.

        If invalid/missing, degrade to DECLINE (conservative, no decision substitution).
        """
        # --- Packaging repair ---
        raw_list = llm_output.get("responses")
        if not isinstance(raw_list, list):
            single = llm_output.get("response")
            if isinstance(single, dict):
                raw_list = [single]
            else:
                raw_list = []
        raw = raw_list[0] if raw_list and isinstance(raw_list[0], dict) else {}

        # --- request_id must match ---
        rid = raw.get("request_id")
        if not isinstance(rid, str):
            rid = request_item.request_id
        if rid != request_item.request_id:
            # Wrong id -> treat as invalid
            return PickerResponseItem(
                request_id=request_item.request_id,
                decision="DECLINE",
                reason="Invalid response_id mismatch.",
            )

        # --- decision validation (ACCEPT / PREFER / DECLINE) ---
        decision = raw.get("decision")
        if not isinstance(decision, str):
            decision = "DECLINE"
        decision = decision.strip().upper()
        if decision not in {"ACCEPT", "PREFER", "DECLINE"}:
            decision = "DECLINE"

        reason = raw.get("reason", "")
        if reason is None:
            reason = ""
        reason = str(reason)

        if decision == "DECLINE":
            return PickerResponseItem(
                request_id=request_item.request_id,
                decision="DECLINE",
                reason=reason,
            )

        # --- chosen rack id (aliases + type fix) ---
        chosen_rack_id = self._safe_int(raw.get("chosen_rack_id"))
        if chosen_rack_id < 0:
            chosen_rack_id = self._safe_int(raw.get("assigned_rack"))
        if chosen_rack_id < 0:
            # Missing choice -> invalid -> DECLINE (no fallback accept)
            return PickerResponseItem(
                request_id=request_item.request_id,
                decision="DECLINE",
                reason=reason or "Invalid output: missing chosen_rack_id.",
            )

        communicated_racks = {int(c.rack_id) for c in request_item.candidates}
        if chosen_rack_id not in communicated_racks:
            return PickerResponseItem(
                request_id=request_item.request_id,
                decision="DECLINE",
                reason=reason or "Invalid output: chosen_rack_id not in communicated candidates.",
            )

        # --- picker_id (alias + type fix) ---
        picker_id = self._safe_int(raw.get("picker_id"))
        if picker_id < 0:
            picker_id = self._safe_int(raw.get("assigned_picker_id"))
        if picker_id < 0:
            return PickerResponseItem(
                request_id=request_item.request_id,
                decision="DECLINE",
                reason=reason or "Invalid output: missing picker_id.",
            )

        # Must be currently available.
        valid_picker_ids = {
            int(agent["id"])
            for agent in agents
            if agent.get("type") == "PICKER" and not bool(agent.get("busy", False))
        }
        if picker_id not in valid_picker_ids:
            return PickerResponseItem(
                request_id=request_item.request_id,
                decision="DECLINE",
                reason=reason or "Invalid output: picker_id not available.",
            )

        # Must be in per-rack picker_options (STRICT match to communicated options).
        allowed_picker_ids = self._allowed_pickers_for_rack(state, valid_picker_ids, chosen_rack_id)
        if picker_id not in allowed_picker_ids:
            return PickerResponseItem(
                request_id=request_item.request_id,
                decision="DECLINE",
                reason=reason or "Invalid output: picker_id not feasible for chosen rack.",
            )

        eta_picker = self._safe_cost(self._agent_cost_map(state, "picker", picker_id), chosen_rack_id)
        if eta_picker is None:
            return PickerResponseItem(
                request_id=request_item.request_id,
                decision="DECLINE",
                reason=reason or "Invalid output: eta_picker missing for chosen rack.",
            )

        return PickerResponseItem(
            request_id=request_item.request_id,
            decision="PREFER" if decision == "PREFER" else "ACCEPT",
            picker_id=int(picker_id),
            eta_picker=int(eta_picker),
            chosen_rack_id=int(chosen_rack_id),
            reason=reason,
        )

    def _allowed_pickers_for_rack(self, state: dict[str, Any], available: set[int], rack_id: int) -> set[int]:
        """Compute which available pickers have a defined ETA to rack_id (this matches how options are built)."""
        allowed: set[int] = set()
        for picker_id in available:
            eta = self._safe_cost(self._agent_cost_map(state, "picker", picker_id), rack_id)
            if eta is not None:
                allowed.add(int(picker_id))
        return allowed

    def _picker_system_prompt(self) -> str:
        """Picker-side system prompt with explicit objective + strict constraints."""
        return (
            "You are the PICKER-side negotiator in a symbiotic coordination protocol.\n"
            "You must decide for ONE AGV request.\n\n"
            "You are ONLY allowed to reason over the candidates provided in the message.\n"
            "You do NOT have access to any global request pool.\n\n"
            "Valid decisions:\n"
            "- ACCEPT: choose the FIRST candidate (index 0) and assign one picker_id from that candidate's picker_options\n"
            "- PREFER: choose a DIFFERENT candidate (index > 0) and assign one picker_id from that candidate's picker_options\n"
            "- DECLINE: only if NO candidate has ANY feasible picker_option\n\n"
            "Objective:\n"
            "Minimize combined cost (eta_agv + eta_picker). If ties: choose smaller rack_id, then smaller picker_id.\n\n"
            "Constraints:\n"
            "- chosen_rack_id MUST be one of the provided candidates' rack_id.\n"
            "- picker_id MUST be one of the provided picker_options for the chosen_rack_id.\n"
            "- Output JSON ONLY. No text outside JSON.\n\n"
            "Required JSON format:\n"
            '{"responses":[{"request_id":"...","decision":"ACCEPT","chosen_rack_id":37,"picker_id":3,"reason":""}]}'
        )

    def _picker_user_prompt(self, payload: dict[str, Any]) -> str:
        """Picker-side user prompt (minimal JSON)."""
        # Minimal: request_id, candidates with rack_id, eta_agv, picker_options
        minimal = {
            "request_id": payload.get("request_id"),
            "agv_id": payload.get("agv_id"),
            "purpose": payload.get("purpose"),
            "candidates": [],
        }
        candidates = payload.get("candidates", [])
        if isinstance(candidates, list):
            for c in candidates:
                if not isinstance(c, dict):
                    continue
                minimal["candidates"].append(
                    {
                        "rack_id": c.get("rack_id"),
                        "eta_agv": c.get("eta_agv"),
                        "picker_options": [
                            {"picker_id": o.get("picker_id"), "eta_picker": o.get("eta_picker")}
                            for o in (c.get("picker_options", []) if isinstance(c.get("picker_options", []), list) else [])
                            if isinstance(o, dict)
                        ],
                    }
                )
        return json.dumps(minimal, ensure_ascii=True)

    # =========================
    # Stage 3: FinalPlan (RULE-BASED, incorporates ACCEPT/PREFER/DECLINE)
    # =========================

    def _build_final_plan(self, state: dict[str, Any], request: AGVRequest, response: PickerResponse) -> FinalPlan:
        agents = self._sorted_agents(state)
        goal_ids = {int(loc_id) for loc_id in state.get("goal_ids", [])}
        response_by_request = {item.request_id: item for item in response.responses}
        request_by_id = {item.request_id: item for item in request.requests}

        action_by_agent: dict[int, int] = {}
        assigned_racks: set[int] = set()
        assigned_pickers: set[int] = set()

        # Rule-based: carrying but not delivered -> go to goal.
        for agent_index, agent in enumerate(agents):
            agent_id = int(agent["id"])
            if bool(agent.get("busy", False)):
                action_by_agent[agent_id] = 0
                continue
            if agent.get("type") != "AGV":
                action_by_agent.setdefault(agent_id, 0)
                continue

            if bool(agent.get("carrying", False)) and not bool(agent.get("has_delivered", False)):
                goal_id = self._choose_best_loc(
                    agent_index=agent_index,
                    candidates=[int(loc_id) for loc_id in state.get("goal_ids", [])],
                    cost_map=self._agent_cost_map(state, "agv", agent_id),
                    valid_action_masks=state.get("valid_action_masks", []),
                    disallowed=set(),
                )
                action_by_agent[agent_id] = goal_id
            else:
                action_by_agent.setdefault(agent_id, 0)

        # Cooperative rack assignments from responses.
        for request_id, req_item in request_by_id.items():
            agent_index = self._agent_index_by_id(agents, req_item.agv_id)
            if agent_index is None:
                continue

            reply = response_by_request.get(request_id)
            if reply is None or reply.decision == "DECLINE":
                action_by_agent[req_item.agv_id] = 0
                continue
            if reply.decision not in {"ACCEPT", "PREFER"}:
                action_by_agent[req_item.agv_id] = 0
                continue

            rack_id = int(getattr(reply, "chosen_rack_id", 0) or 0)
            picker_id = reply.picker_id
            communicated = {int(c.rack_id) for c in req_item.candidates}

            if rack_id == 0 or rack_id not in communicated:
                action_by_agent[req_item.agv_id] = 0
                continue
            if rack_id in goal_ids or rack_id in assigned_racks:
                action_by_agent[req_item.agv_id] = 0
                continue
            if not self._is_valid_action(state.get("valid_action_masks", []), agent_index, rack_id):
                action_by_agent[req_item.agv_id] = 0
                continue
            if picker_id is None or picker_id in assigned_pickers:
                action_by_agent[req_item.agv_id] = 0
                continue

            picker_agent = self._agent_by_id(agents, picker_id)
            if picker_agent is None or picker_agent.get("type") != "PICKER":
                action_by_agent[req_item.agv_id] = 0
                continue
            if bool(picker_agent.get("busy", False)):
                action_by_agent[req_item.agv_id] = 0
                continue
            if self._safe_cost(self._agent_cost_map(state, "picker", picker_id), rack_id) is None:
                action_by_agent[req_item.agv_id] = 0
                continue

            action_by_agent[req_item.agv_id] = rack_id
            action_by_agent[picker_id] = rack_id
            assigned_racks.add(rack_id)
            assigned_pickers.add(picker_id)

        macro_actions = [
            FinalPlanItem(agent_id=int(agent["id"]), location_id=int(action_by_agent.get(int(agent["id"]), 0)))
            for agent in agents
        ]
        return FinalPlan(
            macro_actions=macro_actions,
            explanation="Symbiotic LLM (Mode S) plan: rule-based request + LLM response + rule-based final plan.",
        )

    def _choose_best_loc(
        self,
        agent_index: int,
        candidates: list[int],
        cost_map: dict[str, Any],
        valid_action_masks: list[list[int]],
        disallowed: set[int],
    ) -> int:
        """Rule-based selection: nearest valid by cost."""
        best_loc = 0
        best_cost: int | None = None
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

    # =========================
    # Active assignment bookkeeping (same purpose as in your symbiotic_planner.py)
    # =========================

    def _update_active_assignments_from_state(self, state: dict[str, Any]) -> None:
        agents_by_id = {
            int(agent["id"]): agent
            for agent in self._sorted_agents(state)
            if agent.get("type") == "AGV"
        }
        updated: dict[int, dict[str, Any]] = {}
        for agv_id, assignment in self.active_assignments.items():
            agv_state = agents_by_id.get(int(agv_id))
            if agv_state is None:
                continue
            if self._is_assignment_completed(agv_state, assignment):
                continue
            current_target = int(agv_state.get("target", 0) or 0)
            if current_target not in {0, int(assignment["rack_id"])}:
                continue
            updated[int(agv_id)] = assignment
        self.active_assignments = updated

    def _register_assignments_from_final_plan(self, state: dict[str, Any], final_plan: FinalPlan) -> None:
        agents = self._sorted_agents(state)
        agents_by_id = {int(agent["id"]): agent for agent in agents}
        goal_ids = {int(loc_id) for loc_id in state.get("goal_ids", [])}

        location_to_agvs: dict[int, list[int]] = {}
        location_to_pickers: dict[int, list[int]] = {}

        for item in final_plan.macro_actions:
            loc = int(item.location_id)
            if loc == 0 or loc in goal_ids:
                continue
            agent_state = agents_by_id.get(int(item.agent_id))
            if agent_state is None:
                continue
            if agent_state.get("type") == "AGV":
                location_to_agvs.setdefault(loc, []).append(int(item.agent_id))
            elif agent_state.get("type") == "PICKER":
                location_to_pickers.setdefault(loc, []).append(int(item.agent_id))

        new_assignments = dict(self.active_assignments)
        for rack_id, agv_ids in location_to_agvs.items():
            picker_ids = sorted(location_to_pickers.get(rack_id, []))
            if not picker_ids:
                continue
            for agv_id, picker_id in zip(sorted(agv_ids), picker_ids):
                agv_state = agents_by_id.get(int(agv_id))
                if agv_state is None:
                    continue
                purpose = self._cooperative_purpose_for_agv(agv_state)
                if purpose is None:
                    continue

                existing = new_assignments.get(int(agv_id))
                if (
                    existing is not None
                    and int(existing["picker_id"]) == int(picker_id)
                    and int(existing["rack_id"]) == int(rack_id)
                    and str(existing["purpose"]) == purpose
                ):
                    continue

                new_assignments[int(agv_id)] = {
                    "picker_id": int(picker_id),
                    "rack_id": int(rack_id),
                    "purpose": purpose,
                    "start_step": self.step_counter,
                }

        self.active_assignments = new_assignments

    def _is_assignment_completed(self, agv_state: dict[str, Any], assignment: dict[str, Any]) -> bool:
        purpose = str(assignment.get("purpose", ""))
        carrying = bool(agv_state.get("carrying", False))
        if purpose == "LOAD":
            return carrying
        if purpose == "UNLOAD":
            return not carrying
        return True

    def _cooperative_purpose_for_agv(self, agv_state: dict[str, Any]) -> str | None:
        carrying = bool(agv_state.get("carrying", False))
        has_delivered = bool(agv_state.get("has_delivered", False))
        if not carrying:
            return "LOAD"
        if carrying and has_delivered:
            return "UNLOAD"
        return None

    def _agent_index_by_id(self, agents: list[dict[str, Any]], agent_id: int) -> int | None:
        for idx, agent in enumerate(agents):
            if int(agent["id"]) == int(agent_id):
                return idx
        return None

    def _agent_by_id(self, agents: list[dict[str, Any]], agent_id: int) -> dict[str, Any] | None:
        for agent in agents:
            if int(agent["id"]) == int(agent_id):
                return agent
        return None


if __name__ == "__main__":
    # Minimal smoke test without environment.
    fake_state = {
        "meta": {"num_agents": 4},
        "agents": [
            {"id": 1, "type": "AGV", "busy": False, "carrying": False, "has_delivered": False, "target": 0},
            {"id": 2, "type": "AGV", "busy": False, "carrying": False, "has_delivered": False, "target": 0},
            {"id": 3, "type": "PICKER", "busy": False, "carrying": False, "has_delivered": False, "target": 0},
            {"id": 4, "type": "PICKER", "busy": False, "carrying": False, "has_delivered": False, "target": 0},
        ],
        "goal_ids": [1, 2],
        "requests_rack_ids_topk": [37, 42, 55],
        "empty_rack_ids_topk": [51, 52],
        "valid_action_masks": [[1] * 120 for _ in range(4)],
        "cost_table": {
            "agv": {"1": {"37": 5, "42": 8, "55": 11, "1": 4, "2": 6}, "2": {"37": 6, "42": 7, "55": 9, "1": 3, "2": 5}},
            "picker": {"3": {"37": 4, "42": 6, "55": 10}, "4": {"37": 7, "42": 3, "55": 8}},
        },
    }

    planner = SymbioticLLMPlanner(SymbioticLLMPlannerConfig(debug=True))
    try:
        actions = planner.plan(fake_state)
        print("ACTIONS:", actions)
        print("LAST_REQUEST:", planner.last_request.to_dict() if planner.last_request else None)
        print("LAST_RESPONSE:", planner.last_response.to_dict() if planner.last_response else None)
        print("LAST_FINAL_PLAN:", planner.last_final_plan.to_dict() if planner.last_final_plan else None)
    except Exception as exc:
        print({"error": str(exc)})