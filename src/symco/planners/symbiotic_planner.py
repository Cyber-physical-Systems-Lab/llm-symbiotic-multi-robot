"""Rule-based symbiotic planner stub for TA-RWARE coordination."""

from __future__ import annotations

from typing import Any

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


class SymbioticPlanner:
    """Protocol-faithful symbiotic planner with deterministic rule-based logic.

    The AGV side communicates a small candidate set per cooperative task.
    The picker side reasons only over those communicated candidates, which
    preserves the intended partial-observation protocol while still allowing
    negotiation through ``ACCEPT`` / ``DECLINE`` / ``PREFER``.
    """

    reasonable_eta_threshold = 12
    candidate_limit = 2

    def __init__(self) -> None:
        self.step_counter: int = 0
        self.active_assignments: dict[int, dict[str, Any]] = {}
        self.wait_timeout_steps: int = 20
        self.last_communication_triggered: bool = False
        self.last_request: AGVRequest | None = None
        self.last_response: PickerResponse | None = None
        self.last_final_plan: FinalPlan | None = None

    def plan(self, state: dict[str, Any]) -> list[int]:
        """Run the request/response/final-plan pipeline and return macro actions."""
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
        request = self._build_agv_request(state)
        response = self._build_picker_response(state, request)
        final_plan = self._build_final_plan(state, request, response)

        self.last_request = request
        self.last_response = response
        self.last_final_plan = final_plan
        self._register_assignments_from_final_plan(state, final_plan)

        num_agents = int(state.get("meta", {}).get("num_agents", 0))
        return final_plan_to_action_list(final_plan, num_agents)

    def _should_trigger_communication(self, state: dict[str, Any]) -> bool:
        """Return whether the current state warrants a communication round."""
        if bool(state.get("coordination_alert", False)):
            return True

        request_racks = state.get("requests_rack_ids_topk", [])
        empty_racks = state.get("empty_rack_ids_topk", [])
        agents = self._sorted_agents(state)
        agents_by_id = {int(agent["id"]): agent for agent in agents}

        for agv_id, assignment in self.active_assignments.items():
            agv_state = agents_by_id.get(int(agv_id))
            if agv_state is None:
                continue
            if self._is_assignment_completed(agv_state, assignment):
                continue
            if self.step_counter - int(assignment["start_step"]) >= self.wait_timeout_steps:
                return True

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

        return False

    def _zero_final_plan(self, state: dict[str, Any], explanation: str) -> FinalPlan:
        """Return an all-zero final plan for the current set of agents."""
        macro_actions = [
            FinalPlanItem(agent_id=int(agent["id"]), location_id=0)
            for agent in self._sorted_agents(state)
        ]
        return FinalPlan(macro_actions=macro_actions, explanation=explanation)

    def _build_agv_request(self, state: dict[str, Any]) -> AGVRequest:
        """Build AGV-side cooperation requests with small communicated candidate sets."""
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

        return AGVRequest(requests=requests, notes=["Rule-based AGV intentions over communicated candidates."])

    def _build_picker_response(self, state: dict[str, Any], request: AGVRequest) -> PickerResponse:
        """Build picker-side responses over communicated candidate sets only.

        The picker side does not search global rack pools. It evaluates only the
        candidates explicitly sent by the AGV side and either accepts the first
        candidate, prefers another communicated candidate, or declines.
        """
        agents = self._sorted_agents(state)
        responses: list[PickerResponseItem] = []

        for item in request.requests:
            best_choice = self._choose_best_picker_candidate(agents, state, item)
            if best_choice is None:
                responses.append(
                    PickerResponseItem(
                        request_id=item.request_id,
                        decision="DECLINE",
                        reason="No communicated candidate is feasible for any available picker.",
                    )
                )
                continue

            chosen_rack_id, picker_id, eta_picker = best_choice
            first_candidate_rack_id = item.candidates[0].rack_id
            decision = "ACCEPT" if chosen_rack_id == first_candidate_rack_id else "PREFER"
            responses.append(
                PickerResponseItem(
                    request_id=item.request_id,
                    decision=decision,
                    picker_id=picker_id,
                    eta_picker=eta_picker,
                    chosen_rack_id=chosen_rack_id,
                    reason="Best communicated candidate under combined AGV+picker cost.",
                )
            )

        return PickerResponse(responses=responses, notes=["Rule-based picker responses over communicated candidates."])

    def _build_final_plan(
        self,
        state: dict[str, Any],
        request: AGVRequest,
        response: PickerResponse,
    ) -> FinalPlan:
        """Build the final joint plan after incorporating picker feedback.

        ``PREFER`` is the causal communication case: the AGV may revise its
        target, but only to another rack that was already communicated in the
        request candidate set.
        """
        agents = self._sorted_agents(state)
        goal_ids = set(int(loc_id) for loc_id in state.get("goal_ids", []))
        response_by_request = {item.request_id: item for item in response.responses}
        request_by_id = {item.request_id: item for item in request.requests}
        macro_actions: list[FinalPlanItem] = []
        action_by_agent: dict[int, int] = {}
        assigned_racks: set[int] = set()
        assigned_pickers: set[int] = set()

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

        for request_id, request_item in request_by_id.items():
            agent_index = self._agent_index_by_id(agents, request_item.agv_id)
            if agent_index is None:
                continue

            reply = response_by_request.get(request_id)
            if reply is None or reply.decision == "DECLINE":
                action_by_agent[request_item.agv_id] = 0
                continue

            if reply.decision not in {"ACCEPT", "PREFER"}:
                action_by_agent[request_item.agv_id] = 0
                continue

            rack_id = int(reply.chosen_rack_id or 0)
            picker_id = reply.picker_id
            communicated_rack_ids = {candidate.rack_id for candidate in request_item.candidates}

            if rack_id == 0 or rack_id not in communicated_rack_ids:
                action_by_agent[request_item.agv_id] = 0
                continue
            if rack_id in goal_ids or rack_id in assigned_racks:
                action_by_agent[request_item.agv_id] = 0
                continue
            if not self._is_valid_action(state.get("valid_action_masks", []), agent_index, rack_id):
                action_by_agent[request_item.agv_id] = 0
                continue
            if picker_id is None or picker_id in assigned_pickers:
                action_by_agent[request_item.agv_id] = 0
                continue

            picker_agent = self._agent_by_id(agents, picker_id)
            if picker_agent is None or picker_agent.get("type") != "PICKER":
                action_by_agent[request_item.agv_id] = 0
                continue
            if bool(picker_agent.get("busy", False)):
                action_by_agent[request_item.agv_id] = 0
                continue
            if self._safe_cost(self._agent_cost_map(state, "picker", picker_id), rack_id) is None:
                action_by_agent[request_item.agv_id] = 0
                continue

            action_by_agent[request_item.agv_id] = rack_id
            action_by_agent[picker_id] = rack_id
            assigned_racks.add(rack_id)
            assigned_pickers.add(picker_id)

        for agent in agents:
            agent_id = int(agent["id"])
            macro_actions.append(
                FinalPlanItem(
                    agent_id=agent_id,
                    location_id=int(action_by_agent.get(agent_id, 0)),
                )
            )

        return FinalPlan(
            macro_actions=macro_actions,
            explanation="Rule-based symbiotic plan derived from communicated candidate sets.",
        )

    def _update_active_assignments_from_state(self, state: dict[str, Any]) -> None:
        """Clear stale or completed short-term cooperative assignments."""
        agents_by_id = {
            int(agent["id"]): agent
            for agent in self._sorted_agents(state)
            if agent.get("type") == "AGV"
        }
        active_assignments: dict[int, dict[str, Any]] = {}

        for agv_id, assignment in self.active_assignments.items():
            agv_state = agents_by_id.get(int(agv_id))
            if agv_state is None:
                continue
            if self._is_assignment_completed(agv_state, assignment):
                continue
            current_target = int(agv_state.get("target", 0) or 0)
            if current_target != int(assignment["rack_id"]):
                continue
            active_assignments[int(agv_id)] = assignment

        self.active_assignments = active_assignments

    def _register_assignments_from_final_plan(self, state: dict[str, Any], final_plan: FinalPlan) -> None:
        """Register newly formed AGV-picker cooperative assignments."""
        agents = self._sorted_agents(state)
        agents_by_id = {int(agent["id"]): agent for agent in agents}
        goal_ids = {int(loc_id) for loc_id in state.get("goal_ids", [])}
        location_to_agvs: dict[int, list[int]] = {}
        location_to_pickers: dict[int, list[int]] = {}

        for item in final_plan.macro_actions:
            location_id = int(item.location_id)
            if location_id == 0 or location_id in goal_ids:
                continue
            agent_state = agents_by_id.get(int(item.agent_id))
            if agent_state is None:
                continue
            if agent_state.get("type") == "AGV":
                location_to_agvs.setdefault(location_id, []).append(int(item.agent_id))
            elif agent_state.get("type") == "PICKER":
                location_to_pickers.setdefault(location_id, []).append(int(item.agent_id))

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
                self.active_assignments[int(agv_id)] = {
                    "picker_id": int(picker_id),
                    "rack_id": int(rack_id),
                    "purpose": purpose,
                    "start_step": self.step_counter,
                }

    def _is_assignment_completed(self, agv_state: dict[str, Any], assignment: dict[str, Any]) -> bool:
        """Return whether a cooperative assignment has completed."""
        purpose = str(assignment.get("purpose", ""))
        carrying = bool(agv_state.get("carrying", False))
        if purpose == "LOAD":
            return carrying
        if purpose == "UNLOAD":
            return not carrying
        return True

    def _build_candidate_racks(
        self,
        agent_index: int,
        agv_id: int,
        candidate_pool: list[int],
        state: dict[str, Any],
        reserved_primary_racks: set[int],
    ) -> list[CandidateRack]:
        """Build a ranked communicated candidate list for one AGV task."""
        scored: list[tuple[int, int]] = []
        cost_map = self._agent_cost_map(state, "agv", agv_id)
        valid_action_masks = state.get("valid_action_masks", [])

        for rack_id in candidate_pool:
            rack_id = int(rack_id)
            if not self._is_valid_action(valid_action_masks, agent_index, rack_id):
                continue
            cost = self._safe_cost(cost_map, rack_id)
            if cost is None:
                continue
            scored.append((rack_id, cost))

        scored.sort(key=lambda item: (item[1], item[0]))
        if not scored:
            return []

        ordered: list[tuple[int, int]] = []
        primary_added = False
        for rack_id, cost in scored:
            if not primary_added and rack_id not in reserved_primary_racks:
                ordered.append((rack_id, cost))
                primary_added = True
                break
        if not primary_added:
            ordered.append(scored[0])

        for rack_id, cost in scored:
            if len(ordered) >= self.candidate_limit:
                break
            if rack_id in {item[0] for item in ordered}:
                continue
            ordered.append((rack_id, cost))

        return [CandidateRack(rack_id=rack_id, eta_agv=cost) for rack_id, cost in ordered]

    def _choose_best_picker_candidate(
        self,
        agents: list[dict[str, Any]],
        state: dict[str, Any],
        request_item: AGVRequestItem,
    ) -> tuple[int, int, int] | None:
        """Choose the best communicated candidate and picker for one request."""
        best_choice: tuple[int, int, int] | None = None
        best_total_cost: int | None = None

        for candidate in request_item.candidates:
            picker_choice = self._find_best_picker(agents, state, candidate.rack_id)
            if picker_choice is None:
                continue
            picker_id, eta_picker = picker_choice
            total_cost = int(candidate.eta_agv) + int(eta_picker)
            if best_total_cost is None or total_cost < best_total_cost or (
                total_cost == best_total_cost and candidate.rack_id < best_choice[0]
            ):
                best_choice = (candidate.rack_id, picker_id, eta_picker)
                best_total_cost = total_cost

        return best_choice

    def _find_best_picker(
        self,
        agents: list[dict[str, Any]],
        state: dict[str, Any],
        rack_id: int,
    ) -> tuple[int, int] | None:
        """Return the nearest available picker for a rack target."""
        best_choice: tuple[int, int] | None = None
        best_eta: int | None = None

        for agent in agents:
            if agent.get("type") != "PICKER":
                continue
            if bool(agent.get("busy", False)):
                continue

            picker_id = int(agent["id"])
            eta_picker = self._safe_cost(self._agent_cost_map(state, "picker", picker_id), rack_id)
            if eta_picker is None:
                continue

            if best_eta is None or eta_picker < best_eta or (
                eta_picker == best_eta and picker_id < best_choice[0]
            ):
                best_choice = (picker_id, eta_picker)
                best_eta = eta_picker

        return best_choice

    def _choose_best_loc(
        self,
        agent_index: int,
        candidates: list[int],
        cost_map: dict[str, Any],
        valid_action_masks: list[list[int]],
        disallowed: set[int],
    ) -> int:
        """Choose the nearest valid location id for one AGV."""
        best_loc = 0
        best_cost: int | None = None

        for loc_id in candidates:
            loc_id = int(loc_id)
            if loc_id in disallowed:
                continue
            if not self._is_valid_action(valid_action_masks, agent_index, loc_id):
                continue

            cost = self._safe_cost(cost_map, loc_id)
            if cost is None:
                continue

            if best_cost is None or cost < best_cost or (cost == best_cost and loc_id < best_loc):
                best_loc = loc_id
                best_cost = cost

        return best_loc if best_cost is not None else 0

    def _sorted_agents(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        """Return agents sorted by id."""
        return sorted(state.get("agents", []), key=lambda agent: int(agent["id"]))

    def _agent_cost_map(self, state: dict[str, Any], group: str, agent_id: int) -> dict[str, Any]:
        """Fetch one agent's cost map from the state cost table."""
        return state.get("cost_table", {}).get(group, {}).get(str(int(agent_id)), {})

    def _agent_index_by_id(self, agents: list[dict[str, Any]], agent_id: int) -> int | None:
        """Return the sorted-list index for a given agent id."""
        for index, agent in enumerate(agents):
            if int(agent["id"]) == int(agent_id):
                return index
        return None

    def _agent_by_id(self, agents: list[dict[str, Any]], agent_id: int) -> dict[str, Any] | None:
        """Return an agent dict by id."""
        for agent in agents:
            if int(agent["id"]) == int(agent_id):
                return agent
        return None

    def _cooperative_purpose_for_agv(self, agv_state: dict[str, Any]) -> str | None:
        """Infer cooperative purpose from the AGV state."""
        carrying = bool(agv_state.get("carrying", False))
        has_delivered = bool(agv_state.get("has_delivered", False))
        if not carrying:
            return "LOAD"
        if carrying and has_delivered:
            return "UNLOAD"
        return None

    def _is_valid_action(
        self, valid_action_masks: list[list[int]], agent_index: int, loc_id: int
    ) -> bool:
        """Return whether a location id is currently valid for an AGV."""
        if agent_index < 0 or agent_index >= len(valid_action_masks):
            return False
        mask = valid_action_masks[agent_index]
        if loc_id < 0 or loc_id >= len(mask):
            return False
        return int(mask[loc_id]) == 1

    def _safe_cost(self, cost_map: dict[str, Any], loc_id: int) -> int | None:
        """Read an integer cost from the cost table, returning None if missing."""
        value = cost_map.get(str(int(loc_id)))
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None


if __name__ == "__main__":
    fake_state = {
        "meta": {"num_agents": 4},
        "agents": [
            {"id": 1, "type": "AGV", "busy": False, "carrying": False, "has_delivered": False, "target": 0},
            {"id": 2, "type": "AGV", "busy": False, "carrying": True, "has_delivered": False, "target": 0},
            {"id": 3, "type": "PICKER", "busy": False, "carrying": False, "has_delivered": False, "target": 0},
            {"id": 4, "type": "PICKER", "busy": False, "carrying": False, "has_delivered": False, "target": 0},
        ],
        "goal_ids": [1, 2],
        "requests_rack_ids_topk": [37, 42],
        "empty_rack_ids_topk": [51, 52],
        "valid_action_masks": [
            [1] * 60,
            [1] * 60,
            [1] * 60,
            [1] * 60,
        ],
        "cost_table": {
            "agv": {
                "1": {"37": 5, "42": 7, "1": 8, "2": 6, "51": 7, "52": 10},
                "2": {"37": 11, "42": 7, "1": 3, "2": 4, "51": 6, "52": 8},
            },
            "picker": {
                "3": {"37": 8, "42": 4, "51": 5, "52": 9},
                "4": {"37": 6, "42": 8, "51": 7, "52": 6},
            },
        },
    }

    planner = SymbioticPlanner()
    actions = planner.plan(fake_state)
    print(actions)
    print(planner.last_request.to_dict() if planner.last_request else {})
    print(planner.last_response.to_dict() if planner.last_response else {})
    print(planner.last_final_plan.to_dict() if planner.last_final_plan else {})
