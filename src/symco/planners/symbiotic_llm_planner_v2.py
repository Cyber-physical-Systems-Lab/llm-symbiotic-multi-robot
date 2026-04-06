"""LLM-backed Step-2 symbiotic planner for TA-RWARE coordination.

Design:
- Rule-based AGVRequest generation (inherited from SymbioticPlanner)
- LLM-based PickerResponse generation (inherited from SymbioticLLMPlanner)
- LLM-based final integration at the request level
- Rule-based assembly of macro actions from request-level decisions
- Safe fallback to inherited Step-1 rule finalizer when LLM output is invalid

This version is intentionally conservative:
- Final-plan LLM does NOT output all macro actions directly.
- Instead it decides, per communicated cooperative request:
    USE_FIRST
    USE_PREFERRED
    SKIP
- We only adopt USE_PREFERRED when picker-side preference is valid and
  provides meaningful coordination gain.
- If the LLM is over-conservative (e.g. skips all feasible requests), we
  fall back to the inherited Step-1 rule-based finalizer.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from symco.planners.symbiotic_llm_planner import (
    SymbioticLLMPlanner,
    SymbioticLLMPlannerConfig,
)
from symco.protocol.messages import FinalPlan, FinalPlanItem, final_plan_to_action_list


@dataclass(slots=True)
class SymbioticLLMPlannerV2Config(SymbioticLLMPlannerConfig):
    """Configuration for the Step-2 symbiotic LLM planner."""

    use_llm_final_plan: bool = True
    # Only allow USE_PREFERRED when total cost improves by at least this amount.
    prefer_min_total_gain: int = 2
    # If first candidate is feasible, do not allow SKIP unless picker DECLINEs.
    allow_skip_when_first_feasible: bool = False


class SymbioticLLMPlannerV2(SymbioticLLMPlanner):
    """Step-2 symbiotic planner with LLM picker response and LLM final integration."""

    def __init__(self, config: SymbioticLLMPlannerV2Config | None = None):
        resolved = config or SymbioticLLMPlannerV2Config()
        super().__init__(resolved)
        self.config = resolved

        self.last_final_prompt_payload: dict[str, Any] | None = None
        self.last_final_llm_raw: dict[str, Any] | None = None
        self.last_final_used_fallback: bool = False

    # ------------------------------------------------------------------
    # Final integration
    # ------------------------------------------------------------------

    def _build_final_plan(self, state: dict[str, Any], request, response) -> FinalPlan:
        """Build final plan using request-level LLM integration with safe fallback."""
        if not self.config.use_llm_final_plan:
            self.last_final_used_fallback = False
            final_plan = super()._build_final_plan(state, request, response)
            if self.config.debug:
                print("FINAL_USED_FALLBACK")
                print(False)
                print("FINAL_PLAN")
                print(json.dumps(final_plan.to_dict(), indent=2, ensure_ascii=False))
            return final_plan

        payload = self._build_final_prompt_payload(state, request, response)
        self.last_final_prompt_payload = payload

        if self.config.debug:
            print("FINAL_PROMPT_PAYLOAD")
            print(json.dumps(payload, indent=2, ensure_ascii=False))

        try:
            raw = self.client.chat_json(
                self._final_system_prompt(),
                self._final_user_prompt(payload),
            )
            self.last_final_llm_raw = raw

            if self.config.debug:
                print("LLM_FINAL_OUTPUT_RAW")
                print(json.dumps(raw, indent=2, ensure_ascii=False))

            sanitized_decisions = self._sanitize_final_decisions(payload, raw)
            if sanitized_decisions is not None:
                final_plan = self._assemble_final_plan_from_decisions(
                    state=state,
                    request=request,
                    response=response,
                    decisions=sanitized_decisions,
                )
                self.last_final_used_fallback = False
                if self.config.debug:
                    print("FINAL_USED_FALLBACK")
                    print(False)
                    print("FINAL_PLAN")
                    print(json.dumps(final_plan.to_dict(), indent=2, ensure_ascii=False))
                return final_plan

        except Exception as exc:
            self.last_final_llm_raw = {"error": str(exc)}
            if self.config.debug:
                print("LLM_FINAL_OUTPUT_RAW")
                print(json.dumps(self.last_final_llm_raw, indent=2, ensure_ascii=False))

        self.last_final_used_fallback = True
        fallback_plan = super()._build_final_plan(state, request, response)
        if self.config.debug:
            print("FINAL_USED_FALLBACK")
            print(True)
            print("FINAL_PLAN")
            print(json.dumps(fallback_plan.to_dict(), indent=2, ensure_ascii=False))
        return fallback_plan

    # ------------------------------------------------------------------
    # Prompt payload
    # ------------------------------------------------------------------

    def _build_final_prompt_payload(self, state: dict[str, Any], request, response) -> dict[str, Any]:
        """Build compact request-level final integration payload."""
        fixed_direct_actions = self._build_fixed_direct_actions(state, request)
        agents = self._sorted_agents(state)

        idle_agvs = sum(
            1
            for agent in agents
            if agent.get("type") == "AGV" and not bool(agent.get("busy", False))
        )
        idle_pickers = sum(
            1
            for agent in agents
            if agent.get("type") == "PICKER" and not bool(agent.get("busy", False))
        )

        response_by_request_id = {item.request_id: item for item in response.responses}

        requests_payload: list[dict[str, Any]] = []
        for item in request.requests:
            agv_id = int(item.agv_id)
            candidates = [
                {
                    "rack_id": int(candidate.rack_id),
                    "eta_agv": int(candidate.eta_agv),
                }
                for candidate in item.candidates
            ]

            candidate_details = []
            for candidate in item.candidates:
                rack_id = int(candidate.rack_id)
                eta_agv = int(candidate.eta_agv)
                best_picker_id, best_eta_picker = self._best_picker_for_rack(state, rack_id)
                total_cost = None
                if best_picker_id is not None and best_eta_picker is not None:
                    total_cost = int(eta_agv) + int(best_eta_picker)

                candidate_details.append(
                    {
                        "rack_id": rack_id,
                        "eta_agv": eta_agv,
                        "best_picker_id": best_picker_id,
                        "best_eta_picker": best_eta_picker,
                        "total_cost": total_cost,
                    }
                )

            first_option = candidate_details[0] if candidate_details else None

            reply = response_by_request_id.get(item.request_id)
            picker_response = {
                "decision": "DECLINE",
                "picker_id": None,
                "chosen_rack_id": None,
                "eta_picker": None,
            }
            preferred_option = None

            if reply is not None:
                picker_response = {
                    "decision": reply.decision,
                    "picker_id": reply.picker_id,
                    "chosen_rack_id": reply.chosen_rack_id,
                    "eta_picker": reply.eta_picker,
                }

                if reply.decision == "PREFER" and reply.chosen_rack_id is not None:
                    chosen_rack_id = int(reply.chosen_rack_id)
                    for detail in candidate_details:
                        if int(detail["rack_id"]) == chosen_rack_id:
                            preferred_option = detail
                            break

            requests_payload.append(
                {
                    "request_id": item.request_id,
                    "agv_id": agv_id,
                    "purpose": item.purpose,
                    "priority": int(item.priority),
                    "candidates": candidates,
                    "first_option": first_option,
                    "preferred_option": preferred_option,
                    "picker_response": picker_response,
                }
            )

        return {
            "requests": requests_payload,
            "fixed_direct_actions": [
                {"agent_id": int(agent_id), "location_id": int(location_id)}
                for agent_id, location_id in sorted(fixed_direct_actions.items())
            ],
            "active_assignments_summary": [
                {
                    "agv_id": int(agv_id),
                    "picker_id": int(assignment["picker_id"]),
                    "rack_id": int(assignment["rack_id"]),
                    "purpose": str(assignment["purpose"]),
                }
                for agv_id, assignment in sorted(self.active_assignments.items())
            ],
            "system_pressure": {
                "idle_agvs": int(idle_agvs),
                "idle_pickers": int(idle_pickers),
                "active_cooperative_assignments": int(len(self.active_assignments)),
            },
        }

    # ------------------------------------------------------------------
    # Prompt text
    # ------------------------------------------------------------------

    def _final_system_prompt(self) -> str:
        """Return a short strict final integration prompt suitable for local 7B inference."""
        return (
            "You are the AGV-side final coordination planner for one warehouse step. "
            "For each cooperative request, output exactly one decision: USE_FIRST, USE_PREFERRED, or SKIP. "
            "Rules: "
            "1. USE_FIRST means assign the first communicated candidate. "
            "2. USE_PREFERRED means adopt the picker preferred communicated candidate, but only if picker_response.decision is PREFER. "
            "3. SKIP means do not assign that cooperative request now. "
            "4. If picker_response.decision is DECLINE, choose SKIP. "
            "5. Use USE_PREFERRED only when the preferred option is clearly better than the first option. "
            "6. Keep fixed_direct_actions unchanged. "
            "7. Return JSON only. "
            'Use exactly this schema: {"request_actions":[{"request_id":"...","decision":"USE_FIRST"}],"explanation":""}'
        )

    def _final_user_prompt(self, payload: dict[str, Any]) -> str:
        """Return compact final integration payload."""
        return json.dumps(payload, ensure_ascii=True)

    # ------------------------------------------------------------------
    # Sanitization
    # ------------------------------------------------------------------

    def _sanitize_final_decisions(
        self,
        payload: dict[str, Any],
        raw: dict[str, Any],
    ) -> list[dict[str, Any]] | None:
        """Sanitize LLM request-level decisions.

        Conservative policy:
        - If first candidate is feasible and model says SKIP, normalize to USE_FIRST
          unless picker explicitly DECLINEs.
        - Only allow USE_PREFERRED when the picker response is PREFER AND the
          preferred option improves total cost enough.
        - If the model skips every feasible request, trigger fallback to Step-1.
        """
        if not isinstance(raw, dict):
            return None

        request_actions = raw.get("request_actions")
        if not isinstance(request_actions, list):
            return None

        request_map = {
            str(item["request_id"]): item
            for item in payload.get("requests", [])
            if isinstance(item, dict) and isinstance(item.get("request_id"), str)
        }

        seen: set[str] = set()
        raw_decision_by_request_id: dict[str, str] = {}

        for item in request_actions:
            if not isinstance(item, dict):
                return None
            request_id = item.get("request_id")
            decision = item.get("decision")
            if not isinstance(request_id, str) or request_id not in request_map:
                return None
            if request_id in seen:
                return None
            if decision not in {"USE_FIRST", "USE_PREFERRED", "SKIP"}:
                return None
            seen.add(request_id)
            raw_decision_by_request_id[request_id] = decision

        if len(raw_decision_by_request_id) != len(request_map):
            return None

        sanitized: list[dict[str, Any]] = []
        feasible_requests = 0
        used_any_non_skip = False

        for request_id, item in request_map.items():
            picker_response = item.get("picker_response", {})
            picker_decision = picker_response.get("decision")
            first_option = item.get("first_option")
            preferred_option = item.get("preferred_option")
            requested_decision = raw_decision_by_request_id.get(request_id, "SKIP")

            first_feasible = (
                isinstance(first_option, dict)
                and first_option.get("best_picker_id") is not None
                and first_option.get("best_eta_picker") is not None
            )
            preferred_feasible = (
                isinstance(preferred_option, dict)
                and preferred_option.get("best_picker_id") is not None
                and preferred_option.get("best_eta_picker") is not None
            )

            if first_feasible or preferred_feasible:
                feasible_requests += 1

            normalized_decision = requested_decision

            if picker_decision == "DECLINE":
                normalized_decision = "SKIP"

            elif picker_decision == "ACCEPT":
                if first_feasible:
                    normalized_decision = "USE_FIRST"
                else:
                    normalized_decision = "SKIP"

            elif picker_decision == "PREFER":
                gain = None
                if first_feasible and preferred_feasible:
                    first_total = first_option.get("total_cost")
                    preferred_total = preferred_option.get("total_cost")
                    if first_total is not None and preferred_total is not None:
                        gain = int(first_total) - int(preferred_total)

                if requested_decision == "USE_PREFERRED":
                    if preferred_feasible and (
                        not first_feasible or (gain is not None and gain >= int(self.config.prefer_min_total_gain))
                    ):
                        normalized_decision = "USE_PREFERRED"
                    elif first_feasible:
                        normalized_decision = "USE_FIRST"
                    elif preferred_feasible:
                        normalized_decision = "USE_PREFERRED"
                    else:
                        normalized_decision = "SKIP"

                elif requested_decision == "USE_FIRST":
                    if first_feasible:
                        normalized_decision = "USE_FIRST"
                    elif preferred_feasible:
                        normalized_decision = "USE_PREFERRED"
                    else:
                        normalized_decision = "SKIP"

                elif requested_decision == "SKIP":
                    if not self.config.allow_skip_when_first_feasible:
                        if first_feasible:
                            normalized_decision = "USE_FIRST"
                        elif preferred_feasible:
                            normalized_decision = "USE_PREFERRED"
                        else:
                            normalized_decision = "SKIP"
                    else:
                        normalized_decision = "SKIP"

            if normalized_decision != "SKIP":
                used_any_non_skip = True

            chosen_rack_id = None
            chosen_picker_id = None

            if normalized_decision == "USE_FIRST" and first_feasible:
                chosen_rack_id = int(first_option["rack_id"])
                chosen_picker_id = int(first_option["best_picker_id"])

            elif normalized_decision == "USE_PREFERRED" and preferred_feasible:
                chosen_rack_id = int(preferred_option["rack_id"])
                chosen_picker_id = int(preferred_option["best_picker_id"])

            sanitized.append(
                {
                    "request_id": request_id,
                    "decision": normalized_decision,
                    "chosen_rack_id": chosen_rack_id,
                    "chosen_picker_id": chosen_picker_id,
                }
            )

        if feasible_requests > 0 and not used_any_non_skip:
            return None

        return sanitized

    # ------------------------------------------------------------------
    # Rule-based macro action assembly
    # ------------------------------------------------------------------

    def _assemble_final_plan_from_decisions(
        self,
        state: dict[str, Any],
        request,
        response,
        decisions: list[dict[str, Any]],
    ) -> FinalPlan:
        """Assemble final macro actions from sanitized request-level decisions."""
        agents = self._sorted_agents(state)
        agent_ids = [int(agent["id"]) for agent in agents]
        action_by_agent: dict[int, int] = {agent_id: 0 for agent_id in agent_ids}

        fixed_direct_actions = self._build_fixed_direct_actions(state, request)
        for agent_id, location_id in fixed_direct_actions.items():
            action_by_agent[int(agent_id)] = int(location_id)

        request_by_id = {item.request_id: item for item in request.requests}
        decision_by_request_id = {
            item["request_id"]: item
            for item in decisions
        }

        used_picker_ids: set[int] = set()
        used_racks: set[int] = set()

        for request_id, request_item in request_by_id.items():
            agv_id = int(request_item.agv_id)

            if agv_id in fixed_direct_actions:
                continue

            decision_item = decision_by_request_id.get(request_id)
            if decision_item is None:
                continue

            decision = decision_item["decision"]
            chosen_rack_id = decision_item.get("chosen_rack_id")
            chosen_picker_id = decision_item.get("chosen_picker_id")

            if decision == "SKIP":
                continue
            if chosen_rack_id is None or chosen_picker_id is None:
                continue

            chosen_rack_id = int(chosen_rack_id)
            chosen_picker_id = int(chosen_picker_id)

            candidate_rack_ids = {int(candidate.rack_id) for candidate in request_item.candidates}
            if chosen_rack_id not in candidate_rack_ids:
                continue
            if chosen_rack_id in used_racks:
                continue

            picker_agent = self._agent_by_id(agents, chosen_picker_id)
            if picker_agent is None or picker_agent.get("type") != "PICKER":
                continue
            if bool(picker_agent.get("busy", False)):
                continue
            if chosen_picker_id in used_picker_ids:
                continue
            if chosen_picker_id in fixed_direct_actions:
                continue

            # Ensure the chosen picker can indeed support the chosen rack.
            eta_picker = self._safe_cost(
                self._agent_cost_map(state, "picker", chosen_picker_id),
                chosen_rack_id,
            )
            if eta_picker is None:
                continue

            action_by_agent[agv_id] = chosen_rack_id
            action_by_agent[chosen_picker_id] = chosen_rack_id
            used_picker_ids.add(chosen_picker_id)
            used_racks.add(chosen_rack_id)

        return FinalPlan(
            macro_actions=[
                FinalPlanItem(agent_id=int(agent_id), location_id=int(action_by_agent[int(agent_id)]))
                for agent_id in sorted(agent_ids)
            ],
            explanation="Step-2 LLM request-level integration with rack-consistent picker assignment.",
        )

    # ------------------------------------------------------------------
    # Fixed direct actions
    # ------------------------------------------------------------------

    def _build_fixed_direct_actions(self, state: dict[str, Any], request) -> dict[int, int]:
        """Build actions that must stay fixed and should not be negotiated.

        Preserve only:
        - busy agents -> 0
        - AGVs carrying and not has_delivered -> direct goal action

        Do NOT freeze idle pickers here, because they may still be assigned to
        cooperative racks in the assembled final plan.
        """
        agents = self._sorted_agents(state)
        cooperative_agv_ids = {int(item.agv_id) for item in request.requests}
        fixed_actions: dict[int, int] = {}

        for agent_index, agent in enumerate(agents):
            agent_id = int(agent["id"])
            agent_type = agent.get("type")

            if bool(agent.get("busy", False)):
                fixed_actions[agent_id] = 0
                continue

            if agent_type == "AGV":
                carrying = bool(agent.get("carrying", False))
                has_delivered = bool(agent.get("has_delivered", False))

                if carrying and not has_delivered:
                    fixed_actions[agent_id] = self._choose_best_loc(
                        agent_index=agent_index,
                        candidates=[int(loc_id) for loc_id in state.get("goal_ids", [])],
                        cost_map=self._agent_cost_map(state, "agv", agent_id),
                        valid_action_masks=state.get("valid_action_masks", []),
                        disallowed=set(),
                    )
                    continue

                if agent_id not in cooperative_agv_ids:
                    continue

        return fixed_actions

    # ------------------------------------------------------------------
    # Picker helpers
    # ------------------------------------------------------------------

    def _best_picker_for_rack(self, state: dict[str, Any], rack_id: int) -> tuple[int | None, int | None]:
        """Return the best currently available picker for a rack."""
        agents = self._sorted_agents(state)
        best_picker_id: int | None = None
        best_eta: int | None = None

        for agent in agents:
            if agent.get("type") != "PICKER":
                continue
            if bool(agent.get("busy", False)):
                continue

            picker_id = int(agent["id"])
            eta = self._safe_cost(self._agent_cost_map(state, "picker", picker_id), int(rack_id))
            if eta is None:
                continue
            eta = int(eta)

            if best_eta is None or eta < best_eta or (eta == best_eta and picker_id < int(best_picker_id or 10**9)):
                best_picker_id = picker_id
                best_eta = eta

        return best_picker_id, best_eta


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
        "valid_action_masks": [[1] * 60 for _ in range(4)],
        "cost_table": {
            "agv": {
                "1": {"37": 5, "42": 7, "1": 8, "2": 6, "51": 7, "52": 10},
                "2": {"37": 11, "42": 7, "1": 3, "2": 4, "51": 6, "52": 8},
            },
            "picker": {
                "3": {"37": 4, "42": 14, "51": 5, "52": 9},
                "4": {"37": 6, "42": 8, "51": 7, "52": 6},
            },
        },
    }

    planner = SymbioticLLMPlannerV2(SymbioticLLMPlannerV2Config(debug=True))
    print("SymbioticLLMPlannerV2 ready. Call plan(state) with a configured local vLLM server.")
    print(isinstance(planner.last_final_plan, (type(None), FinalPlan)))
    print(final_plan_to_action_list(FinalPlan(macro_actions=[], explanation=""), 4))