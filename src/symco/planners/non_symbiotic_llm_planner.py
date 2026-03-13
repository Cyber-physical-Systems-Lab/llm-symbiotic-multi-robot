"""Non-symbiotic LLM planner for TA-RWARE coordination."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from symco.llm.vllm_client import VLLMChatClient, load_vllm_config_from_env
from symco.protocol.messages import FinalPlan, FinalPlanItem, final_plan_to_action_list


@dataclass(slots=True)
class NonSymbioticLLMPlannerConfig:
    """Configuration for the non-symbiotic LLM planner."""

    agv_topk_candidates: int = 3
    debug: bool = False


class NonSymbioticLLMPlanner:
    """Two-stage non-symbiotic LLM planner.

    Stage 1:
    The AGV-side LLM chooses one fixed rack target per cooperative AGV task.

    Stage 2:
    The picker-side LLM only returns ``ACK`` or ``BUSY`` for that same fixed
    rack. Picker-side feedback never changes the AGV target.
    """

    def __init__(self, config: NonSymbioticLLMPlannerConfig | None = None):
        self.config = config or NonSymbioticLLMPlannerConfig()
        self.client = VLLMChatClient(load_vllm_config_from_env())
        self.last_agv_requests: dict | None = None
        self.last_picker_responses: dict | None = None
        self.last_final_plan: dict | None = None

    def plan(self, state: dict[str, Any]) -> list[int]:
        """Plan macro actions ordered by agent id."""
        agents = self._sorted_agents(state)
        agv_payload, cooperative_requests = self._build_agv_stage_input(state, agents)

        print("AGV_PROMPT_PAYLOAD:")
        print(json.dumps(agv_payload, indent=2, ensure_ascii=False))
        print("COOPERATIVE_REQUESTS:")
        print(json.dumps(cooperative_requests, indent=2, ensure_ascii=False, default=str))

        agv_raw = {"requests": []}
        if cooperative_requests:
            agv_raw = self.client.chat_json(
                self._agv_system_prompt(),
                self._agv_user_prompt(agv_payload),
            )

        print("LLM_AGV_OUTPUT_RAW:")
        print(json.dumps(agv_raw, indent=2, ensure_ascii=False))

        agv_requests = self._sanitize_agv_requests(agv_raw, cooperative_requests)

        print("SANITIZED_REQUESTS:")
        print(json.dumps(agv_requests, indent=2, ensure_ascii=False))

        self.last_agv_requests = {"requests": agv_requests}

        picker_payload = self._build_picker_stage_input(state, agents, agv_requests)

        print("PICKER_PROMPT_PAYLOAD:")
        print(json.dumps(picker_payload, indent=2, ensure_ascii=False))

        picker_raw = {"responses": []}
        if agv_requests:
            picker_raw = self.client.chat_json(
                self._picker_system_prompt(),
                self._picker_user_prompt(picker_payload),
            )

        print("LLM_PICKER_OUTPUT_RAW:")
        print(json.dumps(picker_raw, indent=2, ensure_ascii=False))

        picker_responses = self._sanitize_picker_responses(state, agents, picker_raw, agv_requests)

        print("SANITIZED_RESPONSES:")
        print(json.dumps(picker_responses, indent=2, ensure_ascii=False))

        self.last_picker_responses = {"responses": picker_responses}

        final_plan = self._build_final_plan(state, agents, agv_requests, picker_responses)

        print("FINAL_PLAN:")
        print(json.dumps(final_plan.to_dict(), indent=2, ensure_ascii=False))

        self.last_final_plan = final_plan.to_dict()
        num_agents = int(state.get("meta", {}).get("num_agents", len(agents)))
        return final_plan_to_action_list(final_plan, num_agents)

    def _build_agv_stage_input(
        self,
        state: dict[str, Any],
        agents: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        """Build the AGV-side LLM payload and request map."""
        cooperative_items: list[dict[str, Any]] = []
        cooperative_requests: dict[str, dict[str, Any]] = {}

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

            candidates = self._top_agv_candidates(
                state=state,
                agent_index=agent_index,
                agv_id=agv_id,
                candidate_pool=[int(loc_id) for loc_id in candidate_pool],
                topk=self.config.agv_topk_candidates,
            )
            if not candidates:
                continue

            request_id = f"agv-{agv_id}-{purpose.lower()}"
            request_payload = {
                "request_id": request_id,
                "agv_id": agv_id,
                "purpose": purpose,
                "priority": 1,
                "candidates": candidates,
            }
            cooperative_items.append(request_payload)
            cooperative_requests[request_id] = request_payload

        payload = {
            "meta": state.get("meta", {}),
            "requests": cooperative_items,
        }
        return payload, cooperative_requests

    def _build_picker_stage_input(
        self,
        state: dict[str, Any],
        agents: list[dict[str, Any]],
        agv_requests: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build picker-side input using fixed AGV-selected racks."""
        available_pickers = [
            {
                "picker_id": int(agent["id"]),
                "coords_yx": agent.get("coords_yx", [0, 0]),
            }
            for agent in agents
            if agent.get("type") == "PICKER" and not bool(agent.get("busy", False))
        ]

        fixed_requests: list[dict[str, Any]] = []
        for item in agv_requests:
            rack_id = int(item["chosen_rack_id"])
            picker_options: list[dict[str, Any]] = []
            for picker in available_pickers:
                picker_id = int(picker["picker_id"])
                eta_picker = self._safe_cost(self._agent_cost_map(state, "picker", picker_id), rack_id)
                if eta_picker is None:
                    continue
                picker_options.append(
                    {
                        "picker_id": picker_id,
                        "eta_picker": eta_picker,
                    }
                )

            fixed_requests.append(
                {
                    "request_id": item["request_id"],
                    "agv_id": int(item["agv_id"]),
                    "purpose": item["purpose"],
                    "chosen_rack_id": rack_id,
                    "zone_id": self._rack_zone_id(state, rack_id),
                    "zone_recent_clashes": 0,
                    "zone_recent_wait": 0,
                    "picker_options": picker_options,
                }
            )

        return {
            "meta": state.get("meta", {}),
            "available_pickers": available_pickers,
            "requests": fixed_requests,
        }

    def _build_final_plan(
        self,
        state: dict[str, Any],
        agents: list[dict[str, Any]],
        agv_requests: list[dict[str, Any]],
        picker_responses: list[dict[str, Any]],
    ) -> FinalPlan:
        """Assemble the final plan without allowing picker-side target revision."""
        action_by_agent: dict[int, int] = {}
        assigned_pickers: set[int] = set()
        responses_by_request = {item["request_id"]: item for item in picker_responses}

        for agent_index, agent in enumerate(agents):
            agent_id = int(agent["id"])
            if bool(agent.get("busy", False)):
                action_by_agent[agent_id] = 0
                continue

            if agent.get("type") != "AGV":
                action_by_agent.setdefault(agent_id, 0)
                continue

            carrying = bool(agent.get("carrying", False))
            has_delivered = bool(agent.get("has_delivered", False))
            if carrying and not has_delivered:
                goal_id = self._choose_best_loc(
                    agent_index=agent_index,
                    candidates=[int(loc_id) for loc_id in state.get("goal_ids", [])],
                    cost_map=self._agent_cost_map(state, "agv", agent_id),
                    valid_action_masks=state.get("valid_action_masks", []),
                )
                action_by_agent[agent_id] = goal_id
            else:
                action_by_agent.setdefault(agent_id, 0)

        for request_item in agv_requests:
            agv_id = int(request_item["agv_id"])
            rack_id = int(request_item["chosen_rack_id"])
            action_by_agent[agv_id] = rack_id

            response_item = responses_by_request.get(request_item["request_id"])
            if response_item is None or response_item.get("status") != "ACK":
                continue

            picker_id = int(response_item["picker_id"])
            if picker_id in assigned_pickers:
                continue
            action_by_agent[picker_id] = rack_id
            assigned_pickers.add(picker_id)

        macro_actions = [
            FinalPlanItem(
                agent_id=int(agent["id"]),
                location_id=int(action_by_agent.get(int(agent["id"]), 0)),
            )
            for agent in agents
        ]
        return FinalPlan(
            macro_actions=macro_actions,
            explanation="Non-symbiotic LLM plan with fixed AGV targets and picker ACK/BUSY responses.",
        )

    def _top_agv_candidates(
        self,
        state: dict[str, Any],
        agent_index: int,
        agv_id: int,
        candidate_pool: list[int],
        topk: int,
    ) -> list[dict[str, Any]]:
        """Rank AGV rack candidates by AGV cost."""
        scored: list[tuple[int, int]] = []
        cost_map = self._agent_cost_map(state, "agv", agv_id)
        valid_masks = state.get("valid_action_masks", [])

        for rack_id in candidate_pool:
            if not self._is_valid_action(valid_masks, agent_index, rack_id):
                continue
            eta_agv = self._safe_cost(cost_map, rack_id)
            if eta_agv is None:
                continue
            scored.append((int(rack_id), int(eta_agv)))

        scored.sort(key=lambda item: (item[1], item[0]))
        return [
            {
                "rack_id": rack_id,
                "eta_agv": eta_agv,
                "zone_id": self._rack_zone_id(state, rack_id),
                "zone_recent_clashes": 0,
                "zone_recent_wait": 0,
            }
            for rack_id, eta_agv in scored[: max(0, topk)]
        ]

    def _sanitize_agv_requests(
        self,
        llm_output: dict[str, Any],
        cooperative_requests: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Sanitize AGV-side LLM outputs against communicated candidates.

        Compatibility behavior:
        - preferred full format: ``{"requests": [...]}``
        - accepted simplified single-task formats when exactly one cooperative
          request exists in the prompt context:
          ``{"chosen_rack_id": 37}`` or ``{"assigned_rack": 37}``

        Candidate validation remains strict in all cases: the selected rack
        must belong to the original communicated candidate list.
        """
        raw_requests = llm_output.get("requests", [])
        if not isinstance(raw_requests, list):
            raw_requests = self._coerce_single_agv_request(llm_output, cooperative_requests)
            if raw_requests is None:
                return []

        sanitized: list[dict[str, Any]] = []
        seen_request_ids: set[str] = set()

        for item in raw_requests:
            if not isinstance(item, dict):
                continue
            request_id = item.get("request_id")
            if not isinstance(request_id, str) or request_id in seen_request_ids:
                continue

            original = cooperative_requests.get(request_id)
            if original is None:
                continue

            chosen_rack_id = self._safe_int(item.get("chosen_rack_id"))
            candidate_map = {
                int(candidate["rack_id"]): candidate
                for candidate in original.get("candidates", [])
                if isinstance(candidate, dict)
            }
            if chosen_rack_id not in candidate_map:
                continue

            candidate = candidate_map[chosen_rack_id]
            sanitized.append(
                {
                    "request_id": request_id,
                    "agv_id": int(original["agv_id"]),
                    "purpose": original["purpose"],
                    "chosen_rack_id": chosen_rack_id,
                    "eta_agv": int(candidate["eta_agv"]),
                    "priority": int(original.get("priority", 1)),
                    "reason": str(item.get("reason", "")),
                }
            )
            seen_request_ids.add(request_id)

        return sanitized

    def _coerce_single_agv_request(
        self,
        llm_output: dict[str, Any],
        cooperative_requests: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        """Coerce simplified single-task AGV output into request-list form.

        Some models collapse a single cooperative task to a minimal payload like
        ``{"chosen_rack_id": 37}`` or ``{"assigned_rack": 37}``. This helper
        reconstructs the missing request context only when there is exactly one
        communicated cooperative request available. Multi-task simplified output
        remains ambiguous and is rejected safely.
        """
        if len(cooperative_requests) != 1:
            return None

        chosen_rack_id = self._safe_int(llm_output.get("chosen_rack_id"))
        if chosen_rack_id < 0:
            chosen_rack_id = self._safe_int(llm_output.get("assigned_rack"))
        if chosen_rack_id < 0:
            return None

        original = next(iter(cooperative_requests.values()))
        candidate_map = {
            int(candidate["rack_id"]): candidate
            for candidate in original.get("candidates", [])
            if isinstance(candidate, dict)
        }
        if chosen_rack_id not in candidate_map:
            return None

        return [
            {
                "request_id": str(original["request_id"]),
                "agv_id": int(original["agv_id"]),
                "purpose": str(original["purpose"]),
                "chosen_rack_id": chosen_rack_id,
                "eta_agv": int(candidate_map[chosen_rack_id]["eta_agv"]),
                "priority": int(original.get("priority", 1)),
                "reason": "",
            }
        ]

    def _sanitize_picker_responses(
        self,
        state: dict[str, Any],
        agents: list[dict[str, Any]],
        llm_output: dict[str, Any],
        agv_requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Sanitize picker-side LLM outputs.

        Invalid outputs fall back to ``BUSY`` so picker-side feedback never
        changes the AGV's fixed target.
        """
        raw_responses = llm_output.get("responses", [])
        if not isinstance(raw_responses, list):
            raw_responses = self._coerce_picker_responses(llm_output)
        if not isinstance(raw_responses, list):
            raw_responses = []

        valid_picker_ids = {
            int(agent["id"])
            for agent in agents
            if agent.get("type") == "PICKER" and not bool(agent.get("busy", False))
        }
        requests_by_id = {item["request_id"]: item for item in agv_requests}
        raw_by_id = {
            item.get("request_id"): item
            for item in raw_responses
            if isinstance(item, dict) and isinstance(item.get("request_id"), str)
        }

        sanitized: list[dict[str, Any]] = []
        for request_id, request_item in requests_by_id.items():
            raw = raw_by_id.get(request_id, {})
            status = str(raw.get("status", "BUSY"))
            if status not in {"ACK", "BUSY"}:
                status = "BUSY"

            chosen_rack_id = int(request_item["chosen_rack_id"])
            reason = str(raw.get("reason", ""))

            if status == "ACK":
                picker_id = self._safe_int(raw.get("picker_id"))
                if picker_id < 0:
                    picker_id = self._safe_int(raw.get("assigned_picker_id"))
                if picker_id in valid_picker_ids:
                    eta_picker = self._safe_cost(self._agent_cost_map(state, "picker", picker_id), chosen_rack_id)
                    if eta_picker is not None:
                        sanitized.append(
                            {
                                "request_id": request_id,
                                "status": "ACK",
                                "picker_id": picker_id,
                                "eta_picker": int(eta_picker),
                                "reason": reason,
                            }
                        )
                        continue
                status = "BUSY"

            sanitized.append(
                {
                    "request_id": request_id,
                    "status": "BUSY",
                    "picker_id": None,
                    "eta_picker": None,
                    "reason": reason or "Invalid or unavailable picker response.",
                }
            )

        return sanitized

    def _coerce_picker_responses(self, llm_output: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Coerce alternate picker response wrappers into ``responses`` form.

        Supported compatibility forms:
        - ``{"responses": [...]}`` handled directly by caller
        - ``{"response": {...}}`` converted to a single-element response list
        """
        raw_response = llm_output.get("response")
        if isinstance(raw_response, dict):
            return [raw_response]
        return None

    def _agv_system_prompt(self) -> str:
        """Return the AGV-side system prompt."""
        return (
            "Choose exactly one rack per request. "
            "chosen_rack_id must come from candidates. "
            "Return JSON only. No explanation outside JSON. "
            "Use exactly this format: "
            '{"requests":[{"request_id":"...","agv_id":1,"purpose":"LOAD","chosen_rack_id":37,"eta_agv":5,"priority":1,"reason":""}]}'
        )

    def _agv_user_prompt(self, payload: dict) -> str:
        """Return the AGV-side user prompt."""
        minimal_payload = {
            "requests": [
                {
                    "request_id": item.get("request_id"),
                    "agv_id": item.get("agv_id"),
                    "purpose": item.get("purpose"),
                    "priority": item.get("priority", 1),
                    "candidates": [
                        {
                            "rack_id": candidate.get("rack_id"),
                            "eta_agv": candidate.get("eta_agv"),
                        }
                        for candidate in item.get("candidates", [])
                        if isinstance(candidate, dict)
                    ],
                }
                for item in payload.get("requests", [])
                if isinstance(item, dict)
            ]
        }
        return json.dumps(minimal_payload, ensure_ascii=True)

    def _picker_system_prompt(self) -> str:
        """Return the picker-side system prompt."""
        return (
            "Rack is fixed. "
            "Only ACK or BUSY. "
            "If ACK, choose one picker_id from picker_options. "
            "Return JSON only. "
            "Use exactly this format: "
            '{"responses":[{"request_id":"...","status":"ACK","picker_id":3,"eta_picker":4,"reason":""}]}'
        )

    def _picker_user_prompt(self, payload: dict) -> str:
        """Return the picker-side user prompt."""
        minimal_payload = {
            "requests": [
                {
                    "request_id": item.get("request_id"),
                    "agv_id": item.get("agv_id"),
                    "purpose": item.get("purpose"),
                    "chosen_rack_id": item.get("chosen_rack_id"),
                    "picker_options": [
                        {
                            "picker_id": option.get("picker_id"),
                            "eta_picker": option.get("eta_picker"),
                        }
                        for option in item.get("picker_options", [])
                        if isinstance(option, dict)
                    ],
                }
                for item in payload.get("requests", [])
                if isinstance(item, dict)
            ]
        }
        return json.dumps(minimal_payload, ensure_ascii=True)

    def _rack_zone_id(self, state: dict[str, Any], rack_id: int) -> int:
        """Return a temporary integer zone id for a rack.

        This is a deterministic placeholder until a proper rack-group-based zone
        map is added to the project.
        """
        del state
        return int(rack_id) % 8

    def _sorted_agents(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        """Return agents sorted by id."""
        return sorted(state.get("agents", []), key=lambda agent: int(agent["id"]))

    def _agent_cost_map(self, state: dict[str, Any], group: str, agent_id: int) -> dict[str, Any]:
        """Fetch one agent cost map from state."""
        return state.get("cost_table", {}).get(group, {}).get(str(int(agent_id)), {})

    def _choose_best_loc(
        self,
        agent_index: int,
        candidates: list[int],
        cost_map: dict[str, Any],
        valid_action_masks: list[list[int]],
    ) -> int:
        """Choose the nearest valid location id for direct rule-based moves."""
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

    def _is_valid_action(
        self, valid_action_masks: list[list[int]], agent_index: int, loc_id: int
    ) -> bool:
        """Return whether a location id is currently valid for an agent."""
        if agent_index < 0 or agent_index >= len(valid_action_masks):
            return False
        mask = valid_action_masks[agent_index]
        if loc_id < 0 or loc_id >= len(mask):
            return False
        return int(mask[loc_id]) == 1

    def _safe_cost(self, cost_map: dict[str, Any], loc_id: int) -> int | None:
        """Read an integer cost from a cost table."""
        value = cost_map.get(str(int(loc_id)))
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _safe_int(self, value: Any) -> int:
        """Convert a value to int, returning -1 on failure."""
        try:
            return int(value)
        except (TypeError, ValueError):
            return -1


if __name__ == "__main__":
    fake_state = {
        "meta": {"num_agents": 4},
        "agents": [
            {"id": 1, "type": "AGV", "coords_yx": [0, 0], "busy": False, "carrying": False, "has_delivered": False, "target": 0},
            {"id": 2, "type": "AGV", "coords_yx": [1, 0], "busy": False, "carrying": True, "has_delivered": False, "target": 0},
            {"id": 3, "type": "PICKER", "coords_yx": [0, 1], "busy": False, "carrying": False, "has_delivered": False, "target": 0},
            {"id": 4, "type": "PICKER", "coords_yx": [1, 1], "busy": False, "carrying": False, "has_delivered": False, "target": 0},
        ],
        "goal_ids": [1, 2],
        "requests_rack_ids_topk": [37, 42, 55],
        "empty_rack_ids_topk": [51, 52],
        "valid_action_masks": [[1] * 80 for _ in range(4)],
        "cost_table": {
            "agv": {
                "1": {"37": 5, "42": 8, "55": 11, "1": 4, "2": 6},
                "2": {"1": 3, "2": 4, "51": 5, "52": 7},
            },
            "picker": {
                "3": {"37": 4, "42": 6, "55": 10, "51": 5, "52": 6},
                "4": {"37": 7, "42": 3, "55": 8, "51": 9, "52": 4},
            },
        },
    }

    try:
        planner = NonSymbioticLLMPlanner()
        actions = planner.plan(fake_state)
        print(actions)
        print(planner.last_agv_requests)
        print(planner.last_picker_responses)
        print(planner.last_final_plan)
    except Exception as exc:
        print({"error": str(exc)})
