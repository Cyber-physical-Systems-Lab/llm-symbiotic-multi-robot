"""Non-symbiotic LLM planner for TA-RWARE coordination (FAIR trigger-aligned).

What changed vs your original:
- Adds an event-triggered *LLM decision update* gate so we don't call LLM every step.
  This is NOT "negotiation" (non-symbiotic has none); it's just "when do we re-plan with LLM".
- Goal stage is ALWAYS rule-based (no LLM), consistent with your earlier decision.
- STRICT semantic-preserving sanitization remains unchanged:
  ✅ Allowed repairs (do NOT change the model's decision semantics)
  - Packaging repair:
      {"chosen_rack_id": 37} or {"assigned_rack": 37}
    -> wrapped into {"requests":[...]} ONLY when exactly one request exists.
  - Field alias repair:
      assigned_rack -> chosen_rack_id
      assigned_picker_id -> picker_id
  - Type coercion:
      "37" -> 37, "3" -> 3
  - Ignore unrelated fields
  - Fill defaults:
      reason="" , priority=1
  ❌ Not allowed:
  - Any decision fallback (no "nearest valid rack", no "best picker" replacement)
  - Any selection outside communicated candidates/picker_options

Also:
- Removes zone-related placeholders.
- Adds explicit decision objectives to prompts (min ETA with tie-breakers).
"""

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

    # --- NEW: trigger-aligned decision update gating ---
    # Minimum number of steps between LLM decision updates (prevents re-planning too frequently).
    min_decision_gap_steps: int = 8
    # If True, coordination_alert forces an LLM update when there is any cooperative need.
    honor_coordination_alert: bool = True


class NonSymbioticLLMPlanner:
    """Two-stage non-symbiotic LLM planner.

    Stage 1 (AGV-side LLM):
      For each cooperative AGV task, choose exactly one rack from a communicated
      candidate list. This AGV rack target is fixed (cannot be revised by picker).

    Stage 2 (Picker-side LLM):
      Given the fixed rack, respond ACK or BUSY. If ACK, choose one picker_id
      strictly from the communicated picker_options for that request.

    IMPORTANT:
    - Goal stage (carrying but not delivered) is rule-based, no LLM.
    - LLM is called only when an AGV needs a NEW cooperative decision (LOAD/UNLOAD)
      and decision-gap allows it.
    """

    def __init__(self, config: NonSymbioticLLMPlannerConfig | None = None):
        self.config = config or NonSymbioticLLMPlannerConfig()
        self.client = VLLMChatClient(load_vllm_config_from_env())

        # Debug/analysis fields (plain dicts).
        self.last_agv_requests: dict | None = None
        self.last_picker_responses: dict | None = None
        self.last_final_plan: dict | None = None

        # Compatibility fields some runners may look for.
        self.last_request: dict | None = None
        self.last_response: dict | None = None

        # --- NEW: last time we called LLM for cooperative decisions ---
        self.step_counter: int = 0
        self.last_llm_step: int = -10**9

    def plan(self, state: dict[str, Any]) -> list[int]:
        """Plan macro actions ordered by agent id."""
        self.step_counter += 1
        agents = self._sorted_agents(state)

        # Decide whether we should call LLM for cooperative decisions this step.
        trigger_llm = self._should_trigger_llm_update(state, agents)

        # -------------------------
        # If NOT triggering LLM:
        # - do NOT fabricate cooperative decisions
        # - still allow rule-based GOAL moves via _build_final_plan (carrying & not delivered)
        # -------------------------
        if not trigger_llm:
            self.last_agv_requests = {"requests": []}
            self.last_picker_responses = {"responses": []}
            self.last_request = None
            self.last_response = None

            final_plan = self._build_final_plan(
                state=state,
                agents=agents,
                agv_requests=[],
                picker_responses=[],
            )
            self.last_final_plan = final_plan.to_dict()

            num_agents = int(state.get("meta", {}).get("num_agents", len(agents)))
            return final_plan_to_action_list(final_plan, num_agents)

        # We will run an LLM decision update now.
        self.last_llm_step = self.step_counter

        # -------------------------
        # Stage 1: AGV-side choice (LLM)
        # -------------------------
        agv_payload, cooperative_requests = self._build_agv_stage_input(state, agents)

        if self.config.debug:
            print("AGV_PROMPT_PAYLOAD:")
            print(json.dumps(agv_payload, indent=2, ensure_ascii=False))

        agv_raw: dict[str, Any] = {"requests": []}
        if cooperative_requests:
            agv_raw = self.client.chat_json(
                self._agv_system_prompt(),
                self._agv_user_prompt(agv_payload),
            )

        if self.config.debug:
            print("LLM_AGV_OUTPUT_RAW:")
            print(json.dumps(agv_raw, indent=2, ensure_ascii=False))

        agv_requests = self._sanitize_agv_requests(agv_raw, cooperative_requests)

        if self.config.debug:
            print("SANITIZED_AGV_REQUESTS:")
            print(json.dumps(agv_requests, indent=2, ensure_ascii=False))

        self.last_agv_requests = {"requests": agv_requests}
        self.last_request = self.last_agv_requests  # compatibility

        # -------------------------
        # Stage 2: Picker-side ACK/BUSY (LLM, fixed rack)
        # -------------------------
        picker_payload = self._build_picker_stage_input(state, agents, agv_requests)

        if self.config.debug:
            print("PICKER_PROMPT_PAYLOAD:")
            print(json.dumps(picker_payload, indent=2, ensure_ascii=False))

        picker_raw: dict[str, Any] = {"responses": []}
        if agv_requests:
            picker_raw = self.client.chat_json(
                self._picker_system_prompt(),
                self._picker_user_prompt(picker_payload),
            )

        if self.config.debug:
            print("LLM_PICKER_OUTPUT_RAW:")
            print(json.dumps(picker_raw, indent=2, ensure_ascii=False))

        allowed_picker_ids_by_request = self._build_allowed_picker_map(picker_payload)

        picker_responses = self._sanitize_picker_responses(
            state=state,
            agents=agents,
            llm_output=picker_raw,
            agv_requests=agv_requests,
            allowed_picker_ids_by_request=allowed_picker_ids_by_request,
        )

        if self.config.debug:
            print("SANITIZED_PICKER_RESPONSES:")
            print(json.dumps(picker_responses, indent=2, ensure_ascii=False))

        self.last_picker_responses = {"responses": picker_responses}
        self.last_response = self.last_picker_responses  # compatibility

        # -------------------------
        # Final plan (rule-based, no target revision by picker)
        # -------------------------
        final_plan = self._build_final_plan(state, agents, agv_requests, picker_responses)

        if self.config.debug:
            print("FINAL_PLAN:")
            print(json.dumps(final_plan.to_dict(), indent=2, ensure_ascii=False))

        self.last_final_plan = final_plan.to_dict()

        num_agents = int(state.get("meta", {}).get("num_agents", len(agents)))
        return final_plan_to_action_list(final_plan, num_agents)

    # ============================================================
    # NEW: Trigger logic (aligned in spirit with symbiotic decision triggering)
    # ============================================================

    def _should_trigger_llm_update(self, state: dict[str, Any], agents: list[dict[str, Any]]) -> bool:
        """Return whether we should run an LLM decision update this step.

        Non-symbiotic has no negotiation; this gate only controls *LLM re-planning*.

        We trigger when:
        - there exists at least one cooperative decision need (LOAD/UNLOAD) for a non-busy AGV with target==0
        - and a minimum decision-gap has elapsed since last LLM update

        We do NOT use LLM for GOAL stage (carrying & not delivered).
        """
        # Gap control
        if (self.step_counter - self.last_llm_step) < int(self.config.min_decision_gap_steps):
            # Allow an override if an explicit alert exists AND we do have a cooperative need.
            if not (self.config.honor_coordination_alert and bool(state.get("coordination_alert", False))):
                return False

        request_racks = state.get("requests_rack_ids_topk", [])
        empty_racks = state.get("empty_rack_ids_topk", [])

        for agent in agents:
            if agent.get("type") != "AGV":
                continue
            if bool(agent.get("busy", False)):
                continue

            carrying = bool(agent.get("carrying", False))
            has_delivered = bool(agent.get("has_delivered", False))
            target = int(agent.get("target", 0) or 0)

            # Cooperative LOAD decision need (only when target==0)
            if (not carrying) and target == 0 and request_racks:
                return True

            # Cooperative UNLOAD/RETURN decision need (only when target==0)
            if carrying and has_delivered and target == 0 and empty_racks:
                return True

            # GOAL stage is rule-based (do not trigger LLM here)
            # carrying and not has_delivered -> handled in final plan without LLM.

        return False

    # ============================================================
    # Stage 1: Build AGV-side input (candidates + eta_agv)
    # ============================================================

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
            target = int(agent.get("target", 0) or 0)

            # GOAL stage is rule-based (no LLM request).
            if carrying and not has_delivered:
                continue

            # Only request an LLM choice when AGV truly needs a new cooperative decision (target==0).
            if target != 0:
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

    def _top_agv_candidates(
        self,
        state: dict[str, Any],
        agent_index: int,
        agv_id: int,
        candidate_pool: list[int],
        topk: int,
    ) -> list[dict[str, Any]]:
        """Rank AGV rack candidates by eta_agv (distance/path length)."""
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
        return [{"rack_id": rack_id, "eta_agv": eta_agv} for rack_id, eta_agv in scored[: max(0, topk)]]

    # ============================================================
    # Stage 2: Build Picker-side input (fixed rack + picker_options)
    # ============================================================

    def _build_picker_stage_input(
        self,
        state: dict[str, Any],
        agents: list[dict[str, Any]],
        agv_requests: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build picker-side input using fixed AGV-selected racks."""
        available_pickers = [
            {"picker_id": int(agent["id"]), "coords_yx": agent.get("coords_yx", [0, 0])}
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
                picker_options.append({"picker_id": picker_id, "eta_picker": int(eta_picker)})

            fixed_requests.append(
                {
                    "request_id": item["request_id"],
                    "agv_id": int(item["agv_id"]),
                    "purpose": item["purpose"],
                    "chosen_rack_id": rack_id,
                    "picker_options": picker_options,
                }
            )

        return {"meta": state.get("meta", {}), "available_pickers": available_pickers, "requests": fixed_requests}

    def _build_allowed_picker_map(self, picker_payload: dict[str, Any]) -> dict[str, set[int]]:
        """Build request_id -> allowed picker_id set from picker_payload.requests[].picker_options."""
        allowed: dict[str, set[int]] = {}
        requests = picker_payload.get("requests", [])
        if not isinstance(requests, list):
            return allowed

        for req in requests:
            if not isinstance(req, dict):
                continue
            request_id = req.get("request_id")
            if not isinstance(request_id, str):
                continue
            options = req.get("picker_options", [])
            if not isinstance(options, list):
                options = []
            ids: set[int] = set()
            for opt in options:
                if not isinstance(opt, dict):
                    continue
                pid = self._safe_int(opt.get("picker_id"))
                if pid >= 0:
                    ids.add(int(pid))
            allowed[request_id] = ids
        return allowed

    # ============================================================
    # Final plan (fixed AGV targets + optional picker assignment)
    # ============================================================

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

        responses_by_request = {
            item["request_id"]: item for item in picker_responses if isinstance(item, dict) and "request_id" in item
        }

        # Rule-based: AGVs carrying but not delivered -> go to goal.
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

        # Apply AGV rack assignments (fixed from stage 1).
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
            FinalPlanItem(agent_id=int(agent["id"]), location_id=int(action_by_agent.get(int(agent["id"]), 0)))
            for agent in agents
        ]
        return FinalPlan(
            macro_actions=macro_actions,
            explanation="Non-symbiotic LLM plan with fixed AGV targets and picker ACK/BUSY responses.",
        )

    # ============================================================
    # Strict semantic-preserving sanitization
    # ============================================================

    def _sanitize_agv_requests(
        self,
        llm_output: dict[str, Any],
        cooperative_requests: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        raw_requests = llm_output.get("requests", None)
        if not isinstance(raw_requests, list):
            raw_requests = self._coerce_single_agv_request(llm_output, cooperative_requests)
        if not isinstance(raw_requests, list):
            raw_requests = []

        raw_by_id: dict[str, dict[str, Any]] = {}
        for item in raw_requests:
            if not isinstance(item, dict):
                continue
            rid = item.get("request_id")
            if isinstance(rid, str) and rid not in raw_by_id:
                raw_by_id[rid] = item

        sanitized: list[dict[str, Any]] = []
        for request_id, original in cooperative_requests.items():
            raw = raw_by_id.get(request_id)
            if raw is None:
                continue

            chosen_rack_id = self._safe_int(raw.get("chosen_rack_id"))
            if chosen_rack_id < 0:
                chosen_rack_id = self._safe_int(raw.get("assigned_rack"))
            if chosen_rack_id < 0:
                continue

            candidate_map = {
                int(candidate.get("rack_id")): candidate
                for candidate in original.get("candidates", [])
                if isinstance(candidate, dict) and candidate.get("rack_id") is not None
            }
            if chosen_rack_id not in candidate_map:
                continue

            candidate = candidate_map[chosen_rack_id]
            eta_agv = self._safe_int(candidate.get("eta_agv"))
            if eta_agv < 0:
                continue

            sanitized.append(
                {
                    "request_id": request_id,
                    "agv_id": int(original["agv_id"]),
                    "purpose": str(original["purpose"]),
                    "chosen_rack_id": int(chosen_rack_id),
                    "eta_agv": int(eta_agv),
                    "priority": int(original.get("priority", 1)),
                    "reason": str(raw.get("reason", "")) if raw.get("reason") is not None else "",
                }
            )

        return sanitized

    def _coerce_single_agv_request(
        self,
        llm_output: dict[str, Any],
        cooperative_requests: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        if len(cooperative_requests) != 1:
            return None

        chosen_rack_id = self._safe_int(llm_output.get("chosen_rack_id"))
        if chosen_rack_id < 0:
            chosen_rack_id = self._safe_int(llm_output.get("assigned_rack"))
        if chosen_rack_id < 0:
            return None

        original = next(iter(cooperative_requests.values()))
        request_id = str(original.get("request_id", ""))
        if not request_id:
            return None

        return [
            {
                "request_id": request_id,
                "agv_id": int(original["agv_id"]),
                "purpose": str(original["purpose"]),
                "chosen_rack_id": int(chosen_rack_id),
                "priority": int(original.get("priority", 1)),
                "reason": str(llm_output.get("reason", "")) if llm_output.get("reason") is not None else "",
            }
        ]

    def _sanitize_picker_responses(
        self,
        state: dict[str, Any],
        agents: list[dict[str, Any]],
        llm_output: dict[str, Any],
        agv_requests: list[dict[str, Any]],
        allowed_picker_ids_by_request: dict[str, set[int]],
    ) -> list[dict[str, Any]]:
        raw_responses = llm_output.get("responses", None)
        if not isinstance(raw_responses, list):
            raw_responses = self._coerce_picker_responses(llm_output)
        if not isinstance(raw_responses, list):
            raw_responses = []

        valid_picker_ids = {
            int(agent["id"])
            for agent in agents
            if agent.get("type") == "PICKER" and not bool(agent.get("busy", False))
        }

        raw_by_id: dict[str, dict[str, Any]] = {}
        for item in raw_responses:
            if not isinstance(item, dict):
                continue
            rid = item.get("request_id")
            if isinstance(rid, str) and rid not in raw_by_id:
                raw_by_id[rid] = item

        requests_by_id = {item["request_id"]: item for item in agv_requests if isinstance(item, dict)}

        sanitized: list[dict[str, Any]] = []
        for request_id, request_item in requests_by_id.items():
            raw = raw_by_id.get(request_id, {})
            status = str(raw.get("status", "BUSY"))
            if status not in {"ACK", "BUSY"}:
                status = "BUSY"

            chosen_rack_id = int(request_item["chosen_rack_id"])
            reason = str(raw.get("reason", "")) if raw.get("reason") is not None else ""
            allowed_picker_ids = allowed_picker_ids_by_request.get(request_id, set())

            if status == "ACK":
                picker_id = self._safe_int(raw.get("picker_id"))
                if picker_id < 0:
                    picker_id = self._safe_int(raw.get("assigned_picker_id"))

                if picker_id in allowed_picker_ids and picker_id in valid_picker_ids:
                    eta_picker = self._safe_cost(
                        self._agent_cost_map(state, "picker", picker_id),
                        chosen_rack_id,
                    )
                    if eta_picker is not None:
                        sanitized.append(
                            {
                                "request_id": request_id,
                                "status": "ACK",
                                "picker_id": int(picker_id),
                                "eta_picker": int(eta_picker),
                                "reason": reason,
                            }
                        )
                        continue

                sanitized.append(
                    {
                        "request_id": request_id,
                        "status": "BUSY",
                        "picker_id": None,
                        "eta_picker": None,
                        "reason": reason
                        or "Invalid ACK (picker_id not in picker_options / unavailable / unreachable).",
                    }
                )
                continue

            sanitized.append(
                {
                    "request_id": request_id,
                    "status": "BUSY",
                    "picker_id": None,
                    "eta_picker": None,
                    "reason": reason,
                }
            )

        return sanitized

    def _coerce_picker_responses(self, llm_output: dict[str, Any]) -> list[dict[str, Any]] | None:
        raw_response = llm_output.get("response")
        if isinstance(raw_response, dict):
            return [raw_response]
        return None

    # ============================================================
    # Prompts with explicit objectives (min ETA + tie-breakers)
    # ============================================================

    def _agv_system_prompt(self) -> str:
        return (
            "You are the AGV-side planner.\n"
            "For EACH request, choose EXACTLY ONE rack from the provided candidates.\n"
            "Objective: minimize eta_agv. Tie-breaker: choose the smallest rack_id.\n"
            "Constraints:\n"
            "- chosen_rack_id MUST be one of the candidate rack_id values.\n"
            "- Output JSON only. No text outside JSON.\n"
            "Required format:\n"
            '{"requests":[{"request_id":"...","chosen_rack_id":37,"reason":""}]}'
        )

    def _agv_user_prompt(self, payload: dict) -> str:
        minimal_payload = {
            "requests": [
                {
                    "request_id": item.get("request_id"),
                    "agv_id": item.get("agv_id"),
                    "purpose": item.get("purpose"),
                    "priority": item.get("priority", 1),
                    "candidates": [
                        {"rack_id": c.get("rack_id"), "eta_agv": c.get("eta_agv")}
                        for c in item.get("candidates", [])
                        if isinstance(c, dict)
                    ],
                }
                for item in payload.get("requests", [])
                if isinstance(item, dict)
            ]
        }
        return json.dumps(minimal_payload, ensure_ascii=True)

    def _picker_system_prompt(self) -> str:
        return (
            "You are the picker-side allocator.\n"
            "The rack target for each request is FIXED.\n"
            "You may only respond with:\n"
            "- ACK: choose ONE picker_id from picker_options\n"
            "- BUSY: no picker can/should be assigned\n"
            "Objective (when ACK is possible): choose the picker_id with minimal eta_picker. "
            "Tie-breaker: choose the smallest picker_id.\n"
            "Constraints:\n"
            "- If status is ACK, picker_id MUST come from picker_options.\n"
            "- Output JSON only. No text outside JSON.\n"
            "Required format:\n"
            '{"responses":[{"request_id":"...","status":"ACK","picker_id":3,"reason":""}]}'
        )

    def _picker_user_prompt(self, payload: dict) -> str:
        minimal_payload = {
            "requests": [
                {
                    "request_id": item.get("request_id"),
                    "agv_id": item.get("agv_id"),
                    "purpose": item.get("purpose"),
                    "chosen_rack_id": item.get("chosen_rack_id"),
                    "picker_options": [
                        {"picker_id": opt.get("picker_id"), "eta_picker": opt.get("eta_picker")}
                        for opt in item.get("picker_options", [])
                        if isinstance(opt, dict)
                    ],
                }
                for item in payload.get("requests", [])
                if isinstance(item, dict)
            ]
        }
        return json.dumps(minimal_payload, ensure_ascii=True)

    # ============================================================
    # Helpers
    # ============================================================

    def _sorted_agents(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        return sorted(state.get("agents", []), key=lambda agent: int(agent["id"]))

    def _agent_cost_map(self, state: dict[str, Any], group: str, agent_id: int) -> dict[str, Any]:
        return state.get("cost_table", {}).get(group, {}).get(str(int(agent_id)), {})

    def _choose_best_loc(
        self,
        agent_index: int,
        candidates: list[int],
        cost_map: dict[str, Any],
        valid_action_masks: list[list[int]],
    ) -> int:
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

    def _is_valid_action(self, valid_action_masks: list[list[int]], agent_index: int, loc_id: int) -> bool:
        if agent_index < 0 or agent_index >= len(valid_action_masks):
            return False
        mask = valid_action_masks[agent_index]
        if loc_id < 0 or loc_id >= len(mask):
            return False
        return int(mask[loc_id]) == 1

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
        "valid_action_masks": [[1] * 120 for _ in range(4)],
        "cost_table": {
            "agv": {"1": {"37": 5, "42": 8, "55": 11, "1": 4, "2": 6}, "2": {"1": 3, "2": 4, "51": 5, "52": 7}},
            "picker": {"3": {"37": 4, "42": 6, "55": 10, "51": 5, "52": 6}, "4": {"37": 7, "42": 3, "55": 8, "51": 9, "52": 4}},
        },
    }

    try:
        planner = NonSymbioticLLMPlanner(NonSymbioticLLMPlannerConfig(debug=True))
        actions = planner.plan(fake_state)
        print("ACTIONS:", actions)
        print("last_request:", planner.last_request)
        print("last_response:", planner.last_response)
        print("last_final_plan:", planner.last_final_plan)
    except Exception as exc:
        print({"error": str(exc)})