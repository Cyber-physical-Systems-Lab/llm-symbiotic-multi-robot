"""LLM-backed Step-3 symbiotic planner for TA-RWARE coordination."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from symco.planners.symbiotic_llm_planner_v2 import (
    SymbioticLLMPlannerV2,
    SymbioticLLMPlannerV2Config,
)
from symco.protocol.messages import AGVRequest, AGVRequestItem


@dataclass(slots=True)
class SymbioticLLMPlannerV3Config(SymbioticLLMPlannerV2Config):
    """Configuration for the Step-3 symbiotic LLM planner."""

    use_llm_agv_request: bool = True
    agv_request_max_candidates: int = 3
    allow_skip_all_requests: bool = False


class SymbioticLLMPlannerV3(SymbioticLLMPlannerV2):
    """Step-3 symbiotic planner with LLM AGV packaging, picker response, and final integration."""

    def __init__(self, config: SymbioticLLMPlannerV3Config | None = None):
        resolved = config or SymbioticLLMPlannerV3Config()
        super().__init__(resolved)
        self.config = resolved
        self.last_agv_prompt_payload: dict | None = None
        self.last_agv_llm_raw: dict | None = None
        self.last_agv_used_fallback: bool = False

    def _build_agv_request(self, state: dict[str, Any]) -> AGVRequest:
        """Build AGV requests via LLM packaging over grounded rule-generated requests."""
        if not self.config.use_llm_agv_request:
            self.last_agv_used_fallback = False
            raw_request = super()._build_agv_request(state)
            if self.config.debug:
                print("AGV_USED_FALLBACK")
                print(False)
                print("SANITIZED_AGV_REQUEST")
                print(json.dumps(raw_request.to_dict(), indent=2, ensure_ascii=False))
            return raw_request

        raw_request = super()._build_agv_request(state)
        if not raw_request.requests:
            self.last_agv_used_fallback = False
            self.last_agv_prompt_payload = {"requests": []}
            self.last_agv_llm_raw = {"requests": []}
            if self.config.debug:
                print("AGV_PROMPT_PAYLOAD")
                print(json.dumps(self.last_agv_prompt_payload, indent=2, ensure_ascii=False))
                print("LLM_AGV_OUTPUT_RAW")
                print(json.dumps(self.last_agv_llm_raw, indent=2, ensure_ascii=False))
                print("AGV_USED_FALLBACK")
                print(False)
                print("SANITIZED_AGV_REQUEST")
                print(json.dumps(raw_request.to_dict(), indent=2, ensure_ascii=False))
            return raw_request

        payload = self._build_agv_prompt_payload(state, raw_request)
        self.last_agv_prompt_payload = payload

        if self.config.debug:
            print("AGV_PROMPT_PAYLOAD")
            print(json.dumps(payload, indent=2, ensure_ascii=False))

        try:
            raw_llm = self.client.chat_json(
                self._agv_system_prompt(),
                self._agv_user_prompt(payload),
            )
            self.last_agv_llm_raw = raw_llm

            if self.config.debug:
                print("LLM_AGV_OUTPUT_RAW")
                print(json.dumps(raw_llm, indent=2, ensure_ascii=False))

            sanitized = self._sanitize_agv_request_output(raw_request, raw_llm)
            if sanitized is not None:
                self.last_agv_used_fallback = False
                if self.config.debug:
                    print("AGV_USED_FALLBACK")
                    print(False)
                    print("SANITIZED_AGV_REQUEST")
                    print(json.dumps(sanitized.to_dict(), indent=2, ensure_ascii=False))
                return sanitized
        except Exception as exc:
            self.last_agv_llm_raw = {"error": str(exc)}
            if self.config.debug:
                print("LLM_AGV_OUTPUT_RAW")
                print(json.dumps(self.last_agv_llm_raw, indent=2, ensure_ascii=False))

        self.last_agv_used_fallback = True
        if self.config.debug:
            print("AGV_USED_FALLBACK")
            print(True)
            print("SANITIZED_AGV_REQUEST")
            print(json.dumps(raw_request.to_dict(), indent=2, ensure_ascii=False))
        return raw_request

    def _build_agv_prompt_payload(self, state: dict[str, Any], raw_request: AGVRequest) -> dict:
        """Build a compact AGV packaging payload from the grounded raw request."""
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

        return {
            "requests": [
                {
                    "request_id": item.request_id,
                    "agv_id": int(item.agv_id),
                    "purpose": item.purpose,
                    "priority": int(item.priority),
                    "candidates": [
                        {
                            "rack_id": int(candidate.rack_id),
                            "eta_agv": int(candidate.eta_agv),
                        }
                        for candidate in item.candidates
                    ],
                }
                for item in raw_request.requests
            ],
            "system_pressure": {
                "idle_agvs": int(idle_agvs),
                "idle_pickers": int(idle_pickers),
                "active_cooperative_assignments": int(len(self.active_assignments)),
            },
        }

    def _agv_system_prompt(self) -> str:
        """Return a short strict AGV packaging prompt."""
        return (
            "Do not invent new rack ids. Use only candidate racks from the input. "
            "For each raw request choose EMIT or SKIP. "
            "If EMIT, keep at least 1 and at most the allowed number of communicated candidates. "
            "You may reorder kept candidates, but only using communicated candidates. "
            "Return JSON only in exactly this format: "
            '{"requests":[{"request_id":"agv-1-load","decision":"EMIT","agv_id":1,"purpose":"LOAD","priority":1,"candidates":[{"rack_id":42,"eta_agv":8},{"rack_id":37,"eta_agv":5}],"reason":""}]}'
        )

    def _agv_user_prompt(self, payload: dict[str, Any]) -> str:
        """Return the compact AGV packaging payload."""
        return json.dumps(payload, ensure_ascii=True)

    def _sanitize_agv_request_output(
        self,
        raw_request: AGVRequest,
        raw_llm: dict,
    ) -> AGVRequest | None:
        """Sanitize AGV packaging output. Return None on any invalid structure."""
        if not isinstance(raw_llm, dict):
            return None

        raw_items = raw_llm.get("requests")
        if not isinstance(raw_items, list):
            return None

        original_by_id = {item.request_id: item for item in raw_request.requests}
        seen_request_ids: set[str] = set()
        emitted_items: list[AGVRequestItem] = []

        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                return None

            request_id = raw_item.get("request_id")
            if not isinstance(request_id, str):
                return None
            if request_id not in original_by_id:
                return None
            if request_id in seen_request_ids:
                return None
            seen_request_ids.add(request_id)

            decision = raw_item.get("decision")
            if decision not in {"EMIT", "SKIP"}:
                return None

            if decision == "SKIP":
                continue

            raw_candidates = raw_item.get("candidates")
            if not isinstance(raw_candidates, list) or not raw_candidates:
                return None

            original_item = original_by_id[request_id]
            original_candidate_by_rack_id = {
                int(candidate.rack_id): candidate for candidate in original_item.candidates
            }

            kept_candidates = []
            seen_rack_ids: set[int] = set()
            for raw_candidate in raw_candidates:
                if not isinstance(raw_candidate, dict):
                    return None
                rack_id = raw_candidate.get("rack_id")
                try:
                    rack_id_int = int(rack_id)
                except (TypeError, ValueError):
                    return None
                if rack_id_int not in original_candidate_by_rack_id:
                    return None
                if rack_id_int in seen_rack_ids:
                    continue
                seen_rack_ids.add(rack_id_int)
                kept_candidates.append(original_candidate_by_rack_id[rack_id_int])
                if len(kept_candidates) >= int(self.config.agv_request_max_candidates):
                    break

            if not kept_candidates:
                return None

            emitted_items.append(
                AGVRequestItem(
                    request_id=original_item.request_id,
                    agv_id=int(original_item.agv_id),
                    purpose=original_item.purpose,
                    candidates=kept_candidates,
                    priority=int(original_item.priority),
                )
            )

        if raw_request.requests and not emitted_items and not self.config.allow_skip_all_requests:
            return None

        return AGVRequest(
            requests=emitted_items,
            notes=["Step-3 LLM AGV packaging sanitized against grounded raw requests."],
        )


if __name__ == "__main__":
    planner = SymbioticLLMPlannerV3(SymbioticLLMPlannerV3Config(debug=True))
    print("SymbioticLLMPlannerV3 ready. Call plan(state) with a configured local vLLM server.")
    print(planner.last_agv_prompt_payload)
