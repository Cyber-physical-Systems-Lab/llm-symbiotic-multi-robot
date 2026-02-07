# src/agents/llm_agent.py
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, Optional

from ..llm.base import LLMProvider


def _safe_parse_action_id(text: str) -> Optional[int]:
    """
    Accept only JSON like {"action_id": 123}.
    If parsing fails, return None.
    """
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "action_id" in obj:
            val = obj["action_id"]
            if isinstance(val, int):
                return val
            # allow numpy int types etc.
            try:
                return int(val)
            except Exception:
                return None
        return None
    except Exception:
        return None


@dataclass
class IndependentLLMAgent:
    agent_id: int
    llm: LLMProvider

    def _system_prompt(self) -> str:
        return (
            "You are a warehouse robot agent in a multi-agent simulation.\n"
            "At each step you must choose exactly one discrete action.\n"
            "Output must be STRICT JSON only, with the format: {\"action_id\": <int>}.\n"
            "Do not include any extra keys, text, or explanations."
        )

    def _user_prompt(self, obs_i: Any, action_space_n: int) -> str:
        # Keep it minimal for baseline; later you can add observation-to-text mapping.
        # For now, just provide the raw observation shape/type and constraints.
        return (
            f"Your agent_id = {self.agent_id}.\n"
            f"Valid action_id range: 0 to {action_space_n - 1}.\n"
            f"Observation (raw): {obs_i}\n"
            "Choose an action_id that helps maximize deliveries over time."
        )

    def decide_action(self, obs_i: Any, action_space, fallback_random) -> int:
        n = int(action_space[self.agent_id].n)
        system = self._system_prompt()
        user = self._user_prompt(obs_i, n)

        resp = self.llm.generate(system, user, temperature=0.0)
        action_id = _safe_parse_action_id(resp.text)

        # Fallback rules
        if action_id is None or action_id < 0 or action_id >= n:
            return int(fallback_random())
        return int(action_id)
