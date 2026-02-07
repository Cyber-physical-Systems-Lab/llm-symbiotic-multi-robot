# src/agents/random_agent.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class RandomAgent:
    agent_id: int

    def decide_action(self, obs_i: Any, action_space) -> int:
        # TA-RWARE action_space is usually a gymnasium Tuple space of Discrete
        return int(action_space[self.agent_id].sample())
