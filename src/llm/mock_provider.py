# src/llm/mock_provider.py
from __future__ import annotations
import json
import random
from .base import LLMProvider, LLMResponse


class MockLLMProvider(LLMProvider):
    """
    A placeholder LLM provider.
    Returns a random valid JSON action.
    Useful for testing the full pipeline before choosing a real LLM.
    """

    def __init__(self, action_space_n: int, seed: int = 0):
        self.rng = random.Random(seed)
        self.action_space_n = action_space_n

    def generate(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> LLMResponse:
        action_id = self.rng.randrange(self.action_space_n)
        return LLMResponse(text=json.dumps({"action_id": action_id}))
