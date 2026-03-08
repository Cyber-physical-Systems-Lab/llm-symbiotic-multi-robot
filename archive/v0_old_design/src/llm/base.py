# src/llm/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional, Dict, Any


@dataclass
class LLMResponse:
    text: str
    raw: Optional[Dict[str, Any]] = None


class LLMProvider(Protocol):
    """A minimal interface so we can swap OpenAI/Azure/local models later."""
    def generate(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> LLMResponse:
        ...
