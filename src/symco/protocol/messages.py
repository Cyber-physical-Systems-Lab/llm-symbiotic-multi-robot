"""Structured message schemas for symbiotic TA-RWARE coordination."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar


def validate_non_negative_int(name: str, value: Any) -> int:
    """Validate and return a non-negative integer."""
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def validate_decision(value: Any, allowed: set[str]) -> str:
    """Validate a decision string against an allowed set."""
    if not isinstance(value, str):
        raise TypeError(f"decision must be a str, got {type(value).__name__}")
    if value not in allowed:
        raise ValueError(f"decision must be one of {sorted(allowed)}, got {value!r}")
    return value


@dataclass(slots=True)
class CandidateRack:
    """One candidate rack proposed by the AGV side."""

    rack_id: int
    eta_agv: int

    def __post_init__(self) -> None:
        self.rack_id = validate_non_negative_int("rack_id", self.rack_id)
        self.eta_agv = validate_non_negative_int("eta_agv", self.eta_agv)

    def to_dict(self) -> dict[str, Any]:
        """Convert this item to a JSON-serializable dict."""
        return {
            "rack_id": self.rack_id,
            "eta_agv": self.eta_agv,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateRack:
        """Create an item from a dict."""
        return cls(
            rack_id=int(data["rack_id"]),
            eta_agv=int(data["eta_agv"]),
        )


@dataclass(slots=True)
class AGVRequestItem:
    """One AGV-side cooperation request with an explicit candidate set."""

    request_id: str
    agv_id: int
    purpose: str
    candidates: list[CandidateRack]
    priority: int = 1

    _ALLOWED_PURPOSES: ClassVar[set[str]] = {"LOAD", "UNLOAD"}

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str):
            raise TypeError("request_id must be a str")
        self.agv_id = validate_non_negative_int("agv_id", self.agv_id)
        self.priority = validate_non_negative_int("priority", self.priority)
        if not isinstance(self.purpose, str):
            raise TypeError("purpose must be a str")
        if self.purpose not in self._ALLOWED_PURPOSES:
            raise ValueError(
                f"purpose must be one of {sorted(self._ALLOWED_PURPOSES)}, got {self.purpose!r}"
            )
        self.candidates = [
            item if isinstance(item, CandidateRack) else CandidateRack.from_dict(item)
            for item in self.candidates
        ]
        if not self.candidates:
            raise ValueError("candidates must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        """Convert this item to a JSON-serializable dict."""
        return {
            "request_id": self.request_id,
            "agv_id": self.agv_id,
            "purpose": self.purpose,
            "candidates": [item.to_dict() for item in self.candidates],
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AGVRequestItem:
        """Create an item from a dict."""
        return cls(
            request_id=str(data["request_id"]),
            agv_id=int(data["agv_id"]),
            purpose=str(data["purpose"]),
            candidates=[CandidateRack.from_dict(item) for item in data.get("candidates", [])],
            priority=int(data.get("priority", 1)),
        )


@dataclass(slots=True)
class AGVRequest:
    """Batch of AGV cooperation requests."""

    requests: list[AGVRequestItem] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.requests = [
            item if isinstance(item, AGVRequestItem) else AGVRequestItem.from_dict(item)
            for item in self.requests
        ]
        self.notes = [str(note) for note in self.notes]

    def to_dict(self) -> dict[str, Any]:
        """Convert this message to a JSON-serializable dict."""
        return {
            "requests": [item.to_dict() for item in self.requests],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AGVRequest:
        """Create a message from a dict."""
        return cls(
            requests=[AGVRequestItem.from_dict(item) for item in data.get("requests", [])],
            notes=[str(note) for note in data.get("notes", [])],
        )


@dataclass(slots=True)
class PickerResponseItem:
    """One picker-side response to an AGV request."""

    request_id: str
    decision: str
    picker_id: int | None = None
    eta_picker: int | None = None
    chosen_rack_id: int | None = None
    reason: str = ""

    _ALLOWED_DECISIONS: ClassVar[set[str]] = {"ACCEPT", "DECLINE", "PREFER"}

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str):
            raise TypeError("request_id must be a str")
        self.decision = validate_decision(self.decision, self._ALLOWED_DECISIONS)
        if self.picker_id is not None:
            self.picker_id = validate_non_negative_int("picker_id", self.picker_id)
        if self.eta_picker is not None:
            self.eta_picker = validate_non_negative_int("eta_picker", self.eta_picker)
        if self.chosen_rack_id is not None:
            self.chosen_rack_id = validate_non_negative_int("chosen_rack_id", self.chosen_rack_id)
        if not isinstance(self.reason, str):
            raise TypeError("reason must be a str")

        if self.decision in {"ACCEPT", "PREFER"}:
            if self.picker_id is None:
                raise ValueError(f"picker_id must not be None when decision == {self.decision!r}")
            if self.chosen_rack_id is None:
                raise ValueError(f"chosen_rack_id must not be None when decision == {self.decision!r}")

    def to_dict(self) -> dict[str, Any]:
        """Convert this item to a JSON-serializable dict."""
        return {
            "request_id": self.request_id,
            "decision": self.decision,
            "picker_id": self.picker_id,
            "eta_picker": self.eta_picker,
            "chosen_rack_id": self.chosen_rack_id,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PickerResponseItem:
        """Create an item from a dict."""
        return cls(
            request_id=str(data["request_id"]),
            decision=str(data["decision"]),
            picker_id=None if data.get("picker_id") is None else int(data["picker_id"]),
            eta_picker=None if data.get("eta_picker") is None else int(data["eta_picker"]),
            chosen_rack_id=None if data.get("chosen_rack_id") is None else int(data["chosen_rack_id"]),
            reason=str(data.get("reason", "")),
        )


@dataclass(slots=True)
class PickerResponse:
    """Batch of picker responses."""

    responses: list[PickerResponseItem] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.responses = [
            item if isinstance(item, PickerResponseItem) else PickerResponseItem.from_dict(item)
            for item in self.responses
        ]
        self.notes = [str(note) for note in self.notes]

    def to_dict(self) -> dict[str, Any]:
        """Convert this message to a JSON-serializable dict."""
        return {
            "responses": [item.to_dict() for item in self.responses],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PickerResponse:
        """Create a message from a dict."""
        return cls(
            responses=[PickerResponseItem.from_dict(item) for item in data.get("responses", [])],
            notes=[str(note) for note in data.get("notes", [])],
        )


@dataclass(slots=True)
class FinalPlanItem:
    """One macro action assignment in the final plan."""

    agent_id: int
    location_id: int

    def __post_init__(self) -> None:
        self.agent_id = validate_non_negative_int("agent_id", self.agent_id)
        self.location_id = validate_non_negative_int("location_id", self.location_id)

    def to_dict(self) -> dict[str, Any]:
        """Convert this item to a JSON-serializable dict."""
        return {
            "agent_id": self.agent_id,
            "location_id": self.location_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FinalPlanItem:
        """Create an item from a dict."""
        return cls(
            agent_id=int(data["agent_id"]),
            location_id=int(data["location_id"]),
        )


@dataclass(slots=True)
class FinalPlan:
    """Final macro-action plan for all agents."""

    macro_actions: list[FinalPlanItem] = field(default_factory=list)
    explanation: str = ""

    def __post_init__(self) -> None:
        self.macro_actions = [
            item if isinstance(item, FinalPlanItem) else FinalPlanItem.from_dict(item)
            for item in self.macro_actions
        ]
        if not isinstance(self.explanation, str):
            raise TypeError("explanation must be a str")

    def to_dict(self) -> dict[str, Any]:
        """Convert this message to a JSON-serializable dict."""
        return {
            "macro_actions": [item.to_dict() for item in self.macro_actions],
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FinalPlan:
        """Create a message from a dict."""
        return cls(
            macro_actions=[FinalPlanItem.from_dict(item) for item in data.get("macro_actions", [])],
            explanation=str(data.get("explanation", "")),
        )


def final_plan_to_action_list(plan: FinalPlan, num_agents: int) -> list[int]:
    """Convert a final plan to an action list ordered by agent id."""
    num_agents = validate_non_negative_int("num_agents", num_agents)
    actions = [0] * num_agents
    seen_agent_ids: set[int] = set()

    for item in plan.macro_actions:
        agent_id = item.agent_id
        if agent_id in seen_agent_ids:
            raise ValueError(f"Duplicate agent_id in final plan: {agent_id}")
        seen_agent_ids.add(agent_id)
        if 1 <= agent_id <= num_agents:
            actions[agent_id - 1] = item.location_id

    return actions


if __name__ == "__main__":
    agv_request = AGVRequest(
        requests=[
            AGVRequestItem(
                request_id="req-1",
                agv_id=1,
                purpose="LOAD",
                candidates=[
                    CandidateRack(rack_id=37, eta_agv=5),
                    CandidateRack(rack_id=42, eta_agv=7),
                ],
                priority=1,
            )
        ],
        notes=["Need picker support for one of the communicated candidates."],
    )
    picker_response = PickerResponse(
        responses=[
            PickerResponseItem(
                request_id="req-1",
                decision="PREFER",
                picker_id=3,
                eta_picker=4,
                chosen_rack_id=42,
                reason="Rack 42 is easier for the picker side to reach.",
            )
        ],
        notes=["Preference returned from picker side."],
    )
    final_plan = FinalPlan(
        macro_actions=[
            FinalPlanItem(agent_id=1, location_id=42),
            FinalPlanItem(agent_id=3, location_id=42),
        ],
        explanation="AGV 1 and picker 3 cooperate at rack 42.",
    )

    agv_request_roundtrip = AGVRequest.from_dict(agv_request.to_dict())
    picker_response_roundtrip = PickerResponse.from_dict(picker_response.to_dict())
    final_plan_roundtrip = FinalPlan.from_dict(final_plan.to_dict())

    print(agv_request_roundtrip.to_dict())
    print(picker_response_roundtrip.to_dict())
    print(final_plan_to_action_list(final_plan_roundtrip, num_agents=4))
