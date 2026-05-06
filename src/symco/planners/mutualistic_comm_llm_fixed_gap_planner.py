from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from symco.planners.symbiotic_comm_llm_planner import (
    SymbioticCommLLMPlanner,
    SymbioticCommLLMPlannerConfig,
)


@dataclass(slots=True)
class MutualisticCommLLMFixedGapPlannerConfig(SymbioticCommLLMPlannerConfig):
    """Config for the mutualistic planner with fixed-gap triggering."""
    fixed_gap_steps: int = 8


class MutualisticCommLLMFixedGapPlanner(SymbioticCommLLMPlanner):
    """
    Mutualistic staged LLM planner with fixed-gap triggering.

    In contrast to the parent event-triggered planner, this variant only allows
    communication once every fixed_gap_steps, provided that a legal idle
    cooperative need currently exists.

    Everything else is inherited from SymbioticCommLLMPlanner:
    - Stage 1 / Stage 2 / Stage 3 logic
    - assignment registration
    - fallback behavior
    - communication accounting
    """

    def __init__(self, config: MutualisticCommLLMFixedGapPlannerConfig | None = None):
        super().__init__(config or MutualisticCommLLMFixedGapPlannerConfig())

    def _should_trigger_communication(self, state: dict[str, Any]) -> bool:
        agents = self._sorted_agents(state)
        request_racks = state.get("requests_rack_ids_topk", [])
        empty_racks = state.get("empty_rack_ids_topk", [])
        trigger_reasons: list[str] = []

        def preview_no_batch_requests_possible() -> bool:
            preview_requests, _ = self._build_batch_request_trace(state)
            return len(preview_requests) == 0

        def record_no_communication_trace(reason: str, **extra: Any) -> None:
            trace = {
                "step": int(self.step_counter),
                "reason": str(reason),
                "eligible_idle_agv_ids": sorted(int(x) for x in eligible_idle_agv_ids),
                "idle_need": bool(idle_need),
                "fixed_gap_steps": int(self.config.fixed_gap_steps),
                "steps_since_last_comm": int(self.step_counter - self.last_communication_step),
                "gap_not_reached": bool(extra.get("gap_not_reached", False)),
                "gap_remaining": (
                    int(extra["gap_remaining"])
                    if extra.get("gap_remaining") is not None
                    else None
                ),
                "budget_throttling": bool(extra.get("budget_throttling", False)),
                "no_batch_requests_possible": bool(extra.get("no_batch_requests_possible", False)),
            }
            self.last_no_communication_reason_trace = trace

        def finish(result: bool) -> bool:
            # keep compatibility with existing runner/debug fields
            self.last_available_picker_count = 0
            if result:
                self.last_no_communication_reason_trace = {}
                deduped_trigger_reasons: list[str] = []
                seen_reasons: set[str] = set()
                for reason in trigger_reasons:
                    if not isinstance(reason, str) or reason in seen_reasons:
                        continue
                    seen_reasons.add(reason)
                    deduped_trigger_reasons.append(reason)
                self.last_trigger_reasons = deduped_trigger_reasons
                for reason in self.last_trigger_reasons:
                    self.trigger_reason_counts[reason] = int(
                        self.trigger_reason_counts.get(reason, 0)
                    ) + 1
                self.trigger_reason_steps.append(
                    {
                        "step": int(self.step_counter),
                        "reasons": list(self.last_trigger_reasons),
                    }
                )
            else:
                self.last_trigger_reasons = []
            return bool(result)

        # ---------------------------------------
        # Legal idle cooperative need only
        # ---------------------------------------
        eligible_idle_agv_ids: list[int] = []
        for agent in agents:
            if agent.get("type") != "AGV":
                continue
            if bool(agent.get("busy", False)):
                continue

            target = int(agent.get("target", 0) or 0)
            if target != 0:
                continue

            carrying = bool(agent.get("carrying", False))
            has_delivered = bool(agent.get("has_delivered", False))
            needs_new_task = (
                ((not carrying) and bool(request_racks))
                or (carrying and has_delivered and bool(empty_racks))
            )
            if needs_new_task:
                eligible_idle_agv_ids.append(int(agent["id"]))

        idle_need = bool(eligible_idle_agv_ids)
        if not idle_need:
            record_no_communication_trace(
                "no_idle_need",
                no_batch_requests_possible=preview_no_batch_requests_possible(),
            )
            return finish(False)

        # ---------------------------------------
        # Fixed-gap timing gate
        # ---------------------------------------
        fixed_gap = max(1, int(self.config.fixed_gap_steps))
        steps_since_last_comm = int(self.step_counter - self.last_communication_step)

        if steps_since_last_comm < fixed_gap:
            gap_remaining = int(fixed_gap - steps_since_last_comm)
            if gap_remaining < 0:
                gap_remaining = 0
            record_no_communication_trace(
                "fixed_gap_not_reached",
                gap_not_reached=True,
                gap_remaining=gap_remaining,
                no_batch_requests_possible=preview_no_batch_requests_possible(),
            )
            return finish(False)

        # Optional: keep rolling budget accounting for consistency
        if self._is_budget_throttled():
            self.planner_throttled_by_budget = True
            record_no_communication_trace(
                "budget_throttling",
                budget_throttling=True,
                no_batch_requests_possible=preview_no_batch_requests_possible(),
            )
            return finish(False)

        trigger_reasons.append("fixed_gap_trigger")
        return finish(True)