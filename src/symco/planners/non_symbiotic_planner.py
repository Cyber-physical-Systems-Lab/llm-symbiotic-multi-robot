"""Non-symbiotic baseline planner for TA-RWARE coordination."""

from __future__ import annotations

from typing import Any


class NonSymbioticPlanner:
    """Distance-first baseline planner without AGV-picker negotiation."""

    def plan(self, state: dict[str, Any]) -> list[int]:
        """Plan one macro action per agent, ordered by agent id.

        Picker assignments are based on AGV rack targets selected in this same
        planning step. The environment's current picker valid-action mask is
        derived from the previous env targets, so it can be stale for newly
        chosen AGV rack targets and would incorrectly suppress cooperation.
        """
        agents = sorted(state.get("agents", []), key=lambda agent: int(agent["id"]))
        num_agents = int(state.get("meta", {}).get("num_agents", len(agents)))
        actions = [0] * num_agents

        goal_ids = [int(loc_id) for loc_id in state.get("goal_ids", [])]
        request_ids = [int(loc_id) for loc_id in state.get("requests_rack_ids_topk", [])]
        empty_ids = [int(loc_id) for loc_id in state.get("empty_rack_ids_topk", [])]
        cost_table = state.get("cost_table", {})
        agv_costs = cost_table.get("agv", {})
        picker_costs = cost_table.get("picker", {})

        assigned_rack_targets: set[int] = set()
        agv_rack_targets: list[int] = []

        for agent_index, agent in enumerate(agents):
            agent_id = int(agent["id"])
            if agent.get("type") != "AGV":
                continue
            if bool(agent.get("busy", False)):
                actions[agent_index] = 0
                continue

            if bool(agent.get("carrying", False)) and not bool(agent.get("has_delivered", False)):
                candidates = goal_ids
                forbid_duplicates = False
            elif bool(agent.get("carrying", False)) and bool(agent.get("has_delivered", False)):
                candidates = empty_ids
                forbid_duplicates = True
            else:
                candidates = request_ids
                forbid_duplicates = True

            chosen = self._choose_best_loc(
                agent_index=agent_index,
                agent_id=agent_id,
                candidates=candidates,
                cost_map=agv_costs.get(str(agent_id), {}),
                valid_action_masks=state.get("valid_action_masks", []),
                disallowed=assigned_rack_targets if forbid_duplicates else set(),
            )
            actions[agent_index] = chosen
            if chosen != 0 and chosen not in goal_ids:
                assigned_rack_targets.add(chosen)
                agv_rack_targets.append(chosen)

        assigned_picker_targets: set[int] = set()
        assigned_picker_ids: set[int] = set()

        for rack_id in agv_rack_targets:
            picker_choice = self._choose_picker_for_rack(
                agents=agents,
                rack_id=rack_id,
                picker_costs=picker_costs,
                assigned_picker_ids=assigned_picker_ids,
                assigned_picker_targets=assigned_picker_targets,
                goal_ids=set(goal_ids),
            )
            if picker_choice is None:
                continue

            picker_index, picker_id = picker_choice
            actions[picker_index] = rack_id
            assigned_picker_ids.add(picker_id)
            assigned_picker_targets.add(rack_id)

        return actions

    def _choose_best_loc(
        self,
        agent_index: int,
        agent_id: int,
        candidates: list[int],
        cost_map: dict[str, Any],
        valid_action_masks: list[list[int]],
        disallowed: set[int],
    ) -> int:
        """Choose the nearest valid location id for one agent."""
        best_loc = 0
        best_cost: int | None = None

        for loc_id in candidates:
            loc_id = int(loc_id)
            if loc_id in disallowed:
                continue
            if not self._is_valid_action(valid_action_masks, agent_index, loc_id):
                continue

            cost = self._safe_cost(cost_map, loc_id)
            if cost is None:
                continue

            if best_cost is None or cost < best_cost or (cost == best_cost and loc_id < best_loc):
                best_loc = loc_id
                best_cost = cost

        return best_loc if best_cost is not None else 0

    def _choose_picker_for_rack(
        self,
        agents: list[dict[str, Any]],
        rack_id: int,
        picker_costs: dict[str, dict[str, Any]],
        assigned_picker_ids: set[int],
        assigned_picker_targets: set[int],
        goal_ids: set[int],
    ) -> tuple[int, int] | None:
        """Choose the nearest available picker for a projected rack target.

        We intentionally do not consult the current picker valid-action masks
        here. Those masks are computed from the env's current AGV targets,
        which lag behind the AGV rack targets selected in this planning step.
        For same-step cooperation, the projected rack targets should be treated
        as valid candidates directly.
        """
        if rack_id in assigned_picker_targets or rack_id in goal_ids:
            return None

        best_choice: tuple[int, int] | None = None
        best_cost: int | None = None

        for agent_index, agent in enumerate(agents):
            if agent.get("type") != "PICKER":
                continue

            picker_id = int(agent["id"])
            if picker_id in assigned_picker_ids:
                continue
            if bool(agent.get("busy", False)):
                continue

            cost = self._safe_cost(picker_costs.get(str(picker_id), {}), rack_id)
            if cost is None:
                continue

            if (
                best_cost is None
                or cost < best_cost
                or (cost == best_cost and picker_id < best_choice[1])
            ):
                best_choice = (agent_index, picker_id)
                best_cost = cost

        return best_choice

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
        """Read a distance from the cost table, returning None when unavailable."""
        value = cost_map.get(str(int(loc_id)))
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
