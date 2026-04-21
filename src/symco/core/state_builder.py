"""State extraction utilities for TA-RWARE coordination."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from tarware.definitions import Action, CollisionLayers


@dataclass(slots=True)
class StateConfig:
    """Configuration for compact state extraction."""

    topk_requests: int = 10
    topk_empty: int = 10
    topk_goals: int = 10
    pickers_to_agvs: bool = True
    block_conflicting_actions: bool = True
    care_for_agents_in_cost: bool = False


class StateBuilder:
    """Build a compact JSON-serializable state dict from a TA-RWARE env."""

    def __init__(self, config: StateConfig):
        self.config = config

    def build(self, env: Any) -> dict[str, Any]:
        """Extract the current environment state into a JSON-serializable dict."""
        env = getattr(env, "unwrapped", env)
        agents_sorted = sorted(env.agents, key=lambda agent: agent.id)
        agvs = [agent for agent in agents_sorted if self._agent_type_name(agent) == "AGV"]
        pickers = [agent for agent in agents_sorted if self._agent_type_name(agent) == "PICKER"]
        goals_all = self._goal_ids(env)

        request_ids_all = self._requested_rack_ids(env)
        empty_ids_all = self._rack_ids_from_indicator(env, env.get_empty_shelf_information())
        goal_ids = self._select_goal_ids(env, goals_all, agvs)

        request_ids = self._select_closest_ids(env, agvs, request_ids_all, self.config.topk_requests)
        empty_ids = self._select_closest_ids(env, agvs, empty_ids_all, self.config.topk_empty)
        empty_rack_pool_trace = {
            "total_empty_racks_in_environment": int(len(empty_ids_all)),
            "topk_limit": int(self.config.topk_empty),
            "selected_empty_racks": int(len(empty_ids)),
            "reason": (
                "no_empty_racks_exist_in_environment"
                if not empty_ids_all
                else "all_empty_racks_filtered_by_topk_construction_logic"
                if not empty_ids
                else "topk_truncated"
                if len(empty_ids) < len(empty_ids_all)
                else "available"
            ),
        }
        empty_rack_semantic_trace = None
        if not empty_ids_all:
            empty_rack_semantic_trace = self._build_empty_rack_semantic_trace(
                env=env,
                empty_ids_all=empty_ids_all,
                empty_ids_topk=empty_ids,
            )
        candidate_loc_ids = self._ordered_union(request_ids, empty_ids, goal_ids)
        cost_table = self._build_cost_table(env, agvs, pickers, candidate_loc_ids)
        valid_action_masks = self._valid_action_masks(env)

        # 添加区域映射
        rack_to_region, region_to_racks = self._build_rack_region_maps(env)

        return {
            "meta": {
                "grid_size": [int(env.grid_size[0]), int(env.grid_size[1])],
                "num_agents": int(env.num_agents),
                "num_agvs": int(env.num_agvs),
                "num_pickers": int(env.num_pickers),
                "num_goals": int(len(goals_all)),
            },
            "agents": [self._serialize_agent(env, agent) for agent in agents_sorted],
            "requests_rack_ids_all": [int(loc_id) for loc_id in request_ids_all],
            "empty_rack_ids_all": [int(loc_id) for loc_id in empty_ids_all],
            "goal_ids": [int(loc_id) for loc_id in goal_ids],
            "requests_rack_ids_topk": [int(loc_id) for loc_id in request_ids],
            "empty_rack_ids_topk": [int(loc_id) for loc_id in empty_ids],
            "empty_rack_pool_trace": empty_rack_pool_trace,
            "empty_rack_semantic_trace": empty_rack_semantic_trace,
            "valid_action_masks": valid_action_masks,
            "cost_table": cost_table,
            "rack_to_region": rack_to_region,
            "region_to_racks": region_to_racks,
            "location_coords_xy": {
                str(int(loc_id)): [int(coords[1]), int(coords[0])]
                for loc_id, coords in env.action_id_to_coords_map.items()
            },
        }

    def _invert_location_map(self, env: Any) -> dict[tuple[int, int], int]:
        coords_to_loc_id: dict[tuple[int, int], int] = {}
        for loc_id, coords in env.action_id_to_coords_map.items():
            yx = (int(coords[0]), int(coords[1]))
            if yx in coords_to_loc_id:
                raise ValueError(f"Duplicate coordinates in action_id_to_coords_map: {yx}")
            coords_to_loc_id[yx] = int(loc_id)
        if len(coords_to_loc_id) != len(env.action_id_to_coords_map):
            raise ValueError("Failed to invert action_id_to_coords_map without collisions.")
        return coords_to_loc_id

    def _goal_ids(self, env: Any) -> list[int]:
        return list(range(1, int(len(env.goals)) + 1))

    def _requested_rack_ids(self, env: Any) -> list[int]:
        """Return requested rack ids from the env indicator map.

        `request_queue` stores shelf objects, but their live coordinates may
        change once shelves are carried or moved, so those coordinates are not
        reliable for recovering the original rack location ids mid-episode.
        """
        return self._rack_ids_from_indicator(env, env.get_shelf_request_information())

    def _rack_ids_from_indicator(self, env: Any, indicator: np.ndarray) -> list[int]:
        rack_ids: list[int] = []
        num_goals = int(len(env.goals))
        for offset, value in enumerate(np.asarray(indicator).astype(int).tolist()):
            if int(value) == 1:
                rack_ids.append(num_goals + offset + 1)
        return rack_ids

    def _select_goal_ids(self, env: Any, goal_ids: list[int], agvs: list[Any]) -> list[int]:
        if len(goal_ids) <= self.config.topk_goals:
            return goal_ids

        carrying_agvs = [agent for agent in agvs if getattr(agent, "carrying_shelf", None)]
        if not carrying_agvs:
            return goal_ids[: self.config.topk_goals]

        def sort_key(goal_id: int) -> tuple[int, int]:
            goal_yx = self._loc_id_to_coords(env, goal_id)
            min_distance = min(
                self._manhattan((int(agent.y), int(agent.x)), goal_yx) for agent in carrying_agvs
            )
            return min_distance, goal_id

        return sorted(goal_ids, key=sort_key)[: self.config.topk_goals]

    def _select_closest_ids(
        self,
        env: Any,
        agvs: list[Any],
        loc_ids: list[int],
        limit: int,
    ) -> list[int]:
        loc_ids = self._dedupe_preserve_order(loc_ids)
        if limit <= 0:
            return []
        if len(loc_ids) <= limit:
            return loc_ids
        if not agvs:
            return loc_ids[:limit]

        def sort_key(loc_id: int) -> tuple[int, int]:
            target_yx = self._loc_id_to_coords(env, loc_id)
            min_distance = min(
                self._manhattan((int(agent.y), int(agent.x)), target_yx) for agent in agvs
            )
            return min_distance, loc_id

        return sorted(loc_ids, key=sort_key)[:limit]

    def _build_empty_rack_semantic_trace(
        self,
        env: Any,
        empty_ids_all: list[int],
        empty_ids_topk: list[int],
    ) -> dict[str, Any]:
        goal_set = {(int(x), int(y)) for x, y in env.goals}
        carried_shelf_ids = sorted(
            int(agent.carrying_shelf.id)
            for agent in env.agents
            if getattr(agent, "carrying_shelf", None) is not None
        )
        request_queue_ids = sorted(int(shelf.id) for shelf in getattr(env, "request_queue", []))
        all_shelf_ids = sorted(int(shelf.id) for shelf in getattr(env, "shelfs", []))
        location_evaluations: list[dict[str, Any]] = []
        env_empty_indicator = np.asarray(env.get_empty_shelf_information()).astype(int).tolist()

        for loc_id, coords in env.action_id_to_coords_map.items():
            y = int(coords[0])
            x = int(coords[1])
            loc_id = int(loc_id)
            if (x, y) in goal_set:
                continue

            shelf_layer_id = int(env.grid[CollisionLayers.SHELVES, y, x])
            carried_shelf_layer_id = int(env.grid[CollisionLayers.CARRIED_SHELVES, y, x])
            agv_layer_id = int(env.grid[CollisionLayers.AGVS, y, x])
            agv_req_action = None
            if agv_layer_id > 0:
                try:
                    agv_req_action = getattr(env.agents[agv_layer_id - 1].req_action, "name", None)
                except Exception:
                    agv_req_action = None

            env_marks_empty = False
            exclusion_reasons: list[str] = []
            if shelf_layer_id != 0:
                exclusion_reasons.append("occupied_by_shelf_layer")
            if carried_shelf_layer_id != 0:
                if agv_layer_id <= 0:
                    exclusion_reasons.append("occupied_by_carried_shelf_layer_without_agv")
                elif agv_req_action in {Action.NOOP.name, Action.TOGGLE_LOAD.name}:
                    exclusion_reasons.append("agv_on_cell_is_loading_or_noop_with_carried_shelf")
            if shelf_layer_id == 0 and (
                carried_shelf_layer_id == 0
                or (agv_layer_id > 0 and agv_req_action not in {Action.NOOP.name, Action.TOGGLE_LOAD.name})
            ):
                env_marks_empty = True
            if env_marks_empty:
                exclusion_reasons = []

            location_evaluations.append(
                {
                    "location_id": int(loc_id),
                    "coords_yx": [int(y), int(x)],
                    "shelf_layer_id": int(shelf_layer_id),
                    "carried_shelf_layer_id": int(carried_shelf_layer_id),
                    "agv_layer_id": int(agv_layer_id),
                    "agv_req_action": agv_req_action,
                    "environment_marks_empty": bool(env_marks_empty),
                    "excluded_reasons": exclusion_reasons,
                }
            )

        return {
            "all_shelf_ids_known_to_environment": all_shelf_ids,
            "request_queue_shelf_ids": request_queue_ids,
            "carried_shelf_ids": carried_shelf_ids,
            "goal_coords_xy": [[int(x), int(y)] for x, y in sorted(goal_set)],
            "empty_rack_ids_all": [int(x) for x in empty_ids_all],
            "empty_rack_ids_topk": [int(x) for x in empty_ids_topk],
            "raw_empty_indicator": env_empty_indicator,
            "location_evaluations": location_evaluations,
        }

    def _build_cost_table(
        self,
        env: Any,
        agvs: list[Any],
        pickers: list[Any],
        candidate_loc_ids: list[int],
    ) -> dict[str, dict[str, dict[str, int]]]:
        return {
            "agv": self._cost_table_for_agents(env, agvs, candidate_loc_ids),
            "picker": self._cost_table_for_agents(env, pickers, candidate_loc_ids),
        }

    def _cost_table_for_agents(
        self, env: Any, agents: list[Any], candidate_loc_ids: list[int]
    ) -> dict[str, dict[str, int]]:
        table: dict[str, dict[str, int]] = {}
        for agent in agents:
            start = (int(agent.y), int(agent.x))
            agent_costs: dict[str, int] = {}
            for loc_id in candidate_loc_ids:
                goal = self._loc_id_to_coords(env, loc_id)
                path = env.find_path(
                    start,
                    goal,
                    agent,
                    care_for_agents=self.config.care_for_agents_in_cost,
                )
                if path:
                    agent_costs[str(int(loc_id))] = int(len(path))
                elif start == goal:
                    agent_costs[str(int(loc_id))] = 0
            table[str(int(agent.id))] = agent_costs
        return table

    def _valid_action_masks(self, env: Any) -> list[list[int]]:
        mask = env.compute_valid_action_masks(
            pickers_to_agvs=self.config.pickers_to_agvs,
            block_conflicting_actions=self.config.block_conflicting_actions,
        )
        return [[int(value) for value in row] for row in np.asarray(mask).astype(int).tolist()]

    def _serialize_agent(self, env: Any, agent: Any) -> dict[str, Any]:
        carrying_shelf = getattr(agent, "carrying_shelf", None)
        target = getattr(agent, "target", 0) or 0
        target_coords_yx = None
        if int(target) != 0:
            coords = env.action_id_to_coords_map.get(int(target))
            if coords is not None:
                target_coords_yx = [int(coords[0]), int(coords[1])]
        return {
            "id": int(agent.id),
            "type": self._agent_type_name(agent),
            "coords_yx": [int(agent.y), int(agent.x)],
            "busy": bool(agent.busy),
            "carrying": bool(carrying_shelf),
            "has_delivered": bool(getattr(agent, "has_delivered", False)),
            "target": int(target),
            "target_coords_yx": target_coords_yx,
        }

    def _agent_type_name(self, agent: Any) -> str:
        raw_type = getattr(agent, "type", None)
        name = getattr(raw_type, "name", str(raw_type))
        if name in {"AGV", "PICKER", "AGENT"}:
            return name
        return "AGENT"

    def _loc_id_to_coords(self, env: Any, loc_id: int) -> tuple[int, int]:
        if loc_id not in env.action_id_to_coords_map:
            raise KeyError(f"Unknown location_id {loc_id}")
        coords = env.action_id_to_coords_map[loc_id]
        return int(coords[0]), int(coords[1])

    def _ordered_union(self, *groups: list[int]) -> list[int]:
        ordered: list[int] = []
        seen: set[int] = set()
        for group in groups:
            for loc_id in group:
                if loc_id not in seen:
                    seen.add(loc_id)
                    ordered.append(int(loc_id))
        return ordered

    def _dedupe_preserve_order(self, values: list[int]) -> list[int]:
        seen: set[int] = set()
        ordered: list[int] = []
        for value in values:
            if value not in seen:
                seen.add(value)
                ordered.append(int(value))
        return ordered

    def _manhattan(self, start: tuple[int, int], goal: tuple[int, int]) -> int:
        return abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    
    # 在 StateBuilder 类中添加一个辅助方法
    def _build_rack_region_maps(self, env: Any) -> tuple[dict[int, int], dict[int, list[int]]]:
        """从环境获取 rack_groups，返回 rack_to_region 和 region_to_racks 映射。"""
        rack_groups = getattr(env.unwrapped, "rack_groups", None)
        if rack_groups is None:
            return {}, {}

        # 构建坐标到 action_id 的映射（坐标顺序 (x, y) -> action_id）
        coord_to_action_id: dict[tuple[int, int], int] = {}
        for action_id, (y, x) in env.action_id_to_coords_map.items():
            coord_to_action_id[(x, y)] = action_id

        rack_to_region: dict[int, int] = {}
        region_to_racks: dict[int, list[int]] = {}
        for region_idx, group in enumerate(rack_groups):
            region_to_racks[region_idx] = []
            for (x, y) in group:
                action_id = coord_to_action_id.get((x, y))
                if action_id is not None:
                    rack_to_region[action_id] = region_idx
                    region_to_racks[region_idx].append(action_id)
        return rack_to_region, region_to_racks


def _smoke_test() -> None:
    import gymnasium as gym
    import tarware

    env = gym.make("tarware-small-2agvs-2pickers-partialobs-v1")
    env.reset(seed=0)
    sb = StateBuilder(StateConfig())
    state = sb.build(env)
    print(state["meta"], len(state["agents"]))


if __name__ == "__main__":
    _smoke_test()
