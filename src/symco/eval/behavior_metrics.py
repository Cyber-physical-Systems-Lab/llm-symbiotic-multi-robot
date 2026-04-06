"""Behavior-level coordination metrics for TA-RWARE JSONL traces."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: str) -> list[dict]:
    """Read a JSONL file into a list of dictionaries."""
    records: list[dict] = []
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {file_path}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(
                    f"Expected object on line {line_no} in {file_path}, got {type(item).__name__}"
                )
            records.append(item)
    return records


def summarize_behavior(records: list[dict]) -> dict:
    """Compute behavior and coordination metrics from JSONL records."""
    records_by_episode = _group_records_by_episode(records)

    total_pair_switches = 0
    total_cooperative_waiting_time = 0
    total_steps_with_clashes = 0
    total_clashes = 0
    total_agv_episode_count = 0
    total_completed_deliveries = 0

    rack_counter: Counter[int] = Counter()
    episode_breakdown: dict[int, dict[str, Any]] = {}

    for episode_idx, episode_records in records_by_episode.items():
        pair_switches, agv_count = _count_pair_switches(episode_records)
        cooperative_waiting_time = _compute_cooperative_waiting_time(episode_records)
        steps_with_clashes = _count_steps_with_clashes(episode_records)
        clashes = _count_total_clashes(episode_records)
        deliveries = _count_completed_deliveries(episode_records)

        _update_rack_counter(episode_records, rack_counter)

        total_pair_switches += pair_switches
        total_agv_episode_count += agv_count
        total_cooperative_waiting_time += cooperative_waiting_time
        total_steps_with_clashes += steps_with_clashes
        total_clashes += clashes
        total_completed_deliveries += deliveries

        episode_breakdown[episode_idx] = {
            "pair_switches": pair_switches,
            "completed_deliveries": deliveries,
            "total_cooperative_waiting_time": cooperative_waiting_time,
            "avg_cooperative_waiting_time": _safe_ratio(cooperative_waiting_time, deliveries),
            "total_clashes": clashes,
            "steps_with_clashes": steps_with_clashes,
        }

    episode_count = len(records_by_episode)

    return {
        "episodes": episode_count,
        "total_pair_switches": total_pair_switches,
        "avg_pair_switches_per_episode": _safe_ratio(total_pair_switches, episode_count),
        "avg_pair_switches_per_agv_per_episode": _safe_ratio(
            total_pair_switches, total_agv_episode_count
        ),
        "total_completed_deliveries": total_completed_deliveries,
        "avg_completed_deliveries_per_episode": _safe_ratio(
            total_completed_deliveries, episode_count
        ),
        "total_cooperative_waiting_time": total_cooperative_waiting_time,
        "avg_cooperative_waiting_time": _safe_ratio(
            total_cooperative_waiting_time, total_completed_deliveries
        ),
        "total_clashes": total_clashes,
        "total_steps_with_clashes": total_steps_with_clashes,
        "most_frequent_rack_targets": [
            [loc_id, count] for loc_id, count in rack_counter.most_common(10)
        ],
        "episode_breakdown": episode_breakdown,
    }


def print_summary(summary: dict) -> None:
    """Pretty print the behavior metrics summary."""
    print("Behavior Summary")
    print(f"episodes: {summary.get('episodes', 0)}")
    print(f"total_pair_switches: {summary.get('total_pair_switches', 0)}")
    print(
        "avg_pair_switches_per_episode: "
        f"{float(summary.get('avg_pair_switches_per_episode', 0.0)):.3f}"
    )
    print(
        "avg_pair_switches_per_agv_per_episode: "
        f"{float(summary.get('avg_pair_switches_per_agv_per_episode', 0.0)):.3f}"
    )
    print(f"total_completed_deliveries: {summary.get('total_completed_deliveries', 0)}")
    print(
        "avg_completed_deliveries_per_episode: "
        f"{float(summary.get('avg_completed_deliveries_per_episode', 0.0)):.3f}"
    )
    print(
        "total_cooperative_waiting_time: "
        f"{float(summary.get('total_cooperative_waiting_time', 0.0)):.3f}"
    )
    print(
        "avg_cooperative_waiting_time: "
        f"{float(summary.get('avg_cooperative_waiting_time', 0.0)):.3f}"
    )
    print(f"total_clashes: {summary.get('total_clashes', 0)}")
    print(f"total_steps_with_clashes: {summary.get('total_steps_with_clashes', 0)}")
    print(f"most_frequent_rack_targets: {summary.get('most_frequent_rack_targets', [])}")

    episode_breakdown = summary.get("episode_breakdown", {})
    if isinstance(episode_breakdown, dict) and episode_breakdown:
        print("episode_breakdown:")
        for episode_idx in sorted(episode_breakdown):
            item = episode_breakdown[episode_idx]
            print(
                f"  ep{episode_idx}: "
                f"pair_switches={item.get('pair_switches', 0)} "
                f"completed_deliveries={item.get('completed_deliveries', 0)} "
                f"total_cooperative_waiting_time={float(item.get('total_cooperative_waiting_time', 0.0)):.3f} "
                f"avg_cooperative_waiting_time={float(item.get('avg_cooperative_waiting_time', 0.0)):.3f} "
                f"total_clashes={item.get('total_clashes', 0)} "
                f"steps_with_clashes={item.get('steps_with_clashes', 0)}"
            )


def _group_records_by_episode(records: list[dict]) -> dict[int, list[dict]]:
    """Group records by episode and sort each episode by step index."""
    grouped: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        grouped[_safe_int(record.get("episode_idx", 0))].append(record)
    for episode_idx in grouped:
        grouped[episode_idx].sort(key=lambda record: _safe_int(record.get("step_idx", 0)))
    return dict(sorted(grouped.items()))


def _extract_pairs_from_record(record: dict) -> dict[int, int]:
    """Extract AGV->picker pairings from a step's final plan."""
    comm_final_plan = record.get("comm_final_plan")
    if not isinstance(comm_final_plan, dict):
        return {}

    macro_actions = comm_final_plan.get("macro_actions", [])
    if not isinstance(macro_actions, list):
        return {}

    state_min = record.get("state_min", {})
    if not isinstance(state_min, dict):
        state_min = {}
    goal_ids = {int(loc_id) for loc_id in state_min.get("goal_ids", []) if _is_int_like(loc_id)}
    agents = state_min.get("agents", [])
    if not isinstance(agents, list):
        agents = []
    agent_types = {
        _safe_int(agent.get("id", 0)): str(agent.get("type", ""))
        for agent in agents
        if isinstance(agent, dict)
    }

    location_to_agvs: dict[int, list[int]] = defaultdict(list)
    location_to_pickers: dict[int, list[int]] = defaultdict(list)

    for item in macro_actions:
        if not isinstance(item, dict):
            continue
        agent_id = _safe_int(item.get("agent_id", 0))
        location_id = _safe_int(item.get("location_id", 0))
        if location_id == 0 or location_id in goal_ids:
            continue
        agent_type = agent_types.get(agent_id)
        if agent_type == "AGV":
            location_to_agvs[location_id].append(agent_id)
        elif agent_type == "PICKER":
            location_to_pickers[location_id].append(agent_id)

    pairs: dict[int, int] = {}
    for location_id, agv_ids in location_to_agvs.items():
        picker_ids = sorted(location_to_pickers.get(location_id, []))
        if not picker_ids:
            continue
        for index, agv_id in enumerate(sorted(agv_ids)):
            if index >= len(picker_ids):
                break
            pairs[agv_id] = picker_ids[index]
    return pairs


def _count_pair_switches(records: list[dict]) -> tuple[int, int]:
    """Count AGV-pair switches within one episode."""
    previous_pairs: dict[int, int] = {}
    agv_ids_seen: set[int] = set()
    pair_switches = 0

    for record in records:
        pairs = _extract_pairs_from_record(record)
        for agv_id, picker_id in pairs.items():
            agv_ids_seen.add(agv_id)
            previous_picker = previous_pairs.get(agv_id)
            if previous_picker is not None and previous_picker != picker_id:
                pair_switches += 1
            previous_pairs[agv_id] = picker_id

    return pair_switches, len(agv_ids_seen)


def _compute_cooperative_waiting_time(records: list[dict]) -> int:
    """
    Compute cumulative cooperative waiting time for one episode.

    Definition:
    - AGV target is a non-goal location (i.e., rack or empty-rack target)
    - AGV has already reached that target location
    - cooperative phase is still unfinished:
        * carrying == False  -> waiting for LOAD completion
        * carrying == True   -> waiting for UNLOAD completion

    This metric counts only waiting after arrival at target, not movement time.
    """
    total_wait_steps = 0

    for record in records:
        state_min = record.get("state_min", {})
        if not isinstance(state_min, dict):
            continue

        agents = state_min.get("agents", [])
        if not isinstance(agents, list):
            continue

        goal_ids = {
            int(loc_id) for loc_id in state_min.get("goal_ids", []) if _is_int_like(loc_id)
        }

        for agent in agents:
            if not isinstance(agent, dict):
                continue
            if str(agent.get("type", "")) != "AGV":
                continue

            target = _safe_int(agent.get("target", 0))
            if target == 0 or target in goal_ids:
                continue

            coords_yx = agent.get("coords_yx")
            target_coords_yx = agent.get("target_coords_yx")

            if not _same_coords(coords_yx, target_coords_yx):
                continue

            # Once the AGV has reached a non-goal target and the cooperative
            # phase is not yet completed, we count one waiting step.
            # - not carrying -> LOAD waiting
            # - carrying     -> UNLOAD waiting
            total_wait_steps += 1

    return total_wait_steps


def _count_steps_with_clashes(records: list[dict]) -> int:
    """Count episode steps where clashes were reported."""
    steps_with_clashes = 0
    for record in records:
        info = record.get("info", {})
        if not isinstance(info, dict):
            continue
        if _safe_int(info.get("clashes", 0)) > 0:
            steps_with_clashes += 1
    return steps_with_clashes


def _count_total_clashes(records: list[dict]) -> int:
    """Count total clash events across all steps in one episode."""
    total_clashes = 0
    for record in records:
        info = record.get("info", {})
        if not isinstance(info, dict):
            continue
        total_clashes += _safe_int(info.get("clashes", 0))
    return total_clashes


def _count_completed_deliveries(records: list[dict]) -> int:
    """Count completed deliveries in one episode by summing per-step shelf deliveries."""
    total_deliveries = 0
    for record in records:
        info = record.get("info", {})
        if not isinstance(info, dict):
            continue
        total_deliveries += _safe_int(info.get("shelf_deliveries", 0))
    return total_deliveries


def _update_rack_counter(records: list[dict], rack_counter: Counter[int]) -> None:
    """
    Accumulate AGV non-goal rack targets directly from state_min agents.
    This is more reliable for your trace format than relying on final plans.
    """
    for record in records:
        state_min = record.get("state_min", {})
        if not isinstance(state_min, dict):
            continue

        goal_ids = {int(loc_id) for loc_id in state_min.get("goal_ids", []) if _is_int_like(loc_id)}
        agents = state_min.get("agents", [])
        if not isinstance(agents, list):
            continue

        for agent in agents:
            if not isinstance(agent, dict):
                continue
            if str(agent.get("type", "")) != "AGV":
                continue

            target = _safe_int(agent.get("target", 0))
            if target == 0 or target in goal_ids:
                continue

            rack_counter[target] += 1


def _same_coords(a: Any, b: Any) -> bool:
    """Return whether two coordinate containers represent the same [y, x]."""
    ayx = _normalize_coords_yx(a)
    byx = _normalize_coords_yx(b)
    if ayx is None or byx is None:
        return False
    return ayx == byx


def _normalize_coords_yx(value: Any) -> tuple[int, int] | None:
    """Normalize [y, x] style coordinates."""
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        if _is_int_like(value[0]) and _is_int_like(value[1]):
            return int(value[0]), int(value[1])
    return None


def _safe_ratio(num: float, den: float) -> float:
    """Safely divide two values."""
    if den == 0:
        return 0.0
    return float(num) / float(den)


def _safe_int(value: Any) -> int:
    """Convert a value to int, falling back to zero."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _is_int_like(value: Any) -> bool:
    """Return whether a value can be interpreted as an int."""
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def main() -> int:
    """CLI entrypoint for behavior metrics."""
    parser = argparse.ArgumentParser(description="Summarize TA-RWARE behavior metrics from JSONL.")
    parser.add_argument("--jsonl", required=True, help="Path to the JSONL trace file.")
    args = parser.parse_args()

    records = load_jsonl(args.jsonl)
    summary = summarize_behavior(records)
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())