"""Trace loading and diagnostics utilities for TA-RWARE JSONL logs."""

from __future__ import annotations

import argparse
import json
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
                raise ValueError(f"Expected object on line {line_no} in {file_path}, got {type(item).__name__}")
            records.append(item)
    return records


def summarize_trace(records: list[dict]) -> dict:
    """Compute aggregate diagnostics for a TA-RWARE run trace."""
    episodes_seen: set[int] = set()
    total_steps = len(records)
    total_deliveries = 0
    total_clashes = 0
    total_stucks = 0
    total_reward = 0.0

    agv_entries = 0
    agv_carrying_entries = 0
    picker_entries = 0
    picker_busy_entries = 0
    agv_load_events = 0
    agv_deliver_events_proxy = 0

    prev_agv_state: dict[tuple[int, int], dict[str, bool]] = {}

    sorted_records = sorted(
        records,
        key=lambda record: (
            int(record.get("episode_idx", 0)),
            int(record.get("step_idx", 0)),
        ),
    )

    for record in sorted_records:
        episode_idx = int(record.get("episode_idx", 0))
        episodes_seen.add(episode_idx)

        info = record.get("info", {})
        if not isinstance(info, dict):
            info = {}

        total_deliveries += _safe_int(info.get("shelf_deliveries", 0))
        total_clashes += _safe_int(info.get("clashes", 0))
        total_stucks += _safe_int(info.get("stucks", 0))
        total_reward += _safe_float(record.get("reward_sum", 0.0))

        state_min = record.get("state_min", {})
        if not isinstance(state_min, dict):
            state_min = {}
        agents = state_min.get("agents", [])
        if not isinstance(agents, list):
            agents = []

        current_episode_agvs: set[int] = set()

        for agent in agents:
            if not isinstance(agent, dict):
                continue

            agent_id = _safe_int(agent.get("id", 0))
            agent_type = str(agent.get("type", ""))
            carrying = bool(agent.get("carrying", False))
            has_delivered = bool(agent.get("has_delivered", False))
            busy = bool(agent.get("busy", False))

            if agent_type == "AGV":
                agv_entries += 1
                agv_carrying_entries += int(carrying)
                key = (episode_idx, agent_id)
                previous = prev_agv_state.get(key)
                if previous is not None:
                    if not previous["carrying"] and carrying:
                        agv_load_events += 1
                    if not previous["has_delivered"] and has_delivered:
                        agv_deliver_events_proxy += 1
                prev_agv_state[key] = {
                    "carrying": carrying,
                    "has_delivered": has_delivered,
                }
                current_episode_agvs.add(agent_id)

            elif agent_type == "PICKER":
                picker_entries += 1
                picker_busy_entries += int(busy)

        if current_episode_agvs:
            stale_keys = [
                key for key in prev_agv_state
                if key[0] == episode_idx and key[1] not in current_episode_agvs
            ]
            for key in stale_keys:
                prev_agv_state.pop(key, None)

    return {
        "episodes": len(episodes_seen),
        "total_steps": total_steps,
        "total_deliveries": total_deliveries,
        "total_clashes": total_clashes,
        "total_stucks": total_stucks,
        "agv_carrying_step_ratio": _ratio(agv_carrying_entries, agv_entries),
        "agv_load_events": agv_load_events,
        "agv_deliver_events_proxy": agv_deliver_events_proxy,
        "picker_busy_ratio": _ratio(picker_busy_entries, picker_entries),
        "avg_reward_per_step": _ratio(total_reward, total_steps),
    }


def print_summary(summary: dict) -> None:
    """Pretty print a compact diagnostics summary."""
    print("Trace Summary")
    print(f"episodes: {summary.get('episodes', 0)}")
    print(f"total_steps: {summary.get('total_steps', 0)}")
    print(f"total_deliveries: {summary.get('total_deliveries', 0)}")
    print(f"total_clashes: {summary.get('total_clashes', 0)}")
    print(f"total_stucks: {summary.get('total_stucks', 0)}")
    print(f"agv_carrying_step_ratio: {float(summary.get('agv_carrying_step_ratio', 0.0)):.3f}")
    print(f"agv_load_events: {summary.get('agv_load_events', 0)}")
    print(f"agv_deliver_events_proxy: {summary.get('agv_deliver_events_proxy', 0)}")
    print(f"picker_busy_ratio: {float(summary.get('picker_busy_ratio', 0.0)):.3f}")
    print(f"avg_reward_per_step: {float(summary.get('avg_reward_per_step', 0.0)):.3f}")


def _ratio(numerator: float, denominator: float) -> float:
    """Safely divide two values."""
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _safe_int(value: Any) -> int:
    """Convert a value to int, falling back to zero."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    """Convert a value to float, falling back to zero."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main() -> int:
    """CLI entrypoint for trace diagnostics."""
    parser = argparse.ArgumentParser(description="Summarize TA-RWARE JSONL trace diagnostics.")
    parser.add_argument("--jsonl", required=True, help="Path to the JSONL trace file.")
    args = parser.parse_args()

    records = load_jsonl(args.jsonl)
    summary = summarize_trace(records)
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
