"""Communication metrics for TA-RWARE symbiotic coordination traces."""

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
                raise ValueError(
                    f"Expected object on line {line_no} in {file_path}, got {type(item).__name__}"
                )
            records.append(item)
    return records


def summarize_communication(records: list[dict]) -> dict:
    """Compute aggregate communication metrics from step-level JSONL records.

    ``total_alternative_proposals`` is kept for backward compatibility, but it
    now counts ``PREFER`` responses under the candidate-based protocol. Counts
    are recomputed from ``comm_response`` directly so they stay consistent with
    TRM logic.
    """
    episodes_seen: set[int] = set()
    total_steps = len(records)
    communication_steps = 0
    total_accepts = 0
    total_declines = 0
    total_alternative_proposals = 0
    total_trm_events = 0
    episode_breakdown: dict[int, dict[str, int]] = {}

    for record in records:
        episode_idx = _safe_int(record.get("episode_idx", 0))
        episodes_seen.add(episode_idx)
        breakdown = episode_breakdown.setdefault(
            episode_idx,
            {
                "steps": 0,
                "communication_steps": 0,
                "accepts": 0,
                "declines": 0,
                "alternatives": 0,
                "trm_events": 0,
            },
        )

        breakdown["steps"] += 1

        used = bool(record.get("communication_used", False))
        communication_steps += int(used)
        breakdown["communication_steps"] += int(used)

        accept_count, decline_count, alternative_count = _count_response_decisions_in_record(record)
        trm_events = _count_trm_in_record(record)

        total_accepts += accept_count
        total_declines += decline_count
        total_alternative_proposals += alternative_count
        total_trm_events += trm_events

        breakdown["accepts"] += accept_count
        breakdown["declines"] += decline_count
        breakdown["alternatives"] += alternative_count
        breakdown["trm_events"] += trm_events

    total_responses = total_accepts + total_declines + total_alternative_proposals
    return {
        "episodes": len(episodes_seen),
        "total_steps": total_steps,
        "communication_steps": communication_steps,
        "communication_ratio": _safe_ratio(communication_steps, total_steps),
        "total_accepts": total_accepts,
        "total_declines": total_declines,
        "total_alternative_proposals": total_alternative_proposals,
        "total_responses": total_responses,
        "accept_ratio": _safe_ratio(total_accepts, total_responses),
        "decline_ratio": _safe_ratio(total_declines, total_responses),
        "alternative_ratio": _safe_ratio(total_alternative_proposals, total_responses),
        "total_trm_events": total_trm_events,
        "trm_per_communication_step": _safe_ratio(total_trm_events, communication_steps),
        "episode_breakdown": episode_breakdown,
    }


def print_summary(summary: dict) -> None:
    """Pretty print the communication summary."""
    print("Communication Summary")
    print(f"episodes: {summary.get('episodes', 0)}")
    print(f"total_steps: {summary.get('total_steps', 0)}")
    print(f"communication_steps: {summary.get('communication_steps', 0)}")
    print(f"communication_ratio: {float(summary.get('communication_ratio', 0.0)):.3f}")
    print(f"total_accepts: {summary.get('total_accepts', 0)}")
    print(f"total_declines: {summary.get('total_declines', 0)}")
    print(f"total_alternative_proposals: {summary.get('total_alternative_proposals', 0)}")
    print(f"total_responses: {summary.get('total_responses', 0)}")
    print(f"accept_ratio: {float(summary.get('accept_ratio', 0.0)):.3f}")
    print(f"decline_ratio: {float(summary.get('decline_ratio', 0.0)):.3f}")
    print(f"alternative_ratio: {float(summary.get('alternative_ratio', 0.0)):.3f}")
    print(f"total_trm_events: {summary.get('total_trm_events', 0)}")
    print(f"trm_per_communication_step: {float(summary.get('trm_per_communication_step', 0.0)):.3f}")

    episode_breakdown = summary.get("episode_breakdown", {})
    if isinstance(episode_breakdown, dict) and episode_breakdown:
        print("episode_breakdown:")
        for episode_idx in sorted(episode_breakdown):
            item = episode_breakdown[episode_idx]
            print(
                f"  ep{episode_idx}: steps={item.get('steps', 0)} "
                f"comm_steps={item.get('communication_steps', 0)} "
                f"accepts={item.get('accepts', 0)} "
                f"declines={item.get('declines', 0)} "
                f"alternatives={item.get('alternatives', 0)} "
                f"trm_events={item.get('trm_events', 0)}"
            )


def _count_trm_in_record(record: dict) -> int:
    """Count target revisions due to message in a single step record.

    A TRM event now means the picker responded with ``PREFER``, selected a
    communicated candidate that is not the first AGV candidate, and the final
    plan assigned the AGV to that preferred communicated rack. If a record has
    no ``PREFER`` responses, its TRM count is zero.
    """
    comm_request = record.get("comm_request")
    comm_response = record.get("comm_response")
    comm_final_plan = record.get("comm_final_plan")

    if not isinstance(comm_request, dict):
        return 0
    if not isinstance(comm_response, dict):
        return 0
    if not isinstance(comm_final_plan, dict):
        return 0

    requests = comm_request.get("requests", [])
    responses = comm_response.get("responses", [])
    macro_actions = comm_final_plan.get("macro_actions", [])

    if not isinstance(requests, list) or not isinstance(responses, list) or not isinstance(macro_actions, list):
        return 0

    request_to_metadata: dict[str, tuple[int, int]] = {}
    for item in requests:
        if not isinstance(item, dict):
            continue
        request_id = item.get("request_id")
        agv_id = item.get("agv_id")
        candidates = item.get("candidates", [])
        if not isinstance(request_id, str):
            continue
        if not isinstance(candidates, list) or not candidates:
            continue
        first_candidate = candidates[0]
        if not isinstance(first_candidate, dict):
            continue
        first_rack_id = _safe_int(first_candidate.get("rack_id", 0))
        request_to_metadata[request_id] = (_safe_int(agv_id), first_rack_id)

    final_actions: dict[int, int] = {}
    for item in macro_actions:
        if not isinstance(item, dict):
            continue
        agent_id = _safe_int(item.get("agent_id", 0))
        location_id = _safe_int(item.get("location_id", 0))
        final_actions[agent_id] = location_id

    prefer_found = False
    trm_events = 0
    for item in responses:
        if not isinstance(item, dict):
            continue
        if item.get("decision") != "PREFER":
            continue
        prefer_found = True
        request_id = item.get("request_id")
        if not isinstance(request_id, str):
            continue
        if item.get("chosen_rack_id") is None:
            continue
        chosen_rack_id = _safe_int(item.get("chosen_rack_id", 0))
        metadata = request_to_metadata.get(request_id)
        if metadata is None:
            continue
        agv_id, first_candidate_rack_id = metadata
        if chosen_rack_id == 0 or chosen_rack_id == first_candidate_rack_id:
            continue
        if final_actions.get(agv_id) == chosen_rack_id:
            trm_events += 1
    if not prefer_found:
        return 0
    return trm_events


def _count_response_decisions_in_record(record: dict) -> tuple[int, int, int]:
    """Count ACCEPT, DECLINE, and PREFER responses in one record.

    The third returned value maps to ``total_alternative_proposals`` for
    backward compatibility, but semantically it now counts ``PREFER`` events.
    """
    comm_response = record.get("comm_response")
    if not isinstance(comm_response, dict):
        return 0, 0, 0
    responses = comm_response.get("responses", [])
    if not isinstance(responses, list):
        return 0, 0, 0

    accept_count = 0
    decline_count = 0
    prefer_count = 0
    for item in responses:
        if not isinstance(item, dict):
            continue
        decision = item.get("decision")
        if decision == "ACCEPT":
            accept_count += 1
        elif decision == "DECLINE":
            decline_count += 1
        elif decision == "PREFER":
            prefer_count += 1
    return accept_count, decline_count, prefer_count


def _safe_ratio(num: float, den: float) -> float:
    """Return a safe division result."""
    if den == 0:
        return 0.0
    return float(num) / float(den)


def _safe_int(value: Any) -> int:
    """Convert a value to int, returning 0 on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def main() -> int:
    """CLI entrypoint for communication metrics."""
    parser = argparse.ArgumentParser(description="Summarize TA-RWARE communication metrics from JSONL.")
    parser.add_argument("--jsonl", required=True, help="Path to the JSONL trace file.")
    args = parser.parse_args()

    records = load_jsonl(args.jsonl)
    summary = summarize_communication(records)
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
