"""Communication metrics for NonMutualisticCommLLMPlanner v2.

Aligned to:
- partner-aware but non-negotiative staged communication
- Stage 1: committed target selection
- Stage 2: ACK / BUSY on committed target
- Stage 3: deterministic integration (no LLM revision)
"""

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


def summarize_communication_non_mutualistic_v2(records: list[dict]) -> dict[str, Any]:
    """Compute aggregate communication metrics for the non-mutualistic v2 protocol."""
    episodes_seen: set[int] = set()
    total_steps = len(records)
    communication_steps = 0

    total_stage1_requests = 0
    total_stage2_requests = 0
    total_ack_requests = 0
    total_busy_requests = 0

    total_picker_candidates = 0
    picker_candidate_count_breakdown = {0: 0, 1: 0, 2: 0, 3: 0, "gt3": 0}
    total_zero_candidate_requests = 0

    total_assigned_requests = 0
    total_illegal_revisions = 0

    fallback_steps = 0

    total_num_assignments_sum = 0
    total_sum_sync_cost = 0
    total_sum_eta_gap = 0
    total_max_sync_cost_sum = 0
    objective_score_steps = 0

    episode_breakdown: dict[int, dict[str, Any]] = {}

    for record in records:
        episode_idx = _safe_int(record.get("episode_idx", 0))
        episodes_seen.add(episode_idx)

        breakdown = episode_breakdown.setdefault(
            episode_idx,
            {
                "steps": 0,
                "communication_steps": 0,
                "stage1_requests": 0,
                "stage2_requests": 0,
                "ack_requests": 0,
                "busy_requests": 0,
                "picker_candidates_total": 0,
                "zero_candidate_requests": 0,
                "picker_candidate_count_breakdown": {0: 0, 1: 0, 2: 0, 3: 0, "gt3": 0},
                "assigned_requests": 0,
                "illegal_revisions": 0,
                "fallback_steps": 0,
                "objective_score_steps": 0,
                "num_assignments_sum": 0,
                "sum_sync_cost": 0,
                "sum_eta_gap": 0,
                "max_sync_cost_sum": 0,
            },
        )
        breakdown["steps"] += 1

        used = bool(record.get("communication_used", False))
        if not used:
            continue

        communication_steps += 1
        breakdown["communication_steps"] += 1

        comm_request = _unwrap_message(record.get("comm_request"))
        comm_response = _unwrap_message(record.get("comm_response"))
        comm_final_plan = _unwrap_message(record.get("comm_final_plan"))

        stage1_requests = _extract_stage1_requests(comm_request)
        stage2_responses = _extract_stage2_responses(comm_response)
        final_assignments = _extract_stage3_assignments(comm_final_plan)

        total_stage1_requests += len(stage1_requests)
        total_stage2_requests += len(stage2_responses)
        breakdown["stage1_requests"] += len(stage1_requests)
        breakdown["stage2_requests"] += len(stage2_responses)

        stage2_metrics = _count_stage2_ack_busy_metrics(stage2_responses)
        total_ack_requests += stage2_metrics["ack_requests"]
        total_busy_requests += stage2_metrics["busy_requests"]
        total_picker_candidates += stage2_metrics["picker_candidates_total"]
        total_zero_candidate_requests += stage2_metrics["zero_candidate_requests"]

        breakdown["ack_requests"] += stage2_metrics["ack_requests"]
        breakdown["busy_requests"] += stage2_metrics["busy_requests"]
        breakdown["picker_candidates_total"] += stage2_metrics["picker_candidates_total"]
        breakdown["zero_candidate_requests"] += stage2_metrics["zero_candidate_requests"]

        for key, value in stage2_metrics["picker_candidate_count_breakdown"].items():
            picker_candidate_count_breakdown[key] += value
            breakdown["picker_candidate_count_breakdown"][key] += value

        assigned_count = len(final_assignments)
        total_assigned_requests += assigned_count
        breakdown["assigned_requests"] += assigned_count

        illegal_revisions = _count_illegal_revision_metrics(stage1_requests, final_assignments)
        total_illegal_revisions += illegal_revisions
        breakdown["illegal_revisions"] += illegal_revisions

        if _is_fallback_plan(comm_final_plan):
            fallback_steps += 1
            breakdown["fallback_steps"] += 1

        objective_scores = _extract_objective_scores(comm_final_plan)
        if objective_scores is not None:
            objective_score_steps += 1
            breakdown["objective_score_steps"] += 1

            num_assignments = _safe_int(objective_scores.get("num_assignments", 0))
            sum_sync_cost = _safe_int(objective_scores.get("sum_sync_cost", 0))
            sum_eta_gap = _safe_int(objective_scores.get("sum_eta_gap", 0))
            max_sync_cost = _safe_int(objective_scores.get("max_sync_cost", 0))

            total_num_assignments_sum += num_assignments
            total_sum_sync_cost += sum_sync_cost
            total_sum_eta_gap += sum_eta_gap
            total_max_sync_cost_sum += max_sync_cost

            breakdown["num_assignments_sum"] += num_assignments
            breakdown["sum_sync_cost"] += sum_sync_cost
            breakdown["sum_eta_gap"] += sum_eta_gap
            breakdown["max_sync_cost_sum"] += max_sync_cost

    return {
        "episodes": len(episodes_seen),
        "total_steps": total_steps,
        "communication_steps": communication_steps,
        "communication_ratio": _safe_ratio(communication_steps, total_steps),
        "total_stage1_requests": total_stage1_requests,
        "total_stage2_requests": total_stage2_requests,
        "total_ack_requests": total_ack_requests,
        "total_busy_requests": total_busy_requests,
        "ack_ratio": _safe_ratio(total_ack_requests, total_stage2_requests),
        "busy_ratio": _safe_ratio(total_busy_requests, total_stage2_requests),
        "total_picker_candidates": total_picker_candidates,
        "avg_picker_candidates_per_request": _safe_ratio(total_picker_candidates, total_stage2_requests),
        "total_zero_candidate_requests": total_zero_candidate_requests,
        "zero_candidate_request_ratio": _safe_ratio(total_zero_candidate_requests, total_stage2_requests),
        "picker_candidate_count_breakdown": picker_candidate_count_breakdown,
        "total_assigned_requests": total_assigned_requests,
        "assignment_rate": _safe_ratio(total_assigned_requests, total_ack_requests),
        "total_illegal_revisions": total_illegal_revisions,
        "illegal_revision_rate": _safe_ratio(total_illegal_revisions, total_assigned_requests),
        "fallback_steps": fallback_steps,
        "fallback_rate": _safe_ratio(fallback_steps, communication_steps),
        "avg_num_assignments_per_comm_step": _safe_ratio(total_num_assignments_sum, objective_score_steps),
        "avg_sum_sync_cost_per_comm_step": _safe_ratio(total_sum_sync_cost, objective_score_steps),
        "avg_sum_eta_gap_per_comm_step": _safe_ratio(total_sum_eta_gap, objective_score_steps),
        "avg_max_sync_cost_per_comm_step": _safe_ratio(total_max_sync_cost_sum, objective_score_steps),
        "episode_breakdown": episode_breakdown,
    }


def print_summary(summary: dict[str, Any]) -> None:
    """Pretty print the non-mutualistic v2 communication summary."""
    print("Communication Summary Non-Mutualistic V2")
    print(f"episodes: {summary.get('episodes', 0)}")
    print(f"total_steps: {summary.get('total_steps', 0)}")
    print(f"communication_steps: {summary.get('communication_steps', 0)}")
    print(f"communication_ratio: {float(summary.get('communication_ratio', 0.0)):.3f}")

    print(f"total_stage1_requests: {summary.get('total_stage1_requests', 0)}")
    print(f"total_stage2_requests: {summary.get('total_stage2_requests', 0)}")
    print(f"total_ack_requests: {summary.get('total_ack_requests', 0)}")
    print(f"total_busy_requests: {summary.get('total_busy_requests', 0)}")
    print(f"ack_ratio: {float(summary.get('ack_ratio', 0.0)):.3f}")
    print(f"busy_ratio: {float(summary.get('busy_ratio', 0.0)):.3f}")

    print(f"total_picker_candidates: {summary.get('total_picker_candidates', 0)}")
    print(
        f"avg_picker_candidates_per_request: "
        f"{float(summary.get('avg_picker_candidates_per_request', 0.0)):.3f}"
    )
    print(f"total_zero_candidate_requests: {summary.get('total_zero_candidate_requests', 0)}")
    print(
        f"zero_candidate_request_ratio: "
        f"{float(summary.get('zero_candidate_request_ratio', 0.0)):.3f}"
    )
    print(f"picker_candidate_count_breakdown: {summary.get('picker_candidate_count_breakdown', {})}")

    print(f"total_assigned_requests: {summary.get('total_assigned_requests', 0)}")
    print(f"assignment_rate: {float(summary.get('assignment_rate', 0.0)):.3f}")
    print(f"total_illegal_revisions: {summary.get('total_illegal_revisions', 0)}")
    print(f"illegal_revision_rate: {float(summary.get('illegal_revision_rate', 0.0)):.3f}")

    print(f"fallback_steps: {summary.get('fallback_steps', 0)}")
    print(f"fallback_rate: {float(summary.get('fallback_rate', 0.0)):.3f}")

    print(
        f"avg_num_assignments_per_comm_step: "
        f"{float(summary.get('avg_num_assignments_per_comm_step', 0.0)):.3f}"
    )
    print(
        f"avg_sum_sync_cost_per_comm_step: "
        f"{float(summary.get('avg_sum_sync_cost_per_comm_step', 0.0)):.3f}"
    )
    print(
        f"avg_sum_eta_gap_per_comm_step: "
        f"{float(summary.get('avg_sum_eta_gap_per_comm_step', 0.0)):.3f}"
    )
    print(
        f"avg_max_sync_cost_per_comm_step: "
        f"{float(summary.get('avg_max_sync_cost_per_comm_step', 0.0)):.3f}"
    )

    episode_breakdown = summary.get("episode_breakdown", {})
    if isinstance(episode_breakdown, dict) and episode_breakdown:
        print("episode_breakdown:")
        for episode_idx in sorted(episode_breakdown):
            item = episode_breakdown[episode_idx]
            print(
                f"  ep{episode_idx}: "
                f"steps={item.get('steps', 0)} "
                f"comm_steps={item.get('communication_steps', 0)} "
                f"stage1_requests={item.get('stage1_requests', 0)} "
                f"stage2_requests={item.get('stage2_requests', 0)} "
                f"ack_requests={item.get('ack_requests', 0)} "
                f"busy_requests={item.get('busy_requests', 0)} "
                f"picker_candidates_total={item.get('picker_candidates_total', 0)} "
                f"zero_candidate_requests={item.get('zero_candidate_requests', 0)} "
                f"assigned={item.get('assigned_requests', 0)} "
                f"illegal_revisions={item.get('illegal_revisions', 0)} "
                f"fallbacks={item.get('fallback_steps', 0)}"
            )


def _unwrap_message(value: Any) -> dict[str, Any]:
    """Best-effort unwrap for comm_* fields."""
    if isinstance(value, dict):
        return value
    if value is not None and hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            unwrapped = value.to_dict()
            if isinstance(unwrapped, dict):
                return unwrapped
        except Exception:
            return {}
    return {}


def _extract_stage1_requests(comm_request: dict[str, Any]) -> list[dict[str, Any]]:
    requests = comm_request.get("requests", [])
    if not isinstance(requests, list):
        return []
    return [item for item in requests if isinstance(item, dict)]


def _extract_stage2_responses(comm_response: dict[str, Any]) -> list[dict[str, Any]]:
    responses = comm_response.get("responses", [])
    if not isinstance(responses, list):
        return []
    return [item for item in responses if isinstance(item, dict)]


def _extract_stage3_assignments(comm_final_plan: dict[str, Any]) -> list[dict[str, Any]]:
    assignments = comm_final_plan.get("assignments", [])
    if not isinstance(assignments, list):
        return []
    return [item for item in assignments if isinstance(item, dict)]


def _extract_objective_scores(comm_final_plan: dict[str, Any]) -> dict[str, Any] | None:
    scores = comm_final_plan.get("objective_scores")
    if not isinstance(scores, dict):
        return None
    return scores


def _normalize_stage2_status(item: dict[str, Any]) -> str:
    status = str(item.get("status", "BUSY")).upper()
    if status == "ACK":
        return "ACK"
    return "BUSY"


def _stage2_picker_candidates_count(item: dict[str, Any]) -> int:
    count = _safe_int(item.get("picker_candidates_count"))
    if count >= 0:
        return count
    return 0


def _count_stage2_ack_busy_metrics(stage2_responses: list[dict[str, Any]]) -> dict[str, Any]:
    ack_requests = 0
    busy_requests = 0
    picker_candidates_total = 0
    zero_candidate_requests = 0
    hist = {0: 0, 1: 0, 2: 0, 3: 0, "gt3": 0}

    for item in stage2_responses:
        status = _normalize_stage2_status(item)
        if status == "ACK":
            ack_requests += 1
        else:
            busy_requests += 1

        count = _stage2_picker_candidates_count(item)
        picker_candidates_total += count
        if count == 0:
            zero_candidate_requests += 1

        if count in hist:
            hist[count] += 1
        elif count > 3:
            hist["gt3"] += 1

    return {
        "ack_requests": ack_requests,
        "busy_requests": busy_requests,
        "picker_candidates_total": picker_candidates_total,
        "zero_candidate_requests": zero_candidate_requests,
        "picker_candidate_count_breakdown": hist,
    }


def _count_illegal_revision_metrics(
    stage1_requests: list[dict[str, Any]],
    final_assignments: list[dict[str, Any]],
) -> int:
    committed_by_request: dict[str, int] = {}
    for item in stage1_requests:
        request_id = item.get("request_id")
        if not isinstance(request_id, str):
            continue
        committed_rack_id = _safe_int(item.get("committed_rack_id", 0))
        if committed_rack_id > 0:
            committed_by_request[request_id] = committed_rack_id

    illegal_revisions = 0
    for item in final_assignments:
        request_id = item.get("request_id")
        if not isinstance(request_id, str):
            continue
        committed_rack_id = committed_by_request.get(request_id)
        if committed_rack_id is None:
            continue
        final_rack_id = _safe_int(item.get("rack_id", 0))
        if final_rack_id > 0 and final_rack_id != committed_rack_id:
            illegal_revisions += 1
    return illegal_revisions


def _is_fallback_plan(comm_final_plan: dict[str, Any]) -> bool:
    explanation = comm_final_plan.get("explanation", "")
    if explanation is None:
        return False
    explanation_text = str(explanation)
    return (
        explanation_text.startswith("FALLBACK_TO_RULE_NON_MUTUALISTIC_PARTNER_AWARE")
        or explanation_text.startswith("FALLBACK_TO_RULE_NON_MUTUALISTIC")
        or explanation_text.startswith("FALLBACK")
    )


def _safe_ratio(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num) / float(den)


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def main() -> int:
    """CLI entrypoint for non-mutualistic v2 communication metrics."""
    parser = argparse.ArgumentParser(
        description="Summarize NonMutualisticCommLLMPlanner v2 communication metrics from JSONL."
    )
    parser.add_argument("--jsonl", required=True, help="Path to the JSONL trace file.")
    args = parser.parse_args()

    records = load_jsonl(args.jsonl)
    summary = summarize_communication_non_mutualistic_v2(records)
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())