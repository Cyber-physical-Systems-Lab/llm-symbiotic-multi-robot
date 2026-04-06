"""Communication metrics v2 for SymbioticCommLLMPlanner JSONL traces."""

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


def summarize_communication_v2(records: list[dict]) -> dict:
    """Compute aggregate communication metrics for the new 3-stage protocol.

    Expected trace structure per record:
      - communication_used: bool
      - comm_request: {"requests": [...]}
      - comm_response: {"responses": [...]}
      - comm_final_plan: {
            "assignments": [...],
            "skipped": [...],
            "objective_scores": {...},
            "explanation": "..."
        }

    Protocol semantics:
      - Stage2 decisions: RECOMMEND / DECLINE
      - Stage3 final assignments: selected conflict-free assignments
      - revision event: final rack_id != Stage1 primary_rack_id for same request
      - fallback step: comm_final_plan["explanation"] starts with "FALLBACK_TO_RULE_SYMBIOTIC"
    """
    episodes_seen: set[int] = set()
    total_steps = len(records)
    communication_steps = 0

    total_stage1_requests = 0
    total_stage2_requests = 0
    total_recommends = 0
    total_declines = 0

    total_options = 0
    total_recommended_requests = 0
    option_count_breakdown = {0: 0, 1: 0, 2: 0, "gt2": 0}

    total_assigned_requests = 0
    total_recommended_assignable_requests = 0  # denominator for assignment_rate

    total_revisions = 0
    total_assigned_requests_for_revision = 0

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
                "recommends": 0,
                "declines": 0,
                "recommended_requests": 0,
                "options_total": 0,
                "option_count_breakdown": {0: 0, 1: 0, 2: 0, "gt2": 0},
                "assigned_requests": 0,
                "recommended_assignable_requests": 0,
                "revisions": 0,
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

        # Stage2 counts
        recommends, declines, options_total, recommended_requests_count, option_hist = _count_stage2_response_metrics(
            stage2_responses
        )
        total_recommends += recommends
        total_declines += declines
        total_options += options_total
        total_recommended_requests += recommended_requests_count

        breakdown["recommends"] += recommends
        breakdown["declines"] += declines
        breakdown["options_total"] += options_total
        breakdown["recommended_requests"] += recommended_requests_count

        for key, value in option_hist.items():
            option_count_breakdown[key] += value
            breakdown["option_count_breakdown"][key] += value

        # Assignment-related metrics
        assigned_count = len(final_assignments)
        total_assigned_requests += assigned_count
        breakdown["assigned_requests"] += assigned_count

        # denominator: recommended requests (i.e., stage2 said RECOMMEND)
        total_recommended_assignable_requests += recommended_requests_count
        breakdown["recommended_assignable_requests"] += recommended_requests_count

        # Revision rate: final rack != stage1 primary
        revisions, assigned_for_revision = _count_revision_metrics(stage1_requests, final_assignments)
        total_revisions += revisions
        total_assigned_requests_for_revision += assigned_for_revision
        breakdown["revisions"] += revisions

        # Fallback detection
        if _is_fallback_plan(comm_final_plan):
            fallback_steps += 1
            breakdown["fallback_steps"] += 1

        # Objective scores
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

    total_stage2_decisions = total_recommends + total_declines

    summary = {
        "episodes": len(episodes_seen),
        "total_steps": total_steps,
        "communication_steps": communication_steps,
        "communication_ratio": _safe_ratio(communication_steps, total_steps),
        "total_stage1_requests": total_stage1_requests,
        "total_stage2_requests": total_stage2_requests,
        "total_recommends": total_recommends,
        "total_declines": total_declines,
        "recommend_ratio": _safe_ratio(total_recommends, total_stage2_decisions),
        "decline_ratio": _safe_ratio(total_declines, total_stage2_decisions),
        "total_options": total_options,
        "avg_options_per_request": _safe_ratio(total_options, total_stage2_requests),
        "avg_options_per_recommended_request": _safe_ratio(total_options, total_recommended_requests),
        "option_count_breakdown": option_count_breakdown,
        "total_assigned_requests": total_assigned_requests,
        "assignment_rate": _safe_ratio(total_assigned_requests, total_recommended_assignable_requests),
        "total_revisions": total_revisions,
        "revision_rate": _safe_ratio(total_revisions, total_assigned_requests_for_revision),
        "fallback_steps": fallback_steps,
        "fallback_rate": _safe_ratio(fallback_steps, communication_steps),
        "avg_num_assignments_per_comm_step": _safe_ratio(total_num_assignments_sum, objective_score_steps),
        "avg_sum_sync_cost_per_comm_step": _safe_ratio(total_sum_sync_cost, objective_score_steps),
        "avg_sum_eta_gap_per_comm_step": _safe_ratio(total_sum_eta_gap, objective_score_steps),
        "avg_max_sync_cost_per_comm_step": _safe_ratio(total_max_sync_cost_sum, objective_score_steps),
        "episode_breakdown": episode_breakdown,
    }
    return summary


def print_summary(summary: dict) -> None:
    """Pretty print the communication summary v2."""
    print("Communication Summary V2")
    print(f"episodes: {summary.get('episodes', 0)}")
    print(f"total_steps: {summary.get('total_steps', 0)}")
    print(f"communication_steps: {summary.get('communication_steps', 0)}")
    print(f"communication_ratio: {float(summary.get('communication_ratio', 0.0)):.3f}")

    print(f"total_stage1_requests: {summary.get('total_stage1_requests', 0)}")
    print(f"total_stage2_requests: {summary.get('total_stage2_requests', 0)}")
    print(f"total_recommends: {summary.get('total_recommends', 0)}")
    print(f"total_declines: {summary.get('total_declines', 0)}")
    print(f"recommend_ratio: {float(summary.get('recommend_ratio', 0.0)):.3f}")
    print(f"decline_ratio: {float(summary.get('decline_ratio', 0.0)):.3f}")

    print(f"total_options: {summary.get('total_options', 0)}")
    print(f"avg_options_per_request: {float(summary.get('avg_options_per_request', 0.0)):.3f}")
    print(
        f"avg_options_per_recommended_request: "
        f"{float(summary.get('avg_options_per_recommended_request', 0.0)):.3f}"
    )
    print(f"option_count_breakdown: {summary.get('option_count_breakdown', {})}")

    print(f"total_assigned_requests: {summary.get('total_assigned_requests', 0)}")
    print(f"assignment_rate: {float(summary.get('assignment_rate', 0.0)):.3f}")

    print(f"total_revisions: {summary.get('total_revisions', 0)}")
    print(f"revision_rate: {float(summary.get('revision_rate', 0.0)):.3f}")

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
                f"recommends={item.get('recommends', 0)} "
                f"declines={item.get('declines', 0)} "
                f"assigned={item.get('assigned_requests', 0)} "
                f"revisions={item.get('revisions', 0)} "
                f"fallbacks={item.get('fallback_steps', 0)}"
            )


def _unwrap_message(value: Any) -> dict:
    """Best-effort unwrap for comm_* fields.

    Supports:
      - plain dict
      - objects exposing to_dict()
      - None -> {}
    """
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


def _extract_stage1_requests(comm_request: dict) -> list[dict]:
    requests = comm_request.get("requests", [])
    if not isinstance(requests, list):
        return []
    return [item for item in requests if isinstance(item, dict)]


def _extract_stage2_responses(comm_response: dict) -> list[dict]:
    responses = comm_response.get("responses", [])
    if not isinstance(responses, list):
        return []
    return [item for item in responses if isinstance(item, dict)]


def _extract_stage3_assignments(comm_final_plan: dict) -> list[dict]:
    assignments = comm_final_plan.get("assignments", [])
    if not isinstance(assignments, list):
        return []
    return [item for item in assignments if isinstance(item, dict)]


def _extract_objective_scores(comm_final_plan: dict) -> dict | None:
    scores = comm_final_plan.get("objective_scores")
    if not isinstance(scores, dict):
        return None
    return scores


def _is_fallback_plan(comm_final_plan: dict) -> bool:
    explanation = comm_final_plan.get("explanation", "")
    if explanation is None:
        return False
    return str(explanation).startswith("FALLBACK_TO_RULE_SYMBIOTIC")


def _count_stage2_response_metrics(
    stage2_responses: list[dict],
) -> tuple[int, int, int, int, dict]:
    """Return:
    (recommend_count, decline_count, total_options, recommended_requests_count, option_hist)
    """
    recommend_count = 0
    decline_count = 0
    total_options = 0
    recommended_requests_count = 0
    option_hist = {0: 0, 1: 0, 2: 0, "gt2": 0}

    for item in stage2_responses:
        decision = str(item.get("decision", "")).upper()
        options = item.get("options", [])
        if not isinstance(options, list):
            options = []

        n_options = len(options)
        if n_options in option_hist:
            option_hist[n_options] += 1
        elif n_options > 2:
            option_hist["gt2"] += 1

        if decision == "RECOMMEND":
            recommend_count += 1
            recommended_requests_count += 1
            total_options += n_options
        elif decision == "DECLINE":
            decline_count += 1

    return recommend_count, decline_count, total_options, recommended_requests_count, option_hist


def _count_revision_metrics(
    stage1_requests: list[dict],
    final_assignments: list[dict],
) -> tuple[int, int]:
    """Return:
    (revision_count, assigned_requests_count)

    revision = final rack != stage1 primary_rack_id for the same request_id
    """
    primary_by_request: dict[str, int] = {}
    for item in stage1_requests:
        request_id = item.get("request_id")
        if not isinstance(request_id, str):
            continue
        primary_rack_id = _safe_int(item.get("primary_rack_id", 0))
        if primary_rack_id > 0:
            primary_by_request[request_id] = primary_rack_id

    revision_count = 0
    assigned_requests_count = 0
    for item in final_assignments:
        request_id = item.get("request_id")
        if not isinstance(request_id, str):
            continue
        rack_id = _safe_int(item.get("rack_id", 0))
        primary = primary_by_request.get(request_id)
        if primary is None:
            continue
        assigned_requests_count += 1
        if rack_id > 0 and rack_id != primary:
            revision_count += 1

    return revision_count, assigned_requests_count


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
    """CLI entrypoint for communication metrics v2."""
    parser = argparse.ArgumentParser(
        description="Summarize SymbioticCommLLMPlanner communication metrics from JSONL."
    )
    parser.add_argument("--jsonl", required=True, help="Path to the JSONL trace file.")
    args = parser.parse_args()

    records = load_jsonl(args.jsonl)
    summary = summarize_communication_v2(records)
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())