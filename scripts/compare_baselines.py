"""Compare non-symbiotic and symbiotic TA-RWARE planners across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import tarware  # noqa: F401


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from symco.eval.behavior_metrics import load_jsonl as load_behavior_jsonl
from symco.eval.behavior_metrics import summarize_behavior
from symco.eval.comm_metrics import load_jsonl as load_comm_jsonl
from symco.eval.comm_metrics import summarize_communication
from symco.planners.non_symbiotic_planner import NonSymbioticPlanner
from symco.planners.symbiotic_planner import SymbioticPlanner
from symco.run.runner import EpisodeRunner, RunnerConfig


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for planner comparison."""
    parser = argparse.ArgumentParser(
        description="Compare non-symbiotic and symbiotic TA-RWARE planners."
    )
    parser.add_argument("--env_id", required=True, help="Gymnasium environment id.")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per seed.")
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seed list.")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum steps per episode.")
    parser.add_argument("--out_dir", type=str, default="outputs_compare", help="Output directory.")
    parser.add_argument("--topk_requests", type=int, default=10, help="Top-k requested racks.")
    parser.add_argument("--topk_empty", type=int, default=10, help="Top-k empty racks.")
    parser.add_argument("--topk_goals", type=int, default=10, help="Top-k goals.")
    parser.add_argument(
        "--pickers_to_agvs",
        type=int,
        choices=(0, 1),
        default=1,
        help="Whether picker valid actions follow AGV targets.",
    )
    parser.add_argument(
        "--block_conflicting_actions",
        type=int,
        choices=(0, 1),
        default=1,
        help="Whether to block conflicting same-type actions.",
    )
    parser.add_argument(
        "--care_for_agents_in_cost",
        type=int,
        choices=(0, 1),
        default=0,
        help="Whether path costs account for other agents.",
    )
    return parser.parse_args()


def parse_seeds(text: str) -> list[int]:
    """Parse a comma-separated seed string."""
    parts = [part.strip() for part in text.split(",")]
    seeds = [int(part) for part in parts if part]
    if not seeds:
        raise ValueError("At least one seed must be provided.")
    return seeds


def build_config(args: argparse.Namespace, seed: int) -> RunnerConfig:
    """Build a runner config for one seed."""
    return RunnerConfig(
        env_id=args.env_id,
        max_steps=args.max_steps,
        seed=seed,
        num_episodes=args.episodes,
        out_dir=args.out_dir,
        topk_requests=args.topk_requests,
        topk_empty=args.topk_empty,
        topk_goals=args.topk_goals,
        pickers_to_agvs=bool(args.pickers_to_agvs),
        block_conflicting_actions=bool(args.block_conflicting_actions),
        care_for_agents_in_cost=bool(args.care_for_agents_in_cost),
    )


def snapshot_jsonl_files(out_dir: Path) -> set[Path]:
    """Return the set of existing JSONL files in the output directory."""
    return set(out_dir.glob("*.jsonl"))


def find_new_jsonl(out_dir: Path, before: set[Path]) -> Path:
    """Find the JSONL file created by the latest run."""
    after = set(out_dir.glob("*.jsonl"))
    new_files = sorted(after - before, key=lambda path: path.stat().st_mtime)
    if len(new_files) == 1:
        return new_files[0]
    if len(new_files) > 1:
        return new_files[-1]
    if not after:
        raise FileNotFoundError(f"No JSONL files found in {out_dir}")
    return max(after, key=lambda path: path.stat().st_mtime)


def mean_episode_metric(episode_summaries: list[dict[str, Any]], key: str) -> float:
    """Compute the mean value of one episode metric."""
    if not episode_summaries:
        return 0.0
    total = sum(float(summary.get(key, 0.0)) for summary in episode_summaries)
    return total / float(len(episode_summaries))


def zero_comm_summary() -> dict[str, Any]:
    """Return an empty-compatible communication summary."""
    return {
        "episodes": 0,
        "total_steps": 0,
        "communication_steps": 0,
        "communication_ratio": 0.0,
        "total_accepts": 0,
        "total_declines": 0,
        "total_alternative_proposals": 0,
        "total_responses": 0,
        "accept_ratio": 0.0,
        "decline_ratio": 0.0,
        "alternative_ratio": 0.0,
        "total_trm_events": 0,
        "trm_per_communication_step": 0.0,
        "episode_breakdown": {},
    }


def run_one_planner(
    planner: Any,
    config: RunnerConfig,
    compute_comm: bool,
) -> dict[str, Any]:
    """Run one planner for one seed and collect summary metrics."""
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    before = snapshot_jsonl_files(out_dir)

    runner = EpisodeRunner(config)
    result = runner.run(planner)
    jsonl_path = find_new_jsonl(out_dir, before)

    episode_summaries = result.get("episode_summaries", [])
    behavior_records = load_behavior_jsonl(str(jsonl_path))
    behavior_summary = summarize_behavior(behavior_records)

    if compute_comm:
        comm_records = load_comm_jsonl(str(jsonl_path))
        comm_summary = summarize_communication(comm_records)
    else:
        comm_summary = zero_comm_summary()

    return {
        "seed": config.seed,
        "jsonl_path": str(jsonl_path),
        "mean_deliveries": mean_episode_metric(episode_summaries, "total_deliveries"),
        "mean_clashes": mean_episode_metric(episode_summaries, "total_clashes"),
        "mean_stucks": mean_episode_metric(episode_summaries, "total_stucks"),
        "mean_agvs_idle_time": mean_episode_metric(episode_summaries, "agvs_idle_time"),
        "mean_pickers_idle_time": mean_episode_metric(episode_summaries, "pickers_idle_time"),
        "mean_agvs_distance_travelled": mean_episode_metric(
            episode_summaries, "agvs_distance_travelled"
        ),
        "mean_pickers_distance_travelled": mean_episode_metric(
            episode_summaries, "pickers_distance_travelled"
        ),
        "episode_summaries": episode_summaries,
        "behavior_summary": behavior_summary,
        "communication_summary": comm_summary,
    }


def aggregate_per_seed(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate comparison metrics across seeds."""
    return {
        "mean_deliveries": _avg(per_seed, lambda item: item["mean_deliveries"]),
        "mean_clashes": _avg(per_seed, lambda item: item["mean_clashes"]),
        "mean_stucks": _avg(per_seed, lambda item: item["mean_stucks"]),
        "mean_agvs_idle_time": _avg(per_seed, lambda item: item["mean_agvs_idle_time"]),
        "mean_pickers_idle_time": _avg(per_seed, lambda item: item["mean_pickers_idle_time"]),
        "mean_agvs_distance_travelled": _avg(
            per_seed, lambda item: item["mean_agvs_distance_travelled"]
        ),
        "mean_pickers_distance_travelled": _avg(
            per_seed, lambda item: item["mean_pickers_distance_travelled"]
        ),
        "avg_pair_switches_per_episode": _avg(
            per_seed,
            lambda item: item["behavior_summary"].get("avg_pair_switches_per_episode", 0.0),
        ),
        "avg_wait_to_load_steps": _avg(
            per_seed,
            lambda item: item["behavior_summary"].get("avg_wait_to_load_steps", 0.0),
        ),
        "communication_ratio": _avg(
            per_seed,
            lambda item: item["communication_summary"].get("communication_ratio", 0.0),
        ),
        "total_trm_events_per_seed": _avg(
            per_seed,
            lambda item: item["communication_summary"].get("total_trm_events", 0.0),
        ),
    }


def _avg(items: list[dict[str, Any]], getter: Any) -> float:
    """Compute the average of a derived value across items."""
    if not items:
        return 0.0
    total = sum(float(getter(item)) for item in items)
    return total / float(len(items))


def sanitize_env_id(env_id: str) -> str:
    """Sanitize an env id for file naming."""
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in env_id)


def main() -> int:
    """Run the comparison study."""
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    non_symbiotic_per_seed: list[dict[str, Any]] = []
    symbiotic_per_seed: list[dict[str, Any]] = []

    for seed in seeds:
        config = build_config(args, seed)
        non_symbiotic_per_seed.append(
            run_one_planner(NonSymbioticPlanner(), config, compute_comm=False)
        )
        symbiotic_per_seed.append(
            run_one_planner(SymbioticPlanner(), config, compute_comm=True)
        )

    comparison = {
        "config": {
            "env_id": args.env_id,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "out_dir": args.out_dir,
            "topk_requests": args.topk_requests,
            "topk_empty": args.topk_empty,
            "topk_goals": args.topk_goals,
            "pickers_to_agvs": bool(args.pickers_to_agvs),
            "block_conflicting_actions": bool(args.block_conflicting_actions),
            "care_for_agents_in_cost": bool(args.care_for_agents_in_cost),
        },
        "seeds": seeds,
        "non_symbiotic": {
            "per_seed": non_symbiotic_per_seed,
            "aggregate": aggregate_per_seed(non_symbiotic_per_seed),
        },
        "symbiotic": {
            "per_seed": symbiotic_per_seed,
            "aggregate": aggregate_per_seed(symbiotic_per_seed),
        },
    }

    output_path = out_dir / f"comparison_{sanitize_env_id(args.env_id)}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, ensure_ascii=True, indent=2)

    print(
        "baseline mean_deliveries: "
        f"{comparison['non_symbiotic']['aggregate']['mean_deliveries']:.3f}"
    )
    print(
        "symbiotic mean_deliveries: "
        f"{comparison['symbiotic']['aggregate']['mean_deliveries']:.3f}"
    )
    print(
        "baseline avg_wait_to_load_steps: "
        f"{comparison['non_symbiotic']['aggregate']['avg_wait_to_load_steps']:.3f}"
    )
    print(
        "symbiotic avg_wait_to_load_steps: "
        f"{comparison['symbiotic']['aggregate']['avg_wait_to_load_steps']:.3f}"
    )
    print(
        "symbiotic communication_ratio: "
        f"{comparison['symbiotic']['aggregate']['communication_ratio']:.3f}"
    )
    print(
        "symbiotic total_trm_events_per_seed: "
        f"{comparison['symbiotic']['aggregate']['total_trm_events_per_seed']:.3f}"
    )
    print(f"comparison_json: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
