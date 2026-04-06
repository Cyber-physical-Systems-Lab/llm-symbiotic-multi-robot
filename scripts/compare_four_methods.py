"""Compare four TA-RWARE planners on the same environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import tarware  # noqa: F401

from symco.eval.behavior_metrics import load_jsonl as load_behavior_jsonl, summarize_behavior
from symco.eval.comm_metrics import load_jsonl as load_comm_jsonl, summarize_communication
from symco.planners.non_symbiotic_llm_planner import NonSymbioticLLMPlanner
from symco.planners.non_symbiotic_planner import NonSymbioticPlanner
from symco.planners.symbiotic_llm_planner import SymbioticLLMPlanner
from symco.planners.symbiotic_planner import SymbioticPlanner
from symco.planners.symbiotic_llm_planner_v2 import SymbioticLLMPlannerV2
from symco.run.runner import EpisodeRunner, RunnerConfig


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for four-method comparison."""
    parser = argparse.ArgumentParser(
        description="Compare four TA-RWARE planners on the same environment."
    )
    parser.add_argument("--env_id", required=True, help="Gymnasium environment id.")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per seed.")
    parser.add_argument("--seeds", type=str, default="0,1", help="Comma-separated seed list.")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum steps per episode.")
    parser.add_argument("--out_dir", type=str, default="outputs_compare_four", help="Output directory.")
    parser.add_argument("--topk_requests", type=int, default=10, help="Top-k requested racks.")
    parser.add_argument("--topk_empty", type=int, default=10, help="Top-k empty racks.")
    parser.add_argument("--topk_goals", type=int, default=10, help="Top-k goals.")
    parser.add_argument(
        "--pickers_to_agvs",
        type=int,
        default=1,
        help="Whether picker valid actions follow AGV targets (0 or 1).",
    )
    parser.add_argument(
        "--block_conflicting_actions",
        type=int,
        default=1,
        help="Whether to block conflicting same-type actions (0 or 1).",
    )
    parser.add_argument(
        "--care_for_agents_in_cost",
        type=int,
        default=0,
        help="Whether path costs account for other agents (0 or 1).",
    )
    return parser.parse_args()


def parse_seeds(text: str) -> list[int]:
    """Parse a comma-separated seed string."""
    seeds = [int(part.strip()) for part in text.split(",") if part.strip()]
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


def mean_episode_metric(episode_summaries: list[dict[str, Any]], key: str) -> float:
    """Compute a mean metric across episode summaries."""
    if not episode_summaries:
        return 0.0
    total = sum(float(item.get(key, 0.0)) for item in episode_summaries)
    return total / float(len(episode_summaries))


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


def zero_comm() -> dict[str, float]:
    """Return zero communication metrics for non-symbiotic methods."""
    return {
        "communication_ratio": 0.0,
        "total_trm_events": 0.0,
    }


def run_one_planner(planner: Any, config: RunnerConfig) -> dict[str, Any]:
    """Run one planner for one seed and collect core metrics."""
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_jsonl_files(out_dir)

    runner = EpisodeRunner(config)
    result = runner.run(planner)
    episode_summaries = result.get("episode_summaries", [])
    jsonl_path = find_new_jsonl(out_dir, snapshot)

    behavior_records = load_behavior_jsonl(str(jsonl_path))
    behavior = summarize_behavior(behavior_records)
    avg_wait = float(behavior.get("avg_wait_to_load_steps", 0.0))

    planner_name = planner.__class__.__name__
    if planner_name in {"SymbioticPlanner", "SymbioticLLMPlanner", "SymbioticLLMPlannerV2"} or any(
        hasattr(planner, attr) for attr in ("last_request", "last_response", "last_final_plan")
    ):
        comm_records = load_comm_jsonl(str(jsonl_path))
        comm = summarize_communication(comm_records)
    else:
        comm = zero_comm()

    return {
        "seed": config.seed,
        "mean_deliveries": mean_episode_metric(episode_summaries, "total_deliveries"),
        "mean_clashes": mean_episode_metric(episode_summaries, "total_clashes"),
        "mean_stucks": mean_episode_metric(episode_summaries, "total_stucks"),
        "avg_wait_to_load_steps": avg_wait,
        "communication_ratio": float(comm.get("communication_ratio", 0.0)),
        "total_trm_events": float(comm.get("total_trm_events", 0.0)),
    }


def aggregate_per_seed(per_seed: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate core metrics across seeds."""
    return {
        "mean_deliveries": _avg(per_seed, "mean_deliveries"),
        "mean_clashes": _avg(per_seed, "mean_clashes"),
        "mean_stucks": _avg(per_seed, "mean_stucks"),
        "avg_wait_to_load_steps": _avg(per_seed, "avg_wait_to_load_steps"),
        "communication_ratio": _avg(per_seed, "communication_ratio"),
        "total_trm_events_per_seed": _avg(per_seed, "total_trm_events"),
    }


def _avg(items: list[dict[str, Any]], key: str) -> float:
    """Compute the average value for a metric key."""
    if not items:
        return 0.0
    return sum(float(item.get(key, 0.0)) for item in items) / float(len(items))


def sanitize_env_id(env_id: str) -> str:
    """Sanitize an env id for file naming."""
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in env_id)


def main() -> int:
    """Run the four-method comparison study."""
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    planner_specs = [
        ("NonSymbioticPlanner", NonSymbioticPlanner),
        ("SymbioticPlanner", SymbioticPlanner),
        ("NonSymbioticLLMPlanner", NonSymbioticLLMPlanner),
        ("SymbioticLLMPlanner", SymbioticLLMPlanner),
        ("SymbioticLLMPlannerV2", SymbioticLLMPlannerV2),
    ]

    comparison: dict[str, Any] = {
        "config": {
            "env_id": args.env_id,
            "episodes": args.episodes,
            "seeds": seeds,
            "max_steps": args.max_steps,
            "out_dir": args.out_dir,
            "topk_requests": args.topk_requests,
            "topk_empty": args.topk_empty,
            "topk_goals": args.topk_goals,
            "pickers_to_agvs": bool(args.pickers_to_agvs),
            "block_conflicting_actions": bool(args.block_conflicting_actions),
            "care_for_agents_in_cost": bool(args.care_for_agents_in_cost),
        },
        "methods": {},
    }

    for planner_name, planner_cls in planner_specs:
        per_seed: list[dict[str, Any]] = []
        for seed in seeds:
            config = build_config(args, seed)
            per_seed.append(run_one_planner(planner_cls(), config))

        comparison["methods"][planner_name] = {
            "per_seed": per_seed,
            "aggregate": aggregate_per_seed(per_seed),
        }

    output_path = out_dir / f"comparison_{sanitize_env_id(args.env_id)}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, ensure_ascii=True, indent=2)

    print("planner | deliveries | wait_to_load | comm_ratio | trm_per_seed | clashes | stucks")
    for planner_name, _ in planner_specs:
        aggregate = comparison["methods"][planner_name]["aggregate"]
        print(
            f"{planner_name} | "
            f"{aggregate['mean_deliveries']:.3f} | "
            f"{aggregate['avg_wait_to_load_steps']:.3f} | "
            f"{aggregate['communication_ratio']:.3f} | "
            f"{aggregate['total_trm_events_per_seed']:.3f} | "
            f"{aggregate['mean_clashes']:.3f} | "
            f"{aggregate['mean_stucks']:.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
