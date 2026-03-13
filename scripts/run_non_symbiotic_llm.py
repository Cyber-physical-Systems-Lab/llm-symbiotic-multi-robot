"""CLI entrypoint for running the non-symbiotic LLM TA-RWARE planner."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import tarware  # noqa: F401

from symco.planners.non_symbiotic_llm_planner import NonSymbioticLLMPlanner
from symco.run.runner import EpisodeRunner, RunnerConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the non-symbiotic LLM run."""
    parser = argparse.ArgumentParser(
        description="Run the non-symbiotic LLM TA-RWARE planner."
    )
    parser.add_argument("--env_id", type=str, required=True, help="Gymnasium environment id.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum steps per episode.")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Directory for run artifacts.")
    parser.add_argument("--render", action="store_true", help="Render the environment during execution.")
    parser.add_argument("--render_mode", type=str, default="human", help="Render mode passed to env.render.")
    parser.add_argument("--topk_requests", type=int, default=10, help="Top-k requested racks in state.")
    parser.add_argument("--topk_empty", type=int, default=10, help="Top-k empty racks in state.")
    parser.add_argument("--topk_goals", type=int, default=10, help="Top-k goals in state.")
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


def build_config(args: argparse.Namespace) -> RunnerConfig:
    """Create a runner config from CLI arguments."""
    return RunnerConfig(
        env_id=args.env_id,
        num_episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        out_dir=args.out_dir,
        render=args.render,
        render_mode=args.render_mode,
        topk_requests=args.topk_requests,
        topk_empty=args.topk_empty,
        topk_goals=args.topk_goals,
        pickers_to_agvs=bool(args.pickers_to_agvs),
        block_conflicting_actions=bool(args.block_conflicting_actions),
        care_for_agents_in_cost=bool(args.care_for_agents_in_cost),
    )


def find_latest_summary(out_dir: Path) -> Path | None:
    """Return the most recently modified summary JSON in the output directory."""
    summary_files = list(out_dir.glob("*_summary.json"))
    if not summary_files:
        return None
    return max(summary_files, key=lambda path: path.stat().st_mtime)


def mean_metric(episode_summaries: list[dict], key: str) -> float:
    """Compute the mean value of a metric across episode summaries."""
    if not episode_summaries:
        return 0.0
    total = sum(float(summary.get(key, 0.0)) for summary in episode_summaries)
    return total / float(len(episode_summaries))


def main() -> int:
    """Run the non-symbiotic LLM planner from the command line."""
    args = parse_args()
    config = build_config(args)
    runner = EpisodeRunner(config)
    planner = NonSymbioticLLMPlanner()
    result = runner.run(planner)

    out_dir = Path(config.out_dir)
    summary_path = find_latest_summary(out_dir)
    episode_summaries = result.get("episode_summaries", [])

    print(f"summary_json: {summary_path if summary_path is not None else 'not found'}")
    print(f"mean_deliveries: {mean_metric(episode_summaries, 'total_deliveries'):.3f}")
    print(f"mean_clashes: {mean_metric(episode_summaries, 'total_clashes'):.3f}")
    print(f"mean_stucks: {mean_metric(episode_summaries, 'total_stucks'):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
