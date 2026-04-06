# /Users/yangliu/master-thesis/llm-symbiotic-multi-robot/scripts/run_symbiotic_comm_llm.py
"""
CLI entrypoint for running the SymbioticCommLLMPlanner (symbiotic comm + LLM, 3-stage).

This script mirrors the repo's other runners:
- Uses EpisodeRunner + RunnerConfig for consistent logging/output.
- Exposes planner-specific knobs with names that match the new experiment design:
  Stage1: pool_k, backups
  Stage2: picker_options_per_rack, max_options_per_request
  Trigger: wait_timeout_steps, min_recommunication_gap_steps
  Constraints: unique_picker, unique_rack
  Debug: debug

Example:
  python scripts/run_symbiotic_comm_llm.py \
    --env_id tarware-small-2agvs-2pickers-partialobs-v1 \
    --episodes 5 --seed 0 --max_steps 500 \
    --out_dir outputs \
    --stage1_pool_k 8 --stage1_backups 2 \
    --stage2_picker_options_per_rack 3 --stage2_max_options_per_request 2 \
    --wait_timeout_steps 40 --min_recommunication_gap_steps 8 \
    --unique_picker 1 --unique_rack 1 \
    --debug
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import tarware  # noqa: F401  # registers gym envs via side-effects

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from symco.planners.symbiotic_comm_llm_planner import (  # noqa: E402
    SymbioticCommLLMPlanner,
    SymbioticCommLLMPlannerConfig,
)
from symco.run.runner import EpisodeRunner, RunnerConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SymbioticCommLLMPlanner (symbiotic comm + LLM, 3-stage)."
    )

    # Runner / environment
    parser.add_argument("--env_id", type=str, required=True, help="Gymnasium environment id.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum steps per episode.")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Directory for run artifacts.")
    parser.add_argument("--render", action="store_true", help="Render the environment during execution.")
    parser.add_argument("--render_mode", type=str, default="human", help="Render mode passed to env.render.")

    # StateBuilder / masks / cost table
    parser.add_argument("--topk_requests", type=int, default=10, help="Top-k requested racks in state.")
    parser.add_argument("--topk_empty", type=int, default=10, help="Top-k empty racks in state.")
    parser.add_argument("--topk_goals", type=int, default=10, help="Top-k goals in state.")
    parser.add_argument(
        "--pickers_to_agvs",
        type=int,
        choices=(0, 1),
        default=1,
        help="Whether picker valid actions follow AGV targets (0/1).",
    )
    parser.add_argument(
        "--block_conflicting_actions",
        type=int,
        choices=(0, 1),
        default=1,
        help="Whether to block conflicting same-type actions (0/1).",
    )
    parser.add_argument(
        "--care_for_agents_in_cost",
        type=int,
        choices=(0, 1),
        default=0,
        help="Whether path costs account for other agents (0/1).",
    )

    # Planner: Stage 1 (AGV proposal)
    parser.add_argument(
        "--stage1_pool_k",
        type=int,
        default=8,
        help="Max candidates per request passed to Stage1 AGV LLM (after mask filtering).",
    )
    parser.add_argument(
        "--stage1_backups",
        type=int,
        default=2,
        help="Max backup racks per request accepted from Stage1 output.",
    )

    # Planner: Stage 2 (Picker feasibility/options)
    parser.add_argument(
        "--stage2_picker_options_per_rack",
        type=int,
        default=3,
        help="Top-P picker options per rack candidate (ranked by eta_picker).",
    )
    parser.add_argument(
        "--stage2_max_options_per_request",
        type=int,
        default=2,
        help="Max joint options per request accepted from Stage2 output.",
    )

    # Planner: Trigger policy
    parser.add_argument(
        "--wait_timeout_steps",
        type=int,
        default=40,
        help="Re-communicate if an active cooperative assignment appears stalled beyond this many steps.",
    )
    parser.add_argument(
        "--min_recommunication_gap_steps",
        type=int,
        default=8,
        help="Minimum steps between communication rounds (anti-spam).",
    )

    # Planner: Global constraints
    parser.add_argument(
        "--unique_picker",
        type=int,
        choices=(0, 1),
        default=1,
        help="Enforce unique picker per step among assigned requests (0/1).",
    )
    parser.add_argument(
        "--unique_rack",
        type=int,
        choices=(0, 1),
        default=1,
        help="Enforce unique rack per step among assigned requests (0/1).",
    )

    # Planner debug
    parser.add_argument("--debug", action="store_true", help="Print prompts and raw/sanitized outputs.")
    return parser.parse_args()


def build_runner_config(args: argparse.Namespace) -> RunnerConfig:
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


def build_planner_config(args: argparse.Namespace) -> SymbioticCommLLMPlannerConfig:
    return SymbioticCommLLMPlannerConfig(
        stage1_pool_k=args.stage1_pool_k,
        stage1_backups=args.stage1_backups,
        stage2_picker_options_per_rack=args.stage2_picker_options_per_rack,
        stage2_max_options_per_request=args.stage2_max_options_per_request,
        wait_timeout_steps=args.wait_timeout_steps,
        min_recommunication_gap_steps=args.min_recommunication_gap_steps,
        unique_picker=bool(args.unique_picker),
        unique_rack=bool(args.unique_rack),
        debug=args.debug,
    )


def find_latest_summary(out_dir: Path) -> Path | None:
    summary_files = list(out_dir.glob("*_summary.json"))
    if not summary_files:
        return None
    return max(summary_files, key=lambda path: path.stat().st_mtime)


def mean_metric(episode_summaries: list[dict], key: str) -> float:
    if not episode_summaries:
        return 0.0
    total = sum(float(summary.get(key, 0.0)) for summary in episode_summaries)
    return total / float(len(episode_summaries))


def main() -> int:
    args = parse_args()
    runner_config = build_runner_config(args)
    planner_config = build_planner_config(args)

    runner = EpisodeRunner(runner_config)
    planner = SymbioticCommLLMPlanner(planner_config)
    result = runner.run(planner)

    out_dir = Path(runner_config.out_dir)
    summary_path = find_latest_summary(out_dir)
    episode_summaries = result.get("episode_summaries", [])

    print(f"summary_json: {summary_path if summary_path is not None else 'not found'}")
    print(f"mean_deliveries: {mean_metric(episode_summaries, 'total_deliveries'):.3f}")
    print(f"mean_clashes: {mean_metric(episode_summaries, 'total_clashes'):.3f}")
    print(f"mean_stucks: {mean_metric(episode_summaries, 'total_stucks'):.3f}")
    print(f"mean_comm_steps: {mean_metric(episode_summaries, 'communication_steps'):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())