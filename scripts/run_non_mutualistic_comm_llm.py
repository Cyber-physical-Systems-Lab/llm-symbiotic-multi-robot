"""
CLI entrypoint for running the NonMutualisticCommLLMPlanner.

This mirrors scripts/run_symbiotic_comm_llm.py as closely as possible while
swapping in the non-mutualistic staged LLM baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import tarware  # noqa: F401  # registers gym envs via side-effects

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from symco.planners.non_mutualistic_comm_llm_planner import (  # noqa: E402
    NonMutualisticCommLLMPlanner,
    NonMutualisticCommLLMPlannerConfig,
)
from symco.run.runner import EpisodeRunner, RunnerConfig  # noqa: E402


class NonMutualisticEpisodeRunner(EpisodeRunner):
    """Episode runner variant that tags traces with the non-mutualistic schema."""

    def _run_episode(
        self,
        env: Any,
        planner: Any,
        episode_idx: int,
        jsonl_handle: Any,
    ) -> dict[str, Any]:
        episode_seed = None if self.config.seed is None else self.config.seed + episode_idx
        try:
            env.reset(seed=episode_seed)
        except Exception as exc:
            raise RuntimeError(f"Failed to reset environment for episode {episode_idx}: {exc}") from exc

        total_reward = 0.0
        steps = 0
        total_deliveries = 0
        total_clashes = 0
        total_stucks = 0
        agvs_distance_travelled = 0
        pickers_distance_travelled = 0
        agvs_idle_time = 0
        pickers_idle_time = 0
        total_acks = 0
        total_busies = 0
        communication_steps = 0
        reachable_candidate_count = 0
        total_candidate_count = 0
        stage2_zero_option_requests = 0
        zero_option_unreachable = 0
        zero_option_no_idle_picker = 0

        while steps < self.config.max_steps:
            if self.config.render:
                env.render(self.config.render_mode)

            state = self.state_builder.build(env)
            try:
                raw_actions = planner.plan(state)
            except Exception as exc:
                raise RuntimeError(
                    f"Planner '{planner.__class__.__name__}' failed at episode {episode_idx}, step {steps}: {exc}"
                ) from exc

            num_agents = int(state["meta"]["num_agents"])
            if len(raw_actions) != num_agents:
                raise ValueError(
                    f"Planner returned {len(raw_actions)} actions for {num_agents} agents "
                    f"at episode {episode_idx}, step {steps}."
                )

            actions = self._sanitize_actions(raw_actions, env, episode_idx, steps)

            try:
                _, rewards, terminateds, _, info = env.step(actions)
            except Exception as exc:
                raise RuntimeError(
                    f"Environment step failed at episode {episode_idx}, step {steps}: {exc}"
                ) from exc

            reward_sum = float(sum(float(reward) for reward in rewards))
            total_reward += reward_sum
            total_deliveries += int(info.get("shelf_deliveries", 0))
            total_clashes += int(info.get("clashes", 0))
            total_stucks += int(info.get("stucks", 0))
            agvs_distance_travelled += int(info.get("agvs_distance_travelled", 0))
            pickers_distance_travelled += int(info.get("pickers_distance_travelled", 0))
            agvs_idle_time += int(info.get("agvs_idle_time", 0))
            pickers_idle_time += int(info.get("pickers_idle_time", 0))

            comm_payload = self._extract_comm_payload(planner)
            ack_count, busy_count = self._count_ack_busy_responses(comm_payload["comm_response"])
            total_acks += ack_count
            total_busies += busy_count
            communication_steps += int(comm_payload["communication_used"])

            comm_diagnostics = self._compute_comm_diagnostics(
                state=state,
                comm_request=comm_payload["comm_request"],
                comm_response=comm_payload["comm_response"],
                communication_used=bool(comm_payload["communication_used"]),
            )
            reachable_candidate_count += int(comm_diagnostics["reachable_candidate_count"])
            total_candidate_count += int(comm_diagnostics["total_candidate_count"])
            stage2_zero_option_requests += int(comm_diagnostics["stage2_zero_option_requests"])
            zero_option_unreachable += int(comm_diagnostics["zero_option_unreachable"])
            zero_option_no_idle_picker += int(comm_diagnostics["zero_option_no_idle_picker"])

            if jsonl_handle is not None:
                record = {
                    "episode_idx": episode_idx,
                    "step_idx": steps,
                    "planner_schema": "non_mutualistic_comm_llm_v1",
                    "stage2_semantics": "ack_busy_committed_target",
                    "actions": actions,
                    "info": self._to_jsonable(info),
                    "reward_sum": reward_sum,
                    "state_min": {
                        "agents": state["agents"],
                        "requests_rack_ids_topk": state["requests_rack_ids_topk"],
                        "empty_rack_ids_topk": state["empty_rack_ids_topk"],
                        "goal_ids": state["goal_ids"],
                    },
                    "comm_request": comm_payload["comm_request"],
                    "comm_response": comm_payload["comm_response"],
                    "comm_final_plan": comm_payload["comm_final_plan"],
                    "communication_used": comm_payload["communication_used"],
                    "planner_last_communication_triggered": comm_payload["planner_last_communication_triggered"],
                    "planner_throttled_by_budget": comm_payload["planner_throttled_by_budget"],
                    "planner_has_request": comm_payload["planner_has_request"],
                    "planner_has_response": comm_payload["planner_has_response"],
                    "planner_has_final_plan": comm_payload["planner_has_final_plan"],
                    "comm_diagnostics": comm_diagnostics,
                    "ack_count": ack_count,
                    "busy_count": busy_count,
                }
                jsonl_handle.write(json.dumps(record, ensure_ascii=True) + "\n")

            steps += 1
            done = all(bool(flag) for flag in terminateds)
            if done:
                break

        return {
            "episode_idx": episode_idx,
            "steps": steps,
            "total_reward": total_reward,
            "total_deliveries": total_deliveries,
            "total_clashes": total_clashes,
            "total_stucks": total_stucks,
            "agvs_distance_travelled": agvs_distance_travelled,
            "pickers_distance_travelled": pickers_distance_travelled,
            "agvs_idle_time": agvs_idle_time,
            "pickers_idle_time": pickers_idle_time,
            "total_acks": total_acks,
            "total_busies": total_busies,
            "communication_steps": communication_steps,
            "reachable_candidate_count": reachable_candidate_count,
            "total_candidate_count": total_candidate_count,
            "reachability_rate": (
                float(reachable_candidate_count) / float(total_candidate_count)
                if total_candidate_count > 0
                else 0.0
            ),
            "stage2_zero_option_requests": stage2_zero_option_requests,
            "zero_option_unreachable": zero_option_unreachable,
            "zero_option_no_idle_picker": zero_option_no_idle_picker,
            "planner_schema": "non_mutualistic_comm_llm_v1",
            "stage2_semantics": "ack_busy_committed_target",
        }

    def _count_ack_busy_responses(self, comm_response: Any) -> tuple[int, int]:
        """Count ACK / BUSY in the non-mutualistic Stage 2 schema."""
        if isinstance(comm_response, dict):
            payload = comm_response
        elif comm_response is not None and hasattr(comm_response, "to_dict") and callable(comm_response.to_dict):
            try:
                payload = comm_response.to_dict()
            except Exception:
                payload = {}
        else:
            payload = {}

        responses = payload.get("responses", [])
        if not isinstance(responses, list):
            return 0, 0

        ack_count = 0
        busy_count = 0
        for item in responses:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status", "")).upper()
            if status == "ACK":
                ack_count += 1
            elif status == "BUSY":
                busy_count += 1
        return ack_count, busy_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the NonMutualisticCommLLMPlanner (non-mutualistic comm + LLM, 3-stage)."
    )

    parser.add_argument("--env_id", type=str, required=True, help="Gymnasium environment id.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run.")
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

    parser.add_argument(
        "--stage1_pool_k",
        type=int,
        default=8,
        help="Max candidates per request passed to Stage1 AGV LLM.",
    )
    parser.add_argument(
        "--stage2_picker_options_per_rack",
        type=int,
        default=3,
        help="Top-P picker candidates considered for the committed rack.",
    )
    parser.add_argument(
        "--max_requests_per_batch",
        type=int,
        default=2,
        help="Maximum requests handled per communication mini-batch.",
    )
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
    parser.add_argument(
        "--idle_probe_gap_steps",
        type=int,
        default=3,
        help="Gap between idle-probe communication attempts.",
    )
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
    parser.add_argument(
        "--enable_rationale",
        action="store_true",
        help="Enable rationale generation and propagation.",
    )
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


def build_planner_config(args: argparse.Namespace) -> NonMutualisticCommLLMPlannerConfig:
    return NonMutualisticCommLLMPlannerConfig(
        stage1_pool_k=args.stage1_pool_k,
        stage2_picker_options_per_rack=args.stage2_picker_options_per_rack,
        max_requests_per_batch=args.max_requests_per_batch,
        wait_timeout_steps=args.wait_timeout_steps,
        min_recommunication_gap_steps=args.min_recommunication_gap_steps,
        idle_probe_gap_steps=args.idle_probe_gap_steps,
        unique_picker=bool(args.unique_picker),
        unique_rack=bool(args.unique_rack),
        enable_rationale=args.enable_rationale,
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

    runner = NonMutualisticEpisodeRunner(runner_config)
    planner = NonMutualisticCommLLMPlanner(planner_config)
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