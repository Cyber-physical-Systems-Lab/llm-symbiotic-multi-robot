"""
CLI entrypoint for running the NonMutualisticCommLLMPlannerV2.

This mirrors the staged communication runners while using the
non-mutualistic-with-partner-awareness planner implementation.
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

from symco.planners.non_mutualistic_comm_llm_planner_v2 import (  # noqa: E402
    NonMutualisticCommLLMPlannerV2,
    NonMutualisticCommLLMPlannerV2Config,
)
from symco.run.runner import EpisodeRunner, RunnerConfig  # noqa: E402


class NonMutualisticEpisodeRunner(EpisodeRunner):
    """Episode runner variant for the non-mutualistic partner-aware planner."""

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
        communication_steps = 0

        total_acks = 0
        total_busies = 0
        total_picker_candidates = 0
        total_zero_candidate_requests = 0

        episode_records: list[dict[str, Any]] = []

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
            communication_steps += int(comm_payload["communication_used"])

            ack_count, busy_count, picker_candidate_total, zero_candidate_requests = (
                self._count_non_mutualistic_stage2(comm_payload["comm_response"])
            )
            total_acks += ack_count
            total_busies += busy_count
            total_picker_candidates += picker_candidate_total
            total_zero_candidate_requests += zero_candidate_requests

            record = {
                "episode_idx": episode_idx,
                "step_idx": steps,
                "planner_schema": "non_mutualistic_partner_aware_comm_llm_v2",
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
                "ack_count": ack_count,
                "busy_count": busy_count,
                "picker_candidate_total": picker_candidate_total,
                "zero_candidate_requests": zero_candidate_requests,
            }
            episode_records.append(record)

            if jsonl_handle is not None:
                jsonl_handle.write(json.dumps(record, ensure_ascii=True) + "\n")

            steps += 1
            done = all(bool(flag) for flag in terminateds)
            if done:
                break

        total_cooperative_waiting_time = self._compute_cooperative_waiting_time(episode_records)
        cooperative_attempts = self._count_cooperative_attempts(episode_records)
        (
            assigned_cooperative_tasks,
            total_assigned_sync_cost,
            total_assigned_eta_gap,
        ) = self._compute_assigned_assignment_quality_metrics(episode_records)

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
            "communication_steps": communication_steps,
            "total_acks": total_acks,
            "total_busies": total_busies,
            "total_picker_candidates": total_picker_candidates,
            "total_zero_candidate_requests": total_zero_candidate_requests,
            "total_cooperative_waiting_time": total_cooperative_waiting_time,
            "cooperative_attempts": cooperative_attempts,
            "avg_cooperative_waiting_time": self._safe_ratio(
                total_cooperative_waiting_time, cooperative_attempts
            ),
            "assigned_cooperative_tasks": assigned_cooperative_tasks,
            "avg_assigned_cooperative_completion_time": self._safe_ratio(
                total_assigned_sync_cost, assigned_cooperative_tasks
            ),
            "avg_assigned_coordination_mismatch": self._safe_ratio(
                total_assigned_eta_gap, assigned_cooperative_tasks
            ),
            "planner_schema": "non_mutualistic_partner_aware_comm_llm_v2",
            "stage2_semantics": "ack_busy_committed_target",
        }

    def _count_non_mutualistic_stage2(self, comm_response: Any) -> tuple[int, int, int, int]:
        """Count ACK/BUSY-style Stage 2 outputs."""
        if isinstance(comm_response, dict):
            response_dict = comm_response
        elif comm_response is not None and hasattr(comm_response, "to_dict") and callable(comm_response.to_dict):
            try:
                response_dict = comm_response.to_dict()
            except Exception:
                response_dict = {}
        else:
            response_dict = {}

        responses = response_dict.get("responses", [])
        if not isinstance(responses, list):
            responses = []

        ack_count = 0
        busy_count = 0
        picker_candidate_total = 0
        zero_candidate_requests = 0

        for item in responses:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status", "BUSY")).upper()
            if status == "ACK":
                ack_count += 1
            else:
                busy_count += 1

            candidate_count = self._safe_int(item.get("picker_candidates_count", 0))
            picker_candidate_total += candidate_count
            if candidate_count == 0:
                zero_candidate_requests += 1

        return ack_count, busy_count, picker_candidate_total, zero_candidate_requests

    def _compute_cooperative_waiting_time(self, records: list[dict[str, Any]]) -> int:
        """
        Compute cumulative cooperative waiting time for one episode.

        Definition:
        - AGV target is a non-goal location
        - AGV has already reached that target location
        - cooperative phase is still unfinished
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
                int(loc_id) for loc_id in state_min.get("goal_ids", []) if self._is_int_like(loc_id)
            }

            for agent in agents:
                if not isinstance(agent, dict):
                    continue
                if str(agent.get("type", "")) != "AGV":
                    continue

                target = self._safe_int(agent.get("target", 0))
                if target == 0 or target in goal_ids:
                    continue

                coords_yx = agent.get("coords_yx")
                target_coords_yx = agent.get("target_coords_yx")

                if not self._same_coords(coords_yx, target_coords_yx):
                    continue

                total_wait_steps += 1

        return total_wait_steps

    def _count_cooperative_attempts(self, records: list[dict[str, Any]]) -> int:
        """
        Count cooperative attempts in one episode.

        A cooperative attempt is counted when an AGV enters a new non-goal target phase.
        """
        attempts = 0
        active_targets_by_agv: dict[int, int] = {}

        for record in records:
            state_min = record.get("state_min", {})
            if not isinstance(state_min, dict):
                continue

            goal_ids = {
                int(loc_id) for loc_id in state_min.get("goal_ids", []) if self._is_int_like(loc_id)
            }
            agents = state_min.get("agents", [])
            if not isinstance(agents, list):
                continue

            seen_agvs_this_step: set[int] = set()

            for agent in agents:
                if not isinstance(agent, dict):
                    continue
                if str(agent.get("type", "")) != "AGV":
                    continue

                agv_id = self._safe_int(agent.get("id", 0))
                target = self._safe_int(agent.get("target", 0))
                seen_agvs_this_step.add(agv_id)

                if target == 0 or target in goal_ids:
                    active_targets_by_agv.pop(agv_id, None)
                    continue

                previous_target = active_targets_by_agv.get(agv_id)
                if previous_target != target:
                    attempts += 1
                    active_targets_by_agv[agv_id] = target

            for agv_id in list(active_targets_by_agv.keys()):
                if agv_id not in seen_agvs_this_step:
                    active_targets_by_agv.pop(agv_id, None)

        return attempts

    def _compute_assigned_assignment_quality_metrics(
        self,
        records: list[dict[str, Any]],
    ) -> tuple[int, float, float]:
        """
        Compute assignment-quality metrics for one episode.

        Returns:
        - assigned_task_count
        - sum_assigned_sync_cost
        - sum_assigned_eta_gap

        Each final assignment is matched back to comm_response.responses[].options
        using:
        - request_id
        - rack_id
        - picker_id
        """
        assigned_task_count = 0
        sum_assigned_sync_cost = 0.0
        sum_assigned_eta_gap = 0.0

        for record in records:
            comm_final_plan = record.get("comm_final_plan")
            if not isinstance(comm_final_plan, dict):
                continue

            assignments = comm_final_plan.get("assignments", [])
            if not isinstance(assignments, list) or not assignments:
                continue

            comm_response = record.get("comm_response")
            if not isinstance(comm_response, dict):
                continue

            responses = comm_response.get("responses", [])
            if not isinstance(responses, list):
                continue

            option_index: dict[tuple[str, int, int], dict[str, Any]] = {}
            for response in responses:
                if not isinstance(response, dict):
                    continue
                request_id = response.get("request_id")
                if not isinstance(request_id, str):
                    continue

                options = response.get("options", [])
                if not isinstance(options, list):
                    continue

                    # unreachable, but safe to leave

                for option in options:
                    if not isinstance(option, dict):
                        continue
                    rack_id = self._safe_int(option.get("rack_id", 0))
                    picker_id = self._safe_int(option.get("picker_id", 0))
                    if rack_id <= 0 or picker_id <= 0:
                        continue
                    option_index[(request_id, rack_id, picker_id)] = option

            for assignment in assignments:
                if not isinstance(assignment, dict):
                    continue

                request_id = assignment.get("request_id")
                if not isinstance(request_id, str):
                    continue

                rack_id = self._safe_int(assignment.get("rack_id", 0))
                picker_id = self._safe_int(assignment.get("picker_id", 0))
                if rack_id <= 0 or picker_id <= 0:
                    continue

                option = option_index.get((request_id, rack_id, picker_id))
                if not isinstance(option, dict):
                    continue

                sync_cost = option.get("sync_cost")
                eta_gap = option.get("eta_gap")
                if not self._is_number_like(sync_cost) or not self._is_number_like(eta_gap):
                    continue

                assigned_task_count += 1
                sum_assigned_sync_cost += float(sync_cost)
                sum_assigned_eta_gap += float(eta_gap)

        return assigned_task_count, sum_assigned_sync_cost, sum_assigned_eta_gap

    @staticmethod
    def _safe_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _safe_ratio(num: float, den: float) -> float:
        if den == 0:
            return 0.0
        return float(num) / float(den)

    @staticmethod
    def _is_int_like(value: Any) -> bool:
        try:
            int(value)
        except (TypeError, ValueError):
            return False
        return True

    @staticmethod
    def _is_number_like(value: Any) -> bool:
        try:
            float(value)
        except (TypeError, ValueError):
            return False
        return True

    def _same_coords(self, a: Any, b: Any) -> bool:
        ayx = self._normalize_coords_yx(a)
        byx = self._normalize_coords_yx(b)
        if ayx is None or byx is None:
            return False
        return ayx == byx

    def _normalize_coords_yx(self, value: Any) -> tuple[int, int] | None:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            if self._is_int_like(value[0]) and self._is_int_like(value[1]):
                return int(value[0]), int(value[1])
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the NonMutualisticCommLLMPlannerV2 (partner-aware, no negotiation, deterministic Stage 3)."
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
        help="Minimum steps between communication rounds.",
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


def build_planner_config(args: argparse.Namespace) -> NonMutualisticCommLLMPlannerV2Config:
    return NonMutualisticCommLLMPlannerV2Config(
        stage1_pool_k=args.stage1_pool_k,
        stage2_picker_options_per_rack=args.stage2_picker_options_per_rack,
        max_requests_per_batch=args.max_requests_per_batch,
        wait_timeout_steps=args.wait_timeout_steps,
        min_recommunication_gap_steps=args.min_recommunication_gap_steps,
        idle_probe_gap_steps=args.idle_probe_gap_steps,
        unique_picker=bool(args.unique_picker),
        unique_rack=bool(args.unique_rack),
        enable_rationale=False,
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
    planner = NonMutualisticCommLLMPlannerV2(planner_config)
    result = runner.run(planner)

    out_dir = Path(runner_config.out_dir)
    summary_path = find_latest_summary(out_dir)
    episode_summaries = result.get("episode_summaries", [])

    print(f"summary_json: {summary_path if summary_path is not None else 'not found'}")
    print(f"mean_deliveries: {mean_metric(episode_summaries, 'total_deliveries'):.3f}")
    print(f"mean_clashes: {mean_metric(episode_summaries, 'total_clashes'):.3f}")
    print(f"mean_stucks: {mean_metric(episode_summaries, 'total_stucks'):.3f}")
    print(f"mean_comm_steps: {mean_metric(episode_summaries, 'communication_steps'):.3f}")
    print(
        "mean_avg_cooperative_waiting_time: "
        f"{mean_metric(episode_summaries, 'avg_cooperative_waiting_time'):.3f}"
    )
    print(
        "mean_avg_assigned_cooperative_completion_time: "
        f"{mean_metric(episode_summaries, 'avg_assigned_cooperative_completion_time'):.3f}"
    )
    print(
        "mean_avg_assigned_coordination_mismatch: "
        f"{mean_metric(episode_summaries, 'avg_assigned_coordination_mismatch'):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())