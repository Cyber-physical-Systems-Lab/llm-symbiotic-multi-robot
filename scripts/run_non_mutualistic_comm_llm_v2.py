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
        seed: int,
        repeat: int,
        global_episode_idx: int,
        jsonl_handle: Any,
    ) -> dict[str, Any]:
        reset_fn = getattr(planner, "reset", None)
        if callable(reset_fn):
            try:
                reset_fn()
            except Exception as exc:
                raise RuntimeError(
                    f"Planner '{planner.__class__.__name__}' failed to reset "
                    f"for seed {seed}, repeat {repeat}, episode_idx {global_episode_idx}: {exc}"
                ) from exc

        try:
            env.reset(seed=seed)
        except Exception as exc:
            raise RuntimeError(f"Failed to reset environment for seed {seed}, repeat {repeat}: {exc}") from exc

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
        assigned_cooperative_tasks = 0
        total_assigned_sync_cost = 0
        total_assigned_eta_gap = 0

        episode_records: list[dict[str, Any]] = []

        while steps < self.config.max_steps:
            if self.config.render:
                env.render(self.config.render_mode)

            state = self.state_builder.build(env)
            try:
                raw_actions = planner.plan(state)
            except Exception as exc:
                raise RuntimeError(
                    f"Planner '{planner.__class__.__name__}' failed at seed {seed}, repeat {repeat}, step {steps}: {exc}"
                ) from exc

            num_agents = int(state["meta"]["num_agents"])
            if len(raw_actions) != num_agents:
                raise ValueError(
                    f"Planner returned {len(raw_actions)} actions for {num_agents} agents "
                    f"at seed {seed}, repeat {repeat}, step {steps}."
                )

            actions = self._sanitize_actions(raw_actions, env, seed, repeat, steps)

            try:
                _, rewards, terminateds, _, info = env.step(actions)
            except Exception as exc:
                raise RuntimeError(
                    f"Environment step failed at seed {seed}, repeat {repeat}, step {steps}: {exc}"
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

            objective_scores = self._extract_objective_scores(comm_payload["comm_final_plan"])
            if objective_scores is not None:
                assigned_cooperative_tasks += self._safe_nonnegative_int(
                    objective_scores.get("num_assignments", 0)
                )
                total_assigned_sync_cost += self._safe_nonnegative_int(
                    objective_scores.get("sum_sync_cost", 0)
                )
                total_assigned_eta_gap += self._safe_nonnegative_int(
                    objective_scores.get("sum_eta_gap", 0)
                )

            record = {
                "episode_idx": global_episode_idx,
                "seed": seed,
                "repeat": repeat,
                "global_episode_idx": global_episode_idx,
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
                    "location_coords_xy": state.get("location_coords_xy", {}),
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
                "debug_idle_unload_agv_count": getattr(planner, "last_debug_idle_unload_agv_count", 0),
                "debug_empty_rack_count": getattr(planner, "last_debug_empty_rack_count", 0),
                "debug_actionable_unload_agv_count": getattr(planner, "last_debug_actionable_unload_agv_count", 0),
                "debug_suspected_unload_deadlock": getattr(planner, "last_debug_suspected_unload_deadlock", False),
                "debug_idle_unload_on_storage_cell_count": getattr(
                    planner, "last_debug_idle_unload_on_storage_cell_count", 0
                ),
                "debug_idle_unload_on_excluded_cell_count": getattr(
                    planner, "last_debug_idle_unload_on_excluded_cell_count", 0
                ),
                "debug_idle_unload_on_current_cell_not_in_empty_topk_count": getattr(
                    planner, "last_debug_idle_unload_on_current_cell_not_in_empty_topk_count", 0
                ),
                "debug_idle_unload_on_current_cell_invalid_action_count": getattr(
                    planner, "last_debug_idle_unload_on_current_cell_invalid_action_count", 0
                ),
                "debug_idle_unload_on_excluded_cell_examples": getattr(
                    planner, "last_debug_idle_unload_on_excluded_cell_examples", []
                ),
                "debug_self_blocking_unload_recovery": getattr(
                    planner, "last_debug_self_blocking_unload_recovery", False
                ),
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
            completed_assigned_task_count,
            total_assigned_task_completion_time,
        ) = self._compute_assigned_task_completion_metrics(episode_records)

        (
            all_assigned_task_count_for_exec,
            total_assigned_task_execution_time_all,
        ) = self._compute_assigned_task_execution_time_all(episode_records)

        (
            completed_assigned_task_count_for_wait,
            total_assigned_target_wait_time_all,
            total_assigned_target_wait_time_completed,
        ) = self._compute_assigned_task_wait_metrics(episode_records)

        trigger_reason_counts = getattr(planner, "trigger_reason_counts", {})
        if not isinstance(trigger_reason_counts, dict):
            trigger_reason_counts = {}
        trigger_reason_steps = getattr(planner, "trigger_reason_steps", [])
        if not isinstance(trigger_reason_steps, list):
            trigger_reason_steps = []

        print(
            "EPISODE_TRIGGER_REASON_COUNTS\n",
            json.dumps(
                {
                    "episode_idx": global_episode_idx,
                    "seed": seed,
                    "repeat": repeat,
                    "global_episode_idx": global_episode_idx,
                    "counts": trigger_reason_counts,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        print(
            "EPISODE_TRIGGER_REASON_STEPS_TAIL\n",
            json.dumps(
                {
                    "episode_idx": global_episode_idx,
                    "seed": seed,
                    "repeat": repeat,
                    "global_episode_idx": global_episode_idx,
                    "num_trigger_steps": int(len(trigger_reason_steps)),
                    "steps_tail": trigger_reason_steps[-8:],
                },
                ensure_ascii=False,
                indent=2,
            ),
        )

        return {
            "episode_idx": global_episode_idx,
            "seed": seed,
            "repeat": repeat,
            "global_episode_idx": global_episode_idx,
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
            "suspected_unload_deadlock_steps": int(
                getattr(planner, "suspected_unload_deadlock_steps", 0)
            ),
            "suspected_unload_deadlock_ge10_events": int(
                getattr(planner, "suspected_unload_deadlock_ge10_events", 0)
            ),
            "idle_unload_on_excluded_cell_steps": int(
                getattr(planner, "idle_unload_on_excluded_cell_steps", 0)
            ),
            "idle_unload_on_excluded_cell_ge10_events": int(
                getattr(planner, "idle_unload_on_excluded_cell_ge10_events", 0)
            ),
            "self_blocking_unload_recovery_steps": int(
                getattr(planner, "self_blocking_unload_recovery_steps", 0)
            ),
            "self_blocking_unload_recovery_with_existing_picker": int(
                getattr(planner, "self_blocking_unload_recovery_with_existing_picker", 0)
            ),
            "self_blocking_unload_recovery_via_comm": int(
                getattr(planner, "self_blocking_unload_recovery_via_comm", 0)
            ),
            "assigned_cooperative_tasks": assigned_cooperative_tasks,
            "avg_planned_sync_cost": self._safe_ratio(
                total_assigned_sync_cost, assigned_cooperative_tasks
            ),
            "avg_agv_picker_arrival_gap": self._safe_ratio(
                total_assigned_eta_gap, assigned_cooperative_tasks
            ),
            "avg_execution_time_per_assignment": self._safe_ratio(
                total_assigned_task_completion_time, completed_assigned_task_count
            ),
            "avg_execution_time_all": self._safe_ratio(
                total_assigned_task_execution_time_all, all_assigned_task_count_for_exec
            ),
            "avg_wait_time_all_assignments": self._safe_ratio(
                total_assigned_target_wait_time_all, assigned_cooperative_tasks
            ),
            "avg_wait_time_after_first_arrival": self._safe_ratio(
                total_assigned_target_wait_time_completed, completed_assigned_task_count_for_wait
            ),
            "assignment_success_rate": self._safe_ratio(
                completed_assigned_task_count_for_wait, assigned_cooperative_tasks
            ),
            "planner_schema": "non_mutualistic_partner_aware_comm_llm_v2",
            "stage2_semantics": "ack_busy_committed_target",
        }

    def _sanitize_actions(
        self,
        actions: list[Any],
        env: Any,
        seed: int,
        repeat: int,
        step_idx: int,
    ) -> list[int]:
        """Convert planner output to safe integer macro actions."""
        action_size = int(getattr(env.unwrapped, "action_size", 0))
        sanitized: list[int] = []

        for agent_idx, action in enumerate(actions):
            safe_action = 0
            try:
                candidate = int(action)
            except (TypeError, ValueError):
                import logging
                logging.warning(
                    "Invalid non-integer action for agent %s at seed %s repeat %s step %s: %r. Using 0.",
                    agent_idx + 1,
                    seed,
                    repeat,
                    step_idx,
                    action,
                )
                sanitized.append(0)
                continue

            if 0 <= candidate < action_size:
                safe_action = candidate
            else:
                import logging
                logging.warning(
                    "Out-of-range action for agent %s at seed %s repeat %s step %s: %s not in [0, %s). Using 0.",
                    agent_idx + 1,
                    seed,
                    repeat,
                    step_idx,
                    candidate,
                    action_size,
                )
            sanitized.append(safe_action)

        return sanitized

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
    parser.add_argument("--num_seeds", type=int, default=10, help="Number of different random seeds.")
    parser.add_argument("--repeats_per_seed", type=int, default=1, help="Number of repeats per seed.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed (seeds will be base, base+1, ...).")
    parser.add_argument("--max_steps", type=int, default=1500, help="Maximum steps per episode.")
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
        default=3,
        help="Maximum requests handled per communication mini-batch.",
    )
    parser.add_argument(
        "--idle_probe_gap_steps",
        type=int,
        default=25,
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
        num_seeds=args.num_seeds,
        repeats_per_seed=args.repeats_per_seed,
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
    print(f"mean_total_deliveries: {mean_metric(episode_summaries, 'total_deliveries'):.3f}")
    print(f"mean_clashes: {mean_metric(episode_summaries, 'total_clashes'):.3f}")
    print(f"mean_stucks: {mean_metric(episode_summaries, 'total_stucks'):.3f}")
    print(f"mean_comm_steps: {mean_metric(episode_summaries, 'communication_steps'):.3f}")
    print(
        "mean_avg_cooperative_waiting_time: "
        f"{mean_metric(episode_summaries, 'avg_cooperative_waiting_time'):.3f}"
    )
    print(
        "mean_avg_planned_sync_cost: "
        f"{mean_metric(episode_summaries, 'avg_planned_sync_cost'):.3f}"
    )
    print(
        "mean_avg_agv_picker_arrival_gap: "
        f"{mean_metric(episode_summaries, 'avg_agv_picker_arrival_gap'):.3f}"
    )
    print(
        "mean_avg_execution_time_per_assignment: "
        f"{mean_metric(episode_summaries, 'avg_execution_time_per_assignment'):.3f}"
    )
    print(
        "mean_avg_execution_time_all: "
        f"{mean_metric(episode_summaries, 'avg_execution_time_all'):.3f}"
    )
    print(
        "mean_avg_wait_time_all_assignments: "
        f"{mean_metric(episode_summaries, 'avg_wait_time_all_assignments'):.3f}"
    )
    print(
        "mean_avg_wait_time_after_first_arrival: "
        f"{mean_metric(episode_summaries, 'avg_wait_time_after_first_arrival'):.3f}"
    )
    print(
        "mean_assignment_success_rate: "
        f"{mean_metric(episode_summaries, 'assignment_success_rate'):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
