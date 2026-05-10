"""Episode runner for TA-RWARE coordination experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any

import gymnasium as gym
import tarware  # noqa: F401  # triggers gym.register side-effects

from symco.core.state_builder import StateBuilder, StateConfig


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RunnerConfig:
    """Configuration for experiment execution and logging."""

    env_id: str
    max_steps: int = 500
    seed: int | None = None
    num_seeds: int = 10           # 不同随机种子的数量（原 num_episodes）
    repeats_per_seed: int = 1     # 每个种子重复运行的次数（新增）
    render: bool = False
    render_mode: str = "human"
    out_dir: str = "outputs"
    save_jsonl: bool = True
    save_episode_summary: bool = True
    pickers_to_agvs: bool = True
    block_conflicting_actions: bool = True
    topk_requests: int = 10
    topk_empty: int = 10
    topk_goals: int = 10
    care_for_agents_in_cost: bool = False


class EpisodeRunner:
    """Run one or more episodes with a planner against a TA-RWARE environment."""

    def __init__(self, config: RunnerConfig):
        self.config = config
        self.state_builder = StateBuilder(
            StateConfig(
                topk_requests=config.topk_requests,
                topk_empty=config.topk_empty,
                topk_goals=config.topk_goals,
                pickers_to_agvs=config.pickers_to_agvs,
                block_conflicting_actions=config.block_conflicting_actions,
                care_for_agents_in_cost=config.care_for_agents_in_cost,
            )
        )

    def make_env(self) -> gym.Env:
        try:
            if str(self.config.env_id).startswith("tarware-"):
                import tarware  # noqa: F401
            return gym.make(self.config.env_id, max_steps=self.config.max_steps)
        except Exception as exc:
            raise RuntimeError(f"Failed to create environment '{self.config.env_id}': {exc}") from exc

    def run(self, planner: Any) -> dict[str, Any]:
        """Run the configured number of episodes and collect summaries."""
        planner_name = planner.__class__.__name__
        out_dir = Path(self.config.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        run_prefix = self._build_run_prefix(planner_name)
        jsonl_path = out_dir / f"{run_prefix}.jsonl"
        summary_path = out_dir / f"{run_prefix}_summary.json"
        episode_summaries: list[dict[str, Any]] = []

        jsonl_handle = None
        if self.config.save_jsonl:
            jsonl_handle = jsonl_path.open("w", encoding="utf-8")

        env = self.make_env()
        global_episode_counter = 0

        try:
            base_seed = self.config.seed if self.config.seed is not None else 0
            seeds = [base_seed + i for i in range(self.config.num_seeds)]

            for seed in seeds:
                for repeat in range(self.config.repeats_per_seed):
                    summary = self._run_episode(
                        env=env,
                        planner=planner,
                        seed=seed,
                        repeat=repeat,
                        global_episode_idx=global_episode_counter,
                        jsonl_handle=jsonl_handle,
                    )
                    episode_summaries.append(summary)
                    global_episode_counter += 1
        finally:
            if jsonl_handle is not None:
                jsonl_handle.close()
            env.close()

        result = {
            "episode_summaries": episode_summaries,
            "config": asdict(self.config),
        }

        if self.config.save_episode_summary:
            with summary_path.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=True, indent=2)

        return result

    def _run_episode(
        self,
        env: gym.Env,
        planner: Any,
        seed: int,
        repeat: int,
        global_episode_idx: int,
        jsonl_handle: Any,
    ) -> dict[str, Any]:
        """Execute a single episode and return aggregated metrics."""

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
        total_recommends = 0
        total_declines = 0
        total_multi_option_recommends = 0
        communication_steps = 0
        reachable_candidate_count = 0
        total_candidate_count = 0
        stage2_zero_option_requests = 0
        zero_option_unreachable = 0
        zero_option_no_idle_picker = 0

        # Assignment-level quality metrics accumulated from comm_final_plan.objective_scores
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

            recommend_count, decline_count, multi_option_recommend_count = self._count_response_decisions(
                comm_payload["comm_response"]
            )
            total_recommends += recommend_count
            total_declines += decline_count
            total_multi_option_recommends += multi_option_recommend_count
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

            objective_scores = self._extract_objective_scores(comm_payload["comm_final_plan"])
            if objective_scores is not None:
                assigned_cooperative_tasks += self._safe_nonnegative_int(objective_scores.get("num_assignments", 0))
                total_assigned_sync_cost += self._safe_nonnegative_int(objective_scores.get("sum_sync_cost", 0))
                total_assigned_eta_gap += self._safe_nonnegative_int(objective_scores.get("sum_eta_gap", 0))

            record = {
                "episode_idx": global_episode_idx,
                "seed": seed,
                "repeat": repeat,
                "global_episode_idx": global_episode_idx,
                "step_idx": steps,
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
                "comm_diagnostics": comm_diagnostics,
                "recommend_count": recommend_count,
                "decline_count": decline_count,
                "multi_option_recommend_count": multi_option_recommend_count,
            }
            episode_records.append(record)

            if jsonl_handle is not None:
                jsonl_handle.write(json.dumps(record, ensure_ascii=True) + "\n")

            steps += 1
            done = all(bool(flag) for flag in terminateds)
            if done:
                break

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
            "total_recommends": total_recommends,
            "total_declines": total_declines,
            "total_multi_option_recommends": total_multi_option_recommends,
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
            "assigned_cooperative_tasks": assigned_cooperative_tasks,
            "avg_planned_sync_cost": (
                float(total_assigned_sync_cost) / float(assigned_cooperative_tasks)
                if assigned_cooperative_tasks > 0
                else 0.0
            ),
            "avg_agv_picker_arrival_gap": (
                float(total_assigned_eta_gap) / float(assigned_cooperative_tasks)
                if assigned_cooperative_tasks > 0
                else 0.0
            ),
            "avg_execution_time_per_assignment": (
                float(total_assigned_task_completion_time) / float(completed_assigned_task_count)
                if completed_assigned_task_count > 0
                else 0.0
            ),
            "avg_execution_time_all": (
                float(total_assigned_task_execution_time_all) / float(all_assigned_task_count_for_exec)
                if all_assigned_task_count_for_exec > 0
                else 0.0
            ),
            "avg_wait_time_all_assignments": (
                float(total_assigned_target_wait_time_all) / float(assigned_cooperative_tasks)
                if assigned_cooperative_tasks > 0
                else 0.0
            ),
            "avg_wait_time_after_first_arrival": (
                float(total_assigned_target_wait_time_completed) / float(completed_assigned_task_count_for_wait)
                if completed_assigned_task_count_for_wait > 0
                else 0.0
            ),
            "assignment_success_rate": (
                float(completed_assigned_task_count_for_wait) / float(assigned_cooperative_tasks)
                if assigned_cooperative_tasks > 0
                else 0.0
            ),
        }

    def _sanitize_actions(
        self,
        actions: list[Any],
        env: gym.Env,
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
                LOGGER.warning(
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
                LOGGER.warning(
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

    def _build_run_prefix(self, planner_name: str) -> str:
        """Create the common filename prefix for run artifacts."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        env_name = self.config.env_id.replace("/", "_").replace(":", "_")
        env_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in env_name)
        planner_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in planner_name)
        repeats_str = f"repeats{self.config.repeats_per_seed}" if self.config.repeats_per_seed > 1 else ""
        seed_range_str = f"seeds{self.config.num_seeds}"
        if repeats_str:
            return f"{timestamp}_{env_name}_{planner_name}_{seed_range_str}_{repeats_str}"
        return f"{timestamp}_{env_name}_{planner_name}_{seed_range_str}"

    def _extract_comm_payload(self, planner: Any) -> dict[str, Any]:
        """Extract optional communication artifacts from a planner."""
        comm_request = self._maybe_to_dict(getattr(planner, "last_request", None))
        comm_response = self._maybe_to_dict(getattr(planner, "last_response", None))
        comm_final_plan = self._maybe_to_dict(getattr(planner, "last_final_plan", None))
        planner_last_communication_triggered = bool(
            getattr(planner, "last_communication_triggered", False)
        )
        planner_throttled_by_budget = bool(
            getattr(planner, "planner_throttled_by_budget", False)
        )
        return {
            "comm_request": comm_request,
            "comm_response": comm_response,
            "comm_final_plan": comm_final_plan,
            "communication_used": planner_last_communication_triggered,
            "planner_last_communication_triggered": planner_last_communication_triggered,
            "planner_throttled_by_budget": planner_throttled_by_budget,
            "planner_has_request": comm_request is not None,
            "planner_has_response": comm_response is not None,
            "planner_has_final_plan": comm_final_plan is not None,
        }

    def _count_response_decisions(self, comm_response_dict: Any) -> tuple[int, int, int]:
        """Count RECOMMEND / DECLINE decisions and multi-option recommends."""
        if not isinstance(comm_response_dict, dict):
            return 0, 0, 0

        responses = comm_response_dict.get("responses", [])
        if not isinstance(responses, list):
            return 0, 0, 0

        recommend_count = 0
        decline_count = 0
        multi_option_recommend_count = 0

        for item in responses:
            if not isinstance(item, dict):
                continue
            decision = str(item.get("decision", "")).upper()
            options = item.get("options", [])
            if not isinstance(options, list):
                options = []

            if decision == "RECOMMEND":
                recommend_count += 1
                if len(options) >= 2:
                    multi_option_recommend_count += 1
            elif decision == "DECLINE":
                decline_count += 1

        return recommend_count, decline_count, multi_option_recommend_count

    def _compute_comm_diagnostics(
        self,
        state: dict[str, Any],
        comm_request: Any,
        comm_response: Any,
        communication_used: bool,
    ) -> dict[str, Any]:
        """Compute non-invasive communication diagnostics for one environment step."""
        if not communication_used:
            return {
                "reachable_candidate_count": 0,
                "total_candidate_count": 0,
                "reachability_rate": 0.0,
                "stage2_zero_option_requests": 0,
                "zero_option_unreachable": 0,
                "zero_option_no_idle_picker": 0,
            }

        requests = comm_request.get("requests", []) if isinstance(comm_request, dict) else []
        responses = comm_response.get("responses", []) if isinstance(comm_response, dict) else []
        if not isinstance(requests, list):
            requests = []
        if not isinstance(responses, list):
            responses = []

        picker_cost_table = state.get("cost_table", {}).get("picker", {})
        agents = state.get("agents", [])
        all_picker_ids = [
            int(agent["id"])
            for agent in agents
            if isinstance(agent, dict) and agent.get("type") == "PICKER" and agent.get("id") is not None
        ]
        idle_picker_ids = [
            int(agent["id"])
            for agent in agents
            if isinstance(agent, dict)
            and agent.get("type") == "PICKER"
            and agent.get("id") is not None
            and not bool(agent.get("busy", False))
        ]

        reachable_candidate_count = 0
        total_candidate_count = 0
        request_has_idle_reachable: dict[str, bool] = {}
        for request in requests:
            if not isinstance(request, dict):
                continue
            request_id = request.get("request_id")
            if not isinstance(request_id, str):
                continue

            idle_reachable = False
            for candidate in request.get("candidates", []):
                if not isinstance(candidate, dict):
                    continue
                rack_id = self._safe_int(candidate.get("rack_id"))
                if rack_id <= 0:
                    continue
                total_candidate_count += 1
                if self._candidate_reachable_by_any_picker(picker_cost_table, all_picker_ids, rack_id):
                    reachable_candidate_count += 1
                if self._candidate_reachable_by_any_picker(picker_cost_table, idle_picker_ids, rack_id):
                    idle_reachable = True
            request_has_idle_reachable[request_id] = idle_reachable

        stage2_zero_option_requests = 0
        zero_option_unreachable = 0
        zero_option_no_idle_picker = 0
        for response in responses:
            if not isinstance(response, dict):
                continue
            request_id = response.get("request_id")
            if not isinstance(request_id, str):
                continue
            options = response.get("options", [])
            if not isinstance(options, list):
                options = []
            if len(options) != 0:
                continue
            stage2_zero_option_requests += 1
            if not idle_picker_ids:
                zero_option_no_idle_picker += 1
            elif not request_has_idle_reachable.get(request_id, False):
                zero_option_unreachable += 1

        return {
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
        }

    def _candidate_reachable_by_any_picker(
        self,
        picker_cost_table: Any,
        picker_ids: list[int],
        rack_id: int,
    ) -> bool:
        """Return whether any listed picker has a finite cost entry to the rack."""
        rack_key = str(int(rack_id))
        if not isinstance(picker_cost_table, dict):
            return False
        for picker_id in picker_ids:
            picker_cost_map = picker_cost_table.get(str(int(picker_id)), {})
            if isinstance(picker_cost_map, dict) and rack_key in picker_cost_map:
                return True
        return False

    def _extract_objective_scores(self, comm_final_plan: Any) -> dict[str, Any] | None:
        """Extract objective_scores from a communication final plan if present."""
        if not isinstance(comm_final_plan, dict):
            return None
        scores = comm_final_plan.get("objective_scores")
        if not isinstance(scores, dict):
            return None
        return scores

    def _compute_assigned_task_completion_metrics(
        self,
        records: list[dict[str, Any]],
    ) -> tuple[int, int]:
        """Compute completed assigned-task durations from per-step records."""
        active_tasks_by_agv: dict[int, dict[str, Any]] = {}
        completed_task_count = 0
        total_completion_time = 0

        for idx, record in enumerate(records):
            step_idx = self._safe_nonnegative_int(record.get("step_idx", idx))
            carrying_by_agv = self._extract_agv_carrying_by_id(record.get("state_min"))

            for agv_id, task in list(active_tasks_by_agv.items()):
                carrying = carrying_by_agv.get(int(agv_id))
                if carrying is None:
                    continue
                purpose = str(task.get("purpose", "")).upper()
                completed = (
                    (purpose == "LOAD" and carrying)
                    or (purpose == "UNLOAD" and not carrying)
                )
                if not completed:
                    continue
                duration = int(step_idx) - int(task.get("start_step", step_idx))
                if duration >= 0:
                    completed_task_count += 1
                    total_completion_time += int(duration)
                active_tasks_by_agv.pop(int(agv_id), None)

            comm_final_plan = record.get("comm_final_plan")
            if not isinstance(comm_final_plan, dict):
                continue
            assignments = comm_final_plan.get("assignments", [])
            if not isinstance(assignments, list):
                continue
            purpose_by_request = self._extract_request_purpose_map(record.get("comm_request"))

            for assignment in assignments:
                if not isinstance(assignment, dict):
                    continue
                agv_id = self._safe_int(assignment.get("agv_id", 0))
                picker_id = self._safe_int(assignment.get("picker_id", 0))
                rack_id = self._safe_int(assignment.get("rack_id", 0))
                request_id = assignment.get("request_id")
                if agv_id <= 0 or picker_id <= 0 or rack_id <= 0 or not isinstance(request_id, str):
                    continue

                purpose = str(purpose_by_request.get(request_id, "")).upper()
                if purpose not in {"LOAD", "UNLOAD"}:
                    carrying = carrying_by_agv.get(int(agv_id))
                    if carrying is None:
                        continue
                    purpose = "UNLOAD" if carrying else "LOAD"

                signature = (str(request_id), int(picker_id), int(rack_id), str(purpose))
                current_task = active_tasks_by_agv.get(int(agv_id))
                if current_task is not None and current_task.get("signature") == signature:
                    continue

                active_tasks_by_agv[int(agv_id)] = {
                    "signature": signature,
                    "purpose": str(purpose),
                    "start_step": int(step_idx),
                }

        return completed_task_count, total_completion_time

    def _compute_assigned_task_execution_time_all(
        self,
        records: list[dict[str, Any]],
    ) -> tuple[int, int]:
        """
        Compute execution time over all assigned cooperative tasks.

        Completed tasks are finalized when the AGV carrying state changes:
        - LOAD completes when AGV becomes carrying=True
        - UNLOAD completes when AGV becomes carrying=False

        Unfinished tasks are finalized at episode end, so long-running assignments
        are included in the average.
        """
        active_tasks_by_agv: dict[int, dict[str, Any]] = {}
        finalized_task_count = 0
        total_execution_time_all = 0

        if not records:
            return 0, 0

        final_step = self._safe_nonnegative_int(records[-1].get("step_idx", len(records) - 1)) + 1

        def finalize_task(agv_id: int, end_step: int) -> None:
            nonlocal finalized_task_count, total_execution_time_all
            task = active_tasks_by_agv.pop(int(agv_id), None)
            if task is None:
                return
            start_step = int(task.get("start_step", end_step))
            duration = int(end_step) - int(start_step)
            if duration < 0:
                duration = 0
            finalized_task_count += 1
            total_execution_time_all += int(duration)

        for idx, record in enumerate(records):
            step_idx = self._safe_nonnegative_int(record.get("step_idx", idx))
            carrying_by_agv = self._extract_agv_carrying_by_id(record.get("state_min"))

            for agv_id, task in list(active_tasks_by_agv.items()):
                carrying = carrying_by_agv.get(int(agv_id))
                if carrying is None:
                    continue
                purpose = str(task.get("purpose", "")).upper()
                completed = (
                    (purpose == "LOAD" and carrying)
                    or (purpose == "UNLOAD" and not carrying)
                )
                if completed:
                    finalize_task(int(agv_id), int(step_idx))

            comm_final_plan = record.get("comm_final_plan")
            if not isinstance(comm_final_plan, dict):
                continue

            assignments = comm_final_plan.get("assignments", [])
            if not isinstance(assignments, list):
                continue

            purpose_by_request = self._extract_request_purpose_map(record.get("comm_request"))

            for assignment in assignments:
                if not isinstance(assignment, dict):
                    continue

                agv_id = self._safe_int(assignment.get("agv_id", 0))
                picker_id = self._safe_int(assignment.get("picker_id", 0))
                rack_id = self._safe_int(assignment.get("rack_id", 0))
                request_id = assignment.get("request_id")

                if agv_id <= 0 or picker_id <= 0 or rack_id <= 0 or not isinstance(request_id, str):
                    continue

                purpose = str(purpose_by_request.get(request_id, "")).upper()
                if purpose not in {"LOAD", "UNLOAD"}:
                    carrying = carrying_by_agv.get(int(agv_id))
                    if carrying is None:
                        continue
                    purpose = "UNLOAD" if carrying else "LOAD"

                signature = (str(request_id), int(picker_id), int(rack_id), str(purpose))
                existing_task = active_tasks_by_agv.get(int(agv_id))

                if existing_task is not None and existing_task.get("signature") == signature:
                    continue

                if existing_task is not None and existing_task.get("signature") != signature:
                    finalize_task(int(agv_id), int(step_idx))

                active_tasks_by_agv[int(agv_id)] = {
                    "signature": signature,
                    "purpose": str(purpose),
                    "start_step": int(step_idx),
                }

        for agv_id in list(active_tasks_by_agv.keys()):
            finalize_task(int(agv_id), int(final_step))

        return finalized_task_count, total_execution_time_all

    def _compute_assigned_task_wait_metrics(
        self,
        records: list[dict[str, Any]],
    ) -> tuple[int, int, int]:
        """Compute assignment-conditioned wait metrics from per-step records."""
        active_tasks_by_agv: dict[int, dict[str, Any]] = {}
        completed_task_count = 0
        total_wait_all = 0
        total_wait_completed = 0

        def finalize_task(agv_id: int, completed: bool) -> None:
            nonlocal completed_task_count, total_wait_all, total_wait_completed
            task = active_tasks_by_agv.pop(int(agv_id), None)
            if task is None:
                return
            wait_steps = int(task.get("wait_steps", 0))
            total_wait_all += wait_steps
            if completed:
                completed_task_count += 1
                total_wait_completed += wait_steps

        for idx, record in enumerate(records):
            agents_by_id = self._extract_agents_by_id(record.get("state_min"))
            purpose_by_request = self._extract_request_purpose_map(record.get("comm_request"))
            rack_coords_yx_by_id = self._extract_location_coords_yx_map(record.get("state_min"))

            comm_final_plan = record.get("comm_final_plan")
            assignments = comm_final_plan.get("assignments", []) if isinstance(comm_final_plan, dict) else []
            if not isinstance(assignments, list):
                assignments = []

            current_assignments_by_agv: dict[int, dict[str, Any]] = {}
            for assignment in assignments:
                if not isinstance(assignment, dict):
                    continue
                agv_id = self._safe_int(assignment.get("agv_id", 0))
                picker_id = self._safe_int(assignment.get("picker_id", 0))
                rack_id = self._safe_int(assignment.get("rack_id", 0))
                request_id = assignment.get("request_id")
                if agv_id <= 0 or picker_id <= 0 or rack_id <= 0 or not isinstance(request_id, str):
                    continue

                purpose = str(purpose_by_request.get(request_id, "")).upper()
                if purpose not in {"LOAD", "UNLOAD"}:
                    agv_state = agents_by_id.get(int(agv_id))
                    if not isinstance(agv_state, dict):
                        continue
                    purpose = "UNLOAD" if bool(agv_state.get("carrying", False)) else "LOAD"

                current_assignments_by_agv[int(agv_id)] = {
                    "signature": (str(request_id), int(picker_id), int(rack_id), str(purpose)),
                    "purpose": str(purpose),
                    "picker_id": int(picker_id),
                    "rack_id": int(rack_id),
                    "wait_steps": 0,
                }

            for agv_id, new_task in current_assignments_by_agv.items():
                existing_task = active_tasks_by_agv.get(int(agv_id))
                if existing_task is not None and existing_task.get("signature") != new_task.get("signature"):
                    finalize_task(int(agv_id), completed=False)
                if existing_task is None or existing_task.get("signature") != new_task.get("signature"):
                    active_tasks_by_agv[int(agv_id)] = dict(new_task)

            for agv_id, task in list(active_tasks_by_agv.items()):
                agv_state = agents_by_id.get(int(agv_id))
                if not isinstance(agv_state, dict):
                    finalize_task(int(agv_id), completed=False)
                    continue

                purpose = str(task.get("purpose", "")).upper()
                carrying = bool(agv_state.get("carrying", False))
                completed = (
                    (purpose == "LOAD" and carrying)
                    or (purpose == "UNLOAD" and not carrying)
                )
                if completed:
                    finalize_task(int(agv_id), completed=True)
                    continue

                rack_id = int(task.get("rack_id", 0))
                picker_id = int(task.get("picker_id", 0))

                picker_state = agents_by_id.get(int(picker_id))
                rack_coords_yx = rack_coords_yx_by_id.get(int(rack_id))
                agv_at_rack = self._agent_at_coords(agv_state, rack_coords_yx)
                picker_at_rack = self._agent_at_coords(picker_state, rack_coords_yx)
                if agv_at_rack or picker_at_rack:
                    task["wait_steps"] = int(task.get("wait_steps", 0)) + 1

        for agv_id in list(active_tasks_by_agv.keys()):
            finalize_task(int(agv_id), completed=False)

        return completed_task_count, total_wait_all, total_wait_completed

    def _extract_request_purpose_map(self, comm_request: Any) -> dict[str, str]:
        """Map request_id to cooperative purpose when present in comm_request."""
        if not isinstance(comm_request, dict):
            return {}
        requests = comm_request.get("requests", [])
        if not isinstance(requests, list):
            return {}

        purpose_by_request: dict[str, str] = {}
        for item in requests:
            if not isinstance(item, dict):
                continue
            request_id = item.get("request_id")
            if not isinstance(request_id, str):
                continue
            purpose = item.get("purpose")
            if purpose is None:
                continue
            purpose_by_request[request_id] = str(purpose)
        return purpose_by_request

    def _extract_agv_carrying_by_id(self, state_min: Any) -> dict[int, bool]:
        """Map AGV id to carrying state from a minimal state snapshot."""
        if not isinstance(state_min, dict):
            return {}
        agents = state_min.get("agents", [])
        if not isinstance(agents, list):
            return {}

        carrying_by_agv: dict[int, bool] = {}
        for agent in agents:
            if not isinstance(agent, dict):
                continue
            if str(agent.get("type", "")) != "AGV":
                continue
            agv_id = self._safe_int(agent.get("id", 0))
            if agv_id <= 0:
                continue
            carrying_by_agv[int(agv_id)] = bool(agent.get("carrying", False))
        return carrying_by_agv

    def _extract_agents_by_id(self, state_min: Any) -> dict[int, dict[str, Any]]:
        """Map agent id to agent state from a minimal state snapshot."""
        if not isinstance(state_min, dict):
            return {}
        agents = state_min.get("agents", [])
        if not isinstance(agents, list):
            return {}

        agents_by_id: dict[int, dict[str, Any]] = {}
        for agent in agents:
            if not isinstance(agent, dict):
                continue
            agent_id = self._safe_int(agent.get("id", 0))
            if agent_id <= 0:
                continue
            agents_by_id[int(agent_id)] = agent
        return agents_by_id

    def _extract_location_coords_yx_map(self, state_min: Any) -> dict[int, tuple[int, int]]:
        """Map location id to rack coordinates in y/x order from a minimal state snapshot."""
        if not isinstance(state_min, dict):
            return {}
        location_coords_xy = state_min.get("location_coords_xy", {})
        if not isinstance(location_coords_xy, dict):
            return {}

        coords_yx_by_id: dict[int, tuple[int, int]] = {}
        for loc_id, coords_xy in location_coords_xy.items():
            try:
                parsed_loc_id = int(loc_id)
            except (TypeError, ValueError):
                continue
            coords = self._normalize_coords_xy(coords_xy)
            if coords is None:
                continue
            coords_yx_by_id[int(parsed_loc_id)] = coords
        return coords_yx_by_id

    def _agent_at_assigned_rack(self, agent: Any, rack_id: int) -> bool:
        """Return whether an agent is currently at the assigned rack location."""
        if not isinstance(agent, dict):
            return False
        target = self._safe_int(agent.get("target", 0))
        if int(target) != int(rack_id):
            return False
        return self._same_coords(agent.get("coords_yx"), agent.get("target_coords_yx"))

    def _agent_at_coords(self, agent: Any, coords_yx: tuple[int, int] | None) -> bool:
        """Return whether an agent is physically at the given y/x coordinates."""
        if not isinstance(agent, dict) or coords_yx is None:
            return False
        return self._same_coords(agent.get("coords_yx"), coords_yx)

    def _same_coords(self, a: Any, b: Any) -> bool:
        a_yx = self._normalize_coords_yx(a)
        b_yx = self._normalize_coords_yx(b)
        if a_yx is None or b_yx is None:
            return False
        return a_yx == b_yx

    def _normalize_coords_yx(self, value: Any) -> tuple[int, int] | None:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return int(value[0]), int(value[1])
            except (TypeError, ValueError):
                return None
        return None

    def _normalize_coords_xy(self, value: Any) -> tuple[int, int] | None:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return int(value[1]), int(value[0])
            except (TypeError, ValueError):
                return None
        return None

    def _safe_int(self, value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return -1

    def _safe_nonnegative_int(self, value: Any) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return 0
        return parsed if parsed >= 0 else 0

    def _maybe_to_dict(self, value: Any) -> Any:
        """Return a JSON-serializable dict when the object exposes to_dict()."""
        if value is None:
            return None
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            return self._to_jsonable(to_dict())
        return None

    def _to_jsonable(self, value: Any) -> Any:
        """Recursively convert values to JSON-serializable builtins."""
        if isinstance(value, dict):
            return {str(key): self._to_jsonable(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(item) for item in value]
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)


if __name__ == "__main__":
    config = RunnerConfig(
        env_id="tarware-small-2agvs-2pickers-partialobs-v1",
        num_seeds=2,
        repeats_per_seed=3,
        seed=0,
    )
    runner = EpisodeRunner(config)
    from symco.planners.non_symbiotic_planner import NonSymbioticPlanner

    result = runner.run(NonSymbioticPlanner())
    print(result["episode_summaries"][0])