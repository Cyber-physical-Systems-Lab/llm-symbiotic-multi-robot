
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
    num_episodes: int = 10
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
    """
    def make_env(self) -> gym.Env:
        
        try:
            return gym.make(self.config.env_id)
        except Exception as exc:
            raise RuntimeError(f"Failed to create environment '{self.config.env_id}': {exc}") from exc
    """
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
        try:
            for episode_idx in range(self.config.num_episodes):
                summary = self._run_episode(
                    env=env,
                    planner=planner,
                    episode_idx=episode_idx,
                    jsonl_handle=jsonl_handle,
                )
                episode_summaries.append(summary)
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
        episode_idx: int,
        jsonl_handle: Any,
    ) -> dict[str, Any]:
        """Execute a single episode and return aggregated metrics."""
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
        total_recommends = 0
        total_declines = 0
        total_multi_option_recommends = 0
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

            if jsonl_handle is not None:
                record = {
                    "episode_idx": episode_idx,
                    "step_idx": steps,
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
                    "recommend_count": recommend_count,
                    "decline_count": decline_count,
                    "multi_option_recommend_count": multi_option_recommend_count,
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
        }

    def _sanitize_actions(
        self,
        actions: list[Any],
        env: gym.Env,
        episode_idx: int,
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
                    "Invalid non-integer action for agent %s at episode %s step %s: %r. Using 0.",
                    agent_idx + 1,
                    episode_idx,
                    step_idx,
                    action,
                )
                sanitized.append(0)
                continue

            if 0 <= candidate < action_size:
                safe_action = candidate
            else:
                LOGGER.warning(
                    "Out-of-range action for agent %s at episode %s step %s: %s not in [0, %s). Using 0.",
                    agent_idx + 1,
                    episode_idx,
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
        return f"{timestamp}_{env_name}_{planner_name}"

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

    def _safe_int(self, value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return -1

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
        num_episodes=2,
        seed=0,
    )
    runner = EpisodeRunner(config)
    from symco.planners.non_symbiotic_planner import NonSymbioticPlanner

    result = runner.run(NonSymbioticPlanner())
    print(result["episode_summaries"][0])