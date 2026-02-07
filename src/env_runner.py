# src/env_runner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

import gymnasium as gym
import tarware  # noqa: F401  # ensure env is registered

from .metrics import EpisodeResult


@dataclass
class RunConfig:
    env_id: str
    seed: int
    episodes: int
    max_steps: int = 500


def _reset_compat(env, seed: int):
    """
    Make env.reset compatible across different gym/gymnasium versions and
    imperfect env implementations.

    - Gymnasium standard: (obs, info_dict)
    - Some envs: obs only
    """
    res = env.reset(seed=seed)

    # gymnasium standard: (obs, info)
    if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], dict):
        return res[0], res[1]

    # non-standard / older env: obs only
    return res, {}


def _step_compat(env, actions):
    """
    Make env.step compatible across different gym/gymnasium versions.

    - Gymnasium: obs, reward, terminated, truncated, info
    - Gym (older): obs, reward, done, info

    NOTE: TA-RWARE usually uses per-agent reward and done flags (lists/tuples).
    """
    res = env.step(actions)

    # gymnasium standard
    if isinstance(res, tuple) and len(res) == 5:
        return res

    # older gym style
    if isinstance(res, tuple) and len(res) == 4:
        obs, reward, done, info = res

        # Convert done -> terminated/truncated
        terminated = done
        # If reward is per-agent list/tuple, create matching truncated flags
        if hasattr(reward, "__len__") and not isinstance(reward, (str, bytes)):
            truncated = [False] * len(reward)
        else:
            truncated = False

        return obs, reward, terminated, truncated, info

    raise RuntimeError(f"Unexpected step() return format: {type(res)} len={len(res) if isinstance(res, tuple) else 'n/a'}")


def _sum_rewards(rewards: Sequence[Any]) -> Tuple[float, List[float]]:
    """
    TA-RWARE typically returns reward as list/tuple per agent.
    Convert to float list and sum.
    """
    per = [float(r) for r in rewards]
    return float(sum(per)), per


def run_episodes(
    cfg: RunConfig,
    agents: List[Any],
    method: str,
) -> List[EpisodeResult]:
    """
    Run multiple episodes and return per-episode results.

    method:
      - "random": env.action_space[i].sample()
      - "independent_llm": agent.decide_action(obs_i, env.action_space, fallback)
    """
    env = gym.make(cfg.env_id)
    results: List[EpisodeResult] = []

    # robust agent count from Tuple action space
    n_agents = len(env.action_space)

    for ep in range(cfg.episodes):
        obs, info = _reset_compat(env, seed=cfg.seed + ep)

        total_reward = 0.0
        rewards_per_agent_acc = [0.0 for _ in range(n_agents)]

        steps_taken = 0

        for t in range(cfg.max_steps):
            actions = []

            for i in range(n_agents):
                # fallback random action for this agent
                def _fallback():
                    return env.action_space[i].sample()

                if method == "random":
                    a = int(env.action_space[i].sample())
                else:
                    # agent must implement decide_action(obs_i, action_space, fallback_random)
                    a = int(agents[i].decide_action(obs[i], env.action_space, _fallback))

                actions.append(a)

            obs, rewards, terminated, truncated, info = _step_compat(env, tuple(actions))

            step_total, step_per = _sum_rewards(rewards)
            total_reward += step_total
            for i in range(n_agents):
                rewards_per_agent_acc[i] += step_per[i]

            steps_taken = t + 1

            # terminated/truncated are often lists per agent in TA-RWARE
            done = False
            if isinstance(terminated, (list, tuple)):
                done = any(terminated) or any(truncated)
            else:
                done = bool(terminated) or bool(truncated)

            if done:
                break

        results.append(
            EpisodeResult(
                episode_idx=ep,
                steps=steps_taken,
                total_reward=total_reward,
                rewards_per_agent=rewards_per_agent_acc,
            )
        )

    env.close()
    return results
