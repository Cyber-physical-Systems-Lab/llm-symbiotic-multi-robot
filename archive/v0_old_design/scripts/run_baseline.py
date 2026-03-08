# scripts/run_baseline.py
from __future__ import annotations
import argparse
import json
from pathlib import Path
from datetime import datetime

from src.env_runner import RunConfig, run_episodes
from src.metrics import mean_std
from src.agents.random_agent import RandomAgent
from src.agents.llm_agent import IndependentLLMAgent
from src.llm.mock_provider import MockLLMProvider

import gymnasium as gym
import tarware  # noqa: F401


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", type=str, default="tarware-tiny-3agvs-2pickers-partialobs-v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--method", type=str, choices=["random", "independent_llm"], default="random")
    p.add_argument("--llm_provider", type=str, choices=["mock"], default="mock")
    p.add_argument("--outdir", type=str, default="outputs")
    args = p.parse_args()

    out_base = Path(args.outdir) / "independent_baseline" / f"env={args.env_id}" / f"method={args.method}" / f"seed={args.seed}"
    out_base.mkdir(parents=True, exist_ok=True)

    
    # Build env once and infer agent count robustly
    env = gym.make(args.env_id)

    # Robust across versions: Tuple(action_space) has one subspace per agent
    n_agents = len(env.action_space)

    agents = []
    if args.method == "random":
        for i in range(n_agents):
            agents.append(RandomAgent(agent_id=i))
    else:
        for i in range(n_agents):
            n_actions_i = int(env.action_space[i].n)
            llm = MockLLMProvider(action_space_n=n_actions_i, seed=args.seed + i)
            agents.append(IndependentLLMAgent(agent_id=i, llm=llm))

    env.close()


    cfg = RunConfig(env_id=args.env_id, seed=args.seed, episodes=args.episodes, max_steps=args.max_steps)
    results = run_episodes(cfg, agents, method=("random" if args.method == "random" else "independent_llm"))

    # Summaries
    total_rewards = [r.total_reward for r in results]
    lengths = [float(r.steps) for r in results]

    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "env_id": args.env_id,
        "seed": args.seed,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "method": args.method,
        "llm_provider": (args.llm_provider if args.method == "independent_llm" else None),
        "metrics": {
            "total_reward": mean_std(total_rewards),
            "episode_length": mean_std(lengths),
        },
    }

    (out_base / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (out_base / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Save per-episode metrics for later plots
    rows = []
    for r in results:
        rows.append({
            "episode": r.episode_idx,
            "steps": r.steps,
            "total_reward": r.total_reward,
            "rewards_per_agent": json.dumps(r.rewards_per_agent),
        })
    (out_base / "episode_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"Saved results to: {out_base}")


if __name__ == "__main__":
    main()
