import json
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import gymnasium as gym

from tarware.heuristic_custom import heuristic_episode_custom

parser = ArgumentParser(
    description="Run custom heuristic baseline with thesis metrics on TA-RWARE",
    formatter_class=ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--env",
    default="tarware-small-2agvs-2pickers-partialobs-v1",
    type=str,
    help="Gym environment ID",
)

parser.add_argument(
    "--num_episodes",
    default=1,
    type=int,
    help="Number of episodes to run",
)

parser.add_argument(
    "--seed",
    default=0,
    type=int,
    help="Starting seed",
)

parser.add_argument(
    "--max_steps",
    default=1000,
    type=int,
    help="Maximum episode steps",
)

parser.add_argument(
    "--render",
    action="store_true",
)

parser.add_argument(
    "--save_json",
    default="",
    type=str,
    help="Optional path to save per-episode results as JSON",
)

args = parser.parse_args()


def info_statistics(infos, global_episode_return, episode_returns, episode_metrics):
    total_deliveries = 0
    total_clashes = 0
    total_stuck = 0

    for info in infos:
        total_deliveries += info["shelf_deliveries"]
        total_clashes += info["clashes"]
        total_stuck += info["stucks"]

    last_info = infos[-1].copy()
    last_info["episode_length"] = episode_metrics["episode_length"]
    last_info["global_episode_return"] = float(global_episode_return)
    last_info["episode_returns"] = [float(x) for x in episode_returns]

    # Original baseline-facing metrics
    last_info["total_deliveries"] = total_deliveries
    last_info["total_clashes"] = total_clashes
    last_info["total_stuck"] = total_stuck

    # Thesis metrics
    last_info["completed_deliveries"] = total_deliveries
    last_info["cooperative_waiting_time"] = episode_metrics["cooperative_wait_steps"]
    last_info["picker_idle_time_ratio"] = episode_metrics["picker_idle_time_ratio"]

    # Execution-time metrics
    last_info["avg_load_execution_time"] = episode_metrics["avg_load_execution_time"]
    last_info["avg_unload_execution_time"] = episode_metrics["avg_unload_execution_time"]
    last_info["avg_execution_time_per_assignment"] = episode_metrics["avg_execution_time_per_assignment"]

    # Arrival-wait and success-rate metrics
    last_info["avg_wait_time_after_first_arrival"] = episode_metrics["avg_wait_time_after_first_arrival"]
    last_info["assignment_success_rate"] = episode_metrics["assignment_success_rate"]

    # Average cooperative waiting time per delivery
    if total_deliveries > 0:
        last_info["avg_cooperative_waiting_time"] = (
            episode_metrics["cooperative_wait_steps"] / total_deliveries
        )
    else:
        last_info["avg_cooperative_waiting_time"] = 0.0

    return last_info


if __name__ == "__main__":
    env = gym.make(args.env, max_steps=args.max_steps)
    base_seed = args.seed
    completed_episodes = 0
    all_episode_results = []

    print(f"env max_steps = {env.unwrapped.max_steps}")

    for i in range(args.num_episodes):
        start = time.time()

        infos, global_episode_return, episode_returns, episode_metrics = heuristic_episode_custom(
            env.unwrapped, args.render, base_seed + i
        )

        end = time.time()

        last_info = info_statistics(infos, global_episode_return, episode_returns, episode_metrics)
        last_info["overall_pick_rate"] = (
            last_info["completed_deliveries"] * 3600 / (5 * last_info["episode_length"])
            if last_info["episode_length"] > 0
            else 0.0
        )

        episode_length = last_info["episode_length"]
        fps = episode_length / (end - start) if (end - start) > 0 else 0.0

        print(
            f"Completed Episode {completed_episodes}: "
            f"| [Env={args.env}]"
            f"| [Overall Pick Rate={last_info['overall_pick_rate']:.2f}]"
            f"| [Global return={last_info['global_episode_return']:.2f}]"
            f"| [Completed deliveries={last_info['completed_deliveries']:.2f}]"
            f"| [Cooperative waiting time={last_info['cooperative_waiting_time']:.2f}]"
            f"| [Avg cooperative waiting time={last_info['avg_cooperative_waiting_time']:.2f}]"
            f"| [Picker idle time ratio={last_info['picker_idle_time_ratio']:.4f}]"
            f"| [Avg load execution time={last_info['avg_load_execution_time']:.2f}]"
            f"| [Avg unload execution time={last_info['avg_unload_execution_time']:.2f}]"
            f"| [Avg execution time per assignment={last_info['avg_execution_time_per_assignment']:.2f}]"
            f"| [Avg wait time after first arrival={last_info['avg_wait_time_after_first_arrival']:.2f}]"
            f"| [Assignment success rate={last_info['assignment_success_rate']:.4f}]"
            f"| [Total clashes={last_info['total_clashes']:.2f}]"
            f"| [Total stuck={last_info['total_stuck']:.2f}]"
            f"| [FPS={fps:.2f}]"
        )

        all_episode_results.append(last_info)
        completed_episodes += 1

    env.close()

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(all_episode_results, f, ensure_ascii=False, indent=2)
        print(f"Saved results to {out_path}")