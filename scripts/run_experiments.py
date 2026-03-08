"""Batch launcher for TA-RWARE comparison experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "compare_baselines.py"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for batch experiment execution."""
    parser = argparse.ArgumentParser(
        description="Run multiple TA-RWARE comparison experiments and collect a master summary."
    )
    parser.add_argument(
        "--env_ids",
        required=True,
        help="Comma-separated list of environment ids.",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per seed.")
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seed list.")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum steps per episode.")
    parser.add_argument("--out_dir", type=str, default="outputs_experiments", help="Output directory.")
    parser.add_argument("--topk_requests", type=int, default=10, help="Top-k requested racks.")
    parser.add_argument("--topk_empty", type=int, default=10, help="Top-k empty racks.")
    parser.add_argument("--topk_goals", type=int, default=10, help="Top-k goals.")
    parser.add_argument(
        "--pickers_to_agvs",
        type=int,
        choices=(0, 1),
        default=1,
        help="Whether picker valid actions follow AGV targets.",
    )
    parser.add_argument(
        "--block_conflicting_actions",
        type=int,
        choices=(0, 1),
        default=1,
        help="Whether to block conflicting same-type actions.",
    )
    parser.add_argument(
        "--care_for_agents_in_cost",
        type=int,
        choices=(0, 1),
        default=0,
        help="Whether path costs account for other agents.",
    )
    return parser.parse_args()


def parse_env_ids(text: str) -> list[str]:
    """Parse a comma-separated list of env ids."""
    env_ids = [part.strip() for part in text.split(",") if part.strip()]
    if not env_ids:
        raise ValueError("At least one env_id must be provided.")
    return env_ids


def sanitize_env_id(env_id: str) -> str:
    """Sanitize an env id for directory and file naming."""
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in env_id)


def build_compare_command(args: argparse.Namespace, env_id: str, env_out_dir: Path) -> list[str]:
    """Build the subprocess command for one comparison run."""
    return [
        sys.executable,
        str(COMPARE_SCRIPT),
        "--env_id",
        env_id,
        "--episodes",
        str(args.episodes),
        "--seeds",
        args.seeds,
        "--max_steps",
        str(args.max_steps),
        "--out_dir",
        str(env_out_dir),
        "--topk_requests",
        str(args.topk_requests),
        "--topk_empty",
        str(args.topk_empty),
        "--topk_goals",
        str(args.topk_goals),
        "--pickers_to_agvs",
        str(args.pickers_to_agvs),
        "--block_conflicting_actions",
        str(args.block_conflicting_actions),
        "--care_for_agents_in_cost",
        str(args.care_for_agents_in_cost),
    ]


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file into a dictionary."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def build_experiment_entry(env_id: str, comparison_json: Path, comparison: dict[str, Any]) -> dict[str, Any]:
    """Build one master-summary entry from a comparison result."""
    baseline_agg = comparison.get("non_symbiotic", {}).get("aggregate", {})
    symbiotic_agg = comparison.get("symbiotic", {}).get("aggregate", {})
    return {
        "env_id": env_id,
        "comparison_json": str(comparison_json),
        "baseline_mean_deliveries": float(baseline_agg.get("mean_deliveries", 0.0)),
        "symbiotic_mean_deliveries": float(symbiotic_agg.get("mean_deliveries", 0.0)),
        "baseline_avg_wait_to_load_steps": float(baseline_agg.get("avg_wait_to_load_steps", 0.0)),
        "symbiotic_avg_wait_to_load_steps": float(symbiotic_agg.get("avg_wait_to_load_steps", 0.0)),
        "symbiotic_communication_ratio": float(symbiotic_agg.get("communication_ratio", 0.0)),
        "symbiotic_total_trm_events_per_seed": float(symbiotic_agg.get("total_trm_events_per_seed", 0.0)),
    }


def print_summary_table(experiments: list[dict[str, Any]]) -> None:
    """Print a concise table-like summary for all experiments."""
    print(
        "env_id | baseline_deliv | symbiotic_deliv | baseline_wait | "
        "symbiotic_wait | symbiotic_comm | symbiotic_trm"
    )
    for item in experiments:
        if "error" in item:
            print(f"{item.get('env_id', 'unknown')} | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR")
            continue
        print(
            f"{item['env_id']} | "
            f"{item['baseline_mean_deliveries']:.3f} | "
            f"{item['symbiotic_mean_deliveries']:.3f} | "
            f"{item['baseline_avg_wait_to_load_steps']:.3f} | "
            f"{item['symbiotic_avg_wait_to_load_steps']:.3f} | "
            f"{item['symbiotic_communication_ratio']:.3f} | "
            f"{item['symbiotic_total_trm_events_per_seed']:.3f}"
        )


def main() -> int:
    """Run batch experiments and write a master summary."""
    args = parse_args()
    env_ids = parse_env_ids(args.env_ids)
    root_out_dir = Path(args.out_dir)
    root_out_dir.mkdir(parents=True, exist_ok=True)

    experiments: list[dict[str, Any]] = []

    for env_id in env_ids:
        env_out_dir = root_out_dir / sanitize_env_id(env_id)
        env_out_dir.mkdir(parents=True, exist_ok=True)
        command = build_compare_command(args, env_id, env_out_dir)

        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            check=False,
            capture_output=True,
            text=True,
        )

        comparison_json = env_out_dir / f"comparison_{sanitize_env_id(env_id)}.json"
        if completed.returncode != 0:
            experiments.append(
                {
                    "env_id": env_id,
                    "comparison_json": str(comparison_json),
                    "error": f"compare_baselines.py failed with exit code {completed.returncode}",
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            )
            continue

        if not comparison_json.exists():
            experiments.append(
                {
                    "env_id": env_id,
                    "comparison_json": str(comparison_json),
                    "error": "comparison json not found after successful run",
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            )
            continue

        try:
            comparison = load_json(comparison_json)
            entry = build_experiment_entry(env_id, comparison_json, comparison)
            experiments.append(entry)
        except Exception as exc:
            experiments.append(
                {
                    "env_id": env_id,
                    "comparison_json": str(comparison_json),
                    "error": f"failed to load comparison json: {exc}",
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            )

    master_summary = {
        "config": {
            "env_ids": env_ids,
            "episodes": args.episodes,
            "seeds": args.seeds,
            "max_steps": args.max_steps,
            "out_dir": args.out_dir,
            "topk_requests": args.topk_requests,
            "topk_empty": args.topk_empty,
            "topk_goals": args.topk_goals,
            "pickers_to_agvs": bool(args.pickers_to_agvs),
            "block_conflicting_actions": bool(args.block_conflicting_actions),
            "care_for_agents_in_cost": bool(args.care_for_agents_in_cost),
        },
        "experiments": experiments,
    }

    master_summary_path = root_out_dir / "master_summary.json"
    with master_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(master_summary, handle, ensure_ascii=True, indent=2)

    print_summary_table(experiments)
    print(f"master_summary_json: {master_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
