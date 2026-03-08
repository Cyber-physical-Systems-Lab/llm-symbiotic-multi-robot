"""Generate thesis figures from experiment comparison summaries."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs_experiments" / "master_summary.json"
DEFAULT_FIG_DIR = ROOT / "outputs_experiments" / "figures"


def load_master_summary(path: Path) -> dict:
    """Load the master summary JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def successful_experiments(summary: dict) -> list[dict]:
    """Return experiment entries without recorded errors."""
    experiments = summary.get("experiments", [])
    if not isinstance(experiments, list):
        return []
    return [item for item in experiments if isinstance(item, dict) and "error" not in item]


def extract_series(experiments: list[dict], key: str) -> list[float]:
    """Extract one numeric series from experiment entries."""
    values: list[float] = []
    for item in experiments:
        values.append(float(item.get(key, 0.0)))
    return values


def make_line_figure(
    env_ids: list[str],
    series_list: list[tuple[str, list[float]]],
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """Create and save a simple academic-style line plot."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = list(range(len(env_ids)))
    markers = ["o", "s", "^"]

    for index, (label, values) in enumerate(series_list):
        ax.plot(
            x,
            values,
            marker=markers[index % len(markers)],
            linewidth=2.0,
            markersize=6,
            label=label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(env_ids, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Environment")
    ax.set_title(title)
    if len(series_list) > 1:
        ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    """Load summary results and generate thesis figures."""
    summary = load_master_summary(DEFAULT_INPUT)
    experiments = successful_experiments(summary)
    if not experiments:
        raise RuntimeError(f"No successful experiments found in {DEFAULT_INPUT}")

    fig_dir = DEFAULT_FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)

    env_ids = [str(item.get("env_id", "")) for item in experiments]
    baseline_deliveries = extract_series(experiments, "baseline_mean_deliveries")
    symbiotic_deliveries = extract_series(experiments, "symbiotic_mean_deliveries")
    baseline_wait = extract_series(experiments, "baseline_avg_wait_to_load_steps")
    symbiotic_wait = extract_series(experiments, "symbiotic_avg_wait_to_load_steps")
    trm_events = extract_series(experiments, "symbiotic_total_trm_events_per_seed")

    make_line_figure(
        env_ids=env_ids,
        series_list=[
            ("Baseline", baseline_deliveries),
            ("Symbiotic", symbiotic_deliveries),
        ],
        ylabel="Mean Deliveries",
        title="Throughput Across Environments",
        output_path=fig_dir / "throughput.png",
    )
    make_line_figure(
        env_ids=env_ids,
        series_list=[
            ("Baseline", baseline_wait),
            ("Symbiotic", symbiotic_wait),
        ],
        ylabel="Average Wait-to-Load Steps",
        title="Wait-to-Load Across Environments",
        output_path=fig_dir / "wait_to_load.png",
    )
    make_line_figure(
        env_ids=env_ids,
        series_list=[("Symbiotic", trm_events)],
        ylabel="TRM Events Per Seed",
        title="Target Revision Due to Messages",
        output_path=fig_dir / "trm_events.png",
    )

    print(fig_dir / "throughput.png")
    print(fig_dir / "wait_to_load.png")
    print(fig_dir / "trm_events.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
