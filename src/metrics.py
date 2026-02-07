# src/metrics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import math


@dataclass
class EpisodeResult:
    episode_idx: int
    steps: int
    total_reward: float
    rewards_per_agent: List[float]


def mean_std(values: List[float]) -> dict:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    m = sum(values) / len(values)
    var = sum((x - m) ** 2 for x in values) / len(values)
    return {"mean": m, "std": math.sqrt(var)}
