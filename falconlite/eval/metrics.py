"""Evaluation metrics for controller rollouts."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Any

import pandas as pd

from falconlite.eval.rollout import EpisodeResult


CRASH_REASONS = {"missed_pad", "hard_landing", "tip_over", "body_contact", "one_foot_contact", "crash"}


def summarize_episode_results(results: list[EpisodeResult]) -> dict[str, float]:
    """Compute aggregate metrics over completed episodes."""

    if not results:
        raise ValueError("Cannot summarize an empty result set.")

    count = len(results)
    done_counts = Counter(result.done_reason for result in results)
    success_count = sum(result.is_success for result in results)
    crash_count = sum(result.done_reason in CRASH_REASONS for result in results)

    return {
        "episodes": float(count),
        "success_rate": success_count / count,
        "crash_rate": crash_count / count,
        "hard_landing_rate": done_counts["hard_landing"] / count,
        "missed_pad_rate": done_counts["missed_pad"] / count,
        "tip_over_rate": done_counts["tip_over"] / count,
        "body_contact_rate": done_counts["body_contact"] / count,
        "one_foot_contact_rate": done_counts["one_foot_contact"] / count,
        "out_of_bounds_rate": done_counts["out_of_bounds"] / count,
        "max_steps_rate": done_counts["max_steps"] / count,
        "average_total_reward": _mean(result.total_reward for result in results),
        "average_fuel_used": _mean(result.fuel_used for result in results),
        "average_touchdown_speed": _mean(result.touchdown_speed for result in results),
        "average_touchdown_vy": _mean(abs(result.final_vy) for result in results),
        "average_max_tilt": _mean(result.max_tilt for result in results),
        "average_episode_length": _mean(result.steps for result in results),
    }


def episode_results_to_frame(results: list[EpisodeResult]) -> pd.DataFrame:
    """Convert per-episode results to a pandas DataFrame."""

    return pd.DataFrame(asdict(result) for result in results)


def metrics_to_frame(metrics: dict[str, float], metadata: dict[str, Any] | None = None) -> pd.DataFrame:
    """Convert one aggregate metrics dict to a single-row DataFrame."""

    row: dict[str, Any] = {}
    if metadata is not None:
        row.update(metadata)
    row.update(metrics)
    return pd.DataFrame([row])


def _mean(values: Any) -> float:
    values = list(values)
    return float(sum(values) / len(values))
