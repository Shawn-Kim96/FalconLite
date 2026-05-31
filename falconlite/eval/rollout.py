"""Rollout orchestration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from falconlite.env import RocketLandingEnv
from falconlite.env.state import RocketState
from falconlite.utils.logger import TelemetryLogger


class Controller(Protocol):
    """Controller interface used by evaluation rollouts."""

    def select_action(self, state: RocketState, info: dict[str, Any]) -> np.ndarray:
        """Return a normalized action."""


@dataclass(frozen=True)
class EpisodeResult:
    """Summary of one completed rollout."""

    episode_id: int
    controller: str
    steps: int
    total_reward: float
    done_reason: str
    is_success: bool
    final_x: float
    final_y: float
    final_vx: float
    final_vy: float
    final_theta: float
    final_omega: float
    final_fuel: float
    fuel_used: float
    touchdown_speed: float
    max_tilt: float
    missed_pad: bool
    hard_landing: bool
    tip_over: bool
    out_of_bounds: bool
    body_contact: bool = False
    one_foot_contact: bool = False
    log_path: str | None = None


def run_episode(
    *,
    env: RocketLandingEnv,
    controller: Controller,
    controller_name: str,
    episode_id: int,
    seed: int | None = None,
    scenario: str | None = None,
    log_dir: str | Path | None = None,
) -> EpisodeResult:
    """Run one episode and return aggregate episode data."""

    if hasattr(controller, "reset"):
        controller.reset()

    reset_options = {"scenario": scenario} if scenario is not None else None
    _, info = env.reset(seed=seed, options=reset_options)
    initial_fuel = info["state"].fuel
    max_tilt = abs(info["state"].theta)
    total_reward = 0.0
    steps = 0
    logger = TelemetryLogger(log_dir=log_dir, episode_id=episode_id) if log_dir is not None else None

    try:
        while True:
            action = controller.select_action(info["state"], info)
            _, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            state = info["state"]
            max_tilt = max(max_tilt, abs(state.theta))

            if logger is not None:
                logger.log_step(
                    state=state,
                    action=info["actual_action"],
                    reward=reward,
                    info=info,
                )

            if terminated or truncated:
                break
    finally:
        if logger is not None:
            logger.close()

    state = info["state"]
    failure_flags = info.get("failure_flags", {})
    touchdown_speed = sqrt(state.vx * state.vx + state.vy * state.vy)
    return EpisodeResult(
        episode_id=episode_id,
        controller=controller_name,
        steps=steps,
        total_reward=float(total_reward),
        done_reason=info["done_reason"],
        is_success=bool(info["is_success"]),
        final_x=state.x,
        final_y=state.y,
        final_vx=state.vx,
        final_vy=state.vy,
        final_theta=state.theta,
        final_omega=state.omega,
        final_fuel=state.fuel,
        fuel_used=max(0.0, initial_fuel - state.fuel),
        touchdown_speed=touchdown_speed,
        max_tilt=max_tilt,
        missed_pad=bool(failure_flags.get("missed_pad", False)),
        hard_landing=bool(failure_flags.get("hard_landing", False)),
        tip_over=bool(failure_flags.get("tip_over", False)),
        out_of_bounds=bool(failure_flags.get("out_of_bounds", False)),
        body_contact=bool(failure_flags.get("body_contact", False)),
        one_foot_contact=bool(failure_flags.get("one_foot_contact", False)),
        log_path=str(logger.path) if logger is not None else None,
    )
