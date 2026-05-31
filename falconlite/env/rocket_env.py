"""Gymnasium environment for FalconLite."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from falconlite.env.physics import PhysicsEngine
from falconlite.env.renderer import Renderer
from falconlite.env.state import RocketAction, RocketState
from falconlite.utils.config import load_config


@dataclass(frozen=True)
class EnvironmentConfig:
    """Episode and normalization settings for the Gymnasium wrapper."""

    max_steps: int = 2000
    max_speed: float = 50.0
    max_angular_speed: float = 20.0
    max_angle: float = np.pi

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "EnvironmentConfig":
        allowed_keys = cls.__dataclass_fields__.keys()
        filtered = {key: values[key] for key in allowed_keys if key in values}
        return cls(**filtered)

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if self.max_speed <= 0:
            raise ValueError("max_speed must be positive.")
        if self.max_angular_speed <= 0:
            raise ValueError("max_angular_speed must be positive.")
        if self.max_angle <= 0:
            raise ValueError("max_angle must be positive.")


@dataclass(frozen=True)
class RewardConfig:
    """Reward shaping and landing classification settings."""

    pad_x: float = 0.0
    landing_tolerance: float = 5.0
    max_touchdown_vx: float = 2.0
    max_touchdown_vy: float = 3.0
    max_touchdown_angle: float = 0.2
    max_touchdown_omega: float = 1.0
    required_stable_time: float = 3.0
    target_descent_rate_gain: float = 0.08
    min_target_descent_rate: float = 1.0
    max_target_descent_rate: float = 8.0
    x_weight: float = 2.0
    vx_weight: float = 0.5
    vy_weight: float = 1.0
    theta_weight: float = 0.75
    omega_weight: float = 0.25
    fuel_weight: float = 0.02
    time_penalty: float = 0.01
    success_bonus: float = 100.0
    crash_penalty: float = -100.0
    out_of_bounds_penalty: float = -100.0
    max_steps_penalty: float = -20.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "RewardConfig":
        allowed_keys = cls.__dataclass_fields__.keys()
        filtered = {key: values[key] for key in allowed_keys if key in values}
        return cls(**filtered)

    def __post_init__(self) -> None:
        if self.landing_tolerance <= 0:
            raise ValueError("landing_tolerance must be positive.")
        if self.max_touchdown_vx <= 0:
            raise ValueError("max_touchdown_vx must be positive.")
        if self.max_touchdown_vy <= 0:
            raise ValueError("max_touchdown_vy must be positive.")
        if self.max_touchdown_angle <= 0:
            raise ValueError("max_touchdown_angle must be positive.")
        if self.max_touchdown_omega <= 0:
            raise ValueError("max_touchdown_omega must be positive.")
        if self.required_stable_time <= 0:
            raise ValueError("required_stable_time must be positive.")
        if self.min_target_descent_rate <= 0 or self.max_target_descent_rate < self.min_target_descent_rate:
            raise ValueError("target descent rates must satisfy 0 < min <= max.")


class RocketLandingEnv(gym.Env):
    """Gymnasium-compatible wrapper around FalconLite physics."""

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        render_mode: str | None = None,
    ) -> None:
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.config = dict(load_config() if config is None else config)
        self.physics_engine = PhysicsEngine(self.config["physics"])
        self.env_config = EnvironmentConfig.from_mapping(self.config["environment"])
        self.reward_config = RewardConfig.from_mapping(self.config.get("reward", {}))
        self.initial_state_config = dict(self.config["environment"]["initial_state"])
        self.render_mode = render_mode
        self.renderer: Renderer | None = None
        self.state = RocketState(**self.initial_state_config)
        self.last_action: RocketAction | None = None
        self.last_info: dict[str, Any] = {"done_reason": "not_started"}
        self.step_count = 0

        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        initial_state_values = dict(self.initial_state_config)
        if options is not None:
            scenario_name = options.get("scenario")
            if scenario_name is not None:
                initial_state_values.update(self._scenario_initial_state(str(scenario_name)))
            if "initial_state" in options:
                initial_state_values.update(options["initial_state"])

        self.state = RocketState(**initial_state_values)
        self.last_action = None
        self.last_info = {
            "done_reason": "reset",
            "terminated": False,
            "truncated": False,
            "is_success": False,
            "failure_flags": self._empty_failure_flags(),
            "reward_terms": self._empty_reward_terms(),
        }
        self.step_count = 0
        return self._get_obs(), self._get_info()

    def _scenario_initial_state(self, scenario_name: str) -> dict[str, Any]:
        scenarios = self.config.get("scenarios", {})
        if scenario_name not in scenarios:
            available = ", ".join(sorted(scenarios)) or "none"
            raise ValueError(f"Unknown scenario '{scenario_name}'. Available scenarios: {available}.")
        return dict(scenarios[scenario_name])

    def step(self, action: object) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != self.action_space.shape:
            raise ValueError(f"Expected action shape {self.action_space.shape}, got {action_array.shape}.")

        clipped_action = np.clip(action_array, self.action_space.low, self.action_space.high)
        physical_action = self.physics_engine.action_from_normalized(clipped_action)
        self.state, physics_info = self.physics_engine.step(self.state, physical_action)
        self.last_action = physical_action
        self.step_count += 1

        physics_terminated = bool(physics_info["terminated"])
        step_limit_reached = self.step_count >= self.env_config.max_steps and not physics_terminated
        terminal_result = self._classify_terminal_state(
            physics_info["done_reason"],
            physics_terminated,
            step_limit_reached,
            physics_info,
        )
        terminated = terminal_result["terminated"]
        truncated = terminal_result["truncated"]
        reward, reward_terms = self._compute_reward(clipped_action[0], terminal_result["done_reason"])

        self.last_info = {
            **physics_info,
            "terminated": terminated,
            "truncated": truncated,
            "done_reason": terminal_result["done_reason"],
            "is_success": terminal_result["is_success"],
            "failure_flags": terminal_result["failure_flags"],
            "reward_terms": reward_terms,
            "step": self.step_count,
            "time": self.step_count * self.physics_engine.config.dt,
            "normalized_action": clipped_action,
        }
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self) -> None:
        if self.render_mode != "human":
            return
        if self.renderer is None:
            self.renderer = Renderer(self.config["physics"], self.config.get("render"))
        self.renderer.render(
            self.state,
            action=self.last_action,
            info=self.last_info,
            step=self.step_count,
        )

    def close(self) -> None:
        """Release environment resources."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def _get_obs(self) -> np.ndarray:
        physics_config = self.physics_engine.config
        max_fuel_mass = max(self.physics_engine.max_fuel_mass, 1e-9)
        obs = np.array(
            [
                self.state.x / physics_config.world_x_limit,
                self.state.y / physics_config.world_y_limit,
                self.state.vx / self.env_config.max_speed,
                self.state.vy / self.env_config.max_speed,
                self.state.theta / self.env_config.max_angle,
                self.state.omega / self.env_config.max_angular_speed,
                self.state.fuel / max_fuel_mass,
                1.0 if self.state.legs_deployed else 0.0,
                self.state.stable_time / self.reward_config.required_stable_time,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _get_info(self) -> dict[str, Any]:
        return {
            **self.last_info,
            "state": self.state,
            "raw_state": np.array(
                [
                    self.state.x,
                    self.state.y,
                    self.state.vx,
                    self.state.vy,
                    self.state.theta,
                    self.state.omega,
                    self.state.fuel,
                    1.0 if self.state.legs_deployed else 0.0,
                    self.state.stable_time,
                ],
                dtype=np.float32,
            ),
        }

    def _compute_reward(self, normalized_thrust: float, done_reason: str) -> tuple[float, dict[str, float]]:
        physics_config = self.physics_engine.config
        target_vy = self._target_vertical_velocity()

        x_penalty = self.reward_config.x_weight * abs(self.state.x - self.reward_config.pad_x) / physics_config.world_x_limit
        vx_penalty = self.reward_config.vx_weight * abs(self.state.vx) / self.env_config.max_speed
        vy_penalty = self.reward_config.vy_weight * abs(self.state.vy - target_vy) / self.env_config.max_speed
        theta_penalty = self.reward_config.theta_weight * abs(self.state.theta) / self.env_config.max_angle
        omega_penalty = self.reward_config.omega_weight * abs(self.state.omega) / self.env_config.max_angular_speed
        fuel_penalty = self.reward_config.fuel_weight * float(normalized_thrust)
        time_penalty = self.reward_config.time_penalty
        terminal_reward = self._terminal_reward(done_reason)

        reward_terms = {
            "x_penalty": x_penalty,
            "vx_penalty": vx_penalty,
            "vy_penalty": vy_penalty,
            "theta_penalty": theta_penalty,
            "omega_penalty": omega_penalty,
            "fuel_penalty": fuel_penalty,
            "time_penalty": time_penalty,
            "terminal_reward": terminal_reward,
            "target_vy": target_vy,
        }
        reward = (
            -x_penalty
            - vx_penalty
            - vy_penalty
            - theta_penalty
            - omega_penalty
            - fuel_penalty
            - time_penalty
            + terminal_reward
        )
        return float(reward), reward_terms

    def _target_vertical_velocity(self) -> float:
        descent_rate = self.reward_config.target_descent_rate_gain * max(self.state.y, 0.0)
        descent_rate = min(
            self.reward_config.max_target_descent_rate,
            max(self.reward_config.min_target_descent_rate, descent_rate),
        )
        return -descent_rate

    def _terminal_reward(self, done_reason: str) -> float:
        if done_reason == "success":
            return self.reward_config.success_bonus
        if done_reason == "out_of_bounds":
            return self.reward_config.out_of_bounds_penalty
        if done_reason == "max_steps":
            return self.reward_config.max_steps_penalty
        if done_reason in {"missed_pad", "hard_landing", "tip_over", "body_contact", "one_foot_contact", "crash"}:
            return self.reward_config.crash_penalty
        return 0.0

    def _classify_terminal_state(
        self,
        physics_done_reason: str,
        terminated: bool,
        truncated: bool,
        physics_info: Mapping[str, Any],
    ) -> dict[str, Any]:
        if truncated:
            return {
                "done_reason": "max_steps",
                "is_success": False,
                "failure_flags": self._empty_failure_flags(),
                "terminated": False,
                "truncated": True,
            }

        contact = physics_info.get("contact", {})
        if physics_done_reason == "leg_contact":
            failure_flags = self._ground_contact_failure_flags(contact)
            if any(failure_flags.values()):
                return {
                    "done_reason": self._primary_failure_reason(failure_flags),
                    "is_success": False,
                    "failure_flags": failure_flags,
                    "terminated": True,
                    "truncated": False,
                }
            if self.state.stable_time >= self.reward_config.required_stable_time:
                return {
                    "done_reason": "success",
                    "is_success": True,
                    "failure_flags": failure_flags,
                    "terminated": True,
                    "truncated": False,
                }
            return {
                "done_reason": "leg_contact",
                "is_success": False,
                "failure_flags": failure_flags,
                "terminated": False,
                "truncated": False,
            }

        if not terminated:
            return {
                "done_reason": "running",
                "is_success": False,
                "failure_flags": self._empty_failure_flags(),
                "terminated": False,
                "truncated": False,
            }

        if physics_done_reason == "out_of_bounds":
            return {
                "done_reason": "out_of_bounds",
                "is_success": False,
                "failure_flags": {
                    **self._empty_failure_flags(),
                    "out_of_bounds": True,
                },
                "terminated": True,
                "truncated": False,
            }

        if physics_done_reason != "ground_contact":
            return {
                "done_reason": physics_done_reason,
                "is_success": False,
                "failure_flags": self._empty_failure_flags(),
                "terminated": True,
                "truncated": False,
            }

        failure_flags = self._ground_contact_failure_flags(contact)
        is_success = not any(failure_flags.values())
        if is_success:
            return {
                "done_reason": "success",
                "is_success": True,
                "failure_flags": failure_flags,
                "terminated": True,
                "truncated": False,
            }

        return {
            "done_reason": self._primary_failure_reason(failure_flags),
            "is_success": False,
            "failure_flags": failure_flags,
            "terminated": True,
            "truncated": False,
        }

    def _ground_contact_failure_flags(self, contact: Mapping[str, Any]) -> dict[str, bool]:
        return {
            "missed_pad": abs(self.state.x - self.reward_config.pad_x) > self.reward_config.landing_tolerance,
            "hard_landing": (
                abs(contact.get("contact_vx", self.state.vx)) > self.reward_config.max_touchdown_vx
                or abs(contact.get("contact_vy", self.state.vy)) > self.reward_config.max_touchdown_vy
            ),
            "tip_over": (
                abs(contact.get("contact_theta", self.state.theta)) > self.reward_config.max_touchdown_angle
                or abs(contact.get("contact_omega", self.state.omega)) > self.reward_config.max_touchdown_omega
            ),
            "body_contact": bool(contact.get("body_contact", False)),
            "one_foot_contact": self._one_foot_contact(contact),
            "out_of_bounds": False,
        }

    def _primary_failure_reason(self, failure_flags: Mapping[str, bool]) -> str:
        for reason in ("out_of_bounds", "body_contact", "missed_pad", "hard_landing", "tip_over", "one_foot_contact"):
            if failure_flags.get(reason, False):
                return reason
        return "crash"

    def _empty_failure_flags(self) -> dict[str, bool]:
        return {
            "missed_pad": False,
            "hard_landing": False,
            "tip_over": False,
            "body_contact": False,
            "one_foot_contact": False,
            "out_of_bounds": False,
        }

    def _empty_reward_terms(self) -> dict[str, float]:
        return {
            "x_penalty": 0.0,
            "vx_penalty": 0.0,
            "vy_penalty": 0.0,
            "theta_penalty": 0.0,
            "omega_penalty": 0.0,
            "fuel_penalty": 0.0,
            "time_penalty": 0.0,
            "terminal_reward": 0.0,
            "target_vy": 0.0,
        }

    def _one_foot_contact(self, contact: Mapping[str, Any]) -> bool:
        left = bool(contact.get("left_foot_contact", False))
        right = bool(contact.get("right_foot_contact", False))
        return left != right
