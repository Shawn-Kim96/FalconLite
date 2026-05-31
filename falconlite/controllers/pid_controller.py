"""PID baseline controller."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from falconlite.env import EnvironmentConfig, PhysicsConfig, RewardConfig
from falconlite.env.state import RocketState


@dataclass(frozen=True)
class PIDGains:
    """PID gains for one scalar control loop."""

    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "PIDGains":
        if values is None:
            return cls()
        allowed_keys = cls.__dataclass_fields__.keys()
        filtered = {key: values[key] for key in allowed_keys if key in values}
        return cls(**filtered)


@dataclass(frozen=True)
class PIDControllerConfig:
    """Configuration for the cascaded PID landing controller."""

    x: PIDGains
    theta: PIDGains
    descent: PIDGains
    max_target_angle: float = 0.35
    leg_deploy_altitude_m: float = 120.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "PIDControllerConfig":
        values = values or {}
        return cls(
            x=PIDGains.from_mapping(values.get("x")),
            theta=PIDGains.from_mapping(values.get("theta")),
            descent=PIDGains.from_mapping(values.get("descent")),
            max_target_angle=float(values.get("max_target_angle", 0.35)),
            leg_deploy_altitude_m=float(values.get("leg_deploy_altitude_m", 120.0)),
        )


class PIDController:
    """Cascaded baseline controller for the 2D landing task.

    The controller returns normalized Gym actions:
    - thrust command in [0, 1]
    - gimbal command in [-1, 1]
    - leg deploy command in {0, 1}
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.physics_config = PhysicsConfig.from_mapping(config["physics"])
        self.env_config = EnvironmentConfig.from_mapping(config["environment"])
        self.reward_config = RewardConfig.from_mapping(config.get("reward", {}))
        self.pid_config = PIDControllerConfig.from_mapping(config.get("pid", {}))
        self.x_integral = 0.0
        self.theta_integral = 0.0
        self.descent_integral = 0.0
        self.last_debug: dict[str, float] = {}

    def reset(self) -> None:
        """Clear integral state between episodes."""

        self.x_integral = 0.0
        self.theta_integral = 0.0
        self.descent_integral = 0.0
        self.last_debug = {}

    def select_action(self, state: RocketState, info: Mapping[str, Any] | None = None) -> np.ndarray:
        """Compute a normalized thrust/gimbal/leg command from the current state."""

        dt = self._dt_from_info(info)
        pad_error = self.reward_config.pad_x - state.x
        self.x_integral += pad_error * dt
        target_theta = (
            self.pid_config.x.kp * pad_error
            + self.pid_config.x.ki * self.x_integral
            - self.pid_config.x.kd * state.vx
        )
        target_theta = self._clip(
            target_theta,
            -self.pid_config.max_target_angle,
            self.pid_config.max_target_angle,
        )

        theta_error = target_theta - state.theta
        self.theta_integral += theta_error * dt
        gimbal_angle = (
            self.pid_config.theta.kp * theta_error
            + self.pid_config.theta.ki * self.theta_integral
            - self.pid_config.theta.kd * state.omega
        )
        gimbal_command = gimbal_angle / self.physics_config.max_gimbal_angle

        target_vy = self._target_vertical_velocity(state)
        descent_error = target_vy - state.vy
        self.descent_integral += descent_error * dt
        hover_command = self._hover_thrust_command(state)
        thrust_command = (
            hover_command
            + self.pid_config.descent.kp * descent_error
            + self.pid_config.descent.ki * self.descent_integral
        )
        leg_deploy_command = 1.0 if state.legs_deployed or state.y <= self.pid_config.leg_deploy_altitude_m else 0.0

        self.last_debug = {
            "pad_error": pad_error,
            "target_theta": target_theta,
            "theta_error": theta_error,
            "target_vy": target_vy,
            "descent_error": descent_error,
            "hover_command": hover_command,
            "raw_thrust_command": thrust_command,
            "raw_gimbal_command": gimbal_command,
            "leg_deploy_command": leg_deploy_command,
        }
        return np.array(
            [
                self._clip(thrust_command, 0.0, 1.0),
                self._clip(gimbal_command, -1.0, 1.0),
                leg_deploy_command,
            ],
            dtype=np.float32,
        )

    def _target_vertical_velocity(self, state: RocketState) -> float:
        descent_rate = self.reward_config.target_descent_rate_gain * max(state.y, 0.0)
        descent_rate = min(
            self.reward_config.max_target_descent_rate,
            max(self.reward_config.min_target_descent_rate, descent_rate),
        )
        return -descent_rate

    def _hover_thrust_command(self, state: RocketState) -> float:
        max_fuel_mass = self.physics_config.mass - self.physics_config.dry_mass
        fuel_mass = self._clip(state.fuel, 0.0, max_fuel_mass)
        mass = self.physics_config.dry_mass + fuel_mass
        hover_thrust = mass * self.physics_config.gravity
        return hover_thrust / self.physics_config.max_thrust

    def _dt_from_info(self, info: Mapping[str, Any] | None) -> float:
        if info is None:
            return self.physics_config.dt
        return float(info.get("dt", self.physics_config.dt))

    def _clip(self, value: float, low: float, high: float) -> float:
        return min(max(float(value), low), high)
