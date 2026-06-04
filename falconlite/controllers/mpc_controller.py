"""Simple Model Predictive Control (MPC) baseline.

The controller plans an N-step thrust/gimbal sequence at every simulation
step, applies only the first action, and re-plans next step using the
remainder as a warm start.

This is the "Phase A" controller from the project's MPC plan: it uses a
simplified internal dynamics model (no drag, point-mass for thrust torque
mirroring physics.py) and a generic SciPy L-BFGS-B solver. It is meant to
demonstrate the MPC pattern; later phases will replace the optimizer with
convex/lossless-convexification solvers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize

from falconlite.env import EnvironmentConfig, PhysicsConfig, RewardConfig
from falconlite.env.state import RocketState


@dataclass(frozen=True)
class MPCWeights:
    """Cost weights. Tune these to change the controller's personality."""

    # Stage cost (per step) — pulls the rocket toward the pad gently.
    stage_pos_x: float = 0.5
    stage_pos_y: float = 0.0  # altitude is encoded by terminal cost
    stage_vel_x: float = 0.5
    stage_vel_y: float = 0.05
    stage_theta: float = 5.0
    stage_omega: float = 1.0
    # Modest thrust penalty: enough to discourage wasted burn but small
    # compared to the terminal-altitude cost so MPC still races toward the pad.
    stage_thrust: float = 0.05
    stage_gimbal: float = 0.5

    # Terminal cost (at horizon end) — drives precision touchdown. The
    # terminal_pos_y weight is intentionally heavy so the optimizer keeps
    # descending instead of stalling in mid-air.
    terminal_pos_x: float = 50.0
    terminal_pos_y: float = 5.0
    terminal_vel_x: float = 20.0
    terminal_vel_y: float = 5.0
    terminal_theta: float = 200.0
    terminal_omega: float = 50.0

    # Soft constraint penalty for going below the ground.
    ground_penalty: float = 1000.0


@dataclass(frozen=True)
class MPCConfig:
    """Top-level MPC settings."""

    horizon: int = 100  # 2.0 s lookahead at dt=0.02
    weights: MPCWeights = field(default_factory=MPCWeights)
    leg_deploy_altitude_m: float = 100.0
    leg_deploy_max_vy: float = 30.0
    max_iterations: int = 30  # L-BFGS-B inner iterations per step
    target_pad_x: float = 0.0
    # Touchdown reference altitude in meters (rocket center y at first contact
    # with ground). Equals the booster's engine_offset by geometry convention.
    target_touchdown_y: float = 20.5

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "MPCConfig":
        values = values or {}
        weights_values = values.get("weights") or {}
        allowed_weight_keys = MPCWeights.__dataclass_fields__.keys()
        weights = MPCWeights(
            **{k: float(weights_values[k]) for k in allowed_weight_keys if k in weights_values}
        )
        scalar_keys = {
            "horizon",
            "leg_deploy_altitude_m",
            "leg_deploy_max_vy",
            "max_iterations",
            "target_pad_x",
            "target_touchdown_y",
        }
        kwargs: dict[str, Any] = {key: values[key] for key in scalar_keys if key in values}
        if "horizon" in kwargs:
            kwargs["horizon"] = int(kwargs["horizon"])
        if "max_iterations" in kwargs:
            kwargs["max_iterations"] = int(kwargs["max_iterations"])
        return cls(weights=weights, **kwargs)


class MPCController:
    """Receding-horizon MPC controller.

    The internal prediction model is intentionally simpler than the full
    physics engine (no aerodynamic drag, no distributed forces) — MPC is
    robust to small plant-model mismatch because we re-plan every step
    with the freshly observed state.
    """

    NUM_CONTROLS = 2  # [thrust_norm in [0, 1], gimbal_norm in [-1, 1]]

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.physics_config = PhysicsConfig.from_mapping(config["physics"])
        self.env_config = EnvironmentConfig.from_mapping(config["environment"])
        self.reward_config = RewardConfig.from_mapping(config.get("reward", {}))
        self.mpc_config = MPCConfig.from_mapping(config.get("mpc"))

        self.dt = float(self.physics_config.dt)
        self.gravity = float(self.physics_config.gravity)
        self.max_thrust = float(self.physics_config.max_thrust)
        self.max_gimbal = float(self.physics_config.max_gimbal_angle)
        self.dry_mass = float(self.physics_config.dry_mass)
        self.max_fuel_mass = float(self.physics_config.mass - self.physics_config.dry_mass)
        self.inertia = float(self.physics_config.inertia)
        self.engine_offset = float(self.physics_config.geometry.engine_offset_m)
        self.fuel_burn_rate = float(self.physics_config.fuel_burn_rate)
        self.min_throttle = float(self.physics_config.min_throttle)

        # Drag-model parameters (single-point approximation of physics.py's
        # distributed drag). Sufficient for MPC to learn that "free fall is
        # naturally braked" without paying for 5-node summation per inner step.
        geom = self.physics_config.geometry
        self.air_density = float(self.physics_config.air_density_kg_m3)
        self.cd_axial = float(self.physics_config.drag_cd_axial)
        self.cd_side = float(self.physics_config.drag_cd_side)
        self.area_axial = math.pi * (geom.width_m / 2) ** 2
        self.area_side = geom.width_m * geom.height_m
        self.wind_x = float(self.physics_config.wind_x_mps)
        self.wind_y = float(self.physics_config.wind_y_mps)

        self.N = self.mpc_config.horizon
        self.weights = self.mpc_config.weights
        self.target_pad_x = float(
            self.mpc_config.target_pad_x
            if self.mpc_config.target_pad_x is not None
            else getattr(self.reward_config, "pad_x", 0.0)
        )
        self.target_touchdown_y = float(self.mpc_config.target_touchdown_y)

        # Box bounds: [thrust_norm in min_throttle..1, gimbal_norm in -1..1].
        # Phase A keeps the engine always on (no off-state) so the optimizer
        # works in a single convex box. Phase B will model the "off OR throttled-up"
        # union via lossless convexification.
        single_step_bounds = [
            (float(self.min_throttle), 1.0),
            (-1.0, 1.0),
        ]
        self._bounds = single_step_bounds * self.N

        self._last_u_seq: np.ndarray | None = None
        self.last_debug: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Required controller interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._last_u_seq = None
        self.last_debug = {}

    def select_action(
        self,
        state: RocketState,
        info: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        x0 = np.array(
            [state.x, state.y, state.vx, state.vy, state.theta, state.omega],
            dtype=np.float64,
        )
        fuel0 = float(state.fuel)

        u_seq_init = self._initial_guess()
        result = minimize(
            self._cost_flat,
            u_seq_init.flatten(),
            args=(x0, fuel0),
            method="L-BFGS-B",
            bounds=self._bounds,
            options={"maxiter": self.mpc_config.max_iterations},
        )
        u_seq = result.x.reshape(self.N, self.NUM_CONTROLS)
        self._last_u_seq = u_seq

        thrust_norm = float(np.clip(u_seq[0, 0], self.min_throttle, 1.0))
        gimbal_norm = float(np.clip(u_seq[0, 1], -1.0, 1.0))
        leg_deploy = self._leg_deploy_logic(state)

        self.last_debug = {
            "cost": float(result.fun),
            "iterations": int(result.nit),
            "converged": bool(result.success),
            "thrust_norm": thrust_norm,
            "gimbal_norm": gimbal_norm,
        }
        return np.array([thrust_norm, gimbal_norm, leg_deploy], dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal: warm start, dynamics, cost
    # ------------------------------------------------------------------

    def _initial_guess(self) -> np.ndarray:
        if self._last_u_seq is not None:
            shifted = np.vstack([self._last_u_seq[1:], self._last_u_seq[-1:]])
            return shifted

        # Cold start: hover thrust + zero gimbal, clipped into the feasible box.
        hover_norm = (self.physics_config.mass * self.gravity) / self.max_thrust
        hover_norm = float(np.clip(hover_norm, self.min_throttle, 1.0))
        u0 = np.array([hover_norm, 0.0])
        return np.tile(u0, (self.N, 1))

    def _normalized_thrust_to_force(self, thrust_norm: float) -> float:
        # Phase A: engine always on between min_throttle and 1.0 (no off state).
        return float(np.clip(thrust_norm, self.min_throttle, 1.0)) * self.max_thrust

    def _dynamics(
        self,
        x: np.ndarray,
        fuel: float,
        u: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """One-step Euler dynamics. Single-point approximation of the plant:
        thrust + gravity + orientation-aware aerodynamic drag. Drag torque
        from rotating-body cross-product is intentionally omitted (Phase A)."""

        thrust_norm = float(u[0])
        gimbal_norm = float(u[1])
        gimbal_angle = gimbal_norm * self.max_gimbal
        thrust = self._normalized_thrust_to_force(thrust_norm)

        # Mass clamp: prediction never assumes fuel below zero.
        fuel = max(0.0, min(fuel, self.max_fuel_mass))
        mass = self.dry_mass + fuel
        if fuel <= 0.0:
            thrust = 0.0

        theta = x[4]
        omega = x[5]
        thrust_angle = theta + gimbal_angle
        # Linear acceleration from thrust + gravity.
        a_x = thrust * math.sin(thrust_angle) / mass
        a_y = thrust * math.cos(thrust_angle) / mass - self.gravity

        # Aerodynamic drag (single-point approximation of physics' per-node
        # model). Projected area depends on the alignment of the body axis
        # with the velocity direction; in vertical free-fall this matches the
        # plant exactly, in side-on flight it's close enough for prediction.
        rel_vx = x[2] - self.wind_x
        rel_vy = x[3] - self.wind_y
        speed = math.hypot(rel_vx, rel_vy)
        if self.air_density > 0.0 and speed > 1e-9:
            dir_x = rel_vx / speed
            dir_y = rel_vy / speed
            body_axis_x = math.sin(theta)
            body_axis_y = math.cos(theta)
            axial_alignment = abs(body_axis_x * dir_x + body_axis_y * dir_y)
            side_alignment = abs(body_axis_x * dir_y - body_axis_y * dir_x)
            cda = (
                self.cd_axial * self.area_axial * axial_alignment
                + self.cd_side * self.area_side * side_alignment
            )
            drag_mag = 0.5 * self.air_density * speed * speed * cda
            a_x -= drag_mag * dir_x / mass
            a_y -= drag_mag * dir_y / mass

        # Angular acceleration: torque from gimballed engine acting at the
        # nozzle, r x F. Matches physics.py's analytical form.
        torque = -self.engine_offset * thrust * math.sin(gimbal_angle)
        a_theta = torque / self.inertia

        dt = self.dt
        next_x = np.array(
            [
                x[0] + x[2] * dt + 0.5 * a_x * dt * dt,
                x[1] + x[3] * dt + 0.5 * a_y * dt * dt,
                x[2] + a_x * dt,
                x[3] + a_y * dt,
                x[4] + omega * dt + 0.5 * a_theta * dt * dt,
                x[5] + a_theta * dt,
            ],
            dtype=np.float64,
        )
        fuel_used = self.fuel_burn_rate * (thrust / self.max_thrust) * dt if thrust > 0 else 0.0
        next_fuel = max(0.0, fuel - fuel_used)
        return next_x, next_fuel

    def _cost_flat(self, u_flat: np.ndarray, x0: np.ndarray, fuel0: float) -> float:
        u_seq = u_flat.reshape(self.N, self.NUM_CONTROLS)
        x = x0
        fuel = fuel0
        w = self.weights
        cost = 0.0
        pad_x = self.target_pad_x
        for t in range(self.N):
            u = u_seq[t]
            x, fuel = self._dynamics(x, fuel, u)
            # Stage cost.
            cost += w.stage_pos_x * (x[0] - pad_x) ** 2
            cost += w.stage_pos_y * (x[1] - self.target_touchdown_y) ** 2
            cost += w.stage_vel_x * x[2] ** 2
            cost += w.stage_vel_y * x[3] ** 2
            cost += w.stage_theta * x[4] ** 2
            cost += w.stage_omega * x[5] ** 2
            cost += w.stage_thrust * float(u[0]) ** 2
            cost += w.stage_gimbal * float(u[1]) ** 2
            # Soft ground constraint: penalize altitude below touchdown ref.
            ground_violation = max(0.0, self.target_touchdown_y - x[1])
            cost += w.ground_penalty * ground_violation ** 2

        # Terminal cost.
        cost += w.terminal_pos_x * (x[0] - pad_x) ** 2
        cost += w.terminal_pos_y * (x[1] - self.target_touchdown_y) ** 2
        cost += w.terminal_vel_x * x[2] ** 2
        cost += w.terminal_vel_y * x[3] ** 2
        cost += w.terminal_theta * x[4] ** 2
        cost += w.terminal_omega * x[5] ** 2
        return float(cost)

    def _leg_deploy_logic(self, state: RocketState) -> float:
        if state.legs_deployed:
            return 1.0
        if (
            state.y <= self.mpc_config.leg_deploy_altitude_m
            and abs(state.vy) <= self.mpc_config.leg_deploy_max_vy
        ):
            return 1.0
        return 0.0
