"""2D rocket physics for FalconLite Stage 1."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any

from falconlite.env.geometry import RocketGeometry
from falconlite.env.state import RocketAction, RocketState


@dataclass(frozen=True)
class PhysicsConfig:
    """Physical parameters for the 2D rocket model.

    Defaults model a returning Falcon 9 booster on its landing burn:
    dry mass ~25.6 t, landing-reserve propellant ~10 t, single Merlin
    throttled between ~40% and 100% of ~845 kN.
    """

    dt: float = 0.02
    gravity: float = 9.81
    mass: float = 35_600.0
    dry_mass: float = 25_600.0
    inertia: float = 3_600_000.0
    max_thrust: float = 845_000.0
    max_gimbal_angle: float = 0.087
    fuel_burn_rate: float = 280.0
    world_x_limit: float = 1_000.0
    world_y_limit: float = 2_500.0
    geometry: RocketGeometry = field(default_factory=RocketGeometry)

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt must be positive.")
        if self.mass <= 0:
            raise ValueError("mass must be positive.")
        if self.dry_mass <= 0 or self.dry_mass > self.mass:
            raise ValueError("dry_mass must be in (0, mass].")
        if self.inertia <= 0:
            raise ValueError("inertia must be positive.")
        if self.max_thrust <= 0:
            raise ValueError("max_thrust must be positive.")
        if self.max_gimbal_angle < 0:
            raise ValueError("max_gimbal_angle must be non-negative.")
        if self.fuel_burn_rate < 0:
            raise ValueError("fuel_burn_rate must be non-negative.")
        if self.world_x_limit <= 0 or self.world_y_limit <= 0:
            raise ValueError("world limits must be positive.")

    @property
    def engine_offset_m(self) -> float:
        return self.geometry.engine_offset_m

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "PhysicsConfig":
        """Build a config from a YAML-loaded mapping."""

        scalar_keys = {key for key in cls.__dataclass_fields__ if key != "geometry"}
        filtered: dict[str, Any] = {key: values[key] for key in scalar_keys if key in values}
        filtered["geometry"] = RocketGeometry.from_mapping(values.get("geometry"))
        return cls(**filtered)


class PhysicsEngine:
    """Minimal Newtonian 2D rocket dynamics.

    Coordinate convention:
    - x is horizontal position.
    - y is altitude, with ground at y = 0.
    - theta = 0 means the rocket body points upward.
    - gimbal_angle is relative to the rocket body.
    """

    def __init__(self, config: PhysicsConfig | Mapping[str, Any] | None = None) -> None:
        if config is None:
            self.config = PhysicsConfig()
        elif isinstance(config, PhysicsConfig):
            self.config = config
        else:
            self.config = PhysicsConfig.from_mapping(config)

    def action_from_normalized(self, command: Sequence[float]) -> RocketAction:
        """Convert normalized [thrust, gimbal] commands to physical units."""

        if len(command) != 2:
            raise ValueError("normalized command must contain thrust and gimbal values.")

        thrust_command = min(max(float(command[0]), 0.0), 1.0)
        gimbal_command = min(max(float(command[1]), -1.0), 1.0)
        return RocketAction(
            thrust=thrust_command * self.config.max_thrust,
            gimbal_angle=gimbal_command * self.config.max_gimbal_angle,
        )

    def step(self, state: RocketState, action: RocketAction | Sequence[float] | Mapping[str, Any]) -> tuple[RocketState, dict]:
        """Advance the rocket state by one fixed timestep."""

        requested_action = self._coerce_action(action)
        clamped_action = self._clamp_action(requested_action)
        actual_thrust, fuel_used = self._consume_fuel(state, clamped_action.thrust)
        actual_action = RocketAction(
            thrust=actual_thrust,
            gimbal_angle=clamped_action.gimbal_angle,
        )

        mass = self._current_mass(state)
        acceleration_x, acceleration_y = self._linear_acceleration(state, actual_action, mass)
        angular_acceleration = self._angular_acceleration(actual_action)

        dt = self.config.dt
        vx = state.vx + acceleration_x * dt
        vy = state.vy + acceleration_y * dt
        x = state.x + vx * dt
        y = state.y + vy * dt
        omega = state.omega + angular_acceleration * dt
        theta = self._wrap_angle(state.theta + omega * dt)
        fuel = max(0.0, state.fuel - fuel_used)

        next_state = RocketState(
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            theta=theta,
            omega=omega,
            fuel=fuel,
        )
        next_state, terminated, done_reason = self._apply_termination(next_state)

        info = {
            "requested_action": requested_action,
            "clamped_action": clamped_action,
            "actual_action": actual_action,
            "acceleration": (acceleration_x, acceleration_y),
            "angular_acceleration": angular_acceleration,
            "fuel_used": fuel_used,
            "mass": mass,
            "terminated": terminated,
            "done_reason": done_reason,
        }
        return next_state, info

    def _coerce_action(self, action: RocketAction | Sequence[float] | Mapping[str, Any]) -> RocketAction:
        if isinstance(action, RocketAction):
            return action
        if isinstance(action, Mapping):
            return RocketAction(
                thrust=float(action["thrust"]),
                gimbal_angle=float(action["gimbal_angle"]),
            )
        if len(action) != 2:
            raise ValueError("action must contain thrust and gimbal_angle.")
        return RocketAction(thrust=float(action[0]), gimbal_angle=float(action[1]))

    def _clamp_action(self, action: RocketAction) -> RocketAction:
        return RocketAction(
            thrust=min(max(action.thrust, 0.0), self.config.max_thrust),
            gimbal_angle=min(
                max(action.gimbal_angle, -self.config.max_gimbal_angle),
                self.config.max_gimbal_angle,
            ),
        )

    def _consume_fuel(self, state: RocketState, requested_thrust: float) -> tuple[float, float]:
        if state.fuel <= 0.0 or requested_thrust <= 0.0:
            return 0.0, 0.0

        full_step_fuel = self.config.fuel_burn_rate * (requested_thrust / self.config.max_thrust) * self.config.dt
        fuel_used = min(state.fuel, full_step_fuel)
        if full_step_fuel <= 0.0:
            return requested_thrust, 0.0

        thrust_scale = fuel_used / full_step_fuel
        return requested_thrust * thrust_scale, fuel_used

    def _current_mass(self, state: RocketState) -> float:
        fuel_fraction = min(max(state.fuel, 0.0), 1.0)
        fuel_mass = (self.config.mass - self.config.dry_mass) * fuel_fraction
        return self.config.dry_mass + fuel_mass

    def _linear_acceleration(self, state: RocketState, action: RocketAction, mass: float) -> tuple[float, float]:
        thrust_angle = state.theta + action.gimbal_angle
        thrust_x = action.thrust * math.sin(thrust_angle)
        thrust_y = action.thrust * math.cos(thrust_angle)
        return thrust_x / mass, thrust_y / mass - self.config.gravity

    def _angular_acceleration(self, action: RocketAction) -> float:
        torque = self.config.engine_offset_m * action.thrust * math.sin(action.gimbal_angle)
        return torque / self.config.inertia

    def _wrap_angle(self, angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _apply_termination(self, state: RocketState) -> tuple[RocketState, bool, str]:
        if state.y <= 0.0:
            grounded_state = RocketState(
                x=state.x,
                y=0.0,
                vx=state.vx,
                vy=state.vy,
                theta=state.theta,
                omega=state.omega,
                fuel=state.fuel,
            )
            return grounded_state, True, "ground_contact"

        out_of_bounds = abs(state.x) > self.config.world_x_limit or state.y > self.config.world_y_limit
        if out_of_bounds:
            return state, True, "out_of_bounds"

        return state, False, "running"
