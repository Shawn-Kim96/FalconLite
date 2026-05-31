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
    min_throttle: float = 0.40
    engine_off_threshold: float = 0.05
    max_gimbal_angle: float = 0.087
    fuel_burn_rate: float = 308.0
    ground_contact_tolerance_m: float = 0.25
    air_density_kg_m3: float = 1.225
    drag_cd_axial: float = 0.65
    drag_cd_side: float = 1.05
    wind_x_mps: float = 0.0
    wind_y_mps: float = 0.0
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
        if not 0.0 <= self.min_throttle < 1.0:
            raise ValueError("min_throttle must be in [0, 1).")
        if not 0.0 <= self.engine_off_threshold <= 1.0:
            raise ValueError("engine_off_threshold must be in [0, 1].")
        if self.max_gimbal_angle < 0:
            raise ValueError("max_gimbal_angle must be non-negative.")
        if self.fuel_burn_rate < 0:
            raise ValueError("fuel_burn_rate must be non-negative.")
        if self.ground_contact_tolerance_m < 0:
            raise ValueError("ground_contact_tolerance_m must be non-negative.")
        if self.air_density_kg_m3 < 0:
            raise ValueError("air_density_kg_m3 must be non-negative.")
        if self.drag_cd_axial < 0 or self.drag_cd_side < 0:
            raise ValueError("drag coefficients must be non-negative.")
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
        """Convert normalized [thrust, gimbal, leg_deploy] commands to physical units."""

        if len(command) not in {2, 3}:
            raise ValueError("normalized command must contain thrust, gimbal, and optional leg deploy values.")

        thrust_command = min(max(float(command[0]), 0.0), 1.0)
        gimbal_command = min(max(float(command[1]), -1.0), 1.0)
        leg_deploy = bool(float(command[2]) >= 0.5) if len(command) == 3 else False
        return RocketAction(
            thrust=self._normalized_thrust_to_force(thrust_command),
            gimbal_angle=gimbal_command * self.config.max_gimbal_angle,
            leg_deploy=leg_deploy,
        )

    def _normalized_thrust_to_force(self, command: float) -> float:
        """Map a [0, 1] command to physical thrust honoring min_throttle.

        Below ``engine_off_threshold`` the engine is off (thrust = 0). Above it,
        the command is scaled into ``[min_throttle, 1.0]`` of ``max_thrust``,
        modeling Merlin-class lower throttle limits.
        """

        if command < self.config.engine_off_threshold:
            return 0.0
        span = 1.0 - self.config.engine_off_threshold
        if span <= 0.0:
            scaled = 1.0
        else:
            scaled = (command - self.config.engine_off_threshold) / span
        throttle = self.config.min_throttle + (1.0 - self.config.min_throttle) * scaled
        return throttle * self.config.max_thrust

    def step(self, state: RocketState, action: RocketAction | Sequence[float] | Mapping[str, Any]) -> tuple[RocketState, dict]:
        """Advance the rocket state by one fixed timestep."""

        requested_action = self._coerce_action(action)
        clamped_action = self._clamp_action(requested_action)
        actual_thrust, fuel_used = self._consume_fuel(state, clamped_action.thrust)
        actual_action = RocketAction(
            thrust=actual_thrust,
            gimbal_angle=clamped_action.gimbal_angle,
            leg_deploy=state.legs_deployed or clamped_action.leg_deploy,
        )

        mass = self._current_mass(state)
        base_acceleration_x, base_acceleration_y = self._linear_acceleration(state, actual_action, mass)
        drag_force_x, drag_force_y, drag_torque, drag = self._drag_force(state)
        drag_acceleration_x = drag_force_x / mass
        drag_acceleration_y = drag_force_y / mass
        acceleration_x = base_acceleration_x + drag_acceleration_x
        acceleration_y = base_acceleration_y + drag_acceleration_y
        angular_acceleration = (
            self._angular_acceleration(state, actual_action)
            + drag_torque / self.config.inertia
        )

        dt = self.config.dt
        vx = state.vx + acceleration_x * dt
        vy = state.vy + acceleration_y * dt
        x = state.x + state.vx * dt + 0.5 * acceleration_x * dt * dt
        y = state.y + state.vy * dt + 0.5 * acceleration_y * dt * dt
        omega = state.omega + angular_acceleration * dt
        theta = self._wrap_angle(state.theta + state.omega * dt + 0.5 * angular_acceleration * dt * dt)
        fuel = max(0.0, state.fuel - fuel_used)
        legs_deployed = state.legs_deployed or actual_action.leg_deploy

        next_state = RocketState(
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            theta=theta,
            omega=omega,
            fuel=fuel,
            legs_deployed=legs_deployed,
            stable_time=state.stable_time,
        )
        next_state, terminated, done_reason, contact = self._apply_termination(next_state)

        info = {
            "requested_action": requested_action,
            "clamped_action": clamped_action,
            "actual_action": actual_action,
            "acceleration": (acceleration_x, acceleration_y),
            "base_acceleration": (base_acceleration_x, base_acceleration_y),
            "drag_acceleration": (drag_acceleration_x, drag_acceleration_y),
            "drag": drag,
            "drag_torque": drag_torque,
            "angular_acceleration": angular_acceleration,
            "fuel_used": fuel_used,
            "mass": mass,
            "terminated": terminated,
            "done_reason": done_reason,
            "contact": contact,
        }
        return next_state, info

    def _coerce_action(self, action: RocketAction | Sequence[float] | Mapping[str, Any]) -> RocketAction:
        if isinstance(action, RocketAction):
            return action
        if isinstance(action, Mapping):
            return RocketAction(
                thrust=float(action["thrust"]),
                gimbal_angle=float(action["gimbal_angle"]),
                leg_deploy=bool(action.get("leg_deploy", False)),
            )
        if len(action) not in {2, 3}:
            raise ValueError("action must contain thrust, gimbal_angle, and optional leg_deploy.")
        return RocketAction(
            thrust=float(action[0]),
            gimbal_angle=float(action[1]),
            leg_deploy=bool(float(action[2]) >= 0.5) if len(action) == 3 else False,
        )

    def _clamp_action(self, action: RocketAction) -> RocketAction:
        return RocketAction(
            thrust=self._clamp_thrust(action.thrust),
            gimbal_angle=min(
                max(action.gimbal_angle, -self.config.max_gimbal_angle),
                self.config.max_gimbal_angle,
            ),
            leg_deploy=action.leg_deploy,
        )

    def _clamp_thrust(self, thrust: float) -> float:
        """Clamp a physical thrust value to {0} ∪ [min_throttle·F_max, F_max].

        Mirrors the engine off-or-throttled-up envelope so direct
        ``RocketAction`` construction (e.g. dict actions in tests) cannot
        bypass the ``min_throttle`` constraint.
        """

        thrust = min(max(float(thrust), 0.0), self.config.max_thrust)
        min_thrust = self.config.min_throttle * self.config.max_thrust
        if thrust <= 0.0:
            return 0.0
        if thrust < min_thrust:
            return 0.0
        return thrust

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
        fuel_mass = min(max(state.fuel, 0.0), self.max_fuel_mass)
        return self.config.dry_mass + fuel_mass

    def _linear_acceleration(self, state: RocketState, action: RocketAction, mass: float) -> tuple[float, float]:
        thrust_angle = state.theta + action.gimbal_angle
        thrust_x = action.thrust * math.sin(thrust_angle)
        thrust_y = action.thrust * math.cos(thrust_angle)
        return thrust_x / mass, thrust_y / mass - self.config.gravity

    def _drag_force(
        self, state: RocketState
    ) -> tuple[float, float, float, dict[str, float]]:
        """Distributed aerodynamic drag.

        Sums drag at each body-axis node. Each node has its own velocity
        (``v_cm + omega x r``) and its own slice of the booster's projected
        area. Output is (Fx, Fy, torque, debug_info). The torque comes from
        the cross product r x F summed over nodes — this is what produces
        passive aerodynamic stabilization when the booster is misaligned with
        its velocity vector.
        """

        if self.config.air_density_kg_m3 <= 0.0:
            return 0.0, 0.0, 0.0, self._empty_drag_info()

        geometry = self.config.geometry
        nodes = geometry.aero_nodes_body()
        cos_theta = math.cos(state.theta)
        sin_theta = math.sin(state.theta)
        body_axis_x = sin_theta
        body_axis_y = cos_theta
        wind_x = self.config.wind_x_mps
        wind_y = self.config.wind_y_mps

        total_fx = 0.0
        total_fy = 0.0
        total_torque = 0.0
        cm_speed = math.hypot(state.vx - wind_x, state.vy - wind_y)
        for node_x_body, node_y_body, side_area, axial_area in nodes:
            # Body-frame point to world relative offset (rotation only).
            r_x = node_x_body * cos_theta + node_y_body * sin_theta
            r_y = -node_x_body * sin_theta + node_y_body * cos_theta

            # Node velocity = v_cm + omega x r (2D cross: omega x r = (-omega*r_y, omega*r_x)).
            node_vx = state.vx - state.omega * r_y
            node_vy = state.vy + state.omega * r_x

            rel_vx = node_vx - wind_x
            rel_vy = node_vy - wind_y
            speed = math.hypot(rel_vx, rel_vy)
            if speed <= 1e-9:
                continue

            dir_x = rel_vx / speed
            dir_y = rel_vy / speed
            axial_alignment = abs(body_axis_x * dir_x + body_axis_y * dir_y)
            side_alignment = abs(body_axis_x * dir_y - body_axis_y * dir_x)

            cda = (
                self.config.drag_cd_axial * axial_area * axial_alignment
                + self.config.drag_cd_side * side_area * side_alignment
            )
            drag_mag = 0.5 * self.config.air_density_kg_m3 * speed * speed * cda
            fx = -drag_mag * dir_x
            fy = -drag_mag * dir_y
            total_fx += fx
            total_fy += fy
            # Torque about CM: r x F (2D scalar = r_x * F_y - r_y * F_x).
            total_torque += r_x * fy - r_y * fx

        info = {
            "force_x": total_fx,
            "force_y": total_fy,
            "torque": total_torque,
            "speed_mps": cm_speed,
            "projected_area_m2": sum(n[2] for n in nodes),
            "cda_m2": 0.0,
            "axial_alignment": 0.0,
            "side_alignment": 0.0,
        }
        return total_fx, total_fy, total_torque, info

    def _empty_drag_info(self) -> dict[str, float]:
        return {
            "force_x": 0.0,
            "force_y": 0.0,
            "torque": 0.0,
            "speed_mps": 0.0,
            "projected_area_m2": 0.0,
            "cda_m2": 0.0,
            "axial_alignment": 1.0,
            "side_alignment": 0.0,
        }

    def _angular_acceleration(self, state: RocketState, action: RocketAction) -> float:
        """Torque from the engine thrust about the booster's center of mass.

        Thrust is applied at the engine node (body coords (0, -engine_offset)),
        directed by ``theta + gimbal_angle`` in world coords. The torque is the
        2D cross product r x F, where r is the engine node position relative
        to the CM in world coords.
        """

        cos_theta = math.cos(state.theta)
        sin_theta = math.sin(state.theta)
        # Engine node body-frame position rotated to world-relative.
        node_y_body = -self.config.engine_offset_m
        r_x = node_y_body * sin_theta
        r_y = node_y_body * cos_theta
        thrust_angle = state.theta + action.gimbal_angle
        f_x = action.thrust * math.sin(thrust_angle)
        f_y = action.thrust * math.cos(thrust_angle)
        torque = r_x * f_y - r_y * f_x
        return torque / self.config.inertia

    def _wrap_angle(self, angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi

    @property
    def max_fuel_mass(self) -> float:
        return self.config.mass - self.config.dry_mass

    def _apply_termination(self, state: RocketState) -> tuple[RocketState, bool, str, dict[str, Any]]:
        initial_contact = self._contact_info(state)
        if initial_contact["has_ground_contact"]:
            contact_velocity = {
                "contact_vx": state.vx,
                "contact_vy": state.vy,
                "contact_theta": state.theta,
                "contact_omega": state.omega,
            }
            shift_y = -initial_contact["lowest_y"]
            shifted_state = RocketState(
                x=state.x,
                y=state.y + shift_y,
                vx=state.vx,
                vy=state.vy,
                theta=state.theta,
                omega=state.omega,
                fuel=state.fuel,
                legs_deployed=state.legs_deployed,
                stable_time=state.stable_time,
            )
            contact = self._contact_info(shifted_state)
            both_feet_contact = contact["left_foot_contact"] and contact["right_foot_contact"]
            foot_supported = state.legs_deployed and both_feet_contact and not contact["body_contact"]
            stable_time = state.stable_time + self.config.dt if foot_supported else 0.0
            grounded_state = RocketState(
                x=shifted_state.x,
                y=shifted_state.y,
                vx=0.0 if foot_supported else state.vx,
                vy=0.0 if foot_supported else state.vy,
                theta=state.theta,
                omega=0.0 if foot_supported else state.omega,
                fuel=state.fuel,
                legs_deployed=state.legs_deployed,
                stable_time=stable_time,
            )
            contact = {
                **contact,
                "ground_shift_y": -initial_contact["lowest_y"],
                "foot_supported": foot_supported,
                **contact_velocity,
            }
            if foot_supported:
                return grounded_state, False, "leg_contact", contact
            return grounded_state, True, "ground_contact", contact

        out_of_bounds = abs(state.x) > self.config.world_x_limit or state.y > self.config.world_y_limit
        if out_of_bounds:
            return state, True, "out_of_bounds", initial_contact

        if state.stable_time > 0.0:
            state = RocketState(
                x=state.x,
                y=state.y,
                vx=state.vx,
                vy=state.vy,
                theta=state.theta,
                omega=state.omega,
                fuel=state.fuel,
                legs_deployed=state.legs_deployed,
                stable_time=0.0,
            )
        return state, False, "running", initial_contact

    def _contact_info(self, state: RocketState) -> dict[str, Any]:
        points = self.config.geometry.points_world(
            x=state.x,
            y=state.y,
            theta=state.theta,
            legs_deployed=state.legs_deployed,
        )
        body_names = (
            "nose",
            "left_body_bottom",
            "right_body_bottom",
            "left_body_top",
            "right_body_top",
        )
        foot_names = ("left_foot", "right_foot")
        relevant_names = (*body_names, *foot_names) if state.legs_deployed else body_names
        lowest_y = min(points[name][1] for name in relevant_names)
        foot_contact_y = self.config.ground_contact_tolerance_m
        contact_tolerance = foot_contact_y if state.legs_deployed else 0.0
        has_ground_contact = lowest_y <= contact_tolerance
        left_foot_contact = state.legs_deployed and has_ground_contact and points["left_foot"][1] <= foot_contact_y
        right_foot_contact = state.legs_deployed and has_ground_contact and points["right_foot"][1] <= foot_contact_y
        body_contact = any(points[name][1] <= 0.0 for name in body_names)
        return {
            "points": points,
            "lowest_y": lowest_y,
            "has_ground_contact": has_ground_contact,
            "left_foot_contact": left_foot_contact,
            "right_foot_contact": right_foot_contact,
            "body_contact": body_contact,
            "foot_supported": False,
            "ground_shift_y": 0.0,
            "contact_vx": state.vx,
            "contact_vy": state.vy,
            "contact_theta": state.theta,
            "contact_omega": state.omega,
        }
