"""Meter-scale rocket geometry for FalconLite.

Side-view 2D convention (matches `physics.py`):
- Body frame: +y is along the rocket axis toward the nose; the engine sits at -y.
- Side projection shows ONE pair of legs and ONE pair of grid fins. The real
  Falcon 9 has four of each in a cross pattern, but the front/back pair overlaps
  the left/right pair in 2D, so modelling them as two avoids double-counting.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class RocketGeometry:
    """Meter-scale dimensions for a Falcon-9-style booster in side view."""

    height_m: float = 41.0
    width_m: float = 3.7
    engine_offset_m: float = 18.0
    nozzle_radius_m: float = 0.9
    leg_length_m: float = 9.0
    leg_span_m: float = 12.0
    leg_stow_angle_rad: float = 0.0
    leg_deploy_angle_rad: float = math.radians(35.0)
    grid_fin_offset_m: float = 18.0
    grid_fin_chord_m: float = 1.5
    grid_fin_span_m: float = 1.5

    def __post_init__(self) -> None:
        if self.height_m <= 0:
            raise ValueError("height_m must be positive.")
        if self.width_m <= 0:
            raise ValueError("width_m must be positive.")
        if self.engine_offset_m <= 0:
            raise ValueError("engine_offset_m must be positive.")
        if self.engine_offset_m > self.height_m:
            raise ValueError("engine_offset_m must not exceed height_m.")
        if self.nozzle_radius_m <= 0:
            raise ValueError("nozzle_radius_m must be positive.")
        if self.leg_length_m <= 0:
            raise ValueError("leg_length_m must be positive.")
        if self.leg_span_m <= 0:
            raise ValueError("leg_span_m must be positive.")
        if self.leg_stow_angle_rad < 0 or self.leg_deploy_angle_rad < 0:
            raise ValueError("leg angles must be non-negative.")
        if self.leg_deploy_angle_rad < self.leg_stow_angle_rad:
            raise ValueError("deploy angle must be >= stow angle.")
        if self.grid_fin_offset_m <= 0:
            raise ValueError("grid_fin_offset_m must be positive.")
        if self.grid_fin_offset_m > self.height_m:
            raise ValueError("grid_fin_offset_m must not exceed height_m.")
        if self.grid_fin_chord_m <= 0 or self.grid_fin_span_m <= 0:
            raise ValueError("grid fin dimensions must be positive.")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "RocketGeometry":
        if values is None:
            return cls()
        allowed_keys = cls.__dataclass_fields__.keys()
        filtered = {key: values[key] for key in allowed_keys if key in values}
        return cls(**filtered)

    def nozzle_position_body(self) -> tuple[float, float]:
        """Engine nozzle exit plane in the body frame."""
        return (0.0, -self.engine_offset_m)

    def nose_position_body(self) -> tuple[float, float]:
        """Nose tip in the body frame."""
        return (0.0, self.height_m - self.engine_offset_m)

    def body_bottom_positions_body(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Left and right body bottom corners in the body frame."""
        radius = self.width_m / 2
        return ((-radius, -self.engine_offset_m), (radius, -self.engine_offset_m))

    def body_top_positions_body(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Left and right body shoulder corners below the nose."""
        radius = self.width_m / 2
        nose_y = self.height_m - self.engine_offset_m
        shoulder_y = nose_y - min(3.0, self.height_m * 0.08)
        return ((-radius, shoulder_y), (radius, shoulder_y))

    def body_outline_body(self) -> tuple[tuple[float, float], ...]:
        """Simple tapered booster body outline in body-frame coordinates."""
        left_bottom, right_bottom = self.body_bottom_positions_body()
        left_top, right_top = self.body_top_positions_body()
        return (left_bottom, right_bottom, right_top, self.nose_position_body(), left_top)

    def leg_hinge_positions_body(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Hinge points of the two side-view landing legs in the body frame."""
        body_radius = self.width_m / 2
        hinge_y = -self.engine_offset_m + 0.5 * self.leg_length_m
        return ((-body_radius, hinge_y), (body_radius, hinge_y))

    def grid_fin_positions_body(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Hinge points of the left and right grid fins in the body frame."""
        x_offset = self.width_m / 2
        return ((-x_offset, self.grid_fin_offset_m), (x_offset, self.grid_fin_offset_m))

    def foot_positions_body(self, deployed: bool) -> tuple[tuple[float, float], tuple[float, float]]:
        """Tips of the two landing legs (left, right) in the body frame.

        When stowed, feet sit flush against the body just above the engine.
        When deployed, feet swing out to ``leg_span_m`` total footprint width
        and reach slightly below the engine bell.
        """
        angle = self.leg_deploy_angle_rad if deployed else self.leg_stow_angle_rad
        body_radius = self.width_m / 2
        hinge_y = -self.engine_offset_m + 0.5 * self.leg_length_m
        if deployed:
            half_span = self.leg_span_m / 2
            foot_y = hinge_y - self.leg_length_m * math.cos(angle)
            return ((-half_span, foot_y), (half_span, foot_y))

        foot_x = body_radius
        foot_y = hinge_y - self.leg_length_m
        return ((-foot_x, foot_y), (foot_x, foot_y))

    def body_to_world(
        self,
        point: tuple[float, float],
        *,
        x: float,
        y: float,
        theta: float,
    ) -> tuple[float, float]:
        """Transform a body-frame point into world coordinates."""
        point_x, point_y = point
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        return (
            x + point_x * cos_theta + point_y * sin_theta,
            y - point_x * sin_theta + point_y * cos_theta,
        )

    def points_world(
        self,
        *,
        x: float,
        y: float,
        theta: float,
        legs_deployed: bool,
    ) -> dict[str, tuple[float, float]]:
        """Return named geometry points in world coordinates."""
        left_foot, right_foot = self.foot_positions_body(legs_deployed)
        left_hinge, right_hinge = self.leg_hinge_positions_body()
        left_bottom, right_bottom = self.body_bottom_positions_body()
        left_top, right_top = self.body_top_positions_body()
        left_fin, right_fin = self.grid_fin_positions_body()
        body_points = {
            "nose": self.nose_position_body(),
            "nozzle": self.nozzle_position_body(),
            "left_body_bottom": left_bottom,
            "right_body_bottom": right_bottom,
            "left_body_top": left_top,
            "right_body_top": right_top,
            "left_leg_hinge": left_hinge,
            "right_leg_hinge": right_hinge,
            "left_foot": left_foot,
            "right_foot": right_foot,
            "left_grid_fin": left_fin,
            "right_grid_fin": right_fin,
        }
        return {
            name: self.body_to_world(point, x=x, y=y, theta=theta)
            for name, point in body_points.items()
        }
