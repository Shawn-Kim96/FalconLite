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
    """Meter-scale dimensions for a Falcon-9-style booster in side view.

    Coordinate convention (body frame):
    - Origin (0, 0) is the geometric center of the booster.
    - +y points toward the nose, -y toward the engine.
    - Body bottom (engine base) is at y = -height_m / 2.
    - Body top (nose) is at y = +height_m / 2.
    - The engine nozzle sits ``engine_offset_m`` below the origin; for a
      Falcon-9-style booster this equals height_m / 2.
    """

    height_m: float = 41.0
    width_m: float = 3.7
    engine_offset_m: float = 20.5
    nozzle_radius_m: float = 0.9
    leg_length_m: float = 9.0
    leg_span_m: float = 12.0
    leg_stow_angle_rad: float = 0.0
    leg_deploy_angle_rad: float = math.radians(35.0)
    # Grid fin offset is now measured upward from origin (body center).
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
        if self.grid_fin_offset_m > self.height_m / 2:
            raise ValueError("grid_fin_offset_m must not exceed height_m / 2.")
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
        return (0.0, self.height_m / 2)

    def body_bottom_positions_body(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Left and right body bottom corners in the body frame.

        These are the structural base of the booster (engine deck), at
        y = -height_m / 2.
        """
        radius = self.width_m / 2
        bottom_y = -self.height_m / 2
        return ((-radius, bottom_y), (radius, bottom_y))

    def body_top_positions_body(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Left and right body shoulder corners below the nose."""
        radius = self.width_m / 2
        nose_y = self.height_m / 2
        shoulder_y = nose_y - min(3.0, self.height_m * 0.08)
        return ((-radius, shoulder_y), (radius, shoulder_y))

    def body_outline_body(self) -> tuple[tuple[float, float], ...]:
        """Simple tapered booster body outline in body-frame coordinates."""
        left_bottom, right_bottom = self.body_bottom_positions_body()
        left_top, right_top = self.body_top_positions_body()
        return (left_bottom, right_bottom, right_top, self.nose_position_body(), left_top)

    def leg_hinge_positions_body(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Hinge points of the two side-view landing legs in the body frame.

        Hinges sit on the body wall a short distance above the engine base.
        With the body center at origin, base = -height/2, so the hinge sits
        slightly higher than that.
        """
        body_radius = self.width_m / 2
        bottom_y = -self.height_m / 2
        hinge_y = bottom_y + 0.5 * self.leg_length_m
        return ((-body_radius, hinge_y), (body_radius, hinge_y))

    def grid_fin_positions_body(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Hinge points of the left and right grid fins in the body frame."""
        x_offset = self.width_m / 2
        return ((-x_offset, self.grid_fin_offset_m), (x_offset, self.grid_fin_offset_m))

    def foot_positions_body(self, deployed: bool) -> tuple[tuple[float, float], tuple[float, float]]:
        """Tips of the two landing legs (left, right) in the body frame.

        When stowed, feet tuck flat against the booster wall and never reach
        below the engine base. When deployed, feet swing out to ``leg_span_m``
        total footprint width and extend slightly below the engine bell.
        """
        body_radius = self.width_m / 2
        bottom_y = -self.height_m / 2
        hinge_y = bottom_y + 0.5 * self.leg_length_m
        if deployed:
            angle = self.leg_deploy_angle_rad
            half_span = self.leg_span_m / 2
            foot_y = hinge_y - self.leg_length_m * math.cos(angle)
            return ((-half_span, foot_y), (half_span, foot_y))

        # Stowed: foot tucks flush with the body wall just above the base.
        foot_x = body_radius
        foot_y = max(bottom_y + 0.5, hinge_y - self.leg_length_m * math.cos(self.leg_stow_angle_rad))
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

    def aero_nodes_body(self) -> tuple[tuple[float, float, float, float], ...]:
        """Distributed aerodynamic sample points along the booster body axis.

        Returns a tuple of (x_body, y_body, side_area, axial_area) for 5 nodes
        evenly spaced from engine base (-height/2) to nose (+height/2). The
        sums of side_area and axial_area equal the full-body projected areas
        used by the legacy single-point drag model, so total drag is conserved
        when the booster is straight-line aligned. Off-axis motion produces
        non-trivial torques because each node's velocity differs by ``omega x r``.
        """

        node_count = 5
        bottom_y = -self.height_m / 2
        top_y = self.height_m / 2
        # Each interior node owns a vertical strip of length (height / (n - 1)).
        # End nodes own half-strips so the totals match a fully integrated area.
        strip = self.height_m / (node_count - 1)
        side_area_total = self.width_m * self.height_m
        axial_area_total = math.pi * (self.width_m / 2) ** 2

        nodes = []
        for index in range(node_count):
            y = bottom_y + index * strip
            # Endpoints: half-strip; middle nodes: full strip.
            strip_fraction = 0.5 if index in (0, node_count - 1) else 1.0
            side_area = self.width_m * strip * strip_fraction
            # Axial drag dominated by the booster end caps; concentrate on
            # engine deck (index 0) and nose (last index).
            if index == 0:
                axial_area = axial_area_total * 0.6
            elif index == node_count - 1:
                axial_area = axial_area_total * 0.4
            else:
                axial_area = 0.0
            nodes.append((0.0, y, side_area, axial_area))

        # Normalize side areas to match the closed-form total exactly so the
        # split version reproduces single-point drag in the linear regime.
        side_sum = sum(n[2] for n in nodes)
        side_scale = side_area_total / side_sum if side_sum > 0 else 1.0
        return tuple((x, y, sa * side_scale, ax) for (x, y, sa, ax) in nodes)

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
