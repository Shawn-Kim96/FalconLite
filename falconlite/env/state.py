"""State containers for FalconLite dynamics."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RocketState:
    """Minimal 2D rocket state used by the simulator."""

    x: float
    y: float
    vx: float
    vy: float
    theta: float
    omega: float
    fuel: float


@dataclass(frozen=True)
class RocketAction:
    """Physical rocket action.

    Thrust is measured in Newtons. Gimbal angle is measured in radians.
    """

    thrust: float
    gimbal_angle: float
