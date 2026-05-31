"""State containers for FalconLite dynamics."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RocketState:
    """2D rocket state.

    Fuel is stored as remaining propellant mass in kilograms.
    """

    x: float
    y: float
    vx: float
    vy: float
    theta: float
    omega: float
    fuel: float
    legs_deployed: bool = False
    stable_time: float = 0.0


@dataclass(frozen=True)
class RocketAction:
    """Physical rocket action.

    Thrust is measured in Newtons. Gimbal angle is measured in radians.
    """

    thrust: float
    gimbal_angle: float
    leg_deploy: bool = False
