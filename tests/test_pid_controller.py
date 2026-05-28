"""Stage 6 PID controller tests."""

import numpy as np

from falconlite.controllers import PIDController
from falconlite.env import RocketState
from falconlite.utils.config import load_config


def make_controller() -> PIDController:
    return PIDController(load_config())


def test_pid_action_shape_and_bounds() -> None:
    controller = make_controller()
    state = RocketState(x=0, y=100, vx=0, vy=0, theta=0, omega=0, fuel=1)

    action = controller.select_action(state)

    assert action.shape == (2,)
    assert action.dtype == np.float32
    assert 0.0 <= action[0] <= 1.0
    assert -1.0 <= action[1] <= 1.0


def test_pid_hover_thrust_is_near_gravity_compensation() -> None:
    controller = make_controller()
    state = RocketState(x=0, y=10, vx=0, vy=-1, theta=0, omega=0, fuel=1)

    action = controller.select_action(state)
    expected_hover = controller.physics_config.gravity * controller.physics_config.mass / controller.physics_config.max_thrust

    assert abs(float(action[0]) - expected_hover) < 0.02
    assert controller.last_debug["descent_error"] == 0.0


def test_pid_lateral_error_tilts_toward_pad() -> None:
    controller = make_controller()
    state = RocketState(x=20, y=80, vx=0, vy=-5, theta=0, omega=0, fuel=1)

    action = controller.select_action(state)

    assert controller.last_debug["target_theta"] < 0
    assert action[1] < 0


def test_pid_attitude_error_commands_corrective_gimbal() -> None:
    controller = make_controller()
    state = RocketState(x=0, y=80, vx=0, vy=-5, theta=0.3, omega=0, fuel=1)

    action = controller.select_action(state)

    assert controller.last_debug["theta_error"] < 0
    assert action[1] < 0


def test_pid_falling_too_fast_increases_thrust() -> None:
    controller = make_controller()
    slow_state = RocketState(x=0, y=50, vx=0, vy=-4, theta=0, omega=0, fuel=1)
    fast_state = RocketState(x=0, y=50, vx=0, vy=-12, theta=0, omega=0, fuel=1)

    slow_action = controller.select_action(slow_state)
    controller.reset()
    fast_action = controller.select_action(fast_state)

    assert fast_action[0] > slow_action[0]
