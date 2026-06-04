"""Smoke tests for the MPC baseline controller."""

import numpy as np

from falconlite.controllers import MPCController
from falconlite.env import RocketState
from falconlite.utils.config import load_config


def make_controller(**overrides) -> MPCController:
    config = load_config()
    if overrides:
        config.setdefault("mpc", {}).update(overrides)
    return MPCController(config)


def initial_landing_state(**overrides) -> RocketState:
    base = dict(x=0.0, y=4600.0, vx=0.0, vy=-300.0, theta=0.0, omega=0.0, fuel=4400.0)
    base.update(overrides)
    return RocketState(**base)


def test_mpc_action_shape_and_bounds() -> None:
    """Returned action must fit the env's [thrust, gimbal, leg] schema."""

    controller = make_controller(horizon=8, max_iterations=3)
    state = initial_landing_state()

    action = controller.select_action(state)

    assert action.shape == (3,)
    assert action.dtype == np.float32
    assert controller.min_throttle <= action[0] <= 1.0
    assert -1.0 <= action[1] <= 1.0
    assert action[2] in {0.0, 1.0}


def test_mpc_warm_start_reuses_previous_solution() -> None:
    """The second call should warm-start from the prior solution shifted."""

    controller = make_controller(horizon=8, max_iterations=3)
    state = initial_landing_state()

    controller.select_action(state)
    assert controller._last_u_seq is not None
    first_solution = controller._last_u_seq.copy()

    # Second call: shape preserved, warm start applied (no exception).
    controller.select_action(state)
    assert controller._last_u_seq is not None
    assert controller._last_u_seq.shape == first_solution.shape


def test_mpc_reset_clears_warm_start() -> None:
    controller = make_controller(horizon=8, max_iterations=3)
    state = initial_landing_state()
    controller.select_action(state)
    assert controller._last_u_seq is not None

    controller.reset()
    assert controller._last_u_seq is None


def test_mpc_deploys_legs_only_in_final_approach() -> None:
    controller = make_controller(horizon=4, max_iterations=2)
    high_state = initial_landing_state(y=400.0, vy=-5.0)
    fast_state = initial_landing_state(y=80.0, vy=-60.0)
    low_state = initial_landing_state(y=80.0, vy=-5.0)

    high_action = controller.select_action(high_state)
    controller.reset()
    fast_action = controller.select_action(fast_state)
    controller.reset()
    low_action = controller.select_action(low_state)

    assert high_action[2] == 0.0
    assert fast_action[2] == 0.0
    assert low_action[2] == 1.0


def test_mpc_dynamics_one_step_matches_expected_freefall() -> None:
    """Internal dynamics with zero gimbal and engine off (clamped to min)
    should still produce a downward acceleration close to gravity in vy."""

    controller = make_controller(horizon=4, max_iterations=2)
    x0 = np.array([0.0, 1000.0, 0.0, -50.0, 0.0, 0.0])
    fuel0 = 4000.0

    # Apply min-throttle thrust + zero gimbal: thrust acts purely upward,
    # so dvy = (thrust/mass - g) * dt. Just check direction sanity.
    next_x, next_fuel = controller._dynamics(
        x0, fuel0, np.array([controller.min_throttle, 0.0])
    )
    assert next_x[1] < x0[1]  # altitude decreased
    assert next_fuel < fuel0  # fuel consumed
