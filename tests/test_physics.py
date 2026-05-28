"""Stage 1 physics tests."""

from falconlite.env import PhysicsConfig, PhysicsEngine, RocketAction, RocketState


def make_engine(**overrides: float) -> PhysicsEngine:
    config = PhysicsConfig(
        dt=0.1,
        gravity=9.81,
        mass=1.0,
        dry_mass=0.7,
        inertia=0.05,
        engine_lever_arm=1.0,
        max_thrust=20.0,
        max_gimbal_angle=0.35,
        fuel_burn_rate=0.2,
        world_x_limit=100.0,
        world_y_limit=150.0,
        **overrides,
    )
    return PhysicsEngine(config)


def test_rocket_falls_under_gravity_without_thrust() -> None:
    engine = make_engine()
    state = RocketState(x=0, y=100, vx=0, vy=0, theta=0, omega=0, fuel=1)

    next_state, info = engine.step(state, RocketAction(thrust=0, gimbal_angle=0))

    assert next_state.vy < 0
    assert next_state.y < state.y
    assert info["acceleration"][1] < 0
    assert not info["terminated"]


def test_thrust_moves_rocket_upward() -> None:
    engine = make_engine()
    state = RocketState(x=0, y=100, vx=0, vy=0, theta=0, omega=0, fuel=1)

    next_state, info = engine.step(state, RocketAction(thrust=20, gimbal_angle=0))

    assert next_state.vy > 0
    assert next_state.y > state.y
    assert info["acceleration"][1] > 0


def test_gimbal_rotates_rocket() -> None:
    engine = make_engine()
    state = RocketState(x=0, y=100, vx=0, vy=0, theta=0, omega=0, fuel=1)

    next_state, info = engine.step(state, RocketAction(thrust=20, gimbal_angle=0.2))

    assert next_state.omega != 0
    assert next_state.theta != 0
    assert info["angular_acceleration"] != 0


def test_thrust_consumes_fuel() -> None:
    engine = make_engine()
    state = RocketState(x=0, y=100, vx=0, vy=0, theta=0, omega=0, fuel=1)

    next_state, info = engine.step(state, RocketAction(thrust=20, gimbal_angle=0))

    assert 0 <= next_state.fuel < state.fuel
    assert info["fuel_used"] > 0


def test_ground_collision_terminates_episode() -> None:
    engine = make_engine()
    state = RocketState(x=0, y=0.01, vx=0, vy=-10, theta=0, omega=0, fuel=1)

    next_state, info = engine.step(state, RocketAction(thrust=0, gimbal_angle=0))

    assert next_state.y == 0
    assert info["terminated"]
    assert info["done_reason"] == "ground_contact"


def test_actions_are_clamped_to_physical_limits() -> None:
    engine = make_engine()
    state = RocketState(x=0, y=100, vx=0, vy=0, theta=0, omega=0, fuel=1)

    _, info = engine.step(state, RocketAction(thrust=100, gimbal_angle=10))

    assert info["clamped_action"].thrust == engine.config.max_thrust
    assert info["clamped_action"].gimbal_angle == engine.config.max_gimbal_angle


def test_theta_is_wrapped_to_pi_range() -> None:
    engine = make_engine()
    state = RocketState(x=0, y=100, vx=0, vy=0, theta=4, omega=0, fuel=1)

    next_state, _ = engine.step(state, RocketAction(thrust=0, gimbal_angle=0))

    assert -3.141592653589793 <= next_state.theta <= 3.141592653589793
