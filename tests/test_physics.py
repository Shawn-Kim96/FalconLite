"""Stage 1 physics tests."""

from falconlite.env import PhysicsConfig, PhysicsEngine, RocketAction, RocketState


def make_engine(**overrides: float) -> PhysicsEngine:
    config = PhysicsConfig(
        dt=0.1,
        gravity=9.81,
        mass=35_600.0,
        dry_mass=25_600.0,
        inertia=3_600_000.0,
        max_thrust=845_000.0,
        max_gimbal_angle=0.087,
        fuel_burn_rate=280.0,
        world_x_limit=1_000.0,
        world_y_limit=2_500.0,
        **overrides,
    )
    return PhysicsEngine(config)


def test_rocket_falls_under_gravity_without_thrust() -> None:
    engine = make_engine()
    state = RocketState(x=0, y=2000, vx=0, vy=0, theta=0, omega=0, fuel=10_000)

    next_state, info = engine.step(state, RocketAction(thrust=0, gimbal_angle=0))

    assert next_state.vy < 0
    assert next_state.y < state.y
    assert info["acceleration"][1] < 0
    assert not info["terminated"]


def test_thrust_moves_rocket_upward() -> None:
    engine = make_engine()
    state = RocketState(x=0, y=2000, vx=0, vy=0, theta=0, omega=0, fuel=10_000)

    next_state, info = engine.step(state, RocketAction(thrust=845_000.0, gimbal_angle=0))

    assert next_state.vy > 0
    assert next_state.y > state.y
    assert info["acceleration"][1] > 0


def test_gimbal_rotates_rocket() -> None:
    engine = make_engine()
    state = RocketState(x=0, y=2000, vx=0, vy=0, theta=0, omega=0, fuel=10_000)

    next_state, info = engine.step(state, RocketAction(thrust=845_000.0, gimbal_angle=0.05))

    assert next_state.omega != 0
    assert next_state.theta != 0
    assert info["angular_acceleration"] != 0


def test_thrust_consumes_fuel() -> None:
    engine = make_engine()
    state = RocketState(x=0, y=2000, vx=0, vy=0, theta=0, omega=0, fuel=10_000)

    next_state, info = engine.step(state, RocketAction(thrust=845_000.0, gimbal_angle=0))

    assert 0 <= next_state.fuel < state.fuel
    assert info["fuel_used"] > 0


def test_ground_collision_terminates_episode() -> None:
    engine = make_engine()
    contact_cm_y = engine.config.geometry.engine_offset_m + 0.01
    state = RocketState(x=0, y=contact_cm_y, vx=0, vy=-10, theta=0, omega=0, fuel=10_000)

    next_state, info = engine.step(state, RocketAction(thrust=0, gimbal_angle=0))

    assert next_state.y == engine.config.geometry.engine_offset_m
    assert info["terminated"]
    assert info["done_reason"] == "ground_contact"
    assert info["contact"]["body_contact"]


def test_actions_are_clamped_to_physical_limits() -> None:
    engine = make_engine()
    state = RocketState(x=0, y=2000, vx=0, vy=0, theta=0, omega=0, fuel=10_000)

    _, info = engine.step(state, RocketAction(thrust=10_000_000.0, gimbal_angle=10))

    assert info["clamped_action"].thrust == engine.config.max_thrust
    assert info["clamped_action"].gimbal_angle == engine.config.max_gimbal_angle


def test_theta_is_wrapped_to_pi_range() -> None:
    engine = make_engine()
    state = RocketState(x=0, y=2000, vx=0, vy=0, theta=4, omega=0, fuel=10_000)

    next_state, _ = engine.step(state, RocketAction(thrust=0, gimbal_angle=0))

    assert -3.141592653589793 <= next_state.theta <= 3.141592653589793


def test_deployed_legs_create_nonterminal_ground_support() -> None:
    engine = make_engine()
    left_foot, right_foot = engine.config.geometry.foot_positions_body(deployed=True)
    contact_cm_y = -min(left_foot[1], right_foot[1]) + 0.01
    state = RocketState(
        x=0,
        y=contact_cm_y,
        vx=0,
        vy=-0.5,
        theta=0,
        omega=0,
        fuel=10_000,
        legs_deployed=True,
    )

    next_state, info = engine.step(state, RocketAction(thrust=0, gimbal_angle=0, leg_deploy=True))

    assert not info["terminated"]
    assert info["done_reason"] == "leg_contact"
    assert info["contact"]["foot_supported"]
    assert info["contact"]["left_foot_contact"]
    assert info["contact"]["right_foot_contact"]
    assert next_state.vy == 0
    assert next_state.stable_time == engine.config.dt


def test_drag_opposes_downward_velocity() -> None:
    vacuum_engine = make_engine(air_density_kg_m3=0.0)
    drag_engine = make_engine(air_density_kg_m3=1.225)
    state = RocketState(x=0, y=2000, vx=0, vy=-100, theta=0, omega=0, fuel=10_000)

    vacuum_state, vacuum_info = vacuum_engine.step(state, RocketAction(thrust=0, gimbal_angle=0))
    drag_state, drag_info = drag_engine.step(state, RocketAction(thrust=0, gimbal_angle=0))

    assert drag_state.vy > vacuum_state.vy
    assert drag_info["drag"]["force_y"] > 0
    assert drag_info["drag_acceleration"][1] > 0
    assert vacuum_info["drag"]["force_y"] == 0


def test_distributed_drag_damps_angular_rate() -> None:
    """A spinning, falling booster should pick up a drag torque opposing the
    spin (per-node velocity = v_cm + omega x r differs along the body, so the
    summed r x F is non-zero and points against omega)."""

    engine = make_engine(air_density_kg_m3=1.225)
    state = RocketState(x=0, y=2000, vx=0, vy=-100, theta=0, omega=0.5, fuel=10_000)

    _, info = engine.step(state, RocketAction(thrust=0, gimbal_angle=0))

    assert info["drag_torque"] < 0  # opposes positive spin


def test_sideways_body_has_more_drag_force_than_upright_body() -> None:
    """With distributed per-node drag, the side-on attitude exposes much more
    side area to the airstream and produces a much larger upward drag force."""

    engine = make_engine(air_density_kg_m3=1.225)
    upright = RocketState(x=0, y=2000, vx=0, vy=-100, theta=0, omega=0, fuel=10_000)
    sideways = RocketState(x=0, y=2000, vx=0, vy=-100, theta=1.5707963267948966, omega=0, fuel=10_000)

    _, upright_info = engine.step(upright, RocketAction(thrust=0, gimbal_angle=0))
    _, sideways_info = engine.step(sideways, RocketAction(thrust=0, gimbal_angle=0))

    assert sideways_info["drag"]["force_y"] > upright_info["drag"]["force_y"]
