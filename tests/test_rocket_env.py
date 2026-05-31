"""Stage 3 Gymnasium environment tests."""

import numpy as np

from falconlite.env import RocketLandingEnv


def leg_contact_cm_y(env: RocketLandingEnv) -> float:
    left_foot, right_foot = env.physics_engine.config.geometry.foot_positions_body(deployed=True)
    return -min(left_foot[1], right_foot[1])


def test_reset_returns_observation_and_info() -> None:
    env = RocketLandingEnv()

    try:
        obs, info = env.reset(seed=123)

        assert obs.shape == env.observation_space.shape
        assert env.observation_space.contains(obs)
        assert info["done_reason"] == "reset"
        assert info["state"].y == 4600.0
    finally:
        env.close()


def test_reset_accepts_named_initial_state_scenario() -> None:
    env = RocketLandingEnv()

    try:
        obs, info = env.reset(options={"scenario": "landing_burn_diagonal"})

        assert env.observation_space.contains(obs)
        assert info["state"].x == -250.0
        assert info["state"].vx == -40.0
        assert info["state"].vy == -300.0
    finally:
        env.close()


def test_step_accepts_sampled_action() -> None:
    env = RocketLandingEnv()

    try:
        env.reset(seed=123)
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

        assert obs.shape == env.observation_space.shape
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert info["step"] == 1
        assert "raw_state" in info
        assert "reward_terms" in info
        assert "failure_flags" in info
    finally:
        env.close()


def test_step_clips_actions_to_action_space() -> None:
    env = RocketLandingEnv()

    try:
        env.reset(seed=123)
        _, _, _, _, info = env.step(np.array([10.0, -10.0, 10.0], dtype=np.float32))

        np.testing.assert_allclose(info["normalized_action"], np.array([1.0, -1.0, 1.0], dtype=np.float32))
        assert info["clamped_action"].thrust == env.physics_engine.config.max_thrust
        assert info["clamped_action"].gimbal_angle == -env.physics_engine.config.max_gimbal_angle
        assert info["actual_action"].leg_deploy
    finally:
        env.close()


def test_max_steps_sets_truncated() -> None:
    config = {
        "physics": {
            "dt": 0.02,
            "gravity": 9.81,
            "mass": 1.0,
            "dry_mass": 0.7,
            "inertia": 0.05,
            "engine_lever_arm": 1.0,
            "max_thrust": 20.0,
            "max_gimbal_angle": 0.35,
            "fuel_burn_rate": 0.02,
            "world_x_limit": 100.0,
            "world_y_limit": 150.0,
        },
        "environment": {
            "max_steps": 1,
            "max_speed": 50.0,
            "max_angular_speed": 20.0,
            "max_angle": 3.141592653589793,
            "initial_state": {
                "x": 0.0,
                "y": 100.0,
                "vx": 0.0,
                "vy": 0.0,
                "theta": 0.0,
                "omega": 0.0,
                "fuel": 1.0,
            },
        },
        "render": {},
        "reward": {},
    }
    env = RocketLandingEnv(config=config)

    try:
        env.reset(seed=123)
        _, _, terminated, truncated, info = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))

        assert not terminated
        assert truncated
        assert info["done_reason"] == "max_steps"
        assert info["reward_terms"]["terminal_reward"] == env.reward_config.max_steps_penalty
    finally:
        env.close()


def test_render_mode_none_is_noop() -> None:
    env = RocketLandingEnv(render_mode=None)

    try:
        env.reset()
        env.render()
        assert env.renderer is None
    finally:
        env.close()


def test_deployed_leg_contact_success_requires_stability_window() -> None:
    env = RocketLandingEnv()

    try:
        env.reset(
            options={
                "initial_state": {
                    "x": 0.0,
                    "y": leg_contact_cm_y(env) + 0.01,
                    "vx": 0.0,
                    "vy": -0.1,
                    "theta": 0.0,
                    "omega": 0.0,
                    "fuel": 10_000.0,
                    "legs_deployed": True,
                }
            }
        )
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for _ in range(int(env.reward_config.required_stable_time / env.physics_engine.config.dt) + 2):
            _, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
            if terminated or truncated:
                break

        assert terminated
        assert not truncated
        assert reward > 0
        assert info["is_success"]
        assert info["done_reason"] == "success"
        assert not any(info["failure_flags"].values())
        assert info["state"].stable_time >= env.reward_config.required_stable_time
        assert info["reward_terms"]["terminal_reward"] == env.reward_config.success_bonus
    finally:
        env.close()


def test_ground_contact_hard_landing_is_classified() -> None:
    env = RocketLandingEnv()

    try:
        env.reset(
            options={
                "initial_state": {
                    "x": 0.0,
                    "y": leg_contact_cm_y(env) + 0.01,
                    "vx": 0.0,
                    "vy": -10.0,
                    "theta": 0.0,
                    "omega": 0.0,
                    "fuel": 10_000.0,
                    "legs_deployed": True,
                }
            }
        )
        _, reward, terminated, _, info = env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))

        assert terminated
        assert reward < 0
        assert not info["is_success"]
        assert info["done_reason"] == "hard_landing"
        assert info["failure_flags"]["hard_landing"]
    finally:
        env.close()


def test_ground_contact_missed_pad_has_priority() -> None:
    env = RocketLandingEnv()

    try:
        env.reset(
            options={
                "initial_state": {
                    "x": 40.0,
                    "y": leg_contact_cm_y(env) + 0.01,
                    "vx": 0.0,
                    "vy": -1.0,
                    "theta": 0.0,
                    "omega": 0.0,
                    "fuel": 10_000.0,
                    "legs_deployed": True,
                }
            }
        )
        _, _, terminated, _, info = env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))

        assert terminated
        assert not info["is_success"]
        assert info["done_reason"] == "missed_pad"
        assert info["failure_flags"]["missed_pad"]
    finally:
        env.close()


def test_ground_contact_tip_over_is_classified() -> None:
    env = RocketLandingEnv()

    try:
        env.reset(
            options={
                "initial_state": {
                    "x": 0.0,
                    "y": leg_contact_cm_y(env) + 0.01,
                    "vx": 0.0,
                    "vy": -1.0,
                    "theta": 0.5,
                    "omega": 0.0,
                    "fuel": 10_000.0,
                    "legs_deployed": True,
                }
            }
        )
        _, _, terminated, _, info = env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))

        assert terminated
        assert not info["is_success"]
        assert info["done_reason"] == "tip_over"
        assert info["failure_flags"]["tip_over"]
    finally:
        env.close()


def test_out_of_bounds_has_terminal_penalty() -> None:
    env = RocketLandingEnv()

    try:
        env.reset(
            options={
                "initial_state": {
                    "x": 1501.0,
                    "y": 100.0,
                    "vx": 0.0,
                    "vy": 0.0,
                    "theta": 0.0,
                    "omega": 0.0,
                    "fuel": 1.0,
                }
            }
        )
        _, reward, terminated, _, info = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))

        assert terminated
        assert reward < 0
        assert info["done_reason"] == "out_of_bounds"
        assert info["failure_flags"]["out_of_bounds"]
        assert info["reward_terms"]["terminal_reward"] == env.reward_config.out_of_bounds_penalty
    finally:
        env.close()
