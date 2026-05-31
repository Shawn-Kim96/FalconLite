"""Stage 7 evaluation tests."""

from falconlite.controllers import PIDController, RandomController
from falconlite.env import RocketLandingEnv
from falconlite.eval.metrics import episode_results_to_frame, summarize_episode_results
from falconlite.eval.rollout import EpisodeResult, run_episode
from falconlite.utils.config import load_config


def test_run_episode_returns_result_for_random_controller() -> None:
    config = load_config()
    env = RocketLandingEnv(config=config)

    try:
        result = run_episode(
            env=env,
            controller=RandomController(seed=0),
            controller_name="random",
            episode_id=1,
            seed=0,
        )
    finally:
        env.close()

    assert result.controller == "random"
    assert result.steps > 0
    assert result.done_reason in {
        "success",
        "missed_pad",
        "hard_landing",
        "tip_over",
        "body_contact",
        "one_foot_contact",
        "out_of_bounds",
        "max_steps",
    }


def test_run_episode_pid_can_succeed() -> None:
    config = load_config()
    env = RocketLandingEnv(config=config)

    try:
        result = run_episode(
            env=env,
            controller=PIDController(config),
            controller_name="pid",
            episode_id=1,
            seed=0,
        )
    finally:
        env.close()

    assert result.is_success
    assert result.done_reason == "success"


def test_run_episode_accepts_diagonal_scenario() -> None:
    config = load_config()
    env = RocketLandingEnv(config=config)

    try:
        result = run_episode(
            env=env,
            controller=PIDController(config),
            controller_name="pid",
            episode_id=1,
            seed=0,
            scenario="terminal_diagonal",
        )
    finally:
        env.close()

    assert result.is_success
    assert result.done_reason == "success"


def test_summarize_episode_results() -> None:
    results = [
        EpisodeResult(
            episode_id=1,
            controller="pid",
            steps=10,
            total_reward=100.0,
            done_reason="success",
            is_success=True,
            final_x=0.0,
            final_y=0.0,
            final_vx=0.0,
            final_vy=-1.0,
            final_theta=0.0,
            final_omega=0.0,
            final_fuel=0.8,
            fuel_used=0.2,
            touchdown_speed=1.0,
            max_tilt=0.1,
            missed_pad=False,
            hard_landing=False,
            tip_over=False,
            out_of_bounds=False,
        ),
        EpisodeResult(
            episode_id=2,
            controller="pid",
            steps=20,
            total_reward=-100.0,
            done_reason="hard_landing",
            is_success=False,
            final_x=0.0,
            final_y=0.0,
            final_vx=0.0,
            final_vy=-10.0,
            final_theta=0.0,
            final_omega=0.0,
            final_fuel=0.7,
            fuel_used=0.3,
            touchdown_speed=10.0,
            max_tilt=0.2,
            missed_pad=False,
            hard_landing=True,
            tip_over=False,
            out_of_bounds=False,
        ),
    ]

    metrics = summarize_episode_results(results)

    assert metrics["episodes"] == 2
    assert metrics["success_rate"] == 0.5
    assert metrics["crash_rate"] == 0.5
    assert metrics["hard_landing_rate"] == 0.5
    assert metrics["average_fuel_used"] == 0.25
    assert metrics["average_episode_length"] == 15


def test_episode_results_to_frame() -> None:
    result = EpisodeResult(
        episode_id=1,
        controller="random",
        steps=1,
        total_reward=0.0,
        done_reason="max_steps",
        is_success=False,
        final_x=0.0,
        final_y=1.0,
        final_vx=0.0,
        final_vy=0.0,
        final_theta=0.0,
        final_omega=0.0,
        final_fuel=1.0,
        fuel_used=0.0,
        touchdown_speed=0.0,
        max_tilt=0.0,
        missed_pad=False,
        hard_landing=False,
        tip_over=False,
        out_of_bounds=False,
    )

    frame = episode_results_to_frame([result])

    assert frame.loc[0, "controller"] == "random"
    assert frame.loc[0, "done_reason"] == "max_steps"
