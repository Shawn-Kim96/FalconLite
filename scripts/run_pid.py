"""Run a PID-controller FalconLite rollout."""

import argparse

from falconlite.controllers import PIDController
from falconlite.env import RocketLandingEnv
from falconlite.utils.config import load_config
from falconlite.utils.logger import TelemetryLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a PID FalconLite rollout.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--scenario", default=None, help="Initial-state scenario from configs/default.yaml.")
    parser.add_argument("--render", action="store_true", help="Render the rollout with Pygame.")
    parser.add_argument("--log", action="store_true", help="Write telemetry CSV for the rollout.")
    parser.add_argument("--log-dir", default=None, help="Telemetry output directory.")
    parser.add_argument("--episode-id", type=int, default=1, help="Telemetry episode id.")
    return parser.parse_args()


def _run_episode(
    *,
    env: RocketLandingEnv,
    controller: PIDController,
    args: argparse.Namespace,
    logger: TelemetryLogger | None,
) -> tuple[int, float, dict]:
    reset_options = {"scenario": args.scenario} if args.scenario is not None else None
    _, last_info = env.reset(options=reset_options)
    controller.reset()
    if env.renderer is not None:
        env.renderer.reset_episode()

    executed_steps = 0
    total_reward = 0.0
    for step in range(args.steps):
        action = controller.select_action(last_info["state"], last_info)
        _, reward, terminated, truncated, last_info = env.step(action)
        total_reward += reward
        executed_steps = step + 1
        if logger is not None:
            logger.log_step(
                state=last_info["state"],
                action=last_info["actual_action"],
                reward=reward,
                info=last_info,
            )
        if args.render:
            env.render()
            if env.renderer is not None and env.renderer.closed:
                break
        if terminated or truncated:
            break

    return executed_steps, total_reward, last_info


def main() -> None:
    args = parse_args()
    config = load_config()
    controller = PIDController(config)
    env = RocketLandingEnv(config=config, render_mode="human" if args.render else None)
    log_config = config.get("logging", {})
    log_dir = args.log_dir or log_config.get("directory", "logs")

    episode_index = args.episode_id
    try:
        while True:
            logger = (
                TelemetryLogger(log_dir=log_dir, episode_id=episode_index)
                if args.log
                else None
            )
            try:
                executed_steps, total_reward, last_info = _run_episode(
                    env=env,
                    controller=controller,
                    args=args,
                    logger=logger,
                )
            finally:
                if logger is not None:
                    logger.close()

            state = last_info["state"]
            print("FalconLite PID rollout")
            print(f"project: {config['project']['name']}")
            print(f"scenario: {args.scenario or 'default'}")
            print(f"steps: {executed_steps}")
            print(f"done_reason: {last_info['done_reason']}")
            print(f"is_success: {last_info['is_success']}")
            print(f"total_reward: {total_reward:.3f}")
            print(
                "final_state: "
                f"x={state.x:.3f}, y={state.y:.3f}, vx={state.vx:.3f}, vy={state.vy:.3f}, "
                f"theta={state.theta:.3f}, omega={state.omega:.3f}, fuel={state.fuel:.3f}, "
                f"legs_deployed={state.legs_deployed}, stable_time={state.stable_time:.3f}"
            )
            if logger is not None:
                print(f"telemetry: {logger.path}")

            if not args.render or env.renderer is None or env.renderer.closed:
                break
            if not env.renderer.wait_for_rerun_or_close():
                break
            episode_index += 1
            print("--- rerunning ---")
    finally:
        env.close()


if __name__ == "__main__":
    main()
