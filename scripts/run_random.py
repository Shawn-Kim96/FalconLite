"""Run a short random-controller physics rollout."""

import argparse

from falconlite.controllers import RandomController
from falconlite.env import RocketLandingEnv
from falconlite.utils.config import load_config
from falconlite.utils.logger import TelemetryLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a random FalconLite physics rollout.")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--render", action="store_true", help="Render the rollout with Pygame.")
    parser.add_argument("--log", action="store_true", help="Write telemetry CSV for the rollout.")
    parser.add_argument("--log-dir", default=None, help="Telemetry output directory.")
    parser.add_argument("--episode-id", type=int, default=None, help="Telemetry episode id.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    seed = config["evaluation"]["seed"] if args.seed is None else args.seed
    controller = RandomController(seed=seed)
    env = RocketLandingEnv(config=config, render_mode="human" if args.render else None)
    _, last_info = env.reset(seed=seed)
    log_config = config.get("logging", {})
    logger = (
        TelemetryLogger(
            log_dir=args.log_dir or log_config.get("directory", "logs"),
            episode_id=args.episode_id or log_config.get("episode_id", 1),
        )
        if args.log
        else None
    )

    executed_steps = 0
    try:
        for step in range(args.steps):
            action = controller.select_action()
            _, reward, terminated, truncated, last_info = env.step(action)
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
    finally:
        if logger is not None:
            logger.close()
        env.close()

    state = last_info["state"]
    print("FalconLite Stage 6 random Gymnasium rollout")
    print(f"project: {config['project']['name']}")
    print(f"steps: {executed_steps}")
    print(f"done_reason: {last_info['done_reason']}")
    print(
        "final_state: "
        f"x={state.x:.3f}, y={state.y:.3f}, vx={state.vx:.3f}, vy={state.vy:.3f}, "
        f"theta={state.theta:.3f}, omega={state.omega:.3f}, fuel={state.fuel:.3f}"
    )
    if logger is not None:
        print(f"telemetry: {logger.path}")


if __name__ == "__main__":
    main()
