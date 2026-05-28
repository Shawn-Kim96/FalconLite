"""Run a no-thrust free-fall rollout for physics debugging."""

import argparse

import numpy as np

from falconlite.env import RocketLandingEnv
from falconlite.utils.config import load_config
from falconlite.utils.logger import TelemetryLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a no-thrust FalconLite free-fall rollout.")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--render", action="store_true", help="Render the rollout with Pygame.")
    parser.add_argument("--log", action="store_true", help="Write telemetry CSV for the rollout.")
    parser.add_argument("--log-dir", default=None, help="Telemetry output directory.")
    parser.add_argument("--episode-id", type=int, default=99, help="Telemetry episode id.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    env = RocketLandingEnv(config=config, render_mode="human" if args.render else None)
    _, last_info = env.reset()
    log_config = config.get("logging", {})
    logger = (
        TelemetryLogger(
            log_dir=args.log_dir or log_config.get("directory", "logs"),
            episode_id=args.episode_id,
        )
        if args.log
        else None
    )

    action = np.array([0.0, 0.0], dtype=np.float32)
    executed_steps = 0
    total_reward = 0.0
    try:
        for step in range(args.steps):
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
    finally:
        if logger is not None:
            logger.close()
        env.close()

    state = last_info["state"]
    t = executed_steps * config["physics"]["dt"]
    expected_vy = -config["physics"]["gravity"] * t
    expected_y = max(0.0, config["environment"]["initial_state"]["y"] - 0.5 * config["physics"]["gravity"] * t * t)
    print("FalconLite no-thrust free-fall rollout")
    print(f"project: {config['project']['name']}")
    print(f"steps: {executed_steps}")
    print(f"time: {t:.3f}")
    print(f"done_reason: {last_info['done_reason']}")
    print(f"total_reward: {total_reward:.3f}")
    print(
        "final_state: "
        f"x={state.x:.3f}, y={state.y:.3f}, vx={state.vx:.3f}, vy={state.vy:.3f}, "
        f"theta={state.theta:.3f}, omega={state.omega:.3f}, fuel={state.fuel:.3f}"
    )
    print(f"analytic_freefall: y={expected_y:.3f}, vy={expected_vy:.3f}")
    if logger is not None:
        print(f"telemetry: {logger.path}")


if __name__ == "__main__":
    main()
