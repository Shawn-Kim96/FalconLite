"""Run a no-thrust free-fall rollout for physics debugging."""

import argparse
import math

import numpy as np

from falconlite.env import RocketLandingEnv
from falconlite.utils.config import load_config
from falconlite.utils.logger import TelemetryLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a no-thrust FalconLite free-fall rollout.")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seconds", type=float, default=None, help="Run for this many simulated seconds instead of --steps.")
    parser.add_argument("--vacuum", action="store_true", help="Disable aerodynamic drag for exact vacuum free-fall.")
    parser.add_argument("--scenario", default=None, help="Initial-state scenario from configs/default.yaml.")
    parser.add_argument("--initial-y", type=float, default=None, help="Override initial center-of-mass altitude in meters.")
    parser.add_argument("--initial-vx", type=float, default=None, help="Override initial horizontal velocity in m/s.")
    parser.add_argument("--initial-vy", type=float, default=None, help="Override initial vertical velocity in m/s.")
    parser.add_argument("--initial-theta-deg", type=float, default=None, help="Override initial body angle in degrees.")
    parser.add_argument("--render", action="store_true", help="Render the rollout with Pygame.")
    parser.add_argument("--log", action="store_true", help="Write telemetry CSV for the rollout.")
    parser.add_argument("--log-dir", default=None, help="Telemetry output directory.")
    parser.add_argument("--episode-id", type=int, default=99, help="Telemetry episode id.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    if args.vacuum:
        config["physics"]["air_density_kg_m3"] = 0.0
    if args.seconds is not None:
        if args.seconds <= 0:
            raise ValueError("seconds must be positive.")
        args.steps = max(1, round(args.seconds / config["physics"]["dt"]))

    env = RocketLandingEnv(config=config, render_mode="human" if args.render else None)
    initial_override = {}
    if args.initial_y is not None:
        initial_override["y"] = args.initial_y
    if args.initial_vx is not None:
        initial_override["vx"] = args.initial_vx
    if args.initial_vy is not None:
        initial_override["vy"] = args.initial_vy
    if args.initial_theta_deg is not None:
        initial_override["theta"] = math.radians(args.initial_theta_deg)
    reset_options = {}
    if args.scenario is not None:
        reset_options["scenario"] = args.scenario
    if initial_override:
        reset_options["initial_state"] = initial_override
    reset_options = reset_options or None
    _, last_info = env.reset(options=reset_options)
    initial_state = last_info["state"]
    log_config = config.get("logging", {})
    logger = (
        TelemetryLogger(
            log_dir=args.log_dir or log_config.get("directory", "logs"),
            episode_id=args.episode_id,
        )
        if args.log
        else None
    )

    action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
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
    ground_cm_y = config["physics"]["geometry"]["engine_offset_m"]
    vacuum_vy = initial_state.vy - config["physics"]["gravity"] * t
    vacuum_y = max(ground_cm_y, initial_state.y + initial_state.vy * t - 0.5 * config["physics"]["gravity"] * t * t)
    vacuum_delta_y = state.y - vacuum_y
    vacuum_delta_vy = state.vy - vacuum_vy
    drag = last_info.get("drag", {})
    drag_acceleration = last_info.get("drag_acceleration", (0.0, 0.0))
    print("FalconLite no-thrust free-fall rollout")
    print(f"project: {config['project']['name']}")
    print(
        "initial_state: "
        f"y={initial_state.y:.3f}, vx={initial_state.vx:.3f}, vy={initial_state.vy:.3f}, "
        f"theta={initial_state.theta:.3f}"
    )
    print(f"air_density: {config['physics']['air_density_kg_m3']:.3f} kg/m^3")
    print(f"steps: {executed_steps}")
    print(f"time: {t:.3f}")
    print(f"done_reason: {last_info['done_reason']}")
    print(f"total_reward: {total_reward:.3f}")
    print(
        "final_state: "
        f"x={state.x:.3f}, y={state.y:.3f}, vx={state.vx:.3f}, vy={state.vy:.3f}, "
        f"theta={state.theta:.3f}, omega={state.omega:.3f}, fuel={state.fuel:.3f}, "
        f"legs_deployed={state.legs_deployed}, stable_time={state.stable_time:.3f}"
    )
    print(f"vacuum_reference: y={vacuum_y:.3f}, vy={vacuum_vy:.3f}")
    print(f"vacuum_delta: dy={vacuum_delta_y:.6f}, dvy={vacuum_delta_vy:.6f}")
    print(
        "drag_last: "
        f"area={drag.get('projected_area_m2', 0.0):.3f} m^2, "
        f"speed={drag.get('speed_mps', 0.0):.3f} m/s, "
        f"force=({drag.get('force_x', 0.0):.1f}, {drag.get('force_y', 0.0):.1f}) N, "
        f"accel=({drag_acceleration[0]:.4f}, {drag_acceleration[1]:.4f}) m/s^2"
    )
    if logger is not None:
        print(f"telemetry: {logger.path}")


if __name__ == "__main__":
    main()
