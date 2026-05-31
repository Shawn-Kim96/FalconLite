"""Evaluate FalconLite controllers over multiple episodes."""

import argparse
from pathlib import Path

from falconlite.controllers import PIDController, RandomController
from falconlite.env import RocketLandingEnv
from falconlite.eval.metrics import episode_results_to_frame, metrics_to_frame, summarize_episode_results
from falconlite.eval.rollout import run_episode
from falconlite.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FalconLite controllers.")
    parser.add_argument("--controller", default="random", choices=["random", "pid"])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scenario", default=None, help="Initial-state scenario from configs/default.yaml.")
    parser.add_argument("--log", action="store_true", help="Write telemetry CSV for each episode.")
    parser.add_argument("--log-dir", default=None, help="Telemetry output directory.")
    parser.add_argument("--summary-csv", default=None, help="Optional aggregate metrics CSV path.")
    parser.add_argument("--episodes-csv", default=None, help="Optional per-episode results CSV path.")
    return parser.parse_args()


def make_controller(name: str, config: dict, seed: int):
    if name == "random":
        return RandomController(seed=seed)
    if name == "pid":
        return PIDController(config)
    raise ValueError(f"Unsupported controller: {name}")


def main() -> None:
    args = parse_args()
    config = load_config()
    episodes = args.episodes or config["evaluation"]["episodes"]
    seed = args.seed if args.seed is not None else config["evaluation"]["seed"]
    if episodes <= 0:
        raise ValueError("episodes must be positive.")

    controller = make_controller(args.controller, config, seed)
    results = []
    log_dir = args.log_dir or config.get("logging", {}).get("directory", "logs")
    for episode_index in range(episodes):
        env = RocketLandingEnv(config=config)
        try:
            result = run_episode(
                env=env,
                controller=controller,
                controller_name=args.controller,
                episode_id=episode_index + 1,
                seed=seed + episode_index,
                scenario=args.scenario,
                log_dir=log_dir if args.log else None,
            )
            results.append(result)
        finally:
            env.close()

    metrics = summarize_episode_results(results)

    print("FalconLite Stage 7 evaluation")
    print(f"controller: {args.controller}")
    print(f"scenario: {args.scenario or 'default'}")
    print(f"episodes: {episodes}")
    print(f"seed: {seed}")
    for name, value in metrics.items():
        if name == "episodes":
            continue
        print(f"{name}: {value:.3f}")

    if args.summary_csv is not None:
        output_path = Path(args.summary_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_to_frame(metrics, metadata={"controller": args.controller, "seed": seed, "scenario": args.scenario}).to_csv(
            output_path,
            index=False,
        )
        print(f"summary_csv: {output_path}")

    if args.episodes_csv is not None:
        output_path = Path(args.episodes_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        episode_results_to_frame(results).to_csv(output_path, index=False)
        print(f"episodes_csv: {output_path}")


if __name__ == "__main__":
    main()
