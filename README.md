# FalconLite

Autonomous rocket landing simulator and evaluation platform.

## Current Stage

Stage 7: Evaluation metrics for random and PID controllers.

## Setup

```bash
poetry env use python3.12
poetry install
```

## Checks

```bash
poetry run python scripts/run_random.py
poetry run python scripts/run_random.py --render
poetry run python scripts/run_freefall.py --render
poetry run python scripts/run_random.py --log
poetry run python scripts/run_pid.py
poetry run python scripts/run_pid.py --render
poetry run python scripts/evaluate.py --controller random --episodes 10
poetry run python scripts/evaluate.py --controller pid --episodes 10
poetry run pytest
```

## Current Model

The simulator includes:

- gravity
- thrust force
- gimbal torque
- fuel consumption
- ground collision termination
- Pygame rendering for rocket body, landing pad, velocity, fuel, and status
- Gymnasium `reset`, `step`, `render`, and `close` API
- Dense reward terms and CM-based landing outcome classification
- Per-step CSV telemetry logging
- PID baseline controller
- Batch evaluation metrics
- No-thrust free-fall debug script

Coordinate convention:

- `x`: horizontal position
- `y`: altitude, with ground at `y = 0`
- `theta = 0`: rocket points upward
- `gimbal_angle`: engine angle relative to the rocket body

## Gymnasium Usage

```python
from falconlite.env import RocketLandingEnv

env = RocketLandingEnv()
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

## Stage 4 Reward Contract

Ground contact is classified as `success`, `missed_pad`, `hard_landing`, or `tip_over` using center-of-mass position, velocity, angle, and angular velocity thresholds. Each `step` returns reward diagnostics in `info["reward_terms"]` and landing diagnostics in `info["failure_flags"]`.

## Telemetry

```bash
poetry run python scripts/run_random.py --steps 200 --seed 0 --log
```

The default CSV path is `logs/episode_000001.csv`.

## PID Baseline

```bash
poetry run python scripts/run_pid.py --steps 1000 --render
```

The Stage 6 PID controller is a cascaded baseline:

- lateral error produces a target tilt
- attitude error produces a gimbal command
- vertical-speed error produces a thrust command

## Evaluation

```bash
poetry run python scripts/evaluate.py --controller random --episodes 100
poetry run python scripts/evaluate.py --controller pid --episodes 100
```

The evaluation reports success rate, crash/failure rates, fuel use, touchdown speed, max tilt, and episode length.
