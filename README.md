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
poetry run python scripts/run_pid.py --scenario terminal_diagonal
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
- fuel consumption in kilograms
- orientation-aware aerodynamic drag with configurable air density and wind
- meter-scale rocket geometry, including body, grid fins, and two side-view landing legs
- geometry-based ground contact
- landing leg deployment and 3-second stable-contact success logic
- Pygame rendering for rocket geometry, landing pad, velocity, fuel, and status
- Gymnasium `reset`, `step`, `render`, and `close` API
- Dense reward terms and geometry-aware landing outcome classification
- Per-step CSV telemetry logging
- PID baseline controller
- Batch evaluation metrics
- No-thrust free-fall debug script

Coordinate convention:

- `x`: horizontal position
- `y`: altitude, with ground at `y = 0`
- `theta = 0`: rocket points upward
- `gimbal_angle`: engine angle relative to the rocket body
- `fuel`: remaining propellant mass in kilograms
- `legs_deployed`: irreversible landing leg deploy state

Gym actions are normalized as `[thrust, gimbal, leg_deploy]`, where leg deployment triggers when the third command is at least `0.5`.

Drag uses a simple projected-area model, not CFD: an upright rocket mostly exposes its circular axial area, while a sideways rocket exposes a much larger body side area.

Named scenarios can override the initial state:

- `terminal_vertical`: starts above the pad with vertical descent.
- `terminal_diagonal`: starts offset from the pad with horizontal velocity toward the target, approximating the crossrange terminal approach before the final vertical landing.

## Gymnasium Usage

```python
from falconlite.env import RocketLandingEnv

env = RocketLandingEnv()
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

## Landing Contract

Landing success requires both landing feet to be supported on the pad without body contact, within velocity/angle limits, for `required_stable_time` seconds. Ground contact can be classified as `success`, `missed_pad`, `hard_landing`, `tip_over`, `body_contact`, `one_foot_contact`, or `out_of_bounds`. Each `step` returns reward diagnostics in `info["reward_terms"]` and landing diagnostics in `info["failure_flags"]`.

## Telemetry

```bash
poetry run python scripts/run_random.py --steps 200 --seed 0 --log
```

The default CSV path is `logs/episode_000001.csv`.

## PID Baseline

```bash
poetry run python scripts/run_pid.py --steps 1000 --render
poetry run python scripts/run_pid.py --scenario terminal_diagonal --steps 1500 --render
```

The Stage 6 PID controller is a cascaded baseline:

- lateral error produces a target tilt
- attitude error produces a gimbal command
- vertical-speed error produces a thrust command
- low altitude produces a landing-leg deploy command

## Evaluation

```bash
poetry run python scripts/evaluate.py --controller random --episodes 100
poetry run python scripts/evaluate.py --controller pid --episodes 100
```

The evaluation reports success rate, crash/failure rates, fuel use, touchdown speed, max tilt, and episode length.

## Free-Fall Debugging

```bash
poetry run python scripts/run_freefall.py --initial-y 100 --initial-vy 0 --seconds 3
poetry run python scripts/run_freefall.py --initial-y 100 --initial-vy 0 --seconds 3 --vacuum
poetry run python scripts/run_freefall.py --initial-y 100 --initial-vy 0 --initial-theta-deg 90 --seconds 3
```

The script prints the simulated state, a vacuum reference, and the latest drag force so the drag effect can be compared directly.
