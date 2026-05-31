# TODO lists

## Completed foundation

- [x] Meter-scale Falcon-9-style booster geometry
  - `RocketGeometry` with body, nozzle, grid fins, side-view landing legs, foot points
  - World transforms for named geometry points
- [x] Fuel represented as propellant mass in kilograms
- [x] Geometry-based renderer
  - Meter-scale body, legs, grid fins, gimballed flame, lowered landing pad visual
- [x] Geometry-based ground contact
  - Body contact, one-foot contact, two-foot support, contact velocity diagnostics
- [x] Landing leg deploy state/action
  - Gym action is `[thrust, gimbal, leg_deploy]`
  - Success requires both feet supported for `required_stable_time`
- [x] Orientation-aware aerodynamic drag
  - Upright rocket uses axial projected area
  - Sideways rocket uses much larger side projected area
  - Free-fall script can compare drag vs `--vacuum`
- [x] Terminal approach scenarios
  - `terminal_vertical`
  - `terminal_diagonal`

## Make terminal landing meaningfully harder

Priority order: raise difficulty with physics/evaluation changes before adding advanced controllers.

- [ ] Randomized terminal scenario generator
  - Sample `x`, `y`, `vx`, `vy`, `theta`, `omega`, and fuel from configured ranges
  - Add `scripts/evaluate.py --scenario randomized_terminal`
  - Report seed and sampled initial state in telemetry
  - Target: PID should no longer score near 100% over broad random cases

- [ ] Stricter landing criteria
  - Reduce `landing_tolerance` from the current broad pad-scale value to a tighter target
  - Tighten `max_touchdown_vx`, `max_touchdown_vy`, `max_touchdown_angle`, and `max_touchdown_omega`
  - Add separate thresholds for "safe landing" vs "survived but rough landing"

- [ ] Realistic leg contact dynamics
  - Replace "supported feet zero velocity" with spring-damper normal forces
  - Add foot friction and sliding
  - Allow one foot to touch first and create torque
  - Preserve post-touchdown rocking instead of snapping to stable contact
  - Success should require stable attitude and low motion after touchdown

- [ ] Throttle realism
  - Add minimum throttle for the active landing engine
  - Optionally add a simple on/off landing burn mode
  - Make hover impossible when thrust-to-weight exceeds 1 at minimum throttle
  - This should make timing matter more than the current smooth hover-like PID descent

- [ ] Finite fuel stress cases
  - Add scenarios with `fuel` ranges such as 6,000-10,000 kg
  - Track fuel-out failure distinctly
  - Evaluate fuel margin at touchdown, not only success/failure

- [ ] Wind and gust disturbances
  - Use existing `wind_x_mps` / `wind_y_mps` for steady wind cases
  - Add simple gust profiles over time
  - Add robustness evaluation columns for wind and gust intensity

- [ ] Sensor noise and state-estimation stress
  - Feed controllers noisy observations while physics keeps true state
  - Log both true and observed state
  - Keep noise optional so deterministic debugging remains possible

- [ ] Distributed aerodynamic forces
  - Split body into 5-9 sample nodes
  - Compute node velocity from `v_cm + omega x r`
  - Sum node drag forces and `r x F` torque
  - Add grid-fin lift/drag as separate force nodes later

- [ ] Terminal guidance metrics
  - Add desired glide-slope / divert corridor diagnostics
  - Track crossrange error, lateral braking distance, and touchdown ellipse
  - Make README compare vertical vs diagonal vs randomized terminal results

## Later realism, not needed yet

- [ ] Engine response delay and throttle slew rate
- [ ] Gimbal actuator rate limit / delay
- [ ] Variable mass, center of mass, and inertia during fuel burn
- [ ] Cold-gas/RCS attitude thrusters for high-altitude flip maneuver
- [ ] Entry burn and grid-fin aerodynamic guidance phase
- [ ] 2.5D extension with lateral `z`, yaw, roll, and four-leg contact geometry
