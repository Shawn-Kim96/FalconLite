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
  - Replaced by `landing_burn_vertical` / `landing_burn_diagonal` /
    `randomized_landing_burn` covering the full Falcon-9 main-engine landing
    burn from ~4.6 km, vy ~ -300 m/s down to touchdown

## Make terminal landing meaningfully harder

Priority order: raise difficulty with physics/evaluation changes before adding advanced controllers.

- [x] Randomized terminal scenario generator
  - [x] Sample `x`, `y`, `vx`, `vy`, `theta`, `omega`, and fuel from configured ranges
        (any scenario field written as `[min, max]` is uniformly sampled at reset)
  - [x] `scripts/evaluate.py --scenario randomized_terminal` works end-to-end
  - [x] Seed and sampled initial state recorded on `EpisodeResult` and `info`
  - [x] Verify "PID no longer scores near 100% over broad random cases" once a
        Python 3.12 / poetry environment is available to run pytest + evaluate
        (50-episode run, seed=0: success_rate 0.40, crash_rate 0.60)

- [x] Stricter landing criteria
  - Two-tier classification: `precision_*` (Mechazilla-class — counts as
    success) and `rough_*` (survived but damaged — separate `rough_landing`
    outcome, not success). Anything outside the rough envelope is a crash.
  - Precision thresholds: tolerance 1.0 m, |vx| 0.3, |vy| 1.5, |theta| 0.017
    rad (~1°), |omega| 0.02 rad/s.

- [ ] Realistic leg contact dynamics
  - Replace "supported feet zero velocity" with spring-damper normal forces
  - Add foot friction and sliding
  - Allow one foot to touch first and create torque
  - Preserve post-touchdown rocking instead of snapping to stable contact
  - Success should require stable attitude and low motion after touchdown

- [x] Throttle realism
  - Merlin-class envelope: `min_throttle = 0.40` of max thrust, with an
    `engine_off_threshold` so commands below ~0.05 fully shut the engine off.
  - Mass flow set to `308 kg/s` (≈ Merlin 1D Isp 283 s × g × max_thrust).
  - At min throttle the booster's TWR > 1 once mass drops near dry, so hover
    is physically impossible — controllers must time the burn (hoverslam-style).

- [ ] Finite fuel stress cases
  - Add scenarios with `fuel` ranges such as 6,000-10,000 kg
  - Track fuel-out failure distinctly
  - Evaluate fuel margin at touchdown, not only success/failure

- [ ] Fuel efficiency as a first-class metric
  - PID currently burns ~4500 kg per landing — clearly wasteful
  - Add `fuel_efficiency_score` (e.g. fuel margin or kg-per-successful-landing)
  - Reward shaping: stronger penalty on fuel use, or bonus for surplus at touchdown
  - Compare controllers on (success_rate, precision_rate, fuel_used) jointly

- [ ] Wind and gust disturbances
  - Use existing `wind_x_mps` / `wind_y_mps` for steady wind cases
  - Add simple gust profiles over time
  - Add robustness evaluation columns for wind and gust intensity

- [ ] Sensor noise and state-estimation stress
  - Feed controllers noisy observations while physics keeps true state
  - Log both true and observed state
  - Keep noise optional so deterministic debugging remains possible

- [x] Geometry origin = body center
  - Origin sits at the booster's geometric center; nozzle at -height/2,
    nose at +height/2. Removed the legacy "nozzle = origin" convention that
    falsely triggered body_contact when the booster's center reached y =
    engine_offset_m before the body actually touched the ground.
  - `body_contact` now excludes the nozzle exit point — only structural
    body corners (and nose) trigger it.

- [x] Camera-following render with mini-map and end-of-episode overlay
  - Main view: rocket-centered camera at fixed meter scale, with the ground
    pinned to the bottom of the viewport so touchdown is visible up close.
  - Mini-map (top-right): full world envelope with pad marker, target cross,
    trajectory, rocket position, and a thrust-direction arrow.
  - HUD shows units (m, m/s, deg, deg/s, kN, deg).
  - End-of-episode overlay names the outcome (success / rough_landing /
    hard_landing / body_contact / missed_pad / tip_over / out_of_bounds /
    max_steps), shows final state, and offers a Rerun button (or `R` key).
  - PID gimbal/nozzle is drawn as a body-axis reference line plus a
    gimballed nozzle segment so the gimbal command is visually distinguishable.

- [x] Distributed aerodynamic forces (drag only)
  - 5 body-axis nodes from engine base to nose, each with its own slice of
    side area (and end nodes share axial area).
  - Node velocity = `v_cm + omega x r`; per-node drag is summed for total
    force and torque (`r x F` summed over nodes).
  - Result: spinning booster picks up an opposing drag torque (rotational
    damping). Aligned free-fall is unchanged from the single-point model.
  - Grid-fin lift/drag still TODO (see "Later realism").

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
