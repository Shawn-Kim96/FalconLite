# TODO lists

## Stage: Falcon-9 booster geometry (in progress)

Scope: 2D side-view geometry + meter-scale renderer. Physics, contact, and reward stay
on CM-based behavior in this pass. Modeled after a returning Falcon 9 first stage on
its landing burn (dry ~25.6 t, wet ~35.6 t, 1× Merlin throttled to 40–100% of ~845 kN).
Side projection shows ONE pair of legs and ONE pair of grid fins (the cross-pattern
front/back pair overlaps in 2D).

- [x] `RocketGeometry` dataclass (`falconlite/env/geometry.py`)
  - `height_m`, `width_m`, `engine_offset_m`, `nozzle_radius_m`
  - `leg_length_m`, `leg_span_m`, `leg_stow_angle_rad`, `leg_deploy_angle_rad`
  - `grid_fin_offset_m`, `grid_fin_chord_m`, `grid_fin_span_m`
  - helpers: `nozzle_position_body`, `grid_fin_positions_body`, `foot_positions_body`
- [x] Wire geometry into `PhysicsConfig` (rename `engine_lever_arm` → `engine_offset_m`)
- [x] Update `configs/default.yaml` to F9-booster scale + add `physics.geometry` block
- [x] Update `tests/test_physics.py` to F9 scale
- [ ] Render rocket in meters with TVC + grid fins
  - Drop `rocket_width_px` / `rocket_height_px` from `RendererConfig` and config
  - Tapered body, nose cone, 2 legs, 2 grid fins, gimballed flame
  - Update `tests/test_renderer.py`
- [ ] Run `poetry run pytest` and confirm green
- [ ] Spot-check rendering: `poetry run python scripts/run_freefall.py --render`

## Next stage: contact, legs, drag

- [ ] Compute contact points (leg tips when deployed; engine bell otherwise)
- [ ] Compute force application point (engine offset already known; consider grid fin
      hinge points for aero forces)
- [ ] Replace ground-contact rule (`y <= 0` on CM) with geometry-based check on the
      lowest contact point in world frame
- [ ] Landing-leg deploy action / state (binary deploy switch + transition timer)
- [ ] Atmospheric drag
  - Body drag: `0.5 * rho * v^2 * Cd * frontal_area`
  - Grid fin lift/drag with deflection (the actual attitude-control authority)
  - Simple exponential `rho(altitude)` model
- [ ] Reward / termination updates for the new contact and drag model

## Follow-ups deferred from this pass

- [ ] Re-tune PID gains for F9 scale (current gains were sized for the 1 kg / 20 N toy)
- [ ] Re-tune reward weights (touchdown thresholds, descent rate gain, fuel weight) so
      `evaluate.py` gives meaningful PID baseline numbers at the new scale
- [ ] Decide on RCS (cold-gas attitude thrusters) — current plan: TVC + grid fins only

## Future: 2.5D extension

- [ ] Add depth-axis legs/fins (the front/back pair currently folded into the side pair)
- [ ] Per-engine modeling if multi-engine landing burns are simulated
