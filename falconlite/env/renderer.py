"""Pygame renderer for FalconLite.

Layout:
- Main view: camera follows the rocket, fixed viewport in meters so the booster
  stays visually large regardless of altitude.
- Mini-map (top-right): full world envelope showing pad, rocket position,
  trajectory history, and a thrust direction marker.
- End-of-episode overlay: outcome summary plus a Rerun button.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
import math
import os
from typing import Any

from falconlite.env.physics import PhysicsConfig
from falconlite.env.state import RocketAction, RocketState


Color = tuple[int, int, int]


# Human-readable outcome summaries keyed by the env's done_reason.
_OUTCOME_TITLES: dict[str, tuple[str, Color]] = {
    "success": ("LANDING SUCCESS", (74, 222, 128)),
    "rough_landing": ("ROUGH LANDING", (245, 196, 64)),
    "missed_pad": ("MISSED THE PAD", (245, 142, 70)),
    "hard_landing": ("HARD LANDING — STRUCTURAL FAIL", (240, 84, 84)),
    "tip_over": ("TIPPED OVER ON TOUCHDOWN", (240, 84, 84)),
    "body_contact": ("BODY HIT THE GROUND", (240, 84, 84)),
    "one_foot_contact": ("ONE FOOT CONTACT — UNSTABLE", (245, 196, 64)),
    "out_of_bounds": ("FLEW OFF THE MAP", (240, 84, 84)),
    "max_steps": ("OUT OF TIME", (160, 172, 188)),
    "crash": ("CRASHED", (240, 84, 84)),
}


_OUTCOME_DETAILS: dict[str, str] = {
    "success": "Booster touched down within precision tolerance. Reusable.",
    "rough_landing": "Survived but outside precision envelope (legs likely bent).",
    "missed_pad": "Touched ground too far from the pad center.",
    "hard_landing": "Vertical or lateral velocity exceeded structural limits.",
    "tip_over": "Attitude or angular rate at touchdown was too high.",
    "body_contact": "Body fuselage struck the ground before legs could support.",
    "one_foot_contact": "Only one leg made contact — booster fell over.",
    "out_of_bounds": "Booster left the simulated world envelope.",
    "max_steps": "Simulation time budget exhausted before landing.",
    "crash": "Generic crash.",
}


@dataclass(frozen=True)
class RendererConfig:
    """Display parameters for the renderer."""

    width: int = 900
    height: int = 700
    margin: int = 60
    fps: int = 50
    rocket_width_px: int = 18
    rocket_height_px: int = 58
    pad_width_m: float = 20.0
    velocity_scale: float = 2.0
    # Main camera viewport half-extents in meters (rocket-centered).
    main_view_half_width_m: float = 80.0
    main_view_half_height_m: float = 100.0
    # Minimap dimensions in pixels and trajectory length in samples.
    minimap_width_px: int = 220
    minimap_height_px: int = 320
    minimap_margin_px: int = 16
    trajectory_max_samples: int = 600

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "RendererConfig":
        if values is None:
            return cls()
        allowed_keys = cls.__dataclass_fields__.keys()
        filtered = {key: values[key] for key in allowed_keys if key in values}
        return cls(**filtered)


class Renderer:
    """Visualize a 2D rocket state without changing simulation dynamics."""

    def __init__(
        self,
        physics_config: PhysicsConfig | Mapping[str, Any] | None = None,
        render_config: RendererConfig | Mapping[str, Any] | None = None,
    ) -> None:
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        import pygame

        self.pygame = pygame
        pygame.init()
        self.physics_config = (
            physics_config
            if isinstance(physics_config, PhysicsConfig)
            else PhysicsConfig.from_mapping(physics_config or {})
        )
        self.config = (
            render_config
            if isinstance(render_config, RendererConfig)
            else RendererConfig.from_mapping(render_config)
        )
        self.scale = self._compute_main_scale()
        self.screen = pygame.display.set_mode((self.config.width, self.config.height))
        pygame.display.set_caption("FalconLite")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.closed = False
        self.trajectory: deque[tuple[float, float]] = deque(maxlen=self.config.trajectory_max_samples)
        self._cam_x = 0.0
        self._cam_y = 0.0
        self.title_font = pygame.font.Font(None, 36)
        self._rerun_requested = False
        self._rerun_button_rect: Any | None = None  # set when overlay is drawn

    def render(
        self,
        state: RocketState,
        action: RocketAction | None = None,
        info: Mapping[str, Any] | None = None,
        step: int | None = None,
        reward: float | None = None,
    ) -> None:
        if self.closed:
            return

        self._handle_events()
        if self.closed:
            return

        self.trajectory.append((state.x, state.y))
        self._cam_x = state.x
        # Lock the camera to keep the ground (y=0) inside the viewport. Once the
        # rocket descends near touchdown, ground appears in the bottom of the
        # main view instead of the rocket flying off into emptiness.
        min_cam_y = self.config.main_view_half_height_m
        self._cam_y = max(state.y, min_cam_y)

        self.screen.fill((12, 16, 22))
        self._draw_main_grid()
        self._draw_main_ground_and_pad()
        self._draw_velocity_vector(state)
        self._draw_rocket(state, action)
        self._draw_minimap(state, action)
        self._draw_hud(state, action, info or {}, step, reward)
        info_map = info or {}
        if self._is_terminal(info_map):
            self._draw_outcome_overlay(info_map, state)
        else:
            self._rerun_button_rect = None
        self.pygame.display.flip()
        self.clock.tick(self.config.fps)

    @staticmethod
    def _is_terminal(info: Mapping[str, Any]) -> bool:
        return bool(info.get("terminated") or info.get("truncated"))

    def reset_episode(self) -> None:
        """Clear trajectory and rerun flag for a new episode."""

        self.trajectory.clear()
        self._rerun_requested = False
        self._rerun_button_rect = None

    @property
    def rerun_requested(self) -> bool:
        return self._rerun_requested

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.pygame.display.quit()
        self.pygame.quit()

    # -- Coordinate transforms ----------------------------------------------

    def world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        """Main viewport: rocket-following camera in meters → pixels."""

        center_x = self.config.width / 2
        center_y = self.config.height / 2
        screen_x = center_x + (x - self._cam_x) * self.scale
        screen_y = center_y - (y - self._cam_y) * self.scale
        return int(round(screen_x)), int(round(screen_y))

    def _compute_main_scale(self) -> float:
        usable_width = self.config.width - 2 * self.config.margin
        usable_height = self.config.height - 2 * self.config.margin
        x_scale = usable_width / (2 * self.config.main_view_half_width_m)
        y_scale = usable_height / (2 * self.config.main_view_half_height_m)
        return min(x_scale, y_scale)

    # -- Event handling -----------------------------------------------------

    def _handle_events(self) -> None:
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.close()
            elif event.type == self.pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self._rerun_button_rect is not None and self._rerun_button_rect.collidepoint(event.pos):
                    self._rerun_requested = True
            elif event.type == self.pygame.KEYDOWN:
                if event.key == self.pygame.K_r and self._rerun_button_rect is not None:
                    self._rerun_requested = True

    # -- Main viewport ------------------------------------------------------

    def _draw_main_grid(self) -> None:
        color = (28, 34, 44)
        spacing_m = 10
        half_w = self.config.main_view_half_width_m
        half_h = self.config.main_view_half_height_m
        x_start = math.floor((self._cam_x - half_w) / spacing_m) * spacing_m
        x_end = math.ceil((self._cam_x + half_w) / spacing_m) * spacing_m
        for x_m in range(int(x_start), int(x_end) + 1, spacing_m):
            start = self.world_to_screen(x_m, self._cam_y - half_h)
            end = self.world_to_screen(x_m, self._cam_y + half_h)
            self.pygame.draw.line(self.screen, color, start, end, 1)

        y_start = math.floor((self._cam_y - half_h) / spacing_m) * spacing_m
        y_end = math.ceil((self._cam_y + half_h) / spacing_m) * spacing_m
        for y_m in range(max(0, int(y_start)), int(y_end) + 1, spacing_m):
            start = self.world_to_screen(self._cam_x - half_w, y_m)
            end = self.world_to_screen(self._cam_x + half_w, y_m)
            self.pygame.draw.line(self.screen, color, start, end, 1)

    def _draw_main_ground_and_pad(self) -> None:
        half_w = self.config.main_view_half_width_m
        ground_left = self.world_to_screen(self._cam_x - half_w, 0)
        ground_right = self.world_to_screen(self._cam_x + half_w, 0)
        self.pygame.draw.line(self.screen, (125, 132, 143), ground_left, ground_right, 2)

        pad_half = self.config.pad_width_m / 2
        pad_left = self.world_to_screen(-pad_half, 0)
        pad_right = self.world_to_screen(pad_half, 0)
        pad_rect = self.pygame.Rect(
            pad_left[0],
            pad_left[1] + 4,
            pad_right[0] - pad_left[0],
            10,
        )
        self.pygame.draw.rect(self.screen, (64, 180, 128), pad_rect, border_radius=2)

    def _draw_velocity_vector(self, state: RocketState) -> None:
        start = self.world_to_screen(state.x, state.y)
        end = (
            int(round(start[0] + state.vx * self.config.velocity_scale)),
            int(round(start[1] - state.vy * self.config.velocity_scale)),
        )
        self.pygame.draw.line(self.screen, (82, 168, 255), start, end, 2)
        self.pygame.draw.circle(self.screen, (82, 168, 255), end, 3)

    def _draw_rocket(self, state: RocketState, action: RocketAction | None) -> None:
        geometry = self.physics_config.geometry
        body_points = [
            geometry.body_to_world(point, x=state.x, y=state.y, theta=state.theta)
            for point in geometry.body_outline_body()
        ]
        body_screen = [self.world_to_screen(x, y) for x, y in body_points]

        self._draw_flame(state, action)
        self.pygame.draw.polygon(self.screen, (232, 238, 245), body_screen)
        self.pygame.draw.polygon(self.screen, (43, 52, 66), body_screen, 2)

        left_hinge, right_hinge = geometry.leg_hinge_positions_body()
        left_foot, right_foot = geometry.foot_positions_body(state.legs_deployed)
        leg_width = max(2, int(round(self.scale * 0.35)))
        self.pygame.draw.line(
            self.screen,
            (126, 134, 148),
            self._body_to_screen(left_hinge, state),
            self._body_to_screen(left_foot, state),
            leg_width,
        )
        self.pygame.draw.line(
            self.screen,
            (126, 134, 148),
            self._body_to_screen(right_hinge, state),
            self._body_to_screen(right_foot, state),
            leg_width,
        )
        for foot in (left_foot, right_foot):
            self.pygame.draw.circle(self.screen, (186, 196, 210), self._body_to_screen(foot, state), max(2, leg_width))

        self._draw_grid_fins(state)
        self.pygame.draw.circle(self.screen, (255, 206, 84), self.world_to_screen(state.x, state.y), 3)

    def _draw_grid_fins(self, state: RocketState) -> None:
        geometry = self.physics_config.geometry
        left_fin, right_fin = geometry.grid_fin_positions_body()
        half_chord = geometry.grid_fin_chord_m / 2
        span = geometry.grid_fin_span_m
        fin_polygons = [
            (
                (left_fin[0], left_fin[1] - half_chord),
                (left_fin[0], left_fin[1] + half_chord),
                (left_fin[0] - span, left_fin[1] + half_chord),
                (left_fin[0] - span, left_fin[1] - half_chord),
            ),
            (
                (right_fin[0], right_fin[1] - half_chord),
                (right_fin[0], right_fin[1] + half_chord),
                (right_fin[0] + span, right_fin[1] + half_chord),
                (right_fin[0] + span, right_fin[1] - half_chord),
            ),
        ]
        for polygon in fin_polygons:
            points = [self._body_to_screen(point, state) for point in polygon]
            self.pygame.draw.polygon(self.screen, (82, 104, 133), points)
            self.pygame.draw.polygon(self.screen, (34, 42, 54), points, 1)

    def _draw_flame(self, state: RocketState, action: RocketAction | None) -> None:
        if action is None or action.thrust <= 0:
            return

        geometry = self.physics_config.geometry
        thrust_fraction = min(action.thrust / self.physics_config.max_thrust, 1.0)
        flame_length = (5.0 + 12.0 * thrust_fraction) * self.scale
        flame_width = max(geometry.nozzle_radius_m * self.scale * 1.2, 3.0)
        thrust_angle = state.theta + action.gimbal_angle
        thrust_x = math.sin(thrust_angle)
        thrust_y = math.cos(thrust_angle)
        perp_x = math.cos(thrust_angle)
        perp_y = -math.sin(thrust_angle)

        nozzle_world = geometry.body_to_world(
            geometry.nozzle_position_body(),
            x=state.x,
            y=state.y,
            theta=state.theta,
        )
        nozzle_screen = self.world_to_screen(*nozzle_world)

        # Body-axis reference line at nozzle: shows where engine would point at
        # gimbal=0. Difference between this and the flame direction visualizes
        # the gimbal command.
        body_axis_len = flame_length * 0.45
        body_axis_x = math.sin(state.theta)
        body_axis_y = math.cos(state.theta)
        body_axis_tip = (
            int(round(nozzle_screen[0] - body_axis_x * body_axis_len)),
            int(round(nozzle_screen[1] + body_axis_y * body_axis_len)),
        )
        self._draw_dashed_line(
            (102, 116, 138),
            nozzle_screen,
            body_axis_tip,
            dash_px=6,
            gap_px=4,
            width=1,
        )

        # Visible nozzle: a short fixed segment along the body axis, then a
        # gimballed segment at the actual thrust angle. The kink between them
        # shows the gimbal angle.
        nozzle_pix = max(geometry.nozzle_radius_m * self.scale * 0.8, 4.0)
        nozzle_kink = (
            int(round(nozzle_screen[0] - body_axis_x * nozzle_pix * 0.4)),
            int(round(nozzle_screen[1] + body_axis_y * nozzle_pix * 0.4)),
        )
        self.pygame.draw.line(
            self.screen,
            (200, 210, 224),
            nozzle_screen,
            nozzle_kink,
            max(2, int(round(nozzle_pix * 0.4))),
        )
        nozzle_tip = (
            int(round(nozzle_screen[0] - thrust_x * nozzle_pix)),
            int(round(nozzle_screen[1] + thrust_y * nozzle_pix)),
        )
        self.pygame.draw.line(
            self.screen,
            (240, 226, 180),
            nozzle_kink,
            nozzle_tip,
            max(2, int(round(nozzle_pix * 0.5))),
        )

        base_left = (
            int(round(nozzle_tip[0] - perp_x * flame_width)),
            int(round(nozzle_tip[1] + perp_y * flame_width)),
        )
        base_right = (
            int(round(nozzle_tip[0] + perp_x * flame_width)),
            int(round(nozzle_tip[1] - perp_y * flame_width)),
        )
        tip = (
            int(round(nozzle_tip[0] - thrust_x * flame_length)),
            int(round(nozzle_tip[1] + thrust_y * flame_length)),
        )
        self.pygame.draw.polygon(self.screen, (255, 174, 75), [base_left, base_right, tip])
        inner_tip = (
            int(round(nozzle_tip[0] - thrust_x * flame_length * 0.55)),
            int(round(nozzle_tip[1] + thrust_y * flame_length * 0.55)),
        )
        self.pygame.draw.polygon(self.screen, (255, 231, 137), [nozzle_tip, base_right, inner_tip])

    def _draw_dashed_line(
        self,
        color: Color,
        start: tuple[int, int],
        end: tuple[int, int],
        *,
        dash_px: int,
        gap_px: int,
        width: int,
    ) -> None:
        sx, sy = start
        ex, ey = end
        dx = ex - sx
        dy = ey - sy
        length = math.hypot(dx, dy)
        if length <= 0:
            return
        ux = dx / length
        uy = dy / length
        step = dash_px + gap_px
        traveled = 0.0
        while traveled < length:
            seg_start = (
                int(round(sx + ux * traveled)),
                int(round(sy + uy * traveled)),
            )
            seg_end_dist = min(traveled + dash_px, length)
            seg_end = (
                int(round(sx + ux * seg_end_dist)),
                int(round(sy + uy * seg_end_dist)),
            )
            self.pygame.draw.line(self.screen, color, seg_start, seg_end, width)
            traveled += step

    def _body_to_screen(self, point: tuple[float, float], state: RocketState) -> tuple[int, int]:
        world_point = self.physics_config.geometry.body_to_world(
            point,
            x=state.x,
            y=state.y,
            theta=state.theta,
        )
        return self.world_to_screen(*world_point)

    # -- Minimap ------------------------------------------------------------

    def _draw_minimap(self, state: RocketState, action: RocketAction | None) -> None:
        cfg = self.config
        mm_w = cfg.minimap_width_px
        mm_h = cfg.minimap_height_px
        mm_x = cfg.width - mm_w - cfg.minimap_margin_px
        mm_y = cfg.minimap_margin_px

        bg_rect = self.pygame.Rect(mm_x, mm_y, mm_w, mm_h)
        self.pygame.draw.rect(self.screen, (18, 24, 34), bg_rect, border_radius=4)
        self.pygame.draw.rect(self.screen, (90, 102, 120), bg_rect, 1, border_radius=4)

        world_x_limit = self.physics_config.world_x_limit
        world_y_limit = self.physics_config.world_y_limit
        pad_inner_x = 8
        pad_inner_y = 8
        inner_w = mm_w - 2 * pad_inner_x
        inner_h = mm_h - 2 * pad_inner_y

        def to_minimap(x_m: float, y_m: float) -> tuple[int, int]:
            x_norm = (x_m + world_x_limit) / (2 * world_x_limit)
            y_norm = y_m / world_y_limit
            sx = mm_x + pad_inner_x + x_norm * inner_w
            sy = mm_y + pad_inner_y + (1.0 - y_norm) * inner_h
            return int(round(sx)), int(round(sy))

        ground_left = to_minimap(-world_x_limit, 0)
        ground_right = to_minimap(world_x_limit, 0)
        self.pygame.draw.line(self.screen, (125, 132, 143), ground_left, ground_right, 1)

        pad_half = self.config.pad_width_m / 2
        pad_left = to_minimap(-pad_half, 0)
        pad_right = to_minimap(pad_half, 0)
        pad_marker_height = 6
        pad_rect = self.pygame.Rect(
            pad_left[0],
            pad_left[1] - pad_marker_height // 2,
            max(2, pad_right[0] - pad_left[0]),
            pad_marker_height,
        )
        self.pygame.draw.rect(self.screen, (64, 180, 128), pad_rect)

        target_marker = to_minimap(0, 0)
        self.pygame.draw.line(
            self.screen,
            (64, 180, 128),
            (target_marker[0], target_marker[1] - 10),
            (target_marker[0], target_marker[1] + 10),
            1,
        )

        if len(self.trajectory) >= 2:
            points = [to_minimap(px, py) for px, py in self.trajectory]
            self.pygame.draw.lines(self.screen, (82, 168, 255), False, points, 1)

        rocket_pos = to_minimap(state.x, state.y)
        self.pygame.draw.circle(self.screen, (255, 206, 84), rocket_pos, 4)

        if action is not None and action.thrust > 0:
            thrust_fraction = min(action.thrust / self.physics_config.max_thrust, 1.0)
            thrust_angle = state.theta + action.gimbal_angle
            arrow_len_px = 10 + 18 * thrust_fraction
            tip = (
                int(round(rocket_pos[0] + math.sin(thrust_angle) * arrow_len_px)),
                int(round(rocket_pos[1] + math.cos(thrust_angle) * arrow_len_px)),
            )
            self.pygame.draw.line(self.screen, (255, 174, 75), rocket_pos, tip, 2)
            self.pygame.draw.circle(self.screen, (255, 174, 75), tip, 2)

        scale_label = self.small_font.render(
            f"map: {int(2 * world_x_limit)} x {int(world_y_limit)} m",
            True,
            (160, 172, 188),
        )
        self.screen.blit(scale_label, (mm_x, mm_y + mm_h + 2))

    # -- HUD ----------------------------------------------------------------

    def _draw_hud(
        self,
        state: RocketState,
        action: RocketAction | None,
        info: Mapping[str, Any],
        step: int | None,
        reward: float | None,
    ) -> None:
        lines = [
            f"step: {step if step is not None else '-'}",
            f"x/y: {state.x:7.2f} m / {state.y:7.2f} m",
            f"vx/vy: {state.vx:6.2f} m/s / {state.vy:6.2f} m/s",
            f"theta/omega: {math.degrees(state.theta):6.2f} deg / {math.degrees(state.omega):6.2f} deg/s",
            f"done: {info.get('done_reason', 'running')}",
        ]
        if action is not None:
            lines.append(
                f"thrust/gimbal: {action.thrust / 1000:6.1f} kN / "
                f"{math.degrees(action.gimbal_angle):5.2f} deg"
            )
        lines.append(f"legs/stable: {'deployed' if state.legs_deployed else 'stowed'} / {state.stable_time:4.2f}s")
        if reward is not None:
            lines.append(f"reward: {reward:7.2f}")

        for index, line in enumerate(lines):
            text = self.font.render(line, True, (222, 229, 239))
            self.screen.blit(text, (18, 16 + index * 24))

        bar_width = 180
        bar_height = 14
        x = 18
        y = self.config.height - 34
        self.pygame.draw.rect(self.screen, (48, 56, 70), (x, y, bar_width, bar_height), border_radius=2)
        max_fuel_mass = max(self.physics_config.mass - self.physics_config.dry_mass, 1e-9)
        fuel_fraction = min(max(state.fuel / max_fuel_mass, 0.0), 1.0)
        fill_width = int(bar_width * fuel_fraction)
        self.pygame.draw.rect(self.screen, (74, 222, 128), (x, y, fill_width, bar_height), border_radius=2)
        label = self.small_font.render(f"fuel {state.fuel:,.0f} kg", True, (222, 229, 239))
        self.screen.blit(label, (x + bar_width + 8, y - 1))

    # -- End-of-episode overlay --------------------------------------------

    def _draw_outcome_overlay(
        self,
        info: Mapping[str, Any],
        state: RocketState,
    ) -> None:
        done_reason = str(info.get("done_reason", "crash"))
        title, color = _OUTCOME_TITLES.get(
            done_reason, (done_reason.upper().replace("_", " "), (240, 84, 84))
        )
        detail = _OUTCOME_DETAILS.get(done_reason, "Episode ended.")

        # Dim background.
        dim = self.pygame.Surface((self.config.width, self.config.height))
        dim.set_alpha(140)
        dim.fill((4, 6, 10))
        self.screen.blit(dim, (0, 0))

        # Card.
        card_w, card_h = 480, 220
        card_x = (self.config.width - card_w) // 2
        card_y = (self.config.height - card_h) // 2
        card_rect = self.pygame.Rect(card_x, card_y, card_w, card_h)
        self.pygame.draw.rect(self.screen, (24, 30, 42), card_rect, border_radius=10)
        self.pygame.draw.rect(self.screen, color, card_rect, 2, border_radius=10)

        title_surface = self.title_font.render(title, True, color)
        self.screen.blit(
            title_surface,
            (card_x + (card_w - title_surface.get_width()) // 2, card_y + 22),
        )

        detail_surface = self.small_font.render(detail, True, (210, 218, 230))
        self.screen.blit(
            detail_surface,
            (card_x + (card_w - detail_surface.get_width()) // 2, card_y + 70),
        )

        stat_lines = [
            f"final altitude: {state.y:7.2f} m",
            f"final speed:    {math.hypot(state.vx, state.vy):7.2f} m/s "
            f"(vy {state.vy:6.2f})",
            f"final tilt:     {math.degrees(state.theta):6.2f} deg",
            f"fuel remaining: {state.fuel:,.0f} kg",
        ]
        for index, line in enumerate(stat_lines):
            text = self.small_font.render(line, True, (180, 190, 206))
            self.screen.blit(text, (card_x + 40, card_y + 100 + index * 18))

        # Rerun button.
        button_w, button_h = 160, 38
        button_x = card_x + (card_w - button_w) // 2
        button_y = card_y + card_h - button_h - 18
        button_rect = self.pygame.Rect(button_x, button_y, button_w, button_h)
        mouse_pos = self.pygame.mouse.get_pos()
        hovered = button_rect.collidepoint(mouse_pos)
        button_color = (74, 134, 222) if hovered else (52, 96, 168)
        self.pygame.draw.rect(self.screen, button_color, button_rect, border_radius=6)
        self.pygame.draw.rect(self.screen, (200, 220, 255), button_rect, 1, border_radius=6)
        button_text = self.font.render("Rerun (R)", True, (240, 246, 255))
        self.screen.blit(
            button_text,
            (
                button_x + (button_w - button_text.get_width()) // 2,
                button_y + (button_h - button_text.get_height()) // 2,
            ),
        )

        self._rerun_button_rect = button_rect

    def wait_for_rerun_or_close(self) -> bool:
        """Block until the user clicks Rerun, presses R, or closes the window.

        Returns True if a rerun was requested, False if the window was closed.
        Caller is responsible for keeping the last frame visible (no extra
        rendering happens here)."""

        if self.closed:
            return False
        self._rerun_requested = False
        while not self.closed and not self._rerun_requested:
            self._handle_events()
            self.pygame.display.flip()
            self.clock.tick(30)
        return self._rerun_requested
