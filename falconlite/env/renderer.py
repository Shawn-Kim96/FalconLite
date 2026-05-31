"""Pygame renderer for FalconLite."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import os
from typing import Any

from falconlite.env.physics import PhysicsConfig
from falconlite.env.state import RocketAction, RocketState


Color = tuple[int, int, int]


@dataclass(frozen=True)
class RendererConfig:
    """Display parameters for the Stage 2 renderer."""

    width: int = 900
    height: int = 700
    margin: int = 60
    fps: int = 50
    rocket_width_px: int = 18
    rocket_height_px: int = 58
    pad_width_m: float = 20.0
    velocity_scale: float = 2.0

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
        self.scale = self._compute_scale()
        self.screen = pygame.display.set_mode((self.config.width, self.config.height))
        pygame.display.set_caption("FalconLite")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.closed = False

    def render(
        self,
        state: RocketState,
        action: RocketAction | None = None,
        info: Mapping[str, Any] | None = None,
        step: int | None = None,
        reward: float | None = None,
    ) -> None:
        """Draw the current state."""

        if self.closed:
            return

        self._handle_events()
        if self.closed:
            return
        self.screen.fill((12, 16, 22))
        self._draw_grid()
        self._draw_ground_and_pad()
        self._draw_velocity_vector(state)
        self._draw_rocket(state, action)
        self._draw_hud(state, action, info or {}, step, reward)
        self.pygame.display.flip()
        self.clock.tick(self.config.fps)

    def close(self) -> None:
        """Release renderer resources."""
        if self.closed:
            return
        self.closed = True
        self.pygame.display.quit()
        self.pygame.quit()

    def world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        """Convert world meters to screen pixels."""

        center_x = self.config.width / 2
        ground_y = self.config.height - self.config.margin
        screen_x = center_x + x * self.scale
        screen_y = ground_y - y * self.scale
        return int(round(screen_x)), int(round(screen_y))

    def _compute_scale(self) -> float:
        usable_width = self.config.width - 2 * self.config.margin
        usable_height = self.config.height - 2 * self.config.margin
        x_scale = usable_width / (2 * self.physics_config.world_x_limit)
        y_scale = usable_height / self.physics_config.world_y_limit
        return min(x_scale, y_scale)

    def _handle_events(self) -> None:
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.close()

    def _draw_grid(self) -> None:
        color = (28, 34, 44)
        for x_m in range(
            -int(self.physics_config.world_x_limit),
            int(self.physics_config.world_x_limit) + 1,
            20,
        ):
            start = self.world_to_screen(x_m, 0)
            end = self.world_to_screen(x_m, self.physics_config.world_y_limit)
            self.pygame.draw.line(self.screen, color, start, end, 1)

        for y_m in range(0, int(self.physics_config.world_y_limit) + 1, 20):
            start = self.world_to_screen(-self.physics_config.world_x_limit, y_m)
            end = self.world_to_screen(self.physics_config.world_x_limit, y_m)
            self.pygame.draw.line(self.screen, color, start, end, 1)

    def _draw_ground_and_pad(self) -> None:
        ground_left = self.world_to_screen(-self.physics_config.world_x_limit, 0)
        ground_right = self.world_to_screen(self.physics_config.world_x_limit, 0)
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
        base_left = (
            int(round(nozzle_screen[0] - perp_x * flame_width)),
            int(round(nozzle_screen[1] + perp_y * flame_width)),
        )
        base_right = (
            int(round(nozzle_screen[0] + perp_x * flame_width)),
            int(round(nozzle_screen[1] - perp_y * flame_width)),
        )
        tip = (
            int(round(nozzle_screen[0] - thrust_x * flame_length)),
            int(round(nozzle_screen[1] + thrust_y * flame_length)),
        )
        self.pygame.draw.polygon(self.screen, (255, 174, 75), [base_left, base_right, tip])
        inner_tip = (
            int(round(nozzle_screen[0] - thrust_x * flame_length * 0.55)),
            int(round(nozzle_screen[1] + thrust_y * flame_length * 0.55)),
        )
        self.pygame.draw.polygon(self.screen, (255, 231, 137), [nozzle_screen, base_right, inner_tip])

    def _body_to_screen(self, point: tuple[float, float], state: RocketState) -> tuple[int, int]:
        world_point = self.physics_config.geometry.body_to_world(
            point,
            x=state.x,
            y=state.y,
            theta=state.theta,
        )
        return self.world_to_screen(*world_point)

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
            f"x/y: {state.x:7.2f} / {state.y:7.2f}",
            f"vx/vy: {state.vx:6.2f} / {state.vy:6.2f}",
            f"theta/omega: {state.theta:6.2f} / {state.omega:6.2f}",
            f"done: {info.get('done_reason', 'running')}",
        ]
        if action is not None:
            lines.append(f"thrust/gimbal: {action.thrust / 1000:6.1f} kN / {action.gimbal_angle:5.2f}")
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
