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
            pad_left[1] - 8,
            pad_right[0] - pad_left[0],
            8,
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
        rocket_surface = self._make_rocket_surface(action)
        angle_degrees = -math.degrees(state.theta)
        rotated = self.pygame.transform.rotate(rocket_surface, angle_degrees)
        center = self.world_to_screen(state.x, state.y)
        rect = rotated.get_rect(center=center)
        self.screen.blit(rotated, rect)

    def _make_rocket_surface(self, action: RocketAction | None) -> Any:
        surface = self.pygame.Surface(
            (self.config.rocket_width_px * 3, self.config.rocket_height_px * 2),
            self.pygame.SRCALPHA,
        )
        cx = surface.get_width() // 2
        top = 10
        bottom = top + self.config.rocket_height_px
        half_width = self.config.rocket_width_px // 2

        body = self.pygame.Rect(cx - half_width, top, self.config.rocket_width_px, self.config.rocket_height_px)
        self.pygame.draw.polygon(
            surface,
            (232, 238, 245),
            [(cx, top - 10), (cx - half_width, top + 8), (cx + half_width, top + 8)],
        )
        self.pygame.draw.rect(surface, (232, 238, 245), body, border_radius=3)
        self.pygame.draw.line(surface, (41, 50, 65), (cx, top + 10), (cx, bottom - 6), 2)

        if action is not None and action.thrust > 0:
            flame_length = 10 + int(18 * min(action.thrust / self.physics_config.max_thrust, 1.0))
            flame_half_width = max(4, half_width - 2)
            self.pygame.draw.polygon(
                surface,
                (255, 174, 75),
                [
                    (cx - flame_half_width, bottom),
                    (cx + flame_half_width, bottom),
                    (cx, bottom + flame_length),
                ],
            )

        return surface

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
            lines.append(f"thrust/gimbal: {action.thrust:5.2f} / {action.gimbal_angle:5.2f}")
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
        fill_width = int(bar_width * min(max(state.fuel, 0.0), 1.0))
        self.pygame.draw.rect(self.screen, (74, 222, 128), (x, y, fill_width, bar_height), border_radius=2)
        label = self.small_font.render(f"fuel {state.fuel:.2f}", True, (222, 229, 239))
        self.screen.blit(label, (x + bar_width + 8, y - 1))
