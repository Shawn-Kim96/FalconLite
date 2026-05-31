"""CSV telemetry logging."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from falconlite.env.state import RocketAction, RocketState


TELEMETRY_COLUMNS = [
    "episode_id",
    "step",
    "time",
    "x",
    "y",
    "vx",
    "vy",
    "theta",
    "omega",
    "fuel",
    "legs_deployed",
    "stable_time",
    "thrust",
    "gimbal_angle",
    "leg_deploy",
    "normalized_thrust",
    "normalized_gimbal",
    "normalized_leg_deploy",
    "reward",
    "done_reason",
    "is_success",
    "terminated",
    "truncated",
    "missed_pad",
    "hard_landing",
    "tip_over",
    "body_contact",
    "one_foot_contact",
    "out_of_bounds",
    "left_foot_contact",
    "right_foot_contact",
    "foot_supported",
    "drag_force_x",
    "drag_force_y",
    "drag_acceleration_x",
    "drag_acceleration_y",
    "drag_area_m2",
    "air_relative_speed",
]


class TelemetryLogger:
    """Write per-step rollout telemetry to CSV."""

    def __init__(
        self,
        log_dir: str | Path = "logs",
        episode_id: int = 1,
        filename: str | None = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.episode_id = episode_id
        self.path = self.log_dir / (filename or f"episode_{episode_id:06d}.csv")
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=TELEMETRY_COLUMNS)
        self._writer.writeheader()
        self.closed = False

    def log_step(
        self,
        *,
        state: RocketState,
        action: RocketAction | None,
        reward: float,
        info: dict[str, Any],
    ) -> None:
        """Append one environment step to the telemetry CSV."""

        if self.closed:
            raise RuntimeError("Cannot write to a closed TelemetryLogger.")

        normalized_action = info.get("normalized_action")
        failure_flags = info.get("failure_flags", {})
        contact = info.get("contact", {})
        drag = info.get("drag", {})
        drag_acceleration = info.get("drag_acceleration", (0.0, 0.0))
        row = {
            "episode_id": self.episode_id,
            "step": info.get("step", 0),
            "time": info.get("time", 0.0),
            "x": state.x,
            "y": state.y,
            "vx": state.vx,
            "vy": state.vy,
            "theta": state.theta,
            "omega": state.omega,
            "fuel": state.fuel,
            "legs_deployed": bool(state.legs_deployed),
            "stable_time": state.stable_time,
            "thrust": action.thrust if action is not None else 0.0,
            "gimbal_angle": action.gimbal_angle if action is not None else 0.0,
            "leg_deploy": bool(action.leg_deploy) if action is not None else False,
            "normalized_thrust": float(normalized_action[0]) if normalized_action is not None else 0.0,
            "normalized_gimbal": float(normalized_action[1]) if normalized_action is not None else 0.0,
            "normalized_leg_deploy": float(normalized_action[2]) if normalized_action is not None else 0.0,
            "reward": reward,
            "done_reason": info.get("done_reason", "unknown"),
            "is_success": bool(info.get("is_success", False)),
            "terminated": bool(info.get("terminated", False)),
            "truncated": bool(info.get("truncated", False)),
            "missed_pad": bool(failure_flags.get("missed_pad", False)),
            "hard_landing": bool(failure_flags.get("hard_landing", False)),
            "tip_over": bool(failure_flags.get("tip_over", False)),
            "body_contact": bool(failure_flags.get("body_contact", False)),
            "one_foot_contact": bool(failure_flags.get("one_foot_contact", False)),
            "out_of_bounds": bool(failure_flags.get("out_of_bounds", False)),
            "left_foot_contact": bool(contact.get("left_foot_contact", False)),
            "right_foot_contact": bool(contact.get("right_foot_contact", False)),
            "foot_supported": bool(contact.get("foot_supported", False)),
            "drag_force_x": float(drag.get("force_x", 0.0)),
            "drag_force_y": float(drag.get("force_y", 0.0)),
            "drag_acceleration_x": float(drag_acceleration[0]),
            "drag_acceleration_y": float(drag_acceleration[1]),
            "drag_area_m2": float(drag.get("projected_area_m2", 0.0)),
            "air_relative_speed": float(drag.get("speed_mps", 0.0)),
        }
        self._writer.writerow(row)

    def close(self) -> None:
        """Flush and close the CSV file."""

        if self.closed:
            return
        self._file.flush()
        self._file.close()
        self.closed = True

    def __enter__(self) -> "TelemetryLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
