"""Stage 5 telemetry logger tests."""

import pandas as pd

from falconlite.env import RocketAction, RocketState
from falconlite.utils.logger import TELEMETRY_COLUMNS, TelemetryLogger


def test_telemetry_logger_writes_expected_schema(tmp_path) -> None:
    logger = TelemetryLogger(log_dir=tmp_path, episode_id=42)
    state = RocketState(x=1, y=2, vx=3, vy=4, theta=0.1, omega=0.2, fuel=900.0, legs_deployed=True)
    action = RocketAction(thrust=5, gimbal_angle=0.1, leg_deploy=True)
    info = {
        "step": 7,
        "time": 0.14,
        "done_reason": "running",
        "is_success": False,
        "terminated": False,
        "truncated": False,
        "normalized_action": [0.25, 0.5, 1.0],
        "failure_flags": {
            "missed_pad": False,
            "hard_landing": False,
            "tip_over": False,
            "body_contact": False,
            "one_foot_contact": False,
            "out_of_bounds": False,
        },
        "contact": {
            "left_foot_contact": True,
            "right_foot_contact": True,
            "foot_supported": True,
        },
    }

    logger.log_step(state=state, action=action, reward=-1.5, info=info)
    logger.close()

    frame = pd.read_csv(logger.path)
    assert frame.columns.tolist() == TELEMETRY_COLUMNS
    assert len(frame) == 1
    assert frame.loc[0, "episode_id"] == 42
    assert frame.loc[0, "step"] == 7
    assert frame.loc[0, "reward"] == -1.5
    assert frame.loc[0, "done_reason"] == "running"
    assert frame.loc[0, "legs_deployed"]
    assert frame.loc[0, "leg_deploy"]
    assert frame.loc[0, "normalized_leg_deploy"] == 1.0
    assert frame.loc[0, "foot_supported"]


def test_telemetry_logger_context_manager(tmp_path) -> None:
    with TelemetryLogger(log_dir=tmp_path, episode_id=1) as logger:
        assert not logger.closed

    assert logger.closed
