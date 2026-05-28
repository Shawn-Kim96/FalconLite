"""Smoke test for the no-thrust free-fall script."""

import subprocess
import sys


def test_freefall_script_runs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_freefall.py", "--steps", "5"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "FalconLite no-thrust free-fall rollout" in result.stdout
    assert "analytic_freefall" in result.stdout
