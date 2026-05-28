"""Random controller used for smoke tests and future baselines."""

from __future__ import annotations

import numpy as np


class RandomController:
    """Sample normalized thrust and gimbal commands."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def select_action(self, *_: object, **__: object) -> np.ndarray:
        thrust = self._rng.uniform(0.0, 1.0)
        gimbal = self._rng.uniform(-1.0, 1.0)
        return np.array([thrust, gimbal], dtype=np.float32)
