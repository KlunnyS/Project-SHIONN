"""Decision decimation without changing observation or per-tick action units."""
from __future__ import annotations

import copy
import math


def decision_interval(control_fps: float, model_fps: float | None) -> int:
    """Require an integer control-tick interval; never silently round a rate."""
    rate = control_fps if model_fps is None else model_fps
    if not all(math.isfinite(x) and x > 0 for x in (control_fps, rate)):
        raise ValueError('Control and model FPS must be positive and finite')
    ratio = control_fps / rate
    if not math.isfinite(ratio):
        raise ValueError("Model/control FPS ratio must be finite")
    interval = round(ratio)
    if interval < 1 or not math.isclose(ratio, interval, rel_tol=1e-9):
        raise ValueError('--model-fps must divide --fps into a positive integer number of ticks')
    return interval


class DecisionRatePolicy:
    """Observe every tick, infer every N ticks, hold per-tick commands in between.

    Mouse deltas repeat unchanged: they remain displacement per control tick,
    rather than displacement per decision. No catch-up decisions after a stall.
    Reset drops both cached actions and the underlying frame/recurrent history.
    """
    def __init__(self, policy, interval: int):
        if interval < 1:
            raise ValueError('Decision interval must be positive')
        self.policy = policy
        self.interval = interval
        self.tick = 0
        self.action = self.diagnostics = None

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def reset(self):
        self.policy.reset()
        self._reset_decision_cache()

    def reset_memory(self):
        """Drop held actions and recurrent state but preserve a selected goal."""
        reset = getattr(self.policy, "reset_memory", self.policy.reset)
        reset()
        self._reset_decision_cache()

    def _reset_decision_cache(self):
        self.tick = 0
        self.action = self.diagnostics = None

    def predict_with_diagnostics(self, frame):
        self.policy.add_frame(frame)
        refreshed = self.tick % self.interval == 0
        if refreshed:
            self.action, self.diagnostics = self.policy.predict_with_diagnostics()
        result = copy.deepcopy(self.diagnostics)
        result['decision_rate'] = {
            'refreshed': refreshed, 'interval_ticks': self.interval,
            'decision_index': self.tick // self.interval,
            'held_ticks': self.tick % self.interval,
            'source_observation_tick': self.tick - self.tick % self.interval,
        }
        self.tick += 1
        return dict(self.action), result
