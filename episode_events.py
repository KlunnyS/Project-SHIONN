"""Parse SHIONN Portal events from arbitrarily chunked netconsole text."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass


EVENT_PATTERN = re.compile(
    r"EVT\|(chamber_ready|goal_reached|episode_failed)\|([^\r\n]*)"
)
MAX_EVENT_LINE_CHARS = 64 * 1024


@dataclass(frozen=True)
class EpisodeEvent:
    name: str
    value: str


class EpisodeEventStream:
    """Turn arbitrarily chunked netconsole text into complete SHIONN events."""

    def __init__(self) -> None:
        self._partial_line = ""
        self._discarding_overlong_line = False
        self._events: deque[EpisodeEvent] = deque()

    def clear(self) -> None:
        self._partial_line = ""
        self._discarding_overlong_line = False
        self._events.clear()

    @property
    def has_partial_line(self) -> bool:
        return bool(self._partial_line)

    def feed(self, text: str) -> None:
        if not text:
            return
        lines = (self._partial_line + text).splitlines(keepends=True)
        self._partial_line = ""
        for line in lines:
            complete = line.endswith(("\n", "\r"))
            if self._discarding_overlong_line:
                if complete:
                    self._discarding_overlong_line = False
                continue
            if len(line) > MAX_EVENT_LINE_CHARS:
                # Ignore malformed/unbounded console lines and resynchronize at
                # the next delimiter instead of retaining arbitrary output.
                self._discarding_overlong_line = not complete
                continue
            if not complete:
                self._partial_line = line
                continue
            match = EVENT_PATTERN.search(line)
            if match:
                self._events.append(EpisodeEvent(match.group(1), match.group(2).strip()))

    def pop(self) -> EpisodeEvent | None:
        return self._events.popleft() if self._events else None
