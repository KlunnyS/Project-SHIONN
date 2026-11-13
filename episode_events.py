"""Parse SHIONN Portal events from arbitrarily chunked netconsole text."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass


EVENT_PATTERN = re.compile(
    r"EVT\|(chamber_ready|goal_reached|episode_failed)\|([^\r\n]*)"
)


@dataclass(frozen=True)
class EpisodeEvent:
    name: str
    value: str


class EpisodeEventStream:
    """Turn arbitrarily chunked netconsole text into complete SHIONN events."""

    def __init__(self) -> None:
        self._partial_line = ""
        self._events: deque[EpisodeEvent] = deque()

    def clear(self) -> None:
        self._partial_line = ""
        self._events.clear()

    @property
    def has_partial_line(self) -> bool:
        return bool(self._partial_line)

    def feed(self, text: str) -> None:
        if not text:
            return
        lines = (self._partial_line + text).splitlines(keepends=True)
        self._partial_line = ""
        if lines and not lines[-1].endswith(("\n", "\r")):
            self._partial_line = lines.pop()
        for line in lines:
            match = EVENT_PATTERN.search(line)
            if match:
                self._events.append(EpisodeEvent(match.group(1), match.group(2).strip()))

    def pop(self) -> EpisodeEvent | None:
        return self._events.popleft() if self._events else None
