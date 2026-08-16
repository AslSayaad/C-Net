"""Shared state every agent in a run reads from and writes to."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
from typing import Any


class RefError(KeyError):
    """A reference pointed at state that does not exist."""


@dataclass(frozen=True)
class Ref:
    """A lazily-resolved pointer into the blackboard.

    The path is dotted, with integer segments indexing into lists:
    ``Ref("research.findings.0.claim")``.
    """

    path: str
    default: Any = None
    has_default: bool = False

    def resolve(self, state: dict[str, Any]) -> Any:
        cursor: Any = state
        for segment in self.path.split("."):
            try:
                if isinstance(cursor, (list, tuple)):
                    cursor = cursor[int(segment)]
                else:
                    cursor = cursor[segment]
            except (KeyError, IndexError, TypeError, ValueError):
                if self.has_default:
                    return self.default
                raise RefError(
                    f"cannot resolve {self.path!r}: no value at segment {segment!r}"
                ) from None
        return cursor


def ref(path: str, *args: Any) -> Ref:
    """Build a `Ref`. Pass a second positional argument to supply a default."""
    if args:
        if len(args) > 1:
            raise TypeError("ref() takes at most one default value")
        return Ref(path, args[0], has_default=True)
    return Ref(path)


def resolve(value: Any, state: dict[str, Any]) -> Any:
    """Recursively replace every `Ref` inside `value` with its resolved value."""
    if isinstance(value, Ref):
        return value.resolve(state)
    if isinstance(value, dict):
        return {key: resolve(item, state) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve(item, state) for item in value]
    if isinstance(value, tuple):
        return tuple(resolve(item, state) for item in value)
    return value


class Blackboard:
    """Async-safe key/value store scoped to one run.

    Step outputs land under their step name, so `ref("analysis.risks")` reads
    the `risks` field of the step named `analysis`.
    """

    def __init__(self, initial: dict[str, Any] | None = None) -> None:
        self._state: dict[str, Any] = dict(initial or {})
        self._lock = asyncio.Lock()
        self._writes: list[str] = []

    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            self._state[key] = value
            self._writes.append(key)

    async def update(self, values: dict[str, Any]) -> None:
        async with self._lock:
            self._state.update(values)
            self._writes.extend(values)

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self._state:
            raise RefError(f"required key {key!r} is not on the blackboard")
        return self._state[key]

    def resolve(self, value: Any) -> Any:
        """Resolve `Ref`s (and nested structures containing them) against state."""
        return resolve(value, self._state)

    def snapshot(self) -> dict[str, Any]:
        """A deep copy, safe to hand to callers without exposing live state."""
        return copy.deepcopy(self._state)

    @property
    def write_log(self) -> list[str]:
        return list(self._writes)

    def __contains__(self, key: object) -> bool:
        return key in self._state

    def __repr__(self) -> str:
        return f"Blackboard(keys={sorted(self._state)})"
