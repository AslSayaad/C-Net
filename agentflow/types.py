"""Core data types shared across the workflow engine."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Status(str, Enum):
    """Terminal state of a step or of a whole run."""

    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class Usage:
    """Token accounting for one or more model calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    calls: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens
            + other.cache_read_input_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens
            + other.cache_creation_input_tokens,
            calls=self.calls + other.calls,
        )

    @property
    def total_input_tokens(self) -> int:
        """Every input token billed, cached or not."""
        return (
            self.input_tokens
            + self.cache_read_input_tokens
            + self.cache_creation_input_tokens
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "calls": self.calls,
        }


@dataclass
class StepResult:
    """Outcome of a single workflow step."""

    step: str
    agent: str
    status: Status
    output: Any = None
    error: str | None = None
    attempts: int = 0
    started_at: float = 0.0
    ended_at: float = 0.0
    usage: Usage = field(default_factory=Usage)

    @property
    def duration(self) -> float:
        return max(0.0, self.ended_at - self.started_at)

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "agent": self.agent,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "attempts": self.attempts,
            "duration": round(self.duration, 3),
            "usage": self.usage.as_dict(),
        }


@dataclass
class RunResult:
    """Outcome of a whole workflow run."""

    run_id: str
    workflow: str
    status: Status
    steps: dict[str, StepResult] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    started_at: float = 0.0
    ended_at: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.ended_at - self.started_at)

    @property
    def usage(self) -> Usage:
        total = Usage()
        for result in self.steps.values():
            total = total + result.usage
        return total

    @property
    def failures(self) -> dict[str, StepResult]:
        return {
            name: result
            for name, result in self.steps.items()
            if result.status is Status.FAILED
        }

    def output(self, step: str) -> Any:
        """Output of a completed step, or None if it never produced one."""
        result = self.steps.get(step)
        return result.output if result else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "workflow": self.workflow,
            "status": self.status.value,
            "duration": round(self.duration, 3),
            "usage": self.usage.as_dict(),
            "steps": {name: r.as_dict() for name, r in self.steps.items()},
        }


def new_run_id() -> str:
    return f"run_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
