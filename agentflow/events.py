"""Run instrumentation: a tiny synchronous event bus plus a console reporter."""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger("agentflow")


@dataclass(frozen=True)
class Event:
    kind: str
    run_id: str
    step: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=time.time)


Listener = Callable[[Event], None]


class EventBus:
    """Fan-out for run events. Listener exceptions never break a run."""

    def __init__(self, listeners: list[Listener] | None = None) -> None:
        self._listeners: list[Listener] = list(listeners or [])

    def subscribe(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def emit(self, kind: str, run_id: str, step: str | None = None, **data: Any) -> None:
        event = Event(kind=kind, run_id=run_id, step=step, data=data)
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:  # a broken reporter must not fail the workflow
                log.exception("event listener raised on %s", kind)


class ConsoleReporter:
    """Human-readable progress, one line per event."""

    ICONS = {
        "run.start": "▶",
        "run.end": "■",
        "step.start": "·",
        "step.ok": "✓",
        "step.failed": "✗",
        "step.skipped": "–",
        "step.retry": "↻",
        "agent.llm": "~",
    }

    def __init__(self, stream: Any = None, verbose: bool = False) -> None:
        self.stream = stream or sys.stderr
        self.verbose = verbose

    def __call__(self, event: Event) -> None:
        if event.kind == "agent.llm" and not self.verbose:
            return
        icon = self.ICONS.get(event.kind, " ")
        label = event.step or event.data.get("workflow", "")
        detail = self._detail(event)
        print(f"  {icon} {event.kind:<12} {label}{detail}", file=self.stream, flush=True)

    def _detail(self, event: Event) -> str:
        bits = []
        if "duration" in event.data:
            bits.append(f"{event.data['duration']:.2f}s")
        if "attempt" in event.data:
            bits.append(f"attempt {event.data['attempt']}")
        if "error" in event.data:
            bits.append(str(event.data["error"])[:160])
        if "reason" in event.data:
            bits.append(str(event.data["reason"]))
        if "status" in event.data:
            bits.append(str(event.data["status"]))
        return ("  " + " | ".join(bits)) if bits else ""
