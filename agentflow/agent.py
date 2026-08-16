"""Agent base class and the deterministic (plain-Python) agent adapter."""

from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .blackboard import Blackboard
from .events import EventBus
from .types import Usage


@dataclass
class AgentContext:
    """Everything an agent is handed when it runs.

    `inputs` is the resolved input mapping declared on the step; `state` is a
    read-only view of the whole blackboard for agents that need more than their
    declared inputs. `item`/`index` are set only inside a `foreach` step.
    """

    run_id: str
    step: str
    inputs: dict[str, Any] = field(default_factory=dict)
    blackboard: Blackboard | None = None
    events: EventBus | None = None
    item: Any = None
    index: int | None = None
    usage: Usage = field(default_factory=Usage)
    attempts: int = 0

    @property
    def state(self) -> dict[str, Any]:
        return self.blackboard.snapshot() if self.blackboard else {}

    def get(self, key: str, default: Any = None) -> Any:
        """Read a declared input, falling back to the blackboard."""
        if key in self.inputs:
            return self.inputs[key]
        if self.blackboard is not None:
            return self.blackboard.get(key, default)
        return default

    def require(self, key: str) -> Any:
        value = self.get(key, _MISSING)
        if value is _MISSING:
            raise KeyError(f"step {self.step!r} requires input {key!r}")
        return value

    def emit(self, kind: str, **data: Any) -> None:
        if self.events is not None:
            self.events.emit(kind, self.run_id, self.step, **data)

    def record_usage(self, usage: Usage) -> None:
        """Accumulate token usage so the engine can report per-step cost."""
        self.usage = self.usage + usage


_MISSING = object()


class Agent(ABC):
    """A unit of work. Subclasses implement `run`.

    Agents are reused across steps and across runs, so they must not hold
    per-run mutable state — everything run-scoped lives on the context.
    """

    name: str = "agent"
    description: str = ""

    def __init__(self, name: str | None = None, description: str | None = None) -> None:
        if name:
            self.name = name
        if description:
            self.description = description

    @abstractmethod
    async def run(self, ctx: AgentContext) -> Any:
        """Do the work and return this step's output."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class FunctionAgent(Agent):
    """Wraps a plain callable as an agent.

    Sync callables run in a worker thread so blocking I/O (file reads, pandas,
    a training loop) never stalls the event loop.
    """

    def __init__(
        self,
        func: Callable[..., Any] | Callable[..., Awaitable[Any]],
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        super().__init__(
            name=name or getattr(func, "__name__", "function"),
            description=description or (func.__doc__ or "").strip().split("\n")[0],
        )
        self.func = func
        self._is_async = inspect.iscoroutinefunction(func)
        self._wants_context = "ctx" in inspect.signature(func).parameters

    async def run(self, ctx: AgentContext) -> Any:
        kwargs = dict(ctx.inputs)
        if self._wants_context:
            kwargs["ctx"] = ctx
        if self._is_async:
            return await self.func(**kwargs)
        return await asyncio.to_thread(self.func, **kwargs)


def agent(name: str | None = None, description: str | None = None):
    """Decorator turning a function into a `FunctionAgent`.

    The wrapped function receives the step's declared inputs as keyword
    arguments; add a `ctx` parameter to also receive the `AgentContext`.
    """

    def decorate(func: Callable[..., Any]) -> FunctionAgent:
        return FunctionAgent(func, name=name, description=description)

    return decorate
