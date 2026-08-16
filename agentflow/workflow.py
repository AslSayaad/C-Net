"""Workflow and step definitions, plus static validation of the graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .agent import Agent
from .blackboard import Ref

Predicate = Callable[[dict[str, Any]], bool]


class WorkflowError(ValueError):
    """The workflow graph is malformed."""


@dataclass
class Step:
    """One node in the workflow graph.

    Dependencies are the union of `depends_on` and every step referenced by a
    `Ref` in `inputs` or `foreach`, so wiring data through refs is usually
    enough to get the ordering right.

    Dependencies are *hard* by default: if one does not succeed, this step is
    skipped rather than run with missing inputs. A `Ref` carrying a default
    (``ref("revision", None)``) is a *soft* dependency instead — it still
    orders the step after that one, but a skipped or failed producer only means
    the default is used. That is how a conditional branch feeds a step that
    must run either way.
    """

    name: str
    agent: str | Agent
    inputs: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    when: Predicate | None = None
    foreach: Ref | None = None
    max_parallel: int = 4
    retries: int = 0
    retry_backoff: float = 1.0
    timeout: float | None = None
    optional: bool = False
    description: str = ""

    @property
    def agent_name(self) -> str:
        return self.agent if isinstance(self.agent, str) else self.agent.name

    def dependencies(self, known_steps: Iterable[str]) -> set[str]:
        """Every step that must finish before this one starts."""
        known = set(known_steps)
        deps = set(self.depends_on) | {
            name
            for name, _ in self._referenced(known)
        }
        deps.discard(self.name)
        return deps

    def hard_dependencies(self, known_steps: Iterable[str]) -> set[str]:
        """Dependencies whose failure or skip blocks this step."""
        known = set(known_steps)
        deps = set(self.depends_on) | {
            name for name, optional in self._referenced(known) if not optional
        }
        deps.discard(self.name)
        return deps

    def _referenced(self, known: set[str]) -> set[tuple[str, bool]]:
        found: set[tuple[str, bool]] = set()
        for value in (self.inputs, self.foreach):
            for name, optional in _referenced_steps(value):
                if name in known:
                    found.add((name, optional))
        # A step referenced both with and without a default is hard overall.
        hard = {name for name, optional in found if not optional}
        return {(name, optional) for name, optional in found if not (optional and name in hard)}


def _referenced_steps(value: Any) -> set[tuple[str, bool]]:
    """(step name, is-optional) for every `Ref` nested anywhere inside `value`."""
    if isinstance(value, Ref):
        return {(value.path.split(".")[0], value.has_default)}
    if isinstance(value, dict):
        return set().union(*(_referenced_steps(v) for v in value.values())) if value else set()
    if isinstance(value, (list, tuple)):
        return set().union(*(_referenced_steps(v) for v in value)) if value else set()
    return set()


@dataclass
class Workflow:
    """A named, validated DAG of steps."""

    name: str
    steps: list[Step]
    description: str = ""

    def __post_init__(self) -> None:
        self.validate()

    @property
    def step_names(self) -> list[str]:
        return [step.name for step in self.steps]

    def step(self, name: str) -> Step:
        for step in self.steps:
            if step.name == name:
                return step
        raise KeyError(name)

    def graph(self) -> dict[str, set[str]]:
        """Step name -> set of steps it must run after."""
        names = self.step_names
        return {step.name: step.dependencies(names) for step in self.steps}

    def hard_graph(self) -> dict[str, set[str]]:
        """Step name -> set of steps whose failure would skip it."""
        names = self.step_names
        return {step.name: step.hard_dependencies(names) for step in self.steps}

    def validate(self) -> None:
        if not self.steps:
            raise WorkflowError(f"workflow {self.name!r} has no steps")

        seen: set[str] = set()
        for step in self.steps:
            if not step.name:
                raise WorkflowError("every step needs a name")
            if step.name in seen:
                raise WorkflowError(f"duplicate step name {step.name!r}")
            seen.add(step.name)

        graph = self.graph()
        for name, deps in graph.items():
            unknown = deps - seen
            if unknown:
                raise WorkflowError(
                    f"step {name!r} depends on unknown step(s): {sorted(unknown)}"
                )
        for step in self.steps:
            missing = set(step.depends_on) - seen
            if missing:
                raise WorkflowError(
                    f"step {step.name!r} depends on unknown step(s): {sorted(missing)}"
                )

        cycle = _find_cycle(graph)
        if cycle:
            raise WorkflowError(f"workflow {self.name!r} has a cycle: {' -> '.join(cycle)}")

    def layers(self) -> list[list[str]]:
        """Steps grouped into waves that can run concurrently — used for plans."""
        graph = {name: set(deps) for name, deps in self.graph().items()}
        done: set[str] = set()
        waves: list[list[str]] = []
        while len(done) < len(graph):
            wave = sorted(n for n, deps in graph.items() if n not in done and deps <= done)
            if not wave:  # unreachable: validate() rejects cycles
                raise WorkflowError("cannot order steps; the graph has a cycle")
            waves.append(wave)
            done.update(wave)
        return waves

    def describe(self) -> str:
        lines = [f"workflow: {self.name}"]
        if self.description:
            lines.append(f"  {self.description}")
        for index, wave in enumerate(self.layers(), start=1):
            lines.append(f"  wave {index}:")
            for name in wave:
                step = self.step(name)
                marks = []
                if step.foreach:
                    marks.append(f"foreach {step.foreach.path}")
                if step.when:
                    marks.append("conditional")
                if step.optional:
                    marks.append("optional")
                if step.retries:
                    marks.append(f"retries={step.retries}")
                suffix = f"  [{', '.join(marks)}]" if marks else ""
                lines.append(f"    - {name} ({step.agent_name}){suffix}")
        return "\n".join(lines)


def _find_cycle(graph: dict[str, set[str]]) -> list[str] | None:
    """Return one cycle as a path, or None. Depth-first with a colour marking."""
    WHITE, GREY, BLACK = 0, 1, 2
    colour = dict.fromkeys(graph, WHITE)
    stack: list[str] = []

    def visit(node: str) -> list[str] | None:
        colour[node] = GREY
        stack.append(node)
        for dep in sorted(graph.get(node, ())):
            if colour.get(dep, BLACK) == GREY:
                return stack[stack.index(dep):] + [dep]
            if colour.get(dep, BLACK) == WHITE:
                found = visit(dep)
                if found:
                    return found
        stack.pop()
        colour[node] = BLACK
        return None

    for node in graph:
        if colour[node] == WHITE:
            found = visit(node)
            if found:
                return found
    return None
