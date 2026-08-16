"""Name -> agent lookup, so workflows can reference agents as strings."""

from __future__ import annotations

from typing import Iterator

from .agent import Agent


class AgentRegistry:
    """A mutable collection of agents keyed by name."""

    def __init__(self, agents: list[Agent] | None = None) -> None:
        self._agents: dict[str, Agent] = {}
        for item in agents or []:
            self.register(item)

    def register(self, agent: Agent, name: str | None = None) -> Agent:
        key = name or agent.name
        if key in self._agents and self._agents[key] is not agent:
            raise ValueError(f"an agent named {key!r} is already registered")
        self._agents[key] = agent
        return agent

    def extend(self, agents: list[Agent]) -> "AgentRegistry":
        for item in agents:
            self.register(item)
        return self

    def get(self, name: str) -> Agent:
        try:
            return self._agents[name]
        except KeyError:
            known = ", ".join(sorted(self._agents)) or "<empty registry>"
            raise KeyError(f"no agent named {name!r}; registered: {known}") from None

    def __contains__(self, name: object) -> bool:
        return name in self._agents

    def __iter__(self) -> Iterator[Agent]:
        return iter(self._agents.values())

    def __len__(self) -> int:
        return len(self._agents)

    def names(self) -> list[str]:
        return sorted(self._agents)
