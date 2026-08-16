"""The plumbing that turns Python functions into tools the model can call."""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

# A confirm callback asks the human "is this OK?" and returns True/False.
ConfirmFn = Callable[[str], bool]


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., str]
    # If set, the human is asked before this tool runs. Used for anything
    # that changes the machine rather than just reading from it.
    confirm_prompt: Callable[[dict[str, Any]], str] | None = None

    def spec(self) -> dict[str, Any]:
        """The shape the Claude API expects in its `tools` list."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass
class ToolResult:
    text: str
    is_error: bool = False


class ToolRegistry:
    """Holds every tool C-Net has and runs them on request."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def add(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any] | None = None,
        confirm_prompt: Callable[[dict[str, Any]], str] | None = None,
    ):
        """Decorator form:

            @registry.add("get_time", "Return the current local time.")
            def get_time() -> str:
                ...
        """

        def decorator(fn: Callable[..., str]) -> Callable[..., str]:
            self.register(
                Tool(
                    name=name,
                    description=description,
                    input_schema=input_schema
                    or {"type": "object", "properties": {}},
                    fn=fn,
                    confirm_prompt=confirm_prompt,
                )
            )
            return fn

        return decorator

    # --- lookups ------------------------------------------------------
    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def all(self) -> list[Tool]:
        return [self._tools[name] for name in sorted(self._tools)]

    def specs(self) -> list[dict[str, Any]]:
        """Every tool, in the format the API wants."""
        return [tool.spec() for tool in self.all()]

    # --- execution ----------------------------------------------------
    def run(
        self,
        name: str,
        args: dict[str, Any],
        confirm: ConfirmFn | None = None,
    ) -> ToolResult:
        """Run one tool call and always come back with a string.

        A tool that raises must never crash the agent — the model is told
        what went wrong and gets to try something else. That error message
        is a normal tool result with is_error set, not an exception.
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                f"No such tool: {name}. Available: {', '.join(self.names())}",
                is_error=True,
            )

        if tool.confirm_prompt is not None and confirm is not None:
            question = tool.confirm_prompt(args)
            if not confirm(question):
                return ToolResult(
                    "The user declined this action. Do not retry it; "
                    "ask what they would prefer instead.",
                    is_error=False,
                )

        try:
            output = tool.fn(**args)
        except TypeError as exc:
            return ToolResult(f"Bad arguments for {name}: {exc}", is_error=True)
        except Exception as exc:  # noqa: BLE001 - tools must never crash C-Net
            detail = traceback.format_exc(limit=2)
            return ToolResult(f"{name} failed: {exc}\n{detail}", is_error=True)

        text = str(output) if output is not None else "(no output)"
        # Keep a runaway tool from flooding the context window.
        limit = 20_000
        if len(text) > limit:
            text = text[:limit] + f"\n...[truncated, {len(text)} chars total]"
        return ToolResult(text)
