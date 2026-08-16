"""Test doubles: run whole workflows without touching the API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .llm import LLMResponse
from .types import Usage


@dataclass
class RecordedCall:
    caller: str | None
    system: str
    prompt: str | None
    output_schema: dict[str, Any] | None


class StubClient:
    """A `ClaudeClient` stand-in.

    Returns, in order of preference: a canned response registered for the
    calling agent, a value synthesised from the agent's output schema, or a
    fixed string. Every call is recorded on `calls` for assertions.
    """

    def __init__(
        self,
        responses: dict[str, Any] | None = None,
        default_text: str = "## Stub output\n\nGenerated without calling the API.",
        fail_for: dict[str, Exception] | None = None,
    ) -> None:
        self.responses = dict(responses or {})
        self.default_text = default_text
        self.fail_for = dict(fail_for or {})
        self.calls: list[RecordedCall] = []

    async def complete(
        self,
        system: str,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        output_schema: dict[str, Any] | None = None,
        effort: str | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        caller: str | None = None,
    ) -> LLMResponse:
        self.calls.append(RecordedCall(caller, system, prompt, output_schema))

        if caller in self.fail_for:
            raise self.fail_for[caller]

        if caller in self.responses:
            value = self.responses[caller]
            if callable(value):
                value = value(prompt)
        elif output_schema is not None:
            value = stub_from_schema(output_schema)
        else:
            value = self.default_text

        parsed = value if output_schema is not None else None
        text = value if isinstance(value, str) else repr(value)
        return LLMResponse(
            text=text,
            parsed=parsed,
            stop_reason="end_turn",
            model=model or "stub-model",
            usage=Usage(input_tokens=100, output_tokens=50, calls=1),
        )

    def calls_for(self, caller: str) -> list[RecordedCall]:
        return [call for call in self.calls if call.caller == caller]

    async def aclose(self) -> None:  # matches ClaudeClient's interface
        return None


def stub_from_schema(schema: dict[str, Any], name: str = "value") -> Any:
    """Synthesise the smallest value satisfying a JSON Schema object."""
    kind = schema.get("type")
    if "enum" in schema:
        return schema["enum"][0]
    if kind == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", list(properties))
        return {
            key: stub_from_schema(sub_schema, key)
            for key, sub_schema in properties.items()
            if key in required
        }
    if kind == "array":
        item_schema = schema.get("items", {"type": "string"})
        return [stub_from_schema(item_schema, name)]
    if kind in ("number", "integer"):
        return 1
    if kind == "boolean":
        return False
    if kind == "null":
        return None
    return f"stub {name}"
