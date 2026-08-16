"""Agents backed by a Claude model."""

from __future__ import annotations

from typing import Any, Callable

from .agent import Agent, AgentContext
from .llm import ClaudeClient, Effort, LLMResponse, default_client

PromptSource = str | Callable[[AgentContext], str]


class LLMAgent(Agent):
    """An agent whose work is a single model call.

    The system prompt defines the role and stays byte-identical across runs so
    it caches well; the user prompt carries the per-run inputs. Give an
    `output_schema` when downstream steps need structured fields rather than
    prose — the model is then constrained to that JSON Schema.
    """

    def __init__(
        self,
        name: str,
        system: PromptSource,
        prompt: PromptSource,
        description: str = "",
        output_schema: dict[str, Any] | None = None,
        client: ClaudeClient | None = None,
        effort: Effort | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(name=name, description=description)
        self.system = system
        self.prompt = prompt
        self.output_schema = output_schema
        self.effort = effort
        self.max_tokens = max_tokens
        self.model = model
        self._client = client

    @property
    def client(self) -> ClaudeClient:
        if self._client is None:
            self._client = default_client()
        return self._client

    @property
    def is_bound(self) -> bool:
        """True once a client has been supplied explicitly."""
        return self._client is not None

    def bind(self, client: ClaudeClient) -> "LLMAgent":
        """Point this agent at a specific client (used by the runner and tests)."""
        self._client = client
        return self

    def render(self, source: PromptSource, ctx: AgentContext) -> str:
        if callable(source):
            return source(ctx)
        return source.format(**self._template_vars(ctx))

    def _template_vars(self, ctx: AgentContext) -> dict[str, Any]:
        variables: dict[str, Any] = dict(ctx.inputs)
        variables.setdefault("item", ctx.item)
        variables.setdefault("index", ctx.index)
        variables.setdefault("step", ctx.step)
        return variables

    async def run(self, ctx: AgentContext) -> Any:
        system = self.render(self.system, ctx)
        prompt = self.render(self.prompt, ctx)
        ctx.emit("agent.llm", agent=self.name, prompt_chars=len(prompt))

        response: LLMResponse = await self.client.complete(
            system=system,
            prompt=prompt,
            output_schema=self.output_schema,
            effort=self.effort,
            max_tokens=self.max_tokens,
            model=self.model,
            caller=self.name,
        )
        ctx.record_usage(response.usage)
        return response.value


def schema(
    properties: dict[str, Any],
    required: list[str] | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Build a strict JSON Schema object.

    Structured outputs require `additionalProperties: false` and an explicit
    `required` list, so this fills both in rather than leaving them to be
    forgotten at each call site.
    """
    built: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required if required is not None else list(properties),
        "additionalProperties": False,
    }
    if description:
        built["description"] = description
    return built


def array_of(items: dict[str, Any], description: str | None = None) -> dict[str, Any]:
    """Shorthand for an array property in a schema."""
    built: dict[str, Any] = {"type": "array", "items": items}
    if description:
        built["description"] = description
    return built
