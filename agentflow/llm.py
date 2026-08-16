"""Claude client wrapper used by LLM-backed agents.

Everything model-facing is funnelled through here so the whole system has one
place to configure the model, effort, thinking, caching, and refusal handling.

Defaults follow current Claude API guidance:
  * `claude-opus-5` with adaptive thinking (no `budget_tokens`, no sampling
    parameters — those are rejected by this model family).
  * `output_config.effort` for the cost/quality dial.
  * Server-side refusal fallbacks, so a declined request is retried on a
    fallback model inside the same call instead of surfacing as a dead end.
  * Streaming whenever `max_tokens` is large enough to risk an HTTP timeout.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal

from .types import Usage

DEFAULT_MODEL = "claude-opus-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
STREAM_THRESHOLD = 16_000
Effort = Literal["low", "medium", "high", "xhigh", "max"]


class LLMError(RuntimeError):
    """Base class for model-call failures."""


class LLMNotConfigured(LLMError):
    """The `anthropic` package or an API credential is missing."""


class RefusalError(LLMError):
    """The request was declined by safety classifiers (and by any fallback)."""

    def __init__(self, category: str | None, explanation: str | None) -> None:
        super().__init__(
            f"model declined the request (category={category or 'unknown'})"
            + (f": {explanation}" if explanation else "")
        )
        self.category = category
        self.explanation = explanation


class OutputParseError(LLMError):
    """A structured response did not contain parseable JSON."""


@dataclass
class LLMResponse:
    text: str
    parsed: Any | None
    stop_reason: str | None
    model: str
    usage: Usage
    raw: Any = None

    @property
    def value(self) -> Any:
        """Parsed JSON when a schema was requested, else the raw text."""
        return self.parsed if self.parsed is not None else self.text


class ClaudeClient:
    """Thin async wrapper over the Anthropic Messages API.

    One instance is shared by every LLM agent in a run; per-agent overrides
    (effort, max_tokens, model) are passed to `complete`.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: Effort = "high",
        max_tokens: int = 16_000,
        api_key: str | None = None,
        enable_fallbacks: bool = True,
        cache_system_prompt: bool = True,
        stream_threshold: int = STREAM_THRESHOLD,
        client: Any = None,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.enable_fallbacks = enable_fallbacks
        self.cache_system_prompt = cache_system_prompt
        self.stream_threshold = stream_threshold
        self._client = client
        self._api_key = api_key
        self._max_retries = max_retries

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as exc:  # pragma: no cover - import guard
                raise LLMNotConfigured(
                    "the 'anthropic' package is required for LLM agents: "
                    "pip install -r agentflow/requirements.txt"
                ) from exc
            kwargs: dict[str, Any] = {"max_retries": self._max_retries}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = AsyncAnthropic(**kwargs)
        return self._client

    async def complete(
        self,
        system: str,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        output_schema: dict[str, Any] | None = None,
        effort: Effort | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        caller: str | None = None,
    ) -> LLMResponse:
        """Run one model call and return its text plus parsed output.

        Supply `output_schema` (a JSON Schema object) to constrain the response
        to that shape — the result is validated server-side and returned in
        `LLMResponse.parsed`. `caller` names the agent making the call; it is
        used by instrumentation and test doubles and ignored by the API.
        """
        if prompt is None and not messages:
            raise ValueError("complete() needs either a prompt or messages")
        if prompt is not None and messages:
            raise ValueError("pass either a prompt or messages, not both")

        max_tokens = max_tokens or self.max_tokens
        params: dict[str, Any] = {
            "model": model or self.model,
            "max_tokens": max_tokens,
            "system": self._system_blocks(system),
            "messages": messages or [{"role": "user", "content": prompt}],
            "thinking": {"type": "adaptive"},
            "output_config": self._output_config(effort, output_schema),
        }
        if tools:
            params["tools"] = tools

        message = await self._send(params, stream=max_tokens > self.stream_threshold)
        return self._to_response(message, expect_json=output_schema is not None)

    def _system_blocks(self, system: str) -> list[dict[str, Any]]:
        block: dict[str, Any] = {"type": "text", "text": system}
        if self.cache_system_prompt:
            # Caching is a prefix match; short system prompts simply won't
            # create an entry, which is harmless.
            block["cache_control"] = {"type": "ephemeral"}
        return [block]

    def _output_config(
        self, effort: Effort | None, output_schema: dict[str, Any] | None
    ) -> dict[str, Any]:
        config: dict[str, Any] = {"effort": effort or self.effort}
        if output_schema:
            config["format"] = {"type": "json_schema", "schema": output_schema}
        return config

    async def _send(self, params: dict[str, Any], stream: bool) -> Any:
        if self.enable_fallbacks:
            params = {**params, "betas": [FALLBACK_BETA], "fallbacks": "default"}
            endpoint = self.client.beta.messages
        else:
            endpoint = self.client.messages

        if stream:
            async with endpoint.stream(**params) as response_stream:
                return await response_stream.get_final_message()
        return await endpoint.create(**params)

    def _to_response(self, message: Any, expect_json: bool) -> LLMResponse:
        stop_reason = getattr(message, "stop_reason", None)
        if stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            raise RefusalError(
                getattr(details, "category", None),
                getattr(details, "explanation", None),
            )

        text = "".join(
            block.text
            for block in getattr(message, "content", [])
            if getattr(block, "type", None) == "text"
        ).strip()

        parsed = None
        if expect_json:
            if not text:
                raise OutputParseError(
                    f"expected JSON output but the response was empty "
                    f"(stop_reason={stop_reason})"
                )
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise OutputParseError(
                    f"response was not valid JSON: {text[:300]}"
                ) from exc

        return LLMResponse(
            text=text,
            parsed=parsed,
            stop_reason=stop_reason,
            model=getattr(message, "model", self.model),
            usage=_usage_from(getattr(message, "usage", None)),
            raw=message,
        )

    async def aclose(self) -> None:
        client = self._client
        if client is not None and hasattr(client, "close"):
            await client.close()


def _usage_from(usage: Any) -> Usage:
    if usage is None:
        return Usage(calls=1)
    return Usage(
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        calls=1,
    )


def default_client(**overrides: Any) -> ClaudeClient:
    """Build a client from environment variables, with keyword overrides.

    Recognised: AGENTFLOW_MODEL, AGENTFLOW_EFFORT, AGENTFLOW_MAX_TOKENS,
    AGENTFLOW_DISABLE_FALLBACKS.
    """
    settings: dict[str, Any] = {
        "model": os.environ.get("AGENTFLOW_MODEL", DEFAULT_MODEL),
        "effort": os.environ.get("AGENTFLOW_EFFORT", "high"),
        "max_tokens": int(os.environ.get("AGENTFLOW_MAX_TOKENS", "16000")),
        "enable_fallbacks": os.environ.get("AGENTFLOW_DISABLE_FALLBACKS", "") == "",
    }
    settings.update(overrides)
    return ClaudeClient(**settings)
