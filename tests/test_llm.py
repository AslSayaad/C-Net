"""The Claude client wrapper: request shape, parsing, and failure modes."""

from types import SimpleNamespace

import pytest

from agentflow.llm import (
    FALLBACK_BETA,
    ClaudeClient,
    OutputParseError,
    RefusalError,
    default_client,
)


def message(text="hello", stop_reason="end_turn", usage=None):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        stop_details=None,
        model="claude-opus-5",
        usage=usage
        or SimpleNamespace(
            input_tokens=120,
            output_tokens=40,
            cache_read_input_tokens=1000,
            cache_creation_input_tokens=0,
        ),
    )


class FakeEndpoint:
    def __init__(self, response):
        self.response = response
        self.calls = []
        self.streamed = False

    async def create(self, **params):
        self.calls.append(params)
        return self.response

    def stream(self, **params):
        self.calls.append(params)
        self.streamed = True
        endpoint = self

        class Stream:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get_final_message(self):
                return endpoint.response

        return Stream()


class FakeAnthropic:
    def __init__(self, response=None):
        response = response or message()
        self.messages = FakeEndpoint(response)
        self.beta = SimpleNamespace(messages=FakeEndpoint(response))


async def test_request_uses_adaptive_thinking_and_effort():
    fake = FakeAnthropic()
    client = ClaudeClient(client=fake, effort="medium", enable_fallbacks=False)
    await client.complete(system="s", prompt="p")

    params = fake.messages.calls[0]
    assert params["model"] == "claude-opus-5"
    assert params["thinking"] == {"type": "adaptive"}
    assert params["output_config"]["effort"] == "medium"
    # Sampling parameters are rejected by this model family.
    assert "temperature" not in params
    assert "top_p" not in params
    assert "top_k" not in params
    # As is the fixed thinking budget.
    assert "budget_tokens" not in str(params["thinking"])


async def test_per_call_overrides_win_over_client_defaults():
    fake = FakeAnthropic()
    client = ClaudeClient(client=fake, effort="low", max_tokens=1000, enable_fallbacks=False)
    await client.complete(system="s", prompt="p", effort="max", max_tokens=2000, model="other")

    params = fake.messages.calls[0]
    assert params["output_config"]["effort"] == "max"
    assert params["max_tokens"] == 2000
    assert params["model"] == "other"


async def test_system_prompt_is_cached_by_default():
    fake = FakeAnthropic()
    client = ClaudeClient(client=fake, enable_fallbacks=False)
    await client.complete(system="stable instructions", prompt="p")

    block = fake.messages.calls[0]["system"][0]
    assert block["text"] == "stable instructions"
    assert block["cache_control"] == {"type": "ephemeral"}


async def test_caching_can_be_disabled():
    fake = FakeAnthropic()
    client = ClaudeClient(client=fake, enable_fallbacks=False, cache_system_prompt=False)
    await client.complete(system="s", prompt="p")
    assert "cache_control" not in fake.messages.calls[0]["system"][0]


async def test_fallbacks_go_through_the_beta_endpoint():
    fake = FakeAnthropic()
    client = ClaudeClient(client=fake, enable_fallbacks=True)
    await client.complete(system="s", prompt="p")

    assert not fake.messages.calls
    params = fake.beta.messages.calls[0]
    assert params["fallbacks"] == "default"
    assert params["betas"] == [FALLBACK_BETA]


async def test_large_max_tokens_streams():
    fake = FakeAnthropic()
    client = ClaudeClient(client=fake, enable_fallbacks=False, stream_threshold=1000)
    await client.complete(system="s", prompt="p", max_tokens=64_000)
    assert fake.messages.streamed is True


async def test_small_requests_do_not_stream():
    fake = FakeAnthropic()
    client = ClaudeClient(client=fake, enable_fallbacks=False, stream_threshold=1000)
    await client.complete(system="s", prompt="p", max_tokens=500)
    assert fake.messages.streamed is False


async def test_structured_output_is_parsed():
    fake = FakeAnthropic(message(text='{"verdict": "ship", "score": 9}'))
    client = ClaudeClient(client=fake, enable_fallbacks=False)
    schema = {
        "type": "object",
        "properties": {"verdict": {"type": "string"}, "score": {"type": "integer"}},
        "required": ["verdict", "score"],
        "additionalProperties": False,
    }
    response = await client.complete(system="s", prompt="p", output_schema=schema)

    assert response.parsed == {"verdict": "ship", "score": 9}
    assert response.value == {"verdict": "ship", "score": 9}
    assert fake.messages.calls[0]["output_config"]["format"] == {
        "type": "json_schema",
        "schema": schema,
    }


async def test_unparseable_structured_output_raises():
    fake = FakeAnthropic(message(text="not json at all"))
    client = ClaudeClient(client=fake, enable_fallbacks=False)
    with pytest.raises(OutputParseError, match="not valid JSON"):
        await client.complete(
            system="s", prompt="p", output_schema={"type": "object", "properties": {}}
        )


async def test_plain_text_response_is_returned_unparsed():
    fake = FakeAnthropic(message(text="  a prose answer  "))
    client = ClaudeClient(client=fake, enable_fallbacks=False)
    response = await client.complete(system="s", prompt="p")
    assert response.parsed is None
    assert response.value == "a prose answer"


async def test_refusal_raises_before_content_is_read():
    refused = SimpleNamespace(
        content=[],
        stop_reason="refusal",
        stop_details=SimpleNamespace(category="cyber", explanation="declined"),
        model="claude-opus-5",
        usage=None,
    )
    client = ClaudeClient(client=FakeAnthropic(refused), enable_fallbacks=False)
    with pytest.raises(RefusalError) as caught:
        await client.complete(system="s", prompt="p")
    assert caught.value.category == "cyber"


async def test_usage_is_reported_including_cache_reads():
    client = ClaudeClient(client=FakeAnthropic(), enable_fallbacks=False)
    response = await client.complete(system="s", prompt="p")
    assert response.usage.input_tokens == 120
    assert response.usage.cache_read_input_tokens == 1000
    assert response.usage.total_input_tokens == 1120
    assert response.usage.calls == 1


async def test_prompt_and_messages_are_mutually_exclusive():
    client = ClaudeClient(client=FakeAnthropic(), enable_fallbacks=False)
    with pytest.raises(ValueError):
        await client.complete(system="s")
    with pytest.raises(ValueError):
        await client.complete(system="s", prompt="p", messages=[{"role": "user", "content": "x"}])


def test_default_client_reads_the_environment(monkeypatch):
    monkeypatch.setenv("AGENTFLOW_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("AGENTFLOW_EFFORT", "low")
    monkeypatch.setenv("AGENTFLOW_MAX_TOKENS", "2048")
    monkeypatch.setenv("AGENTFLOW_DISABLE_FALLBACKS", "1")

    client = default_client()
    assert client.model == "claude-sonnet-5"
    assert client.effort == "low"
    assert client.max_tokens == 2048
    assert client.enable_fallbacks is False


def test_default_client_defaults_to_opus_with_fallbacks(monkeypatch):
    for name in (
        "AGENTFLOW_MODEL",
        "AGENTFLOW_EFFORT",
        "AGENTFLOW_MAX_TOKENS",
        "AGENTFLOW_DISABLE_FALLBACKS",
    ):
        monkeypatch.delenv(name, raising=False)
    client = default_client()
    assert client.model == "claude-opus-5"
    assert client.enable_fallbacks is True
