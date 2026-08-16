"""Deterministic agents and LLM agents driven by a stub client."""

import pytest

from agentflow import AgentContext, Status, Step, Workflow, run_workflow, ref
from agentflow.agents.deterministic import (
    estimate_effort,
    normalize_brief,
    publish_report,
    score_readiness,
    slugify,
)
from agentflow.agents.llm_agents import build_llm_agents
from agentflow.llm_agent import LLMAgent, array_of, schema
from agentflow.testing import StubClient

BRIEF = {
    "client": "Northwind Logistics",
    "objective": "forecast delivery delays",
    "domain": "supply chain",
    "constraints": ["stay inside the client VPC"],
    "timeline_weeks": 10,
    "budget_usd": 100_000,
}


def context(**inputs):
    return AgentContext(run_id="run_test", step="s", inputs=inputs)


# --- deterministic agents ---------------------------------------------------


async def test_normalize_brief_derives_research_questions():
    result = await normalize_brief.run(context(brief=BRIEF))
    assert result["client"] == "Northwind Logistics"
    assert result["day_rate_usd"] > 0
    # One question per standard axis, plus one for the stated constraints.
    assert len(result["research_questions"]) == 4
    assert "stay inside the client VPC" in result["research_questions"][-1]


async def test_normalize_brief_without_constraints_skips_that_question():
    brief = {**BRIEF, "constraints": []}
    result = await normalize_brief.run(context(brief=brief))
    assert len(result["research_questions"]) == 3


async def test_normalize_brief_rejects_a_brief_missing_fields():
    with pytest.raises(ValueError, match="missing required field"):
        await normalize_brief.run(context(brief={"client": "ACME"}))


async def test_normalize_brief_rejects_a_non_dict():
    with pytest.raises(TypeError):
        await normalize_brief.run(context(brief="a string"))


def test_slugify_is_url_safe():
    assert slugify("Northwind Logistics: 48h!") == "northwind-logistics-48h"
    assert slugify("!!!") == "untitled"


async def test_estimate_effort_prices_phases_with_contingency():
    phases = [
        {"name": "discovery", "engineer_days": 10},
        {"name": "build", "engineer_days": 30},
    ]
    result = await estimate_effort.run(
        context(phases=phases, day_rate_usd=1000, budget_usd=60_000)
    )
    assert result["base_days"] == 40
    assert result["base_cost_usd"] == 40_000
    assert result["contingency_usd"] == 8_000
    assert result["total_cost_usd"] == 48_000
    assert result["estimated_weeks"] == 8.0
    assert result["within_budget"] is True


async def test_estimate_effort_flags_going_over_budget():
    result = await estimate_effort.run(
        context(
            phases=[{"name": "build", "engineer_days": 100}],
            day_rate_usd=1000,
            budget_usd=50_000,
        )
    )
    assert result["total_cost_usd"] == 120_000
    assert result["within_budget"] is False


async def test_estimate_effort_rejects_unsized_phases():
    with pytest.raises(ValueError, match="engineer_days"):
        await estimate_effort.run(
            context(phases=[{"name": "build"}], day_rate_usd=1000, budget_usd=0)
        )
    with pytest.raises(ValueError, match="no phases"):
        await estimate_effort.run(context(phases=[], day_rate_usd=1000, budget_usd=0))


async def test_score_readiness_penalises_risk_budget_and_timeline():
    estimate = {"total_cost_usd": 200_000, "estimated_weeks": 20, "within_budget": False}
    result = await score_readiness.run(
        context(
            risks=[{"severity": "high"}, {"severity": "low"}],
            estimate=estimate,
            brief=BRIEF,
        )
    )
    assert result["penalties"]["risk"] == 21
    assert result["penalties"]["budget"] > 0
    assert result["penalties"]["timeline"] > 0
    assert result["score"] < 60
    assert result["recommendation"] in {"revise", "no-go"}


async def test_score_readiness_is_clean_when_everything_fits():
    estimate = {"total_cost_usd": 50_000, "estimated_weeks": 6, "within_budget": True}
    result = await score_readiness.run(context(risks=[], estimate=estimate, brief=BRIEF))
    assert result["score"] == 100
    assert result["recommendation"] == "go"


async def test_score_readiness_is_deterministic():
    args = {
        "risks": [{"severity": "medium"}],
        "estimate": {"total_cost_usd": 10, "estimated_weeks": 1, "within_budget": True},
        "brief": BRIEF,
    }
    first = await score_readiness.run(context(**args))
    second = await score_readiness.run(context(**args))
    assert first == second


async def test_publish_report_writes_a_file(tmp_path):
    ctx = AgentContext(
        run_id="run_x_abc123",
        step="publish",
        inputs={
            "proposal_markdown": "## Approach\n\nDetails.",
            "brief": {**BRIEF, "slug": "northwind"},
            "estimate": {"total_cost_usd": 48_000, "base_days": 40, "estimated_weeks": 8},
            "readiness": {"score": 82, "recommendation": "go"},
            "output_dir": str(tmp_path),
        },
    )
    result = await publish_report.run(ctx)
    written = (tmp_path / "northwind-abc123.md").read_text()
    assert result["path"].endswith("northwind-abc123.md")
    assert "# Proposal — Northwind Logistics" in written
    assert "82/100" in written
    assert "## Approach" in written


# --- LLM agents against the stub client -------------------------------------


async def test_llm_agent_renders_templates_and_returns_text():
    stub = StubClient(responses={"greeter": "hi there"})
    llm_agent = LLMAgent(
        name="greeter",
        system="You greet people.",
        prompt="Greet {name} in {language}.",
        client=stub,
    )
    output = await llm_agent.run(context(name="Ada", language="French"))

    assert output == "hi there"
    assert stub.calls[0].prompt == "Greet Ada in French."
    assert stub.calls[0].caller == "greeter"


async def test_llm_agent_accepts_a_callable_prompt():
    stub = StubClient()
    llm_agent = LLMAgent(
        name="a",
        system="s",
        prompt=lambda ctx: f"step={ctx.step} value={ctx.require('value')}",
        client=stub,
    )
    await llm_agent.run(context(value=7))
    assert stub.calls[0].prompt == "step=s value=7"


async def test_llm_agent_returns_parsed_structured_output():
    output_schema = schema(
        {"verdict": {"type": "string"}, "notes": array_of({"type": "string"})}
    )
    stub = StubClient(responses={"judge": {"verdict": "ship", "notes": ["fine"]}})
    llm_agent = LLMAgent(
        name="judge", system="s", prompt="p", output_schema=output_schema, client=stub
    )
    assert await llm_agent.run(context()) == {"verdict": "ship", "notes": ["fine"]}
    assert stub.calls[0].output_schema == output_schema


async def test_llm_agent_records_usage_on_the_context():
    llm_agent = LLMAgent(name="a", system="s", prompt="p", client=StubClient())
    ctx = context()
    await llm_agent.run(ctx)
    assert ctx.usage.calls == 1
    assert ctx.usage.input_tokens == 100


async def test_llm_agent_binding_is_explicit():
    llm_agent = LLMAgent(name="a", system="s", prompt="p")
    assert llm_agent.is_bound is False
    llm_agent.bind(StubClient())
    assert llm_agent.is_bound is True


def test_schema_helper_produces_strict_schemas():
    built = schema({"a": {"type": "string"}, "b": {"type": "integer"}}, required=["a"])
    assert built["additionalProperties"] is False
    assert built["required"] == ["a"]
    assert array_of({"type": "string"})["type"] == "array"


def test_example_agents_use_strict_schemas():
    for llm_agent in build_llm_agents():
        if llm_agent.output_schema:
            assert llm_agent.output_schema["additionalProperties"] is False
            assert llm_agent.output_schema["required"]


def test_example_agents_have_distinct_names_and_descriptions():
    agents = build_llm_agents()
    assert len({a.name for a in agents}) == len(agents)
    assert all(a.description for a in agents)


async def test_llm_agent_failure_is_retried_by_the_engine():
    stub = StubClient(fail_for={"flaky": RuntimeError("overloaded")})
    llm_agent = LLMAgent(name="flaky", system="s", prompt="p", client=stub)
    workflow = Workflow(
        name="retry",
        steps=[Step(name="s", agent=llm_agent, retries=2, retry_backoff=0.001)],
    )
    result = await run_workflow(workflow)
    assert result.steps["s"].status is Status.FAILED
    assert len(stub.calls) == 3
