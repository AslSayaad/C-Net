"""End-to-end runs of the shipped example workflow, with no API calls."""

import json

from agentflow import Engine, Status
from agentflow.cli import main
from agentflow.testing import StubClient
from agentflow.workflows.ai_ml_ops import EXAMPLE_INPUT, build

ANALYSIS = {
    "approach": "Gradient-boosted per-lane delay model on warehouse-native features.",
    "phases": [
        {"name": "Data readiness", "goal": "profile lane history", "engineer_days": 12},
        {"name": "Baseline model", "goal": "beat the naive baseline", "engineer_days": 18},
        {"name": "Deployment", "goal": "scheduled scoring in-warehouse", "engineer_days": 15},
    ],
    "risks": [
        {"risk": "Lane history is sparse", "severity": "high", "mitigation": "Pool by corridor"},
        {"risk": "Label leakage", "severity": "medium", "mitigation": "Freeze feature cutoffs"},
    ],
    "data_requirements": ["two years of lane-level scans"],
    "open_questions": ["who owns the scoring schedule?"],
}

CRITIQUE_CLEAN = {
    "verdict": "ship",
    "needs_revision": False,
    "issues": [],
    "strengths": ["numbers are consistent"],
}

CRITIQUE_DIRTY = {
    "verdict": "revise",
    "needs_revision": True,
    "issues": [
        {"issue": "Accuracy is overclaimed", "severity": "high", "fix": "State the baseline"}
    ],
    "strengths": ["clear phasing"],
}


def stub(critique):
    return StubClient(
        responses={
            "analyst": ANALYSIS,
            "writer": "## Understanding\n\nDraft body.",
            "critic": critique,
            "reviser": "## Understanding\n\nRevised body.",
        }
    )


async def run_example(client, tmp_path, **engine_kwargs):
    workflow, registry = build()
    engine = Engine(workflow, registry=registry, client=client, **engine_kwargs)
    inputs = {**EXAMPLE_INPUT, "output_dir": str(tmp_path)}
    return await engine.run(inputs)


async def test_clean_review_skips_the_revision_branch(tmp_path):
    client = stub(CRITIQUE_CLEAN)
    result = await run_example(client, tmp_path)

    assert result.status is Status.OK
    assert result.steps["revision"].status is Status.SKIPPED
    assert result.output("final") == "## Understanding\n\nDraft body."
    assert [call.caller for call in client.calls_for("reviser")] == []


async def test_flagged_review_runs_the_revision_branch(tmp_path):
    client = stub(CRITIQUE_DIRTY)
    result = await run_example(client, tmp_path)

    assert result.status is Status.OK
    assert result.steps["revision"].status is Status.OK
    assert result.output("final") == "## Understanding\n\nRevised body."
    assert len(client.calls_for("reviser")) == 1


async def test_research_is_fanned_out_one_call_per_question(tmp_path):
    client = stub(CRITIQUE_CLEAN)
    result = await run_example(client, tmp_path)

    questions = result.output("brief")["research_questions"]
    assert len(client.calls_for("researcher")) == len(questions)
    assert len(result.output("findings")) == len(questions)


async def test_costs_are_computed_in_code_not_by_the_model(tmp_path):
    result = await run_example(stub(CRITIQUE_CLEAN), tmp_path)

    estimate = result.output("estimate")
    assert estimate["base_days"] == 45
    assert estimate["base_cost_usd"] == 45 * 1_200
    assert estimate["total_cost_usd"] == round(45 * 1_200 * 1.2, 2)
    assert estimate["within_budget"] is True


async def test_readiness_reflects_the_analyst_risks(tmp_path):
    result = await run_example(stub(CRITIQUE_CLEAN), tmp_path)
    readiness = result.output("readiness")
    assert readiness["risk_count"] == 2
    assert readiness["penalties"]["risk"] == 26  # one high (18) + one medium (8)
    assert readiness["recommendation"] == "go"


async def test_the_report_lands_on_disk(tmp_path):
    result = await run_example(stub(CRITIQUE_CLEAN), tmp_path)

    published = result.output("publish")
    written = (tmp_path / published["path"].rsplit("/", 1)[-1]).read_text()
    assert "# Proposal — Northwind Logistics" in written
    assert "Draft body." in written
    assert published["bytes"] == len(written.encode("utf-8"))


async def test_every_step_reports_usage_and_the_run_totals_it(tmp_path):
    result = await run_example(stub(CRITIQUE_CLEAN), tmp_path)

    llm_steps = ["findings", "analysis", "proposal", "critique"]
    assert all(result.steps[name].usage.calls > 0 for name in llm_steps)
    assert result.steps["estimate"].usage.calls == 0  # deterministic, no model call
    assert result.usage.calls == sum(result.steps[n].usage.calls for n in result.steps)


async def test_a_failing_model_call_stops_the_run_and_skips_the_rest(tmp_path):
    client = StubClient(
        responses={"analyst": ANALYSIS},
        fail_for={"writer": RuntimeError("service overloaded")},
    )
    result = await run_example(client, tmp_path)

    assert result.status is Status.FAILED
    assert result.steps["proposal"].status is Status.FAILED
    assert "service overloaded" in result.steps["proposal"].error
    assert result.steps["publish"].status is Status.SKIPPED
    assert not list(tmp_path.iterdir())


async def test_an_invalid_brief_fails_fast_without_any_model_call(tmp_path):
    client = stub(CRITIQUE_CLEAN)
    workflow, registry = build()
    engine = Engine(workflow, registry=registry, client=client)
    result = await engine.run({"brief": {"client": "ACME"}})

    assert result.status is Status.FAILED
    assert "missing required field" in result.steps["brief"].error
    assert client.calls == []


async def test_the_engine_binds_its_client_to_unbound_agents(tmp_path):
    client = stub(CRITIQUE_CLEAN)
    workflow, registry = build()  # built without a client
    assert not any(getattr(a, "is_bound", False) for a in registry)

    engine = Engine(workflow, registry=registry, client=client)
    await engine.run({**EXAMPLE_INPUT, "output_dir": str(tmp_path)})
    assert client.calls


def test_cli_plan_prints_the_graph(capsys):
    assert main(["plan"]) == 0
    printed = capsys.readouterr().out
    assert "workflow: proposal-pipeline" in printed
    assert "foreach brief.research_questions" in printed


def test_cli_agents_lists_every_agent(capsys):
    assert main(["agents"]) == 0
    printed = capsys.readouterr().out
    for name in ("researcher", "analyst", "writer", "critic", "reviser"):
        assert name in printed


def test_cli_dry_run_completes_without_an_api_key(tmp_path, capsys):
    exit_code = main(
        ["run", "--example", "--dry-run", "--output-dir", str(tmp_path), "--json"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["steps"]["publish"]["status"] == "ok"
    assert list(tmp_path.iterdir())


def test_cli_reports_a_failed_run_with_a_nonzero_exit_code(tmp_path, capsys):
    exit_code = main(
        ["run", "--dry-run", "--input", json.dumps({"brief": {"client": "ACME"}})]
    )
    assert exit_code == 1
    assert "failed" in capsys.readouterr().out


def test_cli_rejects_a_bad_factory_target():
    import pytest

    with pytest.raises(SystemExit):
        main(["plan", "not-a-target"])
    with pytest.raises(SystemExit):
        main(["plan", "agentflow.workflows.ai_ml_ops:missing"])
