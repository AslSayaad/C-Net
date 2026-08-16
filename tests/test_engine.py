"""Engine scheduling, error handling, and instrumentation."""

import asyncio

import pytest

from agentflow import (
    Agent,
    AgentContext,
    AgentRegistry,
    Engine,
    Status,
    Step,
    Workflow,
    agent,
    ref,
    run_workflow,
)
from agentflow.types import Usage


@agent(name="echo")
def echo(value):
    """Return whatever it is given."""
    return value


@agent(name="double")
def double(value):
    return value * 2


class Boom(Agent):
    """Fails a fixed number of times, then succeeds."""

    name = "boom"

    def __init__(self, failures: int = 99) -> None:
        super().__init__()
        self.failures = failures
        self.calls = 0

    async def run(self, ctx: AgentContext):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError(f"boom {self.calls}")
        return "recovered"


class Tracker(Agent):
    """Records concurrency so tests can assert on parallelism."""

    name = "tracker"

    def __init__(self, delay: float = 0.02) -> None:
        super().__init__()
        self.delay = delay
        self.in_flight = 0
        self.peak = 0
        self.seen: list = []

    async def run(self, ctx: AgentContext):
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
            self.seen.append(ctx.item if ctx.index is not None else ctx.step)
            return ctx.item if ctx.index is not None else ctx.step
        finally:
            self.in_flight -= 1


async def test_outputs_flow_between_steps():
    workflow = Workflow(
        name="chain",
        steps=[
            Step(name="first", agent=echo, inputs={"value": ref("input.n")}),
            Step(name="second", agent=double, inputs={"value": ref("first")}),
        ],
    )
    result = await run_workflow(workflow, {"n": 21})
    assert result.status is Status.OK
    assert result.output("second") == 42
    assert result.state["second"] == 42


async def test_independent_steps_run_concurrently():
    tracker = Tracker(delay=0.05)
    workflow = Workflow(
        name="parallel",
        steps=[Step(name=f"s{i}", agent=tracker) for i in range(4)],
    )
    result = await run_workflow(workflow, max_concurrency=4)
    assert result.status is Status.OK
    assert tracker.peak == 4


async def test_max_concurrency_is_respected():
    tracker = Tracker(delay=0.02)
    workflow = Workflow(
        name="throttled",
        steps=[Step(name=f"s{i}", agent=tracker) for i in range(6)],
    )
    await run_workflow(workflow, max_concurrency=2)
    assert tracker.peak <= 2


async def test_failure_skips_dependents_and_fails_the_run():
    workflow = Workflow(
        name="failing",
        steps=[
            Step(name="bad", agent=Boom()),
            Step(name="after", agent=echo, inputs={"value": ref("bad")}),
        ],
    )
    result = await run_workflow(workflow)
    assert result.status is Status.FAILED
    assert result.steps["bad"].status is Status.FAILED
    assert "boom" in result.steps["bad"].error
    assert result.steps["after"].status is Status.SKIPPED
    assert "upstream bad" in result.steps["after"].error


async def test_retries_recover_before_giving_up():
    flaky = Boom(failures=2)
    workflow = Workflow(
        name="retrying",
        steps=[Step(name="flaky", agent=flaky, retries=3, retry_backoff=0.001)],
    )
    result = await run_workflow(workflow)
    assert result.status is Status.OK
    assert result.output("flaky") == "recovered"
    assert result.steps["flaky"].attempts == 3
    assert flaky.calls == 3


async def test_retries_are_bounded():
    always = Boom(failures=99)
    workflow = Workflow(
        name="retry-exhausted",
        steps=[Step(name="flaky", agent=always, retries=2, retry_backoff=0.001)],
    )
    result = await run_workflow(workflow)
    assert result.status is Status.FAILED
    assert always.calls == 3


async def test_optional_step_failure_does_not_fail_the_run():
    workflow = Workflow(
        name="optional",
        steps=[
            Step(name="nice_to_have", agent=Boom(), optional=True),
            Step(name="core", agent=echo, inputs={"value": 1}),
        ],
    )
    result = await run_workflow(workflow)
    assert result.status is Status.OK
    assert result.steps["nice_to_have"].status is Status.FAILED
    assert result.steps["core"].status is Status.OK


async def test_condition_skips_a_step():
    workflow = Workflow(
        name="conditional",
        steps=[
            Step(name="gate", agent=echo, inputs={"value": False}),
            Step(
                name="guarded",
                agent=echo,
                inputs={"value": 1},
                depends_on=["gate"],
                when=lambda state: state["gate"] is True,
            ),
        ],
    )
    result = await run_workflow(workflow)
    assert result.steps["guarded"].status is Status.SKIPPED
    assert result.steps["guarded"].error == "condition not met"
    assert result.status is Status.OK


async def test_soft_dependency_runs_with_default_when_producer_skipped():
    workflow = Workflow(
        name="soft",
        steps=[
            Step(
                name="maybe",
                agent=echo,
                inputs={"value": "revised"},
                when=lambda state: False,
            ),
            Step(
                name="pick",
                agent=echo,
                inputs={"value": ref("maybe", "original")},
            ),
        ],
    )
    result = await run_workflow(workflow)
    assert result.steps["maybe"].status is Status.SKIPPED
    assert result.output("pick") == "original"


async def test_soft_dependency_uses_the_real_value_when_present():
    workflow = Workflow(
        name="soft-present",
        steps=[
            Step(name="maybe", agent=echo, inputs={"value": "revised"}),
            Step(name="pick", agent=echo, inputs={"value": ref("maybe", "original")}),
        ],
    )
    result = await run_workflow(workflow)
    assert result.output("pick") == "revised"


async def test_foreach_fans_out_and_collects_a_list():
    tracker = Tracker(delay=0.01)
    workflow = Workflow(
        name="fanout",
        steps=[
            Step(name="items", agent=echo, inputs={"value": ["a", "b", "c"]}),
            Step(name="each", agent=tracker, foreach=ref("items"), max_parallel=3),
        ],
    )
    result = await run_workflow(workflow)
    assert result.output("each") == ["a", "b", "c"]
    assert tracker.peak == 3


async def test_foreach_respects_max_parallel():
    tracker = Tracker(delay=0.02)
    workflow = Workflow(
        name="fanout-throttled",
        steps=[
            Step(name="items", agent=echo, inputs={"value": list(range(6))}),
            Step(name="each", agent=tracker, foreach=ref("items"), max_parallel=2),
        ],
    )
    await run_workflow(workflow, max_concurrency=8)
    assert tracker.peak <= 2


async def test_foreach_over_a_non_list_fails_the_step():
    workflow = Workflow(
        name="bad-fanout",
        steps=[
            Step(name="items", agent=echo, inputs={"value": "not-a-list"}),
            Step(name="each", agent=echo, foreach=ref("items")),
        ],
    )
    result = await run_workflow(workflow)
    assert result.steps["each"].status is Status.FAILED
    assert "expected a list" in result.steps["each"].error


async def test_timeout_fails_the_step():
    workflow = Workflow(
        name="slow",
        steps=[Step(name="slow", agent=Tracker(delay=0.5), timeout=0.05)],
    )
    result = await run_workflow(workflow)
    assert result.steps["slow"].status is Status.FAILED
    assert "TimeoutError" in result.steps["slow"].error


async def test_keep_going_runs_independent_branches_after_a_failure():
    workflow = Workflow(
        name="keep-going",
        steps=[
            Step(name="bad", agent=Boom()),
            Step(name="independent", agent=echo, inputs={"value": "done"}),
        ],
    )
    result = await run_workflow(workflow, fail_fast=False)
    assert result.status is Status.FAILED
    assert result.output("independent") == "done"


async def test_fail_fast_marks_unstarted_steps_as_skipped():
    slow = Tracker(delay=0.2)
    workflow = Workflow(
        name="abort",
        steps=[
            Step(name="bad", agent=Boom()),
            Step(name="slow", agent=slow),
            Step(name="later", agent=echo, inputs={"value": ref("slow")}),
        ],
    )
    result = await run_workflow(workflow, max_concurrency=4)
    assert result.status is Status.FAILED
    assert result.steps["later"].status is Status.SKIPPED
    assert set(result.steps) == {"bad", "slow", "later"}


async def test_agents_are_resolved_by_name_from_the_registry():
    registry = AgentRegistry([echo])
    workflow = Workflow(
        name="by-name",
        steps=[Step(name="s", agent="echo", inputs={"value": 5})],
    )
    result = await run_workflow(workflow, registry=registry)
    assert result.output("s") == 5


async def test_unknown_agent_name_fails_the_step_with_a_clear_message():
    workflow = Workflow(name="missing", steps=[Step(name="s", agent="ghost")])
    result = await run_workflow(workflow, registry=AgentRegistry([echo]))
    assert result.steps["s"].status is Status.FAILED
    assert "no agent named 'ghost'" in result.steps["s"].error


async def test_usage_is_aggregated_across_steps():
    class Spender(Agent):
        name = "spender"

        async def run(self, ctx: AgentContext):
            ctx.record_usage(Usage(input_tokens=10, output_tokens=5, calls=1))
            return "ok"

    workflow = Workflow(
        name="usage",
        steps=[Step(name=f"s{i}", agent=Spender()) for i in range(3)],
    )
    result = await run_workflow(workflow)
    assert result.usage.calls == 3
    assert result.usage.input_tokens == 30
    assert result.usage.output_tokens == 15


async def test_events_describe_the_run():
    seen = []
    workflow = Workflow(
        name="events",
        steps=[
            Step(name="ok", agent=echo, inputs={"value": 1}),
            Step(name="bad", agent=Boom(), optional=True),
        ],
    )
    engine = Engine(workflow, listeners=[seen.append], fail_fast=False)
    await engine.run()
    kinds = [event.kind for event in seen]
    assert kinds[0] == "run.start"
    assert kinds[-1] == "run.end"
    assert "step.ok" in kinds
    assert "step.failed" in kinds


async def test_a_broken_listener_cannot_break_the_run():
    def explode(event):
        raise RuntimeError("listener is broken")

    workflow = Workflow(name="robust", steps=[Step(name="s", agent=echo, inputs={"value": 1})])
    engine = Engine(workflow, listeners=[explode])
    result = await engine.run()
    assert result.status is Status.OK


async def test_context_exposes_inputs_and_state():
    class Inspector(Agent):
        name = "inspector"

        async def run(self, ctx: AgentContext):
            return {
                "declared": ctx.require("value"),
                "from_state": ctx.state["input"]["seed"],
                "run_id": ctx.run_id,
            }

    workflow = Workflow(
        name="ctx",
        steps=[Step(name="s", agent=Inspector(), inputs={"value": ref("input.seed")})],
    )
    result = await run_workflow(workflow, {"seed": 7})
    assert result.output("s")["declared"] == 7
    assert result.output("s")["from_state"] == 7
    assert result.output("s")["run_id"] == result.run_id


async def test_missing_required_input_fails_the_step():
    class Needy(Agent):
        name = "needy"

        async def run(self, ctx: AgentContext):
            return ctx.require("absent")

    workflow = Workflow(name="needy", steps=[Step(name="s", agent=Needy())])
    result = await run_workflow(workflow)
    assert result.steps["s"].status is Status.FAILED
    assert "requires input 'absent'" in result.steps["s"].error


async def test_unresolvable_ref_fails_only_that_step():
    workflow = Workflow(
        name="badref",
        steps=[
            Step(name="a", agent=echo, inputs={"value": {}}),
            Step(name="b", agent=echo, inputs={"value": ref("a.nope")}),
        ],
    )
    result = await run_workflow(workflow, fail_fast=False)
    assert result.steps["a"].status is Status.OK
    assert result.steps["b"].status is Status.FAILED
    assert "cannot resolve" in result.steps["b"].error
