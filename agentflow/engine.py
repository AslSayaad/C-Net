"""The workflow engine: schedules steps over their dependency graph."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from .agent import Agent, AgentContext
from .blackboard import Blackboard
from .events import EventBus, Listener
from .llm import ClaudeClient
from .llm_agent import LLMAgent
from .registry import AgentRegistry
from .types import RunResult, Status, StepResult, Usage, new_run_id
from .workflow import Step, Workflow

log = logging.getLogger("agentflow.engine")

_NOT_RUN = StepResult(step="", agent="", status=Status.SKIPPED)


class Engine:
    """Runs a `Workflow`.

    Steps run as soon as their dependencies finish, up to `max_concurrency` at
    once. A step whose dependencies did not all succeed is skipped rather than
    run with missing inputs.
    """

    def __init__(
        self,
        workflow: Workflow,
        registry: AgentRegistry | None = None,
        client: ClaudeClient | None = None,
        max_concurrency: int = 4,
        listeners: list[Listener] | None = None,
        fail_fast: bool = True,
    ) -> None:
        self.workflow = workflow
        self.registry = registry or AgentRegistry()
        self.client = client
        self.max_concurrency = max_concurrency
        self.fail_fast = fail_fast
        self.events = EventBus(listeners)

    def resolve_agent(self, step: Step) -> Agent:
        agent = step.agent if isinstance(step.agent, Agent) else self.registry.get(step.agent)
        if isinstance(agent, LLMAgent) and not agent.is_bound and self.client is not None:
            agent.bind(self.client)
        return agent

    async def run(
        self, inputs: dict[str, Any] | None = None, run_id: str | None = None
    ) -> RunResult:
        run_id = run_id or new_run_id()
        blackboard = Blackboard({"input": dict(inputs or {}), "run_id": run_id})
        result = RunResult(
            run_id=run_id,
            workflow=self.workflow.name,
            status=Status.OK,
            started_at=time.time(),
        )
        self.events.emit("run.start", run_id, workflow=self.workflow.name)

        graph = self.workflow.graph()
        hard_graph = self.workflow.hard_graph()
        pending = set(graph)
        running: dict[asyncio.Task[StepResult], str] = {}
        semaphore = asyncio.Semaphore(self.max_concurrency)
        aborted = False

        try:
            while pending or running:
                if not aborted:
                    # Skipping a step can make its dependents ready, so keep
                    # dispatching until a pass produces nothing new.
                    dispatched = True
                    while dispatched:
                        dispatched = False
                        for name in self._ready(pending, graph, result.steps):
                            pending.discard(name)
                            dispatched = True
                            step = self.workflow.step(name)
                            skip_reason = self._skip_reason(
                                step, hard_graph[name], result, blackboard
                            )
                            if skip_reason:
                                result.steps[name] = self._skipped(step, skip_reason)
                                self.events.emit(
                                    "step.skipped", run_id, name, reason=skip_reason
                                )
                                continue
                            task = asyncio.create_task(
                                self._run_step(step, run_id, blackboard, semaphore),
                                name=name,
                            )
                            running[task] = name

                if not running:
                    break

                done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    name = running.pop(task)
                    step_result = await self._collect(task, name)
                    result.steps[name] = step_result
                    if step_result.status is Status.OK:
                        await blackboard.set(name, step_result.output)
                        self.events.emit(
                            "step.ok", run_id, name, duration=step_result.duration
                        )
                    elif step_result.status is Status.FAILED:
                        self.events.emit(
                            "step.failed",
                            run_id,
                            name,
                            error=step_result.error,
                            duration=step_result.duration,
                        )
                        if self.fail_fast and not self.workflow.step(name).optional:
                            aborted = True
                    else:  # cancelled mid-flight while the run was aborting
                        self.events.emit(
                            "step.skipped", run_id, name, reason=step_result.error
                        )

                if aborted and running:
                    for task in running:
                        task.cancel()
                    await asyncio.gather(*running, return_exceptions=True)
                    for name in running.values():
                        result.steps.setdefault(
                            name, self._skipped(self.workflow.step(name), "cancelled")
                        )
                        self.events.emit("step.skipped", run_id, name, reason="cancelled")
                    running.clear()
        finally:
            for task in running:
                task.cancel()
            if running:
                await asyncio.gather(*running, return_exceptions=True)

        for name in sorted(pending, key=lambda n: len(graph[n])):
            step = self.workflow.step(name)
            reason = self._blocked_reason(hard_graph[name], result) or "run aborted"
            result.steps[name] = self._skipped(step, reason)
            self.events.emit("step.skipped", run_id, name, reason=reason)

        result.status = self._run_status(result)
        result.state = blackboard.snapshot()
        result.ended_at = time.time()
        self.events.emit(
            "run.end",
            run_id,
            workflow=self.workflow.name,
            status=result.status.value,
            duration=result.duration,
        )
        return result

    def _ready(
        self, pending: set[str], graph: dict[str, set[str]], done: dict[str, StepResult]
    ) -> list[str]:
        return sorted(name for name in pending if graph[name] <= set(done))

    def _skip_reason(
        self,
        step: Step,
        deps: set[str],
        result: RunResult,
        blackboard: Blackboard,
    ) -> str | None:
        blocked = self._blocked_reason(deps, result)
        if blocked:
            return blocked
        if step.when is not None and not step.when(blackboard.snapshot()):
            return "condition not met"
        return None

    def _blocked_reason(self, deps: set[str], result: RunResult) -> str | None:
        """Name the dependencies that stop a step from running, if any."""
        blocked = sorted(
            dep
            for dep in deps
            if result.steps.get(dep, _NOT_RUN).status is not Status.OK
        )
        if blocked:
            return f"upstream {', '.join(blocked)} did not succeed"
        return None

    def _skipped(self, step: Step, reason: str) -> StepResult:
        now = time.time()
        return StepResult(
            step=step.name,
            agent=step.agent_name,
            status=Status.SKIPPED,
            error=reason,
            started_at=now,
            ended_at=now,
        )

    async def _collect(self, task: asyncio.Task[StepResult], name: str) -> StepResult:
        try:
            return await task
        except asyncio.CancelledError:
            step = self.workflow.step(name)
            return self._skipped(step, "cancelled")

    async def _run_step(
        self,
        step: Step,
        run_id: str,
        blackboard: Blackboard,
        semaphore: asyncio.Semaphore,
    ) -> StepResult:
        started = time.time()
        record = StepResult(
            step=step.name,
            agent=step.agent_name,
            status=Status.OK,
            started_at=started,
        )
        try:
            # Resolving the agent and the inputs can both fail; a bad step
            # definition should fail its own step, not the whole run.
            agent = self.resolve_agent(step)
            record.agent = agent.name
            self.events.emit("step.start", run_id, step.name, agent=agent.name)
            inputs = blackboard.resolve(step.inputs)
            if step.foreach is not None:
                output, attempts, usage = await self._run_foreach(
                    step, agent, run_id, blackboard, inputs, semaphore
                )
            else:
                async with semaphore:
                    ctx = AgentContext(
                        run_id=run_id,
                        step=step.name,
                        inputs=inputs,
                        blackboard=blackboard,
                        events=self.events,
                    )
                    output = await self._attempt(step, agent, ctx, run_id)
                    attempts, usage = ctx.attempts, ctx.usage
            record.output = output
            record.attempts = attempts
            record.usage = usage
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # a failing agent fails its step, not the process
            log.debug("step %s failed", step.name, exc_info=True)
            record.status = Status.FAILED
            record.error = f"{type(exc).__name__}: {exc}"
        record.ended_at = time.time()
        return record

    async def _run_foreach(
        self,
        step: Step,
        agent: Agent,
        run_id: str,
        blackboard: Blackboard,
        inputs: dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> tuple[list[Any], int, Usage]:
        items = blackboard.resolve(step.foreach)
        if not isinstance(items, (list, tuple)):
            raise TypeError(
                f"foreach on step {step.name!r} resolved to "
                f"{type(items).__name__}, expected a list"
            )
        fan_out = asyncio.Semaphore(max(1, step.max_parallel))
        contexts = [
            AgentContext(
                run_id=run_id,
                step=step.name,
                inputs=inputs,
                blackboard=blackboard,
                events=self.events,
                item=item,
                index=index,
            )
            for index, item in enumerate(items)
        ]

        async def run_one(ctx: AgentContext) -> Any:
            async with fan_out, semaphore:
                return await self._attempt(step, agent, ctx, run_id)

        outputs = await asyncio.gather(*(run_one(ctx) for ctx in contexts))
        usage = Usage()
        attempts = 0
        for ctx in contexts:
            usage = usage + ctx.usage
            attempts += ctx.attempts
        return list(outputs), attempts, usage

    async def _attempt(
        self, step: Step, agent: Agent, ctx: AgentContext, run_id: str
    ) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, step.retries + 2):
            ctx.attempts = attempt
            try:
                call = agent.run(ctx)
                if step.timeout:
                    return await asyncio.wait_for(call, timeout=step.timeout)
                return await call
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt > step.retries:
                    break
                delay = step.retry_backoff * (2 ** (attempt - 1))
                self.events.emit(
                    "step.retry",
                    run_id,
                    step.name,
                    attempt=attempt,
                    error=f"{type(exc).__name__}: {exc}",
                )
                await asyncio.sleep(delay)
        assert last_error is not None
        raise last_error

    def _run_status(self, result: RunResult) -> Status:
        for name, step_result in result.steps.items():
            if step_result.status is Status.FAILED and not self.workflow.step(name).optional:
                return Status.FAILED
        return Status.OK


async def run_workflow(
    workflow: Workflow,
    inputs: dict[str, Any] | None = None,
    registry: AgentRegistry | None = None,
    client: ClaudeClient | None = None,
    **engine_kwargs: Any,
) -> RunResult:
    """Convenience wrapper: build an `Engine` and run it once."""
    engine = Engine(workflow, registry=registry, client=client, **engine_kwargs)
    return await engine.run(inputs)
