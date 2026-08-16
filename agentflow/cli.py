"""Command line entry point: `python -m agentflow ...`."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
from typing import Any

from .engine import Engine
from .events import ConsoleReporter
from .llm import default_client
from .registry import AgentRegistry
from .testing import StubClient
from .types import Status
from .workflow import Workflow

DEFAULT_FACTORY = "agentflow.workflows.ai_ml_ops:build"


def load_factory(target: str) -> tuple[Workflow, AgentRegistry]:
    """Import `module:callable` and call it to get (workflow, registry)."""
    if ":" not in target:
        raise SystemExit(f"expected 'module:callable', got {target!r}")
    module_name, attribute = target.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(f"cannot import {module_name!r}: {exc}") from exc
    try:
        factory = getattr(module, attribute)
    except AttributeError:
        raise SystemExit(f"{module_name!r} has no attribute {attribute!r}") from None

    built = factory()
    if isinstance(built, Workflow):
        return built, AgentRegistry()
    workflow, registry = built
    return workflow, registry


def load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    if args.example:
        module_name = args.workflow.split(":", 1)[0]
        module = importlib.import_module(module_name)
        example = getattr(module, "EXAMPLE_INPUT", None)
        if example is None:
            raise SystemExit(f"{module_name} defines no EXAMPLE_INPUT")
        payload = dict(example)
    elif args.input_file:
        payload = json.loads(open(args.input_file, encoding="utf-8").read())
    elif args.input:
        payload = json.loads(args.input)
    else:
        payload = {}
    if getattr(args, "output_dir", None):
        payload["output_dir"] = args.output_dir
    return payload


def cmd_plan(args: argparse.Namespace) -> int:
    workflow, registry = load_factory(args.workflow)
    print(workflow.describe())
    print(f"\nagents: {', '.join(registry.names())}")
    return 0


def cmd_agents(args: argparse.Namespace) -> int:
    _, registry = load_factory(args.workflow)
    for item in sorted(registry, key=lambda a: a.name):
        print(f"{item.name:<20} {type(item).__name__:<14} {item.description}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    workflow, registry = load_factory(args.workflow)
    inputs = load_inputs(args)
    client = StubClient() if args.dry_run else default_client()
    if args.dry_run:
        print("dry run: model calls are stubbed, no API requests are made\n", file=sys.stderr)

    engine = Engine(
        workflow,
        registry=registry,
        client=client,
        max_concurrency=args.concurrency,
        listeners=[ConsoleReporter(verbose=args.verbose)],
        fail_fast=not args.keep_going,
    )
    result = asyncio.run(engine.run(inputs))

    if args.json:
        print(json.dumps(result.as_dict(), indent=2, default=str))
    else:
        print_summary(result, args.show)
    return 0 if result.status is Status.OK else 1


def print_summary(result: Any, show: str | None) -> None:
    usage = result.usage
    print(f"\nrun {result.run_id} -> {result.status.value} in {result.duration:.1f}s")
    print(
        f"model calls: {usage.calls} | "
        f"input tokens: {usage.total_input_tokens:,} "
        f"(cache reads {usage.cache_read_input_tokens:,}) | "
        f"output tokens: {usage.output_tokens:,}"
    )
    for name, step in result.steps.items():
        marker = {"ok": "✓", "failed": "✗", "skipped": "–"}[step.status.value]
        detail = f" — {step.error}" if step.error else ""
        print(f"  {marker} {name:<12} {step.duration:6.2f}s{detail}")
    if show:
        print(f"\n--- {show} ---")
        value = result.output(show)
        print(value if isinstance(value, str) else json.dumps(value, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentflow", description="Run multi-agent workflows."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(target: argparse.ArgumentParser) -> None:
        target.add_argument(
            "workflow",
            nargs="?",
            default=DEFAULT_FACTORY,
            help=f"module:callable returning (Workflow, AgentRegistry). "
            f"Default: {DEFAULT_FACTORY}",
        )

    plan = sub.add_parser("plan", help="print the execution plan without running")
    add_common(plan)
    plan.set_defaults(func=cmd_plan)

    agents = sub.add_parser("agents", help="list the agents a workflow uses")
    add_common(agents)
    agents.set_defaults(func=cmd_agents)

    run = sub.add_parser("run", help="run a workflow")
    add_common(run)
    run.add_argument("--input", help="inputs as a JSON string")
    run.add_argument("--input-file", help="inputs as a JSON file")
    run.add_argument("--example", action="store_true", help="use the module's EXAMPLE_INPUT")
    run.add_argument("--output-dir", help="where generated artifacts are written")
    run.add_argument("--dry-run", action="store_true", help="stub every model call")
    run.add_argument("--concurrency", type=int, default=4, help="max steps in flight")
    run.add_argument("--keep-going", action="store_true", help="do not stop on first failure")
    run.add_argument("--json", action="store_true", help="print the full result as JSON")
    run.add_argument("--show", help="print one step's output at the end")
    run.add_argument("-v", "--verbose", action="store_true", help="log each model call")
    run.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
