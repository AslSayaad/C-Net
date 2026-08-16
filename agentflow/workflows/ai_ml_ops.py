"""Example workflow: turn a client brief into a reviewed, priced proposal.

Shows every capability of the engine in one graph:

    normalize_brief          deterministic validation, derives the questions
        └─ research          LLM, fanned out one call per question
            └─ analysis      LLM, structured plan + risks
                ├─ estimate  deterministic arithmetic over the plan
                │   ├─ readiness   deterministic scoring (parallel with the draft)
                │   └─ proposal    LLM, Markdown draft
                │       └─ critique    LLM, structured review
                │           └─ revision    LLM, runs only if the critic asks
                │               └─ final       picks revision or draft
                │                   └─ publish  writes the file
"""

from __future__ import annotations

from typing import Any

from ..agent import agent
from ..agents.deterministic import (
    estimate_effort,
    normalize_brief,
    publish_report,
    score_readiness,
)
from ..agents.llm_agents import build_llm_agents
from ..blackboard import ref
from ..llm import ClaudeClient
from ..registry import AgentRegistry
from ..workflow import Step, Workflow


@agent(name="select_final")
def select_final(revised: str | None, original: str) -> str:
    """Prefer the revised draft; fall back to the original if no revision ran."""
    return revised if isinstance(revised, str) and revised.strip() else original


def needs_revision(state: dict[str, Any]) -> bool:
    """Gate for the revision step."""
    critique = state.get("critique") or {}
    return bool(critique.get("needs_revision"))


def build(client: ClaudeClient | None = None) -> tuple[Workflow, AgentRegistry]:
    """Build the workflow and the registry of agents it names."""
    registry = AgentRegistry(
        [normalize_brief, estimate_effort, score_readiness, publish_report, select_final]
    )
    registry.extend(build_llm_agents(client))

    workflow = Workflow(
        name="proposal-pipeline",
        description="Client brief -> research -> plan -> priced, reviewed proposal.",
        steps=[
            Step(
                name="brief",
                agent="normalize_brief",
                inputs={"brief": ref("input.brief")},
                description="Validate the brief and derive research questions.",
            ),
            Step(
                name="findings",
                agent="researcher",
                foreach=ref("brief.research_questions"),
                inputs={"brief": ref("brief")},
                max_parallel=3,
                retries=1,
                timeout=180,
                description="One model call per research question, in parallel.",
            ),
            Step(
                name="analysis",
                agent="analyst",
                inputs={"brief": ref("brief"), "findings": ref("findings")},
                retries=1,
                timeout=300,
            ),
            Step(
                name="estimate",
                agent="estimate_effort",
                inputs={
                    "phases": ref("analysis.phases"),
                    "day_rate_usd": ref("brief.day_rate_usd"),
                    "budget_usd": ref("brief.budget_usd"),
                },
            ),
            Step(
                name="readiness",
                agent="score_readiness",
                inputs={
                    "risks": ref("analysis.risks"),
                    "estimate": ref("estimate"),
                    "brief": ref("brief"),
                },
            ),
            Step(
                name="proposal",
                agent="writer",
                inputs={
                    "brief": ref("brief"),
                    "analysis": ref("analysis"),
                    "estimate": ref("estimate"),
                },
                retries=1,
                timeout=300,
            ),
            Step(
                name="critique",
                agent="critic",
                inputs={"brief": ref("brief"), "proposal": ref("proposal")},
                retries=1,
                timeout=300,
            ),
            Step(
                name="revision",
                agent="reviser",
                inputs={"proposal": ref("proposal"), "critique": ref("critique")},
                when=needs_revision,
                timeout=300,
                description="Runs only when the critic flags a high-severity issue.",
            ),
            Step(
                name="final",
                agent="select_final",
                # A defaulted ref is a soft dependency: `final` still waits for
                # `revision`, but runs with `None` when that branch was skipped.
                inputs={"revised": ref("revision", None), "original": ref("proposal")},
            ),
            Step(
                name="publish",
                agent="publish_report",
                inputs={
                    "proposal_markdown": ref("final"),
                    "brief": ref("brief"),
                    "estimate": ref("estimate"),
                    "readiness": ref("readiness"),
                    "output_dir": ref("input.output_dir", "artifacts"),
                },
            ),
        ],
    )
    return workflow, registry


EXAMPLE_INPUT: dict[str, Any] = {
    "brief": {
        "client": "Northwind Logistics",
        "objective": "forecast per-lane delivery delays 48 hours ahead",
        "domain": "supply chain",
        "constraints": [
            "must run on the client's existing Snowflake warehouse",
            "no PII may leave the client's VPC",
        ],
        "timeline_weeks": 10,
        "budget_usd": 180_000,
    }
}
