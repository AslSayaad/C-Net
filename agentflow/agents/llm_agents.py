"""Claude-backed agents for the example AI/ML business-ops workflow.

Each agent owns one role. System prompts are static (so they cache well and so
behaviour is easy to reason about); everything run-specific arrives in the user
prompt. Steps whose output feeds arithmetic or control flow use structured
outputs rather than prose.
"""

from __future__ import annotations

from typing import Any

from ..agent import AgentContext
from ..llm import ClaudeClient
from ..llm_agent import LLMAgent, array_of, schema

HOUSE_STYLE = (
    "Be concrete and specific. Prefer numbers, named technologies, and stated "
    "assumptions over hedged generalities. If something cannot be known from "
    "the information given, say so plainly and state what you assumed."
)

FINDING_SCHEMA = schema(
    {
        "question": {"type": "string"},
        "summary": {"type": "string", "description": "Two or three sentences."},
        "key_points": array_of({"type": "string"}, "Three to six specific findings."),
        "assumptions": array_of({"type": "string"}),
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    }
)

ANALYSIS_SCHEMA = schema(
    {
        "approach": {"type": "string", "description": "The recommended technical approach."},
        "phases": array_of(
            schema(
                {
                    "name": {"type": "string"},
                    "goal": {"type": "string"},
                    "engineer_days": {
                        "type": "number",
                        "description": "Whole or half days of one engineer's effort.",
                    },
                }
            ),
            "Three to six delivery phases, in order.",
        ),
        "risks": array_of(
            schema(
                {
                    "risk": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "mitigation": {"type": "string"},
                }
            )
        ),
        "data_requirements": array_of({"type": "string"}),
        "open_questions": array_of({"type": "string"}),
    }
)

CRITIQUE_SCHEMA = schema(
    {
        "verdict": {"type": "string", "enum": ["ship", "revise", "reject"]},
        "needs_revision": {"type": "boolean"},
        "issues": array_of(
            schema(
                {
                    "issue": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "fix": {"type": "string", "description": "The specific change to make."},
                }
            )
        ),
        "strengths": array_of({"type": "string"}),
    }
)


def _research_prompt(ctx: AgentContext) -> str:
    brief = ctx.require("brief")
    return (
        f"Client: {brief['client']}\n"
        f"Domain: {brief['domain']}\n"
        f"Objective: {brief['objective']}\n"
        f"Constraints: {'; '.join(brief['constraints']) or 'none stated'}\n\n"
        f"Research question:\n{ctx.item}\n\n"
        "Answer only this question."
    )


def _analysis_prompt(ctx: AgentContext) -> str:
    brief = ctx.require("brief")
    findings = ctx.require("findings")
    rendered = "\n\n".join(
        f"Q: {f.get('question', '?')}\n"
        f"Summary: {f.get('summary', '')}\n"
        f"Points: {'; '.join(f.get('key_points', []))}\n"
        f"Confidence: {f.get('confidence', 'unknown')}"
        for f in findings
    )
    return (
        f"Objective: {brief['objective']} (client: {brief['client']}, "
        f"domain: {brief['domain']})\n"
        f"Stated constraints: {'; '.join(brief['constraints']) or 'none'}\n"
        f"Requested timeline: {brief['timeline_weeks'] or 'unspecified'} weeks\n\n"
        f"Research findings:\n{rendered}\n\n"
        "Produce the delivery plan. Size each phase in engineer-days; do not "
        "compute costs or totals — a downstream step does that."
    )


def _proposal_prompt(ctx: AgentContext) -> str:
    brief = ctx.require("brief")
    analysis = ctx.require("analysis")
    estimate = ctx.require("estimate")
    phase_lines = "\n".join(
        f"- {p['name']}: {p['engineer_days']} engineer-days, ${p['cost_usd']:,.0f}"
        for p in estimate["phases"]
    )
    risk_lines = "\n".join(
        f"- [{r['severity']}] {r['risk']} — mitigation: {r['mitigation']}"
        for r in analysis["risks"]
    )
    return (
        f"Write the proposal body for {brief['client']}.\n\n"
        f"Objective: {brief['objective']}\n"
        f"Recommended approach: {analysis['approach']}\n\n"
        f"Priced phases:\n{phase_lines}\n\n"
        f"Total with {estimate['contingency_pct']:.0%} contingency: "
        f"${estimate['total_cost_usd']:,.0f} over ~{estimate['estimated_weeks']} weeks\n\n"
        f"Risks:\n{risk_lines}\n\n"
        f"Data requirements: {'; '.join(analysis['data_requirements'])}\n"
        f"Open questions: {'; '.join(analysis['open_questions']) or 'none'}\n\n"
        "Use these numbers exactly as given; do not recompute or round them."
    )


def _critique_prompt(ctx: AgentContext) -> str:
    return (
        f"Client brief: {ctx.require('brief')['objective']}\n\n"
        f"Proposal under review:\n\n{ctx.require('proposal')}"
    )


def _revision_prompt(ctx: AgentContext) -> str:
    critique = ctx.require("critique")
    issues = "\n".join(
        f"- [{i['severity']}] {i['issue']} → {i['fix']}" for i in critique["issues"]
    )
    return (
        f"Revise this proposal. Address every issue listed; change nothing else, "
        f"and keep all figures exactly as they are.\n\n"
        f"Issues:\n{issues}\n\n"
        f"Proposal:\n\n{ctx.require('proposal')}"
    )


def build_llm_agents(client: ClaudeClient | None = None) -> list[LLMAgent]:
    """Construct the five model-backed agents, optionally bound to a client."""
    agents = [
        LLMAgent(
            name="researcher",
            description="Answers one scoped research question about the engagement.",
            system=(
                "You are a research analyst at an AI/ML consultancy. You answer one "
                "scoped question at a time about the feasibility and shape of a client "
                "engagement. " + HOUSE_STYLE
            ),
            prompt=_research_prompt,
            output_schema=FINDING_SCHEMA,
            effort="medium",
            max_tokens=4_000,
        ),
        LLMAgent(
            name="analyst",
            description="Turns research into a phased delivery plan with risks.",
            system=(
                "You are a delivery lead at an AI/ML consultancy. From research "
                "findings you produce a phased plan, sized in engineer-days, with "
                "risks and their mitigations. You size work honestly, including data "
                "work and evaluation, and you never invent capabilities the findings "
                "do not support. " + HOUSE_STYLE
            ),
            prompt=_analysis_prompt,
            output_schema=ANALYSIS_SCHEMA,
            effort="high",
            max_tokens=8_000,
        ),
        LLMAgent(
            name="writer",
            description="Drafts the client-facing proposal in Markdown.",
            system=(
                "You write client-facing proposals for an AI/ML consultancy. Output "
                "Markdown only, starting at heading level 2. Sections: Understanding, "
                "Approach, Delivery plan, Risks and mitigations, Commercials, Next "
                "steps. Never alter a number you were given. " + HOUSE_STYLE
            ),
            prompt=_proposal_prompt,
            effort="high",
            max_tokens=8_000,
        ),
        LLMAgent(
            name="critic",
            description="Reviews the draft for accuracy, gaps, and overclaiming.",
            system=(
                "You review draft consultancy proposals before they reach a client. "
                "Look for overclaiming, vague deliverables, unstated assumptions, "
                "figures that contradict the plan, and missing risks. Report every "
                "issue you find with the specific fix; a separate step decides what "
                "to act on. Set needs_revision true when any high-severity issue "
                "exists. " + HOUSE_STYLE
            ),
            prompt=_critique_prompt,
            output_schema=CRITIQUE_SCHEMA,
            effort="high",
            max_tokens=6_000,
        ),
        LLMAgent(
            name="reviser",
            description="Applies the critic's fixes to the draft.",
            system=(
                "You revise consultancy proposals. Apply exactly the fixes you are "
                "given and leave everything else — including all figures and section "
                "structure — untouched. Output the full revised Markdown, nothing "
                "else. " + HOUSE_STYLE
            ),
            prompt=_revision_prompt,
            effort="medium",
            max_tokens=8_000,
        ),
    ]
    if client is not None:
        for llm_agent in agents:
            llm_agent.bind(client)
    return agents


def agent_schemas() -> dict[str, dict[str, Any]]:
    """The structured-output schemas, exposed for tests and documentation."""
    return {
        "researcher": FINDING_SCHEMA,
        "analyst": ANALYSIS_SCHEMA,
        "critic": CRITIQUE_SCHEMA,
    }
