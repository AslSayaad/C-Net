"""Deterministic agents — plain Python, no model calls.

Anything that is arithmetic, validation, formatting, or I/O belongs here.
Keeping it out of the model makes it cheap, reproducible, and unit-testable.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

from ..agent import AgentContext, agent

DEFAULT_DAY_RATE_USD = 1_200
CONTINGENCY = 0.20
REQUIRED_BRIEF_FIELDS = ("client", "objective")


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "untitled"


@agent(name="normalize_brief")
def normalize_brief(brief: dict[str, Any]) -> dict[str, Any]:
    """Validate an incoming client brief and derive the research questions."""
    if not isinstance(brief, dict):
        raise TypeError(f"brief must be a dict, got {type(brief).__name__}")

    missing = [field for field in REQUIRED_BRIEF_FIELDS if not brief.get(field)]
    if missing:
        raise ValueError(f"brief is missing required field(s): {', '.join(missing)}")

    objective = str(brief["objective"]).strip()
    domain = str(brief.get("domain") or "general").strip()
    constraints = [str(c).strip() for c in brief.get("constraints", []) if str(c).strip()]

    normalized = {
        "client": str(brief["client"]).strip(),
        "objective": objective,
        "domain": domain,
        "constraints": constraints,
        "timeline_weeks": int(brief.get("timeline_weeks") or 0),
        "budget_usd": int(brief.get("budget_usd") or 0),
        "day_rate_usd": int(brief.get("day_rate_usd") or DEFAULT_DAY_RATE_USD),
        "slug": slugify(f"{brief['client']}-{objective[:40]}"),
        "research_questions": _research_questions(objective, domain, constraints),
    }
    return normalized


def _research_questions(objective: str, domain: str, constraints: list[str]) -> list[str]:
    questions = [
        f"What data and infrastructure does '{objective}' require in the {domain} domain, "
        "and what is typically already in place?",
        f"Which modelling approaches are the current standard for '{objective}', "
        "and what accuracy or latency do they realistically achieve?",
        f"What are the common failure modes and compliance obligations when deploying "
        f"'{objective}' in {domain}?",
    ]
    if constraints:
        questions.append(
            "How do these stated constraints change the approach: "
            + "; ".join(constraints)
        )
    return questions


@agent(name="estimate_effort")
def estimate_effort(phases: list[dict[str, Any]], day_rate_usd: int, budget_usd: int) -> dict[str, Any]:
    """Turn the analyst's phase breakdown into costs and a schedule.

    Arithmetic stays in code: the model proposes phases and day counts, this
    step decides what they cost.
    """
    if not phases:
        raise ValueError("no phases to estimate")

    priced = []
    for phase in phases:
        days = float(phase.get("engineer_days") or 0)
        if days <= 0:
            raise ValueError(f"phase {phase.get('name')!r} has no positive engineer_days")
        priced.append(
            {
                "name": phase.get("name", "unnamed phase"),
                "engineer_days": days,
                "cost_usd": round(days * day_rate_usd, 2),
            }
        )

    base_days = sum(p["engineer_days"] for p in priced)
    base_cost = round(sum(p["cost_usd"] for p in priced), 2)
    contingency_cost = round(base_cost * CONTINGENCY, 2)
    total_cost = round(base_cost + contingency_cost, 2)

    return {
        "phases": priced,
        "base_days": base_days,
        "base_cost_usd": base_cost,
        "contingency_pct": CONTINGENCY,
        "contingency_usd": contingency_cost,
        "total_cost_usd": total_cost,
        "estimated_weeks": round(base_days / 5.0, 1),
        "day_rate_usd": day_rate_usd,
        "within_budget": bool(budget_usd) and total_cost <= budget_usd,
        "budget_usd": budget_usd,
    }


@agent(name="score_readiness")
def score_readiness(
    risks: list[dict[str, Any]],
    estimate: dict[str, Any],
    brief: dict[str, Any],
) -> dict[str, Any]:
    """Weighted go/no-go score. Deterministic so two runs cannot disagree."""
    severity_weight = {"low": 3, "medium": 8, "high": 18}
    risk_penalty = sum(
        severity_weight.get(str(risk.get("severity", "medium")).lower(), 8)
        for risk in risks
    )

    budget_penalty = 0
    if brief.get("budget_usd") and not estimate.get("within_budget"):
        overage = estimate["total_cost_usd"] - brief["budget_usd"]
        budget_penalty = min(30, round(overage / max(brief["budget_usd"], 1) * 100))

    timeline_penalty = 0
    requested_weeks = brief.get("timeline_weeks") or 0
    if requested_weeks and estimate.get("estimated_weeks", 0) > requested_weeks:
        timeline_penalty = min(
            25, round((estimate["estimated_weeks"] - requested_weeks) * 4)
        )

    score = max(0, 100 - risk_penalty - budget_penalty - timeline_penalty)
    return {
        "score": score,
        "recommendation": "go" if score >= 60 else "revise" if score >= 35 else "no-go",
        "penalties": {
            "risk": risk_penalty,
            "budget": budget_penalty,
            "timeline": timeline_penalty,
        },
        "risk_count": len(risks),
    }


@agent(name="publish_report")
def publish_report(
    proposal_markdown: str,
    brief: dict[str, Any],
    estimate: dict[str, Any],
    readiness: dict[str, Any],
    output_dir: str = "artifacts",
    ctx: AgentContext | None = None,
) -> dict[str, Any]:
    """Assemble the final document and write it to disk."""
    header = [
        f"# Proposal — {brief['client']}",
        "",
        f"*{brief['objective']}*",
        "",
        f"- Generated: {date.today().isoformat()}",
        f"- Estimate: ${estimate['total_cost_usd']:,.0f} "
        f"({estimate['base_days']:.0f} engineer-days, ~{estimate['estimated_weeks']} weeks)",
        f"- Readiness: {readiness['score']}/100 → **{readiness['recommendation']}**",
        "",
        "---",
        "",
    ]
    document = "\n".join(header) + proposal_markdown.strip() + "\n"

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    run_suffix = (ctx.run_id.split("_")[-1] if ctx else "manual")
    path = directory / f"{brief['slug']}-{run_suffix}.md"
    path.write_text(document, encoding="utf-8")

    return {"path": str(path), "bytes": len(document.encode("utf-8"))}
