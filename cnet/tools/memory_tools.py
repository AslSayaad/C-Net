"""Tools that let C-Net keep notes about you between sessions."""

from __future__ import annotations

from ..config import Config
from ..memory import FactStore
from .base import ToolRegistry


def register(registry: ToolRegistry, cfg: Config, facts: FactStore) -> None:
    @registry.add(
        "remember",
        "Save a durable fact about the user or their setup so you still know it "
        "in future conversations. Use it for preferences, names, projects and "
        "standing instructions — not for passing details of the current chat. "
        "Write the fact as one clear self-contained sentence.",
        {
            "type": "object",
            "properties": {
                "fact": {
                    "type": "string",
                    "description": "The fact to store, e.g. 'Prefers Python over JavaScript.'",
                }
            },
            "required": ["fact"],
        },
    )
    def remember(fact: str) -> str:
        return facts.remember(fact)

    @registry.add(
        "recall",
        "Search your saved long-term facts. Leave the query empty to list "
        "everything you remember.",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Optional word to filter facts by. Empty means list all.",
                }
            },
        },
    )
    def recall(query: str = "") -> str:
        found = facts.recall(query)
        if not found:
            return "Nothing remembered yet." if not query else f"No facts matching '{query}'."
        return "\n".join(f"- {fact}" for fact in found)

    @registry.add(
        "forget",
        "Delete saved facts that match the given text. Use when the user says "
        "something you stored is wrong or out of date.",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Text to match against stored facts.",
                }
            },
            "required": ["query"],
        },
        confirm_prompt=lambda args: f"Forget facts matching '{args.get('query', '')}'?",
    )
    def forget(query: str) -> str:
        return facts.forget(query)
