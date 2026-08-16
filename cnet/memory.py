"""C-Net's two kinds of memory.

1. Conversation memory  — the running chat, so it knows what you just said.
2. Long-term facts      — a small JSON file that survives restarts.

The Claude API is stateless: it does not remember anything between calls.
"Memory" is entirely our job — we resend the conversation each turn.
That is the single most important thing to understand about building agents.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Conversation:
    """The message list we resend to the model on every turn."""

    def __init__(self, path: Path, max_messages: int = 60) -> None:
        self.path = path
        self.max_messages = max_messages
        self.messages: list[dict[str, Any]] = []

    # --- adding turns ------------------------------------------------
    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, content: Any) -> None:
        """Store the assistant turn.

        We store the FULL content list (text blocks, thinking blocks, tool_use
        blocks) rather than just the text. The API needs those blocks back
        verbatim on the next turn or the conversation breaks.
        """
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_results(self, results: list[dict[str, Any]]) -> None:
        """Tool results are sent back as a *user* message. That surprises
        everyone the first time. All results from one assistant turn must go
        in a single message."""
        self.messages.append({"role": "user", "content": results})

    # --- housekeeping ------------------------------------------------
    def trim(self) -> None:
        """Drop the oldest messages so the context does not grow forever.

        We never cut in the middle of a tool exchange: a message containing
        tool_result blocks must still be preceded by the assistant message
        that requested them, so we walk forward to the next clean 'user text'
        boundary.
        """
        if len(self.messages) <= self.max_messages:
            return
        start = len(self.messages) - self.max_messages
        while start < len(self.messages):
            msg = self.messages[start]
            if msg["role"] == "user" and isinstance(msg["content"], str):
                break
            start += 1
        if start < len(self.messages):
            self.messages = self.messages[start:]

    def clear(self) -> None:
        self.messages = []

    # --- persistence -------------------------------------------------
    def save(self) -> None:
        """Write the conversation to disk so `cnet` can resume it later."""
        try:
            payload = json.dumps(self.messages, default=_json_safe, indent=2)
        except (TypeError, ValueError):
            return  # A session we cannot serialise is not worth crashing over.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(payload, encoding="utf-8")

    def load(self) -> bool:
        """Restore a saved conversation. Returns True if anything was loaded."""
        if not self.path.is_file():
            return False
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        if isinstance(data, list) and data:
            self.messages = data
            return True
        return False


class FactStore:
    """A tiny long-term memory: a list of strings C-Net chose to keep."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.facts: list[str] = []
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if isinstance(data, list):
            self.facts = [str(item) for item in data]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.facts, indent=2), encoding="utf-8")

    def remember(self, fact: str) -> str:
        fact = fact.strip()
        if not fact:
            return "Nothing to remember — the fact was empty."
        if fact in self.facts:
            return f"Already remembered: {fact}"
        self.facts.append(fact)
        self._save()
        return f"Remembered: {fact}"

    def recall(self, query: str = "") -> list[str]:
        if not query:
            return list(self.facts)
        needle = query.lower()
        return [fact for fact in self.facts if needle in fact.lower()]

    def forget(self, query: str) -> str:
        needle = query.lower().strip()
        if not needle:
            return "Give me something to forget."
        removed = [fact for fact in self.facts if needle in fact.lower()]
        if not removed:
            return f"Nothing matching '{query}' in memory."
        self.facts = [fact for fact in self.facts if needle not in fact.lower()]
        self._save()
        return "Forgot: " + "; ".join(removed)


def _json_safe(obj: Any) -> Any:
    """Turn SDK objects (TextBlock, ToolUseBlock, ...) into plain dicts."""
    for attr in ("model_dump", "to_dict", "dict"):
        method = getattr(obj, attr, None)
        if callable(method):
            try:
                return method()
            except TypeError:
                continue
    return str(obj)
