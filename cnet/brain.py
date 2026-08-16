"""The agent loop: C-Net's actual thinking.

This file is small on purpose, because the loop itself is simple:

    1. Send the conversation + tool list to Claude.
    2. If Claude just talked  -> we're done, show the answer.
    3. If Claude asked for a tool -> run it, append the result, go to 1.

That loop is what separates an "agent" from a chatbot. Everything else in this
project is scaffolding around these ~40 lines.
"""

from __future__ import annotations

from typing import Any, Callable

import anthropic

from .config import Config
from .memory import Conversation, FactStore
from .personality import build_system_prompt
from .tools import ToolRegistry
from .tools.base import ConfirmFn

# Called with each chunk of text as the model writes it, so the CLI can stream.
StreamFn = Callable[[str], None]
# Called when C-Net starts/finishes a tool, so the CLI can show activity.
ToolEventFn = Callable[[str, dict[str, Any], str | None], None]

# A hard stop so a confused model cannot loop forever burning tokens.
MAX_TOOL_ROUNDS = 12


class MissingAPIKey(RuntimeError):
    pass


class Brain:
    def __init__(
        self,
        cfg: Config,
        conversation: Conversation,
        facts: FactStore,
        registry: ToolRegistry,
    ) -> None:
        if not cfg.api_key:
            raise MissingAPIKey(
                "No ANTHROPIC_API_KEY found.\n"
                "Get a key at https://console.anthropic.com, then either:\n"
                "  export ANTHROPIC_API_KEY=sk-ant-...\n"
                "or put it in a .env file next to this project."
            )
        self.cfg = cfg
        self.conversation = conversation
        self.facts = facts
        self.registry = registry
        self.client = anthropic.Anthropic(api_key=cfg.api_key)

    # ------------------------------------------------------------------
    def think(
        self,
        user_input: str,
        on_text: StreamFn | None = None,
        on_tool: ToolEventFn | None = None,
        confirm: ConfirmFn | None = None,
    ) -> str:
        """Handle one user message, all the way to a final answer."""
        self.conversation.add_user(user_input)
        self.conversation.trim()

        final_text = ""
        for _ in range(MAX_TOOL_ROUNDS):
            response = self._call_model(on_text)

            # The model can decline a request outright. That is a normal
            # outcome with a 200 response, not an exception — so check it
            # before touching response.content.
            if response.stop_reason == "refusal":
                message = "I can't help with that one."
                self.conversation.add_assistant(message)
                return message

            # Store the assistant turn *whole* — thinking and tool_use blocks
            # included. The API needs them back verbatim next round.
            self.conversation.add_assistant(response.content)

            text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            if text:
                final_text = text

            if response.stop_reason != "tool_use":
                return final_text  # Nothing left to do: this is the answer.

            # Run every tool the model asked for, then hand the results back.
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                if on_tool:
                    on_tool(block.name, dict(block.input), None)
                result = self.registry.run(block.name, dict(block.input), confirm=confirm)
                if on_tool:
                    on_tool(block.name, dict(block.input), result.text)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result.text,
                        "is_error": result.is_error,
                    }
                )
            self.conversation.add_tool_results(tool_results)

        return (
            final_text
            or f"I used {MAX_TOOL_ROUNDS} tool rounds without finishing. "
            "Try breaking the task into smaller steps."
        )

    # ------------------------------------------------------------------
    def _call_model(self, on_text: StreamFn | None):
        """One request to Claude, streamed so text appears as it is written."""
        system = build_system_prompt(self.cfg, self.facts.facts)

        with self.client.messages.stream(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            system=system,
            messages=self.conversation.messages,
            tools=self.registry.specs(),
            # Adaptive thinking lets the model decide how hard to think per
            # message; effort tunes the overall depth/cost trade-off.
            thinking={"type": "adaptive"},
            output_config={"effort": self.cfg.effort},
        ) as stream:
            if on_text:
                for chunk in stream.text_stream:
                    on_text(chunk)
            return stream.get_final_message()
