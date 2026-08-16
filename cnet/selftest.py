"""Offline check that C-Net is wired up correctly.

Run:  python -m cnet.selftest

This never calls the API, so it costs nothing and works without a key.
It exercises the tools directly — the same code paths the model uses.
"""

from __future__ import annotations

import sys

from .config import Config
from .memory import Conversation, FactStore
from .personality import build_system_prompt
from .tools import build_registry

OK = "  \033[32mok\033[0m  "
BAD = "  \033[31mfail\033[0m"


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"{OK if condition else BAD}  {label}{(' — ' + detail) if detail else ''}")
    return condition


def main() -> int:
    print("\nC-Net self test (no API calls)\n" + "-" * 34)
    passed = True

    cfg = Config.load()
    passed &= check("config loads", True, f"model={cfg.model}, effort={cfg.effort}")
    passed &= check("workspace exists", cfg.workspace.is_dir(), str(cfg.workspace))
    passed &= check(
        "api key present",
        bool(cfg.api_key),
        "set ANTHROPIC_API_KEY before chatting" if not cfg.api_key else "found",
    )

    facts = FactStore(cfg.facts_file)
    registry = build_registry(cfg, facts)
    passed &= check("tools registered", len(registry.names()) > 0, ", ".join(registry.names()))

    # Every tool must produce a valid API spec, or the request will 400.
    for tool in registry.all():
        spec = tool.spec()
        valid = bool(spec["name"] and spec["description"]) and spec["input_schema"]["type"] == "object"
        passed &= check(f"schema: {tool.name}", valid)

    clock = registry.run("get_datetime", {})
    passed &= check("clock tool runs", not clock.is_error, clock.text.split("(")[0].strip())

    info = registry.run("get_system_info", {})
    passed &= check("system tool runs", not info.is_error)

    # The sandbox is the safety-critical piece, so test that it actually holds.
    escape = registry.run("read_file", {"path": "../../../etc/passwd"})
    blocked = "outside the workspace" in escape.text
    passed &= check("sandbox blocks path escape", blocked, escape.text.split(".")[0][:60])

    write = registry.run("write_file", {"path": "cnet-selftest.txt", "content": "hello from C-Net"})
    read = registry.run("read_file", {"path": "cnet-selftest.txt"})
    passed &= check("write + read a file", read.text.strip() == "hello from C-Net")
    (cfg.workspace / "cnet-selftest.txt").unlink(missing_ok=True)

    conversation = Conversation(cfg.session_file, cfg.max_history_messages)
    conversation.add_user("hi")
    passed &= check("conversation memory", len(conversation.messages) == 1)

    prompt = build_system_prompt(cfg, facts.facts)
    passed &= check("system prompt builds", cfg.name in prompt, f"{len(prompt)} chars")

    print("-" * 34)
    if passed:
        print("all good. run:  python -m cnet\n")
        return 0
    print("something is off — see the failures above.\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
