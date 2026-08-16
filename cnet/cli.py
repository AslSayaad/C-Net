"""The terminal you talk to C-Net in.

Run it with:  python -m cnet
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .brain import Brain, MissingAPIKey
from .config import Config
from .memory import Conversation, FactStore
from .tools import build_registry

# --- tiny colour helpers (no dependencies, works in any terminal) ------
RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"


def paint(text: str, colour: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{colour}{text}{RESET}"


BANNER = r"""
   ____      _   _      _
  / ___|    | \ | | ___| |_
 | |   ____ |  \| |/ _ \ __|
 | |__|____|| |\  |  __/ |_
  \____|    |_| \_|\___|\__|
"""

HELP = """
Commands:
  /help              show this
  /tools             list the tools C-Net can use
  /memory            show the facts C-Net remembers about you
  /forget <text>     delete remembered facts matching <text>
  /new               start a fresh conversation (memory of facts is kept)
  /effort <level>    low | medium | high | xhigh | max  (how hard it thinks)
  /model <name>      switch model, e.g. claude-sonnet-5
  /config            show current settings
  /quit              exit (also: Ctrl-D)

Anything else is just talked to C-Net.
"""


def confirm(question: str) -> bool:
    """Ask the human before C-Net does something that changes the machine."""
    print()
    print(paint("  ⚠ " + question, YELLOW))
    try:
        answer = input(paint("  approve? [y/N] ", YELLOW)).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in {"y", "yes"}


def make_tool_reporter(verbose: bool):
    """Show what C-Net is doing while it works."""

    def report(name: str, args: dict, result: str | None) -> None:
        if result is None:
            detail = ", ".join(f"{k}={_short(v)}" for k, v in args.items())
            print(paint(f"\n  ⚙ {name}({detail})", DIM), flush=True)
        elif verbose:
            first = result.strip().splitlines()[0] if result.strip() else ""
            print(paint(f"  → {_short(first, 100)}", DIM), flush=True)

    return report


def _short(value, limit: int = 60) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def run_repl(cfg: Config, brain: Brain, facts: FactStore, verbose: bool) -> int:
    registry = brain.registry
    on_tool = make_tool_reporter(verbose)

    print(paint(BANNER, CYAN))
    print(f"  {cfg.name} v{__version__} · {cfg.model} · effort={cfg.effort}")
    print(paint(f"  workspace: {cfg.workspace}", DIM))
    print(paint(f"  tools: {', '.join(registry.names())}", DIM))
    print(paint("  /help for commands, /quit to leave\n", DIM))

    while True:
        try:
            user_input = input(paint("you › ", BOLD + GREEN)).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nlater.")
            return 0

        if not user_input:
            continue

        if user_input.startswith("/"):
            if handle_command(user_input, cfg, brain, facts) == "quit":
                return 0
            continue

        print(paint(f"\n{cfg.name.lower()} › ", BOLD + CYAN), end="", flush=True)
        try:
            answer = brain.think(
                user_input,
                on_text=lambda chunk: print(chunk, end="", flush=True),
                on_tool=on_tool,
                confirm=confirm,
            )
        except KeyboardInterrupt:
            print(paint("\n  [interrupted]\n", YELLOW))
            continue
        except Exception as exc:  # noqa: BLE001 - never drop the user's session
            print(paint(f"\n  error: {exc}\n", RED))
            continue

        if not answer.strip():
            print(paint("(no reply)", DIM))
        print("\n")
        brain.conversation.save()


def handle_command(line: str, cfg: Config, brain: Brain, facts: FactStore) -> str | None:
    parts = line.split(maxsplit=1)
    command = parts[0].lower()
    argument = parts[1].strip() if len(parts) > 1 else ""

    if command in {"/quit", "/exit", "/q"}:
        brain.conversation.save()
        print("later.")
        return "quit"

    if command == "/help":
        print(HELP)

    elif command == "/tools":
        for tool in brain.registry.all():
            print(f"  {paint(tool.name, BOLD)}: {tool.description.split('.')[0]}.")

    elif command == "/memory":
        if not facts.facts:
            print(paint("  nothing remembered yet.", DIM))
        for fact in facts.facts:
            print(f"  - {fact}")

    elif command == "/forget":
        print("  " + facts.forget(argument))

    elif command == "/new":
        brain.conversation.clear()
        brain.conversation.save()
        print(paint("  fresh conversation. saved facts kept.", DIM))

    elif command == "/effort":
        levels = {"low", "medium", "high", "xhigh", "max"}
        if argument in levels:
            cfg.effort = argument
            print(paint(f"  effort → {argument}", DIM))
        else:
            print(f"  pick one of: {', '.join(sorted(levels))}")

    elif command == "/model":
        if argument:
            cfg.model = argument
            print(paint(f"  model → {argument}", DIM))
        else:
            print(f"  current model: {cfg.model}")

    elif command == "/config":
        print(f"  model:      {cfg.model}")
        print(f"  effort:     {cfg.effort}")
        print(f"  max tokens: {cfg.max_tokens}")
        print(f"  workspace:  {cfg.workspace}")
        print(f"  shell:      {'on' if cfg.allow_shell else 'off'}")
        print(f"  confirm:    {'on' if cfg.confirm_writes else 'off'}")
        print(f"  history:    {len(brain.conversation.messages)} messages")

    else:
        print(f"  unknown command {command} — try /help")

    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cnet", description="C-Net — your local AI agent."
    )
    parser.add_argument("prompt", nargs="*", help="ask one question and exit")
    parser.add_argument("--resume", action="store_true", help="continue the last conversation")
    parser.add_argument("--model", help="override the model for this run")
    parser.add_argument("--effort", help="low | medium | high | xhigh | max")
    parser.add_argument("--verbose", action="store_true", help="show tool results")
    parser.add_argument("--version", action="version", version=f"C-Net {__version__}")
    args = parser.parse_args(argv)

    cfg = Config.load()
    if args.model:
        cfg.model = args.model
    if args.effort:
        cfg.effort = args.effort

    facts = FactStore(cfg.facts_file)
    conversation = Conversation(cfg.session_file, cfg.max_history_messages)
    if args.resume and conversation.load():
        print(paint(f"resumed {len(conversation.messages)} messages\n", DIM))

    registry = build_registry(cfg, facts)

    try:
        brain = Brain(cfg, conversation, facts, registry)
    except MissingAPIKey as exc:
        print(paint(str(exc), RED))
        return 1

    # One-shot mode: `python -m cnet "what time is it"`
    if args.prompt:
        question = " ".join(args.prompt)
        try:
            brain.think(
                question,
                on_text=lambda chunk: print(chunk, end="", flush=True),
                on_tool=make_tool_reporter(args.verbose),
                confirm=confirm,
            )
        except Exception as exc:  # noqa: BLE001
            print(paint(f"error: {exc}", RED))
            return 1
        print()
        conversation.save()
        return 0

    return run_repl(cfg, brain, facts, args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
