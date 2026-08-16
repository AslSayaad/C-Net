"""Shell access — the sharpest tool in the box.

This is only loaded when CNET_ALLOW_SHELL=true, and every single command still
has to be approved by you before it runs. Commands come from a language model:
treat them as untrusted input, read them before saying yes, and leave this
switched off until you actually need it.
"""

from __future__ import annotations

import subprocess

from ..config import Config
from .base import ToolRegistry

TIMEOUT_SECONDS = 60


def register(registry: ToolRegistry, cfg: Config) -> None:
    @registry.add(
        "run_command",
        "Run a shell command in the workspace folder and return its output. "
        "The user must approve every command, so keep them short, single-purpose "
        "and non-destructive. Prefer the dedicated file tools when they can do "
        "the job.",
        {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The exact shell command to run.",
                },
                "reason": {
                    "type": "string",
                    "description": "One short line on why this command is needed.",
                },
            },
            "required": ["command"],
        },
        confirm_prompt=lambda args: (
            f"Run shell command?\n    {args.get('command')}"
            + (f"\n    reason: {args['reason']}" if args.get("reason") else "")
        ),
    )
    def run_command(command: str, reason: str = "") -> str:
        try:
            done = subprocess.run(
                command,
                shell=True,
                cwd=cfg.workspace,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return f"Command timed out after {TIMEOUT_SECONDS}s: {command}"

        parts = [f"exit code: {done.returncode}"]
        if done.stdout.strip():
            parts.append(f"stdout:\n{done.stdout.strip()}")
        if done.stderr.strip():
            parts.append(f"stderr:\n{done.stderr.strip()}")
        if len(parts) == 1:
            parts.append("(no output)")
        return "\n".join(parts)
