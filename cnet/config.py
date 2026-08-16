"""Every knob C-Net has, in one place.

Settings come from environment variables (usually loaded from a .env file).
Nothing here talks to the network — it is only about "what should C-Net do".
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# The folder where C-Net keeps its own state: conversation history, long-term
# facts, logs. Kept out of your project folders on purpose.
HOME = Path(os.environ.get("CNET_HOME", Path.home() / ".cnet")).expanduser()


def _load_dotenv(path: Path) -> None:
    """Read KEY=value lines out of a .env file into os.environ.

    Written by hand so C-Net has one less dependency, and so you can see
    exactly what "loading a .env file" actually means.
    Existing environment variables always win.
    """
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    """The full configuration for one C-Net session."""

    # --- Who C-Net is ---
    name: str = "C-Net"
    user_name: str = "boss"

    # --- Which model it thinks with ---
    # claude-opus-5 is the most capable model. If you want cheaper/faster
    # replies while experimenting, set CNET_MODEL=claude-sonnet-5.
    model: str = "claude-opus-5"
    # How hard the model works before answering: low | medium | high | xhigh | max
    # "medium" is a good default for a chat assistant. Raise it for hard tasks.
    effort: str = "medium"
    max_tokens: int = 8000

    # --- Where it is allowed to touch files ---
    # File tools refuse to read or write anything outside this folder.
    workspace: Path = field(default_factory=lambda: Path(
        os.environ.get("CNET_WORKSPACE", Path.home() / "cnet-workspace")
    ).expanduser())

    # --- Safety switches ---
    # Shell access is off unless you deliberately turn it on, and even then
    # every command asks for your approval first.
    allow_shell: bool = False
    # Ask before writing/deleting files. Leave this on until you trust it.
    confirm_writes: bool = True

    # --- Session state ---
    home: Path = field(default_factory=lambda: HOME)
    # Keep the last N messages in context. Older ones are dropped so long
    # sessions do not grow forever. (Phase 3 in the roadmap replaces this
    # with proper summarisation.)
    max_history_messages: int = 60

    api_key: str | None = None

    @classmethod
    def load(cls) -> "Config":
        """Build a Config from .env + environment variables."""
        # Look for a .env next to the project, then one in ~/.cnet/
        _load_dotenv(Path.cwd() / ".env")
        _load_dotenv(HOME / ".env")

        cfg = cls(
            name=os.environ.get("CNET_NAME", "C-Net"),
            user_name=os.environ.get("CNET_USER_NAME", "boss"),
            model=os.environ.get("CNET_MODEL", "claude-opus-5"),
            effort=os.environ.get("CNET_EFFORT", "medium"),
            max_tokens=int(os.environ.get("CNET_MAX_TOKENS", "8000")),
            allow_shell=_env_bool("CNET_ALLOW_SHELL", False),
            confirm_writes=_env_bool("CNET_CONFIRM_WRITES", True),
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )
        cfg.home.mkdir(parents=True, exist_ok=True)
        cfg.workspace.mkdir(parents=True, exist_ok=True)
        return cfg

    # Files C-Net uses to remember things between runs.
    @property
    def session_file(self) -> Path:
        return self.home / "session.json"

    @property
    def facts_file(self) -> Path:
        return self.home / "facts.json"
