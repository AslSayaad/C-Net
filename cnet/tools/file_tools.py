"""File tools, locked to the workspace folder.

The important idea here is the sandbox. The model chooses the path, and model
output is never trusted input — so every path is resolved to its real location
and checked to be inside the workspace before anything is opened. That check
is what stops '../../.ssh/id_rsa' from working.
"""

from __future__ import annotations

from functools import wraps
from pathlib import Path

from ..config import Config
from .base import ToolRegistry

# Files bigger than this are truncated rather than dumped into the context.
MAX_READ_BYTES = 100_000


class OutsideWorkspace(Exception):
    """Raised when a requested path escapes the sandbox."""


def guard(fn):
    """Turn a sandbox violation into a plain message for the model.

    The model should learn "that path is off limits" and adjust, not see a
    Python traceback.
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except OutsideWorkspace as exc:
            return f"Refused: {exc}"

    return wrapper


def register(registry: ToolRegistry, cfg: Config) -> None:
    root = cfg.workspace.resolve()

    def resolve(relative_path: str) -> Path:
        """Turn a model-supplied path into a real path inside the workspace."""
        candidate = (root / relative_path).expanduser()
        try:
            real = candidate.resolve()
        except OSError as exc:
            raise OutsideWorkspace(f"Cannot resolve path '{relative_path}': {exc}") from exc
        if real != root and root not in real.parents:
            raise OutsideWorkspace(
                f"'{relative_path}' is outside the workspace ({root}). "
                "C-Net can only touch files in there."
            )
        return real

    @registry.add(
        "list_files",
        "List files and folders inside the workspace. Use this before reading "
        "or writing so you know what actually exists. Path is relative to the "
        "workspace root; use '.' for the top level.",
        {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Folder to list, relative to the workspace root. Defaults to '.'.",
                }
            },
        },
    )
    @guard
    def list_files(path: str = ".") -> str:
        target = resolve(path)
        if not target.exists():
            return f"'{path}' does not exist in the workspace."
        if target.is_file():
            return f"{path} is a file ({target.stat().st_size} bytes), not a folder."
        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        if not entries:
            return f"'{path}' is empty."
        lines = []
        for entry in entries:
            rel = entry.relative_to(root)
            if entry.is_dir():
                lines.append(f"{rel}/")
            else:
                lines.append(f"{rel}  ({entry.stat().st_size} bytes)")
        return "\n".join(lines)

    @registry.add(
        "read_file",
        "Read a text file from the workspace and return its contents. Always "
        "read a file before answering questions about it — never guess what is "
        "inside.",
        {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File to read, relative to the workspace root.",
                }
            },
            "required": ["path"],
        },
    )
    @guard
    def read_file(path: str) -> str:
        target = resolve(path)
        if not target.is_file():
            return f"No file at '{path}'. Use list_files to see what exists."
        data = target.read_bytes()[: MAX_READ_BYTES + 1]
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return f"'{path}' is not a UTF-8 text file (probably binary)."
        if len(data) > MAX_READ_BYTES:
            text = text[:MAX_READ_BYTES] + "\n...[file truncated]"
        return text or "(the file is empty)"

    @registry.add(
        "write_file",
        "Create a file in the workspace or replace its entire contents. This "
        "overwrites, so read the file first if you only mean to change part of "
        "it. Parent folders are created automatically.",
        {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File to write, relative to the workspace root.",
                },
                "content": {
                    "type": "string",
                    "description": "The full text to write into the file.",
                },
            },
            "required": ["path", "content"],
        },
        confirm_prompt=(
            lambda args: f"Write {len(args.get('content', ''))} chars to {args.get('path')}?"
        )
        if cfg.confirm_writes
        else None,
    )
    @guard
    def write_file(path: str, content: str) -> str:
        target = resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        existed = target.is_file()
        target.write_text(content, encoding="utf-8")
        verb = "Overwrote" if existed else "Created"
        return f"{verb} {target.relative_to(root)} ({len(content)} chars)."
