"""C-Net's tools — the things it can actually *do*.

A tool is three pieces:
  1. a name + description   -> so the model knows when to reach for it
  2. an input schema (JSON) -> so the model knows what to pass
  3. a Python function      -> the code that actually runs

Adding a capability to C-Net means adding one function in one of these files.
"""

from __future__ import annotations

from ..config import Config
from .base import Tool, ToolRegistry


def build_registry(cfg: Config, facts=None) -> ToolRegistry:
    """Create the registry and load every tool module into it."""
    from . import file_tools, memory_tools, shell_tools, system_tools

    registry = ToolRegistry()
    system_tools.register(registry, cfg)
    file_tools.register(registry, cfg)
    if facts is not None:
        memory_tools.register(registry, cfg, facts)
    if cfg.allow_shell:
        shell_tools.register(registry, cfg)
    return registry


__all__ = ["Tool", "ToolRegistry", "build_registry"]
