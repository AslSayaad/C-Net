"""Tools that let C-Net look at the machine it lives on. Read-only."""

from __future__ import annotations

import os
import platform
import shutil
import socket
from datetime import datetime

from ..config import Config
from .base import ToolRegistry


def register(registry: ToolRegistry, cfg: Config) -> None:
    @registry.add(
        "get_datetime",
        "Get the current local date and time on this machine. Use this whenever "
        "the user asks about the time, today's date, the day of the week, or "
        "anything that depends on 'now'. Never estimate these yourself.",
    )
    def get_datetime() -> str:
        now = datetime.now().astimezone()
        return (
            f"{now.strftime('%A, %d %B %Y, %H:%M:%S')} "
            f"(timezone {now.tzname()}, ISO {now.isoformat(timespec='seconds')})"
        )

    @registry.add(
        "get_system_info",
        "Get facts about this computer: operating system, CPU count, Python "
        "version, hostname, and free disk space in the workspace. Use this when "
        "the user asks about their machine or how much space is left.",
    )
    def get_system_info() -> str:
        usage = shutil.disk_usage(cfg.workspace if cfg.workspace.exists() else os.getcwd())
        gb = 1024 ** 3
        lines = [
            f"OS:          {platform.system()} {platform.release()}",
            f"Machine:     {platform.machine()}",
            f"Hostname:    {socket.gethostname()}",
            f"Python:      {platform.python_version()}",
            f"CPU cores:   {os.cpu_count()}",
            f"Workspace:   {cfg.workspace}",
            f"Disk free:   {usage.free / gb:.1f} GB of {usage.total / gb:.1f} GB",
        ]
        return "\n".join(lines)
