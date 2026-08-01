"""multi_agent_web -- vision-language browser agents.

Phase 1: a single agent loop of screenshot -> policy -> action.
Phase 2 (not built yet): a manager that decomposes one goal into subtasks and
runs several of these agents in parallel.
"""

from __future__ import annotations

from .actions import (
    ACTION_ADAPTER,
    Action,
    Click,
    Done,
    KeyPress,
    Navigate,
    Scroll,
    Type,
    Wait,
)
from .agent import Agent, RunResult, run_task
from .browser import ActionError, BrowserSession, PageInfo
from .config import RunConfig
from .trajectory import Step, Trajectory, load_trajectory

__version__ = "0.1.0"

__all__ = [
    "ACTION_ADAPTER",
    "Action",
    "ActionError",
    "Agent",
    "BrowserSession",
    "Click",
    "Done",
    "KeyPress",
    "Navigate",
    "PageInfo",
    "RunConfig",
    "RunResult",
    "Scroll",
    "Step",
    "Trajectory",
    "Type",
    "Wait",
    "load_trajectory",
    "run_task",
    "__version__",
]
