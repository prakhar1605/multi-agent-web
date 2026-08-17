"""Phase 3: the manager LLM and the plans it produces.

    manager.decompose(task)          -> DAG        (validated on construction)
    manager.replan(dag, outcomes)    -> Replan | None

This package knows nothing about browsers, sessions or the orchestrator. It
turns text into a validated graph and back again, which is what makes it
testable against a stub client with no key and no network -- and what keeps the
dependency pointing one way: ``orchestrator.strategy.dag`` imports this, never
the reverse.
"""

from __future__ import annotations

from .manager import (
    Manager,
    ManagerProtocolError,
    PlanningBudget,
    build_subtask,
    format_context,
)
from .plan import DAG, InvalidPlan, Replan, Subtask, SubtaskOutcome, SubtaskStatus

__all__ = [
    "DAG",
    "InvalidPlan",
    "Manager",
    "ManagerProtocolError",
    "PlanningBudget",
    "Replan",
    "Subtask",
    "SubtaskOutcome",
    "SubtaskStatus",
    "build_subtask",
    "format_context",
]
