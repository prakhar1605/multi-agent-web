"""The Judge interface: pick the best candidate trajectory.

A judge sees, per candidate: the task, the final answer, how the run ended, and
how much work it took. It does not see screenshots -- keeping the interface
this narrow means a judge can be a deterministic rule, a heuristic, or an LLM,
without any of them needing browser access.

PHASE 3 SEAM
------------
``choose`` is ``async`` and takes the task text precisely so that an LLM judge
drops in here unchanged: build a prompt from ``task`` and ``candidates``, call
the model, parse an index and a rationale. That is deliberately NOT written
yet. ``MockJudge`` keeps every test and demo running with no API key.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..session import SessionResult


@dataclass
class Candidate:
    """One agent's outcome, as the judge sees it.

    Failed and truncated runs are candidates too, labelled by ``status``. They
    are shown to the judge rather than filtered out, because "every agent
    crashed" and "one agent answered" need to be distinguishable, and because a
    max_steps run often still carries a partial answer worth considering.
    """

    index: int
    task: str
    answer: str | None
    status: str  # "done" | "max_steps" | "aborted" | "crashed"
    num_steps: int
    num_errors: int
    error: str | None = None
    label: str | None = None

    @classmethod
    def from_session(cls, session: SessionResult) -> "Candidate":
        return cls(
            index=session.index,
            task=session.spec.task,
            answer=session.answer,
            status=session.status,
            num_steps=len(session.steps),
            num_errors=session.num_errors,
            error=session.error,
            label=session.spec.label,
        )

    @property
    def terminated_deliberately(self) -> bool:
        return self.status == "done"

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "answer": self.answer,
            "status": self.status,
            "num_steps": self.num_steps,
            "num_errors": self.num_errors,
            "error": self.error,
            "label": self.label,
        }


@dataclass
class Verdict:
    """The judge's choice.

    ``index`` is ``None`` when no candidate is acceptable -- e.g. every agent
    crashed. That is reported honestly rather than by promoting a failure.
    """

    index: int | None
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"index": self.index, "reason": self.reason}


class Judge(ABC):
    """Chooses among candidate trajectories."""

    name: str = "judge"

    @abstractmethod
    async def choose(self, task: str, candidates: Sequence[Candidate]) -> Verdict:
        """Pick the best candidate, or ``None`` if none is acceptable."""
