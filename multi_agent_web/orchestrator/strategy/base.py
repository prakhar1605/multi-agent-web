"""The Strategy interface: how a task becomes agent sessions and one answer.

DESIGNED FOR THE SECOND IMPLEMENTATION
======================================

Only best-of-N exists today, but the interface is shaped by the one coming in
Phase 3 -- decomposing a goal into a dependency graph of subtasks. Three
choices make that fit without changes here:

1. **A strategy receives the ``Runner``, not a list of results.** It decides
   what to launch and when. Best-of-N launches N copies of one task in a single
   wave; a DAG strategy launches a wave, reads the results, builds the next
   wave from them, and repeats. ``Runner.run_sessions`` is re-entrant and keeps
   assigning fresh agent indices, so both shapes work on one Runner.

2. **``contributing_indices`` is a list, not a winner.** Best-of-N puts exactly
   one index in it. A DAG run's answer is synthesised from several leaf agents
   and puts all of them in it. A single ``winner`` field would have needed
   widening later, and run.json is meant to be stable.

3. **``details`` is a free-form dict, serialized verbatim into run.json.**
   Best-of-N stores the judge's verdict; a DAG strategy will store the graph.
   Strategy-specific data has somewhere to go that does not require a schema
   change for every new strategy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..session import SessionResult

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from ..runner import Runner


@dataclass
class StrategyOutcome:
    """What a strategy produced: an answer, and the trajectories behind it."""

    answer: str | None
    #: Agents whose work the answer rests on. Exactly one for best-of-N.
    contributing_indices: list[int]
    #: Why this answer -- a judge's rationale, or an explanation of failure.
    reason: str
    #: Every session launched, including failures. Never silently dropped.
    sessions: list[SessionResult]
    #: Strategy-specific payload, written verbatim into run.json.
    details: dict[str, Any] = field(default_factory=dict)


class Strategy(ABC):
    """Turns one task into agent sessions, and their results into one answer."""

    name: str = "strategy"

    @abstractmethod
    async def run(self, task: str, runner: "Runner") -> StrategyOutcome:
        """Execute the strategy.

        Implementations should surface every session they launched in
        ``StrategyOutcome.sessions``, including crashed ones -- a partial run
        is a legitimate outcome and the report should show what happened to
        each agent.
        """
