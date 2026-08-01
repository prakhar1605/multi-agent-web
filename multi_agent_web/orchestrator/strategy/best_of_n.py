"""Best-of-N: run the same task N times in parallel, let a judge pick a winner.

The simplest thing that uses the parallel substrate for something other than
throughput. Sampling is stochastic (MolmoWeb runs at temperature 0.7), so N
independent attempts explore different routes through a site, and a judge
choosing among finished trajectories beats taking whichever one you happened to
run. It also gives a natural robustness story: one agent hitting a captcha or a
dead end does not sink the task.

Every session launched is handed to the judge, including the ones that crashed
or ran out of steps -- labelled with what happened. Filtering failures out
before judging would make "three of four agents crashed" indistinguishable from
"one agent answered", which is exactly the thing worth knowing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..judge.base import Candidate, Judge
from ..judge.mock import MockJudge
from ..session import SessionSpec
from .base import Strategy, StrategyOutcome

if TYPE_CHECKING:  # pragma: no cover
    from ..runner import Runner

logger = logging.getLogger(__name__)


class BestOfN(Strategy):
    """Launch ``n`` agents on one task; the judge picks which answer to report."""

    name = "best_of_n"

    def __init__(
        self,
        n: int = 3,
        judge: Judge | None = None,
        start_url: str | None = None,
    ) -> None:
        if n < 1:
            raise ValueError("n must be at least 1")
        self.n = n
        self.judge = judge or MockJudge()
        self.start_url = start_url

    async def run(self, task: str, runner: "Runner") -> StrategyOutcome:
        specs = [
            SessionSpec(
                task=task,
                start_url=self.start_url,
                label=f"candidate {i + 1}/{self.n}",
            )
            for i in range(self.n)
        ]
        logger.info("best_of_n: launching %d agents on the same task", self.n)
        sessions = await runner.run_sessions(specs)

        # Every session becomes a candidate, failures included.
        candidates = [Candidate.from_session(s) for s in sessions]
        verdict = await self.judge.choose(task, candidates)

        by_index = {c.index: c for c in candidates}
        winner = by_index.get(verdict.index) if verdict.index is not None else None

        if winner is None:
            logger.warning("best_of_n: no winner -- %s", verdict.reason)
        else:
            logger.info("best_of_n: agent %d won -- %s", winner.index, verdict.reason)

        return StrategyOutcome(
            answer=winner.answer if winner else None,
            contributing_indices=[winner.index] if winner else [],
            reason=verdict.reason,
            sessions=sessions,
            details={
                "n": self.n,
                # "winner" rather than the Verdict's "index": inside a "judge"
                # object in a format the UI reads, "index" is ambiguous.
                "judge": {
                    "name": self.judge.name,
                    "winner": verdict.index,
                    "reason": verdict.reason,
                },
                "candidates": [c.as_dict() for c in candidates],
            },
        )
