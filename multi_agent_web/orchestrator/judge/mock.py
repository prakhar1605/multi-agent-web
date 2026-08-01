"""A deterministic judge. No model, no API key, no randomness.

Its purpose is the same as ``MockPolicy``'s: make the surrounding machinery
testable on its own. When an LLM judge lands in Phase 3 and best-of-N starts
picking odd winners, swapping this back in says immediately whether the bug is
in the judge or in the orchestration.

Ranking, in order:

1. Only runs that terminated deliberately (``Done``) are eligible. A run that
   hit ``max_steps`` did not decide it was finished, and a crashed run has no
   answer at all -- promoting either would report a non-answer as the result.
2. Among those, prefer a non-empty answer. An agent that emitted ``[EXIT]``
   with no text terminated correctly but told us nothing.
3. Then fewest steps -- of two agents that both answered, the one that got
   there in less work took the more direct route.
4. Then fewest errored steps, then lowest index. Both purely for determinism.
"""

from __future__ import annotations

from collections.abc import Sequence

from .base import Candidate, Judge, Verdict


class MockJudge(Judge):
    """Rule-based, fully deterministic: same candidates always give same verdict."""

    name = "mock"

    async def choose(self, task: str, candidates: Sequence[Candidate]) -> Verdict:
        if not candidates:
            return Verdict(index=None, reason="no candidates were produced")

        eligible = [c for c in candidates if c.terminated_deliberately]
        if not eligible:
            return Verdict(
                index=None,
                reason=(
                    "no agent terminated deliberately; "
                    f"{_describe_statuses(candidates)}. No answer is being "
                    "reported rather than promoting an unfinished run."
                ),
            )

        answered = [c for c in eligible if (c.answer or "").strip()]
        pool = answered or eligible
        winner = min(pool, key=lambda c: (c.num_steps, c.num_errors, c.index))

        if not answered:
            reason = (
                f"agent {winner.index} terminated deliberately but with an empty "
                f"answer; no candidate produced answer text "
                f"({len(eligible)}/{len(candidates)} finished)"
            )
        else:
            reason = (
                f"agent {winner.index} finished in {winner.num_steps} step(s) with "
                f"{winner.num_errors} error(s) -- fewest steps among "
                f"{len(answered)} agent(s) that answered "
                f"(of {len(candidates)} launched)"
            )
        return Verdict(index=winner.index, reason=reason)


def _describe_statuses(candidates: Sequence[Candidate]) -> str:
    counts: dict[str, int] = {}
    for c in candidates:
        counts[c.status] = counts.get(c.status, 0) + 1
    return ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))
