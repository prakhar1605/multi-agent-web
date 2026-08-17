"""A judge that is an LLM. Opt-in; ``MockJudge`` stays the default.

The seam this fills was described in ``base.py`` before it existed: ``choose``
is async and takes the task text precisely so this could drop in unchanged.
Nothing in the interface moved.

WHAT IT SEES, AND WHAT IT DELIBERATELY DOES NOT
===============================================
Task, and per candidate: the answer, how the run ended, how many steps it took
and how many of them errored. No screenshots and no trajectories. That is the
same narrow view ``MockJudge`` gets, and keeping it narrow is what makes the two
comparable -- swap the judge, hold everything else, and the difference in
verdicts is attributable to judgement rather than to one of them having seen
more. It also keeps the judge cheap: one text call per run, no images.

Step and error counts are included because they are evidence about *how* an
answer was reached. Two agents reporting the same price, one in three clean
steps and one in twelve with five failures, are not equally believable.

WHY IT FAILS LOUDLY
===================
A malformed verdict raises. The judge runs after every agent has finished, so
raising here loses the run's headline answer -- but not the work: each agent's
trajectory, screenshots and answer are already written to
``agent_<i>/`` incrementally, so nothing is destroyed and the run can be judged
again offline. Silently falling back to the rule-based judge would be worse: the
run would look successful while the thing being evaluated had quietly not run.

The one out-of-band case handled without a model call is an empty candidate
list: there is nothing to judge and nothing to spend money on.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from ...config import JudgeConfig
from ...ppapi import ChatClient, PPAPIError, parse_json_object
from .base import Candidate, Judge, Verdict

logger = logging.getLogger(__name__)


class JudgeProtocolError(PPAPIError):
    """The judge's reply did not match the requested schema."""


SYSTEM_PROMPT = """\
You are judging several independent attempts at the same web browsing task. \
Each attempt was made by a separate agent working alone. Pick the one whose \
answer should be reported to the user.

Reply with a single JSON object and NOTHING else. No markdown, no code fence, \
no commentary before or after. The object has exactly two keys:

  "index"   the index of the attempt you pick, as an integer, or null if none \
of them is acceptable
  "reason"  one or two sentences on why that attempt and not the others

How to judge, in order:
  1. Does the answer actually address the task that was asked? An agent that \
finished confidently with the wrong kind of answer is worse than one that \
finished with none.
  2. Is it specific and plausible? Prefer a concrete answer over a vague one, \
and be suspicious of an answer that looks guessed rather than read off a page.
  3. Only an attempt with status "done" decided for itself that it was \
finished. "max_steps" ran out of budget, "aborted" hit repeated failures, and \
"crashed" never produced anything. Answers from those are partial at best.
  4. Between two attempts that are equally good on the above, prefer the one \
that took fewer steps and had fewer errors -- it found a more direct route.

Choose null only when no attempt produced a usable answer. Do not promote a \
wrong answer just to have one.
"""


class LLMJudge(Judge):
    """Picks among candidate trajectories with a model instead of a rule."""

    name = "llm"

    def __init__(
        self,
        client: ChatClient,
        config: JudgeConfig | None = None,
    ) -> None:
        self.client = client
        self.config = config or JudgeConfig()

    @property
    def model(self) -> str:
        return self.config.model or self.client.default_model

    async def choose(self, task: str, candidates: Sequence[Candidate]) -> Verdict:
        if not candidates:
            return Verdict(index=None, reason="no candidates were produced")

        raw = await self.client.complete(
            system=SYSTEM_PROMPT,
            user=build_user_message(task, candidates),
            model=self.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        logger.debug("judge raw generation: %s", raw)
        return parse_verdict(raw, [c.index for c in candidates])

    async def close(self) -> None:
        await self.client.close()


def build_user_message(task: str, candidates: Sequence[Candidate]) -> str:
    """The task and one block per attempt, keyed by the real agent index.

    Candidates are labelled with their orchestrator index rather than their
    position in this list, so the index the judge returns is the index the rest
    of the system already uses. Renumbering here would mean translating back,
    and a translation is one more place for a winner to be misattributed.
    """
    lines = [f"# TASK\n{task}\n", "# ATTEMPTS"]
    for candidate in candidates:
        answer = (candidate.answer or "").strip() or "(no answer)"
        lines.append(
            f"\n## Attempt index {candidate.index}\n"
            f"status: {candidate.status}\n"
            f"steps: {candidate.num_steps} ({candidate.num_errors} errored)\n"
            f"answer: {answer}"
        )
        if candidate.error:
            lines.append(f"error: {candidate.error}")
    lines.append(
        "\n# YOUR JOB\nPick one attempt index, or null. Reply with the JSON "
        "object."
    )
    return "\n".join(lines)


def parse_verdict(raw: str, valid_indices: Sequence[int]) -> Verdict:
    """Parse one reply into a ``Verdict``, or raise with the generation attached.

    An index outside ``valid_indices`` is rejected rather than clamped or
    dropped. A judge naming an attempt that does not exist has not judged the
    attempts it was shown, and quietly reinterpreting that as "no winner" would
    make a broken prompt indistinguishable from a run where everything failed.
    """
    payload = parse_json_object(raw, "judge", JudgeProtocolError)

    unexpected = set(payload) - {"index", "reason"}
    if unexpected:
        raise JudgeProtocolError(
            f"judge: reply has unexpected top-level key(s) {sorted(unexpected)}; "
            f"expected exactly 'index' and 'reason'.\nRaw generation: {raw}"
        )
    if "index" not in payload:
        raise JudgeProtocolError(
            f"judge: reply has no 'index' key.\nRaw generation: {raw}"
        )

    index = payload["index"]
    reason = payload.get("reason", "")
    if not isinstance(reason, str):
        raise JudgeProtocolError(
            f"judge: 'reason' must be a string, got {type(reason).__name__}."
            f"\nRaw generation: {raw}"
        )

    if index is None:
        return Verdict(index=None, reason=reason or "no attempt was acceptable")

    # bool is an int in Python, and `true` is not an index.
    if isinstance(index, bool) or not isinstance(index, int):
        raise JudgeProtocolError(
            f"judge: 'index' must be an integer or null, got {index!r}."
            f"\nRaw generation: {raw}"
        )
    if index not in valid_indices:
        raise JudgeProtocolError(
            f"judge: picked attempt {index}, which was not among the candidates "
            f"{sorted(valid_indices)}.\nRaw generation: {raw}"
        )
    return Verdict(index=index, reason=reason)


__all__ = ["JudgeProtocolError", "LLMJudge", "build_user_message", "parse_verdict"]
