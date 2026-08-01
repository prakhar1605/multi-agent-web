"""A scripted policy that replays a fixed list of actions.

This exists so the entire loop -- browser, execution, trajectory logging, CLI,
tests -- can be exercised with no GPU, no API key and no network. When
MolmoWeb is wired up and something breaks, swapping ``MockPolicy`` back in
tells you immediately whether the bug is in the model or in the harness.

It ignores the screenshot entirely. That is the point: it is a control.
"""

from __future__ import annotations

from collections.abc import Sequence

from PIL import Image

from ..actions import Action, Done
from ..browser import PageInfo
from ..trajectory import Step
from .base import AgentPolicy

# Either a bare action, or a (thought, action) pair.
ScriptEntry = Action | tuple[str, Action]


class MockPolicy(AgentPolicy):
    """Replays ``script`` in order, then emits ``Done``.

    Args:
        script: Actions to replay. Entries may be ``(thought, action)`` tuples
            if you want readable trajectories.
        fallback_answer: Answer used by the auto-generated ``Done`` action when
            the script runs out. Guaranteeing termination this way means a
            short script can never hang the loop until ``max_steps``.
    """

    name = "mock"

    def __init__(
        self,
        script: Sequence[ScriptEntry],
        fallback_answer: str = "mock policy: script exhausted",
    ) -> None:
        self._script: list[tuple[str, Action]] = [
            entry if isinstance(entry, tuple) else ("", entry) for entry in script
        ]
        self._fallback_answer = fallback_answer
        self._cursor = 0

    async def reset(self) -> None:
        self._cursor = 0

    async def predict(
        self,
        task: str,
        history: Sequence[Step],
        screenshot: Image.Image,
        page: PageInfo,
    ) -> Step:
        if self._cursor >= len(self._script):
            return Step(
                thought="Script exhausted; finishing.",
                action=Done(answer=self._fallback_answer),
            )

        thought, action = self._script[self._cursor]
        self._cursor += 1
        return Step(
            thought=thought or f"Scripted step {self._cursor}: {action.summary()}",
            action=action,
        )

    @property
    def remaining(self) -> int:
        """How many scripted actions are left. Handy in assertions."""
        return max(0, len(self._script) - self._cursor)
