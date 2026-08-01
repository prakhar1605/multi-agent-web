"""The policy interface -- the seam between "how to browse" and "what to do".

Everything else in this package is fixed infrastructure. This is the part that
gets swapped: a mock script today, MolmoWeb-4B tomorrow, an API-backed model
after that. Nothing in this file may import a model runtime, a tokenizer, or a
serving client -- keeping it dependency-free is what makes the swap cheap.

The interface is intentionally one method:

    predict(task, history, screenshot, page) -> Step

which is a pure function of the observation and the trajectory so far. That
matches how a VLM policy actually works and keeps policies stateless enough to
run many of them in parallel in Phase 2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from PIL import Image

from ..browser import PageInfo
from ..trajectory import Step


class AgentPolicy(ABC):
    """Decides the next browser action from a screenshot and a task."""

    name: str = "policy"

    @abstractmethod
    async def predict(
        self,
        task: str,
        history: Sequence[Step],
        screenshot: Image.Image,
        page: PageInfo,
    ) -> Step:
        """Return the next :class:`Step` (a thought plus one action).

        Args:
            task: The natural-language goal, unchanged for the whole episode.
            history: Steps already taken, oldest first. A policy may use as
                much or as little of it as it likes; short-context models
                typically use only the last few.
            screenshot: The current viewport, ``viewport_width x
                viewport_height`` pixels. Any coordinate the returned action
                carries must be in this image's pixel space -- see
                ``actions.py`` for the full convention.
            page: URL and title, since a screenshot cannot convey those.

        Returns:
            A ``Step`` whose ``action`` is executed next. Return a ``Done``
            action to end the episode.

        Implementations should raise on unrecoverable errors; the agent loop
        catches them, logs them, and decides whether to continue.

        This is ``async`` because real policies do network or GPU I/O. Phase 2
        runs several agents concurrently, and a blocking ``predict`` would
        serialize them on the event loop.
        """

    async def reset(self) -> None:
        """Clear any per-episode state. Called once before each run."""

    async def close(self) -> None:
        """Release resources (HTTP sessions, model handles). Optional."""
