"""MolmoWeb-4B policy -- STUB, NOT IMPLEMENTED.

This file is intentionally non-functional. Wiring it up requires the model's
exact output format, and guessing at that would produce a parser that looks
plausible, silently mis-grounds every click, and is harder to debug than no
parser at all.

TODO(phase-1b): implement once the following are known. Until then this raises.

  1. OUTPUT FORMAT. What does the model actually emit for one step? Raw text
     with points (``<point x="..." y="...">``)? JSON? A DSL like
     ``CLICK [x, y]``? Is the "thought" a separate field or inline prose?
     Paste one or two verbatim generations -- a real sample beats a spec.

  2. COORDINATE CONVENTION. Normalized 0-100, normalized 0-1, or absolute
     pixels? If normalized, normalized against what -- the resized model input,
     or the original screenshot? This determines the conversion into the
     absolute-viewport-pixel space defined in ``actions.py``:

         x_px = round(x_model / SCALE * config.viewport_width)

  3. IMAGE PREPROCESSING. Expected input resolution, resize/pad/crop behaviour,
     and whether aspect ratio is preserved. Letterboxing shifts coordinates and
     must be undone when mapping a predicted point back to the viewport.

  4. PROMPT TEMPLATE. Exact system/user template, how the task string is
     inserted, and how prior steps are represented in history (full text?
     action summaries only? previous screenshots too?).

  5. ACTION VOCABULARY. Which of our actions the model can express, and what it
     emits for anything outside that set. Also its termination signal, so it can
     be mapped onto ``Done(answer=...)``.

  6. SERVING MODE. Local ``transformers`` in-process, or an HTTP endpoint
     (vLLM / SGLang)? This decides whether ``predict`` does a network call or
     blocks on GPU work -- and if the latter, inference must go through
     ``asyncio.to_thread`` so parallel agents in Phase 2 do not serialize.

Everything above translates into code in THIS file only. ``actions.py``,
``browser.py`` and ``agent.py`` stay untouched -- that separation is the
reason the stub can sit here harmlessly.
"""

from __future__ import annotations

from collections.abc import Sequence

from PIL import Image

from ..browser import PageInfo
from ..config import RunConfig
from ..trajectory import Step
from .base import AgentPolicy

_NOT_IMPLEMENTED_MSG = (
    "MolmoWebPolicy is a stub. The model's output format and coordinate "
    "convention are not yet known, so no parser has been written. "
    "See the TODO in multi_agent_web/policy/molmoweb.py, and use MockPolicy "
    "to exercise the loop in the meantime."
)


class MolmoWebPolicy(AgentPolicy):
    """Placeholder for the MolmoWeb-4B vision-language policy."""

    name = "molmoweb"

    def __init__(
        self,
        model_id: str = "allenai/MolmoWeb-4B",
        config: RunConfig | None = None,
        **kwargs: object,
    ) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    async def predict(
        self,
        task: str,
        history: Sequence[Step],
        screenshot: Image.Image,
        page: PageInfo,
    ) -> Step:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
