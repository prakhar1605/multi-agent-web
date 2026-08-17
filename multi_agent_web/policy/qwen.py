"""Qwen policy: a general vision-language model driving our own action schema.

This is the third adapter behind ``AgentPolicy``, and the one that tests whether
the interface was designed right. MolmoWeb is a purpose-trained web agent with a
fixed output grammar we had to reverse-engineer and match. Qwen is a general
model that will emit whatever we ask it to -- so we do not imitate anyone's
grammar here. We ask for **our** action schema directly.

That inverts where the difficulty sits:

* ``MolmoWebPolicy`` has to translate an external vocabulary (percent
  coordinates, ``send_msg_to_user`` with ``[ANSWER]`` prefixes, ``keyboard_type``)
  into our actions, and every mismatch is a source of silent error.
* ``QwenPolicy`` asks for ``actions.py``'s own JSON shape, so validation is
  literally ``ACTION_ADAPTER`` -- the same discriminated union the rest of the
  system uses. Unknown action types and unexpected keys are rejected by Pydantic
  for free, because ``ActionBase`` sets ``extra="forbid"``.

COORDINATE CONVENTION (adapter-level, as with MolmoWeb's percent-space)
=======================================================================
**0-1000 normalized, as an ``[x, y]`` pair. Converted to viewport pixels here.**

This was originally written to ask for absolute pixels, on the reasoning that
the model sees the screenshot at its true size. The grounding gate proved that
wrong on the first probe, and the way it failed is worth recording.

Asked for ``{"type": "click", "x": <px>, "y": <px>}``, the model produced things
like ``{"type": "click", "x": [112, 47]}`` -- cramming a coordinate *pair* into
the ``x`` field -- and ``{"type": "click", "x": 904, 45}``, which is not even
valid JSON. It was breaking the requested schema in order to emit a point pair.

Scoring those numbers against the targets showed why: every one of them matched
the target's true centre to within 0.6% *once divided by 1000 and multiplied by
the viewport*. Qwen-VL is trained to emit points in a 0-1000 normalized space,
and that prior is stronger than the instruction in a system prompt. The model's
grounding was never the problem -- the units were.

So the wire format follows the model rather than fighting it, and this adapter
converts, exactly as ``MolmoWebPolicy`` converts its 0-100 percentages:

    x_px = x_norm / 1000 * screenshot_width
    y_px = y_norm / 1000 * screenshot_height

One rule stays load-bearing either way: **the screenshot is never resized
before being sent.** The conversion above uses the dimensions of the image the
model actually saw, so a silent downscale elsewhere would corrupt every
coordinate. If image cost ever forces a resize, undo it here and nowhere else.

Every spatial number uses this one space -- click points and scroll deltas
alike -- because mixed units within a single schema is exactly the kind of
ambiguity that produced the failure above.

COST AND FAILURE
================
This is a metered third-party API paid for by someone else, not a local server,
so the adapter is deliberately defensive: a shared per-run call budget that
aborts rather than looping, retries with backoff only on 429/5xx, and token
accounting surfaced into run.json.

That transport now lives in ``multi_agent_web/ppapi.py`` and is shared with the
Phase 3 manager and judge, which talk to the same API. The wire contract is
unchanged; ``CallBudget`` and friends are re-exported here so existing imports
keep working. What stays in this file is the part that is actually about
*Qwen as a policy*: the prompt, the 0-1000 coordinate space, and the parser.

Parsing keeps the same discipline as ``MolmoWebPolicy``: fail loudly with the
full raw generation attached, no coercion, no default action. The one tolerance
added here is markdown fences -- general models wrap JSON in ```json blocks
regardless of instructions, so a fence is stripped and the strip is logged.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from collections.abc import Sequence
from typing import Any

import httpx
from PIL import Image
from pydantic import ValidationError

from ..actions import ACTION_ADAPTER, Action, Click, Scroll
from ..browser import PageInfo
from ..config import QwenConfig, RunConfig
from ..ppapi import (
    CallBudget,
    CallBudgetExceeded,
    PPAPIClient,
    PPAPIError,
    extract_message_text,
    strip_code_fence,
)
from ..trajectory import Step
from .base import AgentPolicy

logger = logging.getLogger(__name__)

#: Qwen-VL emits points on a 0-1000 grid. See the module docstring.
NORMALIZED_SCALE = 1000.0


def _norm_to_px(value: float, dim: int, clamp: bool) -> float:
    """0-1000 -> pixels against the screenshot the model was shown.

    Positions are clamped into ``[0, dim - 1]`` so that an edge prediction of
    exactly 1000 stays inside the viewport instead of being rejected as
    out-of-bounds. That is part of the documented mapping, not a correction of
    the model -- distances (scroll deltas) are never clamped, since multi-screen
    and negative scrolls are both legal.
    """
    px = round(value / NORMALIZED_SCALE * dim, 1)
    if clamp:
        return max(0.0, min(px, dim - 1.0))
    return px

SYSTEM_PROMPT = """\
You are a web browsing agent. You see a screenshot of a browser viewport and \
choose ONE action to take next, working towards the user's goal.

COORDINATES. Every position and distance you give is normalized to a 0-1000 \
scale, NOT pixels. x=0 is the left edge and x=1000 the right edge; y=0 is the \
top edge and y=1000 the bottom edge. A point is always a two-element array \
[x, y]. The centre of the screen is [500, 500]. Click the CENTRE of the element \
you intend to hit.

Reply with a single JSON object and NOTHING else. No markdown, no code fence, \
no commentary before or after. The object has exactly two keys:

  "thought": one or two sentences on what you see and why you are acting.
  "action": one of the actions below, exactly as specified.

The available actions are:

  {{"type": "click", "point": [x, y]}}
      Left-click at a point. Use this to press buttons, follow links, and to \
focus a text field before typing.

  {{"type": "type", "text": "<string>", "press_enter": <true|false>}}
      Type text into whatever currently has keyboard focus. Click the field \
first. Set press_enter true to submit, e.g. for a search box.

  {{"type": "scroll", "delta": [dx, dy]}}
      Scroll by a distance on the same 0-1000 scale. Positive dy scrolls DOWN, \
negative scrolls UP; positive dx scrolls RIGHT. [0, 1000] moves down exactly \
one full screen.

  {{"type": "key_press", "key": "<string>"}}
      Press a single key or chord. Examples: "Enter", "Escape", "Tab", \
"ArrowDown", "PageDown", "Control+A".

  {{"type": "navigate", "url": "<string>"}}
      Go straight to a URL. Include the scheme, e.g. "https://example.com".

  {{"type": "wait", "seconds": <number>}}
      Wait for a slow page to finish loading. Use sparingly.

  {{"type": "done", "answer": "<string>"}}
      The goal is achieved. Put the answer to the user's question in "answer". \
If the goal needed no textual answer, use a short confirmation.

Rules:
  - Exactly ONE action per reply.
  - Use only the keys shown. Any extra key is an error.
  - Do not invent action types.
  - Points and deltas are ALWAYS two-element arrays on the 0-1000 scale.
  - Base your decision on what is actually visible in the screenshot. If the \
element you want is not visible, scroll to find it rather than guessing.

Example of a well-formed reply:
{{"thought": "The blue Submit button is just below the form, slightly right of \
centre.", "action": {{"type": "click", "point": [532, 674]}}}}
"""


class QwenProtocolError(PPAPIError):
    """The model's reply did not match the requested schema.

    Carries the full raw generation so a mismatch is diagnosable from the log
    line alone. Subclasses ``PPAPIError`` so that ``except PPAPIError`` catches
    every way a call to this API can fail -- transport, HTTP status, and schema
    alike -- while code that only cares about *this* adapter's parsing can still
    name the narrower type.
    """


def _require_point(value: Any, field_name: str, raw: str) -> tuple[float, float]:
    """A two-element numeric array on the 0-1000 scale."""
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value)
    ):
        raise QwenProtocolError(
            f"{field_name!r} must be a two-element array of numbers on the "
            f"0-1000 scale, got {value!r}.\nRaw generation: {raw}"
        )
    return float(value[0]), float(value[1])


def _translate_action(
    action_json: dict[str, Any], size: tuple[int, int], raw: str
) -> Action:
    """Model-space action dict -> one of our pixel-space actions.

    Only ``click`` and ``scroll`` carry spatial values and so need translating;
    every other action already matches ``actions.py`` exactly and is validated
    by ``ACTION_ADAPTER``, the same discriminated union the rest of the system
    runs on. Keeping the untranslated majority on the shared adapter means the
    vocabulary cannot drift out of sync with ``actions.py``.
    """
    width, height = size
    kind = action_json.get("type")

    if kind == "click":
        unexpected = set(action_json) - {"type", "point"}
        if unexpected:
            raise QwenProtocolError(
                f"'click' carries unexpected key(s) {sorted(unexpected)}; expected "
                f"exactly 'type' and 'point'.\nRaw generation: {raw}"
            )
        if "point" not in action_json:
            raise QwenProtocolError(
                f"'click' is missing required key 'point'.\nRaw generation: {raw}"
            )
        nx, ny = _require_point(action_json["point"], "point", raw)
        return Click(
            x=_norm_to_px(nx, width, clamp=True),
            y=_norm_to_px(ny, height, clamp=True),
        )

    if kind == "scroll":
        unexpected = set(action_json) - {"type", "delta"}
        if unexpected:
            raise QwenProtocolError(
                f"'scroll' carries unexpected key(s) {sorted(unexpected)}; expected "
                f"exactly 'type' and 'delta'.\nRaw generation: {raw}"
            )
        if "delta" not in action_json:
            raise QwenProtocolError(
                f"'scroll' is missing required key 'delta'.\nRaw generation: {raw}"
            )
        dx, dy = _require_point(action_json["delta"], "delta", raw)
        # Deltas are unclamped: multi-screen and negative scrolls are legal.
        return Scroll(
            delta_x=_norm_to_px(dx, width, clamp=False),
            delta_y=_norm_to_px(dy, height, clamp=False),
        )

    try:
        return ACTION_ADAPTER.validate_python(action_json)
    except ValidationError as exc:
        raise QwenProtocolError(
            f"'action' does not match our action schema: {exc}."
            f"\nRaw generation: {raw}"
        ) from exc


def parse_generation(raw: str, size: tuple[int, int]) -> tuple[str, Action]:
    """Parse one reply into ``(thought, action)``.

    ``size`` is ``(width, height)`` of the screenshot the model was shown --
    normalized coordinates are relative to that, never to a viewport constant.
    """
    if not raw or not raw.strip():
        raise QwenProtocolError("model returned an empty reply")

    text, fenced = strip_code_fence(raw)
    if fenced:
        logger.info("stripped a markdown code fence from the model reply")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise QwenProtocolError(
            f"reply is not valid JSON ({exc}).\nRaw generation: {raw}"
        ) from exc

    if not isinstance(payload, dict):
        raise QwenProtocolError(
            f"expected a JSON object, got {type(payload).__name__}."
            f"\nRaw generation: {raw}"
        )

    unexpected = set(payload) - {"thought", "action"}
    if unexpected:
        raise QwenProtocolError(
            f"reply has unexpected top-level key(s) {sorted(unexpected)}; "
            f"expected exactly 'thought' and 'action'.\nRaw generation: {raw}"
        )
    if "action" not in payload:
        raise QwenProtocolError(
            f"reply has no 'action' key.\nRaw generation: {raw}"
        )

    thought = payload.get("thought", "")
    if not isinstance(thought, str):
        raise QwenProtocolError(
            f"'thought' must be a string, got {type(thought).__name__}."
            f"\nRaw generation: {raw}"
        )

    action_json = payload["action"]
    if not isinstance(action_json, dict):
        raise QwenProtocolError(
            f"'action' must be an object, got {type(action_json).__name__}."
            f"\nRaw generation: {raw}"
        )

    return thought, _translate_action(action_json, size, raw)


def build_system_prompt(width: int, height: int) -> str:
    return SYSTEM_PROMPT.format(width=width, height=height)


def build_user_message(
    task: str, history: Sequence[dict[str, Any]], page: PageInfo
) -> str:
    """Task, then prior steps as text, then the current page.

    History is text-only: only the current screenshot is sent, following the
    same ``max_past_images=0`` convention MolmoWeb's client uses. Images
    dominate the token bill on a metered API, and past screenshots add little
    once the corresponding thought and action are written down.
    """
    lines = [f"# GOAL\n{task}\n", "# PREVIOUS STEPS"]
    if history:
        for entry in history:
            lines.append(
                f"## Step {entry['index']}\n"
                f"THOUGHT: {entry['thought']}\n"
                f"ACTION: {json.dumps(entry['action'], ensure_ascii=False)}"
                + (f"\nRESULT: {entry['error']}" if entry.get("error") else "")
            )
    else:
        lines.append("(none yet -- this is the first step)")
    lines.append(
        f"\n# CURRENT PAGE\n{page.title} | {page.url}\n\n"
        "# NEXT STEP\nReply with the JSON object for your next action."
    )
    return "\n".join(lines)


class QwenPolicy(AgentPolicy):
    """Drives a Qwen vision model over an OpenAI-compatible HTTP API.

    Stateful, like ``MolmoWebPolicy``: it keeps its own history of thoughts and
    actions to render into each prompt. One instance per agent -- the
    orchestrator enforces that by taking a factory.
    """

    name = "qwen"

    def __init__(
        self,
        config: QwenConfig,
        budget: CallBudget | None = None,
        run_config: RunConfig | None = None,
        agent_index: int | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self.run_config = run_config or RunConfig()
        # The transport, the retry policy and the spend ledger all live on the
        # shared client now. A private budget when unshared, so a single-agent
        # run is still capped.
        self.api = PPAPIClient(
            config,
            budget=budget or CallBudget(limit=config.max_calls_per_run),
            agent_index=agent_index,
            client=client,
        )
        self.agent_index = agent_index
        self._history: list[dict[str, Any]] = []

    # --- lifecycle ---------------------------------------------------------

    @property
    def budget(self) -> CallBudget:
        """The shared ledger. Kept as a property: callers reach for
        ``policy.budget`` and should not have to know where it moved to."""
        return self.api.budget

    async def reset(self) -> None:
        self._history = []

    async def close(self) -> None:
        await self.api.close()

    # --- inference ---------------------------------------------------------

    @staticmethod
    def _data_url(screenshot: Image.Image) -> str:
        """PNG data URL. The API rejects a bare base64 string with HTTP 400."""
        buf = io.BytesIO()
        screenshot.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    async def predict(
        self,
        task: str,
        history: Sequence[Step],  # unused: we keep our own, see class docstring
        screenshot: Image.Image,
        page: PageInfo,
    ) -> Step:
        width, height = screenshot.size
        window = self._history[-self.config.max_past_steps:]

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": build_system_prompt(width, height)},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_user_message(task, window, page)},
                        {
                            "type": "image_url",
                            # Native size: resizing would move every coordinate
                            # out of viewport space. See the module docstring.
                            "image_url": {"url": self._data_url(screenshot)},
                        },
                    ],
                },
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }

        body = await self.api.chat(payload)
        raw = extract_message_text(body)

        logger.debug("raw generation: %s", raw)
        thought, action = parse_generation(raw, screenshot.size)

        self._history.append(
            {
                "index": len(self._history) + 1,
                "thought": thought,
                "action": action.model_dump(mode="json"),
            }
        )
        return Step(thought=thought, action=action)

    def note_action_error(self, error: str) -> None:
        """Attach an execution failure to the last step, for the next prompt.

        Lets the model see that its click missed or its navigation failed,
        instead of repeating the same action blindly.
        """
        if self._history:
            self._history[-1]["error"] = error


#: ``CallBudget``, ``CallBudgetExceeded`` and ``strip_code_fence`` now live in
#: ``multi_agent_web.ppapi`` and are re-exported here: they were part of this
#: module's interface before the transport was shared, and moving an import
#: path is churn with no payoff.
__all__ = [
    "CallBudget",
    "CallBudgetExceeded",
    "QwenPolicy",
    "QwenProtocolError",
    "build_system_prompt",
    "build_user_message",
    "parse_generation",
    "strip_code_fence",
]
