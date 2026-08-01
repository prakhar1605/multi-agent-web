"""MolmoWeb-4B policy: the adapter between the model's wire format and our actions.

The full wire format is documented in ``docs/molmoweb_format.md``, verified
against the public ``allenai/MolmoWeb-SyntheticTrajs`` training data. In short,
the model emits ONE JSON object per step::

    {"thought": "...", "action": {"name": "click", "x": 71.1, "y": 3.1,
                                  "button": "left", "click_type": "single"}}

with coordinates as **percentages 0-100 of the screenshot dimensions**.

PROVENANCE
----------
The prompt template text, the percent->pixel clamp semantics, and the action
vocabulary are derived from ``allenai/molmoweb`` (Apache License 2.0):

  * prompt template          -> ``train/olmo/models/molmo/data_formatter.py:152-169``
                                (identical to ``agent/multimodal_agent.py:42-59``)
  * ``molmo_web_think:`` prefix -> ``agent/multimodal_agent.py:229``
  * percent -> pixel + clamp -> ``agent/multimodal_agent.py:62-70``
  * action vocabulary        -> ``agent/multimodal_agent.py:73-147``
  * title/url truncation     -> ``agent/multimodal_agent.py:150-159``
  * HTTP server contract     -> ``agent/fastapi_model_server.py:70-108``

No files were copied; this is an independent implementation of the same wire
format against our own action space.

DESIGN
------
This adapter FAILS LOUDLY. An unknown action name, an unexpected key, a
non-``left`` mouse button, a malformed payload -- all raise, with the full raw
generation in the message. Nothing is coerced and nothing falls back to a
default action. If the format spec is wrong, that shows up as a readable error
on step 1, not as a mystery misclick on step 5. The agent loop records the
error on the step and aborts after ``max_consecutive_errors``, so a broken
protocol assumption cannot quietly produce a plausible-looking trajectory.

The one exception is coordinate clamping, which is *conversion*, not repair:
the reference implementation defines the percent->pixel mapping as clamped to
``[1, dim-2]``, and real training data contains clipped values like ``"y": 100``
that map to exactly ``dim``. Honouring the clamp is fidelity to the model's
contract; snapping a click to a nearby element would be repair, and we don't
do that.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from collections import Counter
from collections.abc import Sequence
from typing import Any

import httpx
from PIL import Image

from ..actions import Action, Click, Done, KeyPress, Navigate, Scroll, Type, Wait
from ..browser import PageInfo
from ..config import MolmoWebConfig
from ..trajectory import Step
from .base import AgentPolicy

logger = logging.getLogger(__name__)

# ``NOOP_WAIT_MS`` in the reference implementation (agent/actions.py:7).
NOOP_WAIT_SECONDS = 5.0

# Titles and URLs are truncated before going into the prompt
# (agent/multimodal_agent.py:150-159, max_len=100).
_TRUNCATE_LEN = 100
_TRUNCATE_POSTFIX = "... (truncated)"

# Action names we can execute, mapped to the complete set of keys we accept.
# Any key outside its set is a protocol error, not something to ignore.
_SUPPORTED_KEYS: dict[str, frozenset[str]] = {
    "click": frozenset({"name", "x", "y", "button", "click_type"}),
    "mouse_click": frozenset({"name", "x", "y", "button", "click_type"}),
    "keyboard_type": frozenset({"name", "text"}),
    "type": frozenset({"name", "text"}),
    "keyboard_press": frozenset({"name", "key"}),
    "keypress": frozenset({"name", "key"}),
    "scroll": frozenset({"name", "delta_x", "delta_y"}),
    "goto": frozenset({"name", "url"}),
    "noop": frozenset({"name", "noop_reason"}),
    "send_msg_to_user": frozenset({"name", "msg"}),
}

# Actions the model can emit that our action space does not model yet. Naming
# them explicitly gives a better error than "unknown action", and logging them
# reveals which ones the model actually reaches for in practice.
_KNOWN_UNMODELLED: dict[str, str] = {
    "hover_at": "mouse hover at a point",
    "drag_and_drop": "drag from one point to another",
    "mouse_drag_and_drop": "drag from one point to another",
    "scroll_at": "scroll with the cursor parked at a point",
    "browser_nav": "go_back / new_tab / tab_focus",
    "report_infeasible": "declare the task impossible",
    "dblclick": "double click",
    "gemini_type_text_at": "Gemini-CUA compatibility action (coords are /10)",
}

_TOP_LEVEL_KEYS = frozenset({"thought", "action", "action_description"})


class MolmoWebProtocolError(RuntimeError):
    """The model's output did not match the documented wire format.

    Always carries the full raw generation so the mismatch is diagnosable from
    the log line alone.
    """


def _truncate(text: str, max_len: int = _TRUNCATE_LEN) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - len(_TRUNCATE_POSTFIX)] + _TRUNCATE_POSTFIX


def _pct_to_px(pct: float, dim: int) -> float:
    """Percent (0-100) -> pixels. Used for scroll deltas: signed, unclamped."""
    return round(pct / 100.0 * dim, 1)


def _pct_to_coord(pct: float, dim: int) -> float:
    """Percent (0-100) -> pixels, clamped to [1, dim-2].

    The clamp is the reference implementation's definition of this mapping
    (``agent/multimodal_agent.py:66-70``), not a correction of the model. Real
    training targets contain clipped values -- a click below the fold serializes
    as ``"y": 100`` -- which would otherwise convert to exactly ``dim`` and be
    rejected as outside the viewport.
    """
    px = round(pct / 100.0 * dim, 1)
    return max(1.0, min(px, dim - 2.0))


def _require_number(value: Any, field: str, raw: str) -> float:
    """Coordinates arrive as float OR int -- clipped values serialize as ints."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MolmoWebProtocolError(
            f"expected a number for {field!r}, got {type(value).__name__} "
            f"{value!r}.\nRaw generation: {raw}"
        )
    return float(value)


def _require_str(value: Any, field: str, raw: str) -> str:
    if not isinstance(value, str):
        raise MolmoWebProtocolError(
            f"expected a string for {field!r}, got {type(value).__name__} "
            f"{value!r}.\nRaw generation: {raw}"
        )
    return value


def translate_action(
    action_json: dict[str, Any],
    screenshot_size: tuple[int, int],
    raw: str,
) -> Action:
    """Convert one model action dict into one of our actions.

    ``screenshot_size`` is ``(width, height)`` of the image the model actually
    saw -- percentages are relative to that, never to a viewport constant and
    never to the model's internal preprocessed size.

    Raises ``MolmoWebProtocolError`` for anything malformed or unknown, and
    ``NotImplementedError`` for actions the model supports but we do not model.
    """
    width, height = screenshot_size
    name = action_json.get("name")

    if not isinstance(name, str):
        raise MolmoWebProtocolError(
            f"action has no string 'name' key: {action_json!r}\nRaw generation: {raw}"
        )

    if name in _KNOWN_UNMODELLED:
        raise NotImplementedError(
            f"MolmoWeb emitted {name!r} ({_KNOWN_UNMODELLED[name]}), which our "
            f"action space does not model yet. Add it to actions.py and to "
            f"translate_action() rather than approximating it.\nRaw generation: {raw}"
        )

    allowed = _SUPPORTED_KEYS.get(name)
    if allowed is None:
        raise MolmoWebProtocolError(
            f"unknown action name {name!r}. Known names are "
            f"{sorted(set(_SUPPORTED_KEYS) | set(_KNOWN_UNMODELLED))}."
            f"\nRaw generation: {raw}"
        )

    unexpected = set(action_json) - allowed
    if unexpected:
        raise MolmoWebProtocolError(
            f"action {name!r} carries unexpected key(s) {sorted(unexpected)}; "
            f"expected a subset of {sorted(allowed)}. Refusing to guess what "
            f"they mean.\nRaw generation: {raw}"
        )

    if name in ("click", "mouse_click"):
        for required in ("x", "y"):
            if required not in action_json:
                raise MolmoWebProtocolError(
                    f"{name!r} is missing required key {required!r}."
                    f"\nRaw generation: {raw}"
                )
        button = action_json.get("button", "left")
        click_type = action_json.get("click_type", "single")
        if button != "left":
            raise NotImplementedError(
                f"MolmoWeb requested a {button!r}-button click; our Click action "
                f"only models left-click.\nRaw generation: {raw}"
            )
        if click_type != "single":
            raise NotImplementedError(
                f"MolmoWeb requested click_type={click_type!r}; our Click action "
                f"only models single clicks.\nRaw generation: {raw}"
            )
        return Click(
            x=_pct_to_coord(_require_number(action_json["x"], "x", raw), width),
            y=_pct_to_coord(_require_number(action_json["y"], "y", raw), height),
        )

    if name in ("keyboard_type", "type"):
        if "text" not in action_json:
            raise MolmoWebProtocolError(
                f"{name!r} is missing required key 'text'.\nRaw generation: {raw}"
            )
        # MolmoWeb's keyboard_type never submits; a separate keyboard_press
        # carries Enter. Hardcoding False keeps that faithful.
        return Type(text=_require_str(action_json["text"], "text", raw), press_enter=False)

    if name in ("keyboard_press", "keypress"):
        if "key" not in action_json:
            raise MolmoWebProtocolError(
                f"{name!r} is missing required key 'key'.\nRaw generation: {raw}"
            )
        return KeyPress(key=_require_str(action_json["key"], "key", raw))

    if name == "scroll":
        for required in ("delta_x", "delta_y"):
            if required not in action_json:
                raise MolmoWebProtocolError(
                    f"'scroll' is missing required key {required!r}."
                    f"\nRaw generation: {raw}"
                )
        # Deltas are percentages of the corresponding axis: delta_y=100 is
        # exactly one viewport height. Unclamped -- multi-viewport scrolls and
        # negative values are both legal.
        return Scroll(
            delta_x=_pct_to_px(_require_number(action_json["delta_x"], "delta_x", raw), width),
            delta_y=_pct_to_px(_require_number(action_json["delta_y"], "delta_y", raw), height),
        )

    if name == "goto":
        if "url" not in action_json:
            raise MolmoWebProtocolError(
                f"'goto' is missing required key 'url'.\nRaw generation: {raw}"
            )
        return Navigate(url=_require_str(action_json["url"], "url", raw))

    if name == "noop":
        return Wait(seconds=NOOP_WAIT_SECONDS)

    if name == "send_msg_to_user":
        if "msg" not in action_json:
            raise MolmoWebProtocolError(
                f"'send_msg_to_user' is missing required key 'msg'."
                f"\nRaw generation: {raw}"
            )
        msg = _require_str(action_json["msg"], "msg", raw)
        for sentinel in ("[ANSWER]", "[EXIT]"):
            if msg.startswith(sentinel):
                return Done(answer=msg[len(sentinel):].strip(), sentinel=sentinel)
        raise MolmoWebProtocolError(
            "send_msg_to_user without an [ANSWER] or [EXIT] prefix. Every "
            "training example uses one, so a bare message means our format "
            "understanding is wrong. If prefix-less messages turn out to be "
            "normal and non-terminal, model them as their own action rather "
            f"than treating this as Done.\nRaw generation: {raw}"
        )

    raise MolmoWebProtocolError(  # pragma: no cover - guarded by _SUPPORTED_KEYS
        f"no translation implemented for {name!r}.\nRaw generation: {raw}"
    )


def parse_generation(
    raw: str, screenshot_size: tuple[int, int]
) -> tuple[str, Action, dict[str, Any]]:
    """Parse one raw generation into ``(thought, action, raw_action_dict)``.

    The third element is the model's own percent-space action dict, kept
    verbatim so it can be rendered back into the prompt as history. Re-deriving
    percentages from our pixel-space action would double-round and drift.
    """
    text = raw.strip()
    if not text:
        raise MolmoWebProtocolError("model returned an empty generation")

    # The server returns its own errors as plain strings
    # (agent/fastapi_model_server.py:86, :101).
    if text.startswith("Predictor error:"):
        raise MolmoWebProtocolError(f"model server reported: {text}")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MolmoWebProtocolError(
            f"generation is not valid JSON ({exc}).\nRaw generation: {raw}"
        ) from exc

    if not isinstance(payload, dict):
        raise MolmoWebProtocolError(
            f"expected a JSON object, got {type(payload).__name__}."
            f"\nRaw generation: {raw}"
        )

    if "action" in payload:
        unexpected = set(payload) - _TOP_LEVEL_KEYS
        if unexpected:
            raise MolmoWebProtocolError(
                f"generation has unexpected top-level key(s) {sorted(unexpected)}; "
                f"expected a subset of {sorted(_TOP_LEVEL_KEYS)}."
                f"\nRaw generation: {raw}"
            )
        action_json = payload["action"]
        thought = payload.get("thought", "")
    elif "name" in payload:
        # molmo_web_base style: the whole object is the action, no thought.
        action_json = payload
        thought = ""
    else:
        raise MolmoWebProtocolError(
            "generation has neither an 'action' key nor a bare 'name' key."
            f"\nRaw generation: {raw}"
        )

    if not isinstance(action_json, dict):
        raise MolmoWebProtocolError(
            f"'action' must be an object, got {type(action_json).__name__}."
            f"\nRaw generation: {raw}"
        )
    if not isinstance(thought, str):
        raise MolmoWebProtocolError(
            f"'thought' must be a string, got {type(thought).__name__}."
            f"\nRaw generation: {raw}"
        )

    action = translate_action(action_json, screenshot_size, raw)
    return thought, action, action_json


def build_user_message(
    task: str,
    past_actions: Sequence[dict[str, Any]],
    page: PageInfo,
    page_index: int = 0,
) -> str:
    """Render the user message, byte-identical to the reference template.

    Reproduces ``MOLMOWEB_THINK_TEMPLATE`` (``data_formatter.py:152-169``)
    without a Jinja2 dependency. The whitespace here is load-bearing -- it is
    what the model was trained on -- so ``tests/test_molmoweb_format.py`` pins
    it against a prompt reconstructed from real training data.

    ``past_actions`` entries are ``{"index", "thought", "action"}`` where
    ``action`` is the model's own percent-space dict. Jinja renders a dict with
    ``str()``, so history lines carry **Python repr with single quotes**, not
    JSON. That is what training used; do not "fix" it.
    """
    parts = [f"\n# GOAL\n{task}\n\n# PREVIOUS STEPS\n"]
    for entry in past_actions:
        parts.append(
            f"## Step {entry['index']}\n"
            f"THOUGHT: {entry['thought']}\n"
            f"ACTION: {entry['action']}\n"
        )
    parts.append(
        f"\n# CURRENTLY ACTIVE PAGE\n"
        f"Page {page_index}: {_truncate(page.title)} | {_truncate(page.url)}\n"
        f"\n# NEXT STEP\n"
    )
    return "".join(parts)


class MolmoWebPolicy(AgentPolicy):
    """Drives MolmoWeb-4B over HTTP against a running model server.

    Stateful by design: it keeps its own ``past_actions`` list of raw
    percent-space dicts, exactly as ``MultimodalAgent`` does
    (``agent/multimodal_agent.py:304-313``). The loop's ``history`` of ``Step``
    objects is in *pixel* space, so rebuilding the prompt from it would require
    converting back to percentages and would drift by a rounding step each time.
    """

    name = "molmoweb"

    def __init__(
        self,
        config: MolmoWebConfig,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self._client = client
        self._owns_client = client is None
        self._past_actions: list[dict[str, Any]] = []
        # Which unmodelled actions the model reached for, for reporting.
        self.unmodelled_seen: Counter[str] = Counter()

    # --- lifecycle ---------------------------------------------------------

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.config.timeout_s)
        return self._client

    @property
    def past_actions(self) -> list[dict[str, Any]]:
        """The model's own percent-space history. Read-only view, for tooling."""
        return list(self._past_actions)

    async def reset(self) -> None:
        self._past_actions = []

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    # --- inference ---------------------------------------------------------

    async def _post(self, prompt: str, screenshot: Image.Image) -> str:
        """POST to ``{endpoint}/predict`` and return the raw generation."""
        buf = io.BytesIO()
        screenshot.save(buf, format="PNG")
        payload: dict[str, Any] = {
            "prompt": prompt,
            "image_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
        }
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            payload["top_p"] = self.config.top_p

        url = f"{self.config.endpoint.rstrip('/')}/predict"
        response = await self._get_client().post(url, json=payload)
        response.raise_for_status()

        body = response.json()
        if not isinstance(body, str):
            raise MolmoWebProtocolError(
                f"expected the server to return a JSON string, got "
                f"{type(body).__name__}: {body!r}"
            )
        return body

    async def predict(
        self,
        task: str,
        history: Sequence[Step],  # unused: see class docstring, we keep our own
        screenshot: Image.Image,
        page: PageInfo,
    ) -> Step:
        window = self._past_actions[-self.config.max_past_steps:]
        user_message = build_user_message(task, window, page)
        prompt = f"{self.config.system_message}: {user_message}"

        raw = await self._post(prompt, screenshot)
        logger.debug("raw generation: %s", raw)

        try:
            thought, action, action_json = parse_generation(raw, screenshot.size)
        except NotImplementedError:
            name = raw_action_name(raw)
            if name:
                self.unmodelled_seen[name] += 1
                logger.warning(
                    "model reached for unmodelled action %r (seen %d time(s) this run)",
                    name,
                    self.unmodelled_seen[name],
                )
            raise

        # History is kept in the model's own coordinate space, never
        # back-converted from our pixel-space Step.
        self._past_actions.append(
            {
                "index": len(self._past_actions) + 1,  # 1-based, as in training
                "thought": thought,
                "action": action_json,
            }
        )
        return Step(thought=thought, action=action)


def raw_action_name(raw: str) -> str | None:
    """Best-effort action name from a raw generation, for logging only."""
    try:
        payload = json.loads(raw.strip())
        source = payload.get("action", payload) if isinstance(payload, dict) else {}
        name = source.get("name") if isinstance(source, dict) else None
        return name if isinstance(name, str) else None
    except Exception:
        return None
