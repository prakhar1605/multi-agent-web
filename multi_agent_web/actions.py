"""The action space: what an agent is allowed to do to a browser.

COORDINATE CONVENTION (read this before writing a policy)
=========================================================

All coordinates in this module are **absolute pixels in the browser viewport's
coordinate space**:

    * Origin ``(0, 0)`` is the TOP-LEFT corner of the viewport.
    * ``x`` grows to the RIGHT, ``y`` grows DOWNWARD.
    * Units are CSS pixels. Valid range is ``0 <= x < viewport_width`` and
      ``0 <= y < viewport_height`` (1280x720 by default, see ``RunConfig``).
    * Coordinates are relative to the VIEWPORT, not the document. Scrolling
      down by 400px does not shift the coordinates of the visible content;
      it changes *which* content is visible. There is no "page space" here.
    * The browser is created with ``device_scale_factor = 1``, so one pixel in
      the screenshot handed to the policy is exactly one pixel in this space.
      A click at ``(640, 360)`` lands on whatever the policy sees at pixel
      ``(640, 360)`` of that screenshot. This 1:1 mapping is the whole contract
      between the observation and the action.

Why absolute pixels, and not normalized [0, 1] or [0, 100] coordinates?
-----------------------------------------------------------------------
Vision-language models disagree about this. Molmo-family models are trained to
emit *normalized* points (commonly 0-100), other web agents emit raw pixels,
and some emit element indices from an accessibility tree. Rather than bake one
model's convention into the shared action space, this module fixes ONE
convention -- absolute viewport pixels -- and makes every policy responsible
for converting its own output into it.

Concretely, a policy for a model that emits 0-100 normalized points does:

    x_px = round(x_norm / 100 * config.viewport_width)
    y_px = round(y_norm / 100 * config.viewport_height)

That conversion belongs in ``policy/<model>.py``, never here. It keeps
``browser.py`` model-agnostic and means swapping the policy cannot silently
change what a coordinate means.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union
from urllib.parse import urlparse

from pydantic import BaseModel, Field, TypeAdapter, field_validator

# Schemes we are willing to hand to ``page.goto``. Blocking everything else
# keeps a hallucinating model from poking at ``javascript:`` or local schemes.
ALLOWED_URL_SCHEMES = {"http", "https", "file", "about"}


class ActionBase(BaseModel):
    """Common behaviour for every action.

    ``model_config`` forbids extra fields so that a malformed model output
    (e.g. ``{"type": "click", "x": 1, "y": 2, "selector": "#foo"}``) fails loudly
    at parse time rather than being silently half-executed.
    """

    model_config = {"extra": "forbid"}

    def summary(self) -> str:
        """Short human-readable form, used in logs and trajectory viewers."""
        return self.type  # type: ignore[attr-defined]

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.summary()


class Click(ActionBase):
    """Left-click at a point in the viewport. See module docstring for units."""

    type: Literal["click"] = "click"
    x: int = Field(ge=0, description="Pixels from the left edge of the viewport.")
    y: int = Field(ge=0, description="Pixels from the top edge of the viewport.")

    def summary(self) -> str:
        return f"click({self.x}, {self.y})"


class Type(ActionBase):
    """Type text into whatever element currently has keyboard focus.

    There is deliberately no selector here: this is a pixel-first action space,
    so the intended sequence is ``Click`` (to focus a field) then ``Type``.
    """

    type: Literal["type"] = "type"
    text: str
    press_enter: bool = Field(
        default=False,
        description="Press Enter after typing, e.g. to submit a search box.",
    )

    def summary(self) -> str:
        suffix = " + Enter" if self.press_enter else ""
        preview = self.text if len(self.text) <= 40 else self.text[:37] + "..."
        return f"type({preview!r}){suffix}"


class Scroll(ActionBase):
    """Scroll the page by a pixel amount in one of four directions."""

    type: Literal["scroll"] = "scroll"
    direction: Literal["up", "down", "left", "right"] = "down"
    amount: int | None = Field(
        default=None,
        gt=0,
        description="Distance in pixels. None means RunConfig.default_scroll_amount.",
    )

    def summary(self) -> str:
        amount = self.amount if self.amount is not None else "default"
        return f"scroll({self.direction}, {amount})"


class KeyPress(ActionBase):
    """Press a single key or chord, using Playwright's key names.

    Examples: ``Enter``, ``Escape``, ``Tab``, ``ArrowDown``, ``PageDown``,
    ``Control+A``, ``Meta+C``.
    """

    type: Literal["key_press"] = "key_press"
    key: str = Field(min_length=1)

    def summary(self) -> str:
        return f"key({self.key})"


class Navigate(ActionBase):
    """Load a URL directly, bypassing clicks."""

    type: Literal["navigate"] = "navigate"
    url: str = Field(min_length=1)

    @field_validator("url")
    @classmethod
    def _check_scheme(cls, value: str) -> str:
        scheme = urlparse(value).scheme.lower()
        if scheme not in ALLOWED_URL_SCHEMES:
            raise ValueError(
                f"URL scheme {scheme!r} is not allowed; "
                f"expected one of {sorted(ALLOWED_URL_SCHEMES)}. "
                "Include an explicit scheme, e.g. https://example.com"
            )
        return value

    def summary(self) -> str:
        return f"navigate({self.url})"


class Wait(ActionBase):
    """Idle for a while, e.g. while a slow page finishes rendering."""

    type: Literal["wait"] = "wait"
    seconds: float = Field(default=1.0, gt=0, le=30.0)

    def summary(self) -> str:
        return f"wait({self.seconds}s)"


class Done(ActionBase):
    """Terminate the episode and report an answer.

    This is the only action the agent loop treats as terminal.
    """

    type: Literal["done"] = "done"
    answer: str = ""

    def summary(self) -> str:
        preview = self.answer if len(self.answer) <= 60 else self.answer[:57] + "..."
        return f"done({preview!r})"


# Discriminated union: pydantic reads the ``type`` field first and dispatches to
# exactly one model, so validation errors point at the right action instead of
# reporting seven unrelated failures.
Action = Annotated[
    Union[Click, Type, Scroll, KeyPress, Navigate, Wait, Done],
    Field(discriminator="type"),
]

# Use this to parse an action out of raw JSON/dict, e.g. when replaying a saved
# trajectory or decoding a model's structured output:
#     ACTION_ADAPTER.validate_python({"type": "click", "x": 10, "y": 20})
ACTION_ADAPTER: TypeAdapter[Action] = TypeAdapter(Action)

__all__ = [
    "Action",
    "ACTION_ADAPTER",
    "ActionBase",
    "ALLOWED_URL_SCHEMES",
    "Click",
    "Done",
    "KeyPress",
    "Navigate",
    "Scroll",
    "Type",
    "Wait",
]
