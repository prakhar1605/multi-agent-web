"""Run configuration for a single browsing agent.

Everything that a run needs to know about *how* to browse (as opposed to *what*
to browse) lives here, so a future manager can hand each parallel agent its own
``RunConfig`` without any global state.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    """Knobs for one agent run.

    The viewport size is the single most important setting: it defines the
    coordinate space that the policy predicts in. Change it and every learned
    or hardcoded coordinate changes meaning, so keep it fixed across a
    train/eval setup.
    """

    # --- Observation space -------------------------------------------------
    viewport_width: int = Field(default=1280, gt=0)
    viewport_height: int = Field(default=720, gt=0)

    # Force 1 CSS pixel == 1 screenshot pixel. On a Retina display the default
    # would be 2, and every predicted coordinate would be off by 2x.
    device_scale_factor: float = Field(default=1.0, gt=0)

    # --- Episode limits ----------------------------------------------------
    max_steps: int = Field(default=15, gt=0)
    # Abort the run after this many consecutive failures (policy or action) so
    # a broken page cannot burn the whole step budget in a tight loop.
    max_consecutive_errors: int = Field(default=3, gt=0)

    # --- Browser -----------------------------------------------------------
    headless: bool = True
    # Pause between Playwright operations; useful when watching a headed run.
    slow_mo_ms: int = Field(default=0, ge=0)
    action_timeout_ms: int = Field(default=10_000, gt=0)
    navigation_timeout_ms: int = Field(default=30_000, gt=0)
    # Grace period after each action so the page can react before we screenshot.
    settle_ms: int = Field(default=400, ge=0)

    # --- Logging -----------------------------------------------------------
    runs_dir: Path = Path("runs")
    save_screenshots: bool = True

    @property
    def viewport(self) -> dict[str, int]:
        """Viewport in the shape Playwright expects."""
        return {"width": self.viewport_width, "height": self.viewport_height}

    def contains_point(self, x: float, y: float) -> bool:
        """True if (x, y) falls inside the viewport coordinate space."""
        return 0 <= x < self.viewport_width and 0 <= y < self.viewport_height


class MolmoWebConfig(BaseModel):
    """Connection and prompting settings for the MolmoWeb model server.

    Separate from ``RunConfig`` because it describes the *policy*, not the
    browser -- and because a Phase 2 fleet shares one model server while each
    agent gets its own browser.

    The endpoint is never hardcoded: pass it explicitly or set
    ``MOLMOWEB_ENDPOINT``. The server is expected to be reachable over the
    network, so running the GPU on another machine is just a different URL.
    """

    endpoint: str = Field(
        description="Base URL of the model server, e.g. http://gpu-host:8001. "
        "The adapter POSTs to {endpoint}/predict."
    )
    # "molmo_web_think" is the style tag the model was trained with; it is
    # prepended to the user message as f"{system_message}: {user_message}".
    system_message: str = "molmo_web_think"
    # How many past steps to render into the prompt. The reference client uses
    # 10; MultimodalAgent's own default is 3.
    max_past_steps: int = Field(default=10, gt=0)
    timeout_s: float = Field(default=120.0, gt=0)
    # None means "let the server use its configured default".
    temperature: float | None = None
    top_p: float | None = None

    @classmethod
    def from_env(cls, endpoint: str | None = None, **overrides) -> "MolmoWebConfig | None":
        """Build from ``MOLMOWEB_ENDPOINT`` unless an endpoint is passed.

        Returns ``None`` when no endpoint is configured, so callers can skip
        model-dependent work instead of failing.
        """
        resolved = endpoint or os.environ.get("MOLMOWEB_ENDPOINT", "").strip()
        if not resolved:
            return None
        return cls(endpoint=resolved, **overrides)
