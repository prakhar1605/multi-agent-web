"""Run configuration for a single browsing agent.

Everything that a run needs to know about *how* to browse (as opposed to *what*
to browse) lives here, so a future manager can hand each parallel agent its own
``RunConfig`` without any global state.
"""

from __future__ import annotations

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
    # Default scroll distance in pixels when an action does not specify one.
    default_scroll_amount: int = Field(default=400, gt=0)

    # --- Logging -----------------------------------------------------------
    runs_dir: Path = Path("runs")
    save_screenshots: bool = True

    @property
    def viewport(self) -> dict[str, int]:
        """Viewport in the shape Playwright expects."""
        return {"width": self.viewport_width, "height": self.viewport_height}

    def contains_point(self, x: int, y: int) -> bool:
        """True if (x, y) falls inside the viewport coordinate space."""
        return 0 <= x < self.viewport_width and 0 <= y < self.viewport_height
