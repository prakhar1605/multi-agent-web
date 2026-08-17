#!/usr/bin/env python3
"""Recording-friendly wrapper: several agents, tiled on screen, at a readable pace.

Presentation only. No orchestrator or policy logic is changed -- this composes
the same public pieces ``orchestrate()`` uses, swapping in a browser pool that
tiles windows and a policy wrapper that narrates.

    # rehearse for free, no API calls
    python scripts/run_demo.py --preset local --policy mock --n 4 --yes

    # the real thing
    python scripts/run_demo.py --preset wikipedia --policy qwen --n 4

    # a manager LLM decomposes the task and runs the graph in waves
    python scripts/run_demo.py --preset bookstore --policy qwen --strategy dag --n 4

``--strategy`` defaults to best_of_n, so every command that worked before still
does exactly what it did.

HOW THE WINDOWS GET TILED
-------------------------
Launch flags do not work, and giving each session its OWN browser would not
rescue them. Playwright resizes every window to fit its context viewport, so
with a pinned viewport ``--window-size`` is discarded. Measured, asking for
``700x500 at (900,400)``:

    context viewport=1280x720 -> window 1282x800  (size flag ignored)
                                 viewport 1280x720, screenshot 1280x720  OK
    context no_viewport=True  -> window  700x500  (size flag honoured)
                                 viewport  700x413, screenshot 1400x826  WRONG

The flag only sticks when the viewport is surrendered, which is the one thing
that may not be traded away: the policy predicts coordinates in that space. So
a per-session browser buys N times the memory and still cannot tile -- every
window comes out identical, which is exactly the stacking being fixed.

Windows are therefore positioned AFTER they exist, over CDP:

    Browser.getWindowForTarget -> windowId
    Browser.setWindowBounds    -> exact left/top/width/height

which is honoured precisely (asked 640x480 at (120,90), got exactly that), and
crucially leaves the viewport alone: that same 640x480 window still reports
``innerWidth/innerHeight`` of 1280x720 and still screenshots at 1280x720.

That separation is the whole point. The **window** is what you see and may be
scaled down so N of them fit on screen. The **viewport** is the coordinate space
the policy predicts in and is never negotiable. ``TiledSession.start`` asserts
the viewport after positioning and aborts loudly if it ever drifts, because a
silently shrunken viewport would misplace every click.

The browser pool keeps the runner's production design -- ONE Chromium, N
contexts, one window each -- so the demo path does not diverge from it.

NOTHING ABOUT THE DISPLAY IS GUESSED
------------------------------------
Three numbers decide the grid, and hardcoding them is what made the first
attempt overlap. All three are now measured at startup by ``probe_display``:

  * the usable screen. Read from a ``no_viewport=True`` context, because a
    normal Playwright context EMULATES ``window.screen`` to match its viewport
    -- a 1280x720 context reports the display as 1280x720 however big it
    really is. Measured here: emulated 1280x720 vs. real 1470x859 at (0,33).
  * the browser chrome. A fixed strip that does NOT shrink with the page, so
    it is added after scaling the viewport, never scaled with it. Measured 80px
    (a hardcoded 130 both over-scaled the grid and mismodelled it).
  * the window floor. The window manager silently clamps below 500x375, so
    asking for 517x343 yields 517x375 and a grid laid out on the requested
    number overlaps by the difference.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collections.abc import Sequence  # noqa: E402
from dataclasses import replace  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

from PIL import Image  # noqa: E402
from playwright.async_api import Browser as PWBrowser  # noqa: E402
from playwright.async_api import Playwright, async_playwright  # noqa: E402

from multi_agent_web.browser import BrowserSession, PageInfo  # noqa: E402
from multi_agent_web.config import RunConfig  # noqa: E402
from multi_agent_web.launch import (  # noqa: E402
    LaunchError,
    LaunchSpec,
    estimate_calls,
    prepare,
)
from multi_agent_web.orchestrator import (  # noqa: E402
    OrchestrationResult,
    Runner,
    write_run_json,
)
from multi_agent_web.orchestrator.runner import summarise_timing  # noqa: E402
from multi_agent_web.policy.base import AgentPolicy  # noqa: E402
from multi_agent_web.presets import DEMO_PAGE, PRESETS  # noqa: E402
from multi_agent_web.trajectory import Step, make_run_dir  # noqa: E402


# ---------------------------------------------------------------------------
# window tiling
# ---------------------------------------------------------------------------
#: cols x rows per agent count, chosen to stay close to square.
#:
#: 3 is deliberately 2x2-with-a-gap rather than 3x1. Squeezing three 1280-wide
#: viewports into one row leaves a 490px cell, which is under the 500px window
#: floor, so a 3x1 cannot be drawn at all on a 1470px screen -- while the same
#: three windows tile comfortably in a 2x2 with one slot empty.
GRID: dict[int, tuple[int, int]] = {
    1: (1, 1), 2: (2, 1), 3: (2, 2), 4: (2, 2), 5: (3, 2), 6: (3, 2),
}


@dataclass(frozen=True)
class Screen:
    """The area a window may actually occupy.

    ``top`` matters more than it looks: the OS will not place a window above the
    menu bar, so asking for y=0 silently yields y=33 on macOS. Laying out from
    y=0 therefore pushes every row down by the inset and makes the bottom row
    collide with the one above -- which the overlap check caught. Rows are
    measured from ``top`` within ``height`` so the grid closes exactly.
    """

    left: int
    top: int
    width: int
    height: int

    def __str__(self) -> str:
        origin = "" if (self.left, self.top) == (0, 0) else f" at ({self.left},{self.top})"
        return f"{self.width}x{self.height}{origin}"


@dataclass(frozen=True)
class ChromeMetrics:
    """What the browser and the window manager impose on any window.

    ``chrome_px`` (title bar + tab strip + address bar) and ``side_px`` are
    fixed costs: they do NOT shrink when the window is scaled down, so they are
    added *after* scaling the viewport rather than scaled along with it.

    ``min_w``/``min_h`` are the window manager's floor. It clamps silently --
    ``setWindowBounds`` returns success and then reports the larger size -- so a
    grid computed from the requested size overlaps by exactly the clamp.
    """

    chrome_px: int
    side_px: int
    min_w: int
    min_h: int

    def __str__(self) -> str:
        return (
            f"{self.chrome_px}px chrome, {self.side_px}px sides, "
            f"{self.min_w}x{self.min_h} window floor"
        )


FALLBACK_SCREEN = Screen(0, 0, 1440, 900)
FALLBACK_CHROME = ChromeMetrics(chrome_px=80, side_px=2, min_w=500, min_h=375)


async def probe_display(viewport: dict[str, int]) -> tuple[Screen, ChromeMetrics]:
    """Measure the screen, the browser chrome and the window floor, for real.

    Every one of these was a hardcoded guess once and each guess was wrong in a
    way that silently produced an overlapping grid, so one throwaway headed
    launch (about a second) buys all three:

    * **Screen.** ``screen.availWidth`` must be read from a ``no_viewport=True``
      context. A normal context emulates ``window.screen`` to match its own
      viewport, so a 1280x720 context insists the display is 1280x720 -- and a
      2x2 grid laid out in that imaginary space cannot fit anywhere.
    * **Chrome.** Playwright sizes a window to fit its context viewport, so the
      difference between the granted window and the viewport IS the chrome.
    * **Floor.** Ask for a 1x1 window and read back what the OS allows.
    """
    playwright = await async_playwright().start()
    try:
        browser = await playwright.chromium.launch(headless=False)
        try:
            # Real screen: no viewport emulation, so window.screen is the display.
            bare = await (await browser.new_context(no_viewport=True)).new_page()
            box = await bare.evaluate(
                "() => [screen.availLeft ?? 0, screen.availTop ?? 0,"
                " screen.availWidth, screen.availHeight]"
            )
            screen = Screen(int(box[0]), int(box[1]), int(box[2]), int(box[3]))

            # Chrome: let Playwright size a window to fit exactly this viewport,
            # then read the overhead straight off the granted bounds.
            context = await browser.new_context(
                viewport=viewport, device_scale_factor=1
            )
            page = await context.new_page()
            cdp = await context.new_cdp_session(page)
            window_id = (await cdp.send("Browser.getWindowForTarget"))["windowId"]
            natural = (
                await cdp.send("Browser.getWindowBounds", {"windowId": window_id})
            )["bounds"]

            # Floor: ask for an impossible window and see what is granted.
            await cdp.send(
                "Browser.setWindowBounds",
                {
                    "windowId": window_id,
                    "bounds": {"left": 0, "top": screen.top, "width": 1, "height": 1,
                               "windowState": "normal"},
                },
            )
            floor = (
                await cdp.send("Browser.getWindowBounds", {"windowId": window_id})
            )["bounds"]

            return screen, ChromeMetrics(
                chrome_px=max(0, natural["height"] - viewport["height"]),
                side_px=max(0, natural["width"] - viewport["width"]),
                min_w=int(floor["width"]),
                min_h=int(floor["height"]),
            )
        finally:
            await browser.close()
    except Exception:
        logging.getLogger(__name__).debug("display probe failed", exc_info=True)
        return FALLBACK_SCREEN, FALLBACK_CHROME
    finally:
        await playwright.stop()


@dataclass
class Layout:
    """Where each agent's window goes, and how big it is drawn.

    Two sizes matter and they are NOT the same:

      * the **viewport** -- fixed at 1280x720, the space the policy predicts
        coordinates in. Never negotiable.
      * the **window** -- what you see on screen. Shrunk to make N of them fit.

    ``Browser.setWindowBounds`` changes the window without touching the
    viewport (measured: a 640x480 window still screenshots 1280x720), so a grid
    that would not otherwise fit is achieved by scaling windows down for
    display, never by shrinking the coordinate space.
    """

    n: int
    screen: Screen
    chrome: ChromeMetrics
    viewport_w: int = 1280
    viewport_h: int = 720
    allow_scale: bool = True

    @property
    def grid(self) -> tuple[int, int]:
        return GRID.get(self.n, (3, (self.n + 2) // 3))

    @property
    def cell_size(self) -> tuple[int, int]:
        """The screen area one window gets before any padding."""
        cols, rows = self.grid
        return self.screen.width // cols, self.screen.height // rows

    @property
    def natural_w(self) -> int:
        """Window width at 1:1 -- the viewport plus whatever frame it costs."""
        return self.viewport_w + self.chrome.side_px

    @property
    def natural_h(self) -> int:
        return self.viewport_h + self.chrome.chrome_px

    @property
    def needed(self) -> tuple[int, int]:
        cols, rows = self.grid
        return cols * self.natural_w, rows * self.natural_h

    @property
    def fits_natively(self) -> bool:
        need_w, need_h = self.needed
        return need_w <= self.screen.width and need_h <= self.screen.height

    @property
    def scale(self) -> float:
        """Display scale for the VIEWPORT ONLY, capped at 1:1.

        Chrome is subtracted from the cell before dividing rather than scaled
        with the page, because it does not shrink: a 429px-tall window keeps its
        full 80px of tab strip and has 349px left for content. Scaling the whole
        natural height uniformly -- ``0.54 * (720 + 80)`` -- claims 386px of
        content in that same 349px and clips 37px off the bottom of every
        window. Solve for what actually fits instead:
        ``viewport * scale + chrome <= cell``.
        """
        if self.fits_natively or not self.allow_scale:
            return 1.0
        cell_w, cell_h = self.cell_size
        return max(0.0, min(
            1.0,
            (cell_w - self.chrome.side_px) / self.viewport_w,
            (cell_h - self.chrome.chrome_px) / self.viewport_h,
        ))

    @property
    def window_w(self) -> int:
        """Window width, never below the floor the OS would clamp us to anyway.

        Reporting the clamped size is the point: the grid check below then sees
        the size the window will really be, instead of one the OS will quietly
        refuse and overlap on.
        """
        return max(self.chrome.min_w, int(self.viewport_w * self.scale) + self.chrome.side_px)

    @property
    def window_h(self) -> int:
        return max(
            self.chrome.min_h, int(self.viewport_h * self.scale) + self.chrome.chrome_px
        )

    @property
    def fits(self) -> bool:
        """True if this grid is achievable at all -- window floor included."""
        cell_w, cell_h = self.cell_size
        return self.window_w <= cell_w and self.window_h <= cell_h

    def max_n(self) -> int:
        """Largest agent count that tiles without overlapping on this screen.

        Bounded by the window floor, not by taste: three 500px-wide columns need
        1500px and simply cannot be drawn on a 1470px screen at any scale.
        """
        return max(
            (n for n in sorted(GRID) if replace(self, n=n).fits),
            default=0,
        )

    def cell(self, index: int) -> tuple[int, int, int, int]:
        """(left, top, width, height) for the window in slot ``index``."""
        cols, rows = self.grid
        col, row = index % cols, index // cols
        step_x = self.screen.width // cols
        step_y = self.screen.height // rows
        # Centre each window inside its cell so the grid looks deliberate, and
        # measure from the screen's usable origin, not from (0, 0).
        pad_x = max(0, (step_x - self.window_w) // 2)
        pad_y = max(0, (step_y - self.window_h) // 2)
        return (
            self.screen.left + col * step_x + pad_x,
            self.screen.top + row * step_y + pad_y,
            self.window_w,
            self.window_h,
        )

    def max_n_at_1to1(self) -> int:
        """Largest agent count whose grid fits without any scaling."""
        best = 0
        for n in sorted(GRID):
            cols, rows = GRID[n]
            if (
                cols * self.natural_w <= self.screen.width
                and rows * self.natural_h <= self.screen.height
            ):
                best = n
        return best

    def describe(self) -> str:
        cols, rows = self.grid
        need_w, need_h = self.needed
        head = (
            f"{cols}x{rows} grid of {self.window_w}x{self.window_h} windows "
            f"on a {self.screen} usable screen"
        )
        if self.fits_natively:
            return head + "  (1:1, no scaling)"

        # Scaling refused, so the grid is knowingly left overlapping.
        if not self.allow_scale:
            return (
                head + "  (scaling disabled -- windows WILL overlap)\n"
                f"  note: needs {need_w}x{need_h} at 1:1. Drop --no-scale, or use "
                f"--n {max(1, self.max_n_at_1to1())}."
            )

        # Scaling cannot rescue it: the window floor is larger than the cell.
        if not self.fits:
            cell_w, cell_h = self.cell_size
            achievable = self.max_n()
            return (
                head + "  (DOES NOT FIT)\n"
                f"  note: a {cols}x{rows} grid gives each window a "
                f"{cell_w}x{cell_h} cell, but the window manager will not draw a "
                f"window smaller than\n"
                f"        {self.chrome.min_w}x{self.chrome.min_h} -- it clamps "
                f"silently, so {self.n} windows would overlap however far they "
                f"are scaled.\n"
                f"        Largest N that tiles cleanly on this screen: "
                f"{achievable if achievable else 'none'}. "
                f"Use --n {max(1, achievable)}."
            )

        fits_at = self.max_n_at_1to1()
        return (
            head + f"  (windows scaled to {self.scale:.0%})\n"
            f"  note: a 1:1 {cols}x{rows} tiling of {self.viewport_w}x"
            f"{self.viewport_h} viewports needs {need_w}x{need_h}, which this "
            f"screen has not got.\n"
            f"        Windows are drawn smaller so all {self.n} fit and stay "
            f"non-overlapping. The VIEWPORT is untouched at "
            f"{self.viewport_w}x{self.viewport_h},\n"
            f"        so screenshots and click coordinates are unaffected "
            f"(asserted after positioning).\n"
            f"        At 1:1 this screen fits "
            f"{fits_at if fits_at else 'no'} agent(s); "
            f"scaled, up to {self.max_n()}."
        )


async def position_window(
    session: BrowserSession,
    bounds: tuple[int, int, int, int],
    scale: float,
    attach_reapply: bool = True,
) -> dict[str, int]:
    """Move/resize this session's window via CDP, and report where it landed.

    Launch flags cannot do this: Playwright resizes the window to fit the
    context viewport, so ``--window-size`` is discarded and every window ends up
    identical -- which is why they all stacked. ``Browser.setWindowBounds`` acts
    on the live window and is honoured exactly.
    """
    left, top, width, height = bounds
    page = session.page
    cdp = await page.context.new_cdp_session(page)
    window_id = (await cdp.send("Browser.getWindowForTarget"))["windowId"]
    await cdp.send(
        "Browser.setWindowBounds",
        {
            "windowId": window_id,
            "bounds": {
                "left": left, "top": top,
                "width": width, "height": height,
                "windowState": "normal",
            },
        },
    )
    if scale < 1.0:
        # Draw the full viewport into the smaller window instead of clipping it.
        # Verified not to disturb the viewport: innerWidth/innerHeight and the
        # screenshot both stay 1280x720. Re-applied on navigation because a
        # cross-document load can drop the override.
        override = {
            "width": session.config.viewport_width,
            "height": session.config.viewport_height,
            "deviceScaleFactor": 1,
            "mobile": False,
            "scale": round(scale, 3),
        }
        await cdp.send("Emulation.setDeviceMetricsOverride", override)

        async def _reapply(_frame: object) -> None:
            try:
                await cdp.send("Emulation.setDeviceMetricsOverride", override)
            except Exception:  # pragma: no cover - page may be closing
                pass

        if attach_reapply:
            page.on(
                "framenavigated",
                lambda frame: asyncio.ensure_future(_reapply(frame)),
            )

    return (await cdp.send("Browser.getWindowBounds", {"windowId": window_id}))["bounds"]


class TiledSession(BrowserSession):
    """A normal session that positions its own window once it exists."""

    def __init__(
        self,
        config: RunConfig,
        browser: PWBrowser,
        slot: int,
        layout: Layout,
        geometry: dict[int, dict],
    ) -> None:
        super().__init__(config, browser=browser)
        self.slot = slot
        self.layout = layout
        self._geometry = geometry

    async def place(self, attach_reapply: bool = True) -> None:
        """Move this window into its slot and re-assert the viewport.

        Separate from ``start`` so the tiler can measure where the OS actually
        put the window and place it again against a corrected layout.
        """
        actual = await position_window(
            self, self.layout.cell(self.slot), self.layout.scale, attach_reapply
        )

        # The coordinate conversion depends on this. Check, do not assume.
        shot = await self.screenshot()
        want = (self.config.viewport_width, self.config.viewport_height)
        if shot.size != want:
            raise SystemExit(
                f"\nFAIL: after positioning, agent {self.slot}'s screenshot is "
                f"{shot.size[0]}x{shot.size[1]}, expected {want[0]}x{want[1]}.\n"
                f"Window geometry must never change the viewport -- the policy "
                f"predicts coordinates in that space and every click would be "
                f"wrong.\nAborting rather than recording a broken run."
            )
        inner = await self.page.evaluate("() => [innerWidth, innerHeight]")
        self._geometry[self.slot] = {
            "asked": self.layout.cell(self.slot),
            "actual": actual,
            "viewport": shot.size,
            "inner": tuple(inner),
        }

    async def start(self) -> "TiledSession":
        await super().start()
        await self.place()
        return self


class TiledBrowserPool:
    """Same interface and same shared-browser design as ``BrowserPool``.

    One Chromium, N contexts -- unchanged from the runner's production layout.
    In headed mode each context's first page opens its own OS window, and
    ``TiledSession`` moves that window into its grid slot after it exists.
    Slots are handed out in ``session()`` call order, which under the runner's
    ``asyncio.gather`` is agent-index order, so agent 0 lands top-left.

    Slots wrap at the grid size. Best-of-N asks for exactly N sessions and never
    reaches the wrap, but a DAG run launches wave after wave and would otherwise
    ask for slot 5 of a four-cell grid -- and ``Layout.cell`` would place it on a
    row that does not exist, off the bottom of the screen. Wrapping is safe
    because a wave's sessions are closed before the next wave opens, so the cell
    is genuinely free by the time it is handed out again.
    """

    def __init__(self, config: RunConfig, layout: Layout) -> None:
        self.config = config
        self.layout = layout
        self._playwright: Playwright | None = None
        self._browser: PWBrowser | None = None
        self._next = 0
        #: slot -> geometry, filled in by each TiledSession as it starts.
        self.geometry: dict[int, dict] = {}

    async def start(self) -> "TiledBrowserPool":
        if self._browser is not None:
            return self
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.config.headless,
            slow_mo=self.config.slow_mo_ms or None,
        )
        return self

    def session(self, config: RunConfig | None = None) -> BrowserSession:
        if self._browser is None:
            raise RuntimeError("TiledBrowserPool.start() must be awaited first")
        slot = self._next % max(1, self.layout.n)
        self._next += 1
        return TiledSession(
            config or self.config, self._browser, slot, self.layout, self.geometry
        )

    async def close(self) -> None:
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception:  # pragma: no cover - best effort teardown
                pass
            self._browser = None
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:  # pragma: no cover
                pass
            self._playwright = None

    async def __aenter__(self) -> "TiledBrowserPool":
        return await self.start()

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()


# ---------------------------------------------------------------------------
# narration
# ---------------------------------------------------------------------------
COLOURS = ["\033[36m", "\033[35m", "\033[32m", "\033[33m", "\033[34m", "\033[31m"]


class Narrator:
    """All console output goes through here, so --no-color really means it."""

    def __init__(self, colour: bool = True) -> None:
        self.colour = colour
        self.bold = "\033[1m" if colour else ""
        self.dim = "\033[2m" if colour else ""
        self.reset = "\033[0m" if colour else ""

    def strong(self, text: str) -> str:
        return f"{self.bold}{text}{self.reset}"

    def faint(self, text: str) -> str:
        return f"{self.dim}{text}{self.reset}"

    def tint(self, index: int, text: str) -> str:
        if not self.colour:
            return text
        return f"{COLOURS[index % len(COLOURS)]}{text}{self.reset}"

    def agent(self, index: int, message: str) -> None:
        print(f"{self.tint(index, f'[agent {index}]')} {message}", flush=True)

    def rule(self, title: str = "") -> None:
        line = "-" * 74
        print(f"\n{line}\n{title}\n{line}" if title else f"\n{line}", flush=True)


class NarratingPolicy(AgentPolicy):
    """Wraps any policy and prints one readable line per decision.

    Presentation only: it forwards ``predict`` untouched and never alters the
    step. Pacing is NOT done here -- it comes from ``RunConfig.settle_ms``,
    which the browser already applies after executing an action, which is
    exactly where a viewer needs the pause.
    """

    def __init__(
        self, inner: AgentPolicy, index: int, narrator: Narrator, width: int = 96
    ) -> None:
        self.inner = inner
        self.index = index
        self.narrator = narrator
        self.width = width
        self.name = f"demo({getattr(inner, 'name', 'policy')})"
        self._step = 0

    async def predict(
        self,
        task: str,
        history: Sequence[Step],
        screenshot: Image.Image,
        page: PageInfo,
    ) -> Step:
        step = await self.inner.predict(task, history, screenshot, page)
        self._step += 1
        thought = " ".join((step.thought or "").split())
        if len(thought) > self.width:
            thought = thought[: self.width - 1] + "…"
        self.narrator.agent(
            self.index,
            f"{self.narrator.strong(f'step {self._step}')}  "
            f"{self.narrator.faint(thought)}",
        )
        self.narrator.agent(self.index, f"         → {step.action.summary()}")
        return step

    async def reset(self) -> None:
        self._step = 0
        await self.inner.reset()

    async def close(self) -> None:
        await self.inner.close()


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
def _overlaps(a: dict, b: dict) -> bool:
    return not (
        a["left"] + a["width"] <= b["left"]
        or b["left"] + b["width"] <= a["left"]
        or a["top"] + a["height"] <= b["top"]
        or b["top"] + b["height"] <= a["top"]
    )


async def preflight(pool: TiledBrowserPool, run_config: RunConfig) -> bool:
    """Open every window, report the geometry it actually got, check the grid.

    Reports measured numbers rather than "the calls succeeded": window bounds
    read back from CDP, the live ``innerWidth/innerHeight``, the screenshot
    size, and a pairwise overlap test across all N windows.
    """
    layout = pool.layout
    want = (run_config.viewport_width, run_config.viewport_height)
    print(f"preflight: opening {layout.n} window(s), verifying {want[0]}x{want[1]} "
          f"viewport and grid geometry")

    sessions = [pool.session(run_config) for _ in range(layout.n)]
    ok = True
    try:
        for session in sessions:
            await session.start()  # positions, asserts viewport, records geometry
            await session.goto(DEMO_PAGE.as_uri())

        # Measure, then correct. The window manager silently refuses to place a
        # window above the menu bar, and `screen.availTop` does not always
        # report that floor -- it came back 0 on a machine that clamps to 33.
        # So compare where row 0 was asked to go with where it landed, and if
        # the OS pushed it down, shift the whole grid into the space that
        # actually exists and place again. Without this, every row inherits the
        # offset and the bottom row overlaps the one above it.
        cols, _ = layout.grid
        inset = max(
            (
                pool.geometry[slot]["actual"]["top"] - pool.geometry[slot]["asked"][1]
                for slot in pool.geometry
                if slot < cols
            ),
            default=0,
        )
        if inset > 0:
            print(
                f"  window manager pushed row 0 down by {inset}px "
                f"(menu bar); re-laying out in the space that remains"
            )
            layout.screen = Screen(
                layout.screen.left,
                layout.screen.top + inset,
                layout.screen.width,
                layout.screen.height - inset,
            )
            for session in sessions:
                await session.place(attach_reapply=False)

        print(f"\n  {'agent':<6}{'requested':<22}{'actual window':<24}"
              f"{'viewport':<12}{'innerW/H'}")
        bounds: list[dict] = []
        for slot in sorted(pool.geometry):
            g = pool.geometry[slot]
            ax, ay, aw, ah = g["asked"]
            b = g["actual"]
            bounds.append(b)
            print(
                f"  {slot:<6}"
                f"{f'({ax},{ay}) {aw}x{ah}':<22}"
                f"{f'({b['left']},{b['top']}) {b['width']}x{b['height']}':<24}"
                f"{f'{g['viewport'][0]}x{g['viewport'][1]}':<12}"
                f"{g['inner'][0]}x{g['inner'][1]}"
            )

        # Every viewport exactly 1280x720?
        drifted = [s for s in pool.geometry if pool.geometry[s]["viewport"] != want]
        if drifted:  # pragma: no cover - TiledSession.start already aborts
            print(f"\n  FAIL: viewport drifted on agent(s) {drifted}")
            ok = False
        else:
            print(f"\n  viewport: all {layout.n} exactly {want[0]}x{want[1]} ✓")

        # Non-overlapping?
        clashes = [
            (i, j)
            for i in range(len(bounds))
            for j in range(i + 1, len(bounds))
            if _overlaps(bounds[i], bounds[j])
        ]
        if clashes:
            pairs = ", ".join(f"{i}&{j}" for i, j in clashes)
            print(f"  overlap : FAIL -- windows {pairs} intersect")
            ok = False
        else:
            print(f"  overlap : none, all {layout.n} windows are disjoint ✓")

        # Fully on screen?
        offscreen = [
            slot
            for slot, g in pool.geometry.items()
            if g["actual"]["left"] + g["actual"]["width"]
            > layout.screen.left + layout.screen.width
            or g["actual"]["top"] + g["actual"]["height"]
            > layout.screen.top + layout.screen.height
        ]
        if offscreen:
            print(
                f"  onscreen: FAIL -- agent(s) {offscreen} extend past the "
                f"{layout.screen} screen"
            )
            ok = False
        else:
            print(f"  onscreen: all within {layout.screen} ✓")
    finally:
        for session in sessions:
            try:
                await session.close()
            except Exception:  # pragma: no cover
                pass
        pool._next = 0  # slots are handed out again for the real run
        pool.geometry.clear()
    print()
    return ok


# ---------------------------------------------------------------------------
# cost guard
# ---------------------------------------------------------------------------
def confirm_cost(
    estimate_calls_: int, breakdown: str, free: bool, assume_yes: bool, narrator: Narrator
) -> None:
    """Ask before spending. Free runs (mock policy, mock judge) never ask;
    a mock-agent dag run still does, because the manager is an LLM."""
    if free:
        return
    print(
        f"\n  {narrator.strong('cost estimate')}: {breakdown} = {estimate_calls_} API calls\n"
        f"  (an upper bound -- agents that finish early make fewer)"
    )
    if assume_yes:
        print("  --yes given, proceeding.\n")
        return
    try:
        reply = input("  Proceed? [y/N] ").strip().lower()
    except EOFError:
        reply = ""
    if reply not in {"y", "yes"}:
        raise SystemExit("  Cancelled. Nothing was spent.")
    print()


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
async def run_demo(args: argparse.Namespace) -> int:
    narrator = Narrator(colour=not args.no_color)
    preset = PRESETS[args.preset]
    task = args.task or preset.task
    start_url = args.start_url or preset.start_url
    max_steps = args.max_steps or preset.steps
    # A preset knows which strategy it was written for; an explicit flag wins.
    strategy_name = args.strategy or preset.strategy
    max_subtasks = args.max_subtasks or preset.max_subtasks

    width, _, height = args.viewport.partition("x")
    viewport = {"width": int(width), "height": int(height)}

    # Headless has no windows to tile, so skip the probe rather than flash one up.
    if args.headless:
        detected, chrome = FALLBACK_SCREEN, FALLBACK_CHROME
    else:
        detected, chrome = await probe_display(viewport)
    if args.chrome_px is not None:
        chrome = replace(chrome, chrome_px=args.chrome_px)
    if args.screen == "auto":
        screen = detected
    else:
        sw, _, sh = args.screen.partition("x")
        # Keep the detected origin: the OS still refuses to place a window above
        # the menu bar even when the size is overridden.
        screen = Screen(detected.left, detected.top, int(sw), int(sh) - detected.top)

    spec = LaunchSpec(
        task=task,
        strategy=strategy_name,
        policy=args.policy,
        n=args.n,
        max_steps=max_steps,
        start_url=start_url,
        manager_model=args.manager_model,
        planning_budget=args.planning_budget,
        max_subtasks=max_subtasks,
        max_waves=args.max_waves,
        headless=args.headless,
        viewport_width=viewport["width"],
        viewport_height=viewport["height"],
        # The pause a viewer needs, applied by the browser AFTER each action.
        # Using the existing knob keeps this presentation-only.
        settle_ms=int(args.step_delay * 1000) + 200,
        max_browsers=args.n,
        max_model=args.max_model,
        runs_dir=args.runs_dir,
    )
    estimate = estimate_calls(spec)
    # The API ceiling for this run is exactly the estimate: the demo should
    # never quietly spend more than it just quoted.
    spec = spec.model_copy(update={"max_calls": max(estimate.calls, 1)})
    run_config = spec.run_config()
    layout = Layout(
        n=args.n,
        screen=screen,
        chrome=chrome,
        viewport_w=run_config.viewport_width,
        viewport_h=run_config.viewport_height,
        allow_scale=not args.no_scale,
    )

    narrator.rule(narrator.strong("multi-agent web — live demo"))
    print(f"  task     : {task}")
    print(f"  start    : {start_url}")
    if strategy_name == "dag":
        print(f"  strategy : dag — a manager LLM decomposes this into subtasks "
              f"(budget {args.planning_budget} edits, up to {max_subtasks} at first)")
        print(f"  tiles    : {args.n}   policy: {args.policy}   "
              f"max steps: {max_steps}")
    else:
        print(f"  agents   : {args.n}   policy: {args.policy}   "
              f"max steps: {max_steps}")
    print(f"  display  : measured {chrome}")
    print(f"  layout   : {layout.describe()}")
    print(f"  pacing   : {args.step_delay:.1f}s after each action")
    print(f"  preset   : {preset.key} — {preset.why}")

    # Fail before launching anything, rather than opening N browsers to discover
    # what the arithmetic already knows. --no-scale opts into overlap knowingly.
    if layout.allow_scale and not layout.fits and not args.force:
        raise SystemExit(
            f"\n  {args.n} agents cannot be tiled on this screen (see above). "
            f"Use --n {max(1, layout.max_n())}, a larger --screen, or --force."
        )

    confirm_cost(estimate.calls, estimate.breakdown, estimate.free, args.yes, narrator)

    # Same wiring as run_multi.py and the web UI, plus a narrator around each
    # agent's policy. Presentation only: the narrator forwards predict untouched.
    try:
        prepared = prepare(
            spec,
            policy_wrapper=lambda policy, index: NarratingPolicy(policy, index, narrator),
        )
    except LaunchError as exc:
        raise SystemExit(str(exc)) from exc
    strategy = prepared.strategy
    usage_provider = prepared.usage_provider

    run_dir = make_run_dir(run_config.runs_dir)
    runner = Runner(
        policy_factory=prepared.policy_factory,
        run_config=run_config,
        orchestrator_config=prepared.orchestrator_config,
        run_dir=run_dir,
    )
    # Presentation swap: tiled windows instead of one shared browser. Same
    # interface, so the runner is unaware and unmodified.
    runner.pool = TiledBrowserPool(run_config, layout)

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    started = perf_counter()

    await runner.pool.start()
    try:
        if not await preflight(runner.pool, run_config) and not args.force:
            raise SystemExit(
                "  Preflight failed. Fix the layout (see --n / --screen / "
                "--no-scale) or pass --force to record anyway."
            )
        narrator.rule(narrator.strong("running"))
        if strategy_name == "dag":
            print("  the manager is planning…", flush=True)
        outcome = await strategy.run(task, runner)
    finally:
        await runner.pool.close()
        await prepared.aclose()

    timing = summarise_timing(
        outcome.sessions,
        wall_seconds=perf_counter() - started,
        peak_browsers=runner.peak_concurrent_browsers,
        peak_model=runner.slot.peak_inflight,
    )
    result = OrchestrationResult(
        task=task,
        strategy=strategy.name,
        answer=outcome.answer,
        contributing_indices=outcome.contributing_indices,
        reason=outcome.reason,
        sessions=outcome.sessions,
        timing=timing,
        run_dir=run_dir,
        details=outcome.details,
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        usage=usage_provider() if usage_provider else None,
    )
    write_run_json(result)
    print_summary(result, narrator)
    return 0 if result.answer is not None else 1


def print_dag_summary(details: dict, narrator: Narrator) -> None:
    """The graph the manager built, what it changed, and what it spent.

    DAG growth and replan rate are the numbers the paper reports, so they go on
    screen rather than only into run.json -- a recording should show them.
    """
    growth = details["growth"]
    print(f"\n  {narrator.strong('graph')} (manager: {details['manager']['model']})")
    for subtask in details["final_dag"]["subtasks"]:
        deps = ", ".join(subtask["depends_on"])
        mark = {"done": "✓", "failed": "✗", "blocked": "–"}.get(subtask["status"], "?")
        print(
            f"    {mark} {subtask['id']:<20} wave {subtask['wave'] or '-'}"
            + (f"   after: {deps}" if deps else "")
        )
        print(f"      {narrator.faint(' '.join(subtask['instruction'].split())[:88])}")

    applied = [r for r in details["replans"] if r.get("applied")]
    if applied:
        print(f"\n  {narrator.strong('replans')}")
        for entry in applied:
            change = "".join(
                [f" +{[s['id'] for s in entry['add']]}" if entry.get("add") else "",
                 f" -{entry['remove']}" if entry.get("remove") else ""]
            )
            print(f"    wave {entry.get('wave', '?')}:{change} — {entry.get('reason', '')[:70]}")

    b = details["budget"]
    print(f"\n  {narrator.strong('planning')}")
    print(f"    subtasks     {growth['initial_subtasks']} → "
          f"{growth['final_subtasks']} over {growth['waves']} wave(s)")
    print(f"    replans      {growth['replans_applied']} of "
          f"{growth['replans_proposed']} proposed "
          f"({growth['replan_rate']:.2f} per wave)")
    print(f"    edit budget  {b['spent']}/{b['limit']}")
    if details.get("stopped_early"):
        print(f"    stopped      {details['stopped_early']}")


def print_summary(result: OrchestrationResult, narrator: Narrator) -> None:
    narrator.rule(narrator.strong("result"))

    details = result.details or {}
    is_dag = "final_dag" in details
    # Under best-of-N one agent wins; under a DAG every terminal subtask
    # contributes, so the marker has to say something different.
    credit = "  ← contributed" if is_dag else "  ← judge's pick"

    print(f"  {narrator.strong('per agent')}")
    for session in result.sessions:
        mark = "✓" if session.succeeded else "✗"
        won = credit if session.index in result.contributing_indices else ""
        answer = " ".join((session.answer or "").split())[:64]
        label = f"{session.spec.label}  " if is_dag and session.spec.label else ""
        print(
            f"    {narrator.tint(session.index, f'[{session.index}]')} {mark} "
            f"{session.status:<9} {len(session.steps):>2} steps "
            f"{session.timing.wall_seconds:>6.1f}s  {label}{answer}"
            f"{narrator.strong(won)}"
        )
        if session.error:
            print(f"         error: {session.error[:100]}")

    if is_dag:
        print_dag_summary(details, narrator)
    else:
        judge = details.get("judge", {})
        print(f"\n  {narrator.strong('judge')} ({judge.get('name', '?')})")
        print(f"    picked : agent {judge.get('winner')}")
        print(f"    because: {judge.get('reason', '')}")

    print(f"\n  {narrator.strong('answer')}")
    for line in (result.answer or "(none)").splitlines():
        print(f"    {line}")

    t = result.timing
    total = t.total_model_seconds + t.total_browser_seconds
    share = (t.total_model_seconds / total * 100) if total else 0.0
    print(f"\n  {narrator.strong('time')}")
    print(f"    wall clock   {t.wall_seconds:>7.1f}s")
    print(f"    if serial    {t.sum_agent_wall_seconds:>7.1f}s "
          f"({t.speedup_vs_serial:.2f}x concurrency)")
    print(f"    model        {t.total_model_seconds:>7.1f}s  ({share:.0f}% of agent time)")
    print(f"    browser      {t.total_browser_seconds:>7.1f}s  ({100 - share:.0f}%)")
    if t.total_model_queue_seconds > 0.05:
        print(f"    queued       {t.total_model_queue_seconds:>7.1f}s  "
              f"(waiting for a model slot)")

    if result.usage:
        u = result.usage
        print(f"\n  {narrator.strong('api usage')}")
        print(f"    calls        {u['calls']}/{u['limit']}  ({u['retries']} retries)")
        print(f"    tokens       {u['total_tokens']} "
              f"({u['image_tokens']} from images)")

    print(f"\n  artifacts: {result.run_dir}")
    narrator.rule()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recording-friendly multi-agent demo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="presets:\n"
        + "\n".join(f"  {p.key:<11} {p.task}" for p in PRESETS.values()),
    )
    parser.add_argument("--preset", default="wikipedia", choices=sorted(PRESETS))
    parser.add_argument("--task", default=None, help="Override the preset task.")
    parser.add_argument("--start-url", default=None)
    parser.add_argument(
        "--n", type=int, default=4,
        help="How many agents. Under --strategy dag this is the number of "
        "TILES: the manager decides how many subtasks there are, and at most "
        "this many run (and are visible) at once.",
    )
    parser.add_argument("--policy", default="qwen", choices=["qwen", "mock"])
    parser.add_argument(
        "--strategy", default=None, choices=["best_of_n", "dag"],
        help="best_of_n: N agents race at the same task. dag: a manager LLM "
        "decomposes it into a dependency graph, run in waves. Default: what "
        "the preset was written for (best_of_n for all but 'pricecheck').",
    )
    parser.add_argument(
        "--manager-model", default=None,
        help="Model for the dag manager. Default: PPAPI_MANAGER_MODEL, else the "
        "policy's model.",
    )
    parser.add_argument(
        "--planning-budget", type=int, default=10,
        help="DAG edits the manager may make (default 10). 0 decomposes but "
        "never replans.",
    )
    parser.add_argument(
        "--max-subtasks", type=int, default=None,
        help="Ceiling on the manager's first plan. Default: the preset's.",
    )
    parser.add_argument("--max-waves", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--step-delay", type=float, default=0.8,
        help="Seconds to pause after each action so a viewer can follow.",
    )
    parser.add_argument(
        "--screen", default="auto",
        help="Screen size to tile in, e.g. 2560x1440. Default: detect it.",
    )
    parser.add_argument("--viewport", default="1280x720")
    parser.add_argument(
        "--chrome-px", type=int, default=None,
        help="Override the measured height of the browser's own UI strip.",
    )
    parser.add_argument(
        "--no-scale", action="store_true",
        help="Never shrink windows to fit; overlap instead.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Record even if the preflight grid check fails.",
    )
    parser.add_argument(
        "--max-model", type=int, default=4,
        help="Concurrent model requests. Keep >= n so agents do not queue on video.",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="No windows. Only useful for checking output text.",
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip the cost prompt.")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    # Quiet: only the narration should reach the console during a recording.
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
    for noisy in ("multi_agent_web", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    return asyncio.run(run_demo(args))


if __name__ == "__main__":
    raise SystemExit(main())
