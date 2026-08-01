"""Async Playwright wrapper: the only module that knows about a real browser.

The agent loop talks to this class through three methods -- ``screenshot()``,
``execute()`` and ``page_info()`` -- which together define the environment in
the reinforcement-learning sense. Nothing above this layer imports Playwright,
so the environment can later be swapped for a recorded/replayed one without
touching the policy or the loop.
"""

from __future__ import annotations

import asyncio
import io
import logging
from types import TracebackType

from PIL import Image
from playwright.async_api import Browser as PWBrowser
from playwright.async_api import BrowserContext, Page, Playwright, async_playwright
from pydantic import BaseModel

from .actions import Action, Click, Done, KeyPress, Navigate, Scroll, Type, Wait
from .config import RunConfig

logger = logging.getLogger(__name__)


class PageInfo(BaseModel):
    """Cheap textual context about the current page.

    Kept deliberately small: the screenshot is the observation, this is just
    grounding metadata (a screenshot alone cannot tell you the URL).
    """

    url: str
    title: str

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return f"{self.title} <{self.url}>"


class ActionError(RuntimeError):
    """Raised when an action could not be carried out.

    The agent loop catches this, records it on the step, and keeps going --
    a failed click should not end an episode.
    """


class BrowserSession:
    """Owns a Chromium instance, one context and one page.

    One session == one tab == one agent. Phase 2 gives each parallel agent its
    own ``BrowserSession`` (separate contexts mean separate cookies/storage),
    which is why nothing here is a module-level singleton.
    """

    def __init__(
        self,
        config: RunConfig | None = None,
        browser: PWBrowser | None = None,
    ) -> None:
        """
        Args:
            config: browser settings for this session.
            browser: an already-running Chromium to take a context from, owned
                by the caller. The orchestrator passes one so that N concurrent
                agents share a single browser process but each still gets its
                own **context** -- separate cookies, localStorage and auth
                state. Passing None launches (and later closes) a private one.
        """
        self.config = config or RunConfig()
        self._playwright: Playwright | None = None
        self._browser: PWBrowser | None = browser
        self._owns_browser = browser is None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    # --- lifecycle ---------------------------------------------------------

    async def start(self) -> "BrowserSession":
        if self._page is not None:
            return self
        if self._browser is None:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=self.config.headless,
                slow_mo=self.config.slow_mo_ms or None,
            )
        self._context = await self._browser.new_context(
            viewport=self.config.viewport,
            # 1 CSS pixel == 1 screenshot pixel; see actions.py docstring.
            device_scale_factor=self.config.device_scale_factor,
        )
        self._context.set_default_timeout(self.config.action_timeout_ms)
        self._context.set_default_navigation_timeout(self.config.navigation_timeout_ms)
        self._page = await self._context.new_page()
        logger.debug("browser started (headless=%s)", self.config.headless)
        return self

    async def close(self) -> None:
        """Tear down in reverse order; never raise out of cleanup.

        A borrowed browser is left running -- closing it would take down every
        other session sharing it.
        """
        closers = [self._context]
        if self._owns_browser:
            closers.append(self._browser)
        for closer in closers:
            if closer is not None:
                try:
                    await closer.close()
                except Exception:  # pragma: no cover - best effort teardown
                    logger.debug("error while closing browser resource", exc_info=True)
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:  # pragma: no cover
                logger.debug("error while stopping playwright", exc_info=True)
        self._playwright = self._context = self._page = None
        if self._owns_browser:
            self._browser = None

    async def __aenter__(self) -> "BrowserSession":
        return await self.start()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("BrowserSession.start() must be awaited before use.")
        return self._page

    # --- observation -------------------------------------------------------

    async def screenshot(self) -> Image.Image:
        """Capture the current viewport as a PIL image.

        Returns RGB (no alpha) because that is what image encoders for VLMs
        expect, and the size is exactly ``viewport_width x viewport_height``.
        """
        png_bytes = await self.page.screenshot(type="png", full_page=False)
        image = Image.open(io.BytesIO(png_bytes))
        image.load()  # decode now so the BytesIO buffer can be freed
        return image.convert("RGB")

    async def page_info(self) -> PageInfo:
        try:
            title = await self.page.title()
        except Exception:  # pragma: no cover - title can fail mid-navigation
            title = ""
        return PageInfo(url=self.page.url, title=title)

    async def goto(self, url: str) -> None:
        """Convenience entry point for the run's starting URL."""
        await self.execute(Navigate(url=url))

    # --- action execution --------------------------------------------------

    async def execute(self, action: Action) -> None:
        """Carry out one action. Raises ``ActionError`` if it fails.

        Every branch is a straight translation of the action into Playwright
        calls -- no retries, no heuristics, no "did the model really mean this"
        repair. Keeping the environment dumb means failures show up in the
        trajectory as failures, which is what you want when evaluating a policy.
        """
        try:
            if isinstance(action, Click):
                if not self.config.contains_point(action.x, action.y):
                    raise ActionError(
                        f"click ({action.x:g}, {action.y:g}) is outside the "
                        f"{self.config.viewport_width}x{self.config.viewport_height} viewport"
                    )
                await self.page.mouse.click(action.x, action.y)

            elif isinstance(action, Type):
                # Types into the focused element; a Click should precede this.
                await self.page.keyboard.type(action.text, delay=20)
                if action.press_enter:
                    await self.page.keyboard.press("Enter")

            elif isinstance(action, Scroll):
                await self.page.mouse.wheel(action.delta_x, action.delta_y)

            elif isinstance(action, KeyPress):
                await self.page.keyboard.press(action.key)

            elif isinstance(action, Navigate):
                await self.page.goto(action.url, wait_until="domcontentloaded")

            elif isinstance(action, Wait):
                await asyncio.sleep(action.seconds)

            elif isinstance(action, Done):
                return  # terminal marker; the agent loop handles it

            else:  # pragma: no cover - unreachable while Action stays closed
                raise ActionError(f"unhandled action type: {type(action).__name__}")

        except ActionError:
            raise
        except Exception as exc:
            raise ActionError(f"{action.summary()} failed: {exc}") from exc

        await self._settle()

    async def _settle(self) -> None:
        """Give the page a moment to react before the next screenshot."""
        if self.config.settle_ms:
            try:
                await self.page.wait_for_timeout(self.config.settle_ms)
            except Exception:  # pragma: no cover - page may be navigating
                logger.debug("settle wait interrupted", exc_info=True)
