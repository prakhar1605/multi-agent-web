"""One isolated agent run: its own browser context, its own policy instance.

ISOLATION IS BY CONSTRUCTION, NOT BY CONVENTION
===============================================

Two things must not be shared between concurrent agents, and this module is
built so that sharing them is not expressible rather than merely discouraged.

**Browser context.** Every session takes a fresh ``BrowserContext`` from a
shared Chromium process. A context -- not just a page -- is the isolation unit
in Playwright: cookies, localStorage, sessionStorage, IndexedDB and HTTP auth
all live on the context. Two agents on separate *pages* of one context would
share a login; on separate contexts they cannot. One browser process with N
contexts is far cheaper than N browser processes, and isolates just as well.

**Policy instance.** ``AgentSession`` accepts a *factory*
(``Callable[[int], AgentPolicy]``), never a policy object, and calls it exactly
once for its own index. There is no parameter through which two sessions could
receive the same policy. This matters because policies are stateful:
``MolmoWebPolicy`` carries ``past_actions``, the history it renders into every
prompt. Sharing one instance across concurrent agents would interleave their
histories, and each agent would be prompted with steps another agent took --
a bug that produces plausible-looking trajectories and is near-impossible to
spot after the fact. ``Runner`` additionally asserts the factory did not hand
back the same object twice, in case a factory closes over a cached instance.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image
from playwright.async_api import Browser as PWBrowser
from playwright.async_api import Playwright, async_playwright

from ..agent import Agent
from ..browser import BrowserSession, PageInfo
from ..config import RunConfig
from ..policy.base import AgentPolicy
from ..trajectory import Step, Trajectory
from .events import EventSink

logger = logging.getLogger(__name__)

#: Builds a fresh policy for the agent at ``index``. The index is passed so a
#: factory can vary sampling per agent (useful for best-of-N diversity).
PolicyFactory = Callable[[int], AgentPolicy]

#: Terminal states for a session. The first three come from the agent loop;
#: "crashed" means the session raised before producing a result.
SessionStatus = str  # "done" | "max_steps" | "aborted" | "crashed"


class BrowserPool:
    """One Chromium process that hands out isolated contexts.

    Started once per multi-agent run. ``session()`` returns an unstarted
    ``BrowserSession`` bound to a fresh context of this browser.
    """

    def __init__(self, config: RunConfig | None = None) -> None:
        self.config = config or RunConfig()
        self._playwright: Playwright | None = None
        self._browser: PWBrowser | None = None

    async def start(self) -> "BrowserPool":
        if self._browser is not None:
            return self
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.config.headless,
            slow_mo=self.config.slow_mo_ms or None,
        )
        logger.debug("browser pool started (headless=%s)", self.config.headless)
        return self

    def session(self, config: RunConfig | None = None) -> BrowserSession:
        if self._browser is None:
            raise RuntimeError("BrowserPool.start() must be awaited first")
        return BrowserSession(config or self.config, browser=self._browser)

    async def close(self) -> None:
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception:  # pragma: no cover - best effort teardown
                logger.debug("error closing pooled browser", exc_info=True)
            self._browser = None
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:  # pragma: no cover
                logger.debug("error stopping playwright", exc_info=True)
            self._playwright = None

    async def __aenter__(self) -> "BrowserPool":
        return await self.start()

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()


class ModelSlot:
    """Limits how many ``predict()`` calls are in flight against the model.

    Separate from the browser limit on purpose: the model server is one shared,
    usually GPU-bound resource, while browsers are local and memory-bound. With
    a single GPU, generation serializes, so agents queue here even while their
    browsers sit idle in parallel -- and the time they spend queueing is the
    number that explains where wall-clock goes as N grows.
    """

    def __init__(self, limit: int | None) -> None:
        import asyncio

        self.limit = limit
        self._sem = asyncio.Semaphore(limit) if limit and limit > 0 else None
        self.inflight = 0
        self.peak_inflight = 0

    async def __aenter__(self) -> None:
        if self._sem is not None:
            await self._sem.acquire()
        self.inflight += 1
        self.peak_inflight = max(self.peak_inflight, self.inflight)

    async def __aexit__(self, *exc_info: object) -> None:
        self.inflight -= 1
        if self._sem is not None:
            self._sem.release()


class ThrottledPolicy(AgentPolicy):
    """Wraps a policy so its ``predict`` calls pass through a ``ModelSlot``.

    Also stamps ``model_queue_seconds`` onto the returned step -- the time the
    call spent waiting for a slot, as opposed to generating. The agent loop
    records the total around this, so the two together decompose model time
    into "queued" and "actually running".
    """

    def __init__(self, inner: AgentPolicy, slot: ModelSlot) -> None:
        self.inner = inner
        self.slot = slot
        self.name = f"throttled({getattr(inner, 'name', 'policy')})"

    async def predict(
        self,
        task: str,
        history: Sequence[Step],
        screenshot: Image.Image,
        page: PageInfo,
    ) -> Step:
        queue_started = perf_counter()
        async with self.slot:
            queued = perf_counter() - queue_started
            step = await self.inner.predict(task, history, screenshot, page)
        step.model_queue_seconds = queued
        return step

    async def reset(self) -> None:
        await self.inner.reset()

    async def close(self) -> None:
        await self.inner.close()


@dataclass
class SessionSpec:
    """What one agent should do. The unit a strategy schedules."""

    task: str
    start_url: str | None = None
    #: Free-form label for reporting, e.g. a Phase 3 subtask name.
    label: str | None = None


@dataclass
class SessionTiming:
    wall_seconds: float = 0.0
    model_seconds: float = 0.0
    model_queue_seconds: float = 0.0
    browser_seconds: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "wall_seconds": round(self.wall_seconds, 3),
            "model_seconds": round(self.model_seconds, 3),
            "model_queue_seconds": round(self.model_queue_seconds, 3),
            "browser_seconds": round(self.browser_seconds, 3),
        }


@dataclass
class SessionResult:
    """Outcome of one agent, successful or not.

    A crashed session is still a result, not an omission: strategies and judges
    see every agent that was launched, labelled with what happened to it.
    """

    index: int
    spec: SessionSpec
    status: SessionStatus
    answer: str | None = None
    steps: list[Step] = field(default_factory=list)
    run_dir: Path | None = None
    error: str | None = None
    timing: SessionTiming = field(default_factory=SessionTiming)

    @property
    def succeeded(self) -> bool:
        """True only for a run that terminated deliberately via Done."""
        return self.status == "done"

    @property
    def num_errors(self) -> int:
        return sum(1 for s in self.steps if s.error)

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "dir": self.run_dir.name if self.run_dir else None,
            "task": self.spec.task,
            "label": self.spec.label,
            "status": self.status,
            "answer": self.answer,
            "num_steps": len(self.steps),
            "num_errors": self.num_errors,
            "error": self.error,
            "timing": self.timing.as_dict(),
        }


class AgentSession:
    """One agent: own context, own policy, own trajectory directory."""

    def __init__(
        self,
        index: int,
        spec: SessionSpec,
        policy: AgentPolicy,
        pool: BrowserPool,
        run_config: RunConfig,
        run_dir: Path,
        slot: ModelSlot | None = None,
        sink: EventSink | None = None,
    ) -> None:
        self.index = index
        self.spec = spec
        self.run_config = run_config
        self.run_dir = run_dir
        # The policy arrives already built by the Runner from a factory, so a
        # session can never be handed one that another session is also using.
        self.policy: AgentPolicy = (
            ThrottledPolicy(policy, slot) if slot is not None else policy
        )
        self._inner_policy = policy
        self.pool = pool
        # Observe-only. The default discards everything; see events.py.
        self.sink = sink or EventSink()

    def _announce_step(self, step: Step) -> None:
        """Trajectory -> sink. Called after the step is fully recorded."""
        self.sink.emit(
            "agent_step",
            index=self.index,
            step={
                "index": step.index,
                "thought": step.thought,
                "action": step.action.summary(),
                "action_json": step.action.model_dump(mode="json"),
                "error": step.error,
                # Relative to the RUN directory, so a viewer can fetch it
                # without knowing how agent directories are named.
                "screenshot": (
                    f"{self.run_dir.name}/{step.screenshot}" if step.screenshot else None
                ),
                "url": step.url,
                "title": step.title,
                "model_seconds": step.model_seconds,
                "model_queue_seconds": step.model_queue_seconds,
                "browser_seconds": step.browser_seconds,
            },
        )

    async def run(self) -> SessionResult:
        """Run this agent to completion. Never raises.

        Any exception is captured into a ``crashed`` result so that one bad
        agent cannot take down the rest of the fleet.
        """
        started = perf_counter()
        browser = self.pool.session(self.run_config)
        try:
            await browser.start()
            trajectory = Trajectory(
                task=self.spec.task,
                config=self.run_config,
                run_dir=self.run_dir,
                on_record=self._announce_step,
            )
            agent = Agent(
                policy=self.policy, browser=browser, config=self.run_config
            )
            result = await agent.run(
                self.spec.task,
                start_url=self.spec.start_url,
                trajectory=trajectory,
            )
            return SessionResult(
                index=self.index,
                spec=self.spec,
                status=result.status,
                answer=result.answer,
                steps=result.steps,
                run_dir=Path(result.run_dir),
                timing=_sum_timing(result.steps, perf_counter() - started),
            )
        except Exception as exc:
            logger.exception("agent %d crashed", self.index)
            return SessionResult(
                index=self.index,
                spec=self.spec,
                status="crashed",
                error=f"{type(exc).__name__}: {exc}",
                run_dir=self.run_dir if self.run_dir.exists() else None,
                timing=SessionTiming(wall_seconds=perf_counter() - started),
            )
        finally:
            try:
                await browser.close()
            except Exception:  # pragma: no cover
                logger.debug("error closing session browser", exc_info=True)
            try:
                await self.policy.close()
            except Exception:  # pragma: no cover
                logger.debug("error closing session policy", exc_info=True)


def _sum_timing(steps: Sequence[Step], wall_seconds: float) -> SessionTiming:
    return SessionTiming(
        wall_seconds=wall_seconds,
        model_seconds=sum(s.model_seconds or 0.0 for s in steps),
        model_queue_seconds=sum(s.model_queue_seconds or 0.0 for s in steps),
        browser_seconds=sum(s.browser_seconds or 0.0 for s in steps),
    )
