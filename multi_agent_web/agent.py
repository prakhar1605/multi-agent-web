"""The single-agent ReAct loop.

    observe (screenshot + page info)
        -> policy.predict(...)
        -> log the step
        -> execute the action
        -> repeat until Done or max_steps

This is the whole of Phase 1. In Phase 2 a manager will decompose a goal into
subtasks and run one ``Agent`` per subtask concurrently -- which is why ``run``
takes the task as an argument rather than the constructor, and why an ``Agent``
owns no global state.
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Literal

from .actions import Done, Wait
from .browser import ActionError, BrowserSession
from .config import RunConfig
from .policy.base import AgentPolicy
from .trajectory import Step, Trajectory

logger = logging.getLogger(__name__)

RunStatus = Literal["done", "max_steps", "aborted"]


class RunResult:
    """Outcome of one episode."""

    def __init__(
        self,
        task: str,
        status: RunStatus,
        steps: list[Step],
        run_dir,
        answer: str | None = None,
    ) -> None:
        self.task = task
        self.status = status
        self.steps = steps
        self.run_dir = run_dir
        self.answer = answer

    @property
    def num_errors(self) -> int:
        return sum(1 for s in self.steps if s.error)

    def __repr__(self) -> str:
        return (
            f"RunResult(status={self.status!r}, steps={len(self.steps)}, "
            f"errors={self.num_errors}, answer={self.answer!r})"
        )


class Agent:
    """Drives one policy against one browser session."""

    def __init__(
        self,
        policy: AgentPolicy,
        browser: BrowserSession,
        config: RunConfig | None = None,
    ) -> None:
        self.policy = policy
        self.browser = browser
        self.config = config or browser.config

    async def run(
        self,
        task: str,
        start_url: str | None = None,
        trajectory: Trajectory | None = None,
    ) -> RunResult:
        """Run one episode and return its result.

        Failures are recorded on the step and the loop continues -- one bad
        click on a flaky page should not throw away the rest of the run. The
        exception is ``max_consecutive_errors`` back-to-back failures, which
        means something is genuinely stuck and further steps would just burn
        the budget.
        """
        traj = trajectory or Trajectory(task=task, config=self.config)
        await self.policy.reset()

        if start_url:
            await self.browser.goto(start_url)

        history: list[Step] = []
        status: RunStatus = "max_steps"
        answer: str | None = None
        consecutive_errors = 0

        for step_index in range(self.config.max_steps):
            # --- observe ---------------------------------------------------
            observe_started = perf_counter()
            screenshot = await self.browser.screenshot()
            page = await self.browser.page_info()
            browser_seconds = perf_counter() - observe_started

            # --- decide ----------------------------------------------------
            predict_started = perf_counter()
            try:
                step = await self.policy.predict(task, history, screenshot, page)
            except Exception as exc:
                # Record the failure as a short Wait rather than inventing an
                # action the policy never chose: the agent genuinely does pause
                # and retry, so the trajectory stays an honest record and the
                # wait doubles as backoff.
                logger.warning("policy failed at step %d: %s", step_index, exc)
                step = Step(
                    thought="policy error; backing off and retrying",
                    action=Wait(seconds=0.5),
                    error=f"policy.predict failed: {exc}",
                )
                step.index, step.url, step.title = step_index, page.url, page.title
                step.model_seconds = perf_counter() - predict_started
                step.browser_seconds = browser_seconds
                traj.record(step, screenshot)
                history.append(step)
                consecutive_errors += 1
                if consecutive_errors >= self.config.max_consecutive_errors:
                    status = "aborted"
                    break
                await self.browser.execute(step.action)
                continue

            # A throttled policy stamps model_queue_seconds onto the step; keep
            # it, and record the total around it here.
            step.model_seconds = perf_counter() - predict_started
            step.index = step_index
            step.url, step.title = page.url, page.title

            # --- act -------------------------------------------------------
            # Done is terminal, so it is recorded but never executed.
            if isinstance(step.action, Done):
                step.browser_seconds = browser_seconds
                traj.record(step, screenshot)
                history.append(step)
                status, answer = "done", step.action.answer
                break

            act_started = perf_counter()
            try:
                await self.browser.execute(step.action)
                consecutive_errors = 0
            except ActionError as exc:
                step.error = str(exc)
                consecutive_errors += 1
                logger.warning("action failed at step %d: %s", step_index, exc)
            except Exception as exc:  # unexpected: still do not kill the run
                step.error = f"unexpected error: {exc}"
                consecutive_errors += 1
                logger.exception("unexpected error at step %d", step_index)

            step.browser_seconds = browser_seconds + (perf_counter() - act_started)

            # Recorded after execution so ``error`` is populated, but paired
            # with the pre-action screenshot the policy actually saw.
            traj.record(step, screenshot)
            history.append(step)

            if consecutive_errors >= self.config.max_consecutive_errors:
                logger.error("aborting: %d consecutive failures", consecutive_errors)
                status = "aborted"
                break

        # --- wrap up -------------------------------------------------------
        try:
            traj.save_final_screenshot(await self.browser.screenshot())
        except Exception:  # pragma: no cover - page may be closed
            logger.debug("could not capture final screenshot", exc_info=True)

        traj.finalize(status=status, answer=answer)
        logger.info("run finished: status=%s steps=%d", status, len(history))
        return RunResult(
            task=task,
            status=status,
            steps=history,
            run_dir=traj.run_dir,
            answer=answer,
        )


async def run_task(
    task: str,
    policy: AgentPolicy,
    config: RunConfig | None = None,
    start_url: str | None = None,
) -> RunResult:
    """One-shot helper: open a browser, run the task, close everything."""
    config = config or RunConfig()
    async with BrowserSession(config) as browser:
        agent = Agent(policy=policy, browser=browser, config=config)
        return await agent.run(task, start_url=start_url)
