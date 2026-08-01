"""Runs many agent sessions concurrently under two independent limits.

TWO LIMITS, TWO RESOURCES
=========================

``max_concurrent_browsers`` and ``max_inflight_model_requests`` gate different
things and must be tuned separately:

* **Browsers are local and memory-bound.** Each context costs RAM on this
  machine. The ceiling is "how many Chromium contexts fit".
* **The model is remote and usually GPU-bound.** One GPU generates one sequence
  at a time; a server with a pool of K predictors serves K. The ceiling is "how
  many generations the server can actually overlap".

Conflating them hides the interesting behaviour. With 8 browsers and 1 model
slot, all 8 agents browse in parallel but line up single-file at the model, and
wall-clock is dominated by queueing -- which is exactly what the per-step
``model_queue_seconds`` measures. That is the headline plot for this project:
how the model/browser split moves as N scales.

``Runner.run_sessions`` may be called repeatedly on one ``Runner``. Agent
indices keep incrementing, and the browser pool and semaphores are shared
across calls. That is deliberate: a Phase 3 DAG strategy runs one wave, reads
the results, then launches the next wave against the same substrate.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import BaseModel, Field

from ..config import RunConfig
from ..trajectory import make_run_dir
from .session import (
    AgentSession,
    BrowserPool,
    ModelSlot,
    PolicyFactory,
    SessionResult,
    SessionSpec,
    SessionTiming,
)

logger = logging.getLogger(__name__)

#: Bump when run.json changes shape. Consumers should check it.
RUN_JSON_SCHEMA_VERSION = 1


class OrchestratorConfig(BaseModel):
    """Concurrency limits. See the module docstring for why there are two."""

    max_concurrent_browsers: int | None = Field(
        default=4, description="Local, memory-bound. None means unlimited."
    )
    max_inflight_model_requests: int | None = Field(
        default=2, description="Shared model server, GPU-bound. None means unlimited."
    )


@dataclass
class OrchestrationTiming:
    wall_seconds: float = 0.0
    total_model_seconds: float = 0.0
    total_model_queue_seconds: float = 0.0
    total_browser_seconds: float = 0.0
    sum_agent_wall_seconds: float = 0.0
    max_agent_wall_seconds: float = 0.0
    peak_concurrent_browsers: int = 0
    peak_inflight_model_requests: int = 0

    @property
    def speedup_vs_serial(self) -> float:
        """Sum of agent wall-clock / actual wall-clock.

        The concurrency actually achieved. 1.0 means everything serialized;
        N means perfect overlap across N agents.
        """
        return self.sum_agent_wall_seconds / self.wall_seconds if self.wall_seconds else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "wall_seconds": round(self.wall_seconds, 3),
            "total_model_seconds": round(self.total_model_seconds, 3),
            "total_model_queue_seconds": round(self.total_model_queue_seconds, 3),
            "total_browser_seconds": round(self.total_browser_seconds, 3),
            "sum_agent_wall_seconds": round(self.sum_agent_wall_seconds, 3),
            "max_agent_wall_seconds": round(self.max_agent_wall_seconds, 3),
            "peak_concurrent_browsers": self.peak_concurrent_browsers,
            "peak_inflight_model_requests": self.peak_inflight_model_requests,
            "speedup_vs_serial": round(self.speedup_vs_serial, 3),
        }


class Runner:
    """Owns the browser pool, the two limits, and agent-index assignment."""

    def __init__(
        self,
        policy_factory: PolicyFactory,
        run_config: RunConfig | None = None,
        orchestrator_config: OrchestratorConfig | None = None,
        run_dir: Path | None = None,
    ) -> None:
        self.policy_factory = policy_factory
        self.run_config = run_config or RunConfig()
        self.config = orchestrator_config or OrchestratorConfig()
        self.run_dir = run_dir or make_run_dir(self.run_config.runs_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.pool = BrowserPool(self.run_config)
        self.slot = ModelSlot(self.config.max_inflight_model_requests)
        limit = self.config.max_concurrent_browsers
        self._browser_sem = asyncio.Semaphore(limit) if limit and limit > 0 else None

        self._next_index = 0
        self._live_browsers = 0
        self.peak_concurrent_browsers = 0
        # Strong references so that id()-based duplicate detection stays valid.
        self._issued_policies: list[Any] = []
        self._issued_ids: set[int] = set()

    # --- lifecycle ---------------------------------------------------------

    async def __aenter__(self) -> "Runner":
        await self.pool.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.pool.close()

    # --- scheduling --------------------------------------------------------

    def _build_policy(self, index: int):
        policy = self.policy_factory(index)
        if id(policy) in self._issued_ids:
            raise RuntimeError(
                f"policy factory returned the same instance twice (agent {index}). "
                "Each concurrent agent needs its own policy: stateful policies "
                "like MolmoWebPolicy carry per-episode history, and sharing one "
                "would interleave the agents' prompts. Return a NEW instance."
            )
        self._issued_ids.add(id(policy))
        self._issued_policies.append(policy)
        return policy

    async def _run_one(self, index: int, spec: SessionSpec) -> SessionResult:
        if self._browser_sem is not None:
            await self._browser_sem.acquire()
        self._live_browsers += 1
        self.peak_concurrent_browsers = max(
            self.peak_concurrent_browsers, self._live_browsers
        )
        try:
            session = AgentSession(
                index=index,
                spec=spec,
                policy=self._build_policy(index),
                pool=self.pool,
                run_config=self.run_config,
                run_dir=self.run_dir / f"agent_{index}",
                slot=self.slot,
            )
            return await session.run()
        finally:
            self._live_browsers -= 1
            if self._browser_sem is not None:
                self._browser_sem.release()

    async def run_sessions(self, specs: Sequence[SessionSpec]) -> list[SessionResult]:
        """Run every spec concurrently; return results in spec order.

        One agent crashing does not sink the others -- ``AgentSession.run``
        already converts exceptions into ``crashed`` results, and the
        ``return_exceptions`` guard below catches anything that slips past
        (a failure to even build the policy, say) and reports it the same way.
        """
        if not specs:
            return []

        indices = []
        for _ in specs:
            indices.append(self._next_index)
            self._next_index += 1

        tasks = [
            asyncio.create_task(self._run_one(i, spec), name=f"agent-{i}")
            for i, spec in zip(indices, specs)
        ]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[SessionResult] = []
        for index, spec, outcome in zip(indices, specs, gathered):
            if isinstance(outcome, BaseException):
                logger.error("agent %d failed outside the session: %s", index, outcome)
                results.append(
                    SessionResult(
                        index=index,
                        spec=spec,
                        status="crashed",
                        error=f"{type(outcome).__name__}: {outcome}",
                    )
                )
            else:
                results.append(outcome)
        return results


@dataclass
class OrchestrationResult:
    """Everything a caller, the UI, or run.json needs about one multi-agent run."""

    task: str
    strategy: str
    answer: str | None
    contributing_indices: list[int]
    reason: str
    sessions: list[SessionResult]
    timing: OrchestrationTiming
    run_dir: Path
    details: dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""

    @property
    def num_succeeded(self) -> int:
        return sum(1 for s in self.sessions if s.succeeded)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RUN_JSON_SCHEMA_VERSION,
            "task": self.task,
            "strategy": self.strategy,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "answer": self.answer,
            "contributing_agents": self.contributing_indices,
            "reason": self.reason,
            "details": self.details,
            "agents": [s.as_dict() for s in self.sessions],
            "timing": self.timing.as_dict(),
        }


def summarise_timing(
    sessions: Sequence[SessionResult],
    wall_seconds: float,
    peak_browsers: int,
    peak_model: int,
) -> OrchestrationTiming:
    walls = [s.timing.wall_seconds for s in sessions] or [0.0]
    return OrchestrationTiming(
        wall_seconds=wall_seconds,
        total_model_seconds=sum(s.timing.model_seconds for s in sessions),
        total_model_queue_seconds=sum(s.timing.model_queue_seconds for s in sessions),
        total_browser_seconds=sum(s.timing.browser_seconds for s in sessions),
        sum_agent_wall_seconds=sum(walls),
        max_agent_wall_seconds=max(walls),
        peak_concurrent_browsers=peak_browsers,
        peak_inflight_model_requests=peak_model,
    )


def write_run_json(result: OrchestrationResult) -> Path:
    """Write ``run.json`` at the top of the run directory.

    Stable format -- see ``docs/run_json.md``. The UI and the demo tooling both
    read this, so add fields rather than renaming them, and bump
    ``RUN_JSON_SCHEMA_VERSION`` if the shape changes.
    """
    path = result.run_dir / "run.json"
    path.write_text(
        json.dumps(result.as_dict(), indent=2, default=str), encoding="utf-8"
    )
    return path


async def orchestrate(
    task: str,
    strategy,
    policy_factory: PolicyFactory,
    run_config: RunConfig | None = None,
    orchestrator_config: OrchestratorConfig | None = None,
    run_dir: Path | None = None,
    write_report: bool = True,
) -> OrchestrationResult:
    """Top-level entry point: run one strategy end to end and report.

    ``strategy`` is any :class:`~.strategy.base.Strategy`. It gets the Runner
    and decides how many sessions to launch and with what tasks -- best-of-N
    launches N copies of one task; a Phase 3 DAG strategy will launch waves of
    different subtasks against the same Runner.
    """
    run_config = run_config or RunConfig()
    started = perf_counter()
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    async with Runner(
        policy_factory=policy_factory,
        run_config=run_config,
        orchestrator_config=orchestrator_config,
        run_dir=run_dir,
    ) as runner:
        outcome = await strategy.run(task, runner)

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
        run_dir=runner.run_dir,
        details=outcome.details,
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    if write_report:
        write_run_json(result)
    return result


__all__ = [
    "OrchestrationResult",
    "OrchestrationTiming",
    "OrchestratorConfig",
    "RUN_JSON_SCHEMA_VERSION",
    "Runner",
    "SessionSpec",
    "SessionResult",
    "SessionTiming",
    "orchestrate",
    "summarise_timing",
    "write_run_json",
]
