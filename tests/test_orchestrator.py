"""Phase 2 orchestrator tests. MockPolicy only -- no GPU, no endpoint, no keys.

The concurrency tests assert on *overlap*, not merely on everything finishing:
N tasks that ran strictly one after another would also "all complete", so
completion proves nothing about parallelism.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Sequence
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter

import pytest

pytest.importorskip("playwright", reason="playwright is not installed")

from PIL import Image  # noqa: E402

from multi_agent_web.actions import Done, Wait  # noqa: E402
from multi_agent_web.browser import PageInfo  # noqa: E402
from multi_agent_web.config import RunConfig  # noqa: E402
from multi_agent_web.orchestrator import (  # noqa: E402
    BestOfN,
    BrowserPool,
    MockJudge,
    OrchestratorConfig,
    Runner,
    SessionSpec,
    orchestrate,
)
from multi_agent_web.orchestrator.judge.base import Candidate  # noqa: E402
from multi_agent_web.policy.base import AgentPolicy  # noqa: E402
from multi_agent_web.policy.mock import MockPolicy  # noqa: E402
from multi_agent_web.trajectory import Step  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):  # noqa: D102
        pass


@pytest.fixture(scope="module")
def static_server():
    """Serve tests/fixtures over http.

    Needed for the isolation test: localStorage is unavailable on file:// URLs
    in Chromium, so proving contexts are isolated requires a real origin.
    """
    handler = partial(_QuietHandler, directory=str(FIXTURES))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


class Tracker:
    """Records concurrency and per-agent intervals across policies."""

    def __init__(self) -> None:
        self.inflight = 0
        self.peak = 0
        self.intervals: list[tuple[float, float]] = []
        self.instances: list[int] = []

    def max_simultaneous(self) -> int:
        """Largest number of intervals overlapping at any instant."""
        events = [(s, 1) for s, _ in self.intervals] + [(e, -1) for _, e in self.intervals]
        events.sort()
        best = live = 0
        for _, delta in events:
            live += delta
            best = max(best, live)
        return best


class ProbePolicy(AgentPolicy):
    """MockPolicy plus instrumentation and an artificial model latency."""

    name = "probe"

    def __init__(self, script, tracker: Tracker, delay: float = 0.25) -> None:
        self.inner = MockPolicy(script)
        self.tracker = tracker
        self.delay = delay
        tracker.instances.append(id(self))

    async def predict(
        self,
        task: str,
        history: Sequence[Step],
        screenshot: Image.Image,
        page: PageInfo,
    ) -> Step:
        started = perf_counter()
        self.tracker.inflight += 1
        self.tracker.peak = max(self.tracker.peak, self.tracker.inflight)
        try:
            await asyncio.sleep(self.delay)  # stand in for generation
            return await self.inner.predict(task, history, screenshot, page)
        finally:
            self.tracker.inflight -= 1
            self.tracker.intervals.append((started, perf_counter()))

    async def reset(self) -> None:
        await self.inner.reset()


def simple_script(answer: str, filler_steps: int = 0):
    return [Wait(seconds=0.01) for _ in range(filler_steps)] + [Done(answer=answer)]


def run_config(tmp_path: Path, **kw) -> RunConfig:
    return RunConfig(headless=True, max_steps=8, runs_dir=tmp_path / "runs", **kw)


# --------------------------------------------------------------------------
# concurrency
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_sessions_actually_overlap(tmp_path: Path) -> None:
    """Three agents must run at the same time, not merely all finish."""
    tracker = Tracker()
    async with Runner(
        policy_factory=lambda i: ProbePolicy(simple_script(f"a{i}"), tracker, delay=0.3),
        run_config=run_config(tmp_path),
        orchestrator_config=OrchestratorConfig(
            max_concurrent_browsers=3, max_inflight_model_requests=None
        ),
    ) as runner:
        started = perf_counter()
        results = await runner.run_sessions([SessionSpec(task="t")] * 3)
        wall = perf_counter() - started

    assert [r.status for r in results] == ["done"] * 3
    # Overlap, three independent ways.
    assert tracker.max_simultaneous() == 3, "predict intervals never overlapped"
    assert tracker.peak == 3
    assert runner.peak_concurrent_browsers == 3
    # Serial execution would take >= 3 * 0.3s of model time alone.
    assert wall < 0.9, f"ran serially: {wall:.2f}s"


@pytest.mark.asyncio
async def test_model_slot_limit_is_respected(tmp_path: Path) -> None:
    """With one model slot, generations serialize even though browsers do not."""
    tracker = Tracker()
    async with Runner(
        policy_factory=lambda i: ProbePolicy(simple_script(f"a{i}"), tracker, delay=0.15),
        run_config=run_config(tmp_path),
        orchestrator_config=OrchestratorConfig(
            max_concurrent_browsers=4, max_inflight_model_requests=1
        ),
    ) as runner:
        results = await runner.run_sessions([SessionSpec(task="t")] * 4)

    assert all(r.status == "done" for r in results)
    assert tracker.peak == 1, "more than one generation was in flight"
    assert tracker.max_simultaneous() == 1
    assert runner.slot.peak_inflight == 1
    # Browsers still ran in parallel -- that is the whole point of two limits.
    assert runner.peak_concurrent_browsers == 4
    # Queueing was measured.
    assert sum(r.timing.model_queue_seconds for r in results) > 0


@pytest.mark.asyncio
async def test_browser_limit_is_respected(tmp_path: Path) -> None:
    tracker = Tracker()
    async with Runner(
        policy_factory=lambda i: ProbePolicy(simple_script(f"a{i}"), tracker, delay=0.1),
        run_config=run_config(tmp_path),
        orchestrator_config=OrchestratorConfig(
            max_concurrent_browsers=2, max_inflight_model_requests=None
        ),
    ) as runner:
        results = await runner.run_sessions([SessionSpec(task="t")] * 5)

    assert all(r.status == "done" for r in results)
    assert runner.peak_concurrent_browsers == 2
    assert tracker.peak <= 2, "model concurrency exceeded the browser limit"


# --------------------------------------------------------------------------
# isolation
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_browser_contexts_are_isolated(static_server: str) -> None:
    """State written in one session must be invisible to another."""
    url = f"{static_server}/demo_page.html"
    async with BrowserPool(RunConfig(headless=True)) as pool:
        first = pool.session()
        second = pool.session()
        await first.start()
        await second.start()
        try:
            await first.goto(url)
            await second.goto(url)

            await first.page.evaluate("localStorage.setItem('agent', 'one')")
            await first.page.evaluate("document.cookie = 'who=one; path=/'")

            assert await first.page.evaluate("localStorage.getItem('agent')") == "one"
            # Same origin, different context -> must not see the other's state.
            assert await second.page.evaluate("localStorage.getItem('agent')") is None
            assert "who=one" not in await second.page.evaluate("document.cookie")
        finally:
            await first.close()
            await second.close()


@pytest.mark.asyncio
async def test_each_session_gets_its_own_policy_instance(tmp_path: Path) -> None:
    tracker = Tracker()
    async with Runner(
        policy_factory=lambda i: ProbePolicy(simple_script(f"a{i}"), tracker, delay=0.01),
        run_config=run_config(tmp_path),
    ) as runner:
        await runner.run_sessions([SessionSpec(task="t")] * 3)

    assert len(tracker.instances) == 3
    assert len(set(tracker.instances)) == 3, "a policy instance was reused"


@pytest.mark.asyncio
async def test_shared_policy_instance_is_rejected(tmp_path: Path) -> None:
    """Sharing a stateful policy would interleave agents' histories."""
    shared = MockPolicy(simple_script("x"))
    async with Runner(
        policy_factory=lambda i: shared,
        run_config=run_config(tmp_path),
    ) as runner:
        results = await runner.run_sessions([SessionSpec(task="t")] * 2)

    statuses = [r.status for r in results]
    assert "crashed" in statuses
    crashed = next(r for r in results if r.status == "crashed")
    assert "same instance twice" in (crashed.error or "")


# --------------------------------------------------------------------------
# failure containment
# --------------------------------------------------------------------------
class ExplodingPolicy(AgentPolicy):
    """Fails during reset(), i.e. before the loop can defend itself."""

    name = "exploding"

    async def reset(self) -> None:
        raise RuntimeError("boom: simulated agent failure")

    async def predict(self, task, history, screenshot, page) -> Step:  # pragma: no cover
        raise AssertionError("should never be reached")


@pytest.mark.asyncio
async def test_one_crash_does_not_sink_the_run(tmp_path: Path) -> None:
    tracker = Tracker()

    def factory(index: int) -> AgentPolicy:
        if index == 1:
            return ExplodingPolicy()
        return ProbePolicy(simple_script(f"answer-{index}"), tracker, delay=0.01)

    async with Runner(
        policy_factory=factory,
        run_config=run_config(tmp_path),
    ) as runner:
        results = await runner.run_sessions([SessionSpec(task="t")] * 4)

    assert len(results) == 4
    statuses = {r.index: r.status for r in results}
    assert statuses[1] == "crashed"
    assert [statuses[i] for i in (0, 2, 3)] == ["done"] * 3

    crashed = results[1]
    assert "boom: simulated agent failure" in (crashed.error or "")
    assert crashed.answer is None
    # A partial run is still a run: the survivors kept their answers.
    assert results[0].answer == "answer-0"


# --------------------------------------------------------------------------
# judge
# --------------------------------------------------------------------------
def candidate(index: int, **kw) -> Candidate:
    base = dict(
        task="t", answer=f"a{index}", status="done", num_steps=3, num_errors=0
    )
    base.update(kw)
    return Candidate(index=index, **base)  # type: ignore[arg-type]


class TestMockJudge:
    @pytest.mark.asyncio
    async def test_prefers_fewest_steps_among_answers(self) -> None:
        verdict = await MockJudge().choose(
            "t",
            [candidate(0, num_steps=5), candidate(1, num_steps=2), candidate(2, num_steps=9)],
        )
        assert verdict.index == 1
        assert "2 step" in verdict.reason

    @pytest.mark.asyncio
    async def test_ignores_unfinished_runs_even_if_shorter(self) -> None:
        verdict = await MockJudge().choose(
            "t",
            [
                candidate(0, status="max_steps", num_steps=1),
                candidate(1, status="crashed", num_steps=0, answer=None),
                candidate(2, num_steps=7),
            ],
        )
        assert verdict.index == 2

    @pytest.mark.asyncio
    async def test_no_winner_when_nothing_finished(self) -> None:
        verdict = await MockJudge().choose(
            "t",
            [
                candidate(0, status="crashed", answer=None),
                candidate(1, status="max_steps"),
            ],
        )
        assert verdict.index is None
        assert "no agent terminated deliberately" in verdict.reason
        assert "1 crashed" in verdict.reason and "1 max_steps" in verdict.reason

    @pytest.mark.asyncio
    async def test_is_deterministic(self) -> None:
        cands = [candidate(i, num_steps=4) for i in range(4)]
        verdicts = [await MockJudge().choose("t", cands) for _ in range(5)]
        assert {v.index for v in verdicts} == {0}

    @pytest.mark.asyncio
    async def test_empty_answer_loses_to_a_real_one(self) -> None:
        verdict = await MockJudge().choose(
            "t", [candidate(0, answer="", num_steps=1), candidate(1, answer="found it")]
        )
        assert verdict.index == 1


# --------------------------------------------------------------------------
# strategy + report
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_best_of_n_end_to_end_and_run_json(tmp_path: Path) -> None:
    tracker = Tracker()

    def factory(index: int) -> AgentPolicy:
        if index == 2:
            return ExplodingPolicy()
        # agent 1 takes fewer steps than agent 0, so it should win.
        filler = 2 if index == 0 else 0
        return ProbePolicy(
            simple_script(f"answer-{index}", filler_steps=filler), tracker, delay=0.02
        )

    result = await orchestrate(
        task="find the thing",
        strategy=BestOfN(n=4, judge=MockJudge()),
        policy_factory=factory,
        run_config=run_config(tmp_path),
        orchestrator_config=OrchestratorConfig(
            max_concurrent_browsers=4, max_inflight_model_requests=2
        ),
    )

    assert result.answer == "answer-1"
    assert result.contributing_indices == [1]
    assert result.num_succeeded == 3

    # --- run.json is well-formed ------------------------------------------
    payload = json.loads((result.run_dir / "run.json").read_text())
    assert payload["schema_version"] == 1
    assert payload["task"] == "find the thing"
    assert payload["strategy"] == "best_of_n"
    assert payload["answer"] == "answer-1"
    assert payload["contributing_agents"] == [1]
    assert payload["reason"]
    assert payload["started_at"] and payload["finished_at"]

    agents = payload["agents"]
    assert len(agents) == 4
    assert [a["index"] for a in agents] == [0, 1, 2, 3]
    assert [a["dir"] for a in agents][:2] == ["agent_0", "agent_1"]
    crashed = agents[2]
    assert crashed["status"] == "crashed"
    assert "boom" in crashed["error"]

    timing = payload["timing"]
    for key in (
        "wall_seconds",
        "total_model_seconds",
        "total_model_queue_seconds",
        "total_browser_seconds",
        "sum_agent_wall_seconds",
        "max_agent_wall_seconds",
        "peak_concurrent_browsers",
        "peak_inflight_model_requests",
        "speedup_vs_serial",
    ):
        assert key in timing, f"run.json timing is missing {key!r}"
    assert timing["peak_inflight_model_requests"] <= 2
    assert timing["total_browser_seconds"] > 0

    # --- failures reach the judge, labelled --------------------------------
    judged = payload["details"]["candidates"]
    assert len(judged) == 4, "a failed agent was dropped before judging"
    assert {c["status"] for c in judged} == {"done", "crashed"}
    assert payload["details"]["judge"]["winner"] == 1
    assert payload["details"]["judge"]["name"] == "mock"

    # --- per-agent directories keep the Phase 1 format ---------------------
    for index in (0, 1, 3):
        agent_dir = result.run_dir / f"agent_{index}"
        assert (agent_dir / "steps.jsonl").exists()
        assert (agent_dir / "summary.json").exists()
        assert (agent_dir / "meta.json").exists()
        first_step = json.loads((agent_dir / "steps.jsonl").read_text().splitlines()[0])
        assert "model_seconds" in first_step and "browser_seconds" in first_step


@pytest.mark.asyncio
async def test_runner_is_reentrant_for_phase_3_waves(tmp_path: Path) -> None:
    """A DAG strategy runs wave after wave on one Runner; indices keep going."""
    tracker = Tracker()
    async with Runner(
        policy_factory=lambda i: ProbePolicy(simple_script(f"a{i}"), tracker, delay=0.01),
        run_config=run_config(tmp_path),
    ) as runner:
        first = await runner.run_sessions([SessionSpec(task="wave-1")] * 2)
        second = await runner.run_sessions([SessionSpec(task="wave-2")] * 2)

    assert [r.index for r in first] == [0, 1]
    assert [r.index for r in second] == [2, 3]
    assert all(r.status == "done" for r in first + second)
    assert (runner.run_dir / "agent_3" / "steps.jsonl").exists()
