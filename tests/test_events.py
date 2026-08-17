"""The event sink and the replay stream. MockPolicy and a stub manager only.

Two things are being protected here:

1. **The sink is observe-only.** With nothing attached the orchestrator runs
   the same code and writes the same ``run.json`` -- every other test in the
   suite passes with the default sink, which is most of the proof; the tests
   here add that attaching one changes nothing about the run's outcome.

2. **Replay is complete.** A run reconstructed from ``run.json`` and the
   trajectory files must tell the same story as one recorded live: the same
   agents, steps, plan, waves and replans, in a consistent order. If it does
   not, the viewer would show two different runs depending on how the run was
   launched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("playwright", reason="playwright is not installed")

from multi_agent_web.actions import Done, Wait  # noqa: E402
from multi_agent_web.config import ManagerConfig, RunConfig  # noqa: E402
from multi_agent_web.manager import Manager  # noqa: E402
from multi_agent_web.orchestrator import (  # noqa: E402
    BestOfN,
    DagStrategy,
    EventSink,
    MockJudge,
    OrchestratorConfig,
    RecordingSink,
    load_events,
    orchestrate,
)
from multi_agent_web.orchestrator.events import Event, reconstruct_events  # noqa: E402
from multi_agent_web.policy.mock import MockPolicy  # noqa: E402

from test_manager import (  # noqa: E402
    NO_CHANGE,
    FakeChatClient,
    RecordingPolicy,
    plan_reply,
    replan_reply,
)


def run_config(tmp_path: Path) -> RunConfig:
    return RunConfig(headless=True, max_steps=6, runs_dir=tmp_path / "runs")


def mock_factory(index: int) -> MockPolicy:
    return MockPolicy([Wait(seconds=0.01), Done(answer=f"answer {index}")])


def types(events: list[Event]) -> list[str]:
    return [e.type for e in events]


# ---------------------------------------------------------------------------
# the sink
# ---------------------------------------------------------------------------
def test_the_default_sink_is_a_no_op() -> None:
    sink = EventSink()
    assert sink.emit("anything", x=1) is None  # and does not raise


def test_recording_sink_stamps_and_persists(tmp_path: Path) -> None:
    sink = RecordingSink(path=tmp_path / "events.jsonl")
    seen: list[Event] = []
    sink.subscribe(seen.append)
    sink.emit("run_started", task="t")
    sink.emit("agent_step", index=0, step={"index": 0, "path": Path("x")})
    sink.close()

    assert types(sink.events) == ["run_started", "agent_step"]
    assert sink.events[0].t == 0.0
    assert sink.events[1].t >= 0.0
    assert seen == sink.events
    # Paths were coerced: the file must be plain JSON.
    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["data"]["step"]["path"] == "x"


def test_a_failing_listener_is_dropped_not_propagated() -> None:
    sink = RecordingSink()

    def bad(_: Event) -> None:
        raise RuntimeError("viewer bug")

    sink.subscribe(bad)
    sink.emit("run_started")  # must not raise into the run
    sink.emit("run_finished")
    assert len(sink.events) == 2


@pytest.mark.asyncio
async def test_best_of_n_announces_the_whole_story(tmp_path: Path) -> None:
    sink = RecordingSink(path=tmp_path / "runs" / "events.jsonl")
    result = await orchestrate(
        task="find it",
        strategy=BestOfN(n=2, judge=MockJudge()),
        policy_factory=mock_factory,
        run_config=run_config(tmp_path),
        orchestrator_config=OrchestratorConfig(max_concurrent_browsers=2),
        sink=sink,
    )
    t = types(sink.events)
    assert t[0] == "run_started" and t[-1] == "run_finished"
    assert t.count("agent_started") == 2 and t.count("agent_finished") == 2
    assert t.count("agent_step") == 4  # two agents, two steps each
    assert "judge_decision" in t
    assert t.index("judge_decision") > max(i for i, x in enumerate(t) if x == "agent_finished")

    # Steps carry a screenshot path relative to the run dir that really exists.
    step = next(e for e in sink.events if e.type == "agent_step")
    assert step.data["step"]["screenshot"].startswith("agent_")
    assert (result.run_dir / step.data["step"]["screenshot"]).exists()
    assert step.data["step"]["action"]  # a one-line summary
    # And nothing on the bus is an image.
    assert all("base64" not in json.dumps(e.as_dict()) for e in sink.events)

    finished = sink.events[-1].data
    assert finished["answer"] == result.answer
    assert finished["timing"]["wall_seconds"] > 0
    # Timestamps are monotonic and start at zero.
    ts = [e.t for e in sink.events]
    assert ts == sorted(ts) and ts[0] == 0.0


@pytest.mark.asyncio
async def test_attaching_a_sink_does_not_change_the_run(tmp_path: Path) -> None:
    """Same policy, same task, with and without a sink: same run.json shape
    and outcome."""
    plain = await orchestrate(
        task="t", strategy=BestOfN(n=2), policy_factory=mock_factory,
        run_config=RunConfig(headless=True, max_steps=6, runs_dir=tmp_path / "a"),
    )
    sunk = await orchestrate(
        task="t", strategy=BestOfN(n=2), policy_factory=mock_factory,
        run_config=RunConfig(headless=True, max_steps=6, runs_dir=tmp_path / "b"),
        sink=RecordingSink(),
    )
    a = json.loads((plain.run_dir / "run.json").read_text())
    b = json.loads((sunk.run_dir / "run.json").read_text())
    for key in ("schema_version", "task", "strategy", "answer", "contributing_agents"):
        assert a[key] == b[key]
    assert [x["status"] for x in a["agents"]] == [x["status"] for x in b["agents"]]
    assert set(a) == set(b), "the sink added or removed a run.json field"


@pytest.mark.asyncio
async def test_dag_announces_plan_waves_and_replans(tmp_path: Path) -> None:
    manager = Manager(
        FakeChatClient(
            [
                plan_reply(("a", []), ("b", ["a"])),
                replan_reply(add=[("c", ["a"])], reason="a found more"),
                NO_CHANGE,
                NO_CHANGE,
            ]
        ),
        ManagerConfig(planning_budget=5),
    )
    sink = RecordingSink()
    await orchestrate(
        task="t",
        strategy=DagStrategy(manager),
        policy_factory=lambda i: RecordingPolicy(i, []),
        run_config=run_config(tmp_path),
        sink=sink,
    )
    t = types(sink.events)
    assert t[:2] == ["run_started", "plan_created"]
    assert t.count("wave_started") == 2 and t.count("wave_finished") == 2
    assert "replan_proposed" in t and "replan_applied" in t and "replan_skipped" in t
    # The applied replan comes after wave 1 finished and before wave 2 started.
    wave_starts = [i for i, x in enumerate(t) if x == "wave_started"]
    assert t.index("wave_finished") < t.index("replan_applied") < wave_starts[1]

    applied = next(e for e in sink.events if e.type == "replan_applied")
    assert [s["id"] for s in applied.data["add"]] == ["c"]
    assert {s["id"] for s in applied.data["dag"]["subtasks"]} == {"a", "b", "c"}
    assert applied.data["budget"]["spent"] == 1

    # wave_started carries the label that agent_started will carry: the join key.
    wave = next(e for e in sink.events if e.type == "wave_started")
    labels = {s["label"] for s in wave.data["subtasks"]}
    started = [e for e in sink.events if e.type == "agent_started"]
    assert labels <= {e.data["label"] for e in started}

    # The plan_created snapshot is all pending; the final wave_finished shows outcomes.
    plan = next(e for e in sink.events if e.type == "plan_created")
    assert {s["status"] for s in plan.data["dag"]["subtasks"]} == {"pending"}
    last_wave = [e for e in sink.events if e.type == "wave_finished"][-1]
    assert all(s["status"] == "done" for s in last_wave.data["dag"]["subtasks"])


# ---------------------------------------------------------------------------
# replay: recorded vs reconstructed
# ---------------------------------------------------------------------------
def _story(events: list[Event]) -> dict:
    """The parts of a stream a viewer's state depends on, order-insensitive
    where timing is derived."""
    agents = {}
    for e in events:
        if e.type == "agent_started":
            agents[e.data["index"]] = {"label": e.data["label"], "steps": 0, "status": None}
        elif e.type == "agent_step":
            agents[e.data["index"]]["steps"] += 1
        elif e.type == "agent_finished":
            agents[e.data["index"]]["status"] = e.data["status"]
    return {
        "agents": agents,
        "waves": [tuple(sorted(s["id"] for s in e.data["subtasks"])) for e in events if e.type == "wave_started"],
        "applied": [(e.data["wave"], tuple(s["id"] for s in e.data["add"])) for e in events if e.type == "replan_applied"],
        "plan": sorted(s["id"] for e in events if e.type == "plan_created" for s in e.data["dag"]["subtasks"]),
        "final_dag": {
            s["id"]: s["status"]
            for e in events if e.type in ("wave_finished", "replan_applied")
            for s in e.data["dag"]["subtasks"]
        } if any(e.type == "wave_finished" for e in events) else None,
        "answer": next((e.data["answer"] for e in events if e.type == "run_finished"), None),
        "judge": next((e.data.get("winner") for e in events if e.type == "judge_decision"), "n/a"),
    }


@pytest.mark.asyncio
async def test_reconstruction_matches_the_recording_for_best_of_n(tmp_path: Path) -> None:
    sink = RecordingSink()
    result = await orchestrate(
        task="find it", strategy=BestOfN(n=3, judge=MockJudge()), policy_factory=mock_factory,
        run_config=run_config(tmp_path), sink=sink,
    )
    rebuilt = reconstruct_events(result.run_dir)
    assert _story(rebuilt) == _story(sink.events)
    assert types(rebuilt)[0] == "run_started" and types(rebuilt)[-1] == "run_finished"
    ts = [e.t for e in rebuilt]
    assert ts == sorted(ts)
    # Every step's screenshot path resolves.
    for e in rebuilt:
        if e.type == "agent_step" and e.data["step"]["screenshot"]:
            assert (result.run_dir / e.data["step"]["screenshot"]).exists()


@pytest.mark.asyncio
async def test_reconstruction_matches_the_recording_for_dag(tmp_path: Path) -> None:
    manager = Manager(
        FakeChatClient(
            [
                plan_reply(("a", []), ("bad", []), ("join", ["a", "bad"])),
                replan_reply(add=[("join_a", ["a"])], reason="bad failed; join on a alone"),
                NO_CHANGE,
                NO_CHANGE,
            ]
        ),
        ManagerConfig(planning_budget=5),
    )
    sink = RecordingSink()
    result = await orchestrate(
        task="t",
        strategy=DagStrategy(manager),
        policy_factory=lambda i: RecordingPolicy(i, [], crash_on="bad"),
        run_config=run_config(tmp_path),
        sink=sink,
    )
    rebuilt = reconstruct_events(result.run_dir)
    assert _story(rebuilt) == _story(sink.events)

    # Structural order survives reconstruction: plan before wave 1, wave 1
    # finished before the replan, replan before wave 2, and every agent's
    # events inside its wave.
    t = types(rebuilt)
    starts = [i for i, x in enumerate(t) if x == "wave_started"]
    ends = [i for i, x in enumerate(t) if x == "wave_finished"]
    assert t.index("plan_created") < starts[0]
    assert ends[0] < t.index("replan_applied") < starts[1]
    for s, e in zip(starts, ends):
        inside = [ev for ev in rebuilt[s:e] if ev.type == "agent_started"]
        assert inside, "a wave with no agents inside it"

    # The blocked join is blocked in the reconstructed snapshots too.
    last = [e for e in rebuilt if e.type == "wave_finished"][-1]
    status = {s["id"]: s["status"] for s in last.data["dag"]["subtasks"]}
    assert status["bad"] == "failed" and status["join"] == "blocked" and status["join_a"] == "done"


@pytest.mark.asyncio
async def test_load_events_prefers_the_recording(tmp_path: Path) -> None:
    result = await orchestrate(
        task="t", strategy=BestOfN(n=1), policy_factory=mock_factory,
        run_config=run_config(tmp_path),
    )
    # No recording: reconstructed.
    assert types(load_events(result.run_dir))[0] == "run_started"
    # A recording exists: it wins verbatim, even if it says something else.
    (result.run_dir / "events.jsonl").write_text(
        json.dumps({"type": "run_started", "t": 0, "wall": "", "data": {"task": "recorded"}}) + "\n"
    )
    events = load_events(result.run_dir)
    assert len(events) == 1 and events[0].data["task"] == "recorded"


def test_load_events_on_an_empty_dir_is_empty(tmp_path: Path) -> None:
    assert load_events(tmp_path) == []
