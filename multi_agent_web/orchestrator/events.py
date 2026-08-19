"""Events: what an orchestrated run announces as it happens, and how a UI hears it.

OBSERVE-ONLY, AND OFF BY DEFAULT
================================
``EventSink`` is the seam. The orchestrator calls ``sink.emit(...)`` at a dozen
points -- an agent started, an agent took a step, the manager replanned -- and
the base class ignores every one of them. ``Runner`` and ``orchestrate`` build
that base class when nothing is passed, so an unattached run executes exactly
the code it did before this module existed: same paths, same ``run.json``,
same tests. Nothing here may influence a run; a sink that raised would break a
run for the sake of a viewer, so ``emit`` on the recording sink swallows its own
listener errors.

WHY A SINK AND NOT TAILING THE TRAJECTORY FILES
===============================================
Tailing ``steps.jsonl`` was the tempting alternative -- it already exists and
needs no plumbing. It fails on the events that matter most for Phase 3: the
manager's. A decomposition, a wave boundary and a replan never pass through the
trajectory writer, and a UI that only tails agent files would learn about the
plan when the first agent moved and about a replan never. So the sink is a bus
that everything with something to say can put a message on, and the trajectory
writer is one of its sources rather than the source.

TWO WAYS TO GET AN EVENT STREAM, ONE SHAPE
==========================================
* Live: attach a ``RecordingSink`` and read ``.events`` / subscribe.
* Replay: ``load_events(run_dir)`` -- which reads ``events.jsonl`` if a
  recording sink wrote one, and otherwise **reconstructs** the stream from
  ``run.json`` plus the per-agent trajectory directories. Every run ever made
  is therefore replayable, including the ones from before this module existed,
  and a viewer needs a single code path for both.

Reconstructed timing is honest about its precision. Steps within one agent are
placed by their measured model+browser durations (sub-millisecond, from
``perf_counter``); agents are aligned to each other by their trajectories'
``started_at`` stamps, which are millisecond-resolution from now on but were
whole seconds in older runs. That is why the timeline is a scrubber over
events rather than a clock you can trust to the frame.

EVENT VOCABULARY
================
    run_started      task, strategy
    plan_created     dag                              (dag strategy)
    wave_started     wave, subtasks[{id,label}], dag  (dag strategy)
    agent_started    index, label, task, start_url
    agent_step       index, step{index, thought, action, action_json, error,
                                 screenshot, url, title, model_seconds, ...}
    agent_finished   index, status, answer, error, num_steps, num_errors, timing
    wave_finished    wave, statuses, blocked, dag     (dag strategy)
    replan_proposed  wave, reason, add, remove, edits (dag strategy)
    replan_applied   wave, reason, add, remove, edits, dag, budget
    replan_refused   wave, outcome, reason, ...       (invalid, unaffordable, failed)
    replan_skipped   wave, outcome, reason, called_model  (no change / budget spent)
    judge_decision   name, winner, reason, candidates (best-of-N)
    run_finished     answer, reason, contributing_indices, timing, usage
    run_failed       error                            (emitted by a driver, not the orchestrator)

``screenshot`` is a path relative to the run directory (``agent_2/step_004.png``).
The image itself never travels on the bus -- a four-agent run would flood any
socket -- the viewer fetches it by path.

Every ``dag`` snapshot carries each subtask's ``retry_of`` (see
``manager.plan``), including the reconstructed ones, so a viewer can group the
attempts of one logical subtask no matter which way it got the stream. It is
``null`` throughout for a run recorded before the field existed.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)

Listener = Callable[["Event"], None]


@dataclass
class Event:
    """One announcement. ``t`` is seconds since the first event of the run."""

    type: str
    t: float
    wall: str
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"type": self.type, "t": self.t, "wall": self.wall, "data": self.data}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Event":
        return cls(
            type=str(payload["type"]),
            t=float(payload.get("t", 0.0)),
            wall=str(payload.get("wall", "")),
            data=dict(payload.get("data") or {}),
        )


class EventSink:
    """Where a run reports what it is doing. This base class discards it all.

    Deliberately not abstract: the no-op IS the default, and constructing it is
    what ``Runner`` does when no sink is given. Subclass and override ``emit``
    to listen. ``emit`` is synchronous on purpose -- it is called from inside
    tight paths (every recorded step) and from synchronous code (the trajectory
    writer), and an ``await`` there would ripple through signatures that have
    no other reason to change.
    """

    def emit(self, type: str, **data: Any) -> None:  # noqa: A002 - mirrors Event.type
        """Announce ``type`` with ``data``. Must never raise into the run."""
        return None


class RecordingSink(EventSink):
    """Keeps every event, stamps it, optionally writes it, and fans it out.

    ``t`` is measured from the first ``emit`` -- ``run_started`` under
    ``orchestrate`` -- so replaying the list from ``t=0`` reproduces the run's
    own clock. ``path`` (usually ``<run_dir>/events.jsonl``) is appended to as
    events arrive, so a run that dies mid-way still leaves its story on disk.

    Listeners are called synchronously, in registration order, and a listener
    that raises is logged and dropped rather than allowed to break the run.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.events: list[Event] = []
        self.path = path
        self._t0: float | None = None
        self._listeners: list[Listener] = []
        self._fh = None

    @property
    def elapsed(self) -> float:
        return 0.0 if self._t0 is None else perf_counter() - self._t0

    def subscribe(self, listener: Listener) -> Listener:
        self._listeners.append(listener)
        return listener

    def unsubscribe(self, listener: Listener) -> None:
        try:
            self._listeners.remove(listener)
        except ValueError:
            pass

    def emit(self, type: str, **data: Any) -> None:  # noqa: A002
        now = perf_counter()
        if self._t0 is None:
            self._t0 = now
        event = Event(
            type=type,
            t=round(now - self._t0, 3),
            wall=_now_iso(),
            data=_jsonable(data),
        )
        self.events.append(event)
        self._write(event)
        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception:  # pragma: no cover - defensive by design
                logger.exception("event listener failed; dropping it")
                self.unsubscribe(listener)

    def _write(self, event: Event) -> None:
        if self.path is None:
            return
        try:
            if self._fh is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = self.path.open("a", encoding="utf-8")
            self._fh.write(json.dumps(event.as_dict(), default=str) + "\n")
            self._fh.flush()
        except OSError:  # pragma: no cover - a full disk must not kill the run
            logger.exception("could not write %s; events stay in memory only", self.path)
            self.path = None

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None

    def by_type(self, type: str) -> list[Event]:  # noqa: A002
        return [e for e in self.events if e.type == type]


# ---------------------------------------------------------------------------
# reading a stream back
# ---------------------------------------------------------------------------
def load_events(run_dir: Path) -> list[Event]:
    """The event stream for a finished run, from disk.

    Prefers ``events.jsonl`` when a recording sink wrote one -- that is the run
    exactly as it was seen live. Falls back to reconstructing the stream from
    ``run.json`` and the agent trajectories, which is what makes every run
    ever made replayable rather than only the ones launched with a sink.
    """
    run_dir = Path(run_dir)
    recorded = run_dir / "events.jsonl"
    if recorded.exists():
        events = _read_jsonl_events(recorded)
        # A recording that stops before run_finished is a run that died. Still
        # worth replaying -- but only if it got as far as starting.
        if events:
            return events
    if (run_dir / "run.json").exists():
        return reconstruct_events(run_dir)
    return []


def _read_jsonl_events(path: Path) -> list[Event]:
    events: list[Event] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(Event.from_dict(json.loads(line)))
        except (ValueError, KeyError):
            logger.warning("skipping a malformed line in %s", path)
    return events


def reconstruct_events(run_dir: Path) -> list[Event]:
    """Rebuild the event stream of a run from its artifacts alone.

    Sources, and what each supplies:

    * ``run.json`` -- the task, every agent's outcome and timing, the judge's
      verdict, and for a DAG run the initial graph, each wave, each replan and
      the final graph.
    * ``agent_<i>/meta.json`` -- when that agent started (aligns agents).
    * ``agent_<i>/steps.jsonl`` -- every step with its thought, action, error,
      screenshot filename and measured durations.

    Timing is derived, not recorded, so it is derived conservatively: an
    agent's steps are spread over its own ``wall_seconds`` in proportion to
    their measured model+browser durations, and wave boundaries are forced to
    nest correctly even when two agents' second-resolution start stamps tie.
    """
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    run_start = _parse_iso(run.get("started_at"))
    out = _EventList()

    out.add(0.0, "run_started", task=run.get("task"), strategy=run.get("strategy"))

    # --- agents -------------------------------------------------------------
    # Align agents by their own start stamps, relative to the earliest one, so
    # that agents that started together stay together even when the run-level
    # stamp is coarser than theirs.
    agent_rows = run.get("agents") or []
    starts: dict[int, float] = {}
    for row in agent_rows:
        starts[row["index"]] = _agent_started_at(run_dir, row)
    known = [s for s in starts.values() if s is not None]
    earliest = min(known) if known else run_start
    base = 0.0
    if run_start is not None and earliest is not None:
        base = max(0.0, earliest - run_start)

    finish_t: dict[int, float] = {}
    start_t: dict[int, float] = {}
    for row in agent_rows:
        index = row["index"]
        stamp = starts.get(index)
        start = base + ((stamp - earliest) if (stamp and earliest) else 0.0)
        wall = float((row.get("timing") or {}).get("wall_seconds") or 0.0)
        steps = _load_steps(run_dir, row.get("dir"))
        durations = [
            float(s.get("model_seconds") or 0.0) + float(s.get("browser_seconds") or 0.0)
            for s in steps
        ]
        total = sum(durations)
        # Spread measured durations over the agent's wall-clock. If nothing was
        # measured, space steps evenly -- still monotonic, still bounded.
        if total > 0 and wall > 0:
            scale = wall / total
        else:
            scale = 1.0
        start_t[index] = start
        out.add(
            start,
            "agent_started",
            index=index,
            label=row.get("label"),
            task=row.get("task"),
            start_url=None,
        )
        elapsed = 0.0
        for k, step in enumerate(steps):
            if total > 0:
                elapsed += durations[k] * scale
            elif wall > 0 and steps:
                elapsed = wall * (k + 1) / (len(steps) + 1)
            else:
                elapsed += 0.001
            out.add(start + elapsed, "agent_step", index=index, step=_step_payload(step, row.get("dir")))
        finish = start + max(wall, elapsed + 0.001)
        finish_t[index] = finish
        out.add(
            finish,
            "agent_finished",
            index=index,
            status=row.get("status"),
            answer=row.get("answer"),
            error=row.get("error"),
            num_steps=row.get("num_steps", len(steps)),
            num_errors=row.get("num_errors", 0),
            timing=row.get("timing"),
        )

    details = run.get("details") or {}
    strategy = run.get("strategy")

    # --- best-of-N: the judge -----------------------------------------------
    if strategy == "best_of_n" and details.get("judge"):
        judge = details["judge"]
        after = max(finish_t.values(), default=0.0)
        out.add(
            after + 0.01,
            "judge_decision",
            name=judge.get("name"),
            winner=judge.get("winner"),
            reason=judge.get("reason"),
            candidates=details.get("candidates") or [],
        )

    # --- dag: plan, waves, replans -----------------------------------------
    if strategy == "dag" and details.get("initial_dag"):
        _reconstruct_dag_events(out, details, start_t, finish_t)

    timing = run.get("timing") or {}
    end = float(timing.get("wall_seconds") or 0.0)
    end = max(end, max(finish_t.values(), default=0.0) + 0.01, out.last_t + 0.01)
    out.add(
        end,
        "run_finished",
        answer=run.get("answer"),
        reason=run.get("reason"),
        contributing_indices=run.get("contributing_agents") or [],
        timing=timing,
        usage=run.get("usage"),
    )
    return out.sorted()


def _reconstruct_dag_events(
    out: "_EventList",
    details: dict[str, Any],
    start_t: dict[int, float],
    finish_t: dict[int, float],
) -> None:
    """DAG events, with intermediate graph snapshots rebuilt from the record.

    ``run.json`` keeps the initial graph, the final graph (with every subtask's
    wave, agent and outcome), each wave's members and what it blocked, and each
    replan's edits. That is enough to replay the graph's state at every point:
    structure = initial + applied replans so far; status = final status once a
    subtask's wave has run, ``running`` during it, ``blocked`` from the wave
    that blocked it, ``pending`` otherwise.
    """
    initial = details["initial_dag"]
    final_by_id = {s["id"]: s for s in (details.get("final_dag") or {}).get("subtasks", [])}
    waves = details.get("waves") or []
    replans = details.get("replans") or []

    blocked_since: dict[str, int] = {}
    for w in waves:
        for sid in w.get("blocked") or []:
            blocked_since.setdefault(sid, int(w["wave"]))

    # Structure over time: which subtasks exist after wave w's replans.
    applied_by_wave: dict[int, list[dict[str, Any]]] = {}
    for entry in replans:
        if entry.get("applied") and entry.get("wave") is not None:
            applied_by_wave.setdefault(int(entry["wave"]), []).append(entry)

    def structure_after(wave: int) -> list[dict[str, Any]]:
        """Subtask records (id, instruction, depends_on) after wave ``wave``'s
        replans -- wave 0 means the initial plan."""
        current: dict[str, dict[str, Any]] = {
            s["id"]: {"id": s["id"], "instruction": s["instruction"],
                      "depends_on": list(s.get("depends_on") or []),
                      "retry_of": s.get("retry_of")}
            for s in initial["subtasks"]
        }
        order = [s["id"] for s in initial["subtasks"]]
        for w in sorted(applied_by_wave):
            if w > wave:
                break
            for entry in applied_by_wave[w]:
                for sid in entry.get("remove") or []:
                    current.pop(sid, None)
                    if sid in order:
                        order.remove(sid)
                for added in entry.get("add") or []:
                    current[added["id"]] = {
                        "id": added["id"], "instruction": added["instruction"],
                        "depends_on": list(added.get("depends_on") or []),
                        "retry_of": added.get("retry_of"),
                    }
                    if added["id"] not in order:
                        order.append(added["id"])
        return [current[i] for i in order if i in current]

    def snapshot(structure_wave: int, running_wave: int | None, done_through: int) -> dict[str, Any]:
        subtasks = []
        for rec in structure_after(structure_wave):
            fin = final_by_id.get(rec["id"], {})
            fwave = fin.get("wave")
            status = "pending"
            extra: dict[str, Any] = {"agent_index": None, "wave": None, "answer": None, "error": None}
            if fwave is not None and fwave <= done_through:
                status = fin.get("status", "done")
                extra = {k: fin.get(k) for k in ("agent_index", "wave", "answer", "error")}
            elif running_wave is not None and fwave == running_wave:
                status = "running"
                extra["wave"] = fwave
                extra["agent_index"] = fin.get("agent_index")
            elif rec["id"] in blocked_since and blocked_since[rec["id"]] <= done_through:
                status = "blocked"
                extra["error"] = fin.get("error")
            subtasks.append({**rec, "status": status, **extra})
        return {"subtasks": _topological(subtasks), "counts": _counts(subtasks)}

    # plan_created: just before the first wave's agents move.
    first_wave_agents = [
        s.get("agent_index") for s in final_by_id.values() if s.get("wave") == 1
    ]
    first_start = min(
        (start_t[i] for i in first_wave_agents if i in start_t), default=0.0
    )
    t_plan = max(0.001, first_start - 0.05)
    out.add(t_plan, "plan_created", dag=snapshot(0, None, 0))

    floor = t_plan
    for w in waves:
        wave = int(w["wave"])
        members = list(w.get("subtasks") or [])
        agents = [final_by_id[m].get("agent_index") for m in members if m in final_by_id]
        agents = [a for a in agents if a is not None]
        min_start = min((start_t[a] for a in agents if a in start_t), default=floor + 0.02)
        max_finish = max((finish_t[a] for a in agents if a in finish_t), default=min_start + 0.02)
        t_start = max(min_start - 0.01, floor + 0.01)
        # Second-resolution stamps can put wave 2's start on top of wave 1's
        # end; nudge the wave's agents forward so the structure nests.
        shift = max(0.0, t_start + 0.01 - min_start)
        if shift:
            for a in agents:
                out.shift_agent(a, shift)
            max_finish += shift
        out.add(
            t_start,
            "wave_started",
            wave=wave,
            subtasks=[{"id": m, "label": f"wave {wave}: {m}"} for m in members],
            dag=snapshot(wave - 1, wave, wave - 1),
        )
        t_end = max_finish + 0.01
        out.add(
            t_end,
            "wave_finished",
            wave=wave,
            statuses=w.get("statuses") or {},
            blocked=w.get("blocked") or [],
            dag=snapshot(wave - 1, None, wave),
        )
        floor = t_end
        # Replans decided after this wave, in the order they were recorded.
        k = 0
        for entry in replans:
            if entry.get("wave") != wave:
                continue
            k += 1
            t = floor + 0.005 * k
            common = {
                "wave": wave,
                "reason": entry.get("reason", ""),
                "add": entry.get("add") or [],
                "remove": entry.get("remove") or [],
                "edits": entry.get("edits", 0),
            }
            if entry.get("applied"):
                out.add(t, "replan_proposed", **common)
                out.add(t + 0.001, "replan_applied", **common,
                        dag=snapshot(wave, None, wave), budget=None)
            elif entry.get("called_model") and (entry.get("add") or entry.get("remove")):
                out.add(t, "replan_proposed", **common)
                out.add(t + 0.001, "replan_refused", **common, outcome=entry.get("outcome", ""))
            elif str(entry.get("outcome", "")).startswith(("failed", "rejected")):
                out.add(t, "replan_refused", **common, outcome=entry.get("outcome", ""))
            else:
                out.add(t, "replan_skipped", **common,
                        outcome=entry.get("outcome") or entry.get("reason", ""),
                        called_model=bool(entry.get("called_model")))
            floor = t + 0.002
        # Replan entries recorded without a wave (older runs): attach to the
        # last wave so they still appear.
    orphans = [e for e in replans if e.get("wave") is None]
    for k, entry in enumerate(orphans, start=1):
        out.add(floor + 0.005 * k, "replan_skipped", wave=None,
                reason=entry.get("reason", ""), outcome=entry.get("outcome", ""),
                called_model=bool(entry.get("called_model")), add=[], remove=[], edits=0)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class _EventList:
    """Accumulates events with a sequence number so ties sort stably, and
    lets a whole agent's events be nudged in time after the fact."""

    def __init__(self) -> None:
        self._items: list[tuple[float, int, Event]] = []
        self.last_t = 0.0

    def add(self, t: float, type: str, **data: Any) -> Event:  # noqa: A002
        t = round(max(0.0, float(t)), 3)
        event = Event(type=type, t=t, wall="", data=_jsonable(data))
        self._items.append((t, len(self._items), event))
        self.last_t = max(self.last_t, t)
        return event

    def shift_agent(self, index: int, delta: float) -> None:
        for i, (t, seq, ev) in enumerate(self._items):
            if ev.type.startswith("agent_") and ev.data.get("index") == index:
                ev.t = round(t + delta, 3)
                self._items[i] = (ev.t, seq, ev)
                self.last_t = max(self.last_t, ev.t)

    def sorted(self) -> list[Event]:
        return [ev for _, _, ev in sorted(self._items, key=lambda x: (x[0], x[1]))]


def _agent_started_at(run_dir: Path, row: dict[str, Any]) -> float | None:
    if not row.get("dir"):
        return None
    meta = run_dir / row["dir"] / "meta.json"
    if not meta.exists():
        return None
    try:
        return _parse_iso(json.loads(meta.read_text(encoding="utf-8")).get("started_at"))
    except (ValueError, OSError):
        return None


def _load_steps(run_dir: Path, agent_dir: str | None) -> list[dict[str, Any]]:
    if not agent_dir:
        return []
    path = run_dir / agent_dir / "steps.jsonl"
    if not path.exists():
        return []
    steps = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                steps.append(json.loads(line))
            except ValueError:
                continue
    return steps


def _step_payload(step: dict[str, Any], agent_dir: str | None) -> dict[str, Any]:
    action = step.get("action") or {}
    shot = step.get("screenshot")
    return {
        "index": step.get("index"),
        "thought": step.get("thought") or "",
        "action": _summarise_action(action),
        "action_json": action,
        "error": step.get("error"),
        "screenshot": f"{agent_dir}/{shot}" if (agent_dir and shot) else None,
        "url": step.get("url"),
        "title": step.get("title"),
        "model_seconds": step.get("model_seconds"),
        "model_queue_seconds": step.get("model_queue_seconds"),
        "browser_seconds": step.get("browser_seconds"),
    }


def _summarise_action(action: dict[str, Any]) -> str:
    """A one-line rendering of a serialized action, for viewers.

    Mirrors ``Action.summary()`` closely enough to read the same, without
    round-tripping through the pydantic models -- an old run may hold an action
    shape the current models no longer accept, and a viewer should still show
    it.
    """
    kind = action.get("type", "?")
    if kind == "click":
        return f"click({_num(action.get('x'))}, {_num(action.get('y'))})"
    if kind == "type":
        text = str(action.get("text", ""))
        return f"type({text!r})" + (" + Enter" if action.get("press_enter") else "")
    if kind == "scroll":
        return f"scroll(dx={_num(action.get('delta_x'))}, dy={_num(action.get('delta_y'))})"
    if kind == "key_press":
        return f"key_press({action.get('key')!r})"
    if kind == "navigate":
        return f"navigate({action.get('url')!r})"
    if kind == "wait":
        return f"wait({action.get('seconds')}s)"
    if kind == "done":
        return f"done({str(action.get('answer', ''))!r})"
    return kind


def _num(value: Any) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(f)) if f == int(f) else f"{f:.1f}"


def _topological(subtasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {s["id"]: s for s in subtasks}
    placed: set[str] = set()
    order: list[dict[str, Any]] = []
    remaining = list(subtasks)
    while remaining:
        layer = [s for s in remaining if all(d in placed or d not in by_id for d in s["depends_on"])]
        if not layer:
            order.extend(remaining)
            break
        order.extend(layer)
        placed.update(s["id"] for s in layer)
        remaining = [s for s in remaining if s["id"] not in placed]
    return order


def _counts(subtasks: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in subtasks:
        counts[s["status"]] = counts.get(s["status"], 0) + 1
    return counts


def _parse_iso(stamp: str | None) -> float | None:
    if not stamp:
        return None
    try:
        text = stamp.replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _jsonable(value: Any) -> Any:
    """Coerce to plain JSON types. Paths and unknown objects become strings so
    a stray ``Path`` in an event payload cannot break a socket."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "as_dict"):
        return _jsonable(value.as_dict())
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    return str(value)


__all__ = [
    "Event",
    "EventSink",
    "RecordingSink",
    "load_events",
    "reconstruct_events",
]
