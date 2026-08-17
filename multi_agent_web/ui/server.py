"""FastAPI backend for the viewer. Two modes, one event stream.

REPLAY reads ``runs/``. ``GET /api/runs`` lists what is there; ``GET
/api/runs/{id}`` returns the run's summary and its full event stream, loaded
by ``orchestrator.events.load_events`` -- ``events.jsonl`` if the run was
launched from here, otherwise reconstructed from ``run.json`` and the agent
trajectories. Screenshots are served by path from the run directory. No API
key is involved anywhere in replay.

LIVE starts a run. ``POST /api/runs`` validates a ``LaunchSpec``, prices it,
creates the run directory, attaches a ``RecordingSink`` that both writes
``events.jsonl`` and fans out to WebSocket subscribers, and launches
``orchestrate`` as a background task on the same event loop as the server.
``WS /api/runs/{id}/live`` replays whatever the sink has buffered and then
tails it, so a viewer that connects late sees the whole run. When the run
finishes it is a directory in ``runs/`` like any other, and replay can load it.

The orchestrator is not modified for any of this. The server is a client of
``launch.prepare`` and ``orchestrate`` exactly as the CLIs are; the only thing
it adds is the sink.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

from ..launch import LaunchError, LaunchSpec, estimate_calls, prepare
from ..orchestrator.events import Event, RecordingSink, load_events
from ..orchestrator.runner import orchestrate
from ..presets import presets_as_dicts
from ..trajectory import make_run_dir

logger = logging.getLogger(__name__)

INDEX_HTML = Path(__file__).with_name("index.html")

#: The event types after which a live stream has nothing more to say.
TERMINAL_EVENTS = {"run_finished", "run_failed"}


@dataclass
class LiveRun:
    """A run started from the UI, while its process lives in this server."""

    id: str
    spec: LaunchSpec
    run_dir: Path
    sink: RecordingSink
    task: asyncio.Task | None = None
    done: bool = False
    error: str | None = None
    _queues: set[asyncio.Queue] = field(default_factory=set)

    def subscribe(self) -> tuple[list[Event], asyncio.Queue]:
        """Buffered events so far, plus a queue that receives the rest.

        Atomic with respect to new events: nothing awaits between copying the
        buffer and registering the queue, so a subscriber neither misses an
        event nor sees one twice.
        """
        queue: asyncio.Queue = asyncio.Queue()
        buffered = list(self.sink.events)
        self._queues.add(queue)
        return buffered, queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._queues.discard(queue)

    def _fan_out(self, event: Event) -> None:
        for queue in list(self._queues):
            queue.put_nowait(event)

    @property
    def status(self) -> str:
        if self.error:
            return "failed"
        return "finished" if self.done else "running"


def create_app(runs_dir: Path | str = "runs") -> FastAPI:
    runs_dir = Path(runs_dir)
    app = FastAPI(title="Multi-Agent Web", docs_url=None, redoc_url=None)
    app.state.runs_dir = runs_dir
    app.state.live = {}  # type: dict[str, LiveRun]

    # --- the page --------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        # Read per request: no build step, and an edit shows on reload.
        return INDEX_HTML.read_text(encoding="utf-8")

    @app.get("/api/presets")
    async def presets() -> list[dict[str, Any]]:
        return presets_as_dicts()

    # --- replay ----------------------------------------------------------------

    @app.get("/api/runs")
    async def list_runs() -> list[dict[str, Any]]:
        return scan_runs(runs_dir, app.state.live)

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        run_dir = _safe_run_dir(runs_dir, run_id)
        live: LiveRun | None = app.state.live.get(run_id)
        if live is not None and not live.done:
            return {
                "id": run_id,
                "live": True,
                "spec": live.spec.model_dump(mode="json"),
                "summary": summarise_live(live),
                "events": [],
            }
        if not run_dir.exists():
            raise HTTPException(404, f"no such run: {run_id}")
        events = load_events(run_dir)
        if not events and live is None:
            raise HTTPException(404, f"{run_id} has no run.json or events.jsonl")
        return {
            "id": run_id,
            "live": False,
            "spec": live.spec.model_dump(mode="json") if live else None,
            "summary": summarise_dir(run_dir, live),
            "events": [e.as_dict() for e in events],
        }

    @app.get("/api/runs/{run_id}/files/{path:path}")
    async def run_file(run_id: str, path: str) -> FileResponse:
        run_dir = _safe_run_dir(runs_dir, run_id)
        target = (run_dir / path).resolve()
        # No escaping the run directory, whatever the path says.
        if run_dir.resolve() not in target.parents or not target.is_file():
            raise HTTPException(404, "no such file")
        return FileResponse(target)

    # --- live ------------------------------------------------------------------

    @app.post("/api/estimate")
    async def estimate(spec: LaunchSpec) -> dict[str, Any]:
        return estimate_calls(spec).as_dict()

    @app.post("/api/runs", status_code=202)
    async def start(spec: LaunchSpec) -> dict[str, Any]:
        spec = spec.model_copy(update={"runs_dir": runs_dir})
        est = estimate_calls(spec)
        if est.needs_api and not est.api_configured:
            raise HTTPException(
                400,
                "This run talks to the PP API but PPAPI_KEY / PPAPI_BASE_URL are "
                "not set. Use the mock policy with best_of_n, or configure .env.",
            )
        live = await start_run(spec, runs_dir, app.state.live)
        return {"run_id": live.id, "estimate": est.as_dict()}

    @app.websocket("/api/runs/{run_id}/live")
    async def live_stream(ws: WebSocket, run_id: str) -> None:
        await ws.accept()
        live: LiveRun | None = app.state.live.get(run_id)
        if live is None:
            await ws.send_json({"type": "error", "t": 0, "wall": "",
                                "data": {"error": f"{run_id} is not a live run"}})
            await ws.close()
            return
        buffered, queue = live.subscribe()
        try:
            for event in buffered:
                await ws.send_json(event.as_dict())
            if buffered and buffered[-1].type in TERMINAL_EVENTS:
                return
            while True:
                event = await queue.get()
                await ws.send_json(event.as_dict())
                if event.type in TERMINAL_EVENTS:
                    return
        except WebSocketDisconnect:
            pass
        finally:
            live.unsubscribe(queue)
            try:
                await ws.close()
            except Exception:  # pragma: no cover - already closed
                pass

    return app


# ---------------------------------------------------------------------------
# starting a run
# ---------------------------------------------------------------------------
async def start_run(
    spec: LaunchSpec, runs_dir: Path, registry: dict[str, LiveRun]
) -> LiveRun:
    """Create the run directory, register the run, launch it in the background.

    Split out from the route so a test can drive it without HTTP.
    """
    run_dir = make_run_dir(runs_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    sink = RecordingSink(path=run_dir / "events.jsonl")
    live = LiveRun(id=run_dir.name, spec=spec, run_dir=run_dir, sink=sink)
    sink.subscribe(live._fan_out)
    registry[live.id] = live
    live.task = asyncio.create_task(_execute(live), name=f"run-{live.id}")
    return live


async def _execute(live: LiveRun) -> None:
    """Run to completion. Never raises; a failure becomes a ``run_failed`` event."""
    prepared = None
    try:
        prepared = prepare(live.spec)
        await orchestrate(
            task=live.spec.task,
            strategy=prepared.strategy,
            policy_factory=prepared.policy_factory,
            run_config=prepared.run_config,
            orchestrator_config=prepared.orchestrator_config,
            run_dir=live.run_dir,
            usage_provider=prepared.usage_provider,
            sink=live.sink,
        )
    except LaunchError as exc:
        live.error = str(exc)
        live.sink.emit("run_failed", error=str(exc))
    except Exception as exc:  # noqa: BLE001 - the UI must hear about anything
        logger.exception("live run %s failed", live.id)
        live.error = f"{type(exc).__name__}: {exc}"
        live.sink.emit("run_failed", error=live.error)
    finally:
        live.done = True
        if prepared is not None:
            await prepared.aclose()
        live.sink.close()


# ---------------------------------------------------------------------------
# listing and summarising
# ---------------------------------------------------------------------------
def scan_runs(runs_dir: Path, registry: dict[str, LiveRun]) -> list[dict[str, Any]]:
    """Every replayable run, newest first, plus anything live in memory."""
    rows: dict[str, dict[str, Any]] = {}
    if runs_dir.exists():
        for child in runs_dir.iterdir():
            if not child.is_dir():
                continue
            if (child / "run.json").exists() or (child / "events.jsonl").exists():
                summary = summarise_dir(child, registry.get(child.name))
                if summary is not None:
                    rows[child.name] = summary
    for run_id, live in registry.items():
        if run_id not in rows or not live.done:
            rows[run_id] = summarise_live(live)
    return sorted(rows.values(), key=lambda r: r.get("id", ""), reverse=True)


def summarise_dir(run_dir: Path, live: LiveRun | None = None) -> dict[str, Any] | None:
    run_json = run_dir / "run.json"
    if not run_json.exists():
        # Started from the UI and died before reporting; still worth listing.
        if live is not None:
            return summarise_live(live)
        return _summary_from_events(run_dir)
    try:
        run = json.loads(run_json.read_text(encoding="utf-8"))
    except ValueError:
        return None
    details = run.get("details") or {}
    agents = run.get("agents") or []
    summary = {
        "id": run_dir.name,
        "task": run.get("task"),
        "strategy": run.get("strategy"),
        "started_at": run.get("started_at"),
        "finished_at": run.get("finished_at"),
        "answer": run.get("answer"),
        "reason": run.get("reason"),
        "status": "finished" if run.get("answer") is not None else "no answer",
        "num_agents": len(agents),
        "num_succeeded": sum(1 for a in agents if a.get("status") == "done"),
        "wall_seconds": (run.get("timing") or {}).get("wall_seconds"),
        "usage": run.get("usage"),
        "policy": _infer_policy(run, live),
        "live": False,
        "has_events": (run_dir / "events.jsonl").exists(),
    }
    if run.get("strategy") == "dag":
        growth = details.get("growth") or {}
        summary["dag"] = {
            "subtasks": growth.get("final_subtasks"),
            "waves": growth.get("waves"),
            "replans_applied": growth.get("replans_applied"),
            "manager_model": (details.get("manager") or {}).get("model"),
        }
    elif run.get("strategy") == "best_of_n":
        judge = details.get("judge") or {}
        summary["judge"] = {"name": judge.get("name"), "winner": judge.get("winner")}
    return summary


def _summary_from_events(run_dir: Path) -> dict[str, Any] | None:
    events = load_events(run_dir)
    if not events:
        return None
    started = next((e for e in events if e.type == "run_started"), None)
    failed = next((e for e in events if e.type == "run_failed"), None)
    return {
        "id": run_dir.name,
        "task": (started.data.get("task") if started else None),
        "strategy": (started.data.get("strategy") if started else None),
        "started_at": events[0].wall or None,
        "finished_at": None,
        "answer": None,
        "reason": failed.data.get("error") if failed else "incomplete",
        "status": "failed" if failed else "incomplete",
        "num_agents": sum(1 for e in events if e.type == "agent_started"),
        "num_succeeded": 0,
        "wall_seconds": events[-1].t,
        "usage": None,
        "policy": None,
        "live": False,
        "has_events": True,
    }


def summarise_live(live: LiveRun) -> dict[str, Any]:
    events = live.sink.events
    finished = next((e for e in reversed(events) if e.type == "run_finished"), None)
    return {
        "id": live.id,
        "task": live.spec.task,
        "strategy": live.spec.strategy,
        "started_at": events[0].wall if events else None,
        "finished_at": finished.wall if finished else None,
        "answer": finished.data.get("answer") if finished else None,
        "reason": finished.data.get("reason") if finished else live.error,
        "status": live.status,
        "num_agents": sum(1 for e in events if e.type == "agent_started"),
        "num_succeeded": sum(
            1 for e in events if e.type == "agent_finished" and e.data.get("status") == "done"
        ),
        "wall_seconds": live.sink.elapsed if not live.done else (events[-1].t if events else 0),
        "usage": finished.data.get("usage") if finished else None,
        "policy": live.spec.policy,
        "live": not live.done,
        "has_events": True,
    }


def _infer_policy(run: dict[str, Any], live: LiveRun | None) -> str | None:
    if live is not None:
        return live.spec.policy
    # run.json does not record the policy name; a metered ledger means qwen.
    return "qwen" if run.get("usage") else "mock"


def _safe_run_dir(runs_dir: Path, run_id: str) -> Path:
    if not run_id or "/" in run_id or "\\" in run_id or run_id in (".", ".."):
        raise HTTPException(404, "no such run")
    return runs_dir / run_id


__all__ = ["LiveRun", "create_app", "scan_runs", "start_run"]
