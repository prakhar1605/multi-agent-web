"""The FastAPI backend, driven through Starlette's test client. Mock policy only.

Replay is tested against a run written to disk by ``orchestrate`` with no sink
-- the same artifacts a CLI run leaves -- because that is what the supervisor
will point the viewer at. Live is tested end to end: POST a spec, read the
WebSocket until ``run_finished``, then confirm the run is on disk and replayable
like any other.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("playwright", reason="playwright is not installed")
pytest.importorskip("fastapi", reason="fastapi is not installed")

from fastapi.testclient import TestClient  # noqa: E402

from multi_agent_web.actions import Done, Wait  # noqa: E402
from multi_agent_web.config import RunConfig  # noqa: E402
from multi_agent_web.orchestrator import BestOfN, MockJudge, orchestrate  # noqa: E402
from multi_agent_web.policy.mock import MockPolicy  # noqa: E402
from multi_agent_web.ui.server import create_app  # noqa: E402


def mock_factory(index: int) -> MockPolicy:
    return MockPolicy([Wait(seconds=0.01), Done(answer=f"answer {index}")])


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "runs"
    d.mkdir()
    return d


@pytest.fixture
def finished_run(runs_dir: Path):
    """A best-of-N run on disk, produced with NO sink -- a CLI-style artifact.

    Run to completion in its own loop before the test client starts, so the
    artifact is exactly what a CLI invocation leaves behind.
    """
    import asyncio

    return asyncio.run(
        orchestrate(
            task="find it",
            strategy=BestOfN(n=2, judge=MockJudge()),
            policy_factory=mock_factory,
            run_config=RunConfig(headless=True, max_steps=6, runs_dir=runs_dir),
        )
    )


def test_index_is_served(runs_dir: Path) -> None:
    with TestClient(create_app(runs_dir)) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert "<title>Multi-Agent Web</title>" in r.text
        assert "id=\"dag\"" in r.text  # the DAG panel is in the page


def test_presets_include_the_dag_one(runs_dir: Path) -> None:
    with TestClient(create_app(runs_dir)) as client:
        presets = client.get("/api/presets").json()
        keys = {p["key"] for p in presets}
        assert {"local", "bookstore", "pricecheck"} <= keys
        price = next(p for p in presets if p["key"] == "pricecheck")
        assert price["strategy"] == "dag"
        assert "Silent Cartographer" in price["task"]


def test_replay_lists_loads_and_serves_files(runs_dir: Path, finished_run) -> None:
    run_id = finished_run.run_dir.name
    with TestClient(create_app(runs_dir)) as client:
        rows = client.get("/api/runs").json()
        assert [r["id"] for r in rows] == [run_id]
        row = rows[0]
        assert row["strategy"] == "best_of_n" and row["num_agents"] == 2
        assert row["answer"] == finished_run.answer
        assert row["policy"] == "mock" and row["live"] is False

        run = client.get(f"/api/runs/{run_id}").json()
        assert run["live"] is False
        types = [e["type"] for e in run["events"]]
        assert types[0] == "run_started" and types[-1] == "run_finished"
        assert types.count("agent_step") == 4
        assert "judge_decision" in types

        # A screenshot named by an event is fetchable...
        step = next(e for e in run["events"] if e["type"] == "agent_step")
        shot = step["data"]["step"]["screenshot"]
        img = client.get(f"/api/runs/{run_id}/files/{shot}")
        assert img.status_code == 200 and img.headers["content-type"] == "image/png"
        # ...and nothing outside the run directory is.
        assert client.get(f"/api/runs/{run_id}/files/../run.json").status_code == 404
        assert client.get(f"/api/runs/{run_id}/files/../../etc/passwd").status_code == 404
        assert client.get("/api/runs/..%2F..%2Fetc/files/passwd").status_code == 404
        assert client.get("/api/runs/nope").status_code == 404


def test_estimate_prices_free_and_paid_runs(runs_dir: Path) -> None:
    with TestClient(create_app(runs_dir)) as client:
        free = client.post("/api/estimate", json={"task": "t", "policy": "mock"}).json()
        assert free["free"] is True and free["calls"] == 0
        paid = client.post(
            "/api/estimate",
            json={"task": "t", "policy": "qwen", "n": 3, "max_steps": 5},
        ).json()
        assert paid["free"] is False and paid["calls"] == 15
        dag = client.post(
            "/api/estimate",
            json={"task": "t", "policy": "mock", "strategy": "dag", "max_waves": 4},
        ).json()
        assert dag["needs_api"] is True and dag["calls"] == 5  # decompose + 4 replans
        assert client.post("/api/estimate", json={"task": ""}).status_code == 422


def test_live_mock_run_streams_and_lands_on_disk(runs_dir: Path) -> None:
    with TestClient(create_app(runs_dir)) as client:
        r = client.post(
            "/api/runs",
            json={"task": "live one", "policy": "mock", "n": 2, "max_steps": 4},
        )
        assert r.status_code == 202, r.text
        run_id = r.json()["run_id"]
        assert r.json()["estimate"]["free"] is True

        # While it runs, the listing and the detail both say so.
        listed = client.get("/api/runs").json()
        assert any(x["id"] == run_id and x["status"] in ("running", "finished") for x in listed)

        seen = []
        with client.websocket_connect(f"/api/runs/{run_id}/live") as ws:
            while True:
                ev = ws.receive_json()
                seen.append(ev["type"])
                if ev["type"] in ("run_finished", "run_failed"):
                    break
        assert seen[0] == "run_started" and seen[-1] == "run_finished", seen
        assert seen.count("agent_started") == 2 and seen.count("agent_step") >= 2

        # Now it is a run like any other: run.json, events.jsonl, replayable.
        run_dir = runs_dir / run_id
        assert (run_dir / "run.json").exists()
        assert (run_dir / "events.jsonl").exists()
        recorded = [json.loads(l)["type"] for l in (run_dir / "events.jsonl").read_text().splitlines()]
        assert recorded == seen

        again = client.get(f"/api/runs/{run_id}").json()
        assert again["live"] is False
        assert [e["type"] for e in again["events"]] == seen  # the recording, verbatim
        assert again["summary"]["status"] == "finished"

        # A late subscriber to a finished live run gets the buffered story too.
        with client.websocket_connect(f"/api/runs/{run_id}/live") as ws:
            first = ws.receive_json()
            assert first["type"] == "run_started"


def test_live_run_needing_a_key_is_refused_up_front(runs_dir: Path, monkeypatch) -> None:
    monkeypatch.delenv("PPAPI_KEY", raising=False)
    monkeypatch.delenv("PPAPI_BASE_URL", raising=False)
    # Stop load_env_file from finding the repo's .env.
    monkeypatch.setattr("multi_agent_web.config.load_env_file", lambda *a, **k: None)
    with TestClient(create_app(runs_dir)) as client:
        r = client.post("/api/runs", json={"task": "t", "policy": "qwen"})
        assert r.status_code == 400
        assert "PPAPI_KEY" in r.json()["detail"]
        assert list(runs_dir.iterdir()) == [], "a refused run left a directory behind"


def test_unknown_live_socket_says_so(runs_dir: Path) -> None:
    with TestClient(create_app(runs_dir)) as client:
        with client.websocket_connect("/api/runs/nope/live") as ws:
            ev = ws.receive_json()
            assert ev["type"] == "error"
