"""End-to-end test: the loop runs against a local page and logs a trajectory.

Uses a ``file://`` URL, so it needs no network -- only Chromium, installed via
``python -m playwright install chromium``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("playwright", reason="playwright is not installed")

from multi_agent_web.actions import Click, Done, Scroll, Type  # noqa: E402
from multi_agent_web.agent import run_task  # noqa: E402
from multi_agent_web.config import RunConfig  # noqa: E402
from multi_agent_web.policy.mock import MockPolicy  # noqa: E402
from multi_agent_web.trajectory import load_trajectory  # noqa: E402

DEMO_PAGE = Path(__file__).parent / "fixtures" / "demo_page.html"
ANSWER = "typed the query and scrolled"

# Coordinates target the absolutely-positioned input in demo_page.html.
SCRIPT = [
    ("Focus the input box.", Click(x=150, y=120)),
    ("Type a query and submit.", Type(text="multi-agent web", press_enter=True)),
    ("Scroll down.", Scroll(direction="down", amount=600)),
    ("Task complete.", Done(answer=ANSWER)),
]


@pytest.mark.asyncio
async def test_loop_records_trajectory(tmp_path: Path) -> None:
    config = RunConfig(headless=True, max_steps=10, runs_dir=tmp_path / "runs")
    policy = MockPolicy(SCRIPT)

    result = await run_task(
        task="type a query on the demo page",
        policy=policy,
        config=config,
        start_url=DEMO_PAGE.as_uri(),
    )

    # --- the loop terminated the way the script said it should -------------
    assert result.status == "done"
    assert result.answer == ANSWER
    assert len(result.steps) == len(SCRIPT)
    assert result.num_errors == 0, [s.error for s in result.steps if s.error]

    # --- the trajectory landed on disk -------------------------------------
    run_dir = Path(result.run_dir)
    steps_file = run_dir / "steps.jsonl"
    assert steps_file.exists()

    lines = [ln for ln in steps_file.read_text().splitlines() if ln.strip()]
    assert len(lines) == len(SCRIPT)

    first = json.loads(lines[0])
    assert first["action"] == {"type": "click", "x": 150, "y": 120}
    assert first["thought"] == "Focus the input box."
    assert first["url"].startswith("file://")
    assert first["screenshot"] == "step_000.png"

    # One PNG per step, plus the post-run screenshot.
    for i in range(len(SCRIPT)):
        png = run_dir / f"step_{i:03d}.png"
        assert png.exists() and png.stat().st_size > 0
    assert (run_dir / "final.png").exists()

    # --- screenshots are in the same pixel space as the coordinates --------
    from PIL import Image

    with Image.open(run_dir / "step_000.png") as img:
        assert img.size == (config.viewport_width, config.viewport_height)

    # --- the actions actually reached the page -----------------------------
    # The page rewrites document.title on submit. The scroll step is observed
    # after the type step ran, so its recorded title proves the typing landed.
    scroll_step = json.loads(lines[2])
    assert scroll_step["title"] == "Demo Page - typed: multi-agent web"

    # --- the log round-trips back into Step objects ------------------------
    reloaded = load_trajectory(run_dir)
    assert [s.action.type for s in reloaded] == ["click", "type", "scroll", "done"]

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "done"
    assert summary["num_steps"] == len(SCRIPT)


@pytest.mark.asyncio
async def test_loop_stops_at_max_steps(tmp_path: Path) -> None:
    """A policy that never emits Done must be cut off by the step budget."""
    config = RunConfig(headless=True, max_steps=3, runs_dir=tmp_path / "runs")
    policy = MockPolicy([Scroll(direction="down")] * 10)

    result = await run_task(
        task="scroll forever",
        policy=policy,
        config=config,
        start_url=DEMO_PAGE.as_uri(),
    )

    assert result.status == "max_steps"
    assert len(result.steps) == 3


@pytest.mark.asyncio
async def test_failed_action_does_not_kill_the_run(tmp_path: Path) -> None:
    """An out-of-viewport click is logged as an error, and the run continues."""
    config = RunConfig(headless=True, max_steps=5, runs_dir=tmp_path / "runs")
    policy = MockPolicy(
        [
            Click(x=99999, y=99999),  # outside the viewport -> ActionError
            Done(answer="survived"),
        ]
    )

    result = await run_task(
        task="click out of bounds",
        policy=policy,
        config=config,
        start_url=DEMO_PAGE.as_uri(),
    )

    assert result.status == "done"
    assert result.answer == "survived"
    assert result.num_errors == 1
    assert "outside" in result.steps[0].error
