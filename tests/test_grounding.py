"""Grounding gate, as a test.

Skips (never fails) unless explicitly opted in, so the suite stays green with no
credentials -- and, just as importantly, so a plain ``pytest`` never spends money.
This check makes one paid API call per target, and a test suite that quietly
bills a metered key on every run is a trap.

    GROUNDING_LIVE=1 pytest tests/test_grounding.py -s
    GROUNDING_LIVE=1 GROUNDING_POLICY=molmoweb pytest tests/test_grounding.py -s

Or just run the script, which is the intended entry point:

    python scripts/check_grounding.py --policy qwen
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("playwright", reason="playwright is not installed")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from check_grounding import build_policy, run_grounding_check  # noqa: E402

from multi_agent_web.config import RunConfig  # noqa: E402


@pytest.mark.asyncio
async def test_predicted_clicks_land_on_their_targets(tmp_path: Path) -> None:
    if os.environ.get("GROUNDING_LIVE", "").strip() not in {"1", "true", "yes"}:
        pytest.skip(
            "set GROUNDING_LIVE=1 to run the live grounding gate "
            "(it makes one paid API call per target)"
        )

    policy_name = os.environ.get("GROUNDING_POLICY", "qwen")
    policy = build_policy(policy_name, endpoint=None, calls_budget=30)
    if policy is None:
        pytest.skip(f"{policy_name} is not configured; skipping the live grounding check")

    try:
        report = await run_grounding_check(
            policy,
            RunConfig(headless=True, runs_dir=tmp_path / "runs"),
            out_dir=tmp_path / "grounding",
            verbose=True,
        )
    finally:
        await policy.close()

    assert report.results, "no targets were probed"
    misses = [r for r in report.results if not r.hit]
    assert not misses, "missed: " + "; ".join(
        f"{r.target.label} off by {r.miss_px:.0f}px ({r.detail or 'no detail'})"
        for r in misses
    )
