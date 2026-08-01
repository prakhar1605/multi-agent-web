"""Grounding gate, as a test.

Skips (never fails) when no model endpoint is configured, so the suite stays
green without a GPU. Point it at a live server to actually exercise it:

    MOLMOWEB_ENDPOINT=http://gpu-host:8001 pytest tests/test_grounding.py -s
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("playwright", reason="playwright is not installed")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from check_grounding import run_grounding_check  # noqa: E402

from multi_agent_web.config import MolmoWebConfig, RunConfig  # noqa: E402


@pytest.mark.asyncio
async def test_predicted_click_lands_on_the_target() -> None:
    model_config = MolmoWebConfig.from_env()
    if model_config is None:
        pytest.skip("MOLMOWEB_ENDPOINT is not set; skipping the live grounding check")

    results = await run_grounding_check(
        model_config, RunConfig(headless=True), trials=1, verbose=True
    )

    assert results, "grounding check produced no result"
    result = results[0]
    assert result.hit, (
        f"predicted click {result.predicted_px} (raw {result.predicted_pct}) "
        f"missed the target rect {result.rect}. {result.detail}"
    )
