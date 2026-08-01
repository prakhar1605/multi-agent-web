#!/usr/bin/env python3
"""CLI entry point for a single agent run.

    python scripts/run_single.py --task "find the pricing page"
    python scripts/run_single.py --task "..." --headed
    python scripts/run_single.py --task "..." --start-url https://example.com

Defaults to ``MockPolicy`` so the loop runs with no GPU, no API key and no
network -- the default start URL is the bundled demo page on disk.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_agent_web.actions import Click, Done, Scroll, Type  # noqa: E402
from multi_agent_web.agent import run_task  # noqa: E402
from multi_agent_web.config import RunConfig  # noqa: E402
from multi_agent_web.policy.base import AgentPolicy  # noqa: E402
from multi_agent_web.policy.mock import MockPolicy  # noqa: E402

DEMO_PAGE = REPO_ROOT / "tests" / "fixtures" / "demo_page.html"
DEFAULT_START_URL = DEMO_PAGE.as_uri() if DEMO_PAGE.exists() else "https://example.com"

# Coordinates below match the bundled demo page. They are nonsense on any other
# site -- which is exactly what a mock policy is: a control, not a browser agent.
DEMO_SCRIPT = [
    ("The search box is near the top-left; click it to focus.", Click(x=150, y=120)),
    ("Type the query and submit it.", Type(text="multi-agent web", press_enter=True)),
    ("Scroll down to see the rest of the page.", Scroll(direction="down", amount=600)),
    ("Nothing left in the script.", Done(answer="mock run finished")),
]


def build_policy(name: str) -> AgentPolicy:
    """The one place a policy is chosen -- the seam Phase 1 exists to build."""
    if name == "mock":
        return MockPolicy(DEMO_SCRIPT)
    if name == "molmoweb":
        from multi_agent_web.policy.molmoweb import MolmoWebPolicy

        return MolmoWebPolicy()  # raises NotImplementedError, by design
    raise ValueError(f"unknown policy: {name}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one browsing agent episode.")
    parser.add_argument("--task", required=True, help="Natural-language goal.")
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Show the browser window (default: headless).",
    )
    parser.add_argument("--policy", default="mock", choices=["mock", "molmoweb"])
    parser.add_argument("--start-url", default=DEFAULT_START_URL)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--viewport", default="1280x720", help="e.g. 1280x720")
    parser.add_argument("--slow-mo", type=int, default=0, help="ms between actions")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


async def main_async(args: argparse.Namespace) -> int:
    width, _, height = args.viewport.partition("x")
    config = RunConfig(
        viewport_width=int(width),
        viewport_height=int(height),
        headless=not args.headed,
        max_steps=args.max_steps,
        runs_dir=args.runs_dir,
        slow_mo_ms=args.slow_mo,
    )

    try:
        policy = build_policy(args.policy)
    except NotImplementedError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 2

    print(f"task:      {args.task}")
    print(f"policy:    {policy.name}")
    print(f"start url: {args.start_url}")
    print(f"viewport:  {config.viewport_width}x{config.viewport_height}\n")

    result = await run_task(
        task=args.task,
        policy=policy,
        config=config,
        start_url=args.start_url,
    )

    print(f"\nstatus:  {result.status}")
    print(f"steps:   {len(result.steps)} ({result.num_errors} failed)")
    print(f"answer:  {result.answer}")
    print(f"run dir: {result.run_dir}")
    return 0 if result.status == "done" else 1


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
