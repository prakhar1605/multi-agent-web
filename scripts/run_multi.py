#!/usr/bin/env python3
"""CLI for a multi-agent run.

    python scripts/run_multi.py --task "find the pricing page" --n 4
    python scripts/run_multi.py --task "..." --n 4 --headed
    python scripts/run_multi.py --task "..." --n 4 --policy molmoweb \
        --endpoint http://gpu-host:8001

Defaults to MockPolicy + MockJudge, so it runs anywhere with no GPU, no API key
and no endpoint.

Note ``--max-browsers`` and ``--max-model`` are separate on purpose: browsers
are local and memory-bound, the model server is shared and GPU-bound. Setting
``--max-model 1`` against several browsers is the interesting case -- agents
browse in parallel but queue single-file at the model, and the printed timing
breakdown shows exactly that.
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

from multi_agent_web.config import MolmoWebConfig, RunConfig  # noqa: E402
from multi_agent_web.orchestrator import (  # noqa: E402
    BestOfN,
    MockJudge,
    OrchestratorConfig,
    orchestrate,
)
from multi_agent_web.policy.base import AgentPolicy  # noqa: E402
from multi_agent_web.policy.mock import MockPolicy  # noqa: E402

from run_single import DEFAULT_START_URL, DEMO_SCRIPT  # noqa: E402


def build_policy_factory(name: str, endpoint: str | None = None):
    """Return a factory, never a policy.

    The orchestrator takes a factory precisely so two concurrent agents cannot
    end up sharing one stateful policy -- see orchestrator/session.py.
    """
    if name == "mock":
        # A fresh MockPolicy per agent: it holds a script cursor, which is
        # per-episode state just like MolmoWebPolicy's past_actions.
        return lambda index: MockPolicy(DEMO_SCRIPT)

    if name == "molmoweb":
        from multi_agent_web.policy.molmoweb import MolmoWebPolicy

        config = MolmoWebConfig.from_env(endpoint)
        if config is None:
            raise SystemExit(
                "No model endpoint configured. Pass --endpoint http://host:8001 "
                "or set MOLMOWEB_ENDPOINT."
            )
        # One MolmoWebPolicy per agent; they share the model server over HTTP,
        # and --max-model bounds how many generations are in flight at once.
        return lambda index: MolmoWebPolicy(config)

    raise ValueError(f"unknown policy: {name}")


def build_strategy(name: str, n: int, start_url: str | None):
    if name == "best_of_n":
        return BestOfN(n=n, judge=MockJudge(), start_url=start_url)
    raise ValueError(
        f"unknown strategy: {name} (DAG decomposition is Phase 3, not built)"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run several agents on one task.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--n", type=int, default=3, help="How many agents.")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--policy", default="mock", choices=["mock", "molmoweb"])
    parser.add_argument("--strategy", default="best_of_n", choices=["best_of_n"])
    parser.add_argument("--endpoint", default=None, help="MolmoWeb server base URL.")
    parser.add_argument("--start-url", default=DEFAULT_START_URL)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument(
        "--max-browsers", type=int, default=4, help="Local, memory-bound limit."
    )
    parser.add_argument(
        "--max-model", type=int, default=2, help="Shared model server, GPU-bound limit."
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--viewport", default="1280x720")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


async def main_async(args: argparse.Namespace) -> int:
    width, _, height = args.viewport.partition("x")
    run_config = RunConfig(
        viewport_width=int(width),
        viewport_height=int(height),
        headless=not args.headed,
        max_steps=args.max_steps,
        runs_dir=args.runs_dir,
    )
    orch_config = OrchestratorConfig(
        max_concurrent_browsers=args.max_browsers,
        max_inflight_model_requests=args.max_model,
    )

    policy_factory = build_policy_factory(args.policy, args.endpoint)
    strategy = build_strategy(args.strategy, args.n, args.start_url)

    print(f"task     : {args.task}")
    print(f"strategy : {strategy.name} (n={args.n})")
    print(f"policy   : {args.policy}")
    print(f"limits   : {args.max_browsers} browsers, {args.max_model} model slots\n")

    result = await orchestrate(
        task=args.task,
        strategy=strategy,
        policy_factory=policy_factory,
        run_config=run_config,
        orchestrator_config=orch_config,
    )

    print("\nper-agent:")
    for session in result.sessions:
        answer = (session.answer or "")[:60]
        print(
            f"  agent {session.index}: {session.status:<10} "
            f"{len(session.steps):>2} steps  {session.timing.wall_seconds:>6.2f}s  "
            f"{answer}"
        )
        if session.error:
            print(f"             error: {session.error[:100]}")

    t = result.timing
    print("\ntiming:")
    print(f"  wall               {t.wall_seconds:.2f}s")
    print(f"  sum of agent wall  {t.sum_agent_wall_seconds:.2f}s")
    print(f"  concurrency        {t.speedup_vs_serial:.2f}x")
    print(f"  model total        {t.total_model_seconds:.2f}s "
          f"(queued {t.total_model_queue_seconds:.2f}s)")
    print(f"  browser total      {t.total_browser_seconds:.2f}s")
    print(f"  peak browsers      {t.peak_concurrent_browsers}")
    print(f"  peak model inflight {t.peak_inflight_model_requests}")

    print(f"\nanswer : {result.answer}")
    print(f"reason : {result.reason}")
    print(f"run dir: {result.run_dir}")
    return 0 if result.answer is not None else 1


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
