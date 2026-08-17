#!/usr/bin/env python3
"""CLI for a multi-agent run.

    python scripts/run_multi.py --task "find the pricing page" --n 4
    python scripts/run_multi.py --task "..." --n 4 --headed
    python scripts/run_multi.py --task "..." --n 4 --policy molmoweb \
        --endpoint http://gpu-host:8001

    # Phase 3: a manager LLM decomposes the task into a dependency graph
    python scripts/run_multi.py --task "..." --strategy dag --policy qwen
    # the no-replanning ablation, as a config value rather than a code path
    python scripts/run_multi.py --task "..." --strategy dag --planning-budget 0

Defaults to MockPolicy + MockJudge + best-of-N, so it runs anywhere with no
GPU, no API key and no endpoint. ``--strategy dag`` and ``--judge llm`` are the
two things that always need a key, since both are LLMs by definition -- the
manager plans even when the agents are mocked, which is a cheap way to inspect
a decomposition without paying for the browsing.

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

from multi_agent_web.launch import (  # noqa: E402
    LaunchError,
    LaunchSpec,
    estimate_calls,
    prepare,
)
from multi_agent_web.orchestrator import orchestrate  # noqa: E402
from multi_agent_web.presets import DEMO_PAGE  # noqa: E402

DEFAULT_START_URL = DEMO_PAGE.as_uri() if DEMO_PAGE.exists() else "https://example.com"


def spec_from_args(args: argparse.Namespace) -> LaunchSpec:
    """argparse -> the one launch description every entry point shares.

    The wiring itself (policy factory, strategy, judge, the single API budget
    shared by agents, manager and judge) lives in ``multi_agent_web.launch`` so
    this CLI, the demo and the web UI cannot drift apart.
    """
    width, _, height = args.viewport.partition("x")
    return LaunchSpec(
        task=args.task,
        strategy=args.strategy,
        policy=args.policy,
        n=args.n,
        max_steps=args.max_steps,
        start_url=args.start_url,
        judge=args.judge,
        judge_model=args.judge_model,
        manager_model=args.manager_model,
        planning_budget=args.planning_budget,
        max_subtasks=args.max_subtasks,
        max_waves=args.max_waves,
        max_calls=args.max_calls,
        headless=not args.headed,
        viewport_width=int(width),
        viewport_height=int(height),
        max_browsers=args.max_browsers,
        max_model=args.max_model,
        runs_dir=args.runs_dir,
        endpoint=args.endpoint,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run several agents on one task.")
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--n", type=int, default=3,
        help="How many agents. best_of_n only -- under --strategy dag the "
        "manager decides how many subtasks there are, and --max-browsers caps "
        "how many run at once.",
    )
    parser.add_argument("--headed", action="store_true")
    parser.add_argument(
        "--policy", default="mock", choices=["mock", "molmoweb", "qwen"]
    )
    parser.add_argument(
        "--strategy", default="best_of_n", choices=["best_of_n", "dag"],
        help="best_of_n: N attempts at one task, judged. dag: a manager LLM "
        "decomposes the task into a dependency graph and runs it in waves.",
    )
    parser.add_argument(
        "--judge", default="mock", choices=["mock", "llm"],
        help="best_of_n only. mock is the deterministic rule-based judge.",
    )
    parser.add_argument(
        "--judge-model", default=None,
        help="Model for --judge llm. Default: PPAPI_JUDGE_MODEL, else the "
        "policy's model.",
    )
    parser.add_argument(
        "--manager-model", default=None,
        help="Model for the dag manager. Default: PPAPI_MANAGER_MODEL, else "
        "the policy's model. This is the knob MACU found mattered most.",
    )
    parser.add_argument(
        "--planning-budget", type=int, default=10,
        help="DAG edits the manager may make across the run (default 10, as in "
        "the paper). 0 decomposes but never replans -- the ablation.",
    )
    parser.add_argument(
        "--max-subtasks", type=int, default=6,
        help="Ceiling on the initial decomposition.",
    )
    parser.add_argument(
        "--max-waves", type=int, default=8,
        help="Backstop on how many waves a dag run may execute. The edit budget "
        "already bounds growth; this catches a plan that grows anyway.",
    )
    parser.add_argument("--endpoint", default=None, help="MolmoWeb server base URL.")
    parser.add_argument(
        "--max-calls",
        type=int,
        default=100,
        help="Per-run API call ceiling, shared across agents (metered policies).",
    )
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


def print_dag(details: dict) -> None:
    """The graph, the replans and the two headline numbers.

    DAG growth and replan rate are what the paper reports, so they are printed
    rather than left to be dug out of run.json.
    """
    growth = details["growth"]
    print("\ngraph:")
    for subtask in details["final_dag"]["subtasks"]:
        deps = ", ".join(subtask["depends_on"]) or "-"
        agent = "" if subtask["agent_index"] is None else f"agent {subtask['agent_index']}"
        print(
            f"  {subtask['id']:<20} {subtask['status']:<9} "
            f"wave {subtask['wave'] or '-':<3} {agent:<9} deps: {deps}"
        )
        if subtask["answer"]:
            print(f"    -> {' '.join(subtask['answer'].split())[:90]}")
        if subtask["error"]:
            print(f"    !! {subtask['error'][:90]}")

    replans = [r for r in details["replans"] if r.get("called_model")]
    if replans:
        print("\nreplans:")
        for entry in replans:
            mark = "applied" if entry.get("applied") else (entry.get("outcome") or "-")
            print(f"  [{mark}] {entry.get('reason', '')[:80]}")
            if entry.get("add"):
                print(f"    + {[s['id'] for s in entry['add']]}")
            if entry.get("remove"):
                print(f"    - {entry['remove']}")

    b = details["budget"]
    print("\nplanning:")
    print(f"  subtasks           {growth['initial_subtasks']} -> "
          f"{growth['final_subtasks']}  (net {growth['net_growth']:+d})")
    print(f"  waves              {growth['waves']}")
    print(f"  replans            {growth['replans_applied']} applied of "
          f"{growth['replans_proposed']} proposed  "
          f"(rate {growth['replan_rate']:.2f} per wave)")
    print(f"  edit budget        {b['spent']}/{b['limit']} spent"
          + (f", {b['refused_edits']} refused" if b["refused_edits"] else ""))
    if details.get("stopped_early"):
        print(f"  stopped early      {details['stopped_early']}")


async def main_async(args: argparse.Namespace) -> int:
    spec = spec_from_args(args)
    try:
        prepared = prepare(spec)
    except LaunchError as exc:
        raise SystemExit(str(exc)) from exc
    strategy = prepared.strategy

    print(f"task     : {args.task}")
    if args.strategy == "dag":
        print(f"strategy : dag (manager: {strategy.manager.model}, "
              f"planning budget: {strategy.manager.budget.limit} edits)")
    else:
        print(f"strategy : {strategy.name} (n={args.n}, judge: {args.judge})")
    print(f"policy   : {args.policy}")
    print(f"limits   : {args.max_browsers} browsers, {args.max_model} model slots")

    estimate = estimate_calls(spec)
    if estimate.needs_api and estimate.calls > args.max_calls:
        # Worst case is not --n agents under dag: the manager may plan up to
        # --max-subtasks and add up to --planning-budget more, each a whole
        # agent. Better to say so now than to abort at wave three.
        print(f"note     : worst case is ~{estimate.calls} API calls "
              f"({estimate.breakdown}) but --max-calls is {args.max_calls}. The "
              f"run aborts rather than overspending; raise it or lower the knobs.")
    print()

    try:
        result = await orchestrate(
            task=args.task,
            strategy=strategy,
            policy_factory=prepared.policy_factory,
            run_config=prepared.run_config,
            orchestrator_config=prepared.orchestrator_config,
            usage_provider=prepared.usage_provider,
        )
    finally:
        await prepared.aclose()

    print("\nper-agent:")
    for session in result.sessions:
        answer = (session.answer or "")[:60]
        label = f" [{session.spec.label}]" if session.spec.label else ""
        print(
            f"  agent {session.index}: {session.status:<10} "
            f"{len(session.steps):>2} steps  {session.timing.wall_seconds:>6.2f}s  "
            f"{answer}{label}"
        )
        if session.error:
            print(f"             error: {session.error[:100]}")

    if args.strategy == "dag":
        print_dag(result.details)

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

    if result.usage:
        u = result.usage
        print("\napi usage:")
        print(f"  calls              {u['calls']}/{u['limit']}"
              f"  ({u['retries']} retries)")
        if u.get("non_agent_calls"):
            print(f"  of which planning  {u['non_agent_calls']} "
                  f"(manager and judge; the rest is the agents)")
        print(f"  tokens             {u['total_tokens']} total "
              f"({u['prompt_tokens']} prompt, {u['completion_tokens']} completion, "
              f"{u['reasoning_tokens']} reasoning, {u['image_tokens']} image)")

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
