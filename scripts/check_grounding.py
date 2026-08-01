#!/usr/bin/env python3
"""Grounding acceptance gate: does a predicted click land on the right element?

This is the check to run before trusting MolmoWeb for anything, and before any
multi-agent work. It isolates the one thing most likely to be silently wrong --
the percent -> pixel mapping. A coordinate convention that is off by a factor,
an axis, or a device-scale-factor still produces plausible-looking JSON; it just
misses. This makes that visible in one number.

    python scripts/check_grounding.py --endpoint http://gpu-host:8001
    python scripts/check_grounding.py --trials 5 --headed

Exits 0 on a hit (or when skipped), 1 on a miss or an error. With no endpoint
configured it SKIPS rather than fails, so it is safe in a no-GPU environment.

The target's rectangle is read from the live page via Playwright, so the HTML
and this script cannot drift apart.
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

from multi_agent_web.actions import Click  # noqa: E402
from multi_agent_web.browser import BrowserSession  # noqa: E402
from multi_agent_web.config import MolmoWebConfig, RunConfig  # noqa: E402
from multi_agent_web.policy.molmoweb import MolmoWebPolicy  # noqa: E402

GROUNDING_PAGE = REPO_ROOT / "tests" / "fixtures" / "grounding_page.html"
TARGET_SELECTOR = "#target"
TARGET_LABEL = "Download Report"
TASK = f'Click the button labelled "{TARGET_LABEL}".'


class GroundingResult:
    def __init__(
        self,
        hit: bool,
        predicted_px: tuple[float, float] | None,
        predicted_pct: tuple[float, float] | None,
        rect: dict[str, float],
        detail: str = "",
    ) -> None:
        self.hit = hit
        self.predicted_px = predicted_px
        self.predicted_pct = predicted_pct
        self.rect = rect
        self.detail = detail


def _inside(x: float, y: float, rect: dict[str, float]) -> bool:
    return (
        rect["x"] <= x <= rect["x"] + rect["width"]
        and rect["y"] <= y <= rect["y"] + rect["height"]
    )


async def run_grounding_check(
    model_config: MolmoWebConfig,
    run_config: RunConfig | None = None,
    trials: int = 1,
    verbose: bool = True,
) -> list[GroundingResult]:
    """Ask the model to click a known element; report where the click landed."""
    run_config = run_config or RunConfig()
    policy = MolmoWebPolicy(model_config)
    results: list[GroundingResult] = []

    try:
        async with BrowserSession(run_config) as browser:
            await browser.goto(GROUNDING_PAGE.as_uri())

            rect = await browser.page.locator(TARGET_SELECTOR).bounding_box()
            if rect is None:
                raise RuntimeError(f"target {TARGET_SELECTOR} has no bounding box")

            screenshot = await browser.screenshot()
            page_info = await browser.page_info()

            if verbose:
                print(f"endpoint : {model_config.endpoint}")
                print(f"task     : {TASK}")
                print(f"viewport : {run_config.viewport_width}x{run_config.viewport_height}")
                print(f"screenshot: {screenshot.width}x{screenshot.height}")
                print(
                    "target   : x=[{:.1f}, {:.1f}] y=[{:.1f}, {:.1f}]".format(
                        rect["x"], rect["x"] + rect["width"],
                        rect["y"], rect["y"] + rect["height"],
                    )
                )
                print()

            for trial in range(1, trials + 1):
                await policy.reset()
                try:
                    step = await policy.predict(TASK, [], screenshot, page_info)
                except Exception as exc:
                    results.append(
                        GroundingResult(False, None, None, rect, detail=str(exc))
                    )
                    if verbose:
                        print(f"trial {trial}: ERROR  {exc}")
                    continue

                raw = policy.past_actions[-1]["action"] if policy.past_actions else {}
                pct = (
                    (float(raw["x"]), float(raw["y"]))
                    if {"x", "y"} <= set(raw)
                    else None
                )

                if not isinstance(step.action, Click):
                    detail = f"expected a click, model chose {step.action.summary()}"
                    results.append(GroundingResult(False, None, pct, rect, detail))
                    if verbose:
                        print(f"trial {trial}: MISS   {detail}")
                    continue

                px = (step.action.x, step.action.y)
                hit = _inside(px[0], px[1], rect)
                results.append(GroundingResult(hit, px, pct, rect))

                if verbose:
                    pct_s = f"({pct[0]:g}%, {pct[1]:g}%)" if pct else "(n/a)"
                    print(
                        f"trial {trial}: {'HIT ' if hit else 'MISS'}  "
                        f"predicted {pct_s} -> ({px[0]:.1f}, {px[1]:.1f}) px"
                    )
                    if step.thought:
                        print(f"          thought: {step.thought[:140]}")
    finally:
        await policy.close()

    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--endpoint",
        default=None,
        help="MolmoWeb server base URL. Defaults to $MOLMOWEB_ENDPOINT.",
    )
    parser.add_argument("--trials", type=int, default=1, help="Sampling is stochastic.")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--viewport", default="1280x720")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


async def main_async(args: argparse.Namespace) -> int:
    model_config = MolmoWebConfig.from_env(args.endpoint)
    if model_config is None:
        print(
            "SKIP: no model endpoint configured.\n"
            "      Pass --endpoint http://host:8001 or set MOLMOWEB_ENDPOINT.\n"
            "      (Skipping, not failing, so this is safe without a GPU.)"
        )
        return 0

    width, _, height = args.viewport.partition("x")
    run_config = RunConfig(
        viewport_width=int(width),
        viewport_height=int(height),
        headless=not args.headed,
    )

    results = await run_grounding_check(model_config, run_config, trials=args.trials)
    hits = sum(1 for r in results if r.hit)

    print()
    print(f"RESULT: {hits}/{len(results)} click(s) landed inside the target rect.")
    if hits != len(results):
        print(
            "\nA miss usually means the coordinate convention is wrong, not that\n"
            "the model is bad. Check, in order: is the screenshot the same size as\n"
            "the viewport (device_scale_factor=1)? Are percentages being divided by\n"
            "100 and multiplied by the SCREENSHOT dimensions? Are x and y swapped?"
        )
    return 0 if hits == len(results) and results else 1


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
