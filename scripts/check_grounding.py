#!/usr/bin/env python3
"""Grounding acceptance gate: do predicted clicks land on the right elements?

The check to run before trusting ANY policy, and before building anything on
top of one. It isolates the single thing most likely to be silently wrong --
turning what the model sees into a coordinate the browser acts on. A convention
that is off by a factor, an axis, or a device-scale-factor still produces
perfectly well-formed JSON. It just misses.

    python scripts/check_grounding.py --policy qwen
    python scripts/check_grounding.py --policy molmoweb --endpoint http://gpu:8001
    python scripts/check_grounding.py --policy qwen --trials 3 --headed

Policy-agnostic: ``--policy`` selects the adapter, and the gate only ever talks
to ``AgentPolicy``. With nothing configured for the chosen policy it SKIPS
(exit 0) rather than failing, so it is safe in a no-credentials environment.

Six targets are probed, chosen to stress different failure modes: a small
control near the top edge, a large one mid-screen, one in the bottom band, a
34px icon, and one sitting between two near-identical twins where only the
label disambiguates. Note the gate is single-shot -- one screenshot, one click
-- so "below the fold" is exercised as the bottom band of the viewport rather
than off-screen content, which would require scrolling first.

Every target's true rectangle is read from the live page via Playwright, so the
fixture and this script cannot drift apart. Each probe writes an annotated PNG
with the target rect and the predicted point drawn on it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image, ImageDraw  # noqa: E402

from multi_agent_web.actions import Click  # noqa: E402
from multi_agent_web.browser import BrowserSession  # noqa: E402
from multi_agent_web.config import (  # noqa: E402
    MolmoWebConfig,
    QwenConfig,
    RunConfig,
    load_env_file,
)
from multi_agent_web.policy.base import AgentPolicy  # noqa: E402
from multi_agent_web.trajectory import make_run_dir  # noqa: E402

GROUNDING_PAGE = REPO_ROOT / "tests" / "fixtures" / "grounding_page.html"


@dataclass(frozen=True)
class Target:
    selector: str
    label: str
    why: str  # what failure mode this target probes


TARGETS: tuple[Target, ...] = (
    Target("#t-nav", "Documentation", "top edge, left"),
    Target("#t-signin", "Sign In", "top edge, right, narrow"),
    Target("#t-primary", "Generate Report", "large, mid-screen (control)"),
    Target("#t-twin-a", "Export CSV", "between two near-identical twins"),
    Target("#t-small", "?", "34x34 icon, right edge"),
    Target("#t-bottom", "Accept and Continue", "bottom band of the viewport"),
)


@dataclass
class ProbeResult:
    target: Target
    rect: dict[str, float]
    hit: bool = False
    predicted: tuple[float, float] | None = None
    miss_px: float = 0.0
    detail: str = ""
    image: Path | None = None
    thought: str = ""

    def as_dict(self) -> dict:
        return {
            "selector": self.target.selector,
            "label": self.target.label,
            "probes": self.target.why,
            "hit": self.hit,
            "predicted": list(self.predicted) if self.predicted else None,
            "rect": self.rect,
            "miss_px": round(self.miss_px, 1),
            "detail": self.detail,
            "image": self.image.name if self.image else None,
        }


@dataclass
class GroundingReport:
    policy: str
    results: list[ProbeResult] = field(default_factory=list)
    out_dir: Path | None = None

    @property
    def hits(self) -> int:
        return sum(1 for r in self.results if r.hit)

    @property
    def hit_rate(self) -> float:
        return self.hits / len(self.results) if self.results else 0.0


def _distance_outside(x: float, y: float, rect: dict[str, float]) -> float:
    """0 if inside, else the shortest distance to the rectangle's edge."""
    dx = max(rect["x"] - x, 0.0, x - (rect["x"] + rect["width"]))
    dy = max(rect["y"] - y, 0.0, y - (rect["y"] + rect["height"]))
    return math.hypot(dx, dy)


def annotate(
    screenshot: Image.Image,
    result: ProbeResult,
    path: Path,
) -> None:
    """Draw the target rect and the predicted point so misses are visible."""
    img = screenshot.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    r = result.rect
    x0, y0 = r["x"], r["y"]
    x1, y1 = x0 + r["width"], y0 + r["height"]

    ok = (0, 168, 84)
    bad = (222, 48, 48)
    colour = ok if result.hit else bad

    # Target rectangle, drawn thick enough to survive downscaling.
    for w in range(3):
        draw.rectangle([x0 - w, y0 - w, x1 + w, y1 + w], outline=ok)
    caption = f"target: {result.target.label}"
    cap_x = min(x0, img.width - 8 * len(caption) - 4)
    draw.text((max(2.0, cap_x), max(2.0, y0 - 14)), caption, fill=ok)

    if result.predicted:
        px, py = result.predicted
        arm = 14
        for w in range(2):
            draw.line([px - arm, py + w, px + arm, py + w], fill=colour)
            draw.line([px + w, py - arm, px + w, py + arm], fill=colour)
        draw.ellipse([px - 5, py - 5, px + 5, py + 5], outline=colour, width=2)
        # Place the label away from the right edge so it never clips.
        tag = "HIT" if result.hit else f"MISS {result.miss_px:.0f}px"
        tag_w = 8 * len(tag)
        tag_x = px + 18 if px + 18 + tag_w < img.width else px - 18 - tag_w
        draw.text((max(2.0, tag_x), py - 6), tag, fill=colour)
        if not result.hit:
            # Line from the prediction to the centre it should have hit.
            draw.line(
                [px, py, (x0 + x1) / 2, (y0 + y1) / 2], fill=bad, width=1
            )
    else:
        draw.text((12, 12), f"NO CLICK: {result.detail[:80]}", fill=bad)

    img.save(path)


def build_policy(name: str, endpoint: str | None, calls_budget: int) -> AgentPolicy | None:
    """Return a policy, or None when that policy has no configuration.

    The gate only ever uses ``AgentPolicy``, so adding an adapter here is the
    entire cost of making it gate-able.
    """
    load_env_file()

    if name == "qwen":
        config = QwenConfig.from_env(max_calls_per_run=calls_budget)
        if config is None:
            return None
        from multi_agent_web.policy.qwen import QwenPolicy

        return QwenPolicy(config)

    if name == "molmoweb":
        config = MolmoWebConfig.from_env(endpoint)
        if config is None:
            return None
        from multi_agent_web.policy.molmoweb import MolmoWebPolicy

        return MolmoWebPolicy(config)

    raise ValueError(f"unknown policy: {name}")


def missing_config_message(name: str) -> str:
    if name == "qwen":
        return (
            "SKIP: PPAPI_KEY and PPAPI_BASE_URL are not both set.\n"
            "      Put them in .env at the repo root, or export them."
        )
    return (
        "SKIP: no MolmoWeb endpoint configured.\n"
        "      Pass --endpoint http://host:8001 or set MOLMOWEB_ENDPOINT."
    )


async def run_grounding_check(
    policy: AgentPolicy,
    run_config: RunConfig | None = None,
    targets: tuple[Target, ...] = TARGETS,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> GroundingReport:
    """Probe every target once; return per-target hit/miss with annotations."""
    run_config = run_config or RunConfig()
    out_dir = out_dir or make_run_dir(run_config.runs_dir / "grounding")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = GroundingReport(policy=getattr(policy, "name", "policy"), out_dir=out_dir)

    async with BrowserSession(run_config) as browser:
        await browser.goto(GROUNDING_PAGE.as_uri())
        screenshot = await browser.screenshot()
        page_info = await browser.page_info()

        if verbose:
            print(f"policy    : {report.policy}")
            print(f"viewport  : {run_config.viewport_width}x{run_config.viewport_height}")
            print(f"screenshot: {screenshot.width}x{screenshot.height}")
            print(f"output    : {out_dir}\n")

        for target in targets:
            rect = await browser.page.locator(target.selector).bounding_box()
            if rect is None:
                raise RuntimeError(f"{target.selector} has no bounding box")

            result = ProbeResult(target=target, rect=rect)
            task = f'Click the button labelled "{target.label}".'

            try:
                await policy.reset()
                step = await policy.predict(task, [], screenshot, page_info)
            except Exception as exc:
                result.detail = f"{type(exc).__name__}: {exc}"
            else:
                result.thought = step.thought
                if isinstance(step.action, Click):
                    point = (float(step.action.x), float(step.action.y))
                    result.predicted = point
                    result.miss_px = _distance_outside(point[0], point[1], rect)
                    result.hit = result.miss_px == 0.0
                else:
                    result.detail = (
                        f"expected a click, model chose {step.action.summary()}"
                    )

            path = out_dir / f"{target.selector.lstrip('#')}.png"
            annotate(screenshot, result, path)
            result.image = path
            report.results.append(result)

            if verbose:
                mark = "HIT " if result.hit else "MISS"
                where = (
                    f"({result.predicted[0]:.0f}, {result.predicted[1]:.0f})"
                    if result.predicted
                    else "(no click)"
                )
                extra = "" if result.hit else f"  off by {result.miss_px:.0f}px"
                print(f"  {mark}  {target.label:<22} {where}{extra}")
                if result.detail:
                    print(f"        {result.detail[:160]}")

    (out_dir / "grounding.json").write_text(
        json.dumps(
            {
                "policy": report.policy,
                "hit_rate": round(report.hit_rate, 3),
                "hits": report.hits,
                "targets": len(report.results),
                "results": [r.as_dict() for r in report.results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", default="qwen", choices=["qwen", "molmoweb"])
    parser.add_argument(
        "--endpoint", default=None, help="MolmoWeb server base URL (molmoweb only)."
    )
    parser.add_argument(
        "--trials", type=int, default=1, help="Repeat the whole sweep N times."
    )
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--viewport", default="1280x720")
    parser.add_argument(
        "--max-calls", type=int, default=60, help="API call ceiling for this check."
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


async def main_async(args: argparse.Namespace) -> int:
    policy = build_policy(args.policy, args.endpoint, args.max_calls)
    if policy is None:
        print(missing_config_message(args.policy))
        print("      (Skipping, not failing, so this is safe without credentials.)")
        return 0

    width, _, height = args.viewport.partition("x")
    run_config = RunConfig(
        viewport_width=int(width),
        viewport_height=int(height),
        headless=not args.headed,
    )

    try:
        totals: list[GroundingReport] = []
        for trial in range(1, args.trials + 1):
            if args.trials > 1:
                print(f"--- trial {trial}/{args.trials} ---")
            totals.append(await run_grounding_check(policy, run_config))
            print()
    finally:
        await policy.close()

    hits = sum(r.hits for r in totals)
    total = sum(len(r.results) for r in totals)
    print("=" * 66)
    print(f"HIT RATE: {hits}/{total} = {hits / total:.0%}" if total else "no probes")
    print("=" * 66)

    for report in totals:
        for result in report.results:
            if not result.hit:
                print(
                    f"  miss: {result.target.label:<22} "
                    f"({result.target.why})  {result.detail or f'{result.miss_px:.0f}px off'}"
                )
    print(f"\nannotated images: {totals[-1].out_dir}")

    if hits < total:
        print(
            "\nBefore blaming the model, check the conversion: is the screenshot the\n"
            "same size as the viewport (device_scale_factor=1)? Are coordinates in\n"
            "the space the adapter documents? Are x and y swapped? A systematic\n"
            "offset in one direction across all targets means a conversion bug; a\n"
            "scatter of near-misses means the model is genuinely imprecise."
        )
    return 0 if hits == total else 1


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
