"""Demo presets: tasks chosen for what they show, on sites that tolerate agents.

Shared by the CLI demo and the web UI, so a preset picked in either place is
the same task on the same site with the same expectations. Each one records
*why* it is here -- the site's stability, whether it objects to automation, and
what behaviour the task is meant to draw out -- because a preset chosen without
that reasoning is a task that will fail on the day.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .actions import Click, Done, Scroll, Type

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_PAGE = _REPO_ROOT / "tests" / "fixtures" / "demo_page.html"

#: What ``MockPolicy`` replays. Coordinates match the bundled demo page and are
#: nonsense anywhere else -- which is exactly what a mock policy is: a control,
#: not a browser agent. Lives here rather than in a script so the CLI and the
#: UI drive the same control.
DEMO_SCRIPT = [
    ("The search box is near the top-left; click it to focus.", Click(x=150, y=120)),
    ("Type the query and submit it.", Type(text="multi-agent web", press_enter=True)),
    ("Scroll down to see the rest of the page.", Scroll.by("down", 600)),
    ("Nothing left in the script.", Done(answer="mock run finished")),
]


@dataclass(frozen=True)
class Preset:
    key: str
    task: str
    start_url: str
    why: str
    steps: int
    #: The strategy this task was designed for. best_of_n presets run fine
    #: under dag (the manager will usually emit one subtask); the dag preset
    #: is pointless under best_of_n.
    strategy: str = "best_of_n"
    #: Ceiling on the manager's initial decomposition, when strategy is dag.
    max_subtasks: int = 4

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


PRESETS: dict[str, Preset] = {
    "local": Preset(
        key="local",
        task="Type \"multi-agent web\" into the search box, submit it, then "
        "report exactly what the page says was submitted.",
        start_url=DEMO_PAGE.as_uri(),
        why="Bundled local page. No network, no API variance -- rehearse the "
        "recording and check tiling for free before spending anything.",
        steps=5,
    ),
    "wikipedia": Preset(
        key="wikipedia",
        task="Find out which year the Eiffel Tower was completed, and report "
        "just the year.",
        start_url="https://en.wikipedia.org/wiki/Main_Page",
        why="Stable, plain HTML, no cookie wall and no bot check in the US, and "
        "tolerant of four concurrent readers. Genuinely multi-step -- agents "
        "must search, open an article and read a fact -- so they diverge on "
        "route (search box vs. direct navigation, infobox vs. body text), "
        "which is the point of best-of-N.",
        steps=8,
    ),
    "bookstore": Preset(
        key="bookstore",
        task="Find the book titled \"A Light in the Attic\" and report its price.",
        start_url="https://books.toscrape.com/",
        why="A site published specifically as a scraping/automation sandbox, so "
        "using it is unambiguously fine and it will not throw a bot check. "
        "Catalogue -> product page -> read a field is the classic shopping "
        "pattern, and the grid of near-identical covers makes agents genuinely "
        "diverge: some click the cover, some the title link, some search first.",
        steps=8,
    ),
    "hackernews": Preset(
        key="hackernews",
        task="Report the title of the current number one story on the front page.",
        start_url="https://news.ycombinator.com/",
        why="About as static and lightweight as the web gets: no JavaScript "
        "needed, no banners, no login. Short and highly reliable, so it makes a "
        "good opener or a fallback if a longer task is going badly on the day.",
        steps=5,
    ),
    "pricecheck": Preset(
        key="pricecheck",
        task="On books.toscrape.com, find the listed price of each of these four "
        "books: \"A Light in the Attic\", \"Tipping the Velvet\", \"Soumission\" "
        "and \"The Silent Cartographer\". Then report which of them is the "
        "cheapest, and by how much it undercuts the most expensive of them.",
        start_url="https://books.toscrape.com/",
        why="Built for the DAG strategy, and built to make it replan. Four "
        "independent price lookups run as one parallel wave and a join depends "
        "on all four, so the parallel branches genuinely pay off (the first "
        "three titles are on the front page). \"The Silent Cartographer\" is "
        "not in the catalogue -- checked against all 1000 titles -- and the "
        "site has no search box, so that agent pages through the catalogue "
        "until it runs out of steps. Its failure blocks the join, and the only "
        "way to any answer is for the manager to change the graph mid-run: add "
        "a join over the three prices it has, or retry the lookup another way. "
        "Ground truth for the three: 51.77, 53.74, 50.10 -- Soumission "
        "cheapest, undercutting Tipping the Velvet by 3.64.",
        steps=8,
        strategy="dag",
        max_subtasks=6,
    ),
}


def presets_as_dicts() -> list[dict[str, Any]]:
    return [p.as_dict() for p in PRESETS.values()]


__all__ = ["DEMO_PAGE", "DEMO_SCRIPT", "PRESETS", "Preset", "presets_as_dicts"]
