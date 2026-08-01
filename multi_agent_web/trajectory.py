"""Step records and on-disk trajectory logging.

Layout produced by a run::

    runs/20260801-143012/
        meta.json        # task, config, start time
        step_000.png     # what the policy SAW before choosing step 0
        step_001.png
        ...
        final.png        # the page after the last action
        steps.jsonl      # one JSON object per step
        summary.json     # status, answer, step count (written at the end)

``Step`` lives here rather than in ``agent.py`` on purpose: both the policy and
the logger need it, and putting it in ``agent.py`` would make
``agent -> policy -> agent`` a circular import.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from pydantic import BaseModel, Field

from .actions import Action
from .config import RunConfig

logger = logging.getLogger(__name__)


class Step(BaseModel):
    """One decision by the policy, plus the context it was made in.

    A policy only fills ``thought`` and ``action`` -- everything else is
    observation metadata that the agent loop attaches afterwards. Keeping it in
    one model (instead of a separate ``StepRecord``) means the object a policy
    returns is literally the object that gets logged, so what you replay is
    what the policy produced.
    """

    thought: str = ""
    action: Action

    # Filled in by the agent loop.
    index: int | None = None
    url: str | None = None
    title: str | None = None
    screenshot: str | None = Field(
        default=None, description="Screenshot filename relative to the run directory."
    )
    error: str | None = Field(
        default=None, description="Set when executing the action failed."
    )
    timestamp: str | None = None

    def summary(self) -> str:
        head = f"[{self.index if self.index is not None else '?'}] {self.action.summary()}"
        if self.error:
            head += f"  !! {self.error}"
        return head


class Trajectory:
    """Writes a run to ``runs/<timestamp>/`` incrementally.

    Incrementally matters: if a run crashes at step 9 of 15, the first nine
    steps and their screenshots are already on disk. Buffering everything until
    the end would lose exactly the runs you most want to debug.
    """

    def __init__(
        self,
        task: str,
        config: RunConfig | None = None,
        run_dir: Path | None = None,
    ) -> None:
        self.task = task
        self.config = config or RunConfig()
        self.run_dir = run_dir or self._make_run_dir(self.config.runs_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.steps_path = self.run_dir / "steps.jsonl"
        self.steps: list[Step] = []

        self._write_json(
            self.run_dir / "meta.json",
            {
                "task": task,
                "started_at": _now(),
                "config": self.config.model_dump(mode="json"),
            },
        )

    @staticmethod
    def _make_run_dir(root: Path) -> Path:
        """``runs/<YYYYmmdd-HHMMSS>``, with a numeric suffix on collision."""
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        candidate = root / stamp
        n = 1
        while candidate.exists():
            candidate = root / f"{stamp}-{n}"
            n += 1
        return candidate

    def record(self, step: Step, screenshot: Image.Image | None = None) -> Step:
        """Append one step, saving the screenshot the policy acted on.

        Note the screenshot is the observation *before* the action, which is
        the pairing you need for training or for evaluating a prediction.
        """
        step.index = len(self.steps) if step.index is None else step.index
        step.timestamp = step.timestamp or _now()

        if screenshot is not None and self.config.save_screenshots:
            name = f"step_{step.index:03d}.png"
            screenshot.save(self.run_dir / name)
            step.screenshot = name

        self.steps.append(step)
        with self.steps_path.open("a", encoding="utf-8") as fh:
            fh.write(step.model_dump_json() + "\n")
        logger.info("%s", step.summary())
        return step

    def save_final_screenshot(self, screenshot: Image.Image) -> None:
        if self.config.save_screenshots:
            screenshot.save(self.run_dir / "final.png")

    def finalize(self, status: str, answer: str | None = None) -> None:
        self._write_json(
            self.run_dir / "summary.json",
            {
                "task": self.task,
                "status": status,
                "answer": answer,
                "num_steps": len(self.steps),
                "num_errors": sum(1 for s in self.steps if s.error),
                "finished_at": _now(),
            },
        )

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_trajectory(run_dir: Path) -> list[Step]:
    """Read a saved ``steps.jsonl`` back into ``Step`` objects."""
    path = Path(run_dir) / "steps.jsonl"
    if not path.exists():
        return []
    return [
        Step.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
