"""Run configuration for a single browsing agent.

Everything that a run needs to know about *how* to browse (as opposed to *what*
to browse) lives here, so a future manager can hand each parallel agent its own
``RunConfig`` without any global state.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr

# Repo root, so a .env next to it is found regardless of the working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]


def load_env_file(path: Path | None = None) -> None:
    """Load ``KEY=value`` pairs from a ``.env`` file into ``os.environ``.

    Deliberately minimal and dependency-free. Real environment variables always
    win, so exporting a value overrides the file rather than the reverse -- that
    ordering matters for CI, where secrets arrive as env vars and no file exists.

    Never logs values: this file holds credentials.
    """
    path = path or _REPO_ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


class RunConfig(BaseModel):
    """Knobs for one agent run.

    The viewport size is the single most important setting: it defines the
    coordinate space that the policy predicts in. Change it and every learned
    or hardcoded coordinate changes meaning, so keep it fixed across a
    train/eval setup.
    """

    # --- Observation space -------------------------------------------------
    viewport_width: int = Field(default=1280, gt=0)
    viewport_height: int = Field(default=720, gt=0)

    # Force 1 CSS pixel == 1 screenshot pixel. On a Retina display the default
    # would be 2, and every predicted coordinate would be off by 2x.
    device_scale_factor: float = Field(default=1.0, gt=0)

    # --- Episode limits ----------------------------------------------------
    max_steps: int = Field(default=15, gt=0)
    # Abort the run after this many consecutive failures (policy or action) so
    # a broken page cannot burn the whole step budget in a tight loop.
    max_consecutive_errors: int = Field(default=3, gt=0)

    # --- Browser -----------------------------------------------------------
    headless: bool = True
    # Pause between Playwright operations; useful when watching a headed run.
    slow_mo_ms: int = Field(default=0, ge=0)
    action_timeout_ms: int = Field(default=10_000, gt=0)
    navigation_timeout_ms: int = Field(default=30_000, gt=0)
    # Grace period after each action so the page can react before we screenshot.
    settle_ms: int = Field(default=400, ge=0)

    # --- Logging -----------------------------------------------------------
    runs_dir: Path = Path("runs")
    save_screenshots: bool = True

    @property
    def viewport(self) -> dict[str, int]:
        """Viewport in the shape Playwright expects."""
        return {"width": self.viewport_width, "height": self.viewport_height}

    def contains_point(self, x: float, y: float) -> bool:
        """True if (x, y) falls inside the viewport coordinate space."""
        return 0 <= x < self.viewport_width and 0 <= y < self.viewport_height


class MolmoWebConfig(BaseModel):
    """Connection and prompting settings for the MolmoWeb model server.

    Separate from ``RunConfig`` because it describes the *policy*, not the
    browser -- and because a Phase 2 fleet shares one model server while each
    agent gets its own browser.

    The endpoint is never hardcoded: pass it explicitly or set
    ``MOLMOWEB_ENDPOINT``. The server is expected to be reachable over the
    network, so running the GPU on another machine is just a different URL.
    """

    endpoint: str = Field(
        description="Base URL of the model server, e.g. http://gpu-host:8001. "
        "The adapter POSTs to {endpoint}/predict."
    )
    # "molmo_web_think" is the style tag the model was trained with; it is
    # prepended to the user message as f"{system_message}: {user_message}".
    system_message: str = "molmo_web_think"
    # How many past steps to render into the prompt. The reference client uses
    # 10; MultimodalAgent's own default is 3.
    max_past_steps: int = Field(default=10, gt=0)
    timeout_s: float = Field(default=120.0, gt=0)
    # None means "let the server use its configured default".
    temperature: float | None = None
    top_p: float | None = None

    @classmethod
    def from_env(cls, endpoint: str | None = None, **overrides) -> "MolmoWebConfig | None":
        """Build from ``MOLMOWEB_ENDPOINT`` unless an endpoint is passed.

        Returns ``None`` when no endpoint is configured, so callers can skip
        model-dependent work instead of failing.
        """
        resolved = endpoint or os.environ.get("MOLMOWEB_ENDPOINT", "").strip()
        if not resolved:
            return None
        return cls(endpoint=resolved, **overrides)


class QwenConfig(BaseModel):
    """Connection, prompting and spend settings for the PP API (Qwen) adapter.

    Contract below is not from the vendor docs -- those proved stale -- but was
    verified against the live API by ``scripts/smoke_api.py``:

      * endpoint ``{base_url}/v1/chat/completions``; the ``/v1`` is REQUIRED.
        Without it the host returns HTTP 200 carrying the marketing site's HTML,
        so a 200 alone does not mean the call reached the API.
      * ``Authorization: Bearer <key>``
      * images via OpenAI-style ``image_url`` with a ``data:image/png;base64,``
        data URL. A bare base64 string is rejected with HTTP 400.

    The key is a ``SecretStr``: printing this config, dumping it into a log line
    or serialising it into run.json all yield ``**********`` rather than the
    credential. Reaching the real value takes an explicit
    ``.get_secret_value()``, which is easy to grep for in review.
    """

    api_key: SecretStr
    base_url: str
    model: str = "qwen3.5-27b"

    # --- prompting ---------------------------------------------------------
    max_past_steps: int = Field(default=5, gt=0)
    # Matches MolmoWeb's max_past_images=0 convention: history is text, and only
    # the current screenshot is sent. Images dominate cost on a metered API.
    temperature: float = 0.7
    max_tokens: int = Field(default=1024, gt=0)

    # --- reliability -------------------------------------------------------
    timeout_s: float = Field(default=120.0, gt=0)
    max_attempts: int = Field(default=4, gt=0)
    backoff_base_s: float = Field(default=1.5, gt=0)
    backoff_cap_s: float = Field(default=30.0, gt=0)

    # --- spend -------------------------------------------------------------
    # A 4-agent best-of-N at max_steps=15 is at most 60 calls, so 100 leaves
    # headroom while still bounding a runaway loop -- which would otherwise be
    # unbounded. Every HTTP attempt counts, retries included, because retries
    # cost money too.
    max_calls_per_run: int = Field(default=100, gt=0)

    @classmethod
    def from_env(cls, **overrides) -> "QwenConfig | None":
        """Build from ``PPAPI_KEY`` / ``PPAPI_BASE_URL``.

        Returns ``None`` when either is missing, so callers can skip rather than
        fail -- the same contract as ``MolmoWebConfig.from_env``.
        """
        load_env_file()
        key = os.environ.get("PPAPI_KEY", "").strip()
        base = os.environ.get("PPAPI_BASE_URL", "").strip()
        if not key or not base:
            return None
        if not base.startswith(("http://", "https://")):
            base = "https://" + base
        base = base.rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        model = overrides.pop("model", None) or os.environ.get(
            "PPAPI_MODEL", "qwen3.5-27b"
        )
        return cls(api_key=SecretStr(key), base_url=base, model=model, **overrides)

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/v1/chat/completions"


class ManagerConfig(BaseModel):
    """The Phase 3 manager LLM: which model plans, and how much it may replan.

    Connection settings are NOT here. The manager talks to the same PP API as
    the policy through the same client, so it inherits the key, base URL,
    retries and shared call budget from ``QwenConfig`` -- what is manager-
    specific is the model and the planning budget, and those are the two knobs
    worth having in one obvious place.

    ``model`` defaults to empty, meaning "whatever the policy is using". MACU
    found the manager's strength to be the single setting that mattered most,
    so this is written to be changed: point it at a stronger model and nothing
    else about the run moves.

    ``planning_budget`` is in DAG *edits*, not API calls, and 10 is the paper's
    default. Setting it to 0 is a supported configuration, not a broken one:
    the initial decomposition still runs and replanning is skipped entirely,
    which is the ablation, expressed as a config value rather than a code path.
    """

    model: str = Field(
        default="", description="Empty means: the same model the policy uses."
    )
    planning_budget: int = Field(
        default=10, ge=0, description="DAG edits allowed across the whole run."
    )
    max_subtasks: int = Field(
        default=6,
        gt=0,
        description="Ceiling on the initial decomposition. Every subtask is a "
        "whole browser and a whole model budget.",
    )
    #: Waves, not steps. Bounds a pathological plan-grow-plan loop; the edit
    #: budget already bounds growth, so this is a backstop, not the main limit.
    max_waves: int = Field(default=8, gt=0)

    # Planning is not creative work: near-greedy decoding gives more consistent
    # graphs than the 0.7 the browsing policy runs at.
    temperature: float = Field(default=0.2, ge=0)
    max_tokens: int = Field(default=2048, gt=0)

    @classmethod
    def from_env(cls, **overrides) -> "ManagerConfig":
        """Build from ``PPAPI_MANAGER_MODEL`` / ``MANAGER_PLANNING_BUDGET``.

        Always returns a config: unlike the connection settings, every value
        here has a usable default, so a missing environment means "the
        defaults", not "cannot run".
        """
        load_env_file()
        if "model" not in overrides:
            overrides["model"] = os.environ.get("PPAPI_MANAGER_MODEL", "").strip()
        if "planning_budget" not in overrides:
            raw = os.environ.get("MANAGER_PLANNING_BUDGET", "").strip()
            if raw:
                overrides["planning_budget"] = int(raw)
        return cls(**overrides)


class JudgeConfig(BaseModel):
    """The Phase 3 LLM judge. Opt-in; ``MockJudge`` stays the default.

    Same arrangement as ``ManagerConfig``: connection comes from the shared
    client, this holds only what is judge-specific. Temperature is 0 because a
    judge that picks a different winner from the same trajectories is not a
    judge -- reproducibility is the property that makes its verdicts worth
    comparing across runs.
    """

    model: str = Field(
        default="", description="Empty means: the same model the policy uses."
    )
    temperature: float = Field(default=0.0, ge=0)
    max_tokens: int = Field(default=512, gt=0)

    @classmethod
    def from_env(cls, **overrides) -> "JudgeConfig":
        load_env_file()
        if "model" not in overrides:
            overrides["model"] = os.environ.get("PPAPI_JUDGE_MODEL", "").strip()
        return cls(**overrides)
