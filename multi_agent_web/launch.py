"""Assembling a run from a description of one. The single launch site.

Three things start runs -- ``scripts/run_multi.py``, ``scripts/run_demo.py``
and the web UI -- and each used to wire its own policy factory, strategy,
judge and API client. Three copies of that wiring is three chances for the
budget-sharing rule (one ``CallBudget`` for the agents AND the manager AND the
judge) to be applied in two of them, so it now lives here once.

``LaunchSpec`` is the description: plain values, all with defaults, so it can be
built from argparse, from a JSON body, or from a preset. ``prepare`` turns it
into the objects ``orchestrate`` needs and hands back a closer for the API
client. ``estimate_calls`` prices it before anything is spent -- the same
number ``run_demo.py`` shows before it asks for confirmation.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .config import JudgeConfig, ManagerConfig, MolmoWebConfig, QwenConfig, RunConfig
from .manager import Manager
from .orchestrator.judge.llm import LLMJudge
from .orchestrator.judge.mock import MockJudge
from .orchestrator.runner import OrchestratorConfig
from .orchestrator.session import PolicyFactory
from .orchestrator.strategy.base import Strategy
from .orchestrator.strategy.best_of_n import BestOfN
from .orchestrator.strategy.dag import DagStrategy
from .policy.base import AgentPolicy
from .policy.mock import MockPolicy
from .ppapi import CallBudget, PPAPIClient
from .presets import DEMO_SCRIPT

PolicyWrapper = Callable[[AgentPolicy, int], AgentPolicy]


class LaunchSpec(BaseModel):
    """Everything that decides what a run is. Nothing about how it is watched."""

    task: str = Field(min_length=1)
    strategy: Literal["best_of_n", "dag"] = "best_of_n"
    policy: Literal["mock", "qwen", "molmoweb"] = "mock"
    n: int = Field(default=3, ge=1, description="Agents (best_of_n) or tiles (demo).")
    max_steps: int = Field(default=8, ge=1)
    start_url: str | None = None

    # best_of_n
    judge: Literal["mock", "llm"] = "mock"
    judge_model: str | None = None

    # dag
    manager_model: str | None = None
    planning_budget: int = Field(default=10, ge=0)
    max_subtasks: int = Field(default=6, ge=1)
    max_waves: int = Field(default=6, ge=1)

    # spend
    max_calls: int = Field(default=100, ge=1)

    # browser
    headless: bool = True
    viewport_width: int = Field(default=1280, gt=0)
    viewport_height: int = Field(default=720, gt=0)
    settle_ms: int | None = None
    max_browsers: int = Field(default=4, ge=1)
    max_model: int = Field(default=2, ge=1)
    runs_dir: Path = Path("runs")

    # molmoweb
    endpoint: str | None = None

    def needs_api(self) -> bool:
        """True when something in this run talks to the PP API.

        The manager and the LLM judge are LLMs whatever the policy is, so a
        ``mock`` policy under ``dag`` still needs a key -- and is a genuinely
        useful combination: it prices a decomposition without paying for the
        browsing.
        """
        return self.policy == "qwen" or self.strategy == "dag" or self.judge == "llm"

    def run_config(self) -> RunConfig:
        kwargs: dict[str, Any] = dict(
            viewport_width=self.viewport_width,
            viewport_height=self.viewport_height,
            headless=self.headless,
            max_steps=self.max_steps,
            runs_dir=self.runs_dir,
        )
        if self.settle_ms is not None:
            kwargs["settle_ms"] = self.settle_ms
        return RunConfig(**kwargs)

    def orchestrator_config(self) -> OrchestratorConfig:
        return OrchestratorConfig(
            max_concurrent_browsers=self.max_browsers,
            max_inflight_model_requests=self.max_model,
        )


@dataclass(frozen=True)
class Estimate:
    """An upper bound on API calls, and how it was arrived at."""

    calls: int
    breakdown: str
    needs_api: bool
    api_configured: bool

    @property
    def free(self) -> bool:
        return not self.needs_api

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "breakdown": self.breakdown,
            "needs_api": self.needs_api,
            "api_configured": self.api_configured,
            "free": self.free,
        }


def estimate_calls(spec: LaunchSpec) -> Estimate:
    """Price a run before it starts.

    Two spenders under ``dag``, and the agent count is not ``n``: the manager
    decides how many subtasks there are (up to ``max_subtasks``) and may add up
    to ``planning_budget`` more, each of which is another whole agent. Budgeting
    for ``n`` there would abort a legitimate run partway through, which is a
    worse failure than quoting a larger number up front.
    """
    api_configured = QwenConfig.from_env() is not None
    if not spec.needs_api():
        return Estimate(0, "mock policy, mock judge: no API calls", False, api_configured)

    agent_calls = 0
    if spec.policy == "qwen":
        agent_calls = spec.max_steps
    if spec.strategy == "dag":
        agents = spec.max_subtasks + spec.planning_budget
        planning = 1 + spec.max_waves  # one decompose, then one replan per wave
        calls = agents * agent_calls + planning
        agent_part = (
            f"up to ({spec.max_subtasks} subtasks + {spec.planning_budget} added) "
            f"x {spec.max_steps} steps"
            if agent_calls
            else "mock agents (free)"
        )
        return Estimate(
            calls, f"{agent_part}, plus {planning} manager calls", True, api_configured
        )

    calls = spec.n * agent_calls + (1 if spec.judge == "llm" else 0)
    agent_part = (
        f"up to {spec.n} agents x {spec.max_steps} steps" if agent_calls else "mock agents (free)"
    )
    judge_part = ", plus 1 judge call" if spec.judge == "llm" else ""
    return Estimate(calls, agent_part + judge_part, True, api_configured)


@dataclass
class Prepared:
    """What ``orchestrate`` needs, plus what must be closed afterwards."""

    spec: LaunchSpec
    strategy: Strategy
    policy_factory: PolicyFactory
    run_config: RunConfig
    orchestrator_config: OrchestratorConfig
    usage_provider: Callable[[], dict[str, Any]] | None
    budget: CallBudget | None
    _closers: list[Callable[[], Awaitable[None]]]

    async def aclose(self) -> None:
        for close in self._closers:
            try:
                await close()
            except Exception:  # pragma: no cover - best-effort teardown
                pass


class LaunchError(RuntimeError):
    """The spec cannot be launched as given -- typically a missing key."""


def prepare(spec: LaunchSpec, policy_wrapper: PolicyWrapper | None = None) -> Prepared:
    """Build the strategy and policy factory for ``spec``.

    ``policy_wrapper`` lets a caller decorate each agent's policy -- the demo
    wraps them in a narrator -- without the wiring here knowing or caring. It
    is applied to every policy the factory builds, once, with the agent index.

    Raises ``LaunchError`` rather than ``SystemExit`` so the web UI can turn a
    missing key into an HTTP 400 and the CLIs can print it.
    """
    qwen: QwenConfig | None = None
    budget: CallBudget | None = None
    api: PPAPIClient | None = None
    closers: list[Callable[[], Awaitable[None]]] = []

    if spec.needs_api():
        qwen = QwenConfig.from_env(max_calls_per_run=max(spec.max_calls, 1))
        if qwen is None:
            raise LaunchError(
                "PPAPI_KEY and PPAPI_BASE_URL must both be set (.env or environment) "
                f"-- {spec.policy}/{spec.strategy}/{spec.judge} talks to the API."
            )
        # ONE budget for everything this run spends: agents, manager, judge.
        budget = CallBudget(limit=qwen.max_calls_per_run)
        if spec.strategy == "dag" or spec.judge == "llm":
            api = PPAPIClient(qwen, budget=budget)
            closers.append(api.close)

    inner_factory = _policy_factory(spec, qwen, budget)
    if policy_wrapper is None:
        factory = inner_factory
    else:
        wrapper = policy_wrapper

        def factory(index: int) -> AgentPolicy:
            return wrapper(inner_factory(index), index)

    strategy = _strategy(spec, api)

    return Prepared(
        spec=spec,
        strategy=strategy,
        policy_factory=factory,
        run_config=spec.run_config(),
        orchestrator_config=spec.orchestrator_config(),
        usage_provider=budget.as_dict if budget else None,
        budget=budget,
        _closers=closers,
    )


def _policy_factory(
    spec: LaunchSpec, qwen: QwenConfig | None, budget: CallBudget | None
) -> PolicyFactory:
    """A factory, never a policy: each agent must get its own instance."""
    if spec.policy == "mock":
        # A fresh MockPolicy per agent: it holds a script cursor, which is
        # per-episode state just like the model adapters' history.
        return lambda index: MockPolicy(DEMO_SCRIPT)

    if spec.policy == "molmoweb":
        from .policy.molmoweb import MolmoWebPolicy

        config = MolmoWebConfig.from_env(spec.endpoint)
        if config is None:
            raise LaunchError(
                "No model endpoint configured. Pass an endpoint or set MOLMOWEB_ENDPOINT."
            )
        return lambda index: MolmoWebPolicy(config)

    if spec.policy == "qwen":
        from .policy.qwen import QwenPolicy

        assert qwen is not None and budget is not None
        return lambda index: QwenPolicy(qwen, budget=budget, agent_index=index)

    raise LaunchError(f"unknown policy: {spec.policy}")  # pragma: no cover


def _strategy(spec: LaunchSpec, api: PPAPIClient | None) -> Strategy:
    if spec.strategy == "best_of_n":
        if spec.judge == "llm":
            assert api is not None
            judge = LLMJudge(api, JudgeConfig.from_env(**_model(spec.judge_model)))
        else:
            judge = MockJudge()
        return BestOfN(n=spec.n, judge=judge, start_url=spec.start_url)

    assert api is not None
    manager = Manager(
        api,
        ManagerConfig.from_env(
            planning_budget=spec.planning_budget,
            max_subtasks=spec.max_subtasks,
            max_waves=spec.max_waves,
            **_model(spec.manager_model),
        ),
    )
    return DagStrategy(manager, start_url=spec.start_url)


def _model(model: str | None) -> dict[str, str]:
    """Pass ``model`` through only when given, so the environment still counts."""
    return {"model": model} if model else {}


__all__ = [
    "Estimate",
    "LaunchError",
    "LaunchSpec",
    "Prepared",
    "estimate_calls",
    "prepare",
]
