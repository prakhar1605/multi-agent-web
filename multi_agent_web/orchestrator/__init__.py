"""The parallel execution substrate, and the strategies that use it.

    orchestrate(task, strategy, policy_factory, ...) -> OrchestrationResult

``Runner`` runs N isolated ``AgentSession``s concurrently under two independent
limits (browsers: local, memory-bound; model requests: shared, GPU-bound) and
records where wall-clock went. ``Strategy`` decides what to launch and how to
turn the results into one answer.

Two strategies now: ``BestOfN`` (Phase 2) runs one task N ways and has a judge
pick; ``DagStrategy`` (Phase 3) runs a manager LLM's dependency graph in waves,
replanning between them. The second went in without touching this interface --
``Runner.run_sessions`` was already re-entrant and
``StrategyOutcome.contributing_indices`` was already a list, both for its
benefit. The manager itself lives in ``multi_agent_web.manager``, which knows
nothing about browsers.
"""

from __future__ import annotations

from .judge.base import Candidate, Judge, Verdict
from .judge.llm import LLMJudge
from .judge.mock import MockJudge
from .runner import (
    RUN_JSON_SCHEMA_VERSION,
    OrchestrationResult,
    OrchestrationTiming,
    OrchestratorConfig,
    Runner,
    orchestrate,
    write_run_json,
)
from .session import (
    AgentSession,
    BrowserPool,
    ModelSlot,
    PolicyFactory,
    SessionResult,
    SessionSpec,
    SessionTiming,
    ThrottledPolicy,
)
from .strategy.base import Strategy, StrategyOutcome
from .strategy.best_of_n import BestOfN
from .strategy.dag import DagStrategy

__all__ = [
    "RUN_JSON_SCHEMA_VERSION",
    "AgentSession",
    "BestOfN",
    "BrowserPool",
    "Candidate",
    "DagStrategy",
    "Judge",
    "LLMJudge",
    "MockJudge",
    "ModelSlot",
    "OrchestrationResult",
    "OrchestrationTiming",
    "OrchestratorConfig",
    "PolicyFactory",
    "Runner",
    "SessionResult",
    "SessionSpec",
    "SessionTiming",
    "Strategy",
    "StrategyOutcome",
    "ThrottledPolicy",
    "Verdict",
    "orchestrate",
    "write_run_json",
]
