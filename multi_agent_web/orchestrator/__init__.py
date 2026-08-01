"""Phase 2: the parallel execution substrate, plus best-of-N.

    orchestrate(task, strategy, policy_factory, ...) -> OrchestrationResult

``Runner`` runs N isolated ``AgentSession``s concurrently under two independent
limits (browsers: local, memory-bound; model requests: shared, GPU-bound) and
records where wall-clock went. ``Strategy`` decides what to launch and how to
turn the results into one answer; ``BestOfN`` is the only one implemented.

Deliberately NOT here: a manager LLM and DAG decomposition. Those are Phase 3
and slot in as another ``Strategy`` -- see ``strategy/base.py`` for how the
interface was shaped to accept them.
"""

from __future__ import annotations

from .judge.base import Candidate, Judge, Verdict
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

__all__ = [
    "RUN_JSON_SCHEMA_VERSION",
    "AgentSession",
    "BestOfN",
    "BrowserPool",
    "Candidate",
    "Judge",
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
