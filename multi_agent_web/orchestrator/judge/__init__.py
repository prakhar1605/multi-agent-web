"""Judges: pick the best candidate trajectory.

``MockJudge`` is deterministic, needs no API key, and stays the default.
``LLMJudge`` is the Phase 3 opt-in and fills the seam ``base.py`` describes,
without changing it.
"""

from __future__ import annotations

from .base import Candidate, Judge, Verdict
from .llm import JudgeProtocolError, LLMJudge
from .mock import MockJudge

__all__ = [
    "Candidate",
    "Judge",
    "JudgeProtocolError",
    "LLMJudge",
    "MockJudge",
    "Verdict",
]
