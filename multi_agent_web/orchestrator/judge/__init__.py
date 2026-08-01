"""Judges: pick the best candidate trajectory.

``MockJudge`` is deterministic and needs no API key. An LLM judge is Phase 3;
``base.py`` documents the seam.
"""

from __future__ import annotations

from .base import Candidate, Judge, Verdict
from .mock import MockJudge

__all__ = ["Candidate", "Judge", "MockJudge", "Verdict"]
