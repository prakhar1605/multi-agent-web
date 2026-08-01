"""Strategies: how one task becomes agent sessions and one answer."""

from __future__ import annotations

from .base import Strategy, StrategyOutcome
from .best_of_n import BestOfN

__all__ = ["BestOfN", "Strategy", "StrategyOutcome"]
