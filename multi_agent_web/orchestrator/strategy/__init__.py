"""Strategies: how one task becomes agent sessions and one answer.

``BestOfN`` runs one task N ways and picks. ``DagStrategy`` runs a manager's
dependency graph in waves and composes. Both went in behind the same interface,
unchanged -- which was the point of shaping it around the second one.
"""

from __future__ import annotations

from .base import Strategy, StrategyOutcome
from .best_of_n import BestOfN
from .dag import DagStrategy

__all__ = ["BestOfN", "DagStrategy", "Strategy", "StrategyOutcome"]
