"""Swappable agent policies.

``MolmoWebPolicy`` is imported lazily so that importing this package never
pulls in a model runtime -- ``from multi_agent_web.policy import MockPolicy``
must stay cheap and dependency-free.
"""

from __future__ import annotations

from typing import Any

from .base import AgentPolicy
from .mock import MockPolicy

__all__ = ["AgentPolicy", "MockPolicy", "MolmoWebPolicy"]


def __getattr__(name: str) -> Any:
    if name == "MolmoWebPolicy":
        from .molmoweb import MolmoWebPolicy

        return MolmoWebPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
