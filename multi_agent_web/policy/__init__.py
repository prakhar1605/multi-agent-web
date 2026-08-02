"""Swappable agent policies.

Three adapters sit behind one interface:

  * ``MockPolicy``     -- scripted, for testing the system with no model at all
  * ``MolmoWebPolicy`` -- a purpose-trained web agent, its own output grammar
  * ``QwenPolicy``     -- a general VLM, told to emit our action schema directly

Model-backed policies are imported lazily so that importing this package never
pulls in an HTTP client or a model runtime -- ``from multi_agent_web.policy
import MockPolicy`` must stay cheap and dependency-free.
"""

from __future__ import annotations

from typing import Any

from .base import AgentPolicy
from .mock import MockPolicy

__all__ = ["AgentPolicy", "MockPolicy", "MolmoWebPolicy", "QwenPolicy"]


def __getattr__(name: str) -> Any:
    if name == "MolmoWebPolicy":
        from .molmoweb import MolmoWebPolicy

        return MolmoWebPolicy
    if name == "QwenPolicy":
        from .qwen import QwenPolicy

        return QwenPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
