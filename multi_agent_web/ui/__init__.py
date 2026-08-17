"""The web app: a FastAPI backend and one HTML page.

    python scripts/serve_ui.py            # http://localhost:8000

Replay mode reads ``runs/`` and needs no key. Live mode starts a run through
``launch.prepare`` with a ``RecordingSink`` attached and streams its events over
a WebSocket; when it finishes it is just another run in ``runs/``.
"""

from __future__ import annotations

from .server import create_app

__all__ = ["create_app"]
