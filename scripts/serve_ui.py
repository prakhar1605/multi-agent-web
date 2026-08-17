#!/usr/bin/env python3
"""Serve the viewer at http://localhost:8000.

    python scripts/serve_ui.py
    python scripts/serve_ui.py --runs-dir runs --port 8000

Replay mode needs nothing but a ``runs/`` directory. Live mode with the mock
policy is free; live mode with ``qwen`` or ``--strategy dag`` reads
``PPAPI_KEY`` / ``PPAPI_BASE_URL`` from ``.env`` like every other entry point.
The key is never sent to the page -- the server only reports whether it is
configured.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_agent_web.config import load_env_file  # noqa: E402
from multi_agent_web.ui import create_app  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the multi-agent viewer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    load_env_file()

    import uvicorn

    app = create_app(args.runs_dir)
    print(f"viewer: http://{args.host}:{args.port}   (runs from {args.runs_dir.resolve()})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
