"""Make the repo root importable so `import multi_agent_web` works in tests
without installing the package (no setup.py / pyproject packaging in Phase 1).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
