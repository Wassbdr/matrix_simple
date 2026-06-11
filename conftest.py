"""Root conftest: make the repo root importable so ``serving.*`` modules
(plain directory, not an installed package) resolve in tests."""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
