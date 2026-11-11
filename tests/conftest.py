"""Puts the repo root (main.py, procclass.py, v11.py) and src/ on sys.path and registers the markers."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for entry in (ROOT, ROOT / "src", ROOT / "tests"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))


def pytest_configure(config: object) -> None:
    config.addinivalue_line("markers", "unit: hermetic unit test")  # type: ignore[attr-defined]
    config.addinivalue_line("markers", "integration: needs Docker (Postgres + pgvector)")  # type: ignore[attr-defined]
