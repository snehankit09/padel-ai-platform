"""
Makes the top-level `ml/` package importable from backend code.

`ml/` lives outside `backend/` on purpose (see repo README) — it's the one
piece of the codebase meant to be reusable independently of the
FastAPI/Celery app, so it isn't installed as part of the `app` package and
is never on sys.path by default. This computes where it lives, relative to
this file, and adds its parent directory once.

That relative computation works out in both places this code actually
runs, without needing an explicit PYTHONPATH env var anywhere:

- Locally: this file is at backend/app/core/ml_path.py, so
  parents[3] is the repo root — ml/ sits right next to backend/.
- In the worker container: docker-compose mounts ./ml at /ml (a sibling
  of /app, which is what ./backend maps to via the Dockerfile's
  `COPY . .` into WORKDIR /app). This file then lives at
  /app/app/core/ml_path.py, so parents[3] is "/" — and "/" / "ml" is
  still /ml.

Deliberately NOT called at import time by anything the FastAPI API process
loads eagerly — see the deferred import in
app/services/frame_extraction_stage.py for why: the API container never
mounts ./ml at all (README: "imported by workers, not by the API
directly"), so importing an ml module there would fail. This function
itself is harmless to import from anywhere; it's calling it plus the
subsequent `import ml...` that requires ml/ to actually be present.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ML_DIR = Path(__file__).resolve().parents[3] / "ml"


def ensure_ml_importable() -> None:
    """Idempotent — adds ml/'s parent directory to sys.path if it isn't there already."""
    ml_parent = str(_ML_DIR.parent)
    if ml_parent not in sys.path:
        sys.path.insert(0, ml_parent)
