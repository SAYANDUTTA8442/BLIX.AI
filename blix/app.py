"""
blix/app.py
===========
Package entry point for the ``blix`` console script (C05).

The ``[project.scripts]`` table in ``pyproject.toml`` declares::

    blix = "blix.app:main"

This module bridges that declaration to the actual CLI implementation
in the project root ``app.py``, which contains the full Rich-based
interactive shell.  Keeping the implementation in the root ``app.py``
preserves backwards-compatibility with ``python app.py`` invocations
while making the package properly installable.

Usage after ``pip install -e .``::

    blix            # interactive chat shell
    python app.py   # equivalent direct invocation
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def main() -> None:
    """
    Locate the root-level ``app.py`` and invoke its ``main()`` function.

    Resolution order:
    1. ``<package_root>/../../app.py``  — editable install / development
    2. ``<sys.prefix>/blix/app.py``    — future installed-data layout
    3. Current working directory        — last resort

    Raises ``SystemExit(1)`` with a clear message if ``app.py`` cannot
    be found, rather than producing a cryptic ``ImportError``.
    """
    # blix/ is one level inside the project root in both editable installs
    # and the source tree, so __file__/../.. → project root.
    candidates = [
        Path(__file__).resolve().parent.parent / "app.py",
        Path(sys.prefix) / "blix" / "app.py",
        Path.cwd() / "app.py",
    ]

    app_path: Path | None = None
    for candidate in candidates:
        if candidate.exists():
            app_path = candidate
            break

    if app_path is None:
        searched = "\n  ".join(str(c) for c in candidates)
        print(
            f"blix: could not find app.py. Searched:\n  {searched}\n"
            "Please run 'blix' from the project root or reinstall the package.",
            file=sys.stderr,
        )
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("blix._root_app", app_path)
    if spec is None or spec.loader is None:
        print(f"blix: failed to load {app_path}", file=sys.stderr)
        sys.exit(1)

    module = importlib.util.module_from_spec(spec)
    sys.modules["blix._root_app"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    if not hasattr(module, "main"):
        print(f"blix: {app_path} has no main() function", file=sys.stderr)
        sys.exit(1)

    module.main()
