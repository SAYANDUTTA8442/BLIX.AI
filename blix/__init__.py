"""
Blix — Cognitive AI Agent System.

Single source of truth for the package version.
Version is kept in sync with pyproject.toml; both must be updated together.
"""
from importlib.metadata import version as _pkg_version, PackageNotFoundError as _PNF

try:
    __version__: str = _pkg_version("blix")
except _PNF:
    # Fallback for editable installs where metadata may not yet be generated
    __version__ = "0.3.19.1"
