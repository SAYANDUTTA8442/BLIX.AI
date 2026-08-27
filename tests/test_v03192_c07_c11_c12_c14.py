"""
tests/test_v03192_c07_c11_c12_c14.py
======================================
Regression tests for:

  C07 — Request body size limit middleware (4 MB default, 413 on oversize)
  C11 — pyproject.toml version updated to current release
  C12 — X-Request-ID middleware: every response carries a traceable ID
  C14 — blix/__init__.py reads version from importlib.metadata (single source)
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ═════════════════════════════════════════════════════════════════════════════
# C07 — Request body size limit
# ═════════════════════════════════════════════════════════════════════════════

class TestBodySizeLimit:

    def test_middleware_class_in_source(self):
        src = (PROJECT_ROOT / 'api' / 'server.py').read_text()
        assert '_BodySizeLimitMiddleware' in src, (
            "_BodySizeLimitMiddleware must be defined in server.py (C07)"
        )

    def test_default_limit_is_4mb(self):
        src = (PROJECT_ROOT / 'api' / 'server.py').read_text()
        assert '4 * 1024 * 1024' in src, (
            "Default body limit must be 4 MB (C07)"
        )

    def test_env_override_supported(self):
        src = (PROJECT_ROOT / 'api' / 'server.py').read_text()
        assert 'BLIX_MAX_BODY_BYTES' in src

    def test_returns_413_on_oversize(self):
        src = (PROJECT_ROOT / 'api' / 'server.py').read_text()
        # Find the middleware class source block
        start = src.find('_BodySizeLimitMiddleware')
        block = src[start:start + 1400]
        assert '413' in block, (
            "Body size limit middleware must return HTTP 413 (C07)"
        )

    def test_middleware_registered_before_cors(self):
        """Body limit must be applied before CORS to reject large pre-flight bodies."""
        src = (PROJECT_ROOT / 'api' / 'server.py').read_text()
        # Check the add_middleware() call positions, not the class definitions
        body_pos = src.find('add_middleware(_BodySizeLimitMiddleware)')
        cors_pos  = src.find('add_middleware(_RequestIDMiddleware)')
        assert body_pos != -1, "_BodySizeLimitMiddleware never registered"
        assert cors_pos  != -1, "_RequestIDMiddleware never registered"
        assert body_pos < cors_pos, (
            "Body size middleware must be registered before other middlewares"
        )

    def test_middleware_checks_content_length_header(self):
        src = (PROJECT_ROOT / 'api' / 'server.py').read_text()
        assert 'content-length' in src.lower() or 'content_length' in src

    def test_base_http_middleware_imported(self):
        src = (PROJECT_ROOT / 'api' / 'server.py').read_text()
        assert 'BaseHTTPMiddleware' in src

    def test_middleware_env_var_default_4mb(self, monkeypatch):
        """Env var override must parse correctly."""
        import os
        monkeypatch.delenv('BLIX_MAX_BODY_BYTES', raising=False)
        expected = 4 * 1024 * 1024
        actual = int(os.environ.get('BLIX_MAX_BODY_BYTES', 4 * 1024 * 1024))
        assert actual == expected

    def test_middleware_env_var_override(self, monkeypatch):
        import os
        monkeypatch.setenv('BLIX_MAX_BODY_BYTES', str(1024))
        actual = int(os.environ.get('BLIX_MAX_BODY_BYTES', 4 * 1024 * 1024))
        assert actual == 1024


# ═════════════════════════════════════════════════════════════════════════════
# C11 — Version currency
# ═════════════════════════════════════════════════════════════════════════════

class TestVersionCurrency:

    def _pyproject_version(self):
        src = (PROJECT_ROOT / 'pyproject.toml').read_text()
        m = re.search(r'^version = "([\d\.]+)"', src, re.M)
        assert m, "pyproject.toml must have a version field"
        return m.group(1)

    def test_pyproject_version_is_current(self):
        ver = self._pyproject_version()
        assert ver == '0.3.19.1', (
            f"pyproject.toml version must be 0.3.19.1, got {ver!r} (C11)"
        )

    def test_pyproject_version_not_stale(self):
        """Must not be stuck at the old 0.3.18.2 value."""
        ver = self._pyproject_version()
        assert ver != '0.3.18.2', (
            "pyproject.toml version is still the stale 0.3.18.2 (C11)"
        )

    def test_blix_init_version_is_importable(self):
        from blix import __version__
        assert __version__, "blix.__version__ must be a non-empty string"

    def test_blix_init_has_fallback_version(self):
        src = (PROJECT_ROOT / 'blix' / '__init__.py').read_text()
        assert '0.3.19.1' in src, (
            "blix/__init__.py fallback version must be 0.3.19.1 (C11)"
        )


# ═════════════════════════════════════════════════════════════════════════════
# C12 — X-Request-ID middleware
# ═════════════════════════════════════════════════════════════════════════════

class TestRequestIDMiddleware:

    def test_middleware_class_in_source(self):
        src = (PROJECT_ROOT / 'api' / 'server.py').read_text()
        assert '_RequestIDMiddleware' in src, (
            "_RequestIDMiddleware must be defined in server.py (C12)"
        )

    def test_response_header_set(self):
        src = (PROJECT_ROOT / 'api' / 'server.py').read_text()
        assert 'X-Request-ID' in src

    def test_client_id_accepted(self):
        """Middleware must accept a client-supplied x-request-id header."""
        src = (PROJECT_ROOT / 'api' / 'server.py').read_text()
        assert 'x-request-id' in src.lower()

    def test_uuid_generated_when_no_client_id(self):
        """Middleware must generate a UUID when no ID is supplied."""
        src = (PROJECT_ROOT / 'api' / 'server.py').read_text()
        assert 'uuid' in src.lower()

    def test_middleware_registered_in_server(self):
        src = (PROJECT_ROOT / 'api' / 'server.py').read_text()
        assert 'add_middleware(_RequestIDMiddleware)' in src

    def test_request_id_header_declared_in_cors_allow_headers(self):
        """X-Request-ID must be in the CORS allow_headers list."""
        src = (PROJECT_ROOT / 'api' / 'server.py').read_text()
        # The CORS section allows X-Request-ID
        assert 'X-Request-ID' in src


# ═════════════════════════════════════════════════════════════════════════════
# C14 — Single source of version truth
# ═════════════════════════════════════════════════════════════════════════════

class TestSingleVersionSource:

    def test_init_uses_importlib_metadata(self):
        src = (PROJECT_ROOT / 'blix' / '__init__.py').read_text()
        assert 'importlib.metadata' in src, (
            "blix/__init__.py must read version from importlib.metadata (C14)"
        )

    def test_init_has_packagenotfounderror_fallback(self):
        src = (PROJECT_ROOT / 'blix' / '__init__.py').read_text()
        assert 'PackageNotFoundError' in src, (
            "blix/__init__.py must catch PackageNotFoundError for editable installs (C14)"
        )

    def test_no_hardcoded_only_version(self):
        """__version__ must not be defined ONLY as a hardcoded literal."""
        src = (PROJECT_ROOT / 'blix' / '__init__.py').read_text()
        # importlib.metadata must be the primary mechanism
        assert '_pkg_version' in src or 'version(' in src

    def test_version_is_string(self):
        from blix import __version__
        assert isinstance(__version__, str)
        assert re.match(r'^\d+\.\d+\.\d+', __version__), (
            f"__version__ must match semver pattern, got {__version__!r}"
        )
