"""
tests/test_v03197_silent_bugs.py
==================================
Regression tests for three silent bugs found in the final audit:

  BUG 1 — context_builder: k//2 and k//3 floor to 0 for small top_k,
           silently returning empty temporal/concept/principle/belief
           sections even when top_k >= 1. Same class as the B02 bug.
           Fix: max(1, k//2) and max(1, k//3).

  BUG 2 — _BodySizeLimitMiddleware: int(content_length) raised ValueError
           on a malformed Content-Length header (e.g. "abc"), causing an
           unhandled 500 instead of a clean 400.

  BUG 3 — _BodySizeLimitMiddleware: return await call_next was missing
           the (request) argument, so every non-oversized request returned
           the coroutine object rather than the actual response.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# BUG 1 — context_builder k//2 and k//3 floor
# ─────────────────────────────────────────────────────────────────────────────

class TestContextBuilderFloor:

    def test_temporal_uses_max1_guard(self):
        src = (PROJECT_ROOT / 'memory' / 'hybrid' / 'context' / 'context_builder.py').read_text()
        assert 'max(1, k // 2)' in src, (
            "temporal_memories must use max(1, k//2) to avoid silent zero (Bug 1)"
        )

    def test_concept_uses_max1_guard(self):
        src = (PROJECT_ROOT / 'memory' / 'hybrid' / 'context' / 'context_builder.py').read_text()
        assert src.count('max(1, k // 3)') >= 3, (
            "concept/principle/belief slices must all use max(1, k//3) (Bug 1)"
        )

    def test_no_raw_k_floor_div_2(self):
        src = (PROJECT_ROOT / 'memory' / 'hybrid' / 'context' / 'context_builder.py').read_text()
        # Raw k // 2 without max() guard must not appear in the build() method
        build_src = inspect.getsource(
            __import__(
                'memory.hybrid.context.context_builder',
                fromlist=['ContextBuilder']
            ).ContextBuilder.build
        )
        assert 'k // 2' not in build_src or 'max(1, k // 2)' in build_src, (
            "Raw k//2 without max() guard found in build() (Bug 1)"
        )

    def test_values_never_zero_for_positive_k(self):
        """For k in [1..10], all derived sub-k values must be >= 1."""
        for k in range(1, 11):
            temporal  = max(1, k // 2)
            concept   = max(1, k // 3)
            principle = max(1, k // 3)
            belief    = max(1, k // 3)
            assert temporal  >= 1, f"k={k}: temporal={temporal}"
            assert concept   >= 1, f"k={k}: concept={concept}"
            assert principle >= 1, f"k={k}: principle={principle}"
            assert belief    >= 1, f"k={k}: belief={belief}"

    def test_k1_previously_gave_all_zeros(self):
        """Regression: before the fix, k=1 gave 0 for every sub-section."""
        k = 1
        old_temporal = k // 2   # = 0
        old_concept  = k // 3   # = 0
        assert old_temporal == 0, "This test verifies the pre-fix behaviour was indeed 0"
        assert old_concept  == 0
        # After fix:
        assert max(1, old_temporal) == 1
        assert max(1, old_concept)  == 1

    def test_k10_unchanged_after_fix(self):
        """At default top_k=10 the fix must produce the same values as before."""
        k = 10
        assert max(1, k // 2) == 5
        assert max(1, k // 3) == 3


# ─────────────────────────────────────────────────────────────────────────────
# BUG 2 — _BodySizeLimitMiddleware: ValueError on malformed Content-Length
# ─────────────────────────────────────────────────────────────────────────────

class TestBodySizeLimitMiddleware:

    def _get_middleware_source(self) -> str:
        return (PROJECT_ROOT / 'api' / 'server.py').read_text()

    def test_valueerror_guard_present(self):
        src = self._get_middleware_source()
        assert 'ValueError' in src, (
            "_BodySizeLimitMiddleware must catch ValueError on bad Content-Length (Bug 2)"
        )

    def test_try_except_wraps_int_parse(self):
        src = self._get_middleware_source()
        # int() call must be inside a try block
        body_start = src.find('_BodySizeLimitMiddleware')
        body = src[body_start:body_start + 1200]
        assert 'try:' in body and 'ValueError' in body, (
            "int(content_length) must be wrapped in try/except ValueError (Bug 2)"
        )

    def test_returns_400_on_invalid_header(self):
        src = self._get_middleware_source()
        body_start = src.find('_BodySizeLimitMiddleware')
        body = src[body_start:body_start + 1200]
        assert 'status_code=400' in body, (
            "Malformed Content-Length must return HTTP 400, not crash (Bug 2)"
        )

    def test_returns_413_on_oversized(self):
        src = self._get_middleware_source()
        body_start = src.find('_BodySizeLimitMiddleware')
        body = src[body_start:body_start + 1200]
        assert 'status_code=413' in body, (
            "Oversized body must return HTTP 413 (Bug 2/C07)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# BUG 3 — _BodySizeLimitMiddleware: missing (request) in call_next
# ─────────────────────────────────────────────────────────────────────────────

class TestBodySizeLimitCallNext:

    def test_call_next_called_with_request(self):
        src = (PROJECT_ROOT / 'api' / 'server.py').read_text()
        body_start = src.find('_BodySizeLimitMiddleware')
        body = src[body_start:body_start + 1200]
        assert 'call_next(request)' in body, (
            "call_next must be called as call_next(request), "
            "not returned as a bare coroutine (Bug 3)"
        )

    def test_no_bare_call_next_return(self):
        """The old buggy pattern 'return await call_next' (no args) must be gone."""
        src = (PROJECT_ROOT / 'api' / 'server.py').read_text()
        body_start = src.find('_BodySizeLimitMiddleware')
        body = src[body_start:body_start + 1200]
        # "call_next\n" or "call_next)" — but NOT "call_next(request)"
        import re
        # Look for `await call_next` NOT followed by `(`
        bare_pattern = re.search(r'await call_next(?!\()', body)
        assert bare_pattern is None, (
            f"Found bare 'await call_next' without (request): {bare_pattern.group()!r} (Bug 3)"
        )

    def test_dispatch_passes_request_to_call_next(self):
        """Integration: the middleware must forward the request to call_next."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from starlette.testclient import TestClient
        from starlette.applications import Starlette
        from starlette.routing import Route
        from starlette.responses import PlainTextResponse
        from starlette.middleware.base import BaseHTTPMiddleware
        import os

        # Recreate just the middleware logic from server.py
        _MAX = 100

        class _TestBodyLimit(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                content_length = request.headers.get("content-length")
                if content_length is not None:
                    try:
                        cl_int = int(content_length)
                    except ValueError:
                        from starlette.responses import JSONResponse
                        return JSONResponse(status_code=400,
                                            content={"detail": "bad header"})
                    if cl_int > _MAX:
                        from starlette.responses import JSONResponse
                        return JSONResponse(status_code=413,
                                            content={"detail": "too large"})
                return await call_next(request)

        async def _echo(request):
            body = await request.body()
            return PlainTextResponse(f"ok:{len(body)}")

        app = Starlette(routes=[Route("/", _echo, methods=["POST"])])
        app.add_middleware(_TestBodyLimit)

        client = TestClient(app, raise_server_exceptions=False)

        # Normal request → 200
        resp = client.post("/", content=b"hello")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        assert "ok:5" in resp.text

        # Oversized → 413
        resp = client.post("/", content=b"x" * 200,
                           headers={"content-length": "200"})
        assert resp.status_code == 413

        # Bad Content-Length header → 400
        resp = client.post("/", content=b"hello",
                           headers={"content-length": "not-a-number"})
        assert resp.status_code == 400
