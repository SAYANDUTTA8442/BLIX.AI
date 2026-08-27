"""
FastAPI server — Blix v0.3.3  (Feature 1 flagship)

Creates and returns the FastAPI application with all routers mounted.
Can be run directly:

    uvicorn blix.api.server:app --reload --port 8000

Or imported as a library:

    from api.server import create_app, app

OpenAPI docs:   http://localhost:8000/docs
Redoc:          http://localhost:8000/redoc
Health check:   GET /health
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from api.context import BlixContext
from api.deps import set_context

log = logging.getLogger(__name__)


import os as _os

def _cors_origins() -> list[str]:
    """
    Return the list of allowed CORS origins (C03).

    Defaults to common local dev ports.  Override by setting the
    BLIX_CORS_ORIGINS environment variable to a comma-separated list
    of origins, e.g.::

        BLIX_CORS_ORIGINS=http://localhost:3000,http://localhost:5173
    """
    env = _os.environ.get("BLIX_CORS_ORIGINS", "")
    if env.strip():
        return [o.strip() for o in env.split(",") if o.strip()]
    return [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
    ]


def create_app(memory_dir: Path | None = None) -> FastAPI:
    """
    Build and configure the FastAPI application.

    Parameters
    ----------
    memory_dir:
        Override the default ``memory/`` directory (useful for tests).
    """

    # ----------------------------------------------------------------
    # Lifespan: construct BlixContext on startup, shut down cleanly
    # ----------------------------------------------------------------

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
        ctx = BlixContext.build(memory_dir)
        set_context(ctx)
        log.info("Blix API started — memory_dir=%s", ctx.memory_dir)
        yield
        ctx.shutdown()
        log.info("Blix API stopped cleanly.")

    # ----------------------------------------------------------------
    # App definition
    # ----------------------------------------------------------------

    application = FastAPI(
        title="Blix — Cognitive Knowledge Platform",
        description=(
            "REST API for Blix v0.3.3: chat, memory, knowledge graph, "
            "reflection, document processing, goal tracking, and dashboard statistics.\n\n"
            "All endpoints are async. Memory extraction and graph updates run in the "
            "background — chat latency is never blocked by post-processing."
        ),
        version="0.3.3",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ----------------------------------------------------------------
    # CORS (permissive for local dev; tighten for production)
    # ----------------------------------------------------------------

    # C07: Enforce a maximum request body size to prevent memory exhaustion.
    # Starlette/FastAPI have no built-in body limit; a client can POST
    # an arbitrarily large JSON body and fill server memory.
    # 4 MB covers the largest legitimate payload (long conversation history).
    # File uploads are handled separately with their own 10 MB limit (C08).
    _MAX_BODY_BYTES = int(_os.environ.get("BLIX_MAX_BODY_BYTES", 4 * 1024 * 1024))

    class _BodySizeLimitMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            content_length = request.headers.get("content-length")
            if content_length is not None:
                try:
                    cl_int = int(content_length)
                except ValueError:
                    # Malformed Content-Length header — reject to be safe
                    from starlette.responses import JSONResponse
                    return JSONResponse(
                        status_code=400,
                        content={"detail": f"Invalid Content-Length header: {content_length!r}"},
                    )
                if cl_int > _MAX_BODY_BYTES:
                    from starlette.responses import JSONResponse
                    return JSONResponse(
                        status_code=413,
                        content={"detail": (
                            f"Request body too large: {cl_int:,} bytes. "
                            f"Maximum is {_MAX_BODY_BYTES:,} bytes."
                        )},
                    )
            return await call_next(request)(request)

    application.add_middleware(_BodySizeLimitMiddleware)

    # C12: Attach a unique X-Request-ID to every request so errors can be
    # correlated across log lines. Accepts a client-supplied ID if present,
    # otherwise generates one.  The ID is echoed back in the response header.
    import uuid as _uuid

    class _RequestIDMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            req_id = request.headers.get("x-request-id") or str(_uuid.uuid4())
            response = await call_next(request)
            response.headers["X-Request-ID"] = req_id
            return response

    application.add_middleware(_RequestIDMiddleware)

    application.add_middleware(
        CORSMiddleware,
        # C03: allow_origins=["*"] + allow_credentials=True is rejected by
        # browsers (CORS spec forbids wildcard + credentials). Restrict to
        # known local dev origins; override via BLIX_CORS_ORIGINS env var
        # for non-default setups.
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
    )

    # ----------------------------------------------------------------
    # Routers
    # ----------------------------------------------------------------

    from api.routers.chat import router as chat_router
    from api.routers.memory import router as memory_router
    from api.routers.knowledge import router as knowledge_router
    from api.routers.reflection import router as reflection_router
    from api.routers.graph import router as graph_router
    from api.routers.documents import router as documents_router
    from api.routers.stats_goals import stats_router, goals_router
    from api.routers.reasoning_research import reason_router, research_router
    from api.routers.agent import router as agent_router
    from api.routers.temporal import router as temporal_router
    from api.routers.metacognition import router as metacognition_router
    from api.routers.workspace import router as workspace_router
    from api.routers.ml import router as ml_router
    from api.routers.causality import router as causality_router
    from api.routers.search import router as search_router
    from api.routers.curiosity import router as curiosity_router
    from api.routers.world_model import router as world_model_router
    from api.routers.simulation import router as simulation_router
    from api.routers.agents import router as agents_router
    from api.routers.specialists import router as specialists_router

    application.include_router(chat_router)
    application.include_router(memory_router)
    application.include_router(knowledge_router)
    application.include_router(reflection_router)
    application.include_router(graph_router)
    application.include_router(documents_router)
    application.include_router(stats_router)
    application.include_router(goals_router)
    application.include_router(reason_router)
    application.include_router(research_router)
    application.include_router(agent_router)
    application.include_router(temporal_router)
    application.include_router(metacognition_router)
    application.include_router(workspace_router)
    application.include_router(ml_router)
    application.include_router(causality_router)
    application.include_router(search_router)
    application.include_router(curiosity_router)
    application.include_router(world_model_router)
    application.include_router(simulation_router)
    application.include_router(agents_router)
    application.include_router(specialists_router)

    # ----------------------------------------------------------------
    # Health / root
    # ----------------------------------------------------------------

    @application.get("/health", tags=["System"], summary="Health check")
    async def health() -> dict:
        return {"status": "ok", "version": "0.3.3"}

    @application.get("/", tags=["System"], summary="API info", include_in_schema=False)
    async def root() -> dict:
        return {
            "name": "Blix Cognitive Knowledge Platform",
            "version": "0.3.3",
            "docs": "/docs",
            "health": "/health",
        }

    return application


# ---------------------------------------------------------------------------
# Module-level app instance for uvicorn
# ---------------------------------------------------------------------------

app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
