"""FastAPI application factory and local backend entry point."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.services.benchmark_service import BenchmarkService

STATIC_DIR = Path(__file__).resolve().parent / "static"


def health_check() -> dict[str, str]:
    """Return readiness without loading credentials or contacting OpenRouter."""

    return {"status": "ok"}


def create_app(*, benchmark_service: BenchmarkService | None = None) -> FastAPI:
    """Create an application with an injectable benchmark service.

    The default service uses deterministic mock adapters. Live execution is
    selected explicitly per request and checks configuration before it queues
    any work.
    """

    service = benchmark_service or BenchmarkService()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            service.shutdown(wait=True)

    application = FastAPI(
        title="Rubik's Cube Race",
        version="0.2.0",
        description=(
            "Backend API for deterministic and explicitly opted-in live "
            "Rubik's Cube benchmark runs. Mock mode is the default."
        ),
        lifespan=lifespan,
    )
    application.state.benchmark_service = service
    application.mount(
        "/static",
        StaticFiles(directory=STATIC_DIR, check_dir=True),
        name="static",
    )
    application.add_api_route(
        "/",
        lambda: FileResponse(STATIC_DIR / "index.html"),
        methods=["GET"],
        include_in_schema=False,
    )
    # Register this directly before the included router. Besides keeping health
    # usable without service dependencies, this preserves route introspection
    # for lightweight smoke checks.
    application.add_api_route(
        "/api/health",
        health_check,
        methods=["GET"],
        tags=["health"],
        summary="Check backend readiness",
    )
    application.include_router(router)
    return application


app = create_app()

__all__ = ["app", "create_app", "health_check"]
