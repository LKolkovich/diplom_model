from __future__ import annotations

import logging
import logging.config
import time
from contextlib import asynccontextmanager
from typing import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import router
from app.config import get_settings


LOGGING_CONFIG: dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"],
    },
    "loggers": {
        "app": {"level": "DEBUG", "propagate": True},
        "uvicorn.access": {"level": "WARNING", "propagate": True},
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class MaxBodySizeMiddleware:
    """
    Ограничение тела запроса по Content-Length.
    Если заголовка нет, запрос пропускаем дальше — реальный hard-limit
    лучше еще продублировать на уровне ingress/nginx.
    """

    def __init__(self, app, max_size_bytes: int) -> None:
        self.app = app
        self.max_size_bytes = max_size_bytes

    async def __call__(
            self,
            scope,
            receive: Callable[[], Awaitable],
            send: Callable,
    ) -> None:
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            raw_length = headers.get(b"content-length")
            if raw_length is not None:
                try:
                    content_length = int(raw_length)
                    if content_length > self.max_size_bytes:
                        logger.warning(
                            "Request rejected: content_length=%d max_size_bytes=%d",
                            content_length,
                            self.max_size_bytes,
                        )
                        response = Response(
                            content=(
                                f'{{"detail":"Request body too large. '
                                f'Max allowed: {self.max_size_bytes} bytes"}}'
                            ),
                            status_code=413,
                            media_type="application/json",
                        )
                        await response(scope, receive, send)
                        return
                except ValueError:
                    logger.warning("Invalid Content-Length header: %r", raw_length)

        await self.app(scope, receive, send)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(
        "Starting service | tmp_dir=%s output_dir=%s auto_cleanup=%s max_upload_size_bytes=%s",
        getattr(settings, "tmp_dir", "tmp"),
        getattr(settings, "output_dir", "output"),
        getattr(settings, "auto_cleanup", False),
        getattr(settings, "max_upload_size_bytes", None),
    )
    yield
    logger.info("Shutting down service")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Valorant Voice Pipeline",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=getattr(settings, "cors_allow_origins", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(
        MaxBodySizeMiddleware,
        max_size_bytes=getattr(settings, "max_upload_size_bytes", 2 * 1024 * 1024 * 1024),
    )

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next) -> Response:
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "%s %s | status=%d elapsed_ms=%.1f",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    app.include_router(router, prefix="/tasks")
    return app


app = create_app()