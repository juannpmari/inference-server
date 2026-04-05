"""Request-ID middleware and error handlers for FastAPI services.

Provides:
- ``request_id_ctx`` — a ``ContextVar`` holding the current request ID.
- ``RequestIDMiddleware`` — raw ASGI middleware (not ``BaseHTTPMiddleware``)
  that generates or propagates ``X-Request-ID`` headers without buffering
  streaming responses.
- ``register_error_handlers`` — wires up ``InferenceServerError`` handling
  on a FastAPI app, with optional OpenAI-compatible error format.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from shared.errors import ErrorResponse, InferenceServerError

# ---------------------------------------------------------------------------
# Context variable
# ---------------------------------------------------------------------------

request_id_ctx: ContextVar[str | None] = ContextVar("request_id_ctx", default=None)


# ---------------------------------------------------------------------------
# Raw ASGI middleware (avoids BaseHTTPMiddleware body buffering issues)
# ---------------------------------------------------------------------------

class RequestIDMiddleware:
    """ASGI middleware that sets ``X-Request-ID`` on every request/response.

    If the incoming request carries an ``X-Request-ID`` header its value is
    reused; otherwise a new UUID-4 is generated.  The value is stored in
    ``request_id_ctx`` so that loggers and error responses can include it.
    """

    def __init__(self, app):  # type: ignore[no-untyped-def]
        self.app = app

    async def __call__(self, scope, receive, send):  # type: ignore[no-untyped-def]
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        # Extract or generate request ID
        headers = dict(scope.get("headers", []))
        incoming_id = headers.get(b"x-request-id", b"").decode() or None
        rid = incoming_id or uuid.uuid4().hex
        token = request_id_ctx.set(rid)

        async def send_with_request_id(message):  # type: ignore[no-untyped-def]
            if message["type"] == "http.response.start":
                response_headers = list(message.get("headers", []))
                response_headers.append((b"x-request-id", rid.encode()))
                message["headers"] = response_headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            request_id_ctx.reset(token)


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

_STATUS_TO_OPENAI_TYPE = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    429: "rate_limit_error",
}


def register_error_handlers(
    app: FastAPI,
    *,
    openai_compat: bool = False,
) -> None:
    """Register an ``InferenceServerError`` exception handler on *app*.

    When *openai_compat* is ``True`` (gateway), ``/v1/*`` endpoints render
    errors in the OpenAI ``{"error": {...}}`` envelope.
    """

    @app.exception_handler(InferenceServerError)
    async def _handle_inference_error(request: Request, exc: InferenceServerError):
        rid = request_id_ctx.get()

        # For /v1/* paths on the gateway, use OpenAI error format
        if openai_compat and request.url.path.startswith("/v1/"):
            error_type = _STATUS_TO_OPENAI_TYPE.get(exc.status_code, "server_error")
            content: dict[str, Any] = {
                "error": {
                    "message": exc.message,
                    "type": error_type,
                    "param": None,
                    "code": exc.error_code.value if hasattr(exc.error_code, "value") else exc.error_code,
                }
            }
            headers: dict[str, str] = dict(exc.headers)
            if rid:
                headers["X-Request-ID"] = rid
            return JSONResponse(
                status_code=exc.status_code,
                content=content,
                headers=headers or None,
            )

        # Default structured format
        resp = ErrorResponse(
            error_code=int(exc.error_code),
            message=exc.message,
            request_id=rid,
            details=exc.details,
        )
        json_resp = resp.to_json_response(status_code=exc.status_code)
        for k, v in exc.headers.items():
            json_resp.headers[k] = v
        return json_resp
