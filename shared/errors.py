"""Structured error handling for the inference server.

Provides a canonical ``ErrorCode`` enum, a serializable ``ErrorResponse``
model, and an ``InferenceServerError`` exception that middleware can catch
and render consistently across all three services.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, Optional

from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Error codes
# ---------------------------------------------------------------------------

class ErrorCode(IntEnum):
    """Canonical error codes shared across gateway, engine, and sidecar."""

    # Gateway (1xxx)
    MODEL_NOT_FOUND = 1001
    ENGINE_UNREACHABLE = 1002
    RATE_LIMITED = 1004

    # Engine (2xxx)
    ENGINE_NOT_READY = 2001
    QUEUE_FULL = 2002
    INFERENCE_TIMEOUT = 2003

    # Sidecar (3xxx)
    SIDECAR_NOT_READY = 3001
    MODEL_DOWNLOAD_FAILED = 3002
    DISK_SPACE_INSUFFICIENT = 3005


# Map error codes to HTTP status codes
_CODE_TO_HTTP: dict[ErrorCode, int] = {
    ErrorCode.MODEL_NOT_FOUND: 404,
    ErrorCode.ENGINE_UNREACHABLE: 503,
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.ENGINE_NOT_READY: 503,
    ErrorCode.QUEUE_FULL: 429,
    ErrorCode.INFERENCE_TIMEOUT: 504,
    ErrorCode.SIDECAR_NOT_READY: 503,
    ErrorCode.MODEL_DOWNLOAD_FAILED: 500,
    ErrorCode.DISK_SPACE_INSUFFICIENT: 507,
}


# ---------------------------------------------------------------------------
# Error response model
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    """Serializable error payload returned to callers."""

    error_code: int
    message: str
    request_id: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)

    def to_json_response(self, status_code: Optional[int] = None) -> JSONResponse:
        """Return a FastAPI ``JSONResponse`` for this error."""
        http_status = status_code
        if http_status is None:
            try:
                http_status = _CODE_TO_HTTP[ErrorCode(self.error_code)]
            except (ValueError, KeyError):
                http_status = 500

        headers: dict[str, str] = {}
        if self.request_id:
            headers["X-Request-ID"] = self.request_id

        return JSONResponse(
            status_code=http_status,
            content=self.model_dump(),
            headers=headers or None,
        )


# ---------------------------------------------------------------------------
# Exception class
# ---------------------------------------------------------------------------

class InferenceServerError(Exception):
    """Raise from any service to produce a structured error response.

    Middleware in ``shared.middleware`` catches this and renders it via
    ``ErrorResponse.to_json_response()``.
    """

    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        *,
        details: Optional[Dict[str, Any]] = None,
        status_code: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.details = details or {}
        self.status_code = status_code or _CODE_TO_HTTP.get(error_code, 500)
        self.headers = headers or {}
