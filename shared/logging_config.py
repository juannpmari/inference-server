"""Structured logging configuration for the inference server.

Provides:
- ``JSONFormatter`` — renders log records as single-line JSON objects with
  timestamp, level, component, logger name, message, request_id (from
  ``shared.middleware.request_id_ctx``), and exception info when present.
- ``configure_logging(component, ...)`` — one-call replacement for
  ``logging.basicConfig`` that sets up the root logger, installs the
  JSON formatter (or a human-readable one), and suppresses noisy
  third-party loggers.
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime, timezone


def _get_request_id() -> str | None:
    """Fetch the current request ID without importing at module level."""
    try:
        from shared.middleware import request_id_ctx
        return request_id_ctx.get()
    except Exception:
        return None


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON."""

    def __init__(self, component: str = "unknown") -> None:
        super().__init__()
        self.component = component

    def format(self, record: logging.LogRecord) -> str:
        entry: dict = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "component": self.component,
            "logger": record.name,
            "message": record.getMessage(),
        }

        rid = _get_request_id()
        if rid:
            entry["request_id"] = rid

        if record.exc_info and record.exc_info[1] is not None:
            entry["exception"] = "".join(traceback.format_exception(*record.exc_info))

        return json.dumps(entry, default=str)


class _HumanFormatter(logging.Formatter):
    """Fallback human-readable formatter (used when ``json_output=False``)."""

    def __init__(self, component: str = "unknown") -> None:
        super().__init__(
            fmt=f"%(asctime)s %(levelname)s [{component}/%(name)s] %(message)s",
        )


# Loggers to suppress (noisy HTTP clients, etc.)
_SUPPRESSED_LOGGERS = (
    "httpx",
    "httpcore",
    "httpcore.http11",
    "httpcore.connection",
    "hpack",
    "urllib3",
)


def configure_logging(
    component: str,
    *,
    level: str = "INFO",
    json_output: bool = True,
) -> None:
    """Configure the root logger for *component*.

    Call once during service startup (e.g. in the FastAPI lifespan).  This
    replaces any prior ``logging.basicConfig`` calls.

    Parameters
    ----------
    component:
        Service name (``"gateway"``, ``"engine"``, ``"sidecar"``).
    level:
        Root log level (default ``"INFO"``).
    json_output:
        If ``True`` (default), use ``JSONFormatter``; otherwise use a
        human-readable format.
    """
    root = logging.getLogger()
    # Remove any existing handlers (e.g. from basicConfig)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if json_output:
        handler.setFormatter(JSONFormatter(component=component))
    else:
        handler.setFormatter(_HumanFormatter(component=component))

    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addHandler(handler)

    # Suppress noisy third-party loggers
    for name in _SUPPRESSED_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
