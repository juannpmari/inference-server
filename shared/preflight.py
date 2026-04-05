"""Pre-flight check framework for inference server components.

Run a set of checks at startup and fail fast on critical misconfigurations.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, List

logger = logging.getLogger(__name__)


@dataclass
class PreflightCheck:
    """A single pre-flight check to run at startup."""

    name: str
    check: Callable[[], Awaitable[bool]]
    critical: bool
    message: str


class PreflightError(Exception):
    """Raised when one or more critical pre-flight checks fail."""

    def __init__(self, summary: str, failures: List[str]):
        self.failures = failures
        super().__init__(summary)


async def run_preflight(checks: List[PreflightCheck], component: str) -> None:
    """Run all pre-flight checks for *component*.

    Logs each result.  If any critical check fails, raises :class:`PreflightError`
    with a summary of **all** failed checks.  Warning-only failures are logged at
    WARNING level but do not block startup.
    """
    failures: List[str] = []

    for chk in checks:
        try:
            passed = await chk.check()
        except Exception as exc:
            logger.warning(
                "[%s] Preflight '%s' raised an exception: %s", component, chk.name, exc
            )
            passed = False

        if passed:
            logger.info("[%s] Preflight OK: %s", component, chk.name)
        elif chk.critical:
            logger.error(
                "[%s] Preflight FAILED (critical): %s — %s",
                component,
                chk.name,
                chk.message,
            )
            failures.append(f"{chk.name}: {chk.message}")
        else:
            logger.warning(
                "[%s] Preflight WARN: %s — %s", component, chk.name, chk.message
            )

    if failures:
        summary = (
            f"[{component}] {len(failures)} critical preflight check(s) failed: "
            + "; ".join(failures)
        )
        raise PreflightError(summary, failures)
