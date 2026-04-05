"""Tests for shared.preflight module."""

import pytest

from shared.preflight import PreflightCheck, PreflightError, run_preflight


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _pass():
    return True


async def _fail():
    return False


async def _raise():
    raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_pass():
    checks = [
        PreflightCheck(name="a", check=_pass, critical=True, message="should pass"),
        PreflightCheck(name="b", check=_pass, critical=False, message="should pass"),
    ]
    # Should not raise
    await run_preflight(checks, "test")


@pytest.mark.asyncio
async def test_critical_fail_raises():
    checks = [
        PreflightCheck(name="good", check=_pass, critical=True, message="ok"),
        PreflightCheck(name="bad", check=_fail, critical=True, message="No GPU"),
    ]
    with pytest.raises(PreflightError) as exc_info:
        await run_preflight(checks, "test")
    assert "bad" in str(exc_info.value)
    assert "No GPU" in str(exc_info.value)
    assert len(exc_info.value.failures) == 1


@pytest.mark.asyncio
async def test_warning_only_passes():
    checks = [
        PreflightCheck(name="warn_chk", check=_fail, critical=False, message="low mem"),
    ]
    # Should not raise even though check failed
    await run_preflight(checks, "test")


@pytest.mark.asyncio
async def test_mixed_results():
    checks = [
        PreflightCheck(name="ok1", check=_pass, critical=True, message="fine"),
        PreflightCheck(name="warn1", check=_fail, critical=False, message="warn msg"),
        PreflightCheck(name="crit1", check=_fail, critical=True, message="crit msg 1"),
        PreflightCheck(name="crit2", check=_fail, critical=True, message="crit msg 2"),
    ]
    with pytest.raises(PreflightError) as exc_info:
        await run_preflight(checks, "test")
    assert len(exc_info.value.failures) == 2
    assert "crit1" in str(exc_info.value)
    assert "crit2" in str(exc_info.value)


@pytest.mark.asyncio
async def test_exception_in_check_treated_as_failure():
    checks = [
        PreflightCheck(name="exploder", check=_raise, critical=True, message="exploded"),
    ]
    with pytest.raises(PreflightError) as exc_info:
        await run_preflight(checks, "test")
    assert "exploder" in str(exc_info.value)
