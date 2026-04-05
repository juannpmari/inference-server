"""Tests for shared.logging_config — JSONFormatter and configure_logging."""

import json
import logging

import pytest

from shared.logging_config import JSONFormatter, configure_logging


# ---------------------------------------------------------------------------
# JSONFormatter
# ---------------------------------------------------------------------------

class TestJSONFormatter:

    def _make_record(self, message: str = "hello", level: int = logging.INFO) -> logging.LogRecord:
        return logging.LogRecord(
            name="test.logger",
            level=level,
            pathname="test.py",
            lineno=1,
            msg=message,
            args=(),
            exc_info=None,
        )

    def test_output_is_valid_json(self):
        fmt = JSONFormatter(component="engine")
        record = self._make_record("test message")
        output = fmt.format(record)
        data = json.loads(output)
        assert data["message"] == "test message"

    def test_contains_required_fields(self):
        fmt = JSONFormatter(component="gateway")
        record = self._make_record("check fields")
        data = json.loads(fmt.format(record))
        assert "timestamp" in data
        assert data["level"] == "INFO"
        assert data["component"] == "gateway"
        assert data["logger"] == "test.logger"
        assert data["message"] == "check fields"

    def test_timestamp_is_utc_iso(self):
        fmt = JSONFormatter(component="sidecar")
        record = self._make_record()
        data = json.loads(fmt.format(record))
        ts = data["timestamp"]
        assert "T" in ts  # ISO 8601 format
        assert ts.endswith("+00:00") or ts.endswith("Z")

    def test_output_is_single_line(self):
        fmt = JSONFormatter(component="engine")
        record = self._make_record("multi\nline\nmessage")
        output = fmt.format(record)
        # The JSON itself should be single line
        assert "\n" not in output

    def test_exception_info_included(self):
        fmt = JSONFormatter(component="engine")
        record = self._make_record("boom")
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            record.exc_info = sys.exc_info()
        output = fmt.format(record)
        data = json.loads(output)
        assert "exception" in data
        assert "ValueError" in data["exception"]
        assert "test error" in data["exception"]

    def test_no_exception_field_when_none(self):
        fmt = JSONFormatter(component="engine")
        record = self._make_record("no error")
        data = json.loads(fmt.format(record))
        assert "exception" not in data

    def test_request_id_appears_when_set(self):
        """When request_id_ctx is set, it should appear in the log."""
        from shared.middleware import request_id_ctx
        token = request_id_ctx.set("req-abc-123")
        try:
            fmt = JSONFormatter(component="gateway")
            record = self._make_record("with request id")
            data = json.loads(fmt.format(record))
            assert data["request_id"] == "req-abc-123"
        finally:
            request_id_ctx.reset(token)

    def test_no_request_id_when_unset(self):
        fmt = JSONFormatter(component="gateway")
        record = self._make_record("no request id")
        data = json.loads(fmt.format(record))
        assert "request_id" not in data

    def test_different_log_levels(self):
        fmt = JSONFormatter(component="engine")
        for level, name in [
            (logging.DEBUG, "DEBUG"),
            (logging.WARNING, "WARNING"),
            (logging.ERROR, "ERROR"),
        ]:
            record = self._make_record("msg", level=level)
            data = json.loads(fmt.format(record))
            assert data["level"] == name


# ---------------------------------------------------------------------------
# configure_logging
# ---------------------------------------------------------------------------

class TestConfigureLogging:

    def setup_method(self):
        """Save root logger state before each test."""
        self._original_handlers = logging.root.handlers[:]
        self._original_level = logging.root.level

    def teardown_method(self):
        """Restore root logger state after each test."""
        logging.root.handlers = self._original_handlers
        logging.root.level = self._original_level

    def test_sets_correct_level(self):
        configure_logging("test", level="DEBUG", json_output=False)
        assert logging.root.level == logging.DEBUG

    def test_sets_info_level_by_default(self):
        configure_logging("test", json_output=False)
        assert logging.root.level == logging.INFO

    def test_replaces_existing_handlers(self):
        logging.root.addHandler(logging.StreamHandler())
        logging.root.addHandler(logging.StreamHandler())
        assert len(logging.root.handlers) >= 2
        configure_logging("test", json_output=False)
        assert len(logging.root.handlers) == 1

    def test_json_output_uses_json_formatter(self):
        configure_logging("gateway", json_output=True)
        handler = logging.root.handlers[0]
        assert isinstance(handler.formatter, JSONFormatter)

    def test_human_output_does_not_use_json_formatter(self):
        configure_logging("engine", json_output=False)
        handler = logging.root.handlers[0]
        assert not isinstance(handler.formatter, JSONFormatter)

    def test_suppresses_noisy_loggers(self):
        configure_logging("sidecar", json_output=False)
        for name in ("httpx", "httpcore", "hpack", "urllib3"):
            assert logging.getLogger(name).level == logging.WARNING

    def test_invalid_level_defaults_to_info(self):
        configure_logging("test", level="NONEXISTENT", json_output=False)
        assert logging.root.level == logging.INFO
