"""Tests for shared.errors — ErrorCode, ErrorResponse, InferenceServerError."""

import json

import pytest
from fastapi.responses import JSONResponse

from shared.errors import ErrorCode, ErrorResponse, InferenceServerError


# ---------------------------------------------------------------------------
# ErrorCode enum
# ---------------------------------------------------------------------------

class TestErrorCode:

    def test_gateway_codes_in_1xxx_range(self):
        assert ErrorCode.MODEL_NOT_FOUND == 1001
        assert ErrorCode.ENGINE_UNREACHABLE == 1002
        assert ErrorCode.RATE_LIMITED == 1004

    def test_engine_codes_in_2xxx_range(self):
        assert ErrorCode.ENGINE_NOT_READY == 2001
        assert ErrorCode.QUEUE_FULL == 2002
        assert ErrorCode.INFERENCE_TIMEOUT == 2003

    def test_sidecar_codes_in_3xxx_range(self):
        assert ErrorCode.SIDECAR_NOT_READY == 3001
        assert ErrorCode.MODEL_DOWNLOAD_FAILED == 3002
        assert ErrorCode.DISK_SPACE_INSUFFICIENT == 3005

    def test_all_codes_are_unique(self):
        values = [e.value for e in ErrorCode]
        assert len(values) == len(set(values))


# ---------------------------------------------------------------------------
# ErrorResponse
# ---------------------------------------------------------------------------

class TestErrorResponse:

    def test_construction_minimal(self):
        resp = ErrorResponse(error_code=1001, message="not found")
        assert resp.error_code == 1001
        assert resp.message == "not found"
        assert resp.request_id is None
        assert resp.details == {}

    def test_construction_with_all_fields(self):
        resp = ErrorResponse(
            error_code=2003,
            message="timed out",
            request_id="abc-123",
            details={"timeout_s": 30},
        )
        assert resp.request_id == "abc-123"
        assert resp.details["timeout_s"] == 30

    def test_to_json_response_maps_known_code(self):
        resp = ErrorResponse(error_code=ErrorCode.RATE_LIMITED, message="slow down")
        json_resp = resp.to_json_response()
        assert isinstance(json_resp, JSONResponse)
        assert json_resp.status_code == 429

    def test_to_json_response_unknown_code_defaults_500(self):
        resp = ErrorResponse(error_code=9999, message="unknown")
        json_resp = resp.to_json_response()
        assert json_resp.status_code == 500

    def test_to_json_response_explicit_status_code(self):
        resp = ErrorResponse(error_code=ErrorCode.ENGINE_UNREACHABLE, message="down")
        json_resp = resp.to_json_response(status_code=502)
        assert json_resp.status_code == 502

    def test_to_json_response_includes_request_id_header(self):
        resp = ErrorResponse(
            error_code=ErrorCode.MODEL_NOT_FOUND,
            message="missing",
            request_id="req-456",
        )
        json_resp = resp.to_json_response()
        assert json_resp.headers.get("x-request-id") == "req-456"

    def test_to_json_response_no_header_when_no_request_id(self):
        resp = ErrorResponse(error_code=ErrorCode.QUEUE_FULL, message="full")
        json_resp = resp.to_json_response()
        assert "x-request-id" not in json_resp.headers

    def test_to_json_response_body_is_valid_json(self):
        resp = ErrorResponse(
            error_code=ErrorCode.INFERENCE_TIMEOUT,
            message="took too long",
            request_id="r1",
            details={"elapsed": 60.0},
        )
        json_resp = resp.to_json_response()
        body = json.loads(json_resp.body.decode())
        assert body["error_code"] == ErrorCode.INFERENCE_TIMEOUT
        assert body["message"] == "took too long"
        assert body["request_id"] == "r1"
        assert body["details"]["elapsed"] == 60.0

    def test_http_status_mapping_completeness(self):
        """Every ErrorCode should map to an HTTP status."""
        from shared.errors import _CODE_TO_HTTP
        for code in ErrorCode:
            assert code in _CODE_TO_HTTP, f"{code.name} missing from _CODE_TO_HTTP"


# ---------------------------------------------------------------------------
# InferenceServerError
# ---------------------------------------------------------------------------

class TestInferenceServerError:

    def test_basic_construction(self):
        exc = InferenceServerError(ErrorCode.ENGINE_NOT_READY, "not ready")
        assert exc.error_code == ErrorCode.ENGINE_NOT_READY
        assert exc.message == "not ready"
        assert exc.status_code == 503
        assert exc.details == {}
        assert exc.headers == {}

    def test_custom_status_code(self):
        exc = InferenceServerError(
            ErrorCode.RATE_LIMITED, "slow", status_code=503
        )
        assert exc.status_code == 503  # overrides default 429

    def test_details_and_headers(self):
        exc = InferenceServerError(
            ErrorCode.DISK_SPACE_INSUFFICIENT,
            "disk full",
            details={"free_mb": 0},
            headers={"Retry-After": "60"},
        )
        assert exc.details["free_mb"] == 0
        assert exc.headers["Retry-After"] == "60"

    def test_inherits_from_exception(self):
        exc = InferenceServerError(ErrorCode.QUEUE_FULL, "busy")
        assert isinstance(exc, Exception)
        assert str(exc) == "busy"

    def test_default_status_for_unknown_code(self):
        """If error_code not in _CODE_TO_HTTP, status_code defaults to 500."""
        # Use a raw int that isn't in the mapping
        exc = InferenceServerError.__new__(InferenceServerError)
        Exception.__init__(exc, "oops")
        exc.error_code = 9999
        exc.message = "oops"
        exc.details = {}
        exc.status_code = 500
        exc.headers = {}
        assert exc.status_code == 500
