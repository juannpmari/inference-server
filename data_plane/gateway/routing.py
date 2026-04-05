# OpenAI-compatible API gateway
# Routes /v1/chat/completions, /v1/completions, and /v1/models to engine workers.

import asyncio
import json
import logging
import time
import urllib.parse
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import generate_latest, REGISTRY

from data_plane.gateway.config import GatewayConfig
from data_plane.gateway import metrics as gateway_metrics
from shared.config_loader import get_config
from shared.errors import ErrorCode, InferenceServerError
from shared.preflight import PreflightCheck, run_preflight
from shared.tracing import init_tracing, instrument_app
from shared.logging_config import configure_logging
from shared.middleware import RequestIDMiddleware, register_error_handlers, request_id_ctx
from shared.rate_limiter import TokenBucketRateLimiter
from shared.resilience import CircuitBreaker, CircuitBreakerOpen
from shared.openai_types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
    CompletionChunk,
    CompletionChunkChoice,
    ModelObject,
    ModelListResponse,
    Usage,
    generate_completion_id,
    now_unix,
)

logger = logging.getLogger(__name__)

_config = GatewayConfig()
_draining: bool = False
_startup_complete: bool = False
_in_flight_count: int = 0

# Model-to-Service mapping ("routing table") — prefer config routes, fall
# back to legacy routing section from server_config.yaml.
if _config.routes:
    MODEL_SERVICE_MAP: dict[str, str] = dict(_config.routes)
else:
    _routing_cfg = get_config("routing")
    MODEL_SERVICE_MAP = _routing_cfg.get("model_service_map", {
        "Qwen/Qwen2-0.5B-Instruct": "http://engine:8080",
    })

# ---------------------------------------------------------------------------
# Circuit breaker for engine calls
# ---------------------------------------------------------------------------

_engine_circuit_breaker = CircuitBreaker()

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

_rate_limiter = TokenBucketRateLimiter(
    rate=_config.rate_limit_rps,
    burst=_config.rate_limit_burst,
)

# ---------------------------------------------------------------------------
# Engine health cache (for /ready cascading)
# ---------------------------------------------------------------------------

_engine_health_cache: dict = {"healthy": False, "checked_at": 0.0}
_ENGINE_HEALTH_TTL = 5.0


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _startup_complete, _draining

    init_tracing("gateway", _config.otlp_endpoint)
    instrument_app(app)
    configure_logging(
        "gateway",
        level=_config.log_level,
        json_output=_config.log_json,
    )

    # --- Pre-flight checks ---
    preflight_checks = []

    async def _routes_configured():
        return len(MODEL_SERVICE_MAP) > 0

    preflight_checks.append(PreflightCheck(
        name="At least one route configured",
        check=_routes_configured,
        critical=True,
        message="No routes configured",
    ))

    invalid_urls = []
    for url in MODEL_SERVICE_MAP.values():
        parsed = urllib.parse.urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            invalid_urls.append(url)

    if invalid_urls:
        async def _urls_valid():
            return False

        preflight_checks.append(PreflightCheck(
            name="Engine URLs are valid",
            check=_urls_valid,
            critical=True,
            message=f"Invalid engine URL in routes: {', '.join(invalid_urls)}",
        ))

    await run_preflight(preflight_checks, "gateway")

    # The http_client timeout is the outermost timeout in the cascade:
    # gateway.request_timeout > engine.sidecar_timeout > engine.inference_timeout
    app.state.http_client = httpx.AsyncClient(timeout=_config.request_timeout)
    # Separate client for streaming with no read timeout
    app.state.stream_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)
    )
    _startup_complete = True

    yield

    # Shutdown — drain in-flight requests
    _draining = True
    logger.info("Gateway drain started, rejecting new requests")
    deadline = time.monotonic() + _config.drain_timeout
    while time.monotonic() < deadline:
        if _in_flight_count <= 0:
            break
        logger.info(f"Gateway draining: {_in_flight_count} in-flight requests remaining")
        await asyncio.sleep(0.1)

    if _in_flight_count > 0:
        logger.warning(f"Gateway drain timeout: {_in_flight_count} requests still in-flight")
    else:
        logger.info("Gateway drain complete: all requests finished")

    await app.state.http_client.aclose()
    await app.state.stream_client.aclose()
    logger.info("Gateway shutdown complete")

app = FastAPI(title="Inference Gateway", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request size limit middleware (ASGI)
# ---------------------------------------------------------------------------

_MAX_REQUEST_BODY_BYTES = 1_048_576  # 1 MB


class RequestSizeLimitMiddleware:
    """ASGI middleware that rejects request bodies larger than 1 MB."""

    def __init__(self, app):  # type: ignore[no-untyped-def]
        self.app = app

    async def __call__(self, scope, receive, send):  # type: ignore[no-untyped-def]
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        body_size = 0
        body_parts: list[bytes] = []
        exceeded = False

        async def receive_wrapper():
            nonlocal body_size, exceeded
            message = await receive()
            if message["type"] == "http.request":
                chunk = message.get("body", b"")
                body_size += len(chunk)
                body_parts.append(chunk)
                if body_size > _MAX_REQUEST_BODY_BYTES:
                    exceeded = True
            return message

        # Read the first message to check size
        first_message = await receive()
        if first_message["type"] == "http.request":
            chunk = first_message.get("body", b"")
            body_size += len(chunk)
            body_parts.append(chunk)
            if body_size > _MAX_REQUEST_BODY_BYTES:
                exceeded = True

        if exceeded:
            response = JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "message": "Request body too large (max 1 MB)",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": None,
                    }
                },
            )
            await response(scope, receive, send)
            return

        # Replay the first message and forward
        replay_done = False

        async def replay_receive():
            nonlocal replay_done
            if not replay_done:
                replay_done = True
                return first_message
            msg = await receive()
            return msg

        await self.app(scope, replay_receive, send)


# Wire up middleware (order: outermost first)
app.add_middleware(RequestSizeLimitMiddleware)
app.add_middleware(RequestIDMiddleware)
register_error_handlers(app, openai_compat=True)


# ---------------------------------------------------------------------------
# OpenAI-style error handler (kept for plain HTTPException compat)
# ---------------------------------------------------------------------------

_STATUS_TO_ERROR_TYPE = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    429: "rate_limit_error",
}


@app.exception_handler(HTTPException)
async def openai_error_handler(request: Request, exc: HTTPException):
    error_type = _STATUS_TO_ERROR_TYPE.get(exc.status_code, "server_error")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": error_type,
                "param": None,
                "code": None,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Return pydantic validation errors in structured OpenAI-compatible format."""
    errors = exc.errors()
    messages = []
    for err in errors:
        loc = " -> ".join(str(x) for x in err.get("loc", []))
        msg = err.get("msg", "")
        messages.append(f"{loc}: {msg}" if loc else msg)
    detail = "; ".join(messages)
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": detail,
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }
        },
    )


@app.exception_handler(InferenceServerError)
async def inference_server_error_handler(request: Request, exc: InferenceServerError):
    error_type = _STATUS_TO_ERROR_TYPE.get(exc.status_code, "server_error")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "type": error_type,
                "param": None,
                "code": exc.error_code.value if hasattr(exc.error_code, "value") else exc.error_code,
            }
        },
        headers=exc.headers or None,
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

async def _check_engine_health() -> bool:
    """Check if at least one engine is reachable and ready, with caching."""
    now = time.monotonic()
    if now - _engine_health_cache["checked_at"] < _ENGINE_HEALTH_TTL:
        return _engine_health_cache["healthy"]

    healthy = False
    for url in MODEL_SERVICE_MAP.values():
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{url}/ready")
                if resp.status_code == 200:
                    healthy = True
                    break
        except Exception:
            continue

    _engine_health_cache["healthy"] = healthy
    _engine_health_cache["checked_at"] = now
    return healthy


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/ready")
async def ready():
    if _draining:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "draining"}
        )
    engine_ok = await _check_engine_health()
    if not engine_ok:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "reason": "engine_unreachable",
                "error_code": ErrorCode.ENGINE_UNREACHABLE.value,
            },
        )
    return {"status": "ready"}


@app.get("/healthz")
async def healthz():
    """Liveness probe — always returns 200."""
    return {"status": "alive"}


@app.get("/readyz")
async def readyz():
    """Readiness probe — returns 200 when not draining and engines are reachable."""
    if _draining:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "draining"},
        )
    engine_ok = await _check_engine_health()
    if not engine_ok:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "engines_unreachable"},
        )
    return {"status": "ready"}


@app.get("/startupz")
async def startupz():
    """Startup probe — returns 200 once startup is complete."""
    if not _startup_complete:
        return JSONResponse(
            status_code=503,
            content={"status": "starting"},
        )
    return {"status": "started"}


@app.get("/metrics")
async def metrics_endpoint():
    return Response(content=generate_latest(REGISTRY), media_type="text/plain; charset=utf-8")


def _check_gateway_draining():
    """Raise HTTPException if gateway is draining."""
    if _draining:
        raise HTTPException(
            status_code=503,
            detail="Shutting down",
            headers={"Retry-After": "1"},
        )


def _check_rate_limit():
    """Raise HTTPException(429) if rate limit exceeded."""
    if not _rate_limiter.allow():
        retry_after = _rate_limiter.retry_after()
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(max(1, int(retry_after + 0.5)))},
        )


# ---------------------------------------------------------------------------
# Model listing
# ---------------------------------------------------------------------------

@app.get("/v1/models")
async def list_models():
    data = [
        ModelObject(id=model_id, created=0, owned_by="organization").model_dump()
        for model_id in MODEL_SERVICE_MAP
    ]
    return ModelListResponse(data=data).model_dump()


@app.get("/v1/models/{model_id:path}")
async def get_model(model_id: str):
    if model_id not in MODEL_SERVICE_MAP:
        raise InferenceServerError(
            ErrorCode.MODEL_NOT_FOUND,
            f"Model '{model_id}' not found.",
        )
    return ModelObject(id=model_id, created=0, owned_by="organization").model_dump()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_worker(model: str) -> str:
    worker_url = MODEL_SERVICE_MAP.get(model)
    if worker_url is None:
        raise InferenceServerError(
            ErrorCode.MODEL_NOT_FOUND,
            f"Model '{model}' not found.",
        )
    return worker_url


def _request_id_headers() -> dict[str, str]:
    """Build headers dict with current request ID for upstream calls."""
    rid = request_id_ctx.get()
    if rid:
        return {"X-Request-ID": rid}
    return {}


async def _engine_post(client: httpx.AsyncClient, url: str, json: dict) -> httpx.Response:
    """POST to engine through the circuit breaker."""
    async def _do_post():
        return await client.post(url, json=json, headers=_request_id_headers())

    try:
        return await _engine_circuit_breaker.call(_do_post)
    except CircuitBreakerOpen as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Engine circuit breaker open, retry in {exc.time_remaining:.0f}s",
        )


def _sampling_kwargs(
    temperature=None, top_p=None, max_tokens=None, stop=None,
    presence_penalty=0.0, frequency_penalty=0.0, seed=None,
) -> dict:
    """Build the sampling kwargs dict for the engine /generate request."""
    d: dict = {}
    if temperature is not None:
        d["temperature"] = temperature
    if top_p is not None:
        d["top_p"] = top_p
    if max_tokens is not None:
        d["max_tokens"] = max_tokens
    if stop is not None:
        d["stop"] = stop
    if presence_penalty:
        d["presence_penalty"] = presence_penalty
    if frequency_penalty:
        d["frequency_penalty"] = frequency_penalty
    if seed is not None:
        d["seed"] = seed
    return d


def _adapter_kwargs(adapter_identifier=None, adapter_version=None) -> dict:
    """Build adapter kwargs for the engine request."""
    d: dict = {}
    if adapter_identifier is not None:
        d["adapter_identifier"] = adapter_identifier
    if adapter_version is not None:
        d["adapter_version"] = adapter_version
    return d


# ---------------------------------------------------------------------------
# POST /v1/completions
# ---------------------------------------------------------------------------

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    global _in_flight_count
    _check_gateway_draining()
    _check_rate_limit()
    worker_url = _resolve_worker(request.model)
    sampling = _sampling_kwargs(
        temperature=request.temperature, top_p=request.top_p,
        max_tokens=request.max_tokens, stop=request.stop,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        seed=request.seed,
    )
    adapter = _adapter_kwargs(request.adapter_identifier, request.adapter_version)
    engine_payload = {"prompt": request.prompt, **sampling, **adapter}
    completion_id = generate_completion_id("cmpl")

    if request.stream:
        return _stream_completion(worker_url, engine_payload, request.model, completion_id)

    # Non-streaming
    start_time = time.monotonic()
    _in_flight_count += 1
    try:
        http_client: httpx.AsyncClient = app.state.http_client
        try:
            resp = await _engine_post(http_client, f"{worker_url}/generate", engine_payload)
        except httpx.ConnectError:
            gateway_metrics.gateway_requests_total.labels(model=request.model, status_code="503").inc()
            raise InferenceServerError(
                ErrorCode.ENGINE_UNREACHABLE,
                f"Model service '{request.model}' is unreachable.",
            )

        duration = time.monotonic() - start_time
        gateway_metrics.gateway_request_duration_seconds.labels(model=request.model).observe(duration)
        gateway_metrics.gateway_requests_total.labels(model=request.model, status_code=str(resp.status_code)).inc()

        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        data = resp.json()
        return CompletionResponse(
            id=completion_id,
            created=now_unix(),
            model=request.model,
            choices=[CompletionChoice(
                index=0,
                text=data["text"],
                finish_reason=data.get("finish_reason", "stop"),
            )],
            usage=Usage(
                prompt_tokens=data.get("prompt_tokens", 0),
                completion_tokens=data.get("tokens_generated", 0),
                total_tokens=data.get("prompt_tokens", 0) + data.get("tokens_generated", 0),
            ),
        ).model_dump()
    finally:
        _in_flight_count -= 1


def _stream_completion(worker_url: str, payload: dict, model: str, completion_id: str):
    # Streaming: FastAPI sends HTTP 200 before streaming begins, so we record
    # status_code="200" eagerly. Duration tracking for streams is not meaningful
    # (response time is dominated by generation length), so we skip it.
    async def _event_generator():
        global _in_flight_count
        _in_flight_count += 1
        gateway_metrics.gateway_requests_total.labels(model=model, status_code="200").inc()
        try:
            stream_client: httpx.AsyncClient = app.state.stream_client
            try:
                try:
                    # Check circuit breaker before starting the stream
                    if not _engine_circuit_breaker._should_allow():
                        elapsed = time.monotonic() - _engine_circuit_breaker._last_failure_time
                        remaining = max(0.0, _engine_circuit_breaker.recovery_timeout - elapsed)
                        raise CircuitBreakerOpen(_engine_circuit_breaker.recovery_timeout, remaining)
                except CircuitBreakerOpen:
                    error_chunk = json.dumps({
                        "error": {"message": "Engine circuit breaker open", "type": "server_error"}
                    })
                    yield f"data: {error_chunk}\n\n"
                    return

                async with stream_client.stream(
                    "POST",
                    f"{worker_url}/generate/stream",
                    json=payload,
                    headers=_request_id_headers(),
                ) as resp:
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        raw = line[len("data: "):]
                        if raw == "[DONE]":
                            yield "data: [DONE]\n\n"
                            _engine_circuit_breaker._record_success()
                            break
                        token_data = json.loads(raw)
                        chunk = CompletionChunk(
                            id=completion_id,
                            created=now_unix(),
                            model=model,
                            choices=[CompletionChunkChoice(
                                index=0,
                                text=token_data.get("token", ""),
                                finish_reason=token_data.get("finish_reason"),
                            )],
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"
            except httpx.ConnectError:
                _engine_circuit_breaker._record_failure()
                error_chunk = json.dumps({"error": {"message": "Model service unreachable", "type": "server_error"}})
                yield f"data: {error_chunk}\n\n"
        finally:
            _in_flight_count -= 1

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    global _in_flight_count
    _check_gateway_draining()
    _check_rate_limit()
    # Reject unsupported features clearly
    for msg in request.messages:
        if msg.tool_calls:
            raise HTTPException(status_code=400, detail="tool_calls are not supported.")

    if request.response_format and request.response_format.get("type") == "json_object":
        logger.warning("response_format json_object requested but not enforced by engine")

    worker_url = _resolve_worker(request.model)
    start_time = time.monotonic()
    http_client: httpx.AsyncClient = app.state.http_client

    # 1. Render messages via engine chat template
    messages_dicts = [m.model_dump(exclude_none=True) for m in request.messages]
    try:
        tmpl_resp = await _engine_post(
            http_client,
            f"{worker_url}/chat/apply_template",
            {"messages": messages_dicts, "add_generation_prompt": True},
        )
    except httpx.ConnectError:
        gateway_metrics.gateway_requests_total.labels(model=request.model, status_code="503").inc()
        raise InferenceServerError(
            ErrorCode.ENGINE_UNREACHABLE,
            f"Model service '{request.model}' is unreachable.",
        )

    if tmpl_resp.status_code >= 400:
        raise HTTPException(status_code=tmpl_resp.status_code, detail=tmpl_resp.text)

    prompt = tmpl_resp.json()["prompt"]

    # 2. Build engine request
    sampling = _sampling_kwargs(
        temperature=request.temperature, top_p=request.top_p,
        max_tokens=request.max_tokens, stop=request.stop,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        seed=request.seed,
    )
    adapter = _adapter_kwargs(request.adapter_identifier, request.adapter_version)
    engine_payload = {"prompt": prompt, **sampling, **adapter}
    completion_id = generate_completion_id("chatcmpl")

    if request.stream:
        return _stream_chat_completion(worker_url, engine_payload, request.model, completion_id)

    # 3. Non-streaming generation
    _in_flight_count += 1
    try:
        try:
            resp = await _engine_post(http_client, f"{worker_url}/generate", engine_payload)
        except httpx.ConnectError:
            gateway_metrics.gateway_requests_total.labels(model=request.model, status_code="503").inc()
            raise InferenceServerError(
                ErrorCode.ENGINE_UNREACHABLE,
                f"Model service '{request.model}' is unreachable.",
            )

        duration = time.monotonic() - start_time
        gateway_metrics.gateway_request_duration_seconds.labels(model=request.model).observe(duration)
        gateway_metrics.gateway_requests_total.labels(model=request.model, status_code=str(resp.status_code)).inc()

        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        data = resp.json()
        return ChatCompletionResponse(
            id=completion_id,
            created=now_unix(),
            model=request.model,
            choices=[ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=data["text"]),
                finish_reason=data.get("finish_reason", "stop"),
            )],
            usage=Usage(
                prompt_tokens=data.get("prompt_tokens", 0),
                completion_tokens=data.get("tokens_generated", 0),
                total_tokens=data.get("prompt_tokens", 0) + data.get("tokens_generated", 0),
            ),
        ).model_dump()
    finally:
        _in_flight_count -= 1


def _stream_chat_completion(worker_url: str, payload: dict, model: str, completion_id: str):
    # Streaming: FastAPI sends HTTP 200 before streaming begins, so we record
    # status_code="200" eagerly. Duration tracking for streams is not meaningful
    # (response time is dominated by generation length), so we skip it.
    async def _event_generator():
        global _in_flight_count
        _in_flight_count += 1
        gateway_metrics.gateway_requests_total.labels(model=model, status_code="200").inc()
        try:
            # First chunk: send the role
            first_chunk = ChatCompletionChunk(
                id=completion_id,
                created=now_unix(),
                model=model,
                choices=[ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(role="assistant"),
                )],
            )
            yield f"data: {first_chunk.model_dump_json()}\n\n"

            stream_client: httpx.AsyncClient = app.state.stream_client
            try:
                try:
                    # Check circuit breaker before starting the stream
                    if not _engine_circuit_breaker._should_allow():
                        elapsed = time.monotonic() - _engine_circuit_breaker._last_failure_time
                        remaining = max(0.0, _engine_circuit_breaker.recovery_timeout - elapsed)
                        raise CircuitBreakerOpen(_engine_circuit_breaker.recovery_timeout, remaining)
                except CircuitBreakerOpen:
                    error_chunk = json.dumps({
                        "error": {"message": "Engine circuit breaker open", "type": "server_error"}
                    })
                    yield f"data: {error_chunk}\n\n"
                    return

                async with stream_client.stream(
                    "POST",
                    f"{worker_url}/generate/stream",
                    json=payload,
                    headers=_request_id_headers(),
                ) as resp:
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        raw = line[len("data: "):]
                        if raw == "[DONE]":
                            # Send final chunk with finish_reason
                            final_chunk = ChatCompletionChunk(
                                id=completion_id,
                                created=now_unix(),
                                model=model,
                                choices=[ChatCompletionChunkChoice(
                                    index=0,
                                    delta=ChatCompletionChunkDelta(),
                                    finish_reason="stop",
                                )],
                            )
                            yield f"data: {final_chunk.model_dump_json()}\n\n"
                            yield "data: [DONE]\n\n"
                            _engine_circuit_breaker._record_success()
                            break
                        token_data = json.loads(raw)
                        token_text = token_data.get("token", "")
                        finish = token_data.get("finish_reason")

                        if finish:
                            # This is the last token event before [DONE]
                            if token_text:
                                chunk = ChatCompletionChunk(
                                    id=completion_id,
                                    created=now_unix(),
                                    model=model,
                                    choices=[ChatCompletionChunkChoice(
                                        index=0,
                                        delta=ChatCompletionChunkDelta(content=token_text),
                                    )],
                                )
                                yield f"data: {chunk.model_dump_json()}\n\n"
                            continue  # [DONE] will follow

                        if token_text:
                            chunk = ChatCompletionChunk(
                                id=completion_id,
                                created=now_unix(),
                                model=model,
                                choices=[ChatCompletionChunkChoice(
                                    index=0,
                                    delta=ChatCompletionChunkDelta(content=token_text),
                                )],
                            )
                            yield f"data: {chunk.model_dump_json()}\n\n"
            except httpx.ConnectError:
                _engine_circuit_breaker._record_failure()
                error_chunk = json.dumps({"error": {"message": "Model service unreachable", "type": "server_error"}})
                yield f"data: {error_chunk}\n\n"
        finally:
            _in_flight_count -= 1

    return StreamingResponse(_event_generator(), media_type="text/event-stream")
