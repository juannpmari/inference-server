# OpenAI-compatible API gateway
# Routes /v1/chat/completions, /v1/completions, and /v1/models to engine workers.

import json
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from data_plane.gateway.config import GatewayConfig
from shared.config_loader import get_config
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
_routing_cfg = get_config("routing")

# Model-to-Service mapping ("routing table")
MODEL_SERVICE_MAP: dict[str, str] = _routing_cfg.get("model_service_map", {
    "Qwen/Qwen2-0.5B-Instruct": "http://engine:8080",
})


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient(timeout=_config.request_timeout)
    # Separate client for streaming with no read timeout
    app.state.stream_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)
    )
    yield
    await app.state.http_client.aclose()
    await app.state.stream_client.aclose()

app = FastAPI(title="Inference Gateway", lifespan=lifespan)


# ---------------------------------------------------------------------------
# OpenAI-style error handler
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


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "healthy"}


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
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
    return ModelObject(id=model_id, created=0, owned_by="organization").model_dump()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_worker(model: str) -> str:
    worker_url = MODEL_SERVICE_MAP.get(model)
    if worker_url is None:
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found.")
    return worker_url


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
    http_client: httpx.AsyncClient = app.state.http_client
    try:
        resp = await http_client.post(f"{worker_url}/generate", json=engine_payload)
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail=f"Model service '{request.model}' is unreachable.")

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


def _stream_completion(worker_url: str, payload: dict, model: str, completion_id: str):
    async def _event_generator():
        stream_client: httpx.AsyncClient = app.state.stream_client
        try:
            async with stream_client.stream("POST", f"{worker_url}/generate/stream", json=payload) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[len("data: "):]
                    if raw == "[DONE]":
                        yield "data: [DONE]\n\n"
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
            error_chunk = json.dumps({"error": {"message": f"Model service unreachable", "type": "server_error"}})
            yield f"data: {error_chunk}\n\n"

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    # Reject unsupported features clearly
    for msg in request.messages:
        if msg.tool_calls:
            raise HTTPException(status_code=400, detail="tool_calls are not supported.")

    if request.response_format and request.response_format.get("type") == "json_object":
        logger.warning("response_format json_object requested but not enforced by engine")

    worker_url = _resolve_worker(request.model)
    http_client: httpx.AsyncClient = app.state.http_client

    # 1. Render messages via engine chat template
    messages_dicts = [m.model_dump(exclude_none=True) for m in request.messages]
    try:
        tmpl_resp = await http_client.post(
            f"{worker_url}/chat/apply_template",
            json={"messages": messages_dicts, "add_generation_prompt": True},
        )
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail=f"Model service '{request.model}' is unreachable.")

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
    try:
        resp = await http_client.post(f"{worker_url}/generate", json=engine_payload)
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail=f"Model service '{request.model}' is unreachable.")

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


def _stream_chat_completion(worker_url: str, payload: dict, model: str, completion_id: str):
    async def _event_generator():
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
            async with stream_client.stream("POST", f"{worker_url}/generate/stream", json=payload) as resp:
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
            error_chunk = json.dumps({"error": {"message": "Model service unreachable", "type": "server_error"}})
            yield f"data: {error_chunk}\n\n"

    return StreamingResponse(_event_generator(), media_type="text/event-stream")
