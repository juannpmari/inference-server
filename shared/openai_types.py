"""OpenAI-compatible request/response models for the inference server API."""

import time
import uuid
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: int = 1
    stream: bool = False
    stop: Optional[list[str]] = None
    max_tokens: Optional[int] = None
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[dict[str, float]] = None
    user: Optional[str] = None
    response_format: Optional[dict[str, str]] = None
    seed: Optional[int] = None


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: int = 1
    stream: bool = False
    stop: Optional[list[str]] = None
    max_tokens: Optional[int] = None
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str]


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: Optional[str]


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage


# ---------------------------------------------------------------------------
# Streaming chunk models
# ---------------------------------------------------------------------------

class ChatCompletionChunkDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


class CompletionChunkChoice(BaseModel):
    index: int
    text: str
    finish_reason: Optional[str] = None


class CompletionChunk(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChunkChoice]


# ---------------------------------------------------------------------------
# Error model
# ---------------------------------------------------------------------------

class OpenAIErrorDetail(BaseModel):
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class OpenAIErrorResponse(BaseModel):
    error: OpenAIErrorDetail


# ---------------------------------------------------------------------------
# Model listing
# ---------------------------------------------------------------------------

class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "organization"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelObject]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_completion_id(prefix: str = "cmpl") -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def now_unix() -> int:
    return int(time.time())
