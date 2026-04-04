"""Base dispatcher ABC shared by all dispatch modes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Prompt:
    """A single benchmark prompt with metadata."""

    name: str           # e.g., "short_003"
    text: str           # raw prompt text from file
    input_tokens: int   # from manifest.json
    max_tokens: int     # from workload config (32 or 64)

    def with_max_tokens(self, n: int) -> Prompt:
        """Return a shallow copy with a different max_tokens value."""
        return Prompt(
            name=self.name,
            text=self.text,
            input_tokens=self.input_tokens,
            max_tokens=n,
        )


@dataclass
class ResponseRecord:
    """Timing and metadata for a single benchmark request."""

    prompt_name: str
    input_tokens: int
    max_tokens: int
    ttft: float                # seconds, monotonic
    itl: list[float]           # inter-token intervals in seconds
    e2e: float                 # seconds, monotonic
    output_tokens: int         # count of tokens received
    http_status: int           # 200, 429, 503, etc.
    error: str | None          # None on success


class BaseDispatcher(ABC):
    """Abstract base class for all dispatch modes.

    Subclasses must implement ``run()`` which executes the workload and
    returns one ``ResponseRecord`` per request.
    """

    def __init__(self, client, prompts: list[Prompt], config: dict):
        self.client = client
        self.prompts = prompts
        self.config = config

    async def warmup(self, n: int = 3) -> None:
        """Send *n* throw-away requests to stabilise CUDA kernels."""
        for _ in range(n):
            await self.client.send_request(self.prompts[0])

    @abstractmethod
    async def run(self) -> list[ResponseRecord]:
        """Execute the workload and return one ResponseRecord per request."""
        ...
