"""Quick script to send a single inference request through the gateway and print
the generated text, TTFT, and end-to-end latency."""

import json
import time

import httpx

GATEWAY_URL = "http://localhost:8000"
MODEL = "Qwen/Qwen2-0.5B-Instruct"
MAX_TOKENS = 128

input_query = "The train arrived at Platform 7 without a sound, as if it had been waiting there all along. Mateo almost missed it. He had been staring at"


def main():
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": input_query}],
        "max_tokens": MAX_TOKENS,
        "stream": True,
    }

    tokens: list[str] = []
    ttft: float | None = None

    t_start = time.perf_counter()

    with httpx.Client(timeout=httpx.Timeout(connect=10, read=120, write=10, pool=10)) as client:
        with client.stream("POST", f"{GATEWAY_URL}/v1/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str == "[DONE]":
                    break
                chunk = json.loads(data_str)
                content = chunk["choices"][0]["delta"].get("content")
                if content:
                    if ttft is None:
                        ttft = time.perf_counter() - t_start
                    tokens.append(content)

    e2e = time.perf_counter() - t_start
    generated_text = "".join(tokens)

    print(f"Prompt:    {input_query}")
    print(f"Response:  {generated_text}")
    print(f"TTFT:      {ttft * 1000:.1f} ms")
    print(f"E2E:       {e2e * 1000:.1f} ms")
    print(f"Tokens:    {len(tokens)}")


if __name__ == "__main__":
    main()
