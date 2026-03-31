#!/bin/bash
set -e
cd /app
uv pip install lmcache 2>&1 | tail -3
uv run python -c "import lmcache; print('lmcache installed OK')"
exec uv run python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2-1.5B-Instruct \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.45 \
  --max-model-len 4096 \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config":{"discard_partial_chunks":false,"max_local_cpu_size":0.1}}' \
  --host 0.0.0.0 \
  --port 8000
