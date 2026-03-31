# APC vs LMCache Benchmark Protocol

## Structure

```
1 system prompt (shared by ALL groups)
    └── 16 prefix groups (each = different RAG document set)
         └── 5 questions per group
              = 80 total prompts
```

## Layers

### System prompt

A single ~250-char persona instruction ("You are a senior technical analyst..."). Identical across every request.

### Prefix groups (16)

Each group simulates a different RAG retrieval. A group consists of:

- 2-3 fake internal engineering docs (e.g., Group 0 = K8s autoscaling docs, Group 2 = GPU memory docs, Group 15 = incident response docs)
- Filler appendix paragraphs padded to ~2048 tokens total (selected round-robin per group so each prefix is unique)

The prefix is constructed as 3 chat messages: `system` + `user` (with the RAG docs) + `assistant` ("I have reviewed the documents..."). This is the cacheable part.

### Questions (5 per group)

A final `user` message appended after the prefix. This is the only part that varies within a group. E.g., for the K8s group: "What is the maximum number of nodes?" etc.

## Why this design matters for the benchmark

The key insight is the **`ACTIVE_GROUPS`** parameter (`[1, 2, 3, 4, 5, 6, 7, 8]`). At each benchmark level, only prompts from groups `0..N-1` are used:

- **1 active group** — all 80 requests share the same ~2048-token prefix — near-100% cache hits
- **8 active groups** — 8 distinct prefixes competing for GPU KV cache — cache pressure increases

This is what the x-axis of the plot represents. It tests how well each caching strategy (APC vs LMCache) degrades as the prefix working set grows.

## Request ordering

Prompts are ordered **round-robin**: `g0q0, g1q0, g2q0, ..., g7q0, g0q1, g1q1, ...`. Consecutive requests always hit different groups, so cache pressure scales smoothly with concurrency rather than hitting it in bursts.

## Summary

| Concept        | What it is                              | Count          |
|----------------|-----------------------------------------|----------------|
| System prompt  | Shared persona instruction              | 1              |
| Prefix group   | Unique RAG doc set (~2048 tokens)       | 16             |
| Question       | Unique user query per group             | 5 per group    |
| Prompt         | prefix + question                       | 80 total       |
| Active groups  | How many prefixes compete at each level | 1 → 8 (sweep)  |

## Running the benchmark

```bash
# 1. Generate prompts
python experiments/generate_prompts.py

# 2. Run APC-only scenario (start vLLM with --enable-prefix-caching, no LMCache)
python experiments/run_benchmark.py --scenario apc --url http://localhost:8080

# 3. Restart vLLM with LMCache enabled
bash experiments/start_lmcache.sh
python experiments/run_benchmark.py --scenario lmcache --url http://localhost:8080

# 4. Plot combined results (auto-runs if both results exist, or manually)
python experiments/run_benchmark.py --plot-only
```

### Tunable flags

| Flag                   | Default           | Description                          |
|------------------------|-------------------|--------------------------------------|
| `--concurrency`        | 8                 | Fixed concurrent requests            |
| `--levels`             | 1,2,3,4,5,6,7,8  | Active prefix groups per level       |
| `--requests-per-level` | 80                | Requests sent at each level          |
| `--runs`               | 3                 | Repeat each level N times and average|
| `--output-dir`         | `experiments/`    | Where results/plots go               |
| `--prompts`            | `experiments/prompts.json` | Prompt file path            |
