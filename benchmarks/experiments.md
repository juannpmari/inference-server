# Experiments

## Overview

Experiments are defined as YAML files in `benchmarks/experiments/`. Each experiment specifies a set of **conditions** (parameter combinations to compare), a **dispatch mode** (how requests are sent), and a **protocol** (warmup, repetitions, etc.).

A single generic orchestrator (`experiment_orchestrator.py`) executes any experiment YAML and outputs a JSON with all collected metrics. Separate plotting scripts in `benchmarks/plotting/` consume these JSON files to produce visualizations.

## Running an experiment

```bash
python3 -m benchmarks.experiment_orchestrator benchmarks/experiments/<experiment>.yaml
```

Options:
- `--engine-url URL` — Engine endpoint (default: `http://localhost:8080`)
- `--output PATH` — Output JSON path (default: `benchmarks/results/<experiment_name>.json`)
- `--restart` — Restart engine between conditions (works with both flat `conditions` and `condition_groups`; group-level env restarts still take priority at group boundaries)
- `--resume` — Skip conditions already present in the output file (useful for resuming after a crash)
- `--only cond1,cond2` — Run only the specified conditions (comma-separated)
- `--skip cond1,cond2` — Skip the specified conditions (comma-separated)
- `--dry-run` — Validate YAML and print the experiment plan without executing

**Incremental results:** If the output JSON already exists, the orchestrator loads it and preserves existing conditions. Re-running a condition overwrites its entry; new conditions are added alongside old ones. Use `--resume` to skip already-completed conditions entirely.

## Plotting results

Each experiment has a corresponding plotting script in `benchmarks/plotting/`:

```bash
python3 -m benchmarks.plotting.<plot_script> --input benchmarks/results/<experiment>.json --stat <mean|p50|p90|p99>
```

Output PNGs are saved to `resources/<experiment_name>/` by default (override with `--output-dir`).

## Experiment YAML format

```yaml
experiment:
  name: experiment_name
  description: "What this experiment measures"

  conditions:
    condition_a:
      input_length: 32            # input length in tokens (must be divisible by 32)
      max_tokens: 32              # max output tokens
    condition_b:
      input_length: 64
      max_tokens: 64

  dispatch:
    mode: sequential              # sequential | concurrent | realistic
    # concurrency: 4              # concurrent mode only
    # rps: 2.0                    # realistic mode only
    # duration_seconds: 60        # realistic mode only

  protocol:
    warmup_requests: 3
    runs: 3                       # median of N runs is selected
    prompts_per_condition: 10
    cooldown_seconds: 5
    # timeout_per_condition: 600  # seconds, default 600 — skip condition on timeout
```

### Grouped conditions (alternative format)

When an experiment needs to restart the engine with different settings between groups of conditions (e.g. toggling prefix caching), use `condition_groups` instead of `conditions`:

```yaml
experiment:
  name: experiment_name
  description: "..."

  condition_groups:
    - name: group_a
      engine_env:                    # env vars written to .env before engine restart
        ENGINE_ENABLE_PREFIX_CACHING: "false"
      conditions:
        cond_1:
          input_length: 32
          max_tokens: 32
        cond_2:
          input_length: 64
          max_tokens: 32
    - name: group_b
      engine_env:
        ENGINE_ENABLE_PREFIX_CACHING: "true"
      conditions:
        cond_3:
          input_length: 32
          max_tokens: 32

  dispatch:
    mode: sequential

  protocol:
    warmup_requests: 3
    runs: 3
    prompts_per_condition: 10
    cooldown_seconds: 5
```

The runner automatically restarts the engine (via `docker compose up -d --force-recreate`) when `engine_env` changes between groups. The original `.env` file is restored after the experiment completes.

### Per-condition dispatch overrides

Conditions can override top-level dispatch parameters by including an optional `dispatch` key. The condition-level values are merged on top of the top-level `dispatch` config, so unspecified keys inherit the defaults. This is useful for sweeping a dispatch parameter (e.g. concurrency level) across conditions:

```yaml
experiment:
  name: example
  conditions:
    low_concurrency:
      input_length: 32
      max_tokens: 128
      dispatch:
        concurrency: 4
    high_concurrency:
      input_length: 32
      max_tokens: 128
      dispatch:
        concurrency: 16
  dispatch:
    mode: concurrent
    concurrency: 4          # default, overridden per-condition
```

## Output JSON format

The orchestrator always collects the full set of metrics for every condition:

```
results/<dispatch_mode>/<experiment_name>/<experiment_name>.json
├── experiment: str
├── description: str
├── dispatch_mode: str
├── dispatch_config: { mode, concurrency, ... }
├── metadata:
│   ├── timestamp: ISO 8601 UTC start time
│   ├── git_commit: str
│   ├── experiment_yaml: { full YAML config }
│   ├── engine_url: str
│   └── total_duration_seconds: float
└── conditions:
    └── <condition_name>:
        ├── input_length, max_tokens, dispatch (effective config)
        ├── total/successful/failed requests
        ├── client_side_throughput:
        │   ├── total_output_tokens: int
        │   ├── wall_clock_seconds: float
        │   └── output_tokens_per_second: float
        ├── stats:
        │   ├── ttft:        { mean, p50, p90, p99 }
        │   ├── decode_time: { mean, p50, p90, p99 }
        │   ├── itl:         { mean, p50, p90, p99 }
        │   └── e2e:         { mean, p50, p90, p99 }
        ├── server_side_delta: { latency, throughput, kv_cache, errors }
        └── per_request: [ { prompt_name, input_tokens, output_tokens, ttft, decode_time, e2e, ... } ]
```

Plotting scripts select whichever metrics and aggregation they need from this standard output.

## Current experiments

### 1. Latency Composition

**Goal:** Decompose end-to-end latency into prefill (TTFT) and decode phases across four input/output size combinations.

**Conditions:**

| Condition | Input tokens | Output tokens | Total |
|-----------|-------------|---------------|-------|
| SISO (Short In, Short Out) | 32 | 32 | 64 |
| SILO (Short In, Long Out) | 32 | 512 | 544 |
| LISO (Long In, Short Out) | 512 | 32 | 544 |
| LILO (Long In, Long Out) | 512 | 512 | 1024 |

Long inputs are built by concatenating 16 short (32-token) prompts (16 × 32 = 512 tokens).

**Run:**
```bash
python3 -m benchmarks.experiment_orchestrator benchmarks/experiments/sequential/latency_composition.yaml
# Use --restart to restart the engine between conditions (avoids KV cache carryover):
python3 -m benchmarks.experiment_orchestrator benchmarks/experiments/sequential/latency_composition.yaml --restart
```

**Plot:**
```bash
python3 -m benchmarks.plotting.plot_latency_composition --input benchmarks/results/sequential/latency_composition/latency_composition.json --stat mean
```

**Output:** Stacked barchart showing prefill vs decode time for each condition.

---

### 2. TTFT vs Input Length

**Goal:** Measure how Time To First Token scales as input length increases from 32 to 256 tokens.

**Conditions:** 8 conditions (`input_32` through `input_256`), with `input_length` increasing in steps of 32. Output is kept short (`max_tokens: 32`) so decode doesn't dominate.

| Condition | Input tokens | Output tokens |
|-----------|-------------|---------------|
| input_32 | 32 | 32 |
| input_64 | 64 | 32 |
| input_96 | 96 | 32 |
| input_128 | 128 | 32 |
| input_160 | 160 | 32 |
| input_192 | 192 | 32 |
| input_224 | 224 | 32 |
| input_256 | 256 | 32 |

**Run:**
```bash
python3 -m benchmarks.experiment_orchestrator benchmarks/experiments/sequential/ttft_vs_input_length.yaml --restart
```

**Plot:**
```bash
python3 -m benchmarks.plotting.plot_ttft_vs_input_length --input benchmarks/results/sequential/ttft_vs_input_length/ttft_vs_input_length.json --stat p90 --no-scatter
```

**Output:** Line plot with individual request scatter, stat trend line, and linear fit showing TTFT scaling rate in ms/token.

---

### 3. Prefix Caching Impact on TTFT

**Goal:** Measure how prefix caching affects TTFT across input lengths from 32 to 256 tokens.

**Design:** Uses `condition_groups` to run all input lengths first with prefix caching disabled, then restart the engine with prefix caching enabled and repeat. This ensures clean cache state for each group.

**Conditions:** 16 conditions across 2 groups:

| Group | Condition | Cache | Input tokens | Output tokens |
|-------|-----------|-------|-------------|---------------|
| cache_disabled | nocache_input_32 | OFF | 32 | 32 |
| cache_disabled | nocache_input_64 | OFF | 64 | 32 |
| cache_disabled | nocache_input_96 | OFF | 96 | 32 |
| cache_disabled | nocache_input_128 | OFF | 128 | 32 |
| cache_disabled | nocache_input_160 | OFF | 160 | 32 |
| cache_disabled | nocache_input_192 | OFF | 192 | 32 |
| cache_disabled | nocache_input_224 | OFF | 224 | 32 |
| cache_disabled | nocache_input_256 | OFF | 256 | 32 |
| cache_enabled | cache_input_32 | ON | 32 | 32 |
| cache_enabled | cache_input_64 | ON | 64 | 32 |
| cache_enabled | cache_input_96 | ON | 96 | 32 |
| cache_enabled | cache_input_128 | ON | 128 | 32 |
| cache_enabled | cache_input_160 | ON | 160 | 32 |
| cache_enabled | cache_input_192 | ON | 192 | 32 |
| cache_enabled | cache_input_224 | ON | 224 | 32 |
| cache_enabled | cache_input_256 | ON | 256 | 32 |

**Run:**
```bash
python3 -m benchmarks.experiment_orchestrator benchmarks/experiments/sequential/prefix_caching_ttft.yaml --restart
```

**Plot:**
```bash
python3 -m benchmarks.plotting.plot_prefix_caching_ttft --input benchmarks/results/sequential/prefix_caching_ttft/prefix_caching_ttft.json --stat mean --no-scatter
```

**Output:** Dual-line plot comparing TTFT with and without prefix caching across input lengths, with individual request scatter points.

---

### Summary table

| Experiment | YAML | Plot script | Description |
|---|---|---|---|
| Latency Composition | `latency_composition.yaml` | `plot_latency_composition` | Stacked barchart of prefill vs decode for SISO/SILO/LISO/LILO |
| TTFT vs Input Length | `ttft_vs_input_length.yaml` | `plot_ttft_vs_input_length` | Line plot of TTFT scaling from 32 to 256 input tokens |
| Prefix Caching TTFT | `prefix_caching_ttft.yaml` | `plot_prefix_caching_ttft` | Dual-line TTFT comparison with/without prefix caching |
| Decode Time vs Output Length | `decode_time_vs_output_length.yaml` | `plot_decode_time_vs_output_length` | Line plot of decode time scaling from 32 to 768 output tokens |
| Throughput vs Concurrency | `throughput_vs_concurrency.yaml` | `plot_throughput_vs_concurrency` | Line plot of output tok/s scaling across concurrency levels 4–64 |
| TTFT vs Concurrency | `ttft_vs_concurrency.yaml` | `plot_ttft_vs_concurrency` | Multi-line plot of mean/p50/p90/p99 TTFT across concurrency levels 4–64 |
| Throughput vs Latency Pareto | `throughput_latency_pareto.yaml` | `plot_throughput_latency_pareto` | Pareto frontier of throughput vs e2e latency across concurrency levels 1–64 |

### 4. Decode Time vs Output Length

**Goal:** Measure how decode time scales as output token count increases from 32 to 768 tokens (fixed 32-token input).

**Conditions:** 24 conditions (`output_32` through `output_768`), all with `input_length: 32` and `max_tokens` increasing in steps of 32.

| Condition | Input tokens | Output tokens |
|-----------|-------------|---------------|
| output_32 | 32 | 32 |
| output_64 | 32 | 64 |
| output_96 | 32 | 96 |
| ... | 32 | ... |
| output_768 | 32 | 768 |

**Run:**
```bash
python3 -m benchmarks.experiment_orchestrator benchmarks/experiments/sequential/decode_time_vs_output_length.yaml
```

**Plot:**
```bash
python3 -m benchmarks.plotting.plot_decode_time_vs_output_length --input benchmarks/results/sequential/decode_time_vs_output_length/decode_time_vs_output_length.json --stat mean
```

**Output:** Line plot with individual request scatter, stat trend line, and linear fit showing decode time scaling rate in ms/token.

---

### 5. Throughput vs Concurrency

**Goal:** Measure how aggregate output token throughput (tok/s) scales as the number of concurrent requests increases. All conditions use the same prompt size; only concurrency varies.

**Conditions:** 5 conditions with fixed input/output size and increasing concurrency:

| Condition | Input tokens | Output tokens | Concurrency |
|-----------|-------------|---------------|-------------|
| concurrency_4 | 32 | 128 | 4 |
| concurrency_8 | 32 | 128 | 8 |
| concurrency_16 | 32 | 128 | 16 |
| concurrency_32 | 32 | 128 | 32 |
| concurrency_64 | 32 | 128 | 64 |

Each condition uses a per-condition `dispatch.concurrency` override. Concurrency levels 16+ may trigger HTTP 429 retries when `max_pending` is set below the concurrency level.

**Run:**
```bash
python3 -m benchmarks.experiment_orchestrator benchmarks/experiments/concurrent/throughput_vs_concurrency.yaml --restart
```

**Plot:**
```bash
python3 -m benchmarks.plotting.plot_throughput_vs_concurrency --input benchmarks/results/concurrent/throughput_vs_concurrency/throughput_vs_concurrency.json
# Use server-side throughput instead:
python3 -m benchmarks.plotting.plot_throughput_vs_concurrency --input benchmarks/results/concurrent/throughput_vs_concurrency/throughput_vs_concurrency.json --source server
```

**Output:** Line plot of output token throughput (tok/s) vs concurrency level, with annotated data points.

---

### 6. TTFT vs Concurrency

**Goal:** Measure how Time To First Token (and its tail latency) changes as the number of concurrent requests increases. Shows mean, p50, p90, and p99 simultaneously to reveal how tail latency diverges under load.

**Conditions:** 5 conditions with fixed input/output size and increasing concurrency:

| Condition | Input tokens | Output tokens | Concurrency |
|-----------|-------------|---------------|-------------|
| concurrency_4 | 32 | 128 | 4 |
| concurrency_8 | 32 | 128 | 8 |
| concurrency_16 | 32 | 128 | 16 |
| concurrency_32 | 32 | 128 | 32 |
| concurrency_64 | 32 | 128 | 64 |

Each condition uses a per-condition `dispatch.concurrency` override.

**Run:**
```bash
python3 -m benchmarks.experiment_orchestrator benchmarks/experiments/concurrent/ttft_vs_concurrency.yaml
```

**Plot:**
```bash
python3 -m benchmarks.plotting.plot_ttft_vs_concurrency --input benchmarks/results/concurrent/ttft_vs_concurrency/ttft_vs_concurrency.json
# Hide per-request scatter points:
python3 -m benchmarks.plotting.plot_ttft_vs_concurrency --input benchmarks/results/concurrent/ttft_vs_concurrency/ttft_vs_concurrency.json --no-scatter
```

**Output:** Multi-line plot showing mean, p50, p90, and p99 TTFT (in ms) vs concurrency level, with optional per-request scatter. P99 values are annotated on the chart.

---

### 7. Throughput vs Latency Pareto Frontier

**Goal:** Identify the Pareto-optimal concurrency levels in throughput–latency space. Each concurrency level produces a (latency, throughput) point; the Pareto frontier connects points where throughput cannot be improved without increasing latency. This reveals the best concurrency level for a given latency budget.

**Conditions:** 7 conditions with fixed input/output size and increasing concurrency:

| Condition | Input tokens | Output tokens | Concurrency |
|-----------|-------------|---------------|-------------|
| n_1 | 128 | 128 | 1 |
| n_2 | 128 | 128 | 2 |
| n_4 | 128 | 128 | 4 |
| n_8 | 128 | 128 | 8 |
| n_16 | 128 | 128 | 16 |
| n_32 | 128 | 128 | 32 |
| n_64 | 128 | 128 | 64 |

Each condition uses a per-condition `dispatch.concurrency` override.

**Run:**
```bash
python3 -m benchmarks.experiment_orchestrator benchmarks/experiments/concurrent/throughput_latency_pareto.yaml
```

**Plot:**
```bash
python3 -m benchmarks.plotting.plot_throughput_latency_pareto --input benchmarks/results/concurrent/throughput_latency_pareto/throughput_latency_pareto.json --stat mean
# Use p90 latency instead:
python3 -m benchmarks.plotting.plot_throughput_latency_pareto --input benchmarks/results/concurrent/throughput_latency_pareto/throughput_latency_pareto.json --stat p90
# Use server-side throughput:
python3 -m benchmarks.plotting.plot_throughput_latency_pareto --input benchmarks/results/concurrent/throughput_latency_pareto/throughput_latency_pareto.json --source server
```

**Output:** Scatter plot with all concurrency levels labeled. Pareto-optimal points are connected by a frontier line and highlighted in bold. Dominated points appear faded.

---

## Adding a new experiment

1. Create `benchmarks/experiments/<name>.yaml` following the YAML format above.
2. Run it: `python3 -m benchmarks.experiment_orchestrator benchmarks/experiments/<name>.yaml`
3. Create `benchmarks/plotting/plot_<name>.py` that reads from `benchmarks/results/<dispatch_mode>/<name>.json`.
4. Add a row to the table above.
