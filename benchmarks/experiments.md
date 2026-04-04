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

### Summary table

| # | Experiment | YAML | Plotter subcommand | Description |
|---|---|---|---|---|
| 1 | Latency Composition | `sequential/latency_composition.yaml` | `sequential_plotter latency` | Stacked barchart of prefill vs decode for SISO/SILO/LISO/LILO |
| 2 | Input Length Sweep | `sequential/input_length_sweep.yaml` | `sequential_plotter ttft-input` / `prefix-cache` | TTFT scaling across input lengths, with and without prefix caching |
| 3 | Decode Time vs Output Length | `sequential/decode_time_vs_output_length.yaml` | `sequential_plotter decode` | Decode time scaling from 3840 to 12960 output tokens |
| 4 | Concurrency Sweep | `concurrent/concurrency_sweep.yaml` | `concurrency_plotter throughput` / `ttft` / `pareto` | Throughput, TTFT, and Pareto plots from a single concurrency sweep |
| 5 | Arrival Rate Sweep | `realistic/arrival_rate_sweep.yaml` | `realistic_plotter throughput` / `queue-depth` / `ttft` | Throughput, queue depth, and TTFT under increasing Poisson arrival rates |

---

### 1. Latency Composition

**Goal:** Decompose end-to-end latency into prefill (TTFT) and decode phases across four input/output size combinations.

**Conditions:**

| Condition | Input tokens | Output tokens |
|-----------|-------------|---------------|
| SISO (Short In, Short Out) | 32 | 32 |
| SILO (Short In, Long Out) | 32 | 320 |
| LISO (Long In, Short Out) | 320 | 32 |
| LILO (Long In, Long Out) | 320 | 320 |

**Run:**
```bash
python3 -m benchmarks.experiment_orchestrator benchmarks/experiments/sequential/latency_composition.yaml
# Use --restart to restart the engine between conditions (avoids KV cache carryover):
python3 -m benchmarks.experiment_orchestrator benchmarks/experiments/sequential/latency_composition.yaml --restart
```

**Plot (optional):**
```bash
python3 -m benchmarks.plotting.sequential_plotter latency --stat mean
# Custom input/output paths:
python3 -m benchmarks.plotting.sequential_plotter latency --input benchmarks/results/sequential/latency_composition/latency_composition.json --stat p50
```

**Output:** Stacked barchart showing prefill vs decode time for each condition.

---

### 2. Input Length Sweep

**Goal:** Measure how TTFT scales as input length increases from 3840 to 12960 tokens, with and without prefix caching.

**Design:** Uses `condition_groups` to run all input lengths first with prefix caching disabled, then restart the engine with prefix caching enabled and repeat. This ensures clean cache state for each group.

**Conditions:** 8 conditions across 2 groups:

| Group | Condition | Cache | Input tokens | Output tokens |
|-------|-----------|-------|-------------|---------------|
| cache_disabled | nocache_input_3840 | OFF | 3840 | 32 |
| cache_disabled | nocache_input_5760 | OFF | 5760 | 32 |
| cache_disabled | nocache_input_8640 | OFF | 8640 | 32 |
| cache_disabled | nocache_input_12960 | OFF | 12960 | 32 |
| cache_enabled | cache_input_3840 | ON | 3840 | 32 |
| cache_enabled | cache_input_5760 | ON | 5760 | 32 |
| cache_enabled | cache_input_8640 | ON | 8640 | 32 |
| cache_enabled | cache_input_12960 | ON | 12960 | 32 |

**Run:**
```bash
python3 -m benchmarks.experiment_orchestrator benchmarks/experiments/sequential/input_length_sweep.yaml --restart
```

**Plot (optional) — TTFT vs input length (cache-disabled group only):**
```bash
python3 -m benchmarks.plotting.sequential_plotter ttft-input --stat p90
python3 -m benchmarks.plotting.sequential_plotter ttft-input --stat mean --no-scatter
```

**Plot (optional) — prefix caching comparison (both groups):**
```bash
python3 -m benchmarks.plotting.sequential_plotter prefix-cache --stat mean
python3 -m benchmarks.plotting.sequential_plotter prefix-cache --stat mean --no-scatter
```


**Output:** Line plot of TTFT vs input length with linear fit, or dual-line comparison with/without prefix caching.

---

### 3. Decode Time vs Output Length

**Goal:** Measure how decode time scales as output token count increases from 3840 to 12960 tokens (fixed 32-token input).

**Conditions:** 4 conditions with `input_length: 32` and increasing `max_tokens`:

| Condition | Input tokens | Output tokens |
|-----------|-------------|---------------|
| output_3840 | 32 | 3840 |
| output_5760 | 32 | 5760 |
| output_8640 | 32 | 8640 |
| output_12960 | 32 | 12960 |

**Run:**
```bash
python3 -m benchmarks.experiment_orchestrator benchmarks/experiments/sequential/decode_time_vs_output_length.yaml
```

**Plot (optional):**
```bash
python3 -m benchmarks.plotting.sequential_plotter decode --stat mean
python3 -m benchmarks.plotting.sequential_plotter decode --stat p90 --no-scatter
```

**Output:** Line plot with individual request scatter, stat trend line, and linear fit showing decode time scaling rate in ms/token.

---

### 4. Concurrency Sweep

**Goal:** Sweep concurrency levels to measure throughput scaling, TTFT degradation, and the throughput–latency Pareto frontier — all from a single experiment run.

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
python3 -m benchmarks.experiment_orchestrator benchmarks/experiments/concurrent/concurrency_sweep.yaml
# Use --restart to restart the engine between conditions:
python3 -m benchmarks.experiment_orchestrator benchmarks/experiments/concurrent/concurrency_sweep.yaml --restart
```

**Plot (optional) — throughput vs concurrency:**
```bash
python3 -m benchmarks.plotting.concurrency_plotter throughput
# Use server-side throughput:
python3 -m benchmarks.plotting.concurrency_plotter throughput --source server
```

**Plot (optional) — TTFT vs concurrency:**
```bash
python3 -m benchmarks.plotting.concurrency_plotter ttft
# Hide per-request scatter points:
python3 -m benchmarks.plotting.concurrency_plotter ttft --no-scatter
```

**Plot (optional) — throughput-latency Pareto frontier:**
```bash
python3 -m benchmarks.plotting.concurrency_plotter pareto --stat mean
# Use p90 latency:
python3 -m benchmarks.plotting.concurrency_plotter pareto --stat p90
# Use server-side throughput:
python3 -m benchmarks.plotting.concurrency_plotter pareto --source server
```

**Output:** Throughput line plot, multi-stat TTFT line plot, or Pareto scatter with frontier line — depending on subcommand.

---

### 5. Arrival Rate Sweep

**Goal:** Measure throughput, queue depth, and TTFT under increasing Poisson arrival rates with cache on/off.

**Conditions:** 20 per group (lambda 0.5–10.0), with `input_length: 1920` and `max_tokens: 128`.

**Run:**
```bash
python3 -m benchmarks.experiment_orchestrator benchmarks/experiments/realistic/arrival_rate_sweep.yaml
```

**Plot (optional) — throughput vs arrival rate:**
```bash
python3 -m benchmarks.plotting.realistic_plotter throughput
```

**Plot (optional) — queue depth vs arrival rate:**
```bash
python3 -m benchmarks.plotting.realistic_plotter queue-depth
```

**Plot (optional) — TTFT percentiles vs arrival rate:**
```bash
python3 -m benchmarks.plotting.realistic_plotter ttft
```

**Output:** 3 plots — throughput vs arrival rate, queue depth vs arrival rate, TTFT percentiles vs arrival rate.

---

## Adding a new experiment

1. Create `benchmarks/experiments/<dispatch_mode>/<name>.yaml` following the YAML format above.
2. Run it: `python3 -m benchmarks.experiment_orchestrator benchmarks/experiments/<dispatch_mode>/<name>.yaml`
3. Add a plot method to the appropriate plotter (`sequential_plotter.py` or `concurrency_plotter.py`) and register it in `PLOT_REGISTRY`.
4. Add a row to the summary table above.
