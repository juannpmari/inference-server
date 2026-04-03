# Experiments

## Overview

Experiments are defined as YAML files in `benchmarks/experiments/`. Each experiment specifies a set of **conditions** (parameter combinations to compare), a **dispatch mode** (how requests are sent), and a **protocol** (warmup, repetitions, etc.).

A single generic runner (`run_experiment.py`) executes any experiment YAML and outputs a JSON with all collected metrics. Separate plotting scripts in `benchmarks/plotting/` consume these JSON files to produce visualizations.

## Running an experiment

```bash
python -m benchmarks.run_experiment benchmarks/experiments/<experiment>.yaml
```

Options:
- `--engine-url URL` — Engine endpoint (default: `http://localhost:8080`)
- `--output PATH` — Output JSON path (default: `benchmarks/results/<experiment_name>.json`)
- `--restart` — Restart engine between conditions

## Plotting results

Each experiment has a corresponding plotting script in `benchmarks/plotting/`:

```bash
python -m benchmarks.plotting.<plot_script> --input benchmarks/results/<experiment>.json --stat <mean|p50|p90|p99>
```

Output PNGs are saved to `resources/<experiment_name>/` by default (override with `--output-dir`).

## Experiment YAML format

```yaml
experiment:
  name: experiment_name
  description: "What this experiment measures"

  conditions:
    condition_a:
      prompt_pool: short          # which prompt pool from prompts/manifest.json
      max_tokens: 32              # max output tokens
      concat: 1                   # optional: concatenate N pool prompts into one (default: 1)
    condition_b:
      prompt_pool: long
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
```

## Output JSON format

The runner always collects the full set of metrics for every condition:

```
results/<experiment_name>.json
├── experiment: str
├── description: str
├── dispatch_mode: str
└── conditions:
    └── <condition_name>:
        ├── prompt_pool, max_tokens, total/successful/failed requests
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
| SILO (Short In, Long Out) | 32 | 256 | 288 |
| LISO (Long In, Short Out) | 256 | 32 | 288 |
| LILO (Long In, Long Out) | 256 | 256 | 512 |

Long inputs are built by concatenating 2 long-pool prompts (2 × 128 = 256 tokens).

**Run:**
```bash
python -m benchmarks.run_experiment benchmarks/experiments/latency_composition.yaml
# Use --restart to restart the engine between conditions (avoids KV cache carryover):
python -m benchmarks.run_experiment benchmarks/experiments/latency_composition.yaml --restart
```

**Plot:**
```bash
python -m benchmarks.plotting.plot_latency_composition --input benchmarks/results/latency_composition.json --stat mean
```

**Output:** Stacked barchart showing prefill vs decode time for each condition.

---

### 2. TTFT vs Input Length

**Goal:** Measure how Time To First Token scales as input length increases from 32 to 256 tokens.

**Conditions:** 8 conditions (`input_32` through `input_256`), all using the short prompt pool with increasing `concat` values (1 through 8) to build progressively longer inputs. Output is kept short (`max_tokens: 32`) so decode doesn't dominate.

| Condition | Concat | Input tokens | Output tokens |
|-----------|--------|-------------|---------------|
| input_32 | 1 | 32 | 32 |
| input_64 | 2 | 64 | 32 |
| input_96 | 3 | 96 | 32 |
| input_128 | 4 | 128 | 32 |
| input_160 | 5 | 160 | 32 |
| input_192 | 6 | 192 | 32 |
| input_224 | 7 | 224 | 32 |
| input_256 | 8 | 256 | 32 |

**Run:**
```bash
python -m benchmarks.run_experiment benchmarks/experiments/ttft_vs_input_length.yaml
```

**Plot:**
```bash
python -m benchmarks.plotting.plot_ttft_vs_input_length --input benchmarks/results/ttft_vs_input_length.json --stat mean
```

**Output:** Line plot with individual request scatter, stat trend line, and linear fit showing TTFT scaling rate in ms/token.

---

### Summary table

| Experiment | YAML | Plot script | Description |
|---|---|---|---|
| Latency Composition | `latency_composition.yaml` | `plot_latency_composition` | Stacked barchart of prefill vs decode for SISO/SILO/LISO/LILO |
| TTFT vs Input Length | `ttft_vs_input_length.yaml` | `plot_ttft_vs_input_length` | Line plot of TTFT scaling from 32 to 256 input tokens |

## Adding a new experiment

1. Create `benchmarks/experiments/<name>.yaml` following the YAML format above.
2. Run it: `python -m benchmarks.run_experiment benchmarks/experiments/<name>.yaml`
3. Create `benchmarks/plotting/plot_<name>.py` that reads from `benchmarks/results/<name>.json`.
4. Add a row to the table above.
