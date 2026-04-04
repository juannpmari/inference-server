You are guiding the user through creating a new benchmark experiment. Follow these steps in order.

## Step 1: Understand the goal

Ask the user: **What do you want to measure?** Get a clear description of the hypothesis or performance question. Also ask what kind of visualization they want (bar chart, line plot, heatmap, scatter, etc.).

## Step 2: Design conditions

Based on the user's goal, propose a set of experiment conditions. Each condition needs:

- **input_length**: desired input length in tokens (must be divisible by 32). The runner concatenates 32-token prompts from `benchmarks/prompts/data/` to reach this length.
- **max_tokens**: max output tokens to generate.

If the experiment requires toggling engine settings (e.g. prefix caching), use `condition_groups` with `engine_env` instead of flat `conditions`. See `benchmarks/experiments/prefix_caching_ttft.yaml` for an example.

Present the proposed conditions and confirm with the user before proceeding.

## Step 3: Choose dispatch mode

Ask which dispatch mode fits the experiment:

| Mode | Fields | Use when |
|------|--------|----------|
| `sequential` | (none extra) | Measuring per-request latency without interference |
| `concurrent` | `concurrency: N` | Testing server behavior under parallel load |
| `realistic` | `rps: X`, `duration_seconds: Y` | Simulating real traffic patterns |

## Step 4: Set protocol parameters

**Think carefully about which protocol parameters actually matter for this experiment.** Not every parameter applies to every experiment — including an irrelevant parameter adds noise to the YAML and misleads future readers into thinking it was a deliberate choice.

For each parameter below, decide whether it is relevant given the experiment's dispatch mode, independent variable, and what is being measured. **Only include parameters that meaningfully affect the experiment. Omit any that don't apply — the runner has sensible defaults.**

Guidelines:
- `prompts_per_condition`: Controls how many unique prompts are loaded. In concurrent mode, the dispatcher builds batches of size `concurrency` via round-robin over the prompt pool — so `prompts_per_condition` does NOT control how many requests are sent. Only include it if prompt diversity matters (e.g. avoiding prefix cache hits). If the experiment doesn't care about prompt content, omit it and let the runner load all available prompts by default.
- `runs`: Number of full repetitions (median is selected). Useful for reducing noise in latency measurements. Less important when the metric is aggregate throughput over many concurrent requests.
- `warmup_requests`: Almost always applicable — include unless there's a reason not to.
- `cooldown_seconds`: Relevant when running multiple conditions sequentially to avoid interference. Omit for single-condition experiments.

Propose only the applicable parameters with justification, and confirm with the user.

## Step 5: Create the experiment YAML

Write the file to `benchmarks/experiments/<experiment_name>.yaml` using this template:

```yaml
experiment:
  name: <experiment_name>
  description: "<user's description>"

  conditions:
    <condition_name>:
      input_length: <int>         # must be divisible by 32
      max_tokens: <int>

  dispatch:
    mode: sequential | concurrent | realistic
    # concurrency: <int>       # concurrent only
    # rps: <float>             # realistic only
    # duration_seconds: <int>  # realistic only

  protocol:
    warmup_requests: 3
    runs: 3
    prompts_per_condition: 10
    cooldown_seconds: 5
```

## Step 6: Define plots in the YAML

Every experiment YAML must include a `plots` key that declaratively describes the visualizations to produce from the results. This allows a generic plotter to render all plots without per-experiment scripts.

### Plot definition protocol

```yaml
  plots:
    <plot_name>:
      description: "Human-readable description of the plot"
      x: <field>                    # x-axis data source (dot-path into condition config or results)
      y: <field or list>            # y-axis data source(s)
      series: [...]                 # optional: multiple curves/bars
      style: <style>                # optional: rendering hint (default: line)
```

### Field references

`x` and `y` fields use dot-notation paths that resolve against either:
- **Condition config**: `input_length`, `max_tokens`, `dispatch.concurrency` — values from the YAML condition definition itself.
- **Result metrics**: `stats.ttft.mean`, `stats.decode_time.p90`, `client_side_throughput.output_tokens_per_second`, etc.

Available result metrics per condition:
- `stats.ttft` — time to first token: `{mean, p50, p90, p99}`
- `stats.decode_time` — decode phase duration: `{mean, p50, p90, p99}`
- `stats.itl` — inter-token latency: `{mean, p50, p90, p99}`
- `stats.e2e` — end-to-end latency: `{mean, p50, p90, p99}`
- `client_side_throughput.output_tokens_per_second`
- `server_side_delta` — `{latency, throughput, kv_cache, errors}`

### Single y-field (one series)

When `y` is a string, there's one data series. Use `series` to give it a label:

```yaml
    decode_time_vs_output_length:
      x: max_tokens
      y: stats.decode_time.mean
      series:
        - label: "Decode time"
```

### Multiple y-fields (multiple series from different metrics)

When plotting several metrics on the same axes (e.g. percentile breakdown), `y` is a list:

```yaml
    ttft_vs_concurrency:
      x: dispatch.concurrency
      y:
        - label: "Mean"
          field: stats.ttft.mean
        - label: "P50"
          field: stats.ttft.p50
        - label: "P90"
          field: stats.ttft.p90
        - label: "P99"
          field: stats.ttft.p99
```

### Multiple series from condition groups

When the experiment uses `condition_groups` and each group becomes a separate curve, reference it via `condition_group`:

```yaml
    ttft_vs_input_length:
      x: input_length
      y: stats.ttft.mean
      series:
        - label: "Cache disabled"
          condition_group: cache_disabled
        - label: "Cache enabled"
          condition_group: cache_enabled
```

### Style hints

Use `style` to indicate the plot type. The plotter uses this to select the rendering strategy:

| Style | Use when |
|-------|----------|
| `line` (default) | Continuous metric vs ordered x-axis |
| `stacked_bar` | Decomposing a total into parts (e.g. TTFT + decode = e2e) |
| `scatter_pareto` | Scatter with Pareto frontier overlay |

Example for stacked bar:

```yaml
    latency_composition:
      x: condition
      y:
        - label: "Prefill (TTFT)"
          field: stats.ttft.mean
        - label: "Decode"
          field: stats.decode_time.mean
      style: stacked_bar
```

### Existing experiments as reference

- `concurrent/concurrency_sweep.yaml` — 3 plots: line, multi-series percentile, scatter Pareto
- `sequential/input_length_sweep.yaml` — condition_group-based series (cache on/off)
- `sequential/latency_composition.yaml` — stacked bar
- `sequential/decode_time_vs_output_length.yaml` — simple single-series line

## Step 7: Update documentation

**This step is mandatory. Do not skip it.**

Read `benchmarks/experiments.md` and update it with all of the following:

1. **Add a detailed section** for the new experiment, following the same format as the existing experiment sections. The section must include:
   - **Goal**: what the experiment measures.
   - **Conditions table**: listing every condition with its `input_length` and `max_tokens` values.
   - **Run command**: the exact `python3 -m benchmarks.experiment_orchestrator` invocation.
   - **Plot command**: the exact `python3 -m benchmarks.plotting.plot_<name>` invocation.
   - **Output**: what the plot shows.

2. **Add a row to the summary table** at the bottom of the file with the experiment name, YAML filename, plot script name, and short description.

3. **Verify consistency**: ensure the values in the documentation match the YAML exactly (token counts, condition names, number of conditions). Do not use approximate or rounded values.

## Step 8: Summary

Print a summary showing:
- The YAML path and how to run: `python3 -m benchmarks.experiment_orchestrator benchmarks/experiments/<name>.yaml`
- The plotting script path and how to plot: `python3 -m benchmarks.plotting.plot_<name> --input benchmarks/results/<name>.json --stat mean`
- Any caveats (input_length constraints, prompt pool sizes, etc.)
