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

## Step 6: Create the plotting script

Create `benchmarks/plotting/plot_<experiment_name>.py`. The script must:

1. Accept CLI args: `--input` (path to results JSON) and `--stat` (one of mean/p50/p90/p99, default mean).
2. Load the JSON results file.
3. Extract the relevant metrics from the results. Available metrics per condition:
   - `stats.ttft` — time to first token: `{mean, p50, p90, p99}`
   - `stats.decode_time` — decode phase duration: `{mean, p50, p90, p99}`
   - `stats.itl` — inter-token latency: `{mean, p50, p90, p99}`
   - `stats.e2e` — end-to-end latency: `{mean, p50, p90, p99}`
   - `server_side_delta` — `{latency, throughput, kv_cache, errors}`
   - `per_request` — array of individual request data
4. Produce the visualization the user requested (matplotlib).
5. Save the figure to `benchmarks/results/<experiment_name>.png`.

Look at existing plotting scripts in `benchmarks/plotting/` for style conventions before writing.

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
