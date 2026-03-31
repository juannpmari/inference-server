## Hypothesis
vLLM's automatic prefix caching (APC) reuses KV blocks across requests as long as they remain in GPU memory. Under low concurrency and short contexts, this should be sufficient — hit rate will be high and prefill savings real. But as concurrent requests increase and GPU memory fills, APC blocks get evicted and subsequent requests recompute from scratch, collapsing hit rate under pressure. LMCache should degrade more gracefully: by explicitly persisting KV blocks to CPU memory, evicted blocks remain retrievable at the cost of a PCIe transfer rather than a full recompute. The hypothesis is that under sufficient memory pressure — enough concurrent requests with long shared prefixes to force eviction — LMCache maintains a meaningfully higher cache hit ratio and lower average prefill time than APC alone, and that this gap widens as concurrency increases. Under low memory pressure, we expect both systems to perform similarly, with LMCache adding overhead for little gain.

## Prompts
For your hypothesis to show clearly, you need requests that maximize shared prefix reuse while also stressing GPU memory. The juiciest setup is:
Long shared system prompt + RAG context + short varied question
Something like:

2048 token fixed prefix (system prompt + 3-4 retrieved documents (invented by you), same every request)
50-100 token varied suffix (different user question each time)
Short generation (64-128 tokens)

This is juicy because:

The prefix is long enough to actually fill GPU memory under concurrency
The suffix variation means requests aren't identical, so naive caching wouldn't help — but prefix caching does
Short generation keeps the experiment fast to run without losing the interesting signal
It directly mirrors a real RAG workload, which makes the post relevant beyond just a benchmark

Why this hurts APC specifically:
Under concurrency, each request brings 2048 tokens of KV blocks into GPU. With 16+ concurrent requests that's 32k+ tokens of KV state competing for the same GPU memory pool. APC evicts older blocks to make room, so request 10 recomputes what request 1 already computed. LMCache hits from CPU instead.
What to vary across requests:
Keep the system prompt and retrieved documents identical. Only change the final user question. You can generate 50-100 different questions about the same documen

## Metrics
Given this hypothesis specifically, your metrics should map directly to the two conditions you're testing — low pressure and high pressure:
Primary metric: cache hit ratio vs concurrency level Increase the number of concurrent requests incrementally and measure hit rate for both systems at each level. This is the direct test of your hypothesis — you expect the lines to diverge as concurrency grows and GPU fills up. This is your main result.
Secondary metric: prefill time vs concurrency level Same sweep, but measuring average prefill time per request. This converts the hit ratio story into a compute cost story — when APC starts missing and recomputing, prefill time climbs. When LMCache hits from CPU, it stays lower. The gap between the two lines under high concurrency is your "worth it" argument.
Supporting metric: GPU memory utilization during the sweep torch.cuda.memory_allocated() sampled continuously. This is what lets you pinpoint the crossover — the concurrency level where GPU fills up, APC starts evicting, and the two systems diverge. Without this you can't explain why the lines diverge, only that they do.
Those three together tell a complete story: GPU fills up at concurrency level N → APC hit rate drops → LMCache hit rate holds → prefill time reflects that difference.

## Plot specification summary
The visualization consists of two vertically stacked panels sharing the same x-axis, which represents the number of concurrent requests increasing from 1 to 32. The top panel has two y-axes: the left shows GPU memory utilization as a filled area curve, and the right shows cache hit rate as two separate lines, one for vLLM APC and one for LMCache. A horizontal dashed line marks the eviction threshold (85%) on the left axis, and a faint horizontal line marks the 80% hit rate floor on the right axis. The key visual story in this panel is that both hit rate lines track together until GPU memory crosses the eviction threshold, at which point the APC line drops sharply while LMCache degrades gradually. A vertical shaded region begins at the concurrency level where the threshold is crossed and extends to the right — this region should appear in both panels. The bottom panel shows average prefill time in milliseconds for both systems on a single y-axis. Before the shaded region both lines are nearly identical, confirming that LMCache adds no meaningful overhead at low pressure. Inside the shaded region the APC line steepens significantly while LMCache grows at a slower rate. A double-headed arrow annotation at peak concurrency labels the millisecond gap between the two systems — this is the headline number the post is built around.
