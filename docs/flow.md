# Cold Start (Bootstrap Phase)

## Gateway 

1. Pod scheduled and started by Kubernetes — The gateway Deployment is reconciled, the Pod is scheduled, and the FastAPI application initializes.
2. Routing logic loads — The gateway loads its internal routing algorithms and configuration, but its backend pool is initially empty.
3. Cluster discovery begins — The gateway connects to the Kubernetes API and starts watching Services / EndpointSlices for inference backends.
4. Inference Pods become Ready — As inference Pods finish model loading and pass readiness probes, Kubernetes adds them to the Service endpoints (Kubernetes API).
5. Backend registry populated — The gateway receives watch events and updates its internal list of routable Pods.
6. Gateway becomes fully operational — Once at least one healthy backend exists, incoming requests can be dynamically routed according to the configured strategy.

## Control plane

### Model Metadata Store

Contains the model CRDs, providing the desired configuration. It defines: "This model should exist and have N replicas."
For example, a model CRD might specify:

```yaml
model: model-X
replicas: 1
engine: vLLM
weights_uri: s3://...
gpu: A100
```

### Model metadata controller

Watches the current state of the deployments and compares it with the desired state defined in the model CRDs. If there's a mismatch, it updates the deployments accordingly.

### Custom EndpointSlice
The routing map that tells the gateway which Pods are currently healthy and eligible to receive traffic. After an inference pod becomes ready to serve requests, the EndpointSlice controller updates the EndpointSlice to include the pod's IP and port.


## Engine + Sidecar Cold Start (Replica-Driven Bootstrap)

There's a tradeoff: if the server supports many models, they cannot be loaded all at once due to GPU memory constraints. Therefore, the server should only load the models that are actually needed. This is defined by the user in the model CRD. Some models will have replicas = 0, meaning they are not loaded yet.

When a model specifies replicas = N (with minReplicas possibly 0 or >0), the controller creates a Kubernetes Deployment with that desired replica count. Kubernetes schedules each Pod, and inside every Pod the engine container (e.g., vLLM) and the sidecar container start in parallel. The model identity and configuration are injected into the Pod via environment variables, arguments, or mounted configs derived from the CRD.

At startup, the engine initializes GPU context, allocators, and runtime structures. Meanwhile, the sidecar downloads the model from the external storage (e.g. S3) into a path known by the engine. Then, the engine loads the model weights into GPU memory. The sidecar monitors the engine's health and aggregates readiness status. While the model is loading, the Pod remains NotReady.

Once the engine successfully loads the model and its health endpoint responds positively, the readiness probe defined in the Pod spec passes. Kubernetes then marks the Pod as Ready in the Kubernetes API server and adds it to the Service endpoints. From that moment, the Pod is considered eligible to receive traffic, and the gateway can route requests to it.

### LoRA Adapters
While the base model is loaded during Pod cold start according to the configured replica count, LoRA adapters follow a different lifecycle. LoRAs are typically not preloaded at Pod startup; instead, they are fetched by the sidecar when a request references a specific adapter. Then the engine retrieves the adapter weights from the shared volume prepared by the sidecar, loads them into runtime structures, and may move them into GPU memory. Subsequent requests reuse the cached adapter until it is evicted based on internal memory policies. Kubernetes does not track LoRA state; adapter loading, caching, and eviction are fully managed inside the engine process.

## Distributed KV cache

# Steady state

## Gateway
The gateway is responsible for routing requests to the appropriate engine pods. It maintains a list of available pods and routes requests based on:
- model name + adapter name (if applicable)
- Pod Readiness / Load State: Routing only considers pods marked healthy and accepting traffic
- Load-aware routing (Throughput and latency control):
  - Current batch size
  - In-flight requests
  - GPU memory pressure
  - Queue length
- Prefix aware routing: If a pod already has a matching prompt prefix in its KV cache,

### Load aware routing
The inference container exposes a /metrics endpoint with the following metrics that the gateway monitors:

- In-flight requests per pod:
  - Active requests
  - Tokens being generated
  - Batch size

- Internal engine queue depth:
  - Waiting requests
  - Backpressure signal

- GPU memory pressure, Indirect metrics:
  - Free memory %
  - Fragmentation
  - KV block usage

- Throughput stats
  - Tokens/sec
  - Average latency
  - Tail latency (p95/p99)

### Prefix-aware routing
It tries to answer: "Which pod already has the KV cache entries for this prompt prefix?"
TODO: Who exposes it?


## Engine + sidecar

### Cold start manager
Imagine we have N podes serving models. We now receive a request for a new model, which doesn't exist in any pod yet.

1. The routing layer checks the custom EndpointSlice and finds no ready endpoints for that model, meaning there is no routable backend.
2. This model-level miss triggers the Cold Start Manager, which queries the model metadata (weights URI, GPU requirements, replica policy, etc.) to determine how the model should be instantiated.
3. The Cold Start Manager instructs the controller to create a new Deployment or scale the model from zero replicas to one.
4. Kubernetes schedules a new GPU-backed pod for that model.
5. Inside the pod, the sidecar downloads the model weights from object storage, and the inference engine (e.g., vLLM) loads the weights into GPU memory and initializes the runtime.
6. Once the pod passes readiness checks, the controller updates the custom EndpointSlice to include the pod IP for that model.
7. The model now has a ready endpoint, the gateway can route traffic to it, and the system returns to steady state.

## Distributed KV cache

### Eviction policy

# Autoscaling
Kubernetes has a traditional HPA (Horizontal Pod Autoscaler) that can scale pods based on CPU and memory usage. However, for inference workloads, we need a more sophisticated autoscaler that can scale pods based on request queue length, model load states, KV cache hit ratios, and token throughput.

AIBrix has an LLM-specific autoscalr: A separate controller (control-plane service), which is a piece of code that watches LLM-specific metrics (exposed by the inference pod) and adjusts replica counts accordingly.

## Monitored metrics
They are exposed by the inference engine, through Prometheus.

```
Pods → expose engine metrics
        ↓
Metrics backend (Prometheus or internal API)
        ↓
LLM Autoscaler Controller
        ↓
Update Model CR / Deployment replicas
        ↓
Kubernetes reconciles
```

## Request Queue Length
- Pending requests
- Waiting tokens
- Backpressure state

If queue length grows:
    Demand > current serving capacity
then scale up.

This is more accurate than CPU because LLM pods can be CPU-light but latency-heavy due to long contexts.

## Model Load State

This matters because:

- Scaling from 0 has cold-start cost
- Some pods may still be loading weights
- Some replicas may be warming

The autoscaler may:
- Avoid scaling down if pods are still warming
- Pre-scale before load state transitions
- Maintain a minimum warm replica count

## KV Cache Hit Ratio

If KV hit ratio drops:

- Prefix locality is lost
- TTFT increases
- More compute per request

Autoscaler might:
- Scale up to reduce cache pressure
- Avoid eviction cascades
- Keep more replicas warm to preserve locality

This is something HPA would never consider.

## Token Throughput

Measured as:

- Tokens/sec per pod
- Aggregate cluster throughput
- Target token rate

If pods are saturating token generation capacity, scale up.

This is the most direct measure of LLM serving load.
