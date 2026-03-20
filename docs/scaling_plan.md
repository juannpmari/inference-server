# Kubernetes Scaling Plan — LLM Inference Server

## Current State Summary

The system today runs via docker-compose with three data-plane services (gateway, engine, sidecar) and fully stub control-plane components. Key facts:

- **Gateway** (`data_plane/gateway/routing.py`): Hardcoded `MODEL_SERVICE_MAP` dict, single-backend routing, no K8s awareness.
- **Engine** (`data_plane/inference/engine/api.py`): vLLM wrapper with `/inference`, `/health`, `/ready`, `/metrics`. 11 Prometheus metrics defined (3 actively recorded). Supports continuous batching and LoRA hot-loading.
- **Sidecar** (`data_plane/inference/sidecar/api.py`): ArtifactManager downloads models from HuggingFace, persists registry to `/mnt/models/registry.json`. Exposes `/load/{model}`, `/registry/models`, `/adapter/fetch/{adapter}`.
- **Control Plane** (`control_plane/`): `autoscaler.py`, `admission_controller.py`, `lora_manager.py` are single-comment stubs.
- **KV Cache**: L1 (CPU DRAM) and L2 (Redis connector) are scaffolded but NOT integrated into the serving path.
- **K8s Manifests** (`manifests/`, `kubernetes/crds/`): All empty stubs except `05-inference-service-deploy.yaml` which has a bare Pod skeleton.

```
                      Current Architecture (docker-compose)
                      ====================================

  Client
    |
    | POST /generate {model, prompt, ...}
    v
+-----------+     POST /inference     +------------+     /adapter/fetch    +----------+
|  Gateway  |------------------------>|   Engine    |<-------------------->|  Sidecar  |
|  :8000    |                         |   :8080     |                      |  :8001    |
+-----------+                         +------+------+                      +-----+----+
                                             |        shared volume              |
                                             +----------- /mnt/models -----------+

+-----------+
| Prometheus|  scrapes /metrics from all three
|  :9090    |
+-----------+

+-----------+
|  Redis    |  L2 cache backend (not yet integrated)
|  :6379    |
+-----------+
```

---

## Target Architecture (Kubernetes)

```
                              Target Architecture
                              ===================

  Client
    |
    v
+-----------+   watches EndpointSlices    +-------------------+
|  Gateway  |<----------------------------|  Endpoint         |
|  (N reps) |   + load ConfigMaps         |  Controller       |
+-----+-----+                             +-------------------+
      |                                           |
      | direct pod IP routing                     | scrapes /metrics
      |                                           |
      v                                           v
+---Pod (model-A)----+  +---Pod (model-A)----+  +---Pod (model-B)----+
| +-------+ +------+ |  | +-------+ +------+ |  | +-------+ +------+ |
| |Engine | |Sidecar| |  | |Engine | |Sidecar| |  | |Engine | |Sidecar| |
| |:8080  | |:8001  | |  | |:8080  | |:8001  | |  | |:8080  | |:8001  | |
| +---+---+ +--+---+  |  | +---+---+ +--+---+  |  | +---+---+ +--+---+  |
|     |  /mnt/models|  |  |     |  /mnt/models|  |  |     |  /mnt/models|  |
+-----+-------------+  +-----+-------------+  +-----+-------------+

+-----------------------+   +-------------------+   +-------------------+
| InferenceModel CRDs  |-->| Model Controller  |-->| Deployments       |
| (desired state)       |   | (kopf)            |   | (actual state)    |
+-----------------------+   +-------------------+   +-------------------+

+-------------------+
| LLM Autoscaler    |  queries Prometheus, patches CRD replicas
+-------------------+
```

---

## Phase 1: Pod Architecture and Kubernetes Manifests

### Overview

Define the fundamental Kubernetes resource topology. Each inference workload runs as a Pod with two containers (engine + sidecar) sharing an `emptyDir` volume. Model identity is injected via environment variables derived from a Deployment template. One Deployment per base model. A headless Service exposes each model's pods for gateway discovery.

### Pod Spec Design

The Pod spec mirrors the current docker-compose relationship between engine and sidecar. Both containers share a volume at `/mnt/models` and the sidecar downloads model weights before the engine loads them.

```yaml
# manifests/05-inference-pod-template.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-{{ MODEL_SLUG }}
  namespace: inference
  labels:
    app.kubernetes.io/component: inference
    inference.server/model: "{{ MODEL_NAME }}"
spec:
  replicas: 1
  selector:
    matchLabels:
      app.kubernetes.io/component: inference
      inference.server/model: "{{ MODEL_NAME }}"
  template:
    metadata:
      labels:
        app.kubernetes.io/component: inference
        inference.server/model: "{{ MODEL_NAME }}"
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: inference-pod
      terminationGracePeriodSeconds: 60
      containers:
        # --- Engine container ---
        - name: engine
          image: inference-server/engine:latest
          ports:
            - name: http
              containerPort: 8080
              protocol: TCP
          env:
            - name: ENGINE_MODEL_NAME
              value: "{{ MODEL_NAME }}"
            - name: ENGINE_SIDECAR_URL
              value: "http://localhost:8001"
            - name: ENGINE_HOST
              value: "0.0.0.0"
            - name: ENGINE_PORT
              value: "8080"
            - name: ENGINE_MAX_PENDING
              value: "10"
            - name: ENGINE_ENABLE_LORA
              value: "{{ ENABLE_LORA }}"
            - name: POD_NAME
              valueFrom:
                fieldRef:
                  fieldPath: metadata.name
            - name: POD_IP
              valueFrom:
                fieldRef:
                  fieldPath: status.podIP
          resources:
            requests:
              cpu: "2"
              memory: "8Gi"
              nvidia.com/gpu: "1"
            limits:
              cpu: "4"
              memory: "16Gi"
              nvidia.com/gpu: "1"
          readinessProbe:
            httpGet:
              path: /ready
              port: 8080
            initialDelaySeconds: 30
            periodSeconds: 10
            failureThreshold: 30     # allow up to 5 min for model loading
            timeoutSeconds: 5
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 60
            periodSeconds: 15
            failureThreshold: 3
            timeoutSeconds: 5
          volumeMounts:
            - name: model-store
              mountPath: /mnt/models

        # --- Sidecar container ---
        - name: sidecar
          image: inference-server/sidecar:latest
          ports:
            - name: http
              containerPort: 8001
              protocol: TCP
          env:
            - name: SIDECAR_INITIAL_MODEL
              value: "{{ MODEL_NAME }}"
            - name: SIDECAR_SHARED_VOLUME
              value: "/mnt/models"
            - name: SIDECAR_HOST
              value: "0.0.0.0"
            - name: SIDECAR_PORT
              value: "8001"
          resources:
            requests:
              cpu: "500m"
              memory: "2Gi"
            limits:
              cpu: "1"
              memory: "4Gi"
          readinessProbe:
            httpGet:
              path: /ready
              port: 8001
            initialDelaySeconds: 10
            periodSeconds: 5
            failureThreshold: 60    # sidecar ready = model downloaded
          livenessProbe:
            httpGet:
              path: /health
              port: 8001
            initialDelaySeconds: 5
            periodSeconds: 10
          volumeMounts:
            - name: model-store
              mountPath: /mnt/models

      volumes:
        - name: model-store
          emptyDir:
            sizeLimit: 50Gi   # adjust per model size
```

### Service Definition

A headless Service enables discovery of individual pod IPs (needed for the custom EndpointSlice controller in Phase 3 and for direct gateway routing).

```yaml
# manifests/06-inference-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: inference-{{ MODEL_SLUG }}
  namespace: inference
  labels:
    app.kubernetes.io/component: inference
    inference.server/model: "{{ MODEL_NAME }}"
spec:
  clusterIP: None           # headless: no load balancing, just DNS records
  selector:
    app.kubernetes.io/component: inference
    inference.server/model: "{{ MODEL_NAME }}"
  ports:
    - name: engine
      port: 8080
      targetPort: 8080
    - name: sidecar
      port: 8001
      targetPort: 8001
```

### Namespace Strategy

Use a dedicated `inference` namespace for inference pods and a separate `inference-system` namespace for control-plane controllers.

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: inference
---
apiVersion: v1
kind: Namespace
metadata:
  name: inference-system
```

### Key Design Decisions

1. **emptyDir vs PVC for model store**: Use `emptyDir` by default. Models are downloaded from HuggingFace on each pod start (the sidecar already handles this). PVCs would share models across pod restarts but add complexity with ReadWriteOnce constraints on GPU nodes. For large models (70B+), consider ReadOnlyMany PVCs pre-populated by a download Job.

2. **One Deployment per model**: This allows independent scaling per model. The alternative (a generic Deployment pool with dynamic model assignment) is more complex and deferred to a future iteration.

3. **Readiness probe with high failureThreshold**: Model download + GPU loading can take 5+ minutes for large models. The `failureThreshold: 30` with `periodSeconds: 10` gives up to 5 minutes. The existing `/ready` endpoint on both engine and sidecar returns 503 until the model is loaded.

4. **GPU resource requests**: Uses the `nvidia.com/gpu` extended resource. Requires the NVIDIA device plugin DaemonSet in the cluster. One GPU per pod.

### Deliverables

- [ ] Pod spec template with engine + sidecar containers, shared emptyDir volume
- [ ] Headless Service per model
- [ ] Namespace manifests (inference, inference-system)
- [ ] NVIDIA device plugin DaemonSet reference
- [ ] Readiness/liveness probes wired to existing `/ready` and `/health` endpoints
- [ ] Documentation of env var injection for model identity

### Dependencies

- Working Dockerfiles for engine and sidecar (exist at `docker/inference.dockerfile` and `docker/sidecar.dockerfile`)
- A Kubernetes cluster with GPU nodes and NVIDIA device plugin

---

## Phase 2: InferenceModel CRD and Controller

### Overview

Define a Custom Resource Definition (CRD) called `InferenceModel` that declaratively specifies which models should be served, their resource requirements, and scaling parameters. A Python-based controller (using kopf) watches these CRDs and reconciles Deployments to match the desired state.

### InferenceModel CRD

```yaml
# kubernetes/crds/InferenceModel.yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: inferencemodels.inference.server
spec:
  group: inference.server
  versions:
    - name: v1alpha1
      served: true
      storage: true
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              required: [modelName, weightsURI]
              properties:
                modelName:
                  type: string
                  description: "HuggingFace model ID or S3 URI"
                modelVersion:
                  type: string
                  default: "main"
                engineType:
                  type: string
                  enum: [vllm, mock]
                  default: vllm
                gpu:
                  type: object
                  properties:
                    type:
                      type: string
                      description: "GPU type label (e.g., A100, L4, T4)"
                    count:
                      type: integer
                      minimum: 1
                      default: 1
                    memoryUtilization:
                      type: number
                      minimum: 0.1
                      maximum: 0.95
                      default: 0.7
                replicas:
                  type: integer
                  minimum: 0
                  default: 1
                minReplicas:
                  type: integer
                  minimum: 0
                  default: 0
                maxReplicas:
                  type: integer
                  minimum: 1
                  default: 10
                maxModelLen:
                  type: integer
                  default: 512
                enableLora:
                  type: boolean
                  default: false
                adapters:
                  type: array
                  items:
                    type: object
                    properties:
                      name:
                        type: string
                      weightsURI:
                        type: string
                      version:
                        type: string
                        default: "latest"
                  default: []
            status:
              type: object
              properties:
                readyReplicas:
                  type: integer
                availableReplicas:
                  type: integer
                conditions:
                  type: array
                  items:
                    type: object
                    properties:
                      type:
                        type: string
                      status:
                        type: string
                      lastTransitionTime:
                        type: string
                      reason:
                        type: string
                      message:
                        type: string
      subresources:
        status: {}
      additionalPrinterColumns:
        - name: Model
          type: string
          jsonPath: .spec.modelName
        - name: Replicas
          type: integer
          jsonPath: .spec.replicas
        - name: Ready
          type: integer
          jsonPath: .status.readyReplicas
        - name: Age
          type: date
          jsonPath: .metadata.creationTimestamp
  scope: Namespaced
  names:
    plural: inferencemodels
    singular: inferencemodel
    kind: InferenceModel
    shortNames:
      - im
```

### Example CRD Instance

```yaml
apiVersion: inference.server/v1alpha1
kind: InferenceModel
metadata:
  name: qwen2-7b
  namespace: inference
spec:
  modelName: "Qwen/Qwen2.5-7B-Instruct-1M"
  weightsURI: "Qwen/Qwen2.5-7B-Instruct-1M"
  modelVersion: "main"
  engineType: vllm
  gpu:
    type: A100
    count: 1
    memoryUtilization: 0.7
  replicas: 2
  minReplicas: 1
  maxReplicas: 5
  maxModelLen: 4096
  enableLora: true
  adapters:
    - name: medical-qwen-lora
      weightsURI: "org/medical-qwen-lora"
      version: "latest"
```

### Model Metadata Controller

The controller watches `InferenceModel` CRDs and reconciles Kubernetes Deployments. It lives at `control_plane/model_controller.py`.

**Framework**: kopf (Python Kubernetes Operator Framework). Chosen because:
- The entire codebase is Python (pyproject.toml confirms Python 3.12+)
- kopf integrates well with existing pydantic patterns
- Simpler than writing a controller-runtime equivalent from scratch
- Add `kopf>=1.37.0` and `kubernetes>=29.0.0` to `pyproject.toml` dependencies

**Controller logic** (pseudocode):

```python
# control_plane/model_controller.py
import kopf
import kubernetes

@kopf.on.create('inference.server', 'v1alpha1', 'inferencemodels')
async def on_model_create(spec, name, namespace, **kwargs):
    """Create a Deployment for the new InferenceModel."""
    deployment = build_deployment(name, namespace, spec)
    apps_v1 = kubernetes.client.AppsV1Api()
    apps_v1.create_namespaced_deployment(namespace, deployment)
    # Also create headless Service
    core_v1 = kubernetes.client.CoreV1Api()
    service = build_headless_service(name, namespace, spec)
    core_v1.create_namespaced_service(namespace, service)

@kopf.on.update('inference.server', 'v1alpha1', 'inferencemodels')
async def on_model_update(spec, old, new, name, namespace, **kwargs):
    """Update the Deployment when CRD spec changes."""
    apps_v1 = kubernetes.client.AppsV1Api()
    patch = build_deployment_patch(name, namespace, new['spec'])
    apps_v1.patch_namespaced_deployment(f"inference-{name}", namespace, patch)

@kopf.on.delete('inference.server', 'v1alpha1', 'inferencemodels')
async def on_model_delete(name, namespace, **kwargs):
    """Delete the Deployment and Service when CRD is removed."""
    apps_v1 = kubernetes.client.AppsV1Api()
    apps_v1.delete_namespaced_deployment(f"inference-{name}", namespace)
    core_v1 = kubernetes.client.CoreV1Api()
    core_v1.delete_namespaced_service(f"inference-{name}", namespace)

@kopf.timer('inference.server', 'v1alpha1', 'inferencemodels', interval=30)
async def update_status(spec, name, namespace, **kwargs):
    """Periodically sync status.readyReplicas from the Deployment."""
    apps_v1 = kubernetes.client.AppsV1Api()
    deployment = apps_v1.read_namespaced_deployment(f"inference-{name}", namespace)
    return {
        'readyReplicas': deployment.status.ready_replicas or 0,
        'availableReplicas': deployment.status.available_replicas or 0,
    }
```

**Key function `build_deployment`**: Generates the Deployment YAML from Phase 1, injecting `spec.modelName` into `ENGINE_MODEL_NAME` and `SIDECAR_INITIAL_MODEL` env vars, `spec.gpu.count` into `nvidia.com/gpu` resources, `spec.gpu.memoryUtilization` into `ENGINE_GPU_MEMORY_UTILIZATION`, etc.

### Key Design Decisions

1. **kopf vs Go controller-runtime**: kopf is chosen for consistency with the Python codebase. If performance becomes an issue at scale (hundreds of CRDs), migration to a Go controller is straightforward since the reconciliation logic is simple.

2. **One Deployment per InferenceModel**: Direct 1:1 mapping. The Deployment name is `inference-{crd.metadata.name}`. Labels include `inference.server/model: {spec.modelName}` for selection by the EndpointSlice controller and gateway.

3. **Status subresource**: The controller writes `readyReplicas` and `conditions` to the CRD status, enabling `kubectl get inferencemodels` to show live state.

4. **LoRA adapter list in CRD**: The adapter list is informational for Phase 4 routing. Adapters are still fetched on-demand by the engine (the existing `engine.py` adapter fetch flow). The CRD list lets the gateway know which pods CAN serve which adapters.

### Deliverables

- [ ] `InferenceModel` CRD YAML (`kubernetes/crds/InferenceModel.yaml`)
- [ ] Model controller implementation (`control_plane/model_controller.py`)
- [ ] `build_deployment()` helper that maps CRD spec to Deployment YAML
- [ ] Status reconciliation timer
- [ ] `kopf` and `kubernetes` added to `pyproject.toml`
- [ ] Unit tests with mock K8s API client

### Dependencies

- Phase 1 (Pod spec template finalized)
- Working Dockerfiles

---

## Phase 3: Custom EndpointSlice Controller

### Overview

A controller that watches inference Pods and maintains a custom EndpointSlice resource per model. The EndpointSlice contains enriched metadata beyond what standard Kubernetes endpoints provide: model name, load state, adapter list, and metrics snapshot. This is the data source the gateway watches for routing decisions.

### Why Custom EndpointSlice

Standard Kubernetes EndpointSlices track pod IP + port + ready/not-ready. LLM inference routing needs additional per-pod metadata:
- Which model the pod serves
- Load state (downloading, warming, ready, overloaded)
- Which LoRA adapters are cached
- Current load metrics (pending requests, GPU memory usage)

### EndpointSlice Format

```yaml
apiVersion: discovery.k8s.io/v1
kind: EndpointSlice
metadata:
  name: inference-qwen2-7b-abc123
  namespace: inference
  labels:
    inference.server/model: "Qwen/Qwen2.5-7B-Instruct-1M"
    kubernetes.io/service-name: inference-qwen2-7b
  ownerReferences:
    - apiVersion: inference.server/v1alpha1
      kind: InferenceModel
      name: qwen2-7b
addressType: IPv4
ports:
  - name: engine
    port: 8080
    protocol: TCP
  - name: sidecar
    port: 8001
    protocol: TCP
endpoints:
  - addresses:
      - "10.244.1.5"
    conditions:
      ready: true
      serving: true
    targetRef:
      kind: Pod
      name: inference-qwen2-7b-abc12-xyz
      namespace: inference
```

### Load Metadata Storage

Standard EndpointSlice does not support arbitrary per-endpoint annotations. The controller maintains a parallel ConfigMap per model with per-pod load metadata, updated every 10s. The gateway watches both the EndpointSlice (for IP/ready state) and the ConfigMap (for load data).

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: inference-load-qwen2-7b
  namespace: inference
  labels:
    inference.server/model: "Qwen/Qwen2.5-7B-Instruct-1M"
data:
  endpoints: |
    [
      {
        "pod": "inference-qwen2-7b-abc12-xyz",
        "ip": "10.244.1.5",
        "ready": true,
        "loadState": "ready",
        "pendingRequests": 3,
        "gpuMemoryPct": 65,
        "tokensPerSecond": 142.5,
        "cachedAdapters": ["medical-qwen-lora"],
        "lastUpdated": "2026-02-24T10:30:00Z"
      }
    ]
```

### Controller Implementation

The controller lives at `control_plane/endpoint_controller.py`.

```python
# control_plane/endpoint_controller.py (pseudocode)
import kopf
import httpx
import kubernetes

@kopf.on.event('', 'v1', 'pods', labels={'app.kubernetes.io/component': 'inference'})
async def on_pod_event(event, body, **kwargs):
    """React to pod create/update/delete events."""
    pod = body
    pod_name = pod['metadata']['name']
    pod_ip = pod.get('status', {}).get('podIP')
    model_name = pod['metadata']['labels'].get('inference.server/model')

    if not model_name or not pod_ip:
        return

    # Determine load state by querying the pod's sidecar /ready endpoint
    load_state = await _probe_load_state(pod_ip)
    # Update the EndpointSlice for this model
    await _reconcile_endpoint_slice(model_name, pod_name, pod_ip, load_state)

@kopf.timer('', 'v1', 'pods', labels={'app.kubernetes.io/component': 'inference'}, interval=10)
async def refresh_load_metrics(body, **kwargs):
    """Periodically scrape /metrics from each pod and update load metadata."""
    pod_ip = body.get('status', {}).get('podIP')
    if not pod_ip:
        return
    metrics = await _scrape_pod_metrics(pod_ip)
    model_name = body['metadata']['labels'].get('inference.server/model')
    await _update_load_configmap(model_name, body['metadata']['name'], metrics)

async def _scrape_pod_metrics(pod_ip: str) -> dict:
    """Scrape Prometheus metrics from engine and parse key values."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get(f"http://{pod_ip}:8080/metrics")
        # Parse prometheus text format for:
        # - engine_pending_requests
        # - engine_gpu_memory_used_bytes
        # - engine_tokens_per_second
        # - engine_requests_total
        return parse_prometheus_text(resp.text)
```

### Key Design Decisions

1. **Separate ConfigMap for load metrics**: EndpointSlice updates trigger gateway cache invalidation. Updating load metrics every 10s directly in the EndpointSlice would cause excessive churn. A separate ConfigMap decouples topology changes (rare) from load changes (frequent).

2. **10-second scrape interval**: Balances freshness vs. overhead. The existing `prometheus.yml` uses 15s. The endpoint controller scrapes more aggressively because routing decisions depend on it.

3. **Scraping /metrics directly**: The engine already exposes Prometheus metrics at `/metrics`. Metrics defined in `engine/metrics.py` include `engine_pending_requests`, `engine_gpu_memory_used_bytes`, `engine_tokens_per_second`. Several of these are not yet recorded — they need to be wired in engine code as a prerequisite.

### Deliverables

- [ ] EndpointSlice controller (`control_plane/endpoint_controller.py`)
- [ ] Load metadata ConfigMap reconciliation
- [ ] Prometheus text format parser utility
- [ ] Wire missing engine metrics: `engine_pending_requests`, `engine_gpu_memory_used_bytes`, `engine_tokens_per_second` in `engine/api.py` and `engine/engine.py`
- [ ] Unit tests with mock pod events

### Dependencies

- Phase 1 (Pod labels and Service)
- Phase 2 (InferenceModel CRD for ownerReferences)
- Engine metrics must be fully wired (currently only 3 of 11 are recorded)

---

## Phase 4: Gateway — Dynamic Discovery and Routing

### Overview

Replace the hardcoded `MODEL_SERVICE_MAP` in `data_plane/gateway/routing.py` with a Kubernetes-aware dynamic routing system. The gateway watches EndpointSlices and load ConfigMaps to maintain a live view of available pods. Incoming requests pass through a pipeline of filters and scorers to select the optimal pod.

### Routing Pipeline Architecture

```
  Incoming Request
       |
       v
+------+------+
| Model Router |  lookup model -> candidate pods from EndpointSlice
+------+------+
       |
       v
+------+------+
| Ready Filter |  drop pods where conditions.ready = false
+------+------+
       |
       v
+------+------+
| Load State   |  drop pods in "downloading" or "warming" state
| Filter       |
+------+------+
       |
       v
+------+---------+
| Prefix-Aware   |  score pods by KV cache prefix match (future)
| Scorer         |  boost score if prompt prefix hash matches
+------+---------+
       |
       v
+------+---------+
| Load-Aware     |  score pods by composite load metric
| Scorer         |  (pending_requests, gpu_mem, tokens/sec)
+------+---------+
       |
       v
+------+---------+
| Pod Selector   |  pick highest-scoring pod (or weighted random)
+------+---------+
       |
       v
  Forward to pod_ip:8080/inference
```

### Implementation

The gateway needs three new modules:

**1. K8s Watcher** (`data_plane/gateway/k8s_watcher.py`): Watches EndpointSlices and load ConfigMaps.

```python
# data_plane/gateway/k8s_watcher.py
from kubernetes import client, watch
import asyncio
from typing import Dict, List
from dataclasses import dataclass, field

@dataclass
class PodEndpoint:
    pod_name: str
    ip: str
    port: int = 8080
    model: str = ""
    ready: bool = False
    load_state: str = "unknown"   # downloading | warming | ready | overloaded
    pending_requests: int = 0
    gpu_memory_pct: float = 0.0
    tokens_per_second: float = 0.0
    cached_adapters: List[str] = field(default_factory=list)
    prefix_hashes: List[str] = field(default_factory=list)

class EndpointRegistry:
    """Thread-safe registry of model -> list of PodEndpoints."""
    def __init__(self):
        self._endpoints: Dict[str, List[PodEndpoint]] = {}
        self._lock = asyncio.Lock()

    async def get_endpoints(self, model: str) -> List[PodEndpoint]:
        async with self._lock:
            return list(self._endpoints.get(model, []))

    async def update_from_endpointslice(self, slice_data): ...
    async def update_from_load_configmap(self, cm_data): ...

async def watch_endpoint_slices(registry: EndpointRegistry):
    """Long-running task that watches EndpointSlice changes."""
    ...

async def watch_load_configmaps(registry: EndpointRegistry):
    """Long-running task that watches load ConfigMap changes."""
    ...
```

**2. Routing Pipeline** (`data_plane/gateway/router.py`):

```python
# data_plane/gateway/router.py
from typing import List, Optional, Protocol

class Filter(Protocol):
    def apply(self, candidates: List[PodEndpoint], request: dict) -> List[PodEndpoint]: ...

class Scorer(Protocol):
    def score(self, candidate: PodEndpoint, request: dict) -> float: ...

class ReadyFilter:
    def apply(self, candidates, request):
        return [c for c in candidates if c.ready]

class LoadStateFilter:
    def apply(self, candidates, request):
        return [c for c in candidates if c.load_state == "ready"]

class LoadAwareScorer:
    """Composite score: lower pending requests + lower GPU memory = higher score."""
    def score(self, candidate, request):
        pending_score = 1.0 - min(candidate.pending_requests / 10.0, 1.0)
        gpu_score = 1.0 - (candidate.gpu_memory_pct / 100.0)
        throughput_score = min(candidate.tokens_per_second / 200.0, 1.0)
        return 0.5 * pending_score + 0.3 * gpu_score + 0.2 * throughput_score

class PrefixAwareScorer:
    """Boost score for pods that have the request's prompt prefix cached."""
    def score(self, candidate, request):
        prompt = request.get("prompt", "")
        prefix_hash = compute_prefix_hash(prompt, prefix_len=64)
        if prefix_hash in candidate.prefix_hashes:
            return 1.0   # strong boost
        return 0.0

class RoutingPipeline:
    def __init__(self, filters: List[Filter], scorers: List[Scorer]):
        self.filters = filters
        self.scorers = scorers

    def select(self, candidates: List[PodEndpoint], request: dict) -> Optional[PodEndpoint]:
        for f in self.filters:
            candidates = f.apply(candidates, request)
            if not candidates:
                return None

        scored = []
        for c in candidates:
            total_score = sum(s.score(c, request) for s in self.scorers)
            scored.append((total_score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored else None
```

**3. Updated Gateway** (`data_plane/gateway/routing.py`): Replace `MODEL_SERVICE_MAP` with the registry + pipeline.

```python
# Updated routing.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = EndpointRegistry()
    pipeline = RoutingPipeline(
        filters=[ReadyFilter(), LoadStateFilter()],
        scorers=[LoadAwareScorer()],
    )
    app.state.registry = registry
    app.state.pipeline = pipeline
    app.state.http_client = httpx.AsyncClient(timeout=300.0)

    watcher_task = asyncio.create_task(watch_endpoint_slices(registry))
    load_task = asyncio.create_task(watch_load_configmaps(registry))
    yield
    watcher_task.cancel()
    load_task.cancel()
    await app.state.http_client.aclose()

@app.post("/generate")
async def generate(event: dict):
    model_name = event['model']
    candidates = await app.state.registry.get_endpoints(model_name)

    if not candidates:
        # Trigger cold start (Phase 5)
        raise HTTPException(status_code=503, detail=f"No endpoints for model {model_name}")

    pod = app.state.pipeline.select(candidates, event)
    if not pod:
        raise HTTPException(status_code=503, detail="All endpoints busy or not ready")

    target_url = f"http://{pod.ip}:{pod.port}/inference"
    response = await app.state.http_client.post(target_url, json=event)
    return response.json()
```

### Fallback for Non-K8s Environments

Add a config flag `GATEWAY_DISCOVERY_MODE` for docker-compose development:

```python
class GatewayConfig(BaseSettings):
    model_config = {"env_prefix": "GATEWAY_"}
    host: str = "0.0.0.0"
    port: int = 8000
    request_timeout: float = 300.0
    discovery_mode: str = "static"           # "static" or "kubernetes"
    static_backends: dict = {"llama-3-8b": "http://engine:8080"}
```

### Key Design Decisions

1. **Filter/Scorer pipeline pattern**: Composable, testable, extensible. New routing strategies (e.g., cost-aware, latency-SLO-aware) can be added as new Scorer implementations without modifying existing code.

2. **Direct pod IP routing (not through Service)**: The gateway routes to individual pod IPs obtained from the EndpointSlice. This bypasses kube-proxy load balancing, giving the gateway full control over which pod receives each request.

3. **Metrics scraping by endpoint controller, not by gateway**: The gateway reads the ConfigMap rather than scraping each pod directly. This reduces N*M scrape connections (N gateways * M pods) to one controller scraping M pods.

4. **Prefix-aware scoring deferred**: Full prefix-aware routing requires pods to advertise their cached prefix hashes. This is not available from the engine today. The PrefixAwareScorer is included in the pipeline design but will produce zero scores until the engine exposes prefix hash information (tied to KV cache integration in Phase 7).

### Deliverables

- [ ] K8s watcher module (`data_plane/gateway/k8s_watcher.py`)
- [ ] Routing pipeline module (`data_plane/gateway/router.py`)
- [ ] Updated `routing.py` with dynamic discovery
- [ ] `GatewayConfig` extended with `discovery_mode` and `static_backends`
- [ ] `kubernetes` client added to gateway Docker image and `pyproject.toml`
- [ ] Unit tests for each Filter and Scorer
- [ ] Integration test: gateway discovers mock EndpointSlices

### Dependencies

- Phase 3 (EndpointSlice and load ConfigMap must exist)
- Engine metrics wired (for LoadAwareScorer to have meaningful data)

---

## Phase 5: Cold Start Manager

### Overview

When a request arrives for a model with no ready endpoints (either the model has never been deployed, or it was scaled to zero), the Cold Start Manager queues the request, triggers a scale-up, and responds once a pod becomes ready or a timeout is reached.

### Architecture

```
  Gateway receives POST /generate {model: "new-model"}
       |
       | registry.get_endpoints("new-model") -> []
       v
  +----+----+
  | Cold    |  1. Check InferenceModel CRD exists for "new-model"
  | Start   |  2. If CRD exists with replicas=0: patch replicas to 1
  | Manager |  3. If CRD does not exist: return 404
  +---------+  4. Queue the request with a Future
       |       5. Watch EndpointSlice for "new-model" to appear
       |       6. When ready endpoint appears: resolve Future, route request
       v       7. If timeout: resolve Future with 504
  Response to client (or 504 timeout)
```

### Implementation

The Cold Start Manager is part of the gateway process, not a separate service. It lives at `data_plane/gateway/cold_start.py`.

```python
# data_plane/gateway/cold_start.py
import asyncio
from typing import Dict, Optional
from kubernetes import client as k8s_client

class ColdStartManager:
    def __init__(self, registry, timeout: float = 300.0, max_queue_size: int = 50):
        self.registry = registry
        self.timeout = timeout
        self.max_queue_size = max_queue_size
        self._pending: Dict[str, list] = {}  # model -> list of Futures

    async def handle_cold_start(self, model_name: str) -> Optional[str]:
        """
        Trigger scale-up for a model and wait for an endpoint.
        Returns the pod IP when ready, or raises TimeoutError.
        """
        queue = self._pending.setdefault(model_name, [])
        if len(queue) >= self.max_queue_size:
            raise QueueFullError(f"Cold start queue full for {model_name}")

        crd_spec = await self._get_inference_model(model_name)
        if crd_spec is None:
            raise ModelNotFoundError(f"No InferenceModel CRD for {model_name}")

        if crd_spec.get('spec', {}).get('replicas', 0) == 0:
            await self._scale_model(model_name, target_replicas=1)

        future = asyncio.Future()
        queue.append(future)

        try:
            pod_ip = await asyncio.wait_for(future, timeout=self.timeout)
            return pod_ip
        except asyncio.TimeoutError:
            raise ColdStartTimeoutError(
                f"Model {model_name} did not become ready within {self.timeout}s"
            )
        finally:
            if future in queue:
                queue.remove(future)

    async def notify_endpoint_ready(self, model_name: str, pod_ip: str):
        """Called by the registry watcher when a new endpoint becomes ready."""
        queue = self._pending.get(model_name, [])
        for future in queue:
            if not future.done():
                future.set_result(pod_ip)

    async def _get_inference_model(self, model_name: str) -> Optional[dict]:
        """Query the K8s API for the InferenceModel CRD."""
        custom_api = k8s_client.CustomObjectsApi()
        try:
            return custom_api.get_namespaced_custom_object(
                group="inference.server", version="v1alpha1",
                namespace="inference", plural="inferencemodels",
                name=model_name
            )
        except k8s_client.ApiException as e:
            if e.status == 404:
                return None
            raise

    async def _scale_model(self, model_name: str, target_replicas: int):
        """Patch the InferenceModel CRD replicas field."""
        custom_api = k8s_client.CustomObjectsApi()
        custom_api.patch_namespaced_custom_object(
            group="inference.server", version="v1alpha1",
            namespace="inference", plural="inferencemodels",
            name=model_name,
            body={"spec": {"replicas": target_replicas}}
        )
```

### Integration with Gateway

In `routing.py`, the `/generate` endpoint becomes:

```python
@app.post("/generate")
async def generate(event: dict):
    model_name = event['model']
    candidates = await app.state.registry.get_endpoints(model_name)

    if not candidates:
        # Cold start path
        try:
            pod_ip = await app.state.cold_start.handle_cold_start(model_name)
            target_url = f"http://{pod_ip}:8080/inference"
            response = await app.state.http_client.post(target_url, json=event)
            return response.json()
        except ModelNotFoundError:
            raise HTTPException(404, detail=f"Model {model_name} not found")
        except ColdStartTimeoutError:
            raise HTTPException(504, detail=f"Model {model_name} cold start timed out")
        except QueueFullError:
            raise HTTPException(429, detail="Too many requests queued for cold start")

    pod = app.state.pipeline.select(candidates, event)
    ...
```

### Key Design Decisions

1. **Cold start manager inside gateway process**: Not a separate controller. The gateway already watches EndpointSlices and is the natural place to queue requests. A separate cold-start service would add latency and complexity.

2. **CRD replicas patch (not Deployment patch)**: The cold start manager patches the InferenceModel CRD `spec.replicas` from 0 to 1. The model controller (Phase 2) then reconciles the Deployment. This preserves the CRD as the single source of truth.

3. **Queue per model with max size**: Prevents unbounded memory growth. Default 50 queued requests per model during cold start. Excess requests get 429.

4. **Timeout of 300s**: Model download + GPU loading can take minutes. The existing `EngineConfig.sidecar_timeout` is 600s. A 300s cold-start timeout is aggressive but prevents clients from waiting too long. Configurable.

### Deliverables

- [ ] `ColdStartManager` class (`data_plane/gateway/cold_start.py`)
- [ ] Integration into gateway `/generate` endpoint
- [ ] EndpointSlice watcher callback to `notify_endpoint_ready`
- [ ] Configuration: timeout, max_queue_size
- [ ] Unit tests: cold start success, timeout, queue full, model not found
- [ ] Integration test: scale-from-zero end-to-end with mock K8s

### Dependencies

- Phase 2 (InferenceModel CRD must exist and be reconciled)
- Phase 3 (EndpointSlice must update when pods become ready)
- Phase 4 (Gateway registry and watcher infrastructure)

---

## Phase 6: LLM-Specific Autoscaler

### Overview

A custom controller that watches LLM-specific metrics from inference pods and adjusts replica counts on InferenceModel CRDs. Unlike standard HPA (which uses CPU/memory), this autoscaler understands request queue depth, token throughput, GPU memory pressure, and model load state.

### Architecture

```
  +-------------------+
  | Prometheus        |  stores scraped metrics from all engine pods
  |  (or direct poll) |
  +--------+----------+
           |
           | PromQL queries (or /metrics scrape)
           v
  +--------+----------+
  | LLM Autoscaler    |  control_plane/autoscaler.py
  | Controller        |
  |                   |
  | For each model:   |
  |  1. Query metrics |
  |  2. Compute       |
  |     scaling       |
  |     decision      |
  |  3. Patch CRD     |
  |     replicas      |
  +--------+----------+
           |
           | PATCH InferenceModel.spec.replicas
           v
  +--------+----------+
  | Model Controller  |  reconciles Deployment to match CRD
  +-------------------+
```

### Metrics Source

Two options for getting metrics into the autoscaler:

**Option A (recommended): Prometheus queries**. The cluster already has Prometheus scraping engine pods (see `docker/prometheus.yml`). The autoscaler queries Prometheus for aggregate metrics per model.

**Option B: Direct /metrics scrape**. The autoscaler reads the load ConfigMap from Phase 3, which already contains per-pod metrics. Simpler, no Prometheus dependency, but less sophisticated aggregation.

Recommend Option A for production, Option B as a fallback for simpler setups.

### Scaling Logic

```python
# control_plane/autoscaler.py
from dataclasses import dataclass

@dataclass
class ScalingConfig:
    evaluation_interval: float = 30.0    # seconds between evaluations
    scale_up_threshold_pending: int = 5  # avg pending requests per pod
    scale_down_threshold_pending: int = 1
    scale_up_threshold_gpu_pct: float = 85.0
    scale_down_threshold_gpu_pct: float = 30.0
    scale_up_threshold_latency_p95: float = 5.0  # seconds
    scale_down_idle_period: float = 300.0  # 5 min idle before scale down
    cooldown_scale_up: float = 60.0    # seconds after a scale-up before next
    cooldown_scale_down: float = 300.0  # seconds after a scale-down before next

@dataclass
class ModelMetrics:
    avg_pending_requests: float
    total_tokens_per_second: float
    avg_gpu_memory_pct: float
    p95_latency: float
    total_ready_replicas: int
    total_warming_replicas: int

class LLMAutoscaler:
    def __init__(self, config: ScalingConfig):
        self.config = config
        self._last_scale_up: dict = {}   # model -> timestamp
        self._last_scale_down: dict = {} # model -> timestamp

    def compute_desired_replicas(
        self, model_name: str, current: int, min_r: int, max_r: int, m: ModelMetrics
    ) -> int:
        """Determine target replica count based on metrics."""
        desired = current

        # Rule 1: Scale up on queue pressure
        if m.avg_pending_requests > self.config.scale_up_threshold_pending:
            desired = max(desired, int(
                current * m.avg_pending_requests / self.config.scale_up_threshold_pending
            ))

        # Rule 2: Scale up on GPU memory pressure
        if m.avg_gpu_memory_pct > self.config.scale_up_threshold_gpu_pct:
            desired = max(desired, current + 1)

        # Rule 3: Scale up on latency SLO violation
        if m.p95_latency > self.config.scale_up_threshold_latency_p95:
            desired = max(desired, current + 1)

        # Rule 4: Scale down on low utilization
        if (m.avg_pending_requests < self.config.scale_down_threshold_pending
            and m.avg_gpu_memory_pct < self.config.scale_down_threshold_gpu_pct
            and m.total_warming_replicas == 0):   # don't scale down while warming
            desired = min(desired, current - 1)

        # Clamp to [minReplicas, maxReplicas]
        desired = max(min_r, min(desired, max_r))
        return desired
```

### Controller Loop

```python
@kopf.timer('inference.server', 'v1alpha1', 'inferencemodels', interval=30)
async def autoscale_model(spec, status, name, namespace, **kwargs):
    """Evaluate scaling for each InferenceModel every 30s."""
    model_name = spec['modelName']
    current_replicas = spec.get('replicas', 1)
    min_replicas = spec.get('minReplicas', 0)
    max_replicas = spec.get('maxReplicas', 10)

    metrics = await query_model_metrics(model_name)

    desired = autoscaler.compute_desired_replicas(
        model_name, current_replicas, min_replicas, max_replicas, metrics
    )

    if desired != current_replicas:
        if not _in_cooldown(model_name, desired > current_replicas):
            custom_api.patch_namespaced_custom_object(
                group="inference.server", version="v1alpha1",
                namespace=namespace, plural="inferencemodels", name=name,
                body={"spec": {"replicas": desired}}
            )
            _record_scale_event(model_name, desired > current_replicas)
```

### Scale-to-Zero

When `minReplicas: 0` is set in the CRD:
- The autoscaler can scale the Deployment to zero replicas when no requests have been received for `scale_down_idle_period` seconds.
- Incoming requests for a zero-replica model trigger the Cold Start Manager (Phase 5).
- The CRD's `spec.replicas` is patched to 0, and the model controller reconciles the Deployment accordingly.

### Prometheus Queries

Example PromQL for the autoscaler:

```promql
# Average pending requests per pod for a model
avg(engine_pending_requests{model="Qwen/Qwen2.5-7B-Instruct-1M"})

# p95 request latency
histogram_quantile(0.95, rate(engine_request_duration_seconds_bucket{model="..."}[5m]))

# Total tokens/sec across all pods
sum(engine_tokens_per_second{model="Qwen/Qwen2.5-7B-Instruct-1M"})

# Average GPU memory usage percentage
avg(engine_gpu_memory_used_bytes{device=~".*"}) / (8 * 1024^3) * 100
```

### Key Design Decisions

1. **Custom controller, not HPA**: HPA cannot query LLM-specific metrics like pending requests or token throughput. The custom metrics API adapter pattern (Prometheus Adapter -> HPA) is possible but less flexible than a purpose-built controller that understands model semantics.

2. **Scaling the CRD, not the Deployment**: The autoscaler patches `InferenceModel.spec.replicas`. The model controller (Phase 2) reconciles the Deployment. This keeps the CRD as the single source of truth and avoids conflicts between autoscaler and controller both patching the Deployment.

3. **Cooldown periods**: Scale-up cooldown (60s) is shorter than scale-down cooldown (300s). This is asymmetric by design: scaling up is urgent (latency impact), scaling down should be cautious (avoid flapping).

4. **Warming pod protection**: The autoscaler never scales down if there are pods in "warming" state. This prevents killing pods that are still loading model weights.

### Deliverables

- [ ] `LLMAutoscaler` class (`control_plane/autoscaler.py`)
- [ ] Kopf timer for periodic evaluation
- [ ] Prometheus query utility
- [ ] Scale-to-zero support with Cold Start Manager integration
- [ ] Cooldown tracking
- [ ] Scaling event logging and metrics
- [ ] Unit tests for scaling logic edge cases
- [ ] `prometheus-api-client` added to `pyproject.toml`

### Dependencies

- Phase 2 (InferenceModel CRD with min/maxReplicas)
- Phase 3 (Endpoint controller wiring engine metrics)
- Phase 5 (Cold Start Manager for scale-from-zero)
- Prometheus running in-cluster
- Engine metrics fully wired (`engine_pending_requests`, `engine_gpu_memory_used_bytes`, `engine_tokens_per_second`)

---

## Phase 7: Distributed KV Cache (Placeholder)

### Overview

The L1 (CPU DRAM) and L2 (Redis) KV cache tiers are scaffolded in the codebase but not integrated into the inference serving path. This phase is a placeholder for future work.

### Current State

Code exists at:
- `data_plane/inference/sidecar/l1_cache/` — L1CacheAPI, L1Allocator, LRU EvictionPolicy, GPUTransferHandler (all simulated)
- `data_plane/inference/sidecar/l2_cache/` — L2Connector with ConsistentHashRing, KVCacheWatcherClient
- `data_plane/inference/distributed_cache/` — KVCacheWatcher gRPC servicer, StorageNode
- `data_plane/inference/sidecar/cache_manager.py` — MultiTieredCacheManager orchestrating L1 -> L2 offload/fetch

None of these are called from the serving path. The engine uses vLLM's built-in PagedAttention for GPU KV cache management.

### Key Questions to Resolve

1. **Cache coherence**: When a pod evicts a KV prefix from GPU memory to L1 DRAM, how do other pods discover that the prefix exists in another pod's L1? Options: gossip protocol, centralized registry, or broadcast via the EndpointSlice metadata.

2. **Eviction coordination**: If the autoscaler scales down a pod, its L1 cache is lost. Should the pod pre-emptively offload hot entries to L2 (Redis) before termination? This requires a `preStop` hook that signals the cache manager.

3. **Transfer protocol**: Moving KV blocks between pods (L1-to-L1) or from L1 to L2 (Redis) requires serialization. The gRPC proto stubs are planned but not defined. Key considerations: block size (typical KV block is 16-64 KB), latency budget (must be faster than recomputation), and network bandwidth.

4. **Prefix hash tracking for routing**: The gateway's PrefixAwareScorer (Phase 4) needs pods to advertise which prompt prefix hashes have cached KV entries. This requires the engine to expose prefix hashes, likely via the sidecar's `/metrics` or a new endpoint.

5. **Integration point with vLLM**: vLLM manages its own KV cache internally via PagedAttention. Offloading to L1/L2 requires hooking into vLLM's cache eviction callbacks or using vLLM's upcoming external KV cache API.

### Placeholder Deliverables

- [ ] Document L1/L2/L3 cache tier boundaries
- [ ] Define `kv_cache.proto` for inter-pod KV transfer
- [ ] Spike: measure L1 offload latency vs. recomputation cost
- [ ] Spike: evaluate vLLM external cache API compatibility

---

## Cross-Cutting Concerns

### Helm Chart

Package the entire system as a Helm chart at `charts/inference-server/`.

```
charts/inference-server/
  Chart.yaml
  values.yaml
  templates/
    _helpers.tpl
    namespace.yaml
    rbac.yaml
    crd-inferencemodel.yaml
    controller-deployment.yaml
    gateway-deployment.yaml
    gateway-service.yaml
    prometheus-servicemonitor.yaml
```

The chart does NOT template inference Deployments directly — those are created by the model controller in response to InferenceModel CRDs. The chart deploys: gateway, model controller, endpoint controller, autoscaler, CRD definitions, and RBAC.

`values.yaml` key sections:

```yaml
gateway:
  replicas: 2
  image: inference-server/gateway:latest
  discoveryMode: kubernetes
  resources:
    requests: { cpu: "500m", memory: "512Mi" }

controller:
  image: inference-server/controller:latest
  resources:
    requests: { cpu: "200m", memory: "256Mi" }

autoscaler:
  enabled: true
  evaluationInterval: 30
  prometheus:
    url: http://prometheus.monitoring:9090

monitoring:
  serviceMonitor:
    enabled: true
    interval: 15s
```

### RBAC

```yaml
# Controller ServiceAccount + ClusterRole
apiVersion: v1
kind: ServiceAccount
metadata:
  name: inference-controller
  namespace: inference-system
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: inference-controller
rules:
  - apiGroups: ["inference.server"]
    resources: ["inferencemodels", "inferencemodels/status"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
  - apiGroups: ["apps"]
    resources: ["deployments"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
  - apiGroups: [""]
    resources: ["services"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
  - apiGroups: ["discovery.k8s.io"]
    resources: ["endpointslices"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get", "list", "watch"]
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["get", "list", "watch", "create", "update", "patch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: inference-controller
subjects:
  - kind: ServiceAccount
    name: inference-controller
    namespace: inference-system
roleRef:
  kind: ClusterRole
  name: inference-controller
  apiGroup: rbac.authorization.k8s.io
---
# Gateway ServiceAccount (read-only + cold start patch)
apiVersion: v1
kind: ServiceAccount
metadata:
  name: inference-gateway
  namespace: inference
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: inference-gateway
rules:
  - apiGroups: ["discovery.k8s.io"]
    resources: ["endpointslices"]
    verbs: ["get", "list", "watch"]
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["get", "list", "watch"]
  - apiGroups: ["inference.server"]
    resources: ["inferencemodels"]
    verbs: ["get", "list", "watch", "patch"]   # patch needed for cold start
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: inference-gateway
subjects:
  - kind: ServiceAccount
    name: inference-gateway
    namespace: inference
roleRef:
  kind: ClusterRole
  name: inference-gateway
  apiGroup: rbac.authorization.k8s.io
```

### Monitoring

**Prometheus ServiceMonitor** (requires prometheus-operator):

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: inference-engines
  namespace: inference
spec:
  selector:
    matchLabels:
      app.kubernetes.io/component: inference
  endpoints:
    - port: engine
      path: /metrics
      interval: 15s
    - port: sidecar
      path: /metrics
      interval: 30s
```

**Grafana Dashboard panels**:
- Requests per second per model (from `engine_requests_total`)
- p50/p95/p99 latency (from `engine_request_duration_seconds`)
- Pending requests per pod (from `engine_pending_requests`)
- GPU memory per pod (from `engine_gpu_memory_used_bytes`)
- Tokens/sec aggregate (from `engine_tokens_per_second`)
- Replica count per model (from kube-state-metrics or CRD status)
- Autoscaler decisions (custom metric from autoscaler)

### Testing Strategy

1. **Unit tests** (no K8s required):
   - Routing pipeline filters/scorers with mock PodEndpoints
   - Autoscaler scaling logic with synthetic ModelMetrics
   - Cold start manager with mock K8s API
   - Controller build_deployment output validation

2. **Integration tests** (kind cluster):
   - Install CRD + controller, apply InferenceModel, verify Deployment created
   - Verify EndpointSlice populated when pod becomes ready
   - Gateway routes to mock engine pod

3. **E2E tests** (kind cluster + mock engine):
   - Full flow: CRD -> Deployment -> Pod -> EndpointSlice -> Gateway routes -> Response
   - Scale-to-zero -> cold start -> scale-up
   - Autoscaler reacts to simulated load

```makefile
test-k8s:
	kind create cluster --name inference-test
	kubectl apply -f kubernetes/crds/
	helm install inference-server charts/inference-server/ \
	  --set controller.image=... --set gateway.image=...
	uv run pytest tests/e2e/ -v
	kind delete cluster --name inference-test
```

---

## Implementation Sequencing

```
Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5 ──► Phase 6
  Pod         CRD &      Endpoint    Gateway     Cold Start   Autoscaler
  Spec       Controller  Controller  Routing     Manager

  |                                                            |
  |           (cross-cutting: RBAC, Helm, monitoring)          |
  |                                                            |
  └──── Phase 7 (placeholder, can start spike work anytime) ───┘
```

### Sprint Plan

| Sprint | Phases | Estimate | Key Milestone |
|--------|--------|----------|---------------|
| 1 | Phase 1 | 1 sprint | Pod spec validated on kind cluster with mock engine |
| 2 | Phase 2 | 1-2 sprints | `kubectl apply` InferenceModel CRD creates a working Deployment |
| 3 | Phase 3 + wire engine metrics | 1 sprint | EndpointSlice auto-populated when pods start |
| 4 | Phase 4 | 1-2 sprints | Gateway dynamically routes to pods via EndpointSlice |
| 5 | Phase 5 | 1 sprint | Scale-from-zero works end-to-end |
| 6 | Phase 6 + Helm + RBAC | 2 sprints | Autoscaler adjusts replicas based on load |
| 7 | Monitoring, CI/CD, testing | 1 sprint | Full chart installable, dashboards, E2E tests |

### Prerequisite Work (parallel with Sprint 1)

- Wire the 8 unrecorded engine metrics in `engine/metrics.py` (needed by Phases 3, 4, 6)
- Build working Dockerfiles for all services (exist but need validation)
- Add `kopf` and `kubernetes` to `pyproject.toml`

### Critical Files

| File | Phases | Change |
|------|--------|--------|
| `data_plane/gateway/routing.py` | 4, 5 | Replace hardcoded routing with K8s-aware pipeline |
| `data_plane/inference/engine/metrics.py` | 3, 4, 6 | Wire 8 unrecorded Prometheus metrics |
| `data_plane/inference/engine/api.py` | 1, 3 | Record metrics in request handlers |
| `control_plane/autoscaler.py` | 6 | Implement LLM-specific autoscaler (currently stub) |
| `control_plane/model_controller.py` | 2 | New file: kopf operator for InferenceModel CRD |
| `control_plane/endpoint_controller.py` | 3 | New file: EndpointSlice + load ConfigMap controller |
| `data_plane/gateway/k8s_watcher.py` | 4 | New file: EndpointSlice/ConfigMap watcher |
| `data_plane/gateway/router.py` | 4 | New file: Filter/Scorer routing pipeline |
| `data_plane/gateway/cold_start.py` | 5 | New file: Cold start request queuing |
| `kubernetes/crds/InferenceModel.yaml` | 2 | New CRD definition |
| `manifests/05-inference-service-deploy.yaml` | 1 | Expand skeleton into full Deployment + Service |
