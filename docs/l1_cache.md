# L1 KV Cache Offloading — Complete Walkthrough

## The Analogy: A Library with a Reading Room

Think of the GPU as a **reading room** with limited desk space. Each desk holds a book (a KV cache block) being actively used by a reader (an inference request). When all desks are full and a new reader arrives, the librarian must move some books to a **storage closet** (the L1 sidecar cache) nearby. When a reader needs a book again, it's fetched back from the closet. If the closet fills up, the least-recently-used book gets discarded.

The librarian communicates with the closet attendant using **written notes sent through a tube** (gRPC). The note says "store this book" or "bring me book #42." The closet attendant organizes books by numbered slots and keeps a card catalog (the registry) of what's where.

---

## Architecture Overview

```
┌─────────────────────────────────────┐    ┌──────────────────────────────────────────┐
│         ENGINE CONTAINER            │    │          SIDECAR CONTAINER               │
│                                     │    │                                          │
│  ┌─────────────────────────┐        │    │   ┌──────────────────────────────────┐   │
│  │     vLLM Scheduler      │        │    │   │      FastAPI (api.py :8001)      │   │
│  │  (memory pressure →     │        │    │   │   /cache/stats  /cache/blocks    │   │
│  │   offload decision)     │        │    │   │   /metrics      /health          │   │
│  └───────────┬─────────────┘        │    │   └──────────────────────────────────┘   │
│              │ calls                │    │                                          │
│  ┌───────────▼─────────────┐        │    │   ┌──────────────────────────────────┐   │
│  │  SidecarOffloadingSpec  │        │    │   │   gRPC Server (:50051)           │   │
│  │    (sidecar_spec.py)    │        │    │   │     grpc_server.py               │   │
│  │                         │        │    │   │         │                        │   │
│  │  ┌───────────────────┐  │        │    │   │   ┌─────▼──────────────────┐     │   │
│  │  │  SidecarBackend   │  │        │    │   │   │  KVCacheServicer      │     │   │
│  │  │ (block ID tracker)│  │        │    │   │   │  (kv_cache_api.py)    │     │   │
│  │  └───────────────────┘  │        │    │   │   └─────┬──────────────────┘     │   │
│  │  ┌───────────────────┐  │        │    │   └─────────┼────────────────────────┘   │
│  │  │SidecarOffloading  │  │        │    │             │ delegates                  │
│  │  │    Handler        │  │        │    │   ┌─────────▼────────────────────────┐   │
│  │  │(sidecar_handler)  │  │        │    │   │  MultiTieredCacheManager         │   │
│  │  └────────┬──────────┘  │        │    │   │     (cache_manager.py)           │   │
│  └───────────┼─────────────┘        │    │   └──┬──────────────────┬────────────┘   │
│              │ gRPC calls           │    │      │                  │                │
│  ┌───────────▼─────────────┐        │    │   ┌──▼──────────┐  ┌───▼────────────┐   │
│  │  SidecarCacheClient     │════════════════▶│ L1ByteStore  │  │ KVBlockRegistry│   │
│  │ (sidecar_cache_client)  │  gRPC  │    │   │ (l1/api.py)  │  │(kv_block_reg.) │   │
│  └─────────────────────────┘  :50051│    │   │              │  └────────────────┘   │
│                                     │    │   │  ┌─────────┐ │                       │
│  ┌──────────────────┐               │    │   │  │Allocator│ │                       │
│  │   GPU Memory     │               │    │   │  │  + LRU  │ │                       │
│  │ (KV cache blocks)│               │    │   │  └─────────┘ │                       │
│  └──────────────────┘               │    │   └──────────────┘                       │
│                                     │    │      CPU DRAM                            │
└─────────────────────────────────────┘    └──────────────────────────────────────────┘
```

---

## File Map (14 files involved)

| File | Role | Side |
|------|------|------|
| `engine/engine.py:52-64` | Registers offloading via `--kv-transfer-config` CLI | Engine |
| `engine/kv_offload/sidecar_spec.py` | vLLM plugin entry point (`OffloadingSpec`) | Engine |
| `engine/kv_offload/sidecar_backend.py` | Block ID bookkeeping (no I/O) | Engine |
| `engine/kv_offload/sidecar_handler.py` | Async GPU-to-gRPC transfer logic | Engine |
| `engine/sidecar_cache_client.py` | Async gRPC client wrapper | Engine |
| `shared/proto/kv_cache.proto` | gRPC service + message definitions | Shared |
| `sidecar/api.py:26-75` | FastAPI lifespan (creates gRPC server) | Sidecar |
| `sidecar/grpc_server.py` | gRPC server factory | Sidecar |
| `sidecar/kv_cache_api.py` | gRPC servicer (RPC implementations) | Sidecar |
| `sidecar/cache_manager.py` | Orchestrates L1 + L2 + registry | Sidecar |
| `sidecar/l1_cache/api.py` | In-memory byte store with LRU | Sidecar |
| `sidecar/l1_cache/allocator.py` | Free-list block slot manager | Sidecar |
| `sidecar/l1_cache/eviction_policy.py` | LRU via `OrderedDict` | Sidecar |
| `sidecar/kv_block_registry.py` | Metadata catalog (location, timestamps) | Sidecar |

---

## Phase 1: System Startup — How Everything Gets Wired

### Engine side (`engine.py:52-64`)

When `enable_kv_offload` is true, the engine passes this JSON to vLLM:

```python
"--kv-transfer-config", json.dumps({
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "spec_name": "SidecarOffloadingSpec",
        "spec_module_path": "data_plane.inference.engine.kv_offload.sidecar_spec",
        "sidecar_grpc_url": config.sidecar_grpc_url,  # "localhost:50051"
        "num_blocks": config.kv_offload_num_blocks,    # 1024
    },
})
```

vLLM dynamically imports `SidecarOffloadingSpec` from the module path and calls its constructor. This creates:
1. A **`SidecarBackend`** — a local free-list of 1024 integer IDs (no network)
2. A **`SidecarCacheClient`** — opens an async gRPC channel to `localhost:50051`

Then vLLM calls:
- `spec.get_manager()` → returns `LRUOffloadingManager(SidecarBackend)` — this is what the **scheduler** uses to decide *which* blocks to offload
- `spec.get_handlers()` → yields a `SidecarOffloadingHandler` — this is what performs the *actual data movement*

### Sidecar side (`api.py:26-75`)

The FastAPI `lifespan` context manager boots the storage layer:

```
SidecarConfig loaded from env vars
        |
        |-> L1ByteStore(num_blocks=1024, block_size_bytes=131072)
        |       |-> BlockSlotAllocator(1024)   <- free-list [0..1023]
        |       |-> LRUPolicy()                <- empty OrderedDict
        |
        |-> L2Connector()                      <- Redis (placeholder)
        |
        |-> KVBlockRegistry()                  <- empty metadata dict
        |
        |-> MultiTieredCacheManager(l1, l2, registry)
                |
                |-> create_grpc_server(manager, port=50051)
                        |-> KVCacheServicer(manager)
                        |-> server.start()  <- listening on [::]:50051
```

At this point, both sides are ready. The engine has a gRPC client; the sidecar has a gRPC server.

---

## Phase 2: The STORE Path — A Block's Journey from GPU to Sidecar

Let's trace a single KV block with hash `"abc123"` being offloaded.

```
 GPU                 Engine                    Network              Sidecar
  |                    |                          |                    |
  |   1. Scheduler     |                          |                    |
  |   detects memory   |                          |                    |
  |   pressure         |                          |                    |
  |<==================>|                          |                    |
  |                    |                          |                    |
  |   2. LRUOffloading |                          |                    |
  |   Manager selects  |                          |                    |
  |   block "abc123"   |                          |                    |
  |                    |                          |                    |
  |   3. transfer_     |                          |                    |
  |   async(job_id=0,  |                          |                    |
  |    src=gpu,        |                          |                    |
  |    dst=SidecarSpec)|                          |                    |
  |                    |                          |                    |
  |  +--CUDA copy--+   |                          |                    |
  |  |GPU -> pinned|   |                          |                    |
  |  | CPU buffer  |   |                          |                    |
  |  +--(async)----+   |                          |                    |
  |                    |                          |                    |
  |   4. get_finished()|                          |                    |
  |   polls pending    |                          |                    |
  |   stores           |                          |                    |
  |                    |                          |                    |
  |   tensor -> bytes  |  5. StoreBlock RPC       |                    |
  |   serialization    |=========================>|                    |
  |                    |  block_id=42             |  6. KVCacheServicer|
  |                    |  block_hash="abc123"     |     .StoreBlock()  |
  |                    |  data=<128KB bytes>      |         |          |
  |                    |  model_id="llama"        |         v          |
  |                    |                          |  7. cache_manager  |
  |                    |                          |     .store_block() |
  |                    |                          |         |          |
  |                    |                          |         v          |
  |                    |                          |  8. l1.store()     |
  |                    |                          |  _data[42]=<bytes> |
  |                    |                          |  LRU.track_new()   |
  |                    |                          |         |          |
  |                    |                          |         v          |
  |                    |                          |  9. registry       |
  |                    |  10. StoreBlockResponse  |     .register()    |
  |                    |<=========================|  KVBlockEntry(     |
  |                    |  success=true            |   key="abc123",    |
  |                    |                          |   location="L1")   |
  |                    |                          |                    |
  |  11. returns       |                          |                    |
  |  (job_id=0, True)  |                          |                    |
```

### Step-by-step walkthrough:

**Step 1-2: Scheduler decides to offload** — vLLM's internal scheduler detects GPU memory pressure. It asks `LRUOffloadingManager` (which wraps our `SidecarBackend`) which blocks to evict. The manager picks the least-recently-used GPU blocks.

**Step 3: `transfer_async()` is called** (`sidecar_handler.py:72-105`)

```python
def transfer_async(self, job_id, src_spec, dst_spec):
    if isinstance(dst_spec, SidecarLoadStoreSpec):
        # dst is sidecar -> this is a STORE operation
        pending = _PendingStore(
            job_id=job_id,
            block_ids=dst_spec.block_ids,    # e.g. [42]
            block_hashes=dst_spec.block_hashes,  # e.g. ["abc123"]
            model_id="",
        )
        self._pending_stores.append(pending)
```

The key insight: `transfer_async` does **not** do the transfer immediately. It queues a `_PendingStore` dataclass that holds the job metadata. In production, it would also kick off an async CUDA `memcpy` from GPU to a pinned CPU staging buffer, tracked by a `torch.cuda.Event`.

**Step 4-5: `get_finished()` does the actual work** (`sidecar_handler.py:107-153`)

vLLM periodically calls `get_finished()`. For each pending store:

```python
for bid, bh in zip(store.block_ids, store.block_hashes):
    data = b"\x00" * self._block_size  # or real staging data
    ok = await self._client.store_block(
        block_id=bid, block_hash=bh, data=data, model_id=store.model_id,
    )
```

This is where the gRPC call happens. Each block is sent individually.

**Step 5 (gRPC wire):** The `SidecarCacheClient` (`sidecar_cache_client.py:34-57`) builds a protobuf `StoreBlockRequest` and sends it:

```python
resp = await self._stub.StoreBlock(
    kv_cache_pb2.StoreBlockRequest(
        block_id=block_id, block_hash=block_hash,
        data=data, model_id=model_id, layer_name=layer_name,
    ),
    timeout=5.0,
)
```

The wire format is defined in `kv_cache.proto:22-28`:
```protobuf
message StoreBlockRequest {
  int32 block_id = 1;      // 42
  string block_hash = 2;   // "abc123"
  bytes data = 3;           // 128 KB of raw tensor bytes
  string model_id = 4;     // "llama"
  string layer_name = 5;   // optional
}
```

**Step 6-7: Sidecar receives the RPC** — `KVCacheServicer.StoreBlock()` (`kv_cache_api.py:24-39`) delegates to `MultiTieredCacheManager.store_block()` (`cache_manager.py:40-60`).

**Step 8: L1ByteStore stores the bytes** (`l1_cache/api.py:70-87`):

```python
def store(self, block_id, data, block_hash):
    self._data[block_id] = data                        # dict[int, bytes]
    self._id_to_hash[block_id] = block_hash            # 42 -> "abc123"
    self._hash_to_id[block_hash] = block_id            # "abc123" -> 42
    self.eviction_policy.track_new(block_hash, len(data))  # LRU: move to end
```

Three parallel data structures are updated atomically. The `_data` dict is the actual storage — plain Python `bytes` objects in CPU DRAM.

**Step 9: Registry records metadata** (`cache_manager.py:51-59`):

```python
self.registry.register(KVBlockEntry(
    key=block_hash,        # "abc123"
    location="L1",
    size_bytes=len(data),  # 131072
    model_id=model_id,
    prefix_hash=block_hash,
    created_at=time.time(),
    last_accessed=time.time(),
))
```

**Step 10-11:** Response flows back through gRPC -> client -> handler -> scheduler.

---

## Phase 3: The LOAD Path — Bringing a Block Back to GPU

Now the scheduler needs block `"abc123"` back on the GPU:

```
 GPU                 Engine                    Network              Sidecar
  |                    |                          |                    |
  |  1. Scheduler      |                          |                    |
  |  needs block       |                          |                    |
  |  "abc123" back     |                          |                    |
  |                    |                          |                    |
  |  2. transfer_      |                          |                    |
  |  async(job_id=1,   |                          |                    |
  |   src=SidecarSpec, |                          |                    |
  |   dst=gpu)         |                          |                    |
  |                    |                          |                    |
  |  3. get_finished() |  4. LoadBlock RPC        |                    |
  |  polls pending     |=========================>|                    |
  |  loads             |  block_id=42             |  5. l1.load(42)   |
  |                    |                          |  _data.get(42)     |
  |                    |                          |  LRU.record_access |
  |                    |                          |  registry.record   |
  |                    |  6. LoadBlockResponse     |  _access("abc123")|
  |                    |<=========================|                    |
  |                    |  success=true            |                    |
  |                    |  data=<128KB bytes>      |                    |
  |                    |                          |                    |
  |  7. bytes -> tensor|                          |                    |
  |  +--CUDA copy--+   |                          |                    |
  |  |pinned CPU ->|   |                          |                    |
  |  |   GPU HBM   |   |                          |                    |
  |  +-------------+   |                          |                    |
  |                    |                          |                    |
  |  8. returns        |                          |                    |
  |  (job_id=1, True)  |                          |                    |
```

### Step-by-step:

**Step 2:** `transfer_async` detects `src_spec` is a `SidecarLoadStoreSpec`, so it creates a `_PendingLoad`:

```python
elif isinstance(src_spec, SidecarLoadStoreSpec):
    pending = _PendingLoad(job_id=job_id, block_ids=src_spec.block_ids)
    self._pending_loads.append(pending)
```

**Step 3-4:** `get_finished()` processes pending loads:

```python
for bid in load.block_ids:
    data = await self._client.load_block(bid)  # gRPC call
    load.staging_data.append(data)
```

**Step 5:** The L1ByteStore does an O(1) dict lookup (`l1_cache/api.py:89-105`):

```python
def load(self, block_id):
    data = self._data.get(block_id)       # O(1) lookup
    block_hash = self._id_to_hash.get(block_id)
    self.eviction_policy.record_access(block_hash)  # move to end of LRU
    return data
```

The `record_access` call is critical — it moves this block to the **end** of the `OrderedDict`, marking it as most-recently-used:

```python
# eviction_policy.py:30-32
def record_access(self, key):
    if key in self._cache_map:
        self._cache_map.move_to_end(key)  # <- protects from eviction
```

**Step 6-7:** Bytes flow back through gRPC and get copied to GPU via CUDA staging.

---

## Phase 4: Eviction — When L1 is Full

When all 1024 slots are occupied and new blocks need to be stored:

```
allocate_blocks(["new_hash_1", "new_hash_2"])
        |
        v
 needed = 2 - 0 = 2  (0 free slots)
        |
        v
 +------------------------------+
 |  LRU OrderedDict             |
 |                               |
 |  "oldest"  --> "middle" --> "newest"    <- recency order
 |  (victim)                               |
 +------------------------------+
        |
        v
 select_victim() -> "oldest"     <- first item in OrderedDict
        |
        v
 _evict(victim_id):
   |-- del _data[victim_id]      <- free the bytes
   |-- del _id_to_hash[victim_id]
   |-- del _hash_to_id["oldest"]
   |-- eviction_policy.remove("oldest")
   +-- allocator.free(victim_id) <- return ID to free-list
        |
        v
 needed = 1  -> repeat for "middle"
        |
        v
 needed = 0  -> allocate 2 fresh slots
```

This happens inside `L1ByteStore.allocate_blocks()` (`l1_cache/api.py:46-68`):

```python
def allocate_blocks(self, block_hashes):
    needed = len(block_hashes) - self.allocator.num_free
    while needed > 0:
        victim_hash = self.eviction_policy.select_victim()  # first in OrderedDict
        victim_id = self._hash_to_id.get(victim_hash)
        self._evict(victim_id)           # remove from all data structures
        l1_metrics.l1_cache_evictions_total.labels(reason="capacity").inc()
        needed -= 1
    ids = self.allocator.allocate_n(len(block_hashes))  # pop from free-list
    for block_id, block_hash in zip(ids, block_hashes):
        self._id_to_hash[block_id] = block_hash
        self._hash_to_id[block_hash] = block_id
    return ids
```

---

## The gRPC Protocol — All 5 RPCs

Defined in `shared/proto/kv_cache.proto`:

```
+--------------------+--------------------------------------------+
|                    KVCacheService                               |
+--------------------+--------------------------------------------+
| RPC                | Purpose                                    |
+--------------------+--------------------------------------------+
| StoreBlock         | Engine -> Sidecar: send block bytes         |
| LoadBlock          | Engine <- Sidecar: retrieve block bytes     |
| GetFreeBlocks      | Query how many slots are available          |
| AllocateBlocks     | Reserve slots by hash (returns block IDs)   |
| FreeBlock          | Release a single slot                       |
+--------------------+--------------------------------------------+
| Transport          | Insecure channel (no TLS)                   |
| Port               | 50051 (configurable)                        |
| Max message        | 16 MB (both directions)                     |
| Timeout            | 5 seconds per RPC                           |
| Pattern            | All unary-unary (request -> response)       |
+--------------------+--------------------------------------------+
```

---

## The L1 Data Structures — 5 Parallel Maps

```
                      L1ByteStore internals
  +----------------------------------------------------------+
  |                                                          |
  |  _data: dict[int, bytes]          BlockSlotAllocator     |
  |  +-----+--------------+           +----------------+    |
  |  |  42 | b"\x00..."   |           | _free_ids:     |    |
  |  |  17 | b"\xff..."   |           |  [0,1,3,5,...] |    |
  |  | 999 | b"\xab..."   |           | _allocated:    |    |
  |  +-----+--------------+           |  {42,17,999}   |    |
  |                                   +----------------+    |
  |  _id_to_hash: dict[int, str]                            |
  |  +-----+--------------+     _hash_to_id: dict[str, int] |
  |  |  42 | "abc123"     |     +-----------+-----+         |
  |  |  17 | "def456"     |     | "abc123"  |  42 |         |
  |  | 999 | "ghi789"     |     | "def456"  |  17 |         |
  |  +-----+--------------+     | "ghi789"  | 999 |         |
  |                             +-----------+-----+         |
  |  LRU OrderedDict                                        |
  |  +---------------------------------------+              |
  |  | "def456" -> "ghi789" -> "abc123"      |              |
  |  |  oldest               most recent     |              |
  |  +---------------------------------------+              |
  |                                                          |
  +----------------------------------------------------------+
```

---

## Deduplication — The Hash Trick

`SidecarBackend.allocate_blocks()` (`sidecar_backend.py:72-86`) detects duplicates:

```python
for bh in block_hashes:
    if bh in self._hash_to_id:
        ids.append(self._hash_to_id[bh])  # reuse existing slot!
        continue
    block_id = self._free_ids.pop()       # new allocation
```

If two sequences share the same prefix, their KV blocks have the same hash. The system returns the **existing block ID** instead of allocating a new slot — zero-copy deduplication.

---

## Gotchas and Important Details

**1. Two separate block ID free-lists exist.** The `SidecarBackend` (engine-side) and `BlockSlotAllocator` (sidecar-side) both track block IDs independently. They must stay in sync. If the engine thinks block 42 is allocated but the sidecar disagrees, the `store()` call will fail with an `is_allocated` check error at `l1_cache/api.py:73`.

**2. Evicted blocks are currently *lost*, not demoted to L2.** The `MultiTieredCacheManager` has an `L2Connector` slot, but eviction from L1 does not push to L2 — the block's bytes are simply deleted. The L2 path is scaffolded but not wired for automatic spillover.

**3. gRPC calls are synchronous from the handler's perspective.** Even though the channel is `grpc.aio`, the handler `await`s each `store_block`/`load_block` call sequentially inside `get_finished()`. Blocks within a single job are sent one-by-one in a loop, not as a batch/stream.

**4. The 16 MB gRPC limit vs. 128 KB block size.** Default blocks are 128 KB, well within the 16 MB limit. But if `block_size_bytes` is configured larger (e.g., multi-layer blocks), you could hit the message ceiling silently — the gRPC error would surface as a generic `RpcError` caught at `sidecar_cache_client.py:55`.

**5. `transfer_async` doesn't actually transfer.** The name is misleading — it only *queues* the intent. The real work happens in `get_finished()`, which is polled by vLLM's scheduler loop. This two-phase design allows the scheduler to continue scheduling while transfers complete asynchronously.
