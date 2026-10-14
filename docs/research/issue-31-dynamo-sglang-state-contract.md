# Wayfinder issue #31: minimum Dynamo↔SGLang state contract

## Scope and conclusion

This is a local-code investigation only. The minimum useful v1 contract is a **versioned, worker-incarnation-scoped snapshot plus ordered KV events**, not a collection of independent health probes. Dynamo can make unified PD decisions only when it can distinguish a live worker incarnation, bounded-freshness load, exact cache ownership, and P→D transfer admission. SGLang currently exposes useful ingredients, but no single authoritative endpoint combines them; this remains an unresolved contract question and issue #31 should not be closed.

## Existing local evidence

1. **Dynamo HTTP transport is not telemetry.** `lib/llm/src/openai_http_engine.rs` does rendezvous affinity from `prompt_cache_key` (or a hash of prompt-forming fields), explicitly calls affinity only a routing hint, and sends `cache_salt` to workers. It does not learn residency, worker epoch, queue state, or transfer state from responses.
2. **Dynamo worker discovery is mostly static metadata.** `components/src/dynamo/planner/monitoring/worker_info.py` carries endpoint/model/capacity (`total_kv_blocks`, block size, max sequences/batched tokens, context length), sourced from MDC/DGD/defaults. It has no incarnation, heartbeat TTL, readiness state, queue snapshot, or freshness field. Frontend config has discovery-plane selection, but not a unified serving-state schema.
3. **Dynamo KV authority is event/ownership based.** `lib/kv-router/src/protocols.rs` defines `KvCacheEvent { event_id, data, dp_rank }`, with Stored/Removed/Cleared; Stored has parent hash, position, and blocks, and Remove has block hashes. `ResidencyProjection` is an immutable snapshot from exact cache owners to workers; its comments distinguish advisory router metadata from persistent cache ownership. Listener code (`services/indexer/listener.rs`) gates attempts by a generation and rejects stale attempts. This is the right foundation for incarnation and stale-update handling, but event IDs are only documented as ordered within a publisher sequence, not as a worker-restart-safe global epoch.
4. **SGLang has a real KV event descriptor.** `python/sglang/srt/runtime_context.py::describe_kv_events_publisher` publishes endpoint host/port base, topic, block size, DP size, and optional load/replay ports under `/server_info`. It requires valid page size and routable TCP configuration. Consumers must use advertised block size and resolved load base; wildcard hosts are replaced with the worker URL host.
5. **SGLang scheduler already computes the core load signals.** `python/sglang/srt/managers/scheduler_components/kv_events_publisher.py` reports active request slots, total slots, active/total KV blocks, waiting requests, cache usage, recent prefix-cache hit rate, and DP rank. `scheduler.py` wires load inquiry to running batch, waiting queue, prefix matches, uncached prefill tokens/busy time, and disaggregation queues: prefill bootstrap/inflight and decode preallocation/transfer. These are implementation signals, not yet a stable Dynamo-facing schema.
6. **SGLang health is liveness only.** The dummy health server returns HTTP 200 for `/health` and `/health_generate`; this does not mean Ready for admission, nor does it report Busy/Draining/Unhealthy, queues, cache, or transfer state. TokenizerManager owns request ingress and disaggregation service setup, while scheduler owns admission and queues; state must be sampled from the scheduler, not inferred from HTTP health.
7. **SGLang transfer paths are queue/stateful.** Scheduler load inquiry receives disaggregation bootstrap, inflight, decode-prealloc, and decode-transfer queues. `srt/disaggregation/utils.py` explicitly distinguishes staging and incomplete transfers. Therefore a P→D worker needs transfer state and capacity reservation in the contract; a health check or cache-hit rate is insufficient.

## Proposed minimum v1 wire contract

A single `WorkerState` snapshot, periodically published and also returned by discovery/server-info, should contain:

- `schema_version` and `worker_id` (stable logical identity), `incarnation_id`/`epoch` (new random/monotonic value on every process start), role (`aggregated|prefill|decode`), model identity, DP rank/size, and timestamp.
- `state`: `Ready | Busy | Draining | Unhealthy`; `Ready` means local admission is enabled and the worker passed readiness, not merely HTTP liveness. Include `admission_enabled` and a **heartbeat TTL/lease expiry** (or `last_heartbeat` plus agreed TTL). Missing/expired heartbeat makes state Unhealthy and invalidates residency attachment for that incarnation.
- Load snapshot: `running_requests`, `waiting_requests`, `pending_prefill_tokens`, `pending_decode_tokens`, and `queue_wait` (define units and whether p50/max/oldest; v1 should use oldest wait or a clearly defined scalar). Include capacities (`max_running_requests`, `max_total_num_tokens`, KV total blocks/tokens) so values are comparable across workers. Values must be one atomic snapshot with `sample_ts` and `valid_until`/sequence.
- Cache: block size/hash semantics, cache event stream identity `(worker_id, incarnation_id, dp_rank, stream_generation)`, monotonic `event_id`, and event watermark/`event_ts`; event kinds Stored/Removed/Cleared. A snapshot needs `cache_residency_generation`/watermark and freshness. Cache hit rate and aggregate usage are advisory load signals, never proof that a requested prefix resides locally. `Cleared` (or lease expiry/restart) removes all prior ownership.
- P→D: role, request/transfer identifier, source and destination incarnation, token count, and state `Queued | Admitted | Staging | InFlight | Complete | Failed | Cancelled`; include reserved/available local KV capacity. The scheduler must not admit decode based on a stale “cache stored” event before transfer Complete.
- Ordering/restart: all snapshots/events carry incarnation and a monotonic per-incarnation sequence. Consumers discard older sequences, reject events from old incarnations, and require a new Ready snapshot before accepting events after restart. On restart publish/implicitly apply `Cleared`; never let old KV ownership survive a new epoch. A `Draining` worker stops new admission but may finish existing requests/transfers; `Unhealthy` expires immediately and is removed from scheduling.

### Suggested v1 semantics

- Dynamo is authoritative for **worker membership, lease expiry, and scheduling projection**; SGLang is authoritative for **local scheduler queues/counters, admission bit, and cache event production**. Persistent KV ownership remains authoritative for exact residency where configured; telemetry never upgrades a hint into proof.
- A state update is accepted only when `(worker_id, incarnation_id)` matches the currently attached worker and its sequence is newer; event streams are accepted only for the matching stream generation. Out-of-order or late updates are ignored and counted as telemetry errors.
- Freshness is explicit: scheduler state older than its TTL is unusable for load-aware routing. The report should choose concrete TTL/heartbeat intervals in a follow-up performance experiment; code inspection cannot justify numbers.
- Request `queue_wait` and pending token fields must be defined from enqueue/admission timestamps and token accounting, not reconstructed by Dynamo from logs. v1 should avoid using cache hit rate as a control-plane predicate.

## v1 versus later online calibration

**v1 (hard correctness/admission):** worker identity/incarnation; role/model; lease/heartbeat TTL; Ready/Busy/Draining/Unhealthy and local admission; atomic running/waiting counts; pending prefill/decode token counts; one queue-wait scalar; capacities; cache block/hash semantics, event generation/watermark/freshness, Stored/Removed/Cleared; P→D transfer lifecycle and reservation; sequence numbers, stale/out-of-order rejection, and restart clearing.

**Later online calibration (optimization only):** per-worker service-rate/EWMA, queue-wait distributions and tail percentiles, prefill/decode throughput, token-level latency, cache-hit/prefix-match prediction, transfer bandwidth/ETA, adaptive TTLs, and learned cost models. These may improve ranking but cannot override v1 admission, lease, epoch, transfer-complete, or exact-residency rules.

## Unresolved questions / risks

- No local path currently defines a Dynamo↔SGLang canonical `WorkerState` schema or assigns an incarnation epoch. `/health` is too weak, and `/server_info` is mostly static plus KV endpoint descriptors.
- SGLang's `KvMetrics` has no worker epoch, event stream generation, timestamp/watermark, or explicit freshness. `KVEventBatch` has a timestamp, but restart scoping and replay/watermark semantics need an explicit cross-project decision.
- The exact definition of pending prefill/decode tokens and `queue_wait` across chunked prefill, speculative decoding, batching, and disaggregation is not established by these paths.
- P→D lifecycle ownership (source scheduler, destination scheduler, or Dynamo transfer coordinator), request ID correlation, timeout, and cancellation semantics remain to be specified.
- Heartbeat/TTL values and whether Dynamo or SGLang owns state transitions require an interoperability test; this report intentionally does not guess numeric values.

## Files inspected

- Dynamo: `lib/llm/src/openai_http_engine.rs`; `components/src/dynamo/frontend/frontend_args.py`; `components/src/dynamo/planner/monitoring/worker_info.py`; `components/src/dynamo/vllm/kv_hints.py`; `lib/kv-router/src/protocols.rs`; `lib/kv-router/src/services/indexer/listener.rs`.
- SGLang fork: `python/sglang/srt/managers/tokenizer_manager.py`; `python/sglang/srt/managers/scheduler.py`; `python/sglang/srt/managers/scheduler_components/kv_events_publisher.py`; `python/sglang/srt/runtime_context.py`; `python/sglang/srt/disaggregation/utils.py`.

No production code or cluster was modified by this investigation.

## Recommendation

Keep issue #31 open. First agree and test the `WorkerState`/event envelope (epoch, sequence, freshness, restart) and P→D lifecycle semantics; then implement the smallest adapter at SGLang scheduler/server-info and Dynamo discovery/cache authority boundaries. Add online calibration only after correctness under restart, delayed messages, stale cache events, draining, and transfer failure is demonstrated.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Concise local-code findings, proposed minimum v1 contract, later calibration split, and unresolved risks are recorded above."
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [],
  "validationOutput": ["Inspected the specified Dynamo and SGLang paths only; no production code or cluster was modified."],
  "residualRisks": ["Canonical wire schema, heartbeat TTL values, event restart generation, and P→D ownership/lifecycle remain unresolved."],
  "noStagedFiles": true,
  "diffSummary": "Research-only report; no production implementation.",
  "reviewFindings": ["no blockers"],
  "manualNotes": "The requested repository report path is overridden by the runtime artifact path; this artifact is authoritative for the run."
}
```