<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OpenAI HTTP worker transport

The `openai-http-frontend` binary registers an HTTP `StreamingEngineAdapter` with
upstream `ModelManager` and `HttpService`. The existing upstream handlers own
request validation, tools, unary aggregation, Chat Completions, Responses, and
Anthropic Messages conversion. The adapter only transports native SSE chunks;
there is no private authentication, signed receipt, TTL/cache authority, synthetic
usage chunk, deployment-proof reader, or discovery-card emulation.

Configuration is an explicitly supplied JSON file:

```json
{
  "listen": "127.0.0.1:30302",
  "model": "glm-5.3-flash",
  "workers": ["http://127.0.0.1:28100"]
}
```

Only HTTP(S) origins are accepted for workers. Credentials, paths, query strings,
fragments, redirects, empty worker lists, and unknown config keys are rejected.
The binary enables upstream Chat Completions and Messages routes and disables
frontend admin mutation routes. SIGINT/SIGTERM drains in-flight HTTP requests
before stopping the listener. It does not stop/reconfigure workers or load weights.

The adapter forwards the canonical cache salt as the HTTP worker's top-level
`cache_salt`. It uses upstream precedence (`x-tenant-id` header override,
`nvext.cache_salt`, then legacy top-level fallback), rather than leaving namespace
isolation hidden inside an unrecognized worker extension. The header is not an
authentication mechanism: an API gateway must enforce which tenant identity a
caller is allowed to use. The adapter does not add a private authority scheme.

An explicit OpenAI `prompt_cache_key` is preserved from Chat Completions and
Responses and provides deterministic rendezvous affinity, scoped by model and
cache namespace. Responses conversion normalizes the same non-empty
`nvext.cache_salt` / legacy root `cache_salt` precedence as Chat before dropping
unrecognized options. Header tenant override remains owned by ingress. Requests
without a key retain round-robin routing. This is a
routing hint, not real-time KV-residency discovery, load-aware scheduling or a
cache-hit guarantee. Successful inference is never retried after an ambiguous
worker response.

Unsupported behavioral controls are rejected at their existing semantic owners:
Messages accepts block-level Anthropic `cache_control` with `5m`/`1h` TTLs and lowers
proven rendered boundaries into exact worker-token prefix counts, while rejecting
ambiguous, unsupported, or malformed controls. Responses conversion rejects
unrepresentable `prompt_cache_retention`, and the HTTP backend rejects Chat retention
requests. Ordinary automatic prefix caching remains available.
OpenAI error responses are emitted directly by the existing error DTO as an
`error` envelope with a string code; the error reader accepts both that envelope
and legacy flat worker errors.

Worker requests always request native final usage, including when a public unary
request is internally streamed. Public OpenAI usage behavior remains governed by
`stream_options.include_usage`. Required completion identity is decoded directly
from the backend. Invalid chunks, transport errors, premature EOF, duplicate
choice terminals, oversized events, and DONE without every choice terminal are
errors, not fabricated successful completions. There is no inference retry after
an ambiguous backend response.

Native cache-write accounting is optional, not a derived prefill-miss count.
At the HTTP worker boundary, vLLM 0.26's
`usage.prompt_tokens_details.created_cache_tokens` is normalized to the SDK's
`cache_write_tokens`; a supplied canonical field wins, including explicit null.
Absent accounting stays absent and measured zero stays zero. Negative, non-integer
or greater-than-prompt write counts are rejected. Retire the input alias when
these worker revisions leave the supported N-2 window.

Chat unary/stream serialization preserves that canonical field. A bounded output
shim compensates for dynamo-protocols 8.1.0's usage shadow dropping it; remove the
shim after upgrading to a dependency that preserves the field. It borrows the
response envelope/choices and materializes only the fixed-size usage object.
Responses unary/stream conversion projects the count to
`usage.input_tokens_details.cache_write_tokens`. Anthropic unary/stream conversion
projects it to `cache_creation_input_tokens`; `input_tokens` excludes both native
read and created subsets. Neither projection invents a write count when absent.
These counters do not by themselves prove TTL, durable residency or a monetary
bill; live worker accounting semantics and authoritative billing require separate
verification.

## CPU regression (actual PR code)

```sh
cargo test --locked -p dynamo-llm --no-default-features --lib openai_http_engine
cargo test --locked -p dynamo-llm --no-default-features --lib protocols::anthropic
cargo test --locked -p dynamo-llm --no-default-features --test openai_http_transport
```

The Rust integration test starts the compiled frontend and a loopback worker
fixture. It covers text/tool × unary/streaming × Chat/Responses/Anthropic (12 cases),
native read/write usage and envelope fidelity, disjoint input/read/write accounting,
tool name/ID/JSON arguments, fragmented mixed LF/CRLF transport, and actual protocol
terminal events. All 12 forms assert that a public tenant header overrides the body namespace
and reaches the worker's native `cache_salt` field. A library regression covers
canonical/legacy precedence, empty canonical fallback and absent salt.
The fixture proves public block-level cache-control acceptance but cannot prove
worker-side TTL execution, GPU prefix-cache hits, model behavior, live billing, or
deployment readiness. No mock integration is used as a substitute for a live canary.

Production replacement must be built from committed source and pass bounded
real-worker tests before Flash routing changes. Retired private implementations
must not be restored as rollback targets. Delete confirmed stopped, unreferenced
legacy containers; retain current mounted artifacts. Worker-source replacement
requires its own qualified candidate and authorized rolling maintenance; do not
change other models, weights, VPN or shared discovery state.
