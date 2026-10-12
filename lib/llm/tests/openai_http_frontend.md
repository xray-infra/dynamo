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

Worker requests always request native final usage, including when a public unary
request is internally streamed. Public OpenAI usage behavior remains governed by
`stream_options.include_usage`. Required completion identity is decoded directly
from the backend. Invalid chunks, transport errors, premature EOF, duplicate
choice terminals, oversized events, and DONE without every choice terminal are
errors, not fabricated successful completions. There is no inference retry after
an ambiguous backend response.

Anthropic projection preserves explicitly known cache-read zero while absent
native cache-read remains absent. Input tokens exclude the native cached subset.
Cache-write behavior is the upstream OpenAI-to-Anthropic policy: OpenAI's typed
usage has no cache-write count and upstream emits `cache_creation_input_tokens=0`.
This is not proof of a backend-created-cache extension, explicit 5m/1h TTL, signed
accounting, or physical cache residency. Those retired features are not implemented
or claimed by this adapter; live cache billing/residency still needs separate
verification against actual workers.

## CPU regression (actual PR code)

```sh
cargo test --locked -p dynamo-llm --no-default-features --lib openai_http_engine
cargo test --locked -p dynamo-llm --no-default-features --lib protocols::anthropic
cargo test --locked -p dynamo-llm --no-default-features --test openai_http_transport
```

The Rust integration test starts the compiled frontend and a loopback worker
fixture. It covers text/tool × unary/streaming × OpenAI/Anthropic (eight cases),
native usage/envelope fidelity, disjoint cached/input accounting, tool name/ID/JSON
arguments, fragmented mixed LF/CRLF transport, and actual protocol terminal events.
All eight forms assert that a public tenant header overrides the body namespace
and reaches the worker's native `cache_salt` field. A library regression covers
canonical/legacy precedence, empty canonical fallback and absent salt.
The fixture cannot prove GPU prefix-cache hits, model behavior, live billing,
explicit cache-control support, or deployment readiness. No mock integration is
used as a substitute for a live canary.

Production replacement must be built from this PR's committed source and pass
bounded real-worker tests before Flash routing changes. Keep the original image
and config for rollback; do not change other models, GPU processes, weights, VPN,
or shared discovery state.
