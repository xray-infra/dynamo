<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native terminal cache usage envelope regression

## Scope and provenance

This is the narrow legacy-artifact maintenance repair tracked by
[issue #14](https://github.com/xray-infra/dynamo/issues/14). The reviewed source
snapshot is `13c50272d53f1d22e9ae91df4e07d301f038815d` from closed, unmerged
[PR #11](https://github.com/xray-infra/dynamo/pull/11). Its full custom cache
authority/signed-receipt integration is abandoned; this repair does not reopen,
complete or reintroduce it into `main`.

Fork `main` at `8907f0110e554b8748c50120602b65dba179a259` has no defective native
producer. The maintenance Draft PR therefore uses a frozen old-source review
base solely to expose this small diff, not a new integration line. It fixes a
frontend-produced usage-only frame, not vLLM counters or the receipt verifier.
The whole source snapshot has not been matched to the deployed binary; this is
not permission to rebuild it and replace a running artifact.

The producer stores usage and identity from the **same real native chunk**,
then projects `id/object/created/model`, optional `system_fingerprint/service_tier`
and `choices=[]`. It does not replay choices, tools, internal metrics or opaque
chunk metadata. It preserves native completion/reasoning counters and continues
to call the unchanged verified-accounting validation/overlay function.
A missing/malformed required envelope fails closed. A null usage update cannot
resurrect earlier counters. Unknown cache accounting is not zero billing.

Protocol ownership: the native producer owns this synthetic completion envelope;
Chat, Messages and Responses own their respective external projection. No
protocol DTO is relaxed, no worker payload/configuration is changed, and no
receipt or cache-authority validation is bypassed.

N-2 compatibility: this is an internal producer/API change, not a new worker
field requirement or a version-dependent shim. Existing required OpenAI stream
identity fields are reused. Optional identity fields may remain absent. Ordinary
streams without an explicit cache guard and unary responses keep their existing
paths. Actual N-2 worker/frontend combinations have not been tested.

## Bounded producer gate

From the repository root, with a working Rust toolchain:

```sh
python3 scripts/check_native_cache_terminal_usage.py
# When dependencies are already cached:
python3 scripts/check_native_cache_terminal_usage.py --offline
```

The gate extracts production functions and synchronous regression functions
verbatim, imports the production cache-accounting module, and uses the
workspace-pinned `dynamo-protocols` **base** stream DTO rather than the Nv wrapper.
Only the surrounding module wiring/error constructors are substituted; it does
not test error classification, the Nv extensions, SSE codecs, HTTP handlers,
worker dispatch or signed receipts. It prints source hashes for provenance.
Temporary Cargo files do not modify the repository lockfile.

Observed on Rust 1.96.1: **12 tests passed**. These include text/tool × cold/hot ×
5m/1h synthetic accounting fixtures, absent optional identity, missing/malformed
required identity, unknown accounting, stale/null usage, token consistency and
reasoning-detail preservation. These are local fixtures, not live TTL tests.

Before the repair the envelope regression failed with `missing field id`
(6 other tests passed). After repair, independently dropping each of the six
carried identity fields from the production projection in temporary copies
caused the regression gate to fail: **6/6 targeted mutations killed**.
The optional-field mutations fail preservation, not required-field parsing.

## Full-crate protocol converter gate — blocked

```sh
cargo test -p dynamo-llm --no-default-features --lib native_cache_terminal_usage_tests
```

The registered test module exercises the real Nv completion wrapper and real
Anthropic/Responses stream converters: no error, terminal events exactly once,
native output counters preserved and tool blocks not replayed. TTL projection is
asserted at the separate native Messages usage helper seam; this does **not**
claim that the public handler applied verified TTL accounting end-to-end.

The PR #11 base and patched crate both fail compilation with the same 13 errors:
missing `hyper`/`hyper_util`, missing service `ErrorMessage`, missing Anthropic
`native_messages_to_chat` export, three native metrics methods, and State
`seal`/`resume`/`is_sealed` methods. The root lockfile also requires regeneration
before `--locked` builds; no dependency/lockfile change is part of this patch.
Therefore the full-crate converter tests have **not run**. This repair does not
silently substitute no-op implementations for the missing integration pieces.

macOS verification used a rustup toolchain instead of the broken Homebrew
Rust/LLVM ABI pairing and standalone `protoc` instead of Homebrew's broken
abseil linkage. Rust-analyzer additionally reported bool/slice errors that the
CLI did not report; LSP output is not treated as a clean compile.

Before any rollout on the still-running legacy artifact: match the actual complete
build source, compile the complete frontend, run both converter and public HTTP
tests (stream/unary, text/tool, cold/hot, 5m/1h where supported), and verify final
usage and protocol termination. This does not require completing or reviving
PR #11. The separate minimal native/HiCache/NIXL replacement work is not changed.
No production service, model configuration or remote spreadsheet was changed.
