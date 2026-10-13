<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native cache-write accounting — issue #14

## Correction and observed source contract

The earlier claim that the current Flash worker lacked native write-token
accounting was incorrect. Direct worker SSE captured on 2026-10-13 contains:

| Request | Prompt | Cached | `created_cache_tokens` | Output |
| --- | ---: | ---: | ---: | ---: |
| Fresh namespace / cold prefix | 4942 | 0 | 4096 | 2 |
| Same namespace / warm prefix | 4942 | 4096 | 0 | 2 |

Raw evidence is retained in the existing operations receipt
`remaining-live-acceptance-1791858336/native-write-proof-{cold,warm}.sse`
and the corresponding `.usage.json` files. No write tokens were inferred by the
frontend from bytes, prompt misses or a tariff.

The current vLLM 0.26.1rc0 Flash worker emits this optional field through
`chat_completion/serving.py::_make_prompt_tokens_details()`, sourced from
`RequestOutput.num_cache_creation_tokens`. The output processor obtains that
value from `PrefillStats`, finalized by the scheduler with
`KVCacheManager.estimate_cached_tokens(request)`. That method measures block-hash
prefix progress conservatively across participating cache groups; `finalize()`
subtracts the request's initial cached subset and caps progress at the prompt.
This is the worker's local prefix-cache write accounting, not a durable-filesystem
write receipt, an expiry guarantee, or a monetary invoice.

## Where the frontend lost the count

- The HTTP decoder ignored the worker's `created_cache_tokens` spelling; the SDK
  field is `PromptTokensDetails.cache_write_tokens`.
- Both Responses converters explicitly assigned `cache_write_tokens: None`.
- Deployed revision `838c180` assigned Anthropic creation zero. Main revision
  `f3666da` correctly omitted unknown creation but still did not project a
  supplied native write count.
- `dynamo-protocols` 8.1.0's Chat usage serialization shadow dropped the SDK field
  even after the typed decoder retained it.

The fix normalizes the legacy input alias once, preserves the typed count in Chat
serialization, and projects it through Responses and Anthropic unary/stream
paths. Canonical input wins, including explicit null. Unknown remains omitted;
known zero remains zero. Anthropic input excludes both the read and created
subsets. A cold count of 4096 therefore projects to input 846, read 0, creation
4096, not input 4942 plus another 4096.

The dependency output shim borrows the completion envelope and choices and
materializes only the fixed-size usage object. It does not reparse generated
content. Remove it when the dependency preserves the field; remove the input
alias when these legacy workers leave the supported N-2 window.

## Executed qualification and limits

- Old compiled frontend: the updated real-HTTP fixture failed because Chat usage
  lacked the supplied native write count (expected 24).
- Fixed frontend: 12 loopback HTTP cases passed, covering Chat/Responses/Messages
  × text/tool × unary/stream, native counters, parsed tool arguments and terminal
  events. The fixture supplies prompt 100, read 60, write 24; Anthropic input is 16.
- macOS locked/offline tests: engine 5, Anthropic 106, OpenAI 676, HTTP integration
  1 passed; 10 OpenAI tests intentionally ignored. Formatting and Clippy passed
  with existing warnings. LSP coverage was inconclusive, not a clean qualification.
- The first integration wrapper run failed only because its expected case count
  was still eight; the updated 12-case assertion passed.

These are source and CPU/HTTP mapping results, not production deployment results.
A committed Linux build and a bounded live candidate are still required before
public replacement and public write-count acceptance. Public serving remained on
`838c180` during these checks.

Restart readback failed in the prior live experiments; TTL controls remain
unsupported. Authoritative tariffs and invoice/ledger reconciliation remain
unqualified. Neither the raw worker field nor this mapping fix closes those gates.
