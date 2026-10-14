# Issue #29: cache/load-aware routing for GLM-5.3 all-PD

See the complete research artifact at `/Users/auturora/.pi/agent/sessions/--private-tmp--/subagent-artifacts/outputs/1d57b04e-d3d7-4057-8d72-a538c989342a/task-1.md`.

## Resolution
Use **one authoritative Dynamo frontend/KV router**, not a second SGLang/Wayfinder router. Extend/configure Dynamo’s existing routing primitives to combine exact KV-event prefix overlap with projected prefill pending-token load and atomic admission. The selected P worker performs prefill and KV transfer; Dynamo then selects D by decode load, transfer compatibility, and topology. D should not chase prefix affinity unless transferred blocks are proven reusable.

Use server-originated KV events as cache truth; treat event state as eventually consistent and degrade stale/missing state to load-aware routing. P admission should reserve predicted prefill tokens with Dynamo’s existing booking/lease primitives. Dynamo, not SGLang Router, owns retries: retry only before irreversible handoff/streaming, avoid retry loops, and surface no-capacity failures.

## Sources

- [Dynamo Router overview](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/router/overview)
- [Dynamo disaggregated serving](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/router/disaggregated-serving)
- [Dynamo kv_router.rs](https://github.com/ai-dynamo/dynamo/blob/main/lib/llm/src/kv_router.rs)
- [llm-d precise-prefix routing](https://llm-d.ai/docs/well-lit-paths/foundations/precise-prefix-cache-routing)
- [llm-d precise-prefix guide](https://github.com/llm-d/llm-d/blob/main/guides/precise-prefix-cache-routing/README.md)
- [vLLM Production Stack load-aware routing](https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/loadaware-routing.html)
- [Ray prefix-aware routing](https://docs.ray.io/en/latest/serve/llm/user-guides/prefix-aware-routing.html)
- [SGLang Model Gateway](https://sgl-project.github.io/advanced_features/sgl_model_gateway.html)

The complete artifact contains the comparison table, detailed P/D design, risks, missing evidence, and validation plan. No production code or cluster was modified.

> Git commit/push and GitHub issue mutation require shell/GitHub tools unavailable to this subagent.

```acceptance-report
{"criteriaSatisfied":[{"id":"criterion-1","status":"satisfied","evidence":"Report written with source-linked comparison and concrete architecture recommendation covering all requested routing concerns."}],"changedFiles":["docs/research/issue-29-cache-load-aware-pd-router.md"],"testsAddedOrUpdated":[],"commandsRun":[],"validationOutput":["Research artifact written; official docs and Dynamo source inspected."],"residualRisks":["Exact GLM-5.3 deployment compatibility and GitHub mutation remain unverified."],"noStagedFiles":true,"diffSummary":"Added research report pointer and resolution summary.","reviewFindings":["no blockers"],"manualNotes":"Full report is in the authoritative subagent artifact path."}
```

## Contradictions
None found among kept primary sources; exact deployment evidence remains missing.

## Missing evidence
Exact GLM-5.3 cluster topology, deployed fork commits, and event/transfer compatibility require local deployment validation.