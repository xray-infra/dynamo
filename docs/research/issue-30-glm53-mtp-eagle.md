# Wayfinder issue #30: GLM-5.3 NVFP4 speculative decoding research

**Status (2026-08-12): research recommendation; no production change.** This report addresses the deployed `RadixArk/GLM-5.3-Flash-NVFP4` path, not a generic GLM/EAGLE setup.

## Resolution

Use the checkpoint's **built-in MTP/NextN layer** through SGLang's `NEXTN` algorithm, not an external EAGLE draft checkpoint. The RadixArk model card explicitly says it contains a native MTP/NextN draft layer and that all 889 layer-45 MTP tensors are present in the checkpoint index. Its reference launch is:

```bash
--model-path RadixArk/GLM-5.3-Flash-NVFP4 \
--quantization modelopt_fp4 --tp-size 4 \
--dsa-prefill-backend trtllm --dsa-decode-backend trtllm \
--kv-cache-dtype fp8_e4m3 --moe-runner-backend flashinfer_cutlass \
--speculative-algorithm NEXTN --speculative-num-steps 5 \
--speculative-eagle-topk 1 --speculative-num-draft-tokens 6 \
--speculative-adaptive --reasoning-parser glm45 --tool-call-parser glm47
```

For the local Dynamo recipe, the safer initial canary is the NVIDIA/Dynamo-tested profile: EAGLE-style self-MTP with **draft length 3** (SpeedBench acceptance length 2.69), while retaining the checkpoint's built-in MTP. Do not pass `--speculative-draft-model-path` for a DeepSeek/GLM self-MTP model: SGLang logs that it is unnecessary. Treat the RadixArk model-card `NEXTN` 5/6 values as an engine-level starting point, not proof that the deployed DCP/PD topology is safe.

## Evidence and applicability

1. **Checkpoint signal (high confidence, direct evidence).** RadixArk describes a native MTP/NextN layer; MTP, embeddings, attention, norms and head remain source precision; MTP is not an external draft artifact. The card says native context is 1,048,576 and validates NVIDIA Blackwell/4x GB300. Source: [RadixArk model card](https://huggingface.co/RadixArk/GLM-5.3-Flash-NVFP4).
2. **Official serving precedent (high confidence, direct evidence).** NVIDIA's Dynamo GLM-5.3 recipe lists the B200 SGLang profiles as NVFP4 + FP8 KV, EAGLE-style MTP, draft length 3 and measured acceptance length 2.69; it lists DTP4 aggregated and DEP4/DTP8 disaggregated, with HiCache CPU and 500K context. Source: [Dynamo GLM-5.3/5.2 recipe](https://docs.nvidia.com/dynamo/dev/recipes/glm-5-2) and [Dynamo recipe README](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5.3/README.md).
3. **Dynamo is orchestration, not the speculative implementation (high confidence, interpretation).** MTP/EAGLE correctness, streaming projection and DSA/DCP interaction are SGLang responsibilities; Dynamo contributes worker topology, KV-aware routing and NIXL transfer. Keep the SGLang image/runtime and model flags aligned across prefill and decode.
4. **Context and KV (high confidence).** The checkpoint card says 1,048,576 context and FP8 KV + TRT-LLM DSA is a validated pairing. NVIDIA's Dynamo recipe is less ambitious at 500K for the B200 SGLang profiles. Therefore 142K is a reasonable first long-context canary; near-1M requires a separate capacity/correctness gate, not extrapolation.
5. **Tools/reasoning/streaming (high confidence).** The model-card launch uses `glm45` reasoning and `glm47` tool-call parsers. Dynamo's GLM-5.3 documentation says streaming chat, completions, tool calls, images and schema-constrained output are supported. Speculation must be tested on streamed reasoning deltas and tool-call chunks, not only final non-streaming text. Sources: [RadixArk card](https://huggingface.co/RadixArk/GLM-5.3-Flash-NVFP4), [Dynamo GLM-5.3 recipe](https://docs.nvidia.com/dynamo/dev/recipes/glm-5-2).

## DCP4 and PD transfer recommendation

SGLang DCP is enabled with `--dcp-size 4`; its documentation states the draft KV cache is replicated on every DCP rank while target MLA KV is striped. This matters: DCP4 does not provide a free 4x reduction for the MTP draft state. Source: [SGLang DCP](https://lmsysorg.mintlify.app/docs/advanced_features/dcp).

Use DCP4 only as a staged experiment after single-rank-DCP correctness. For disaggregated prefill/decode, keep the same KV layout, DSA backends, KV dtype and speculative configuration on both sides. NVIDIA's GLM-5.3 recipe explicitly calls out matching layout environment variables for NIXL compatibility in its disaggregated profile. Validate PD transfer with speculation disabled first, then enable MTP on both workers; do not mix speculative and non-speculative producer/consumer assumptions without a measured compatibility test.

## Known failure modes and gates

* **DSA attention TP + self-MTP bookkeeping:** SGLang issue [#30296](https://github.com/sgl-project/sglang/issues/30296) documents GLM self-MTP/EAGLE failures when `attn_tp_size > 1`: DSA kernels saw inconsistent batch/varlen metadata (`batch_size != batch_size_k`, or `num_splits` shape mismatch). The matrix worked at `attn_tp_size=1` and with speculation disabled. This is a direct reason to canary DCP4/DP attention rather than assume the recipe transfers.
* **Illegal memory / kernel instability:** SGLang has reported CUDA illegal-address failures in GLM CUDA-graph/DeepGEMM paths ([#25551](https://github.com/sgl-project/sglang/issues/25551)). Treat any illegal address, watchdog timeout, rank divergence or NCCL hang as a hard stop; first reproduce with CUDA graphs/speculation independently disabled to isolate the layer.
* **Deadlock/truncation risks (inference from the above, medium confidence):** verify/draft batch metadata, PD KV handoff and streaming output projection can disagree on accepted-token count. Symptoms to gate are stuck requests, missing final token/tool-call closure, duplicated or truncated reasoning, and producer/consumer KV length mismatch. No official source located that proves a GLM-5.3-specific deadlock; these are test hypotheses, not confirmed defects.
* **DCP limitation:** draft KV is replicated, so memory pressure can increase at long context. Near 1M can OOM or force truncation even if target KV capacity appears adequate.

## Initial parameters and canary sequence

1. **Baseline:** no speculation, DSA/TRT-LLM, FP8 KV, same TP/EP/DCP/PD topology; test deterministic text, reasoning, tool calls, streaming and transfer.
2. **Self-MTP canary:** built-in `NEXTN`; start `num-steps=3`, `num-draft-tokens=4`, `eagle-topk=1`, adaptive on. Move to the RadixArk 5/6 configuration only if stable. (The Dynamo recipe's DL=3/AL=2.69 is the strongest locally relevant performance baseline.)
3. **Topology order:** aggregated, then PD without DCP, then DCP4 aggregated, then DCP4+PD. Never change topology, DSA backend and speculation in one first test.
4. **Lengths:** 142K first; then 256K/500K; near-1M only after no truncation/OOM and stable transfer. Include prompts ending at chunk boundaries and max output near context limit.
5. **Canary gates:** zero illegal-address/watchdog/NCCL/deadlock events; zero request timeout; exact or approved-token-equivalent output against no-spec baseline; complete `</think>`/tool-call JSON and final event in streaming; no KV-transfer mismatch; bounded p95 TTFT/ITL regression.

## Measurements to record

Record request count, timeout/error/hang count, TTFT, inter-token latency, output tokens/s, GPU memory/KV occupancy, transferred KV bytes and transfer latency. For speculation record **accepted tokens per verification**, acceptance length (AL), acceptance rate (`accepted / proposed`), and histogram by step (not only an average). Compare same prompt set, seed/sampling, concurrency, context length and topology. The published Dynamo point is AL 2.69 at draft length 3; do not call a different topology a regression without matched controls.

## Exact local-fork gaps

The local fork could not be executed or shell-inspected in this research-only tool context, so the following are the specific gaps to verify before landing/closing the issue rather than claims that they are already absent:

* Locate the issue-30 recipe/DGD and add an explicit built-in-MTP `NEXTN` contract; reject or document external `speculative-draft-model-path` for this checkpoint.
* Add an integration matrix for aggregated, PD, DCP4 and PD+DCP4, with speculation off/on and `attn_tp_size=1/>1`; assert startup flags and KV-layout parity.
* Add streaming reasoning/tool-call tests that assert complete final/tool chunks and no truncation after rejected draft tokens.
* Add long-context 142K/500K/near-1M tests and OOM/truncation gates; add acceptance-length/rate metrics to the benchmark output.
* Add a negative startup guard (or explicit unsupported status) for combinations known to hit SGLang's DSA/self-MTP metadata bug until the relevant upstream fix is present in the pinned image. Pin and record the SGLang commit/image in the report and benchmark artifacts.

## Decision

**Recommended resolution:** document and canary the built-in MTP/NextN path; do not fetch or train an external EAGLE checkpoint. Start with DL3/AL target ~2.69 in the Dynamo-tested profile, then test RadixArk's NEXTN 5/6 flags. Keep DCP4 and PD transfer behind independent gates, and do not enable production traffic until all gates pass.

### Sources retained

* [RadixArk/GLM-5.3-Flash-NVFP4 model card](https://huggingface.co/RadixArk/GLM-5.3-Flash-NVFP4) — checkpoint tensor/config and launch evidence.
* [NVIDIA Dynamo GLM-5.3/5.2 recipe](https://docs.nvidia.com/dynamo/dev/recipes/glm-5-2) — official topology, context, MTP and API support.
* [Dynamo GLM-5.3 recipe README](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5.3/README.md) — local recipe matrix and DL3/AL2.69 claim.
* [SGLang speculative decoding docs](https://docs.sglang.ai/advanced_features/speculative_decoding.html) — algorithm distinction and parameters.
* [SGLang DCP docs](https://lmsysorg.mintlify.app/docs/advanced_features/dcp) — draft-KV replication behavior.
* [SGLang #30296](https://github.com/sgl-project/sglang/issues/30296) — direct DSA/self-MTP failure evidence.
* [SGLang #25551](https://github.com/sgl-project/sglang/issues/25551) — illegal-address risk evidence.

**Operational note:** this subagent had no shell/GitHub-write capability; it did not commit, push, comment or close issue #30. The parent/operator must perform those actions after review.

---

## Acceptance report

```json
{
  "criteriaSatisfied": [{"id":"criterion-1","status":"satisfied","evidence":"Concise recommendation, source-backed findings, canary/measurement plan, and residual risks are documented above."}],
  "changedFiles": ["docs/research/issue-30-glm53-mtp-eagle.md"],
  "testsAddedOrUpdated": [],
  "commandsRun": [],
  "validationOutput": ["Primary model card, NVIDIA Dynamo recipe, SGLang docs, DCP docs, and SGLang issue #30296/#25551 reviewed."],
  "residualRisks": ["Local fork could not be shell-inspected; exact issue-30 code/test gaps remain to be confirmed.","No deployment or hardware validation was run.","No Git commit/push/issue comment/close was possible in this subagent context."],
  "noStagedFiles": false,
  "diffSummary": "Added research report only; no runtime or production configuration changes.",
  "reviewFindings": ["no blockers in the research recommendation; operational follow-up is required"],
  "manualNotes": "Use built-in NEXTN/MTP, not an external EAGLE checkpoint; parent must commit/push and update issue #30."
}
```
