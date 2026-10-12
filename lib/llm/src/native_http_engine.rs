// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-process transport from Dynamo directly to native HTTP workers.
//! Native payloads remain JSON values; typed request views are never serialized
//! to construct the worker request. Accepted/unknown generation POSTs are never replayed.

use std::collections::{BTreeMap, BTreeSet};

use crate::cache_authority::Ttl;
use crate::cache_authority::native_cache_frontend::{
    CacheRequest, NativeCacheFrontend, RenderedCount,
};
use crate::http::service::metrics::ResponseMetricCollector;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};

use async_stream::stream;
use async_trait::async_trait;
use dynamo_runtime::engine::{AsyncEngineContext, AsyncEngineContextProvider, ResponseStream};
use dynamo_runtime::error::{DynamoError, ErrorType};
use dynamo_runtime::pipeline::{Error, ManyOut, SingleIn};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::engines::StreamingEngine;
use crate::native_contract::{
    ForcedZeroThinkingQualification, ModelContract, NativeTier, NativeWorkerIdentity,
    ThinkingBudgetQualification, VerifiedForcedZeroThinkingQualification,
    VerifiedModelForcedZeroThinking, VerifiedModelThinkingBudget,
    VerifiedThinkingBudgetQualification,
};
use crate::native_fencing::{
    ExpectedRequestIds, FenceError, NativeFence, NativeRequestPhase, WorkerLoad,
};
use crate::native_tool_schema::{StrictStreamBuffer, StrictToolSchemas};
use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
};
use crate::protocols::openai::completions::{
    NvCreateCompletionRequest, NvCreateCompletionResponse,
};

pub const NATIVE_HTTP_BODY: &str = "native_http_body";

pub(crate) fn native_error_annotation(value: &Value) -> Annotated<Value> {
    let fields = crate::native_pd::dispatch::native_error_fields_from_value(value)
        .unwrap_or_else(|| serde_json::json!({"message":"native worker reported an error"}));
    Annotated::from_error(fields.to_string())
}
/// Created by HTTP middleware before the body is read. Never deserialized.
#[derive(Clone, Debug)]
pub struct HttpIngressOrigin {
    pub request_id: String,
    pub started_at: Instant,
}

/// Server-owned Messages plan; private fields never travel through client JSON.
pub struct MessagesCacheContext {
    pub origin: HttpIngressOrigin,
    pub plan: crate::native_cache_plan::CachePlan,
}

impl MessagesCacheContext {
    fn request(&self) -> Result<CacheRequest, Error> {
        let writes = self
            .plan
            .writes
            .iter()
            .map(|point| {
                let id = point["endpoint_id"]
                    .as_str()
                    .ok_or_else(|| invalid("cache endpoint ID missing"))?;
                let ttl = match point["ttl_ms"].as_u64() {
                    Some(300_000) => Ttl::FiveMinutes,
                    Some(3_600_000) => Ttl::OneHour,
                    _ => return Err(invalid("cache TTL invalid")),
                };
                Ok((id.to_owned(), ttl))
            })
            .collect::<Result<BTreeMap<_, _>, Error>>()?;
        let reads = self
            .plan
            .reads
            .iter()
            .map(|point| {
                point["endpoint_id"]
                    .as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| invalid("cache read endpoint ID missing"))
            })
            .collect::<Result<BTreeSet<_>, Error>>()?;
        Ok(CacheRequest::new(
            self.origin.request_id.clone(),
            self.origin.started_at,
            writes,
            reads,
        ))
    }
}

fn reject_client_cache_authority(body: &Value) -> Result<(), Error> {
    let fields = body
        .as_object()
        .ok_or_else(|| invalid("native body must be an object"))?;
    if fields.keys().any(|key| {
        (key.starts_with("prompt_cache_")
            && !matches!(key.as_str(), "prompt_cache_key" | "prompt_cache_retention"))
            || matches!(
                key.as_str(),
                "reservation_id"
                    | "authority_revision"
                    | "cache_request_started_at_ms"
                    | "canonical_native_body"
            )
    }) {
        return Err(invalid(
            "native cache authority fields are owned by the gateway",
        ));
    }
    Ok(())
}

fn strip_native_cache_authority(value: &mut Value) {
    strip_verified_cache_accounting_marker(value);
    if let Some(fields) = value.as_object_mut() {
        fields.retain(|key, _| {
            !key.starts_with("prompt_cache_")
                && !matches!(
                    key.as_str(),
                    "reservation_id"
                        | "authority_revision"
                        | "cache_request_started_at_ms"
                        | "canonical_native_body"
                )
        });
    }
}

pub(crate) const VERIFIED_CACHE_ACCOUNTING_FIELD: &str = "_dynamo_verified_cache_accounting";

/// Internal typed accounting is never accepted from a worker or serialized at
/// a public Chat/Responses boundary. Messages consumes it before serialization.
pub(crate) fn strip_verified_cache_accounting_marker(value: &mut Value) {
    match value {
        Value::Object(fields) => {
            fields.remove(VERIFIED_CACHE_ACCOUNTING_FIELD);
            for field in fields.values_mut() {
                strip_verified_cache_accounting_marker(field);
            }
        }
        Value::Array(values) => {
            for field in values {
                strip_verified_cache_accounting_marker(field);
            }
        }
        _ => {}
    }
}

pub(crate) fn strip_unsigned_cache_accounting(value: &mut Value) {
    if let Some(usage) = value.get_mut("usage").and_then(Value::as_object_mut) {
        usage.retain(|key, _| {
            !key.starts_with("cache_")
                && key != "cached_tokens"
                && key != "input_tokens"
                && key != "prompt_cache_hit_tokens"
        });
        if let Some(details) = usage
            .get_mut("prompt_tokens_details")
            .and_then(Value::as_object_mut)
        {
            details.remove("cached_tokens");
        }
    }
}

fn attach_verified_cache_accounting(
    value: &mut Value,
    accounting: Option<&crate::cache_authority::cache_accounting::VerifiedCacheAccounting>,
    exact_prompt_tokens: u64,
) -> Result<(), Error> {
    strip_unsigned_cache_accounting(value);
    let raw_prompt = value
        .get("usage")
        .and_then(|usage| usage.get("prompt_tokens"))
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid("usage_unavailable"))?;
    let completion = value["usage"].get("completion_tokens").and_then(Value::as_u64)
        .ok_or_else(|| invalid("usage_unavailable"))?;
    let total = raw_prompt.checked_add(completion).ok_or_else(|| invalid("usage_unavailable"))?;
    if raw_prompt != exact_prompt_tokens
        || value["usage"].get("total_tokens").filter(|value| !value.is_null())
            .is_some_and(|value| value.as_u64() != Some(total)) {
        return Err(invalid("usage_unavailable"));
    }
    if let Some(accounting) = accounting {
        accounting
            .validate(raw_prompt)
            .map_err(|_| invalid("usage_unavailable"))?;
    }
    value["usage"][VERIFIED_CACHE_ACCOUNTING_FIELD] = serde_json::to_value(accounting)
        .expect("typed cache accounting contains only unsigned token counts");
    Ok(())
}

/// Retain only completion identity, not choice content or per-chunk metadata.
fn native_terminal_usage_template(native_chunk: &Value) -> Value {
    let mut value = json!({"choices": []});
    for field in [
        "id",
        "object",
        "created",
        "model",
        "system_fingerprint",
        "service_tier",
    ] {
        if let Some(native) = native_chunk.get(field) {
            value[field] = native.clone();
        }
    }
    value
}

fn record_native_usage_chunk(last: &mut Option<Value>, native_chunk: &Value) {
    if let Some(usage) = native_chunk.get("usage") {
        let mut value = native_terminal_usage_template(native_chunk);
        value["usage"] = usage.clone();
        // Null usage replaces an earlier count; it must not revive stale billing.
        *last = Some(value);
    }
}

fn verified_ttl_terminal_usage(
    exact_prompt_tokens: u64,
    native_chunk: Option<&Value>,
    accounting: Option<&crate::cache_authority::cache_accounting::VerifiedCacheAccounting>,
) -> Result<Value, Error> {
    let native_chunk = native_chunk.ok_or_else(|| invalid("usage_unavailable"))?;
    let usage = native_chunk
        .get("usage")
        .filter(|usage| usage.is_object())
        .ok_or_else(|| invalid("usage_unavailable"))?;
    let mut value = native_terminal_usage_template(native_chunk);
    // Messages and Responses consume the same required completion envelope.
    // Validate identity without rewriting native counters or inventing defaults.
    serde_json::from_value::<NvCreateChatCompletionStreamResponse>(value.clone())
        .map_err(|_| invalid("usage_unavailable"))?;
    // Preserve the same actual native output accounting as the JSON path.
    // Only verified P cache input accounting is overlaid below.
    value["usage"] = usage.clone();
    attach_verified_cache_accounting(&mut value, accounting, exact_prompt_tokens)?;
    Ok(value)
}

fn attach_server_render_plan(body: &mut Value, plan: &crate::native_cache_plan::CachePlan) {
    let rendered = crate::native_cache_plan::native_render_body(plan);
    for field in [
        "prompt_cache_blocks",
        "prompt_cache_read_candidates",
        "prompt_cache_message_order",
    ] {
        if let Some(value) = rendered.get(field) {
            body[field] = value.clone();
        }
    }
}

#[cfg(test)]
mod cache_ingress_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn native_cache_bridge_risk_nested_json_and_sse_markers() {
        let arguments = r#"{"_dynamo_verified_cache_accounting":"user text"}"#;
        for member in ["message", "delta"] {
            let mut response = json!({"_dynamo_verified_cache_accounting":"worker","usage":{"_dynamo_verified_cache_accounting":"worker"},"choices":[{(member):{"_dynamo_verified_cache_accounting":"worker","content":arguments,"tool_calls":[{"function":{"arguments":arguments,"_dynamo_verified_cache_accounting":"worker"}}]}}]});
            strip_native_cache_authority(&mut response);
            assert!(response.get(VERIFIED_CACHE_ACCOUNTING_FIELD).is_none());
            assert!(
                response["usage"]
                    .get(VERIFIED_CACHE_ACCOUNTING_FIELD)
                    .is_none()
            );
            assert!(
                response["choices"][0][member]
                    .get(VERIFIED_CACHE_ACCOUNTING_FIELD)
                    .is_none()
            );
            assert!(
                response["choices"][0][member]["tool_calls"][0]["function"]
                    .get(VERIFIED_CACHE_ACCOUNTING_FIELD)
                    .is_none()
            );
            assert_eq!(response["choices"][0][member]["content"], arguments);
            assert_eq!(
                response["choices"][0][member]["tool_calls"][0]["function"]["arguments"],
                arguments
            );
        }
    }

    fn stream_usage_fixture(usage: Value) -> Value {
        json!({"id":"chatcmpl-counter-fixture", "object":"chat.completion.chunk",
            "created":1, "model":"glm-5.3-flash", "choices":[], "usage":usage})
    }

    #[test]
    fn native_cache_bridge_risk_terminal_requires_native_counter_agreement() {
        use crate::cache_authority::cache_accounting::VerifiedCacheAccounting;
        let report = VerifiedCacheAccounting {
            input_tokens: 40,
            cache_read_input_tokens: 10,
            cache_creation_5m_input_tokens: 20,
            cache_creation_1h_input_tokens: 30,
        };
        for native_prompt in [None, Some(99), Some(101)] {
            let usage = native_prompt.map(|prompt| stream_usage_fixture(json!({"prompt_tokens":prompt,"completion_tokens":41})));
            assert!(
                verified_ttl_terminal_usage(100, usage.as_ref(), Some(&report))
                    .unwrap_err()
                    .to_string()
                    .contains("usage_unavailable")
            );
            assert!(verified_ttl_terminal_usage(100, usage.as_ref(), None).is_err());
        }
        assert!(verified_ttl_terminal_usage(100, Some(&stream_usage_fixture(json!({"prompt_tokens":100}))), Some(&report)).is_err());
        let usage = stream_usage_fixture(json!({"prompt_tokens":100,"completion_tokens":41}));
        let known = verified_ttl_terminal_usage(100, Some(&usage), Some(&report)).unwrap();
        assert_eq!(known["usage"]["prompt_tokens"], 100);
        assert_eq!(
            known["usage"][VERIFIED_CACHE_ACCOUNTING_FIELD]["cache_read_input_tokens"],
            10
        );
        let unknown = verified_ttl_terminal_usage(100, Some(&usage), None).unwrap();
        assert!(unknown["usage"][VERIFIED_CACHE_ACCOUNTING_FIELD].is_null());
    }

    #[test]
    fn native_cache_usage_keeps_completion_details_with_verified_prefill_accounting() {
        use crate::cache_authority::cache_accounting::VerifiedCacheAccounting;
        let report = VerifiedCacheAccounting {
            input_tokens: 40, cache_read_input_tokens: 10,
            cache_creation_5m_input_tokens: 20, cache_creation_1h_input_tokens: 30,
        };
        for details in [
            json!({"reasoning_tokens":0,"completion_tokens_details":{"reasoning_tokens":0,"accepted_prediction_tokens":3}}),
            json!({"reasoning_tokens":17,"completion_tokens_details":{"reasoning_tokens":17,"accepted_prediction_tokens":3}}),
            json!({}),
            json!({"completion_tokens_details":null}),
            json!({"reasoning_tokens":null,"completion_tokens_details":{"reasoning_tokens":null}}),
            json!({"reasoning_tokens":null,"completion_tokens_details":{"reasoning_tokens":17}}),
        ] {
            let mut usage=json!({"prompt_tokens":100,"completion_tokens":41,"total_tokens":141,
                "cached_tokens":99,"cache_creation_input_tokens":99,
                "prompt_tokens_details":{"cached_tokens":99,"audio_tokens":2}});
            for (key,value) in details.as_object().unwrap() {usage[key]=value.clone();}
            let mut json_response=json!({"choices":[{"index":0}],"usage":usage.clone()});
            attach_verified_cache_accounting(&mut json_response,Some(&report),100).unwrap();
            let terminal=verified_ttl_terminal_usage(100,Some(&stream_usage_fixture(usage.clone())),Some(&report)).unwrap();
            assert_eq!(terminal["usage"],json_response["usage"]);
            assert_eq!(terminal["usage"].get("reasoning_tokens"),usage.get("reasoning_tokens"));
            assert_eq!(terminal["usage"].get("completion_tokens_details"),usage.get("completion_tokens_details"));
            assert_eq!(terminal["usage"]["total_tokens"],141);
            assert!(terminal["usage"].get("cached_tokens").is_none());
            assert!(terminal["usage"]["prompt_tokens_details"].get("cached_tokens").is_none());
            assert_eq!(terminal["usage"]["prompt_tokens_details"]["audio_tokens"],2);
            assert_eq!(terminal["usage"][VERIFIED_CACHE_ACCOUNTING_FIELD],serde_json::to_value(&report).unwrap());
            assert_eq!(native_reasoning_tokens(&terminal),native_reasoning_tokens(&json!({"usage":usage.clone()})));
            let expected_reasoning=usage.get("reasoning_tokens").and_then(Value::as_u64)
                .or_else(||usage.get("completion_tokens_details").and_then(|details|details.get("reasoning_tokens")).and_then(Value::as_u64));
            assert_eq!(native_reasoning_tokens(&terminal).map(|(tokens,_)|tokens),expected_reasoning);
            let unknown=verified_ttl_terminal_usage(100,Some(&stream_usage_fixture(usage.clone())),None).unwrap();
            assert!(unknown["usage"][VERIFIED_CACHE_ACCOUNTING_FIELD].is_null());
            assert_eq!(unknown["usage"].get("completion_tokens_details"),usage.get("completion_tokens_details"));
        }
        for usage in [json!({"prompt_tokens":100,"completion_tokens":41}),
            json!({"prompt_tokens":100,"completion_tokens":41,"total_tokens":null})] {
            let terminal=verified_ttl_terminal_usage(100,Some(&stream_usage_fixture(usage.clone())),Some(&report)).unwrap();
            assert_eq!(terminal["usage"].get("total_tokens"),usage.get("total_tokens"));
        }
        for bad in [Value::Null,json!({"prompt_tokens":99,"completion_tokens":41}),
            json!({"prompt_tokens":100}),json!({"prompt_tokens":100,"completion_tokens":41,"total_tokens":142}),
            json!({"prompt_tokens":100,"completion_tokens":u64::MAX})] {
            assert!(verified_ttl_terminal_usage(100,Some(&stream_usage_fixture(bad)),Some(&report)).is_err());
        }
        assert!(verified_ttl_terminal_usage(100,None,Some(&report)).is_err());
        for bad in [
            json!({"completion_tokens":41,"reasoning_tokens":1,"completion_tokens_details":{"reasoning_tokens":2}}),
            json!({"completion_tokens":41,"reasoning_tokens":"unknown"}),
            json!({"completion_tokens":41,"completion_tokens_details":{"reasoning_tokens":42}}),
        ] {assert!(native_reasoning_tokens(&json!({"usage":bad})).is_none());}
    }

    #[test]
    fn private_cache_authority_cannot_enter_public_dispatch() {
        for field in [
            "prompt_cache_commit_points",
            "prompt_cache_authority_proof",
            "reservation_id",
            "authority_revision",
            "cache_request_started_at_ms",
            "canonical_native_body",
        ] {
            let mut body = json!({"model":"qwen3.8-flash-next", "messages":[]});
            body[field] = json!("client value");
            assert!(reject_client_cache_authority(&body).is_err(), "{field}");
        }
        assert!(
            reject_client_cache_authority(&json!({
                "model":"qwen3.8-flash-next", "messages":[], "cache_salt":"tenant",
                "prompt_cache_key":"tenant", "prompt_cache_retention":"in_memory"
            }))
            .is_ok()
        );
    }

    #[test]
    fn typed_messages_cache_request_preserves_ingress_origin_and_distinct_hold_ids() {
        let started_at = Instant::now().checked_sub(Duration::from_secs(12)).unwrap();
        let context = MessagesCacheContext {
            origin: HttpIngressOrigin {
                request_id: "owned-http-id".into(),
                started_at,
            },
            plan: crate::native_cache_plan::CachePlan {
                body: json!({"messages":[]}),
                writes: vec![json!({"endpoint_id":"write:p", "ttl_ms":300_000})],
                reads: vec![json!({"endpoint_id":"read:p"})],
                message_order: vec![],
            },
        };
        let request = context.request().unwrap();
        assert_eq!(request.http_request_id, "owned-http-id");
        assert_eq!(request.http_started_at, started_at);
        assert!(matches!(request.writes["write:p"], Ttl::FiveMinutes));
        assert!(request.reads.contains("read:p"));
        assert!(!request.reads.contains("write:p"));
    }
}

const MAX_EVENT_BYTES: usize = 8 * 1024 * 1024;
const MAX_LOAD_BODY_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeIdFormat {
    SglangPairedRid,
    VllmChatId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum NativeIdMatch {
    Prefill,
    Choice(u32, Option<String>),
}

impl NativeIdFormat {
    fn match_id(self, rid: &str, n: u32, id: &str) -> Option<NativeIdMatch> {
        match self {
            Self::SglangPairedRid => match ExpectedRequestIds::new(rid, n).ok()?.phase(id)? {
                NativeRequestPhase::Prefill => Some(NativeIdMatch::Prefill),
                NativeRequestPhase::Sample(index) => Some(NativeIdMatch::Choice(index, None)),
            },
            Self::VllmChatId if n == 1 => {
                let salt = id.strip_prefix(&format!("chatcmpl-{rid}-"))?;
                valid_vllm_salt(salt).then(|| NativeIdMatch::Choice(0, Some(salt.to_owned())))
            }
            Self::VllmChatId => {
                let (index, native_id) = id.split_once('_')?;
                let index = canonical_choice_index(index, n)?;
                let salt = native_id.strip_prefix(&format!("chatcmpl-{rid}-"))?;
                valid_vllm_salt(salt).then(|| NativeIdMatch::Choice(index, Some(salt.to_owned())))
            }
        }
    }
}

fn canonical_choice_index(index: &str, n: u32) -> Option<u32> {
    if index.is_empty()
        || (index != "0" && index.starts_with('0'))
        || !index.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    index.parse::<u32>().ok().filter(|index| *index < n)
}

fn valid_vllm_salt(value: &str) -> bool {
    value.len() == 8
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

struct NativeRequestTrace {
    rid: String,
    model: String,
    ctx: Arc<dyn AsyncEngineContext>,
    started: Instant,
    first_native_chunk: bool,
    first_native_output: bool,
    finished: bool,
    streaming: bool,
    metrics: Option<ResponseMetricCollector>,
}

impl NativeRequestTrace {
    fn new(
        ctx: Arc<dyn AsyncEngineContext>,
        model: &str,
        streaming: bool,
        metrics: Option<ResponseMetricCollector>,
    ) -> Self {
        let trace = Self {
            rid: ctx.id().to_owned(),
            model: model.to_owned(),
            ctx,
            started: Instant::now(),
            first_native_chunk: false,
            first_native_output: false,
            finished: false,
            streaming,
            metrics,
        };
        trace.phase("engine_arrival");
        trace
    }

    fn phase(&self, stage: &str) {
        tracing::info!(target: "dynamo_llm::native_phase", parent: None, rid = %self.rid, model = %self.model, stage,
            elapsed_ms = self.started.elapsed().as_millis() as u64,
            "native_generation_stage");
    }

    fn count(&mut self, prompt_tokens: u64) {
        if let Some(metrics) = &mut self.metrics {
            metrics.observe_native_input(prompt_tokens);
        }
        tracing::info!(target: "dynamo_llm::native_phase", parent: None, rid = %self.rid, model = %self.model, stage = "count_end",
            prompt_tokens, elapsed_ms = self.started.elapsed().as_millis() as u64,
            "native_generation_stage");
    }

    fn usage(
        &mut self,
        prompt_tokens: u64,
        cached_tokens: Option<u64>,
        completion_tokens: Option<u64>,
        reasoning_tokens: Option<(u64, &'static str)>,
    ) {
        if let Some((reasoning_tokens, reasoning_source)) = reasoning_tokens {
            tracing::info!(target: "dynamo_llm::native_phase", parent: None, rid = %self.rid, model = %self.model, stage = "native_usage",
                prompt_tokens, cached_tokens, completion_tokens, reasoning_tokens,
                reasoning_source,
                elapsed_ms = self.started.elapsed().as_millis() as u64,
                "native_generation_stage");
        } else {
            tracing::info!(target: "dynamo_llm::native_phase", parent: None, rid = %self.rid, model = %self.model, stage = "native_usage",
                prompt_tokens, cached_tokens, completion_tokens,
                elapsed_ms = self.started.elapsed().as_millis() as u64,
                "native_generation_stage");
            self.phase("native_reasoning_usage_unavailable");
        }
    }

    fn response_usage(&mut self, value: &Value) {
        if let Some(metrics) = &mut self.metrics {
            let cached = native_cache_receipt(value).and_then(|(_, cached)| cached);
            let completion = value
                .get("usage")
                .and_then(|u| u.get("completion_tokens"))
                .and_then(Value::as_u64);
            metrics.observe_native_usage(cached, completion, self.streaming);
        }
    }

    fn native_request(&self, native_id: &str, choices: usize) {
        // No raw native UUID or client correlation value is emitted. A private
        // getter sample can use this same domain-separated key for joining.
        let native_key = native_observation_key(native_id);
        tracing::info!(target: "dynamo_llm::native_phase", parent: None, rid = %self.rid,
            model = %self.model, stage = "native_request_bound", native_key = %native_key, choices,
            "native_generation_stage");
    }

    fn classified(&self, tier: NativeTier) {
        if self.metrics.is_some() {
            tracing::info!(target: "dynamo_llm::native_phase", parent: None, rid = %self.rid,
                model = %self.model, stage = "generation_classified", tier = ?tier,
                streaming = self.streaming, elapsed_ms = self.started.elapsed().as_millis() as u64,
                arrival_source = "owned_http_arrival", "native_generation_stage");
        }
    }

    fn worker(&self, url: &str, tier: NativeTier, native_id_format: NativeIdFormat) {
        let origin = reqwest::Url::parse(url)
            .map(|url| url.origin().ascii_serialization())
            .unwrap_or_default();
        let digest = Sha256::digest(origin.as_bytes());
        let worker_key = digest[..8]
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        tracing::info!(target: "dynamo_llm::native_phase", parent: None, rid = %self.rid, model = %self.model, stage = "worker_selected",
            worker_key, tier = ?tier, native_id_format = ?native_id_format, elapsed_ms = self.started.elapsed().as_millis() as u64,
            "native_generation_stage");
    }

    fn first_chunk(&mut self) {
        if !self.first_native_chunk {
            self.first_native_chunk = true;
            self.phase("first_native_chunk");
        }
    }

    fn first_output(&mut self, value: &Value) {
        if self.first_native_output {
            return;
        }
        let kind = value
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| {
                choices.iter().find_map(|choice| {
                    let content = choice.get("delta").or_else(|| choice.get("message"))?;
                    if content
                        .get("content")
                        .and_then(Value::as_str)
                        .is_some_and(|s| !s.is_empty())
                    {
                        Some("content")
                    } else if ["reasoning", "reasoning_content"].iter().any(|key| {
                        content
                            .get(*key)
                            .and_then(Value::as_str)
                            .is_some_and(|s| !s.is_empty())
                    }) {
                        Some("reasoning")
                    } else if content
                        .get("tool_calls")
                        .and_then(Value::as_array)
                        .is_some_and(|calls| {
                            calls.iter().any(|call| {
                                call.get("function")
                                    .and_then(|function| function.get("arguments"))
                                    .and_then(Value::as_str)
                                    .is_some_and(|s| !s.is_empty())
                            })
                        })
                    {
                        Some("tool_arguments")
                    } else {
                        None
                    }
                })
            });
        if let Some(kind) = kind {
            self.first_native_output = true;
            if let Some(metrics) = &mut self.metrics {
                metrics.observe_native_first_output(self.streaming);
            }
            tracing::info!(target: "dynamo_llm::native_phase", parent: None, rid = %self.rid, model = %self.model,
                stage = "first_native_output", kind,
                elapsed_ms = self.started.elapsed().as_millis() as u64,
                "native_generation_stage");
        }
    }

    fn finish(&mut self, status: &str) {
        if self.finished {
            return;
        }
        self.finished = true;
        tracing::info!(target: "dynamo_llm::native_phase", parent: None, rid = %self.rid, model = %self.model, stage = "terminal", status,
            elapsed_ms = self.started.elapsed().as_millis() as u64,
            "native_generation_stage");
        // The native reader's collector ends here. The HTTP guard still owns
        // client completion/disconnection; this is never labeled GPU busy time.
        self.metrics.take();
    }
}

fn native_observation_key(native_id: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(b"native-observation-v1\0");
    digest.update(native_id.as_bytes());
    format!("{:x}", digest.finalize())
}

impl Drop for NativeRequestTrace {
    fn drop(&mut self) {
        if !self.finished {
            let status = if self.ctx.is_stopped() {
                "cancelled"
            } else {
                "aborted"
            };
            self.finish(status);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeWorkerConfig {
    pub url: String,
    pub tier: NativeTier,
    /// Optional gateway request limit. DS uses native admission policy and must
    /// not acquire an invented hard cap from this local counter.
    pub max_inflight: Option<usize>,
    /// Required native core load endpoint, never another router. Missing or
    /// incomplete exact-load metadata fails construction or skips that worker.
    pub load_path: Option<String>,
    /// Inclusive native input boundary, from the exact worker's server metadata.
    /// None is allowed for CPU harnesses but is not production qualification.
    pub max_input_tokens: Option<u64>,
    pub capability_source: Option<String>,
    /// Source proving same-snapshot native active_request_ids support.
    pub active_request_ids_source: Option<String>,
    /// Qualified native scheduler DP size, needed to reject partial load snapshots.
    pub native_dp_size: Option<usize>,
    /// Exact native request-ID grammar proven for this pinned worker family.
    pub native_id_format: NativeIdFormat,
    /// Exact identity of this pinned native worker instance.
    #[serde(default)]
    pub identity: Option<NativeWorkerIdentity>,
    /// Evidence that this exact instance enforces a native thinking-token budget.
    #[serde(default)]
    pub thinking_budget_qualification: Option<ThinkingBudgetQualification>,
    /// Evidence that this exact instance enforces a native final-phase mode.
    #[serde(default)]
    pub forced_zero_qualification: Option<ForcedZeroThinkingQualification>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeHttpEngineConfig {
    pub model: String,
    pub workers: Vec<NativeWorkerConfig>,
    /// Private loopback-only qualification of one actual native tier.
    #[serde(default)]
    pub qualification_tier: Option<NativeTier>,
}

fn ds_prefix_renderer_identity(
    model: &str,
    worker_url: &str,
    metadata: &Value,
) -> Result<String, Error> {
    let bytes = serde_json::to_vec(metadata).map_err(|e| invalid(e.to_string()))?;
    let artifact =
        crate::native_cleanup_cohort::parse_current_artifact(model, &bytes).map_err(invalid)?;
    if artifact.model != model
        || artifact.native_origin.trim_end_matches('/') != worker_url.trim_end_matches('/')
    {
        return Err(invalid(
            "loaded renderer model or origin differs from route",
        ));
    }
    let layout = &metadata["profile"]["layout"];
    let expected = layout["rank_ids"]
        .as_array()
        .filter(|ids| !ids.is_empty())
        .ok_or_else(|| invalid("loaded renderer rank set missing"))?;
    let mut rank_ids = BTreeSet::new();
    for id in expected {
        let rank = id
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| invalid("loaded renderer rank ID invalid"))?;
        if !rank_ids.insert(rank) {
            return Err(invalid("loaded renderer rank IDs duplicate"));
        }
    }
    let rows = layout["kv_compat_descriptor"]["ranks"]
        .as_array()
        .ok_or_else(|| invalid("loaded renderer rank descriptors missing"))?;
    let mut seen = BTreeSet::new();
    let mut tokenizer: Option<&Value> = None;
    for row in rows {
        let rank = row["rank_id"]
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| invalid("loaded renderer descriptor rank invalid"))?;
        if !seen.insert(rank) {
            return Err(invalid("loaded renderer descriptor ranks duplicate"));
        }
        let actual = &row["descriptor"]["tokenizer"];
        let fields = actual
            .as_object()
            .ok_or_else(|| invalid("loaded tokenizer descriptor missing"))?;
        actual["class"]
            .as_str()
            .filter(|s| !s.is_empty() && s.len() <= 4096)
            .ok_or_else(|| invalid("loaded tokenizer class missing"))?;
        let digest = actual["serialization_sha256"]
            .as_str()
            .ok_or_else(|| invalid("loaded tokenizer serialization missing"))?;
        if fields.len() != 2
            || digest.len() != 64
            || digest.bytes().all(|b| b == b'0')
            || !digest
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(invalid("loaded tokenizer descriptor invalid"));
        }
        if tokenizer.is_some_and(|prior| prior != actual) {
            return Err(invalid("loaded tokenizer differs across native ranks"));
        }
        tokenizer = Some(actual);
    }
    if seen != rank_ids {
        return Err(invalid("loaded renderer descriptors omit native rank"));
    }
    // Native serialization covers actual vocabulary, special tokens and chat
    // template. This routing hint does not claim resident KV or a file receipt.
    crate::cache_authority::cache_auth::digest(&json!({"model":artifact.model,"family":"deepseek","tokenizer":tokenizer.ok_or_else(||invalid("loaded tokenizer missing"))?})).map_err(invalid)
}

#[cfg(test)]
mod loaded_prefix_renderer_tests {
    use super::*;
    use crate::native_cleanup_cohort::{CleanupCohortArtifact, CleanupCohortParticipant};
    fn metadata(origin: &str, capacity: u64) -> Value {
        // Declared CPU runtime state, never a production qualification receipt.
        let participants = vec![CleanupCohortParticipant {
            participant_id: "CPU-tokenizer".into(),
            role: "tokenizer-owner".into(),
            process_birth: "CPU-birth".into(),
            source_digest: "a".repeat(64),
            runtime_registration_digest: "b".repeat(64),
        }];
        let actual = CleanupCohortArtifact {
            schema_version: 1,
            native_origin: origin.into(),
            native_incarnation: "CPU-incarnation".into(),
            model: "deepseek-ai/DeepSeek-V4.1-Flash".into(),
            source_digest: "c".repeat(64),
            runtime_profile_digest: "d".repeat(64),
            cleanup_cohort_digest: crate::cache_authority::cache_auth::digest(&json!([
                "CPU-tokenizer"
            ]))
            .unwrap(),
            participants,
        };
        actual.validate().unwrap();
        let tokenizer =
            json!({"class":"CPU-actual-tokenizer","serialization_sha256":"e".repeat(64)});
        let layout = json!({"rank_ids":[0,1],"device_tokens":capacity,"kv_compat_descriptor":{"ranks":[{"rank_id":0,"descriptor":{"tokenizer":tokenizer}},{"rank_id":1,"descriptor":{"tokenizer":tokenizer}}]}});
        json!({"qualified":false,"profile":{"family":"deepseek","native_origin_uuid":actual.native_incarnation,"source_digest":actual.source_digest,"runtime_profile_digest":actual.runtime_profile_digest,"layout":layout},"ordinary":{"discovery":{"artifact":actual,"layout":layout,"qualified":true},"registration_report":{},"installation":{"artifact":actual,"artifact_digest":actual.artifact_digest().unwrap(),"participant_ids":actual.participant_ids(),"cleanup_cohort_digest":actual.cleanup_cohort_digest}}})
    }
    #[test]
    fn ds_prefix_uses_loaded_tokenizer_without_config_identity() {
        let model = "deepseek-ai/DeepSeek-V4.1-Flash";
        let short = "http://short:30000";
        let long = "http://long:30000";
        let a = metadata(short, 32768);
        let b = metadata(long, 1048576);
        let identity = ds_prefix_renderer_identity(model, short, &a).unwrap();
        assert_eq!(
            identity,
            ds_prefix_renderer_identity(model, long, &b).unwrap()
        );
        let mut missing = a.clone();
        missing["profile"]["layout"]["kv_compat_descriptor"]["ranks"][0]["descriptor"]
            .as_object_mut()
            .unwrap()
            .remove("tokenizer");
        // Preserve the real normal metadata binding while changing its layout.
        missing["ordinary"]["discovery"]["layout"] = missing["profile"]["layout"].clone();
        assert!(ds_prefix_renderer_identity(model, short, &missing).is_err());
        let mut changed = a.clone();
        changed["profile"]["layout"]["kv_compat_descriptor"]["ranks"][0]["descriptor"]["tokenizer"]
            ["serialization_sha256"] = "f".repeat(64).into();
        changed["ordinary"]["discovery"]["layout"] = changed["profile"]["layout"].clone();
        assert!(ds_prefix_renderer_identity(model, short, &changed).is_err());
        changed["profile"]["layout"]["kv_compat_descriptor"]["ranks"][1]["descriptor"]["tokenizer"]
            ["serialization_sha256"] = "f".repeat(64).into();
        changed["ordinary"]["discovery"]["layout"] = changed["profile"]["layout"].clone();
        assert_ne!(
            identity,
            ds_prefix_renderer_identity(model, short, &changed).unwrap()
        );
        assert!(ds_prefix_renderer_identity(model, long, &a).is_err());
        let engine = NativeHttpEngine::new(NativeHttpEngineConfig {
            model: model.into(),
            qualification_tier: None,
            workers: [(short, NativeTier::Short), (long, NativeTier::Long)]
                .into_iter()
                .map(|(url, tier)| NativeWorkerConfig {
                    url: url.into(),
                    tier,
                    max_inflight: None,
                    load_path: Some("/v1/loads?include=core".into()),
                    max_input_tokens: None,
                    capability_source: None,
                    active_request_ids_source: Some("CPU source only".into()),
                    native_dp_size: Some(1),
                    native_id_format: NativeIdFormat::SglangPairedRid,
                    identity: None,
                    thinking_budget_qualification: None,
                    forced_zero_qualification: None,
                })
                .collect(),
        })
        .unwrap();
        let engine = engine
            .with_prefix_renderer_metadata(BTreeMap::from([(short.into(), a), (long.into(), b)]))
            .unwrap();
        assert_eq!(
            engine.prefix_renderer_identity.as_deref(),
            Some(identity.as_str())
        );
    }
}

struct Worker {
    config: NativeWorkerConfig,
    inflight: AtomicUsize,
    verified_thinking_budget: Option<VerifiedThinkingBudgetQualification>,
    native_thinking_budget: bool,
    verified_forced_zero: Option<VerifiedForcedZeroThinkingQualification>,
    native_forced_zero: bool,
}

impl Worker {
    fn supports_thinking_budget(&self) -> bool {
        self.native_thinking_budget
            || self
                .verified_thinking_budget
                .as_ref()
                .is_some_and(|receipt| {
                    self.config
                        .identity
                        .as_ref()
                        .is_some_and(|identity| receipt.matches_current(identity))
                })
    }

    fn supports_forced_zero(&self) -> bool {
        self.native_forced_zero
            || self.verified_forced_zero.as_ref().is_some_and(|receipt| {
                self.config
                    .identity
                    .as_ref()
                    .is_some_and(|identity| receipt.matches_current(identity))
            })
    }

    fn supports_request(&self, requires_budget: bool, requires_forced_zero: bool) -> bool {
        (!requires_budget || self.supports_thinking_budget())
            && (!requires_forced_zero || self.supports_forced_zero())
    }
}

struct Reservation(Arc<Worker>, Option<ExactPermit>);
// The tracked cleanup task owns this hold after HTTP cancellation. Local load
// accounting is released only when its verified native physical receipt arrives.
struct RequestAdmissionHold {
    _reservation: Reservation,
    permit: Option<crate::native_fencing::GenerationPermit>,
}
impl Drop for Reservation {
    fn drop(&mut self) {
        self.0.inflight.fetch_sub(1, Ordering::AcqRel);
    }
}

#[derive(Clone)]
struct PendingNative {
    worker_url: String,
    n: u32,
    incarnation: Option<String>,
    accepted_choices: BTreeSet<u32>,
    vllm_salt: Option<String>,
    live_prefill: bool,
    restarted: bool,
}

impl PendingNative {
    fn reconcile(
        &mut self,
        snapshot: &ExactLoadSnapshot,
        format: NativeIdFormat,
        rid: &str,
    ) -> bool {
        if self.restarted {
            return true;
        }
        if self.incarnation.as_deref() != Some(&snapshot.incarnation) {
            if self.incarnation.is_some() {
                // A reservation born under the old scheduler cannot be proved
                // accepted by a new one, even if a stale RID reappears later.
                self.restarted = true;
                self.accepted_choices.clear();
                self.vllm_salt = None;
                self.live_prefill = false;
                return true;
            }
            self.incarnation = Some(snapshot.incarnation.clone());
            self.accepted_choices.clear();
            self.vllm_salt = None;
        }
        self.live_prefill = false;
        for id in &snapshot.ids {
            let Some(matched) = format.match_id(rid, self.n, id) else {
                continue;
            };
            let (index, salt) = match matched {
                NativeIdMatch::Prefill => {
                    self.live_prefill = true;
                    continue;
                }
                NativeIdMatch::Choice(index, salt) => (index, salt),
            };
            if let Some(salt) = salt {
                if self.vllm_salt.as_ref().is_some_and(|known| known != &salt) {
                    return false;
                }
                self.vllm_salt.get_or_insert(salt);
            }
            self.accepted_choices.insert(index);
        }
        true
    }

    fn pending_weight(&self) -> u64 {
        let live_prefill = u64::from(self.live_prefill && !self.accepted_choices.contains(&0));
        u64::from(self.n) - self.accepted_choices.len() as u64 - live_prefill
    }
}

type ExactLoadState = Arc<Mutex<std::collections::BTreeMap<String, PendingNative>>>;

struct ExactPermit {
    state: ExactLoadState,
    rid: String,
}

impl Drop for ExactPermit {
    fn drop(&mut self) {
        self.state.lock().unwrap().remove(&self.rid);
    }
}

struct ExactLoadSnapshot {
    native_total: u64,
    ids: Vec<String>,
    incarnation: String,
    sampled_at: tokio::time::Instant,
}

enum RequestLifecycleBinding {
    Configured(Arc<crate::native_cleanup_cohort::OrdinaryCleanupStartupMaterial>),
    Qualified(Arc<crate::native_request_lifecycle::QualifiedRequestLifecycle>),
}

/// Current exact native startup identity and established authority material are
/// requalified for each explicit-cache request. A request retains its returned
/// frontend through Count, recount, generation and physical cleanup.
#[async_trait]
pub trait NativeCacheFrontendFactory: Send + Sync {
    async fn create(
        &self,
        client: reqwest::Client,
        registry: Arc<crate::native_cleanup::NativeCleanupRegistry>,
    ) -> Result<Arc<NativeCacheFrontend>, Error>;
}

impl RequestLifecycleBinding {
    async fn current(
        &self,
        client: &reqwest::Client,
        registry: crate::native_cleanup::NativeCleanupRegistry,
    ) -> Result<Arc<crate::native_request_lifecycle::QualifiedRequestLifecycle>, String> {
        match self {
            Self::Configured(material) => material.qualify_current(
                client, Duration::from_secs(3), registry,
            ).await,
            Self::Qualified(lifecycle) => lifecycle.refresh_current(
                client, Duration::from_secs(3),
            ).await,
        }
    }
}

pub struct NativeHttpEngine {
    contract: ModelContract,
    qualification_tier: Option<NativeTier>,
    workers: Vec<Arc<Worker>>,
    client: reqwest::Client,
    counter_cursor: AtomicUsize,
    fence: Option<Arc<NativeFence>>,
    prefix_renderer_identity: Option<String>,
    exact_load: ExactLoadState,
    cache_frontend: Option<Arc<NativeCacheFrontend>>,
    cache_frontend_factory: Option<Arc<dyn NativeCacheFrontendFactory>>,
    current_cache_discovery: Mutex<Option<Arc<NativeCacheFrontend>>>,
    cleanup: Arc<crate::native_cleanup::NativeCleanupRegistry>,
    request_lifecycles: Option<
        std::collections::BTreeMap<
            String,
            RequestLifecycleBinding,
        >,
    >,
    pd_dispatchers:
        std::collections::BTreeMap<String, Arc<crate::native_pd::dispatch::PdDispatcher>>,
}

fn invalid(message: impl Into<String>) -> Error {
    DynamoError::builder()
        .error_type(ErrorType::InvalidArgument)
        .message(message.into())
        .build()
        .into()
}

fn fencing_error(error: FenceError) -> Error {
    let kind = match error {
        FenceError::Saturated => ErrorType::ResourceExhausted,
        FenceError::DuplicateRequest | FenceError::Config(_) => ErrorType::InvalidArgument,
        _ => ErrorType::Unavailable,
    };
    DynamoError::builder()
        .error_type(kind)
        .message(error.to_string())
        .build()
        .into()
}

fn explicit_cache_unavailable() -> Error {
    DynamoError::builder()
        .error_type(ErrorType::Unavailable)
        .message("explicit cache requires a qualified native cache frontend")
        .build()
        .into()
}

fn native_failover_unavailable() -> Error {
    DynamoError::builder()
        .error_type(ErrorType::Unavailable)
        .message("no compatible healthy native worker remains before generation POST")
        .build()
        .into()
}

impl NativeHttpEngine {
    /// The same lifecycle registry is shared by generation and physical cache cleanup.
    pub fn cleanup_registry(&self) -> Arc<crate::native_cleanup::NativeCleanupRegistry> {
        self.cleanup.clone()
    }

    /// Use the existing HTTP pool for same-port native controls and generation.
    pub fn native_http_client(&self) -> reqwest::Client {
        self.client.clone()
    }

    pub fn with_cache_frontend(
        mut self,
        frontend: Arc<NativeCacheFrontend>,
    ) -> Result<Self, Error> {
        self.validate_cache_frontend(&frontend)?;
        *self.current_cache_discovery.lock().unwrap() = Some(frontend.clone());
        self.cache_frontend = Some(frontend);
        Ok(self)
    }

    fn validate_cache_frontend(&self, frontend: &Arc<NativeCacheFrontend>) -> Result<(), Error> {
        if !frontend
            .cleanup_registry()
            .is_some_and(|registry| Arc::ptr_eq(registry, &self.cleanup))
        {
            return Err(invalid(
                "cache and generation must share the same physical cleanup registry",
            ));
        }
        if self.workers.iter().any(|worker| {
            let origin = self
                .pd_dispatchers
                .get(worker.config.url.trim_end_matches('/'))
                .map_or(worker.config.url.as_str(), |dispatcher| {
                    dispatcher.prefill_url()
                });
            !frontend.has_worker(origin)
        }) {
            return Err(invalid(
                "cache frontend lacks an actual generation route's native cache worker",
            ));
        }
        Ok(())
    }

    pub fn with_cache_frontend_factory(mut self, factory: Arc<dyn NativeCacheFrontendFactory>) -> Self {
        self.cache_frontend_factory = Some(factory);
        self
    }

    async fn request_cache_frontend(
        &self,
        ctx: &Arc<dyn AsyncEngineContext>,
        deadline: tokio::time::Instant,
    ) -> Result<Arc<NativeCacheFrontend>, Error> {
        let Some(factory) = &self.cache_frontend_factory else {
            return self.cache_frontend.clone().ok_or_else(explicit_cache_unavailable);
        };
        let result = tokio::select! { biased;
            _ = ctx.stopped() => return Err(anyhow::anyhow!("request cancelled")),
            _ = tokio::time::sleep_until(deadline) => Err(count_deadline_exceeded()),
            result = factory.create(self.client.clone(), self.cleanup.clone()) => result,
        };
        let result = result.and_then(|frontend| {
            self.validate_cache_frontend(&frontend)?;
            Ok(frontend)
        });
        match result {
            Ok(frontend) => {
                *self.current_cache_discovery.lock().unwrap() = Some(frontend.clone());
                Ok(frontend)
            }
            Err(error) => {
                *self.current_cache_discovery.lock().unwrap() = None;
                Err(error)
            }
        }
    }

    pub fn cache_frontend_configured(&self) -> bool {
        self.cache_frontend.is_some() || self.cache_frontend_factory.is_some()
    }

    pub fn with_request_lifecycles(
        self,
        lifecycles: Vec<crate::native_request_lifecycle::QualifiedRequestLifecycle>,
    ) -> Result<Self, Error> {
        self.with_request_lifecycle_bindings(lifecycles.into_iter().map(Arc::new).collect())
    }

    pub fn with_request_lifecycle_bindings(
        mut self,
        lifecycles: Vec<Arc<crate::native_request_lifecycle::QualifiedRequestLifecycle>>,
    ) -> Result<Self, Error> {
        let mut expected: BTreeSet<_> = self
            .workers
            .iter()
            .map(|worker| worker.config.url.trim_end_matches('/').to_owned())
            .collect();
        let mut bindings = std::collections::BTreeMap::new();
        for lifecycle in lifecycles {
            if lifecycle.model() != self.contract.model
                || !self.cleanup.shares_registry(lifecycle.registry())
                || !expected.remove(lifecycle.worker_url())
            {
                return Err(invalid(
                    "ordinary cleanup must bind the model, exact worker and shared registry",
                ));
            }
            bindings.insert(lifecycle.worker_url().to_owned(), RequestLifecycleBinding::Qualified(lifecycle));
        }
        if !expected.is_empty() || bindings.is_empty() {
            return Err(invalid(
                "ordinary cleanup qualification must cover every configured native worker",
            ));
        }
        self.request_lifecycles = Some(bindings);
        Ok(self)
    }

    /// Every configured worker has validated private material. Current runtime
    /// qualification happens on selected admission, so offline siblings neither
    /// block startup nor need a fabricated/prior cohort.
    pub fn with_request_lifecycle_materials(
        mut self,
        materials: Vec<Arc<crate::native_cleanup_cohort::OrdinaryCleanupStartupMaterial>>,
    ) -> Result<Self, Error> {
        let mut expected: BTreeSet<_> = self.workers.iter()
            .map(|worker| worker.config.url.trim_end_matches('/').to_owned()).collect();
        let mut bindings = BTreeMap::new();
        for material in materials {
            let origin = reqwest::Url::parse(material.native_origin())?
                .origin().ascii_serialization();
            if material.model() != self.contract.model || !expected.remove(&origin) {
                return Err(invalid("ordinary private material must bind the model and exact configured worker"));
            }
            bindings.insert(origin, RequestLifecycleBinding::Configured(material));
        }
        if !expected.is_empty() || bindings.is_empty() {
            return Err(invalid("ordinary private material must cover every configured native worker"));
        }
        self.request_lifecycles = Some(bindings);
        Ok(self)
    }

    pub fn request_cleanup_configured(&self) -> bool {
        self.request_lifecycles.is_some()
    }

    /// P/D qualification is independent of cache activation and ordinary worker
    /// admission. The configured decode member keeps the existing tier policy.
    pub fn with_pd_dispatchers(
        mut self,
        dispatchers: Vec<crate::native_pd::dispatch::PdDispatcher>,
    ) -> Result<Self, Error> {
        let mut bindings = std::collections::BTreeMap::new();
        for dispatcher in dispatchers {
            let decode = dispatcher.decode_url().trim_end_matches('/').to_owned();
            if !self
                .workers
                .iter()
                .any(|worker| worker.config.url.trim_end_matches('/') == decode)
                || !dispatcher.registry().shares_registry(&self.cleanup)
                || dispatcher.native_model() != self.contract.model
                || bindings.insert(decode, Arc::new(dispatcher)).is_some()
            {
                return Err(invalid(
                    "P/D pair lacks an exact configured decode member, model or cleanup registry",
                ));
            }
        }
        self.pd_dispatchers = bindings;
        Ok(self)
    }

    pub fn with_fence(mut self, fence: Arc<NativeFence>) -> Self {
        self.fence = Some(fence);
        self
    }
    pub fn fence(&self) -> Option<&Arc<NativeFence>> {
        self.fence.as_ref()
    }
    pub fn model(&self) -> &str {
        self.contract.model
    }
    pub fn contract(&self) -> &ModelContract {
        &self.contract
    }
    pub fn model_card(&self) -> Value {
        let mut card = self.contract.model_card();
        let input_tier = if let Some(tier) = self.qualification_tier {
            // A private canary must not advertise full-model capability.
            card = ModelContract::for_model(self.contract.model)
                .expect("configured model was validated during construction")
                .model_card();
            card["qualification"] = serde_json::json!({
                "scope": "single_tier",
                "tier": tier,
                "full_model_qualified": false
            });
            tier
        } else if self.contract.short_limit.is_some() {
            NativeTier::Long
        } else {
            NativeTier::Shared
        };
        // Model input capacity comes from its full-context tier. Smaller short
        // workers do not reduce it; dispatch retains each worker's own guard.
        let input_limit = self
            .workers
            .iter()
            .filter(|worker| {
                worker.config.tier == input_tier && worker.config.capability_source.is_some()
            })
            .filter_map(|worker| worker.config.max_input_tokens)
            .max();
        if let Some(limit) = input_limit {
            card["max_input_tokens"] = serde_json::json!(limit);
        } else {
            // Context capacity is not evidence of a native input boundary.
            card.as_object_mut()
                .expect("model card is an object")
                .remove("max_input_tokens");
        }
        // TTL follows the actual cache renderer: P for a P/D pair, otherwise
        // the generation worker. Discovery must have loaded every such origin.
        let current_cache = self.current_cache_discovery.lock().unwrap().clone();
        let cache_discovery = if self.cache_frontend_factory.is_some() {
            current_cache.as_ref()
        } else { self.cache_frontend.as_ref() };
        let ttl = cache_discovery.is_some_and(|frontend| {
            self.workers.iter().all(|worker| {
                let cache_origin = self
                    .pd_dispatchers
                    .get(worker.config.url.trim_end_matches('/'))
                    .map_or(worker.config.url.as_str(), |dispatcher| {
                        dispatcher.prefill_url()
                    });
                frontend.has_worker(cache_origin)
            })
        });
        if ttl {
            let cache = &mut card["gateway_capabilities"]["prompt_cache"];
            cache["mode"] = serde_json::json!("explicit_ttl");
            cache["block_breakpoints_supported"] = serde_json::json!(true);
            cache["ttl_guaranteed"] = serde_json::json!(true);
            cache["cache_read_tokens_source"] = serde_json::json!("verified_native_cache_receipts");
            cache["cache_write_billing"] = serde_json::json!(true);
        }
        card
    }

    pub async fn count_tokens(
        &self,
        body: Value,
        ctx: Arc<dyn AsyncEngineContext>,
    ) -> Result<u64, Error> {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
        reject_client_cache_authority(&body)?;
        crate::native_pd::reject_client_routing(&body).map_err(invalid)?;
        let body = self
            .contract
            .normalize(body)
            .map_err(|e| invalid(e.to_string()))?;
        self.count(&body, &ctx, deadline).await
    }

    /// Qualified native rendering only: no registry lookup, reservation,
    /// publication, or cache refresh occurs in a public Count request.
    pub async fn count_messages_with_cache_plan(
        &self,
        plan: crate::native_cache_plan::CachePlan,
        ctx: Arc<dyn AsyncEngineContext>,
        origin: HttpIngressOrigin,
    ) -> Result<u64, Error> {
        reject_client_cache_authority(&plan.body)?;
        crate::native_pd::reject_client_routing(&plan.body).map_err(invalid)?;
        let mut body = self
            .contract
            .normalize(plan.body.clone())
            .map_err(|e| invalid(e.to_string()))?;
        if body.get("rid").is_some() || body.get("request_id").is_some() {
            return Err(invalid("native request ID is owned by the gateway"));
        }
        body[if self.contract.model == "glm-5.3-flash" {
            "request_id"
        } else {
            "rid"
        }] = uuid::Uuid::new_v4().simple().to_string().into();
        let cache = MessagesCacheContext { origin, plan };
        let request = cache.request()?;
        let deadline = request.control_deadline()?;
        if request.http_request_id != ctx.id() {
            return Err(invalid(
                "cache HTTP origin does not match the owned request context",
            ));
        }
        let frontend = self.request_cache_frontend(&ctx, deadline).await?;
        attach_server_render_plan(&mut body, &cache.plan);
        Ok(self
            .signed_count_using(&frontend, &body, &ctx, &request, deadline)
            .await?
            .prompt_tokens)
    }

    pub fn with_prefix_renderer_metadata(
        mut self,
        metadata: BTreeMap<String, Value>,
    ) -> Result<Self, Error> {
        if self.contract.model != "deepseek-ai/DeepSeek-V4.1-Flash" {
            return Err(invalid("prefix renderer discovery is DeepSeek only"));
        }
        let expected = self
            .workers
            .iter()
            .map(|worker| worker.config.url.trim_end_matches('/').to_owned())
            .collect::<BTreeSet<_>>();
        if metadata.keys().cloned().collect::<BTreeSet<_>>() != expected {
            return Err(invalid(
                "loaded renderer metadata must cover every native worker",
            ));
        }
        let mut identity = None;
        for (url, value) in &metadata {
            let actual = ds_prefix_renderer_identity(&self.contract.model, url, value)?;
            if identity.as_ref().is_some_and(|prior| prior != &actual) {
                return Err(invalid(
                    "loaded renderer differs across native worker pools",
                ));
            }
            identity = Some(actual);
        }
        self.prefix_renderer_identity = identity;
        Ok(self)
    }

    pub fn new(config: NativeHttpEngineConfig) -> Result<Self, Error> {
        let mut contract =
            ModelContract::for_model(&config.model).map_err(|e| invalid(e.to_string()))?;
        if config.workers.is_empty() {
            return Err(invalid("native worker list is empty"));
        }
        let required_tiers: &[NativeTier] = if contract.short_limit.is_some() {
            &[NativeTier::Short, NativeTier::Long]
        } else {
            &[NativeTier::Shared]
        };
        if config
            .qualification_tier
            .is_some_and(|tier| !required_tiers.contains(&tier))
        {
            return Err(invalid(
                "private qualification tier is invalid for this model",
            ));
        }
        let mut workers = Vec::with_capacity(config.workers.len());
        let mut verified = Vec::new();
        let mut verified_forced_zero = Vec::new();
        for worker in config.workers {
            if config
                .qualification_tier
                .is_some_and(|tier| worker.tier != tier)
            {
                return Err(invalid(
                    "private qualification may contain only its actual native tier",
                ));
            }
            let parsed = reqwest::Url::parse(&worker.url)?;
            if !matches!(parsed.scheme(), "http" | "https")
                || parsed.host_str().is_none()
                || !parsed.username().is_empty()
                || parsed.password().is_some()
                || parsed.query().is_some()
                || parsed.fragment().is_some()
                || parsed.path() != "/"
                || worker.max_inflight == Some(0)
            {
                return Err(invalid(
                    "worker URL must be an HTTP origin and max_inflight must be positive",
                ));
            }
            if contract.model == "deepseek-ai/DeepSeek-V4.1-Flash" && worker.max_inflight.is_some()
            {
                return Err(invalid(
                    "DeepSeek uses native routing policy; max_inflight must be null",
                ));
            }
            if worker.native_dp_size.is_none_or(|size| size == 0) {
                return Err(invalid(
                    "exact native load requires a positive qualified DP size",
                ));
            }
            if worker
                .active_request_ids_source
                .as_ref()
                .is_none_or(|source| source.trim().is_empty() || source.len() > 1024)
            {
                return Err(invalid("exact native load requires active-ID provenance"));
            }
            if worker.load_path.as_deref() != Some("/v1/loads?include=core") {
                return Err(invalid("exact native load requires /v1/loads?include=core"));
            }
            let expected_id_format = if contract.model == "glm-5.3-flash" {
                NativeIdFormat::VllmChatId
            } else {
                NativeIdFormat::SglangPairedRid
            };
            if worker.native_id_format != expected_id_format {
                return Err(invalid(
                    "native request-ID format does not match the pinned model family",
                ));
            }
            if worker.max_input_tokens.is_some() != worker.capability_source.is_some()
                || worker
                    .max_input_tokens
                    .is_some_and(|limit| limit == 0 || limit > contract.context_limit)
                || worker
                    .capability_source
                    .as_ref()
                    .is_some_and(|source| source.trim().is_empty() || source.len() > 1024)
            {
                return Err(invalid(
                    "native input boundary must be positive and paired with capability provenance",
                ));
            }
            if let Some(identity) = &worker.identity {
                identity.validate().map_err(|e| invalid(e.to_string()))?;
                if identity.model != contract.model {
                    return Err(invalid("native worker identity names another public model"));
                }
            }
            let verified_thinking_budget = match (
                &worker.identity,
                &worker.thinking_budget_qualification,
            ) {
                (Some(identity), Some(receipt)) => {
                    if !required_tiers.contains(&worker.tier) {
                        return Err(invalid(
                            "thinking budget receipt is attached to an incompatible worker tier",
                        ));
                    }
                    let receipt = receipt
                        .verify(identity)
                        .map_err(|e| invalid(e.to_string()))?;
                    verified.push((worker.tier, receipt.clone()));
                    Some(receipt)
                }
                (None, Some(_)) => {
                    return Err(invalid(
                        "thinking budget receipt requires a native worker identity",
                    ));
                }
                _ => None,
            };
            let forced_zero_receipt = match (&worker.identity, &worker.forced_zero_qualification) {
                (Some(identity), Some(receipt)) => {
                    if !required_tiers.contains(&worker.tier) {
                        return Err(invalid(
                            "forced zero thinking receipt is attached to an incompatible worker tier",
                        ));
                    }
                    let receipt = receipt
                        .verify(identity)
                        .map_err(|e| invalid(e.to_string()))?;
                    verified_forced_zero.push((worker.tier, receipt.clone()));
                    Some(receipt)
                }
                (None, Some(_)) => {
                    return Err(invalid(
                        "forced zero thinking receipt requires a native worker identity",
                    ));
                }
                _ => None,
            };
            workers.push(Arc::new(Worker {
                config: worker,
                inflight: AtomicUsize::new(0),
                verified_thinking_budget,
                native_thinking_budget: contract.supports_fixed_thinking_budget(),
                verified_forced_zero: forced_zero_receipt,
                native_forced_zero: contract.supports_forced_zero_thinking(),
            }));
        }
        if config.qualification_tier.is_some_and(|scope| {
            verified
                .iter()
                .any(|(worker_tier, _)| *worker_tier == scope)
        }) || (config.qualification_tier.is_none()
            && required_tiers
                .iter()
                .all(|tier| verified.iter().any(|(worker_tier, _)| worker_tier == tier)))
        {
            let qualification = if let Some(scope) = config.qualification_tier {
                VerifiedModelThinkingBudget::verify_single_tier(contract.model, scope, &verified)
            } else {
                VerifiedModelThinkingBudget::verify(contract.model, &verified)
            }
            .map_err(|e| invalid(e.to_string()))?;
            contract = contract
                .with_thinking_budget_qualification(&qualification)
                .map_err(|e| invalid(e.to_string()))?;
        }
        if config.qualification_tier.is_some_and(|scope| {
            verified_forced_zero
                .iter()
                .any(|(worker_tier, _)| *worker_tier == scope)
        }) || (config.qualification_tier.is_none()
            && required_tiers.iter().all(|tier| {
                verified_forced_zero
                    .iter()
                    .any(|(worker_tier, _)| worker_tier == tier)
            }))
        {
            let qualification = if let Some(scope) = config.qualification_tier {
                VerifiedModelForcedZeroThinking::verify_single_tier(
                    contract.model,
                    scope,
                    &verified_forced_zero,
                )
            } else {
                VerifiedModelForcedZeroThinking::verify(contract.model, &verified_forced_zero)
            }
            .map_err(|e| invalid(e.to_string()))?;
            contract = contract
                .with_forced_zero_thinking_qualification(&qualification)
                .map_err(|e| invalid(e.to_string()))?;
        }
        Ok(Self {
            contract,
            qualification_tier: config.qualification_tier,
            workers,
            client: reqwest::Client::builder()
                .no_proxy()
                .connect_timeout(Duration::from_secs(3))
                .timeout(Duration::from_secs(10800))
                .redirect(reqwest::redirect::Policy::none())
                .retry(reqwest::retry::never())
                .build()?,
            counter_cursor: AtomicUsize::new(0),
            fence: None,
            prefix_renderer_identity: None,
            exact_load: Arc::new(Mutex::new(std::collections::BTreeMap::new())),
            cache_frontend: None,
            cache_frontend_factory: None,
            current_cache_discovery: Mutex::new(None),
            cleanup: Arc::new(crate::native_cleanup::NativeCleanupRegistry::new()),
            request_lifecycles: None,
            pd_dispatchers: std::collections::BTreeMap::new(),
        })
    }

    pub fn inflight(&self) -> Vec<usize> {
        self.workers
            .iter()
            .map(|w| w.inflight.load(Ordering::Acquire))
            .collect()
    }

    /// Refresh every configured short worker in parallel. The qualified native
    /// timestamp or immutable-payload heartbeat establishes sample age.
    pub async fn refresh_short_loads(
        &self,
        ctx: &Arc<dyn AsyncEngineContext>,
    ) -> Result<(), Error> {
        let fence = self
            .fence
            .as_ref()
            .ok_or_else(|| fencing_error(FenceError::NotAuthority))?;
        let polls =
            self.workers
                .iter()
                .filter(|w| w.config.tier == NativeTier::Short)
                .map(|worker| async {
                    let path = worker.config.load_path.as_ref().ok_or_else(|| {
                        anyhow::anyhow!("short worker requires native load endpoint")
                    })?;
                    let reply = self
                        .client
                        .get(format!(
                            "{}{}",
                            worker.config.url.trim_end_matches('/'),
                            path
                        ))
                        .timeout(Duration::from_secs(3))
                        .send()
                        .await?
                        .error_for_status()?;
                    let value = bounded_load_json(reply).await.ok_or_else(|| {
                        anyhow::anyhow!("native load body is invalid or exceeds the HTTP bound")
                    })?;
                    let (running, waiting) = native_load_parts(&value)
                        .ok_or_else(|| anyhow::anyhow!("invalid native load sample"))?;
                    if worker.config.active_request_ids_source.is_none() {
                        anyhow::bail!("native active request IDs have no qualified provenance");
                    }
                    let active_request_ids = native_request_ids(&value).ok_or_else(|| {
                        anyhow::anyhow!("native active request IDs are missing or inconsistent")
                    })?;
                    let sampled_at = native_snapshot_time(&value, worker.config.native_id_format)
                        .ok_or_else(|| {
                        anyhow::anyhow!("native scheduler snapshot timestamp is stale or invalid")
                    })?;
                    let native_incarnation = native_incarnation(
                        &value,
                        worker.config.native_dp_size.ok_or_else(|| {
                            anyhow::anyhow!("native DP size has no qualified provenance")
                        })?,
                    )
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "native scheduler incarnation or DP snapshot is missing or inconsistent"
                        )
                    })?;
                    Ok::<_, Error>(WorkerLoad {
                        worker: worker.config.url.clone(),
                        healthy: true,
                        running,
                        waiting,
                        sampled_at,
                        active_request_ids: Some(active_request_ids),
                        native_incarnation,
                    })
                });
        let results = tokio::select! { biased; _ = ctx.stopped() => return Err(anyhow::anyhow!("request cancelled")), results = futures::future::join_all(polls) => results };
        match results.into_iter().collect::<Result<Vec<_>, _>>() {
            Ok(samples) => fence.update_samples(samples).map_err(fencing_error),
            Err(_) => {
                let _ = fence.update_samples(Vec::new());
                Err(fencing_error(FenceError::IncompleteSamples))
            }
        }
    }

    async fn count(
        &self,
        body: &Value,
        ctx: &Arc<dyn AsyncEngineContext>,
        deadline: tokio::time::Instant,
    ) -> Result<u64, Error> {
        let count_body = crate::native_contract::count_body(body);
        let requires_budget = ModelContract::requires_thinking_budget(body);
        let requires_forced_zero = ModelContract::requires_forced_zero_thinking(body);
        let start = self.counter_cursor.fetch_add(1, Ordering::Relaxed);
        for offset in 0..self.workers.len() {
            let worker = &self.workers[(start + offset) % self.workers.len()];
            if !worker.supports_request(requires_budget, requires_forced_zero) {
                continue;
            }
            let call = async {
                // Timer-wheel rounding must not permit a new HTTP request once
                // the exact monotonic deadline has already passed.
                if tokio::time::Instant::now() >= deadline {
                    return Err(count_deadline_exceeded());
                }
                let request = self
                    .client
                    .post(format!(
                        "{}{}",
                        worker.config.url.trim_end_matches('/'),
                        self.contract.count_path()
                    ))
                    .json(&count_body);
                let remaining = deadline
                    .checked_duration_since(tokio::time::Instant::now())
                    .filter(|remaining| !remaining.is_zero())
                    .ok_or_else(count_deadline_exceeded)?;
                let response = request.timeout(remaining).send().await?;
                if response.status().as_u16() == 400 || response.status().as_u16() == 422 {
                    return Err(invalid("native tokenizer rejected canonical request"));
                }
                if !response.status().is_success() {
                    anyhow::bail!("native counter unavailable");
                }
                let value: Value = response.json().await?;
                if self.contract.model == "deepseek-ai/DeepSeek-V4.1-Flash"
                    && value
                        .get("max_model_len")
                        .and_then(Value::as_u64)
                        .is_none_or(|length| length < self.contract.context_limit)
                {
                    anyhow::bail!(
                        "native DeepSeek counter context capability is below public contract"
                    );
                }
                value
                    .get("count")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| anyhow::anyhow!("native tokenizer returned invalid count"))
            };
            let result = tokio::select! { biased;
                _ = ctx.stopped() => return Err(anyhow::anyhow!("request cancelled")),
                _ = tokio::time::sleep_until(deadline) => break,
                r = call => r,
            };
            match result {
                Ok(count) => return Ok(count),
                Err(error) if error.downcast_ref::<DynamoError>().is_some() => return Err(error),
                Err(_) => continue,
            }
        }
        Err(DynamoError::builder()
            .error_type(ErrorType::Unavailable)
            .message("all native token counters unavailable")
            .build()
            .into())
    }

    #[cfg(test)]
    async fn signed_count(
        &self, body: &Value, ctx: &Arc<dyn AsyncEngineContext>,
        request: &CacheRequest, deadline: tokio::time::Instant,
    ) -> Result<RenderedCount, Error> {
        let frontend = self.request_cache_frontend(ctx, deadline).await?;
        self.signed_count_using(&frontend, body, ctx, request, deadline).await
    }

    async fn signed_count_using(
        &self,
        frontend: &Arc<NativeCacheFrontend>,
        body: &Value,
        ctx: &Arc<dyn AsyncEngineContext>,
        request: &CacheRequest,
        deadline: tokio::time::Instant,
    ) -> Result<RenderedCount, Error> {
        let start = self.counter_cursor.fetch_add(1, Ordering::Relaxed);
        let requires_budget = ModelContract::requires_thinking_budget(body);
        let requires_forced_zero = ModelContract::requires_forced_zero_thinking(body);
        let mut last_error = None;
        let mut last_native_refusal = None;
        for offset in 0..self.workers.len() {
            let worker = &self.workers[(start + offset) % self.workers.len()];
            if !worker.supports_request(requires_budget, requires_forced_zero) {
                continue;
            }
            let counter_url = self
                .pd_dispatchers
                .get(worker.config.url.trim_end_matches('/'))
                .map_or(worker.config.url.as_str(), |dispatcher| {
                    dispatcher.prefill_url()
                });
            let result = tokio::select! { biased;
                _ = ctx.stopped() => return Err(anyhow::anyhow!("request cancelled")),
                result = self.render_count_using(frontend, counter_url, body, ctx, request, deadline) => result,
            };
            match result {
                Ok(rendered) => return Ok(rendered),
                Err(error) if error.downcast_ref::<DynamoError>().is_some() => return Err(error),
                Err(error) if error.downcast_ref::<crate::cache_authority::Error>()
                    .is_some_and(|error| matches!(error, crate::cache_authority::Error::NativeHttp { .. })) => {
                    last_native_refusal = Some(error);
                }
                Err(error) => last_error = Some(error),
            }
        }
        // A later disconnected counter cannot erase a real native refusal.
        // A successful qualified counter above still ends failover normally.
        Err(last_native_refusal.or(last_error)
            .map(Into::into)
            .unwrap_or_else(|| invalid("no compatible native cache counter is available")))
    }

    #[cfg(test)]
    async fn render_count_until(
        &self, worker_url: &str, body: &Value, ctx: &Arc<dyn AsyncEngineContext>,
        request: &CacheRequest, deadline: tokio::time::Instant,
    ) -> Result<RenderedCount, Error> {
        let frontend = self.request_cache_frontend(ctx, deadline).await?;
        self.render_count_using(&frontend, worker_url, body, ctx, request, deadline).await
    }

    async fn render_count_using(
        &self,
        frontend: &Arc<NativeCacheFrontend>,
        worker_url: &str,
        body: &Value,
        ctx: &Arc<dyn AsyncEngineContext>,
        request: &CacheRequest,
        deadline: tokio::time::Instant,
    ) -> Result<RenderedCount, Error> {
        // Rendering is read-only: dropping its HTTP future creates no native
        // reservation or hold. Its clock/signature/decode share this deadline.
        tokio::select! { biased;
            _ = ctx.stopped() => Err(anyhow::anyhow!("request cancelled")),
            _ = tokio::time::sleep_until(deadline) => Err(count_deadline_exceeded()),
            result = async {
                if tokio::time::Instant::now() >= deadline {
                    return Err(count_deadline_exceeded());
                }
                frontend.render_count(worker_url, body, request, deadline).await.map_err(Into::into)
            } => result,
        }
    }

    async fn exact_snapshot(&self, worker: &Worker) -> Option<ExactLoadSnapshot> {
        let response = self
            .client
            .get(format!(
                "{}{}",
                worker.config.url.trim_end_matches('/'),
                worker.config.load_path.as_deref()?,
            ))
            .timeout(Duration::from_secs(3))
            .send()
            .await
            .ok()?;
        if !response.status().is_success() {
            return None;
        }
        let value = bounded_load_json(response).await?;
        let native_total = native_load(&value)?;
        let ids = native_request_ids(&value)?;
        let sampled_at = native_snapshot_time(&value, worker.config.native_id_format)?;
        let incarnation = native_incarnation(&value, worker.config.native_dp_size?)?;
        Some(ExactLoadSnapshot {
            native_total,
            ids,
            incarnation,
            sampled_at,
        })
    }

    async fn reserve(
        &self,
        tier: NativeTier,
        prompt: u64,
        ctx: &Arc<dyn AsyncEngineContext>,
        excluded_workers: &BTreeSet<String>,
        requires_budget: bool,
        requires_forced_zero: bool,
        rid: &str,
        n: u32,
    ) -> Result<Reservation, Error> {
        let eligible: Vec<_> = self
            .workers
            .iter()
            .filter(|worker| {
                !excluded_workers.contains(&worker.config.url)
                    && worker.supports_request(requires_budget, requires_forced_zero)
                    && worker.config.tier == tier
                    && worker
                        .config
                        .max_input_tokens
                        .is_none_or(|limit| prompt <= limit)
                    && worker
                        .config
                        .max_inflight
                        .is_none_or(|max| worker.inflight.load(Ordering::Acquire) < max)
            })
            .cloned()
            .collect();
        if eligible.is_empty() {
            return Err(DynamoError::builder()
                .error_type(ErrorType::ResourceExhausted)
                .message("no compatible native worker has local capacity")
                .build()
                .into());
        }
        let polled = tokio::select! { biased;
            _ = ctx.stopped() => return Err(anyhow::anyhow!("request cancelled")),
            rows = futures::future::join_all(eligible.iter().map(|worker| self.exact_snapshot(worker))) => rows,
        };
        let mut state = self.exact_load.lock().unwrap();
        let mut candidates = Vec::new();
        for (worker, snapshot) in eligible.into_iter().zip(polled) {
            let Some(snapshot) = snapshot else {
                continue;
            };
            if snapshot.sampled_at.elapsed() > Duration::from_millis(2500) {
                continue;
            }
            let mut pending = 0u64;
            let mut updates = Vec::new();
            let mut contradictory = false;
            for (pending_rid, reservation) in state.iter() {
                if reservation.worker_url == worker.config.url {
                    let mut candidate = reservation.clone();
                    if !candidate.reconcile(&snapshot, worker.config.native_id_format, pending_rid)
                    {
                        contradictory = true;
                        break;
                    }
                    pending = pending.saturating_add(candidate.pending_weight());
                    updates.push((pending_rid.clone(), candidate));
                }
            }
            if contradictory {
                continue;
            }
            for (pending_rid, candidate) in updates {
                state.insert(pending_rid, candidate);
            }
            let score = snapshot.native_total.saturating_add(pending);
            candidates.push((score, worker));
        }
        candidates.sort_by_key(|(score, _)| *score);
        if candidates.is_empty() {
            return Err(DynamoError::builder()
                .error_type(ErrorType::Unavailable)
                .message("every compatible native worker lacks a complete exact load snapshot")
                .build()
                .into());
        }
        for (_, worker) in candidates {
            let max = worker.config.max_inflight;
            if worker
                .inflight
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                    if max.is_some_and(|max| n >= max) {
                        None
                    } else {
                        n.checked_add(1)
                    }
                })
                .is_ok()
            {
                if state.contains_key(rid) {
                    worker.inflight.fetch_sub(1, Ordering::AcqRel);
                    return Err(invalid("owned native request ID already has a reservation"));
                }
                state.insert(
                    rid.to_owned(),
                    PendingNative {
                        worker_url: worker.config.url.clone(),
                        n,
                        incarnation: None,
                        accepted_choices: BTreeSet::new(),
                        vllm_salt: None,
                        live_prefill: false,
                        restarted: false,
                    },
                );
                return Ok(Reservation(
                    worker,
                    Some(ExactPermit {
                        state: self.exact_load.clone(),
                        rid: rid.to_owned(),
                    }),
                ));
            }
        }
        Err(DynamoError::builder()
            .error_type(ErrorType::ResourceExhausted)
            .message("no healthy compatible native worker has capacity")
            .build()
            .into())
    }

    /// This raw entry point also preserves native usage fields that Dynamo's
    /// current OpenAI DTO does not represent. HTTP handlers should use it.
    pub async fn forward_chat(
        &self,
        body: Value,
        ctx: Arc<dyn AsyncEngineContext>,
    ) -> Result<ManyOut<Annotated<Value>>, Error> {
        self.forward_chat_with_count(body, ctx)
            .await
            .map(|(_, stream)| stream)
    }

    /// Return the native exact prompt count along with the same dispatch, for
    /// protocols whose initial envelope needs input accounting before deltas.
    pub async fn forward_chat_with_count(
        &self,
        body: Value,
        ctx: Arc<dyn AsyncEngineContext>,
    ) -> Result<(u64, ManyOut<Annotated<Value>>), Error> {
        self.forward_native_with_count(body, ctx, None, None).await
    }

    pub async fn forward_messages_with_cache_count(
        &self,
        body: Value,
        ctx: Arc<dyn AsyncEngineContext>,
        cache: MessagesCacheContext,
    ) -> Result<(u64, ManyOut<Annotated<Value>>), Error> {
        self.forward_native_with_count(body, ctx, Some(cache), None)
            .await
    }

    /// Public native HTTP supplies its existing response collector at dispatch.
    /// Internal engine callers retain their own existing observation ownership.
    pub(crate) async fn forward_http_with_count(
        &self,
        body: Value,
        ctx: Arc<dyn AsyncEngineContext>,
        cache: Option<MessagesCacheContext>,
        metrics: ResponseMetricCollector,
    ) -> Result<(u64, ManyOut<Annotated<Value>>), Error> {
        self.forward_native_with_count(body, ctx, cache, Some(metrics))
            .await
    }

    async fn forward_native_with_count(
        &self,
        body: Value,
        ctx: Arc<dyn AsyncEngineContext>,
        cache: Option<MessagesCacheContext>,
        metrics: Option<ResponseMetricCollector>,
    ) -> Result<(u64, ManyOut<Annotated<Value>>), Error> {
        reject_client_cache_authority(&body)?;
        crate::native_pd::reject_client_routing(&body).map_err(invalid)?;
        let cache_request = cache
            .as_ref()
            .map(MessagesCacheContext::request)
            .transpose()?;
        let request_frontend = if let Some(request) = &cache_request {
            if request.http_request_id != ctx.id() {
                return Err(invalid(
                    "cache HTTP origin does not match the owned request context",
                ));
            }
            Some(self.request_cache_frontend(&ctx, request.control_deadline()?).await?)
        } else { None };
        let streaming = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
        let mut trace =
            NativeRequestTrace::new(ctx.clone(), self.contract.model, streaming, metrics);
        let request_started_at_ms =
            if self.request_lifecycles.is_some() || !self.pd_dispatchers.is_empty() {
                trace.phase("native_clock_begin");
                let clock = crate::cache_authority::frontend_clock::ChronyClock
                    .qualify()
                    .await
                    .map_err(|_| {
                        DynamoError::builder()
                            .error_type(ErrorType::Unavailable)
                            .message("native request authorization clock unavailable")
                            .build()
                    })?;
                clock.ensure_fresh().map_err(|_| {
                    DynamoError::builder()
                        .error_type(ErrorType::Unavailable)
                        .message("native request authorization clock is stale")
                        .build()
                })?;
                trace.phase("native_clock_complete");
                Some(clock.request_start_lower_ms())
            } else {
                None
            };
        let is_deepseek = self.contract.model == "deepseek-ai/DeepSeek-V4.1-Flash";
        if is_deepseek {
            self.fence
                .as_ref()
                .ok_or_else(|| fencing_error(FenceError::NotAuthority))?
                .check_generation()
                .map_err(fencing_error)?;
        }
        let mut body = self
            .contract
            .normalize(body)
            .map_err(|e| invalid(e.to_string()))?;
        let requires_budget = ModelContract::requires_thinking_budget(&body);
        let requires_forced_zero = ModelContract::requires_forced_zero_thinking(&body);
        let strict_tools = StrictToolSchemas::compile(&body).map_err(invalid)?;
        let choices = parallel_choices(&body)?;
        if body.get("rid").is_some() || body.get("request_id").is_some() {
            return Err(invalid("native request ID is owned by the gateway"));
        }
        let native_rid = uuid::Uuid::new_v4().simple().to_string();
        trace.native_request(&native_rid, choices);
        if self.contract.model == "glm-5.3-flash" {
            body["request_id"] = native_rid.clone().into();
        } else {
            body["rid"] = native_rid.clone().into();
        }
        let output = self
            .contract
            .output_budget(&body)
            .map_err(|e| invalid(e.to_string()))?;
        if let Some(cache) = &cache {
            // The public body was validated before these server-generated fields.
            attach_server_render_plan(&mut body, &cache.plan);
        }
        trace.phase("count_begin");
        // Initial Count and any selected-worker recount consume one budget.
        let count_deadline = if let Some(request) = &cache_request {
            request.control_deadline()?
        } else {
            tokio::time::Instant::now() + Duration::from_secs(30)
        };
        let mut rendered_count = if let Some(request) = &cache_request {
            Some(
                self.signed_count_using(request_frontend.as_ref().expect("cache factory qualified"), &body, &ctx, request, count_deadline)
                    .await?,
            )
        } else {
            None
        };
        let prompt = if let Some(rendered) = &rendered_count {
            rendered.prompt_tokens
        } else {
            self.count(&body, &ctx, count_deadline).await?
        };
        trace.count(prompt);
        trace.phase("admission_begin");
        let mut tier = self
            .contract
            .admit(prompt, output)
            .map_err(|e| invalid(e.to_string()))?;
        if self.qualification_tier.is_some_and(|scope| scope != tier) {
            return Err(invalid(
                "request requires a native tier outside this private qualification scope",
            ));
        }
        // Input capability filtering is decided before dispatch. A short worker
        // that cannot accept the prompt does not receive a generation probe.
        if self.qualification_tier.is_none()
            && tier == NativeTier::Short
            && !self.workers.iter().any(|worker| {
                worker.config.tier == NativeTier::Short
                    && worker.supports_request(requires_budget, requires_forced_zero)
                    && worker
                        .config
                        .max_input_tokens
                        .is_none_or(|limit| prompt <= limit)
            })
        {
            tier = NativeTier::Long;
        }
        if !self.workers.iter().any(|worker| {
            worker.config.tier == tier
                && worker.supports_request(requires_budget, requires_forced_zero)
                && worker
                    .config
                    .max_input_tokens
                    .is_none_or(|limit| prompt <= limit)
        }) {
            return Err(invalid(
                "prompt exceeds every compatible native worker input boundary",
            ));
        }
        // Classifies authenticated, validated generation traffic. The timestamp
        // used for business arrival remains the prior middleware arrival with
        // this same owned opaque RID, never this later post-Count event time.
        trace.classified(tier);
        if is_deepseek && tier == NativeTier::Short {
            self.refresh_short_loads(&ctx).await?;
        }
        let wants_stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
        let deadline = if tier == NativeTier::Short {
            1800
        } else {
            10800
        };
        let mut error_body_fence = tokio::time::Instant::now() + Duration::from_secs(deadline);
        let mut excluded_workers = if (requires_budget || requires_forced_zero)
            && is_deepseek
            && tier == NativeTier::Short
        {
            self.workers
                .iter()
                .filter(|worker| {
                    worker.config.tier == NativeTier::Short
                        && !worker.supports_request(requires_budget, requires_forced_zero)
                })
                .map(|worker| worker.config.url.clone())
                .collect::<BTreeSet<_>>()
        } else {
            BTreeSet::new()
        };
        let mut retried_connect = false;
        let (response, admission, request_guard, cache_guard, pd_generation) = loop {
            trace.phase("worker_reservation_begin");
            let (reservation, permit) = if is_deepseek && tier == NativeTier::Short {
                let fence = self.fence.as_ref().unwrap();
                fence.check_generation().map_err(fencing_error)?;
                let identity = ExpectedRequestIds::new(
                    native_rid.clone(),
                    u32::try_from(choices).map_err(|_| invalid("n exceeds native u32 range"))?,
                )
                .map_err(fencing_error)?;
                let affinity = crate::native_fencing::affinity_for_chat(
                    &body,
                    self.prefix_renderer_identity
                        .as_deref()
                        .ok_or_else(|| invalid("DeepSeek loaded renderer metadata missing"))?,
                )
                .map_err(fencing_error)?;
                let permit = fence
                    .take_short_permit(&native_rid, &affinity, identity, prompt, &excluded_workers)
                    .await
                    .map_err(|error| {
                        if retried_connect {
                            native_failover_unavailable()
                        } else {
                            fencing_error(error)
                        }
                    })?;
                let worker = self
                    .workers
                    .iter()
                    .find(|worker| {
                        worker.config.url == permit.worker
                            && worker.config.tier == NativeTier::Short
                            && worker.supports_request(requires_budget, requires_forced_zero)
                            && !excluded_workers.contains(&worker.config.url)
                            && worker
                                .config
                                .max_input_tokens
                                .is_none_or(|limit| prompt <= limit)
                    })
                    .ok_or_else(|| invalid("native fencing selected an incompatible worker"))?
                    .clone();
                worker
                    .inflight
                    .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                        count.checked_add(1)
                    })
                    .map_err(|_| invalid("local in-flight accounting overflow"))?;
                (Reservation(worker, None), Some(permit))
            } else {
                if is_deepseek {
                    self.fence
                        .as_ref()
                        .unwrap()
                        .check_generation()
                        .map_err(fencing_error)?;
                }
                let reservation = self
                    .reserve(
                        tier,
                        prompt,
                        &ctx,
                        &excluded_workers,
                        requires_budget,
                        requires_forced_zero,
                        &native_rid,
                        u32::try_from(choices)
                            .map_err(|_| invalid("n exceeds native u32 range"))?,
                    )
                    .await;
                let reservation = match reservation {
                    Ok(reservation) => reservation,
                    Err(error) if self.contract.model == "qwen3.8-flash-next"
                        && self.qualification_tier.is_none()
                        && tier == NativeTier::Short
                        && error.downcast_ref::<DynamoError>().is_some_and(|error| matches!(
                            error.error_type(), ErrorType::Unavailable | ErrorType::ResourceExhausted
                        )) => {
                        // Existing full-context workers can admit this same
                        // short request. This is still before any native POST.
                        tier = NativeTier::Long;
                        trace.phase("native_short_pool_failover");
                        continue;
                    }
                    Err(error) if (retried_connect || !excluded_workers.is_empty())
                        && error.downcast_ref::<DynamoError>().is_some_and(|error| matches!(
                            error.error_type(), ErrorType::Unavailable | ErrorType::ResourceExhausted
                        )) => {
                        return Err(native_failover_unavailable());
                    }
                    Err(error) => return Err(error),
                };
                (reservation, None)
            };
            trace.worker(
                &reservation.0.config.url,
                tier,
                reservation.0.config.native_id_format,
            );
            let worker_url = reservation.0.config.url.clone();
            let admission = Arc::new(RequestAdmissionHold {
                _reservation: reservation,
                permit,
            });
            if let Some(dispatcher) = self.pd_dispatchers.get(worker_url.trim_end_matches('/')) {
                // TTL belongs to the actual P renderer/ledger. Recount against
                // that worker within the original shared Count budget; neither
                // D nor the room transport may stand in for its token proof.
                let cached = if let Some(request) = &cache_request {
                    let rendered = match rendered_count.take() {
                        Some(rendered) if rendered.worker_url == dispatcher.prefill_url() => {
                            rendered
                        }
                        _ => {
                            self.render_count_using(
                                request_frontend.as_ref().expect("cache factory qualified"),
                                dispatcher.prefill_url(),
                                &body,
                                &ctx,
                                request,
                                count_deadline,
                            )
                            .await?
                        }
                    };
                    if rendered.prompt_tokens != prompt {
                        return Err(invalid("native P render differs from admission count"));
                    }
                    let epoch = self
                        .fence
                        .as_ref()
                        .map(|fence| fence.check_generation())
                        .transpose()
                        .map_err(fencing_error)?;
                    Some((rendered, epoch))
                } else {
                    None
                };
                let start = cached
                    .as_ref()
                    .map(|(rendered, _)| rendered.request_started_at_ms())
                    .or(request_started_at_ms)
                    .ok_or_else(|| anyhow::anyhow!("qualified request start missing"))?;
                let mut pd_body = body.clone();
                if let Some((rendered, _)) = &cached {
                    pd_body["cache_salt"] = rendered.cache_salt()?.into();
                }
                // The dispatcher binds this unchanged root/n to every sample
                // room and the warmup room. P's cache signature covers that
                // exact map; its ledger consumes once and publishes from the
                // warmup, while the shared guards retain every child's hold.
                let n =
                    u32::try_from(choices).map_err(|_| invalid("n exceeds native u32 range"))?;
                let plan = if let Some((rendered, _)) = &cached {
                    dispatcher
                        .allocate_cached_plan(pd_body, &native_rid, n, start, rendered)
                        .await?
                } else {
                    dispatcher
                        .allocate_plan(pd_body, &native_rid, n, start)
                        .await?
                };
                trace.phase("native_pd_prepare_begin");
                let (mut generation, cache_guard) = if let Some((rendered, epoch)) = cached {
                    let frontend = request_frontend.as_ref().expect("cache factory qualified").clone();
                    let request = cache_request
                        .as_ref()
                        .expect("cache request prepared above");
                    let (generation, guard) = tokio::select! { biased;
                        _ = ctx.stopped() => return Err(anyhow::anyhow!("request cancelled")),
                        result = plan.dispatch_cached(Box::new(admission.clone()), tokio_util::sync::CancellationToken::new(), frontend, request, rendered, epoch) => result?,
                    };
                    (generation, Some(guard))
                } else {
                    let generation = tokio::select! { biased;
                        _ = ctx.stopped() => return Err(anyhow::anyhow!("request cancelled")),
                        result = plan.dispatch(Box::new(admission.clone()), tokio_util::sync::CancellationToken::new()) => result?,
                    };
                    (generation, None)
                };
                let response = generation.take_decode_response()?;
                break (response, admission, None, cache_guard, Some(generation));
            }
            let lifecycle = self
                .request_lifecycles
                .as_ref()
                .map(|bindings| bindings.get(worker_url.trim_end_matches('/')))
                .flatten();
            if self.request_lifecycles.is_some() && lifecycle.is_none() {
                return Err(invalid("selected worker lacks qualified ordinary cleanup"));
            }
            let lifecycle = if let Some(installed) = lifecycle {
                // Loads observe scheduler restarts independently. Bind this new
                // admission to current opaque cleanup owners before any native
                // POST; old armed guards keep their original identity/receipts.
                let current = tokio::select! { biased;
                    _ = ctx.stopped() => return Err(anyhow::anyhow!("request cancelled")),
                    result = installed.current(&self.client, (*self.cleanup).clone()) => result,
                };
                match current {
                    Ok(current) => Some(current),
                    Err(error) => {
                        // Discovery is read-only: no envelope, registration or
                        // native POST exists. Release this local hold and choose
                        // another current origin once; the excluded set bounds
                        // attempts. Prepared/accepted requests never enter here.
                        trace.phase("native_cleanup_discovery_failover");
                        tracing::warn!(worker = %worker_url, cause = %error,
                            "selected native cleanup discovery unavailable before generation POST");
                        drop(admission);
                        excluded_workers.insert(worker_url);
                        continue;
                    }
                }
            } else { None };
            let mut attempt_request_guard = None;
            let mut attempt_request_registration = None;
            let (send_body, mut attempt_cache_guard) = if let Some(request) = &cache_request {
                let frontend = request_frontend.as_ref().expect("cache factory qualified");
                let rendered = match rendered_count.take() {
                    Some(rendered) if rendered.worker_url == worker_url => rendered,
                    _ => {
                        self.render_count_using(frontend, &worker_url, &body, &ctx, request, count_deadline)
                            .await?
                    }
                };
                if rendered.prompt_tokens != prompt {
                    return Err(invalid(
                        "selected native worker render differs from admission count",
                    ));
                }
                let generation_epoch = self
                    .fence
                    .as_ref()
                    .map(|fence| fence.check_generation())
                    .transpose()
                    .map_err(fencing_error)?;
                let mut generation_body = body.clone();
                if let Some(lifecycle) = &lifecycle {
                    generation_body["cache_salt"] = rendered.cache_salt()?.into();
                    let prepared = lifecycle
                        .prepare_cached_generation_draft(
                            generation_body,
                            &native_rid,
                            u32::try_from(choices)
                                .map_err(|_| invalid("n exceeds native u32 range"))?,
                            rendered.request_started_at_ms(),
                            Box::new(admission.clone()),
                        )
                        .await?;
                    let (prepared_body, registration) = prepared.into_parts();
                    error_body_fence = error_body_fence.min(registration.admission_fence());
                    generation_body = prepared_body;
                    attempt_request_registration = Some(registration);
                }
                let prepared = tokio::select! { biased;
                    _ = ctx.stopped() => return Err(anyhow::anyhow!("request cancelled")),
                    result = frontend.prepare_generation(&worker_url, generation_body, request, rendered, generation_epoch,
                        attempt_request_registration.as_ref().map(|registration|registration.admission_fence())
                            .unwrap_or(request.control_deadline()?)) => result?,
                };
                (prepared.body, Some(prepared.guard))
            } else {
                let generation_body = if let Some(lifecycle) = &lifecycle {
                    let prepared = lifecycle
                        .prepare_generation(
                            body.clone(),
                            &native_rid,
                            u32::try_from(choices)
                                .map_err(|_| invalid("n exceeds native u32 range"))?,
                            request_started_at_ms.ok_or_else(|| {
                                anyhow::anyhow!("qualified request start missing")
                            })?,
                            Box::new(admission.clone()),
                        )
                        .await?;
                    error_body_fence = error_body_fence.min(prepared.recovery.admission_fence());
                    attempt_request_guard = Some(prepared.guard);
                    prepared.body
                } else {
                    body.clone()
                };
                (generation_body, None)
            };
            // Finish durable admission and large-body preparation before the
            // cache task seals its fresh generation envelope. No JSON traversal
            // or asynchronous preparation follows that final signature.
            let send_bytes = if let Some(guard) = &attempt_cache_guard {
                guard.begin_generation().await?
            } else {
                serde_json::to_vec(&send_body)?
            };
            if ctx.is_stopped() {
                return Err(anyhow::anyhow!("request cancelled"));
            }
            if let Some(registration) = attempt_request_registration {
                let (entry, guard, _) = registration.into_registration()?;
                lifecycle
                    .as_ref()
                    .expect("draft has an ordinary lifecycle")
                    .registry()
                    .register_group(vec![entry])?;
                attempt_request_guard = Some(guard);
            }
            let send = self
                .client
                .post(format!(
                    "{}/v1/chat/completions",
                    worker_url.trim_end_matches('/')
                ))
                .timeout(Duration::from_secs(deadline))
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(send_bytes)
                .send();
            trace.phase("native_post_begin");
            let result = tokio::select! { biased; _ = ctx.stopped() => return Err(anyhow::anyhow!("request cancelled")), r = send => r };
            match result {
                Ok(response) => {
                    break (
                        response,
                        admission,
                        attempt_request_guard,
                        attempt_cache_guard,
                        None,
                    );
                }
                Err(error) if error.is_connect() && !retried_connect => {
                    if attempt_cache_guard.is_some() {
                        // A claimed native ID remains claimed across cancellation
                        // and revisions. A new HTTP request receives a new ID.
                        return Err(error.into());
                    }
                    trace.phase("native_connect_failover");
                    retried_connect = true;
                    if let Some(guard) = &mut attempt_cache_guard {
                        guard.query_finish(true).await?;
                    }
                    drop(attempt_cache_guard);
                    // Discard the caller's hold before waiting for the tracked
                    // exact no-POST receipt, so failover cannot retain a ghost.
                    drop(admission);
                    if let Some(guard) = &mut attempt_request_guard {
                        guard.not_dispatched(&error).await?;
                    }
                    drop(attempt_request_guard);
                    excluded_workers.insert(worker_url);
                }
                Err(error) => return Err(error.into()),
            }
        };
        trace.phase("native_response_headers");
        if !response.status().is_success() {
            // Reuse the bounded error-envelope summary; never expose the full
            // response or change the public status classification.
            let failure = tokio::select! { biased;
                _ = ctx.stopped() => return Err(anyhow::anyhow!("request cancelled")),
                status = crate::native_pd::dispatch::observe_generation_http_failure(
                    response, "worker", error_body_fence
                ) => status,
            };
            let status = failure.status;
            let kind = match status {
                400 | 422 => ErrorType::InvalidArgument,
                429 => ErrorType::ResourceExhausted,
                _ => ErrorType::Unavailable,
            };
            trace.finish("native_http_error");
            return Err(DynamoError::builder()
                .error_type(kind)
                .message(format!("native generation_post returned HTTP {status}: {}", failure.cause()))
                .build()
                .into());
        }
        if wants_stream
            && !response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .is_some_and(|v| {
                    v.split(';')
                        .next()
                        .is_some_and(|v| v.trim() == "text/event-stream")
                })
        {
            anyhow::bail!("native stream response is not text/event-stream");
        }
        let stream_ctx = ctx.clone();
        let output = stream! {
            let admission = admission;
            let _request_guard = request_guard;
            let permit = &admission.permit;
            let mut cache_guard = cache_guard;
            let mut pd_generation = pd_generation;
            if !wants_stream {
                let reply = tokio::select! { biased;
                    _ = stream_ctx.stopped() => return,
                    _error = pd_failure(&mut pd_generation) => {
                        trace.finish("native_pd_prefill_error");
                        yield Annotated::from_error("native prefill failed"); return;
                    },
                    r = response.json::<Value>() => r
                };
                match reply {
                    Ok(mut value) => {
                        if value.get("error").is_some() {
                            if pd_generation.is_some() { crate::native_pd::dispatch::observe_generation_error_packet(&value, "decode", "generation_json"); }
                            trace.finish("native_protocol_error");
                            yield native_error_annotation(&value); return;
                        }
                        strip_native_cache_authority(&mut value);
                        if cache_guard.is_some() { strip_unsigned_cache_accounting(&mut value); }
                        trace.phase("native_json_decoded");
                        trace.first_output(&value);
                        trace.response_usage(&value);
                        if valid_terminal(&value) && complete_json_choices(&value, choices)
                            && strict_tools.validate_json(&value).is_ok() {
                            let receipt = native_cache_receipt(&value);
                            let (actual_prompt, cached) = receipt.unwrap_or((prompt, None));
                            if receipt.is_some() {
                                trace.usage(
                                    actual_prompt,
                                    cached,
                                    value.get("usage").and_then(|u| u.get("completion_tokens")).and_then(Value::as_u64),
                                    native_reasoning_tokens(&value),
                                );
                            } else {
                                trace.phase("native_usage_unavailable");
                                trace.phase("native_reasoning_usage_unavailable");
                            }
                            if let Some(generation) = &mut pd_generation {
                                if generation.qualify_successful_terminal().await.is_err() {
                                    trace.finish("native_pd_terminal_error");
                                    yield Annotated::from_error("native P/D physical transfer terminal is unverified");
                                    return;
                                }
                            }
                            if let Some(guard) = &mut cache_guard {
                                match guard.query_finish(false).await {
                                    Ok(accounting) => {
                                        if attach_verified_cache_accounting(&mut value, accounting.as_ref(), prompt).is_err() {
                                            yield Annotated::from_error("usage_unavailable");
                                            return;
                                        }
                                    },
                                    Err(error) => {
                                        let cause = match error { crate::cache_authority::Error::Evidence(cause) => cause, crate::cache_authority::Error::Store(_) => "store_unavailable", crate::cache_authority::Error::Contention => "cas_contention", crate::cache_authority::Error::NativeHttp { .. } => "native_http_rejection" };
                                        tracing::warn!(target: "dynamo_llm::native_cache", native_request_id = %native_rid, phase = "query_finish", cause, "native cache receipt verification failed");
                                        trace.finish("native_cache_receipt_error");
                                        yield Annotated::from_error("native cache receipt verification failed");
                                        return;
                                    }
                                }
                            }
                            if let Some(permit) = permit { permit.complete_success(actual_prompt, cached); }
                            trace.finish("ok");
                            yield Annotated::from_data(value);
                        } else {
                            trace.finish("native_protocol_error");
                            yield Annotated::from_error("native JSON response has an invalid terminal or tool call");
                        }
                    },
                    Err(_) => { trace.finish("native_protocol_error"); yield Annotated::from_error("invalid native JSON response"); },
                }
                return;
            }
            let mut bytes = response.bytes_stream();
            let mut buffer = Vec::new();
            let mut completed_choices = BTreeSet::new();
            let mut saw_done = false;
            let mut cache_receipt = None;
            let mut native_usage_chunk: Option<Value> = None;
            let mut completion_tokens: Option<u64> = None;
            let mut reasoning_tokens = None;
            let mut strict_buffer = StrictStreamBuffer::new(strict_tools);
            let mut pd_terminal_frames = Vec::new();
            let mut pd_terminal_bytes = 0usize;
            while let Some(part) = tokio::select! { biased;
                _ = stream_ctx.stopped() => return,
                error = pd_failure(&mut pd_generation) => {
                    tracing::warn!(target: "dynamo_llm::native_phase", role = "prefill", phase = "generation_monitor", error = %error, "native P/D prefill failed");
                    trace.finish("native_pd_prefill_error");
                    yield Annotated::from_error("native prefill failed"); return;
                },
                part = bytes.next() => part
            } {
                let part = match part { Ok(part) => part, Err(_) => { trace.finish("native_transport_error"); yield Annotated::from_error("native stream transport failed"); return; } };
                if !part.is_empty() { trace.first_chunk(); }
                buffer.extend_from_slice(&part);
                loop {
                    let Some((end, separator)) = event_boundary(&buffer) else { break; };
                    if end > MAX_EVENT_BYTES { trace.finish("native_protocol_error"); yield Annotated::from_error("native SSE event exceeds size limit"); return; }
                    let frame = match std::str::from_utf8(&buffer[..end]) { Ok(frame) => frame, Err(_) => { trace.finish("native_protocol_error"); yield Annotated::from_error("native SSE is not UTF-8"); return; } };
                    let data = frame.lines().filter_map(|line| line.strip_prefix("data:").map(|d| d.strip_prefix(' ').unwrap_or(d))).collect::<Vec<_>>().join("\n");
                    if data == "[DONE]" {
                        trace.phase("native_stream_eof");
                        if let Some(generation) = &mut pd_generation {
                            if generation.qualify_successful_terminal().await.is_err() {
                                trace.finish("native_pd_terminal_error");
                                yield Annotated::from_error("native P/D physical transfer terminal is unverified");
                                return;
                            }
                        }
                        let mut final_cache_accounting = None;
                        if let Some(guard) = &mut cache_guard {
                            match guard.query_finish(false).await {
                                Ok(accounting) => final_cache_accounting = Some(accounting),
                                Err(error) => {
                                    let cause = match error { crate::cache_authority::Error::Evidence(cause) => cause, crate::cache_authority::Error::Store(_) => "store_unavailable", crate::cache_authority::Error::Contention => "cas_contention", crate::cache_authority::Error::NativeHttp { .. } => "native_http_rejection" };
                                    tracing::warn!(target: "dynamo_llm::native_cache", native_request_id = %native_rid, phase = "query_finish", cause, "native cache receipt verification failed");
                                    trace.finish("native_cache_receipt_error");
                                    yield Annotated::from_error("native cache receipt verification failed");
                                    return;
                                }
                            }
                        }
                        if completed_choices.len() != choices { trace.finish("native_protocol_error"); yield Annotated::from_error("native SSE ended without every choice finish_reason"); return; }
                        let held = match strict_buffer.finish() {
                            Ok(held) => held,
                            Err(message) => { trace.finish("native_protocol_error"); yield Annotated::from_error(message); return; }
                        };
                        let terminal_usage = match final_cache_accounting {
                            Some(accounting) => match verified_ttl_terminal_usage(prompt,native_usage_chunk.as_ref(),accounting.as_ref()) {
                                Ok(value) => Some(value),
                                Err(_) => { trace.finish("native_usage_unavailable"); yield Annotated::from_error("usage_unavailable"); return; }
                            },
                            None => None,
                        };
                        let (actual_prompt, cached) = cache_receipt.unwrap_or((prompt, None));
                        if cache_receipt.is_some() {
                            trace.usage(actual_prompt, cached, completion_tokens, reasoning_tokens);
                        } else {
                            trace.phase("native_usage_unavailable");
                            trace.phase("native_reasoning_usage_unavailable");
                        }
                        trace.finish("ok");
                        for frame in pd_terminal_frames.drain(..) { yield Annotated::from_data(frame); }
                        for frame in held { yield Annotated::from_data(frame); }
                        if let Some(terminal_usage) = terminal_usage { yield Annotated::from_data(terminal_usage); }
                        saw_done = true;
                        break;
                    }
                    if !data.is_empty() {
                        match serde_json::from_str::<Value>(&data) {
                            Ok(value) if value.get("error").is_some() => { if pd_generation.is_some() { crate::native_pd::dispatch::observe_generation_error_packet(&value, "decode", "generation_sse"); } trace.finish("native_protocol_error"); yield native_error_annotation(&value); return; }
                            Ok(mut value) if value.get("choices").is_some_and(Value::is_array) => {
                                strip_native_cache_authority(&mut value);
                                if cache_guard.is_some() { strip_unsigned_cache_accounting(&mut value); }
                                if !record_choice_terminals(&value, choices, &mut completed_choices) { trace.finish("native_protocol_error"); yield Annotated::from_error("native SSE choice index or terminal is invalid"); return; }
                                trace.first_output(&value);
                                trace.response_usage(&value);
                                // Ordinary usage is scheduling accounting. TTL authority
                                // comes only from the authenticated native receipt query.
                                if cache_guard.is_some() { record_native_usage_chunk(&mut native_usage_chunk, &value); }
                                if value.get("usage").is_some_and(|usage| !usage.is_null()) {
                                    cache_receipt = native_cache_receipt(&value);
                                    completion_tokens = value.get("usage").and_then(|u| u.get("completion_tokens")).and_then(Value::as_u64);
                                    reasoning_tokens = native_reasoning_tokens(&value);
                                } else if cache_guard.is_some() && value.get("usage").is_some_and(Value::is_null) {
                                    cache_receipt = None;
                                    completion_tokens = None;
                                    reasoning_tokens = None;
                                }
                                match strict_buffer.push(value) {
                                    Ok(frames) => for frame in frames {
                                        let pd_terminal = pd_generation.is_some() && frame.get("choices").and_then(Value::as_array)
                                            .is_some_and(|choices| choices.iter().any(|choice| choice.get("finish_reason").is_some_and(Value::is_string)));
                                        if pd_terminal {
                                            let size = match serde_json::to_vec(&frame) { Ok(bytes) => bytes.len(), Err(_) => {
                                                yield Annotated::from_error("invalid native P/D terminal frame"); return;
                                            }};
                                            pd_terminal_bytes = match pd_terminal_bytes.checked_add(size).filter(|size| *size <= MAX_EVENT_BYTES) {
                                                Some(size) => size,
                                                None => { yield Annotated::from_error("native P/D terminal frames exceed size limit"); return; }
                                            };
                                            pd_terminal_frames.push(frame);
                                        } else { yield Annotated::from_data(frame); }
                                    },
                                    Err(message) => { trace.finish("native_protocol_error"); yield Annotated::from_error(message); return; }
                                }
                            }
                            _ => { trace.finish("native_protocol_error"); yield Annotated::from_error("malformed native SSE JSON chunk"); return; }
                        }
                    }
                    buffer.drain(..end + separator);
                }
                if saw_done { break; }
                if buffer.len() > MAX_EVENT_BYTES { trace.finish("native_protocol_error"); yield Annotated::from_error("native SSE event exceeds size limit"); return; }
            }
            if !saw_done { trace.finish("native_protocol_error"); yield Annotated::from_error("native SSE closed without DONE"); }
            else {
                let (actual_prompt, cached) = cache_receipt.unwrap_or((prompt, None));
                if let Some(permit) = permit { permit.complete_success(actual_prompt, cached); }
            }
        };
        Ok((prompt, ResponseStream::new(Box::pin(output), ctx)))
    }
}

async fn pd_failure(
    generation: &mut Option<crate::native_pd::dispatch::PdGeneration>,
) -> crate::native_pd::dispatch::PdDispatchError {
    match generation {
        Some(generation) => generation.prefill_monitor.failure().await,
        None => futures::future::pending().await,
    }
}

fn count_deadline_exceeded() -> Error {
    DynamoError::builder()
        .error_type(ErrorType::Unavailable)
        .message("native token count exceeded its total 30s deadline")
        .build()
        .into()
}

async fn bounded_load_json(response: reqwest::Response) -> Option<Value> {
    let mut bytes = response.bytes_stream();
    let mut body = Vec::new();
    while let Some(part) = bytes.next().await {
        let part = part.ok()?;
        let size = body.len().checked_add(part.len())?;
        if size > MAX_LOAD_BODY_BYTES {
            return None;
        }
        body.extend_from_slice(&part);
    }
    serde_json::from_slice(&body).ok()
}

fn native_load(value: &Value) -> Option<u64> {
    let (running, waiting) = native_load_parts(value)?;
    (running as u64).checked_add(waiting as u64)
}

fn parallel_choices(body: &Value) -> Result<usize, Error> {
    body.get("n").map_or(Ok(1), |value| {
        value
            .as_u64()
            .filter(|n| (1..=u32::MAX as u64).contains(n))
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| invalid("n must be a positive integer within the native u32 range"))
    })
}

fn native_cache_receipt(value: &Value) -> Option<(u64, Option<u64>)> {
    let usage = value.get("usage")?;
    let prompt = usage
        .get("prompt_tokens")?
        .as_u64()
        .filter(|prompt| *prompt > 0)?;
    let fields = [
        usage.get("cached_tokens"),
        usage.get("cache_read_input_tokens"),
        usage
            .get("prompt_tokens_details")
            .and_then(|details| details.get("cached_tokens")),
    ];
    let mut cached = None;
    for field in fields.into_iter().flatten() {
        let count = field.as_u64().filter(|cached| *cached <= prompt)?;
        if cached.is_some_and(|old| old != count) {
            return None;
        }
        cached = Some(count);
    }
    Some((prompt, cached))
}

pub(crate) fn native_reasoning_tokens(value: &Value) -> Option<(u64, &'static str)> {
    let usage = value.get("usage")?;
    let completion = usage.get("completion_tokens")?.as_u64()?;
    let flat = match usage.get("reasoning_tokens") {
        None | Some(Value::Null) => None,
        Some(value) => Some(value.as_u64()?),
    };
    let nested = match usage.get("completion_tokens_details") {
        None | Some(Value::Null) => None,
        Some(Value::Object(details)) => match details.get("reasoning_tokens") {
            None | Some(Value::Null) => None,
            Some(value) => Some(value.as_u64()?),
        },
        Some(_) => return None,
    };
    let (reasoning, source) = match (flat, nested) {
        (Some(flat), Some(nested)) if flat == nested => (
            flat,
            "usage.reasoning_tokens+usage.completion_tokens_details.reasoning_tokens",
        ),
        (Some(flat), None) => (flat, "usage.reasoning_tokens"),
        (None, Some(nested)) => (nested, "usage.completion_tokens_details.reasoning_tokens"),
        _ => return None,
    };
    (reasoning <= completion).then_some((reasoning, source))
}

#[cfg(test)]
mod native_reasoning_usage_tests {
    use super::native_reasoning_tokens;
    use serde_json::json;

    #[test]
    fn records_only_explicit_consistent_bounded_native_reasoning_count() {
        let flat = "usage.reasoning_tokens";
        let nested = "usage.completion_tokens_details.reasoning_tokens";
        let both = "usage.reasoning_tokens+usage.completion_tokens_details.reasoning_tokens";
        assert_eq!(
            native_reasoning_tokens(
                &json!({"usage":{"completion_tokens":41,"reasoning_tokens":16}})
            ),
            Some((16, flat))
        );
        assert_eq!(
            native_reasoning_tokens(
                &json!({"usage":{"completion_tokens":10,"completion_tokens_details":{"reasoning_tokens":0}}})
            ),
            Some((0, nested))
        );
        assert_eq!(
            native_reasoning_tokens(
                &json!({"usage":{"completion_tokens":10,"reasoning_tokens":7,"completion_tokens_details":{"reasoning_tokens":7}}})
            ),
            Some((7, both))
        );
        assert_eq!(
            native_reasoning_tokens(
                &json!({"usage":{"completion_tokens":10,"reasoning_tokens":7,"completion_tokens_details":null}})
            ),
            Some((7, flat))
        );
        for usage in [
            json!({"completion_tokens":10,"reasoning_tokens":7,"completion_tokens_details":{"reasoning_tokens":8}}),
            json!({"completion_tokens":10,"reasoning_tokens":true}),
            json!({"completion_tokens":10,"reasoning_tokens":7.5}),
            json!({"completion_tokens":10,"reasoning_tokens":-1}),
            json!({"completion_tokens":10,"reasoning_tokens":11}),
            json!({"completion_tokens":10,"reasoning_tokens":7,"completion_tokens_details":{"reasoning_tokens":"7"}}),
            json!({"completion_tokens":10,"completion_tokens_details":{"reasoning_tokens":null}}),
            json!({"completion_tokens":10,"completion_tokens_details":[]}),
            json!({"completion_tokens":10}),
            json!({"completion_tokens":null,"reasoning_tokens":0}),
        ] {
            assert_eq!(native_reasoning_tokens(&json!({"usage":usage})), None);
        }
    }
}

fn native_request_ids(value: &Value) -> Option<Vec<String>> {
    let rows = value.get("loads")?.as_array()?;
    if rows.is_empty() {
        return None;
    }
    let mut ids = Vec::new();
    let mut unique = BTreeSet::new();
    for row in rows {
        let counted = row
            .get("num_running_reqs")?
            .as_u64()?
            .checked_add(row.get("num_waiting_reqs")?.as_u64()?)?;
        let raw = row.get("active_request_ids")?.as_array()?;
        if u64::try_from(raw.len()).ok()? != counted {
            return None;
        }
        // The producer bounds its complete MsgPack snapshot and publishes null
        // IDs on overflow. JSON expansion is not an equivalent receiver bound.
        for id in raw {
            let id = id.as_str()?;
            if id.is_empty() || !unique.insert(id.to_owned()) {
                return None;
            }
            ids.push(id.to_owned());
        }
    }
    Some(ids)
}

fn native_timestamp_within_clock_skew(timestamp: f64, now: f64) -> bool {
    const MAX_FUTURE_CLOCK_SKEW_SECS: f64 = 0.1;
    timestamp.is_finite() && timestamp > 0.0 && timestamp - now <= MAX_FUTURE_CLOCK_SKEW_SECS
}

fn native_snapshot_time(value: &Value, format: NativeIdFormat) -> Option<tokio::time::Instant> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs_f64();
    let rows = value.get("loads")?.as_array()?;
    let oldest = rows
        .iter()
        .map(|row| {
            let captured = row.get("timestamp")?.as_f64()?;
            if !native_timestamp_within_clock_skew(captured, now) {
                return None;
            }
            if format == NativeIdFormat::VllmChatId {
                // The vLLM utility samples on the scheduler thread at GET time.
                // It has no frozen-payload heartbeat publication phase.
                return Some(captured);
            }
            let frozen = row.get("frozen")?.as_bool()?;
            let generation = row.get("snapshot_generation")?.as_u64()?;
            if generation == 0 {
                return None;
            }
            let heartbeat = row.get("heartbeat_timestamp")?.as_f64()?;
            if heartbeat != 0.0 && !native_timestamp_within_clock_skew(heartbeat, now) {
                return None;
            }
            if frozen {
                // The producer refreshes this only while the captured payload
                // belongs to an immutable scheduler region.
                if heartbeat < captured {
                    return None;
                }
                Some(heartbeat)
            } else if heartbeat == 0.0 {
                Some(captured)
            } else {
                None
            }
        })
        .collect::<Option<Vec<_>>>()?
        .into_iter()
        .reduce(f64::min)?;
    if !native_timestamp_within_clock_skew(oldest, now) || now - oldest > 2.5 {
        return None;
    }
    tokio::time::Instant::now().checked_sub(Duration::from_secs_f64((now - oldest).max(0.0)))
}

#[cfg(test)]
mod snapshot_freshness_tests {
    use super::{NativeIdFormat, native_snapshot_time};
    use serde_json::{Value, json};

    fn valid(value: &Value, format: NativeIdFormat) -> bool {
        native_snapshot_time(value, format).is_some()
    }

    fn sgl(value: &Value) -> bool {
        valid(value, NativeIdFormat::SglangPairedRid)
    }

    #[test]
    fn frozen_heartbeat_only_extends_freshness_with_immutable_proof() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        let captured = now - 10.0;
        let fresh = now - 0.1;
        let row = json!({"timestamp":captured,"frozen":true,"snapshot_generation":7,"heartbeat_timestamp":fresh});
        assert!(sgl(&json!({"loads":[row]})));
        assert!(!sgl(
            &json!({"loads":[{"timestamp":captured,"frozen":false,"snapshot_generation":7,"heartbeat_timestamp":0}]})
        ));
        assert!(!sgl(
            &json!({"loads":[{"timestamp":captured,"frozen":false,"snapshot_generation":7,"heartbeat_timestamp":fresh}]})
        ));
        assert!(!sgl(
            &json!({"loads":[{"timestamp":captured,"frozen":true,"snapshot_generation":0,"heartbeat_timestamp":fresh}]})
        ));
        assert!(!sgl(
            &json!({"loads":[{"timestamp":captured,"frozen":true,"snapshot_generation":1.5,"heartbeat_timestamp":fresh}]})
        ));
        assert!(!sgl(
            &json!({"loads":[{"timestamp":captured,"frozen":true,"snapshot_generation":7,"heartbeat_timestamp":captured-1.0}]})
        ));
    }

    #[test]
    fn every_dp_row_must_have_complete_and_fresh_metadata() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        let recent = now - 0.1;
        let old = now - 10.0;
        assert!(valid(
            &json!({"loads":[{"timestamp":recent}]}),
            NativeIdFormat::VllmChatId
        ));
        assert!(!valid(
            &json!({"loads":[{"timestamp":old}]}),
            NativeIdFormat::VllmChatId
        ));
        assert!(sgl(
            &json!({"loads":[{"timestamp":recent,"frozen":false,"snapshot_generation":1,"heartbeat_timestamp":0}]})
        ));
        assert!(!sgl(&json!({"loads":[{"timestamp":recent}]})));
        assert!(!sgl(&json!({"loads":[
            {"timestamp":old,"frozen":true,"snapshot_generation":2,"heartbeat_timestamp":recent},
            {"timestamp":old,"frozen":true,"snapshot_generation":2,"heartbeat_timestamp":old}
        ]})));
        assert!(!sgl(&json!({"loads":[
            {"timestamp":recent},
            {"timestamp":recent,"frozen":false,"snapshot_generation":2,"heartbeat_timestamp":0}
        ]})));
        assert!(!sgl(
            &json!({"loads":[{"timestamp":recent,"frozen":true,"snapshot_generation":2}]})
        ));
        assert!(!sgl(
            &json!({"loads":[{"timestamp":recent,"frozen":false,"snapshot_generation":0,"heartbeat_timestamp":0}]})
        ));
    }

    #[test]
    fn delayed_rpc_uses_actual_capture_or_frozen_heartbeat_age() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        let stale = now - 2.7;
        let fresh = now - 0.1;
        // An RPC may have started before the 2.5-second bound. The native
        // scheduler's later capture is still fresh; the RPC start is not a
        // substitute for the qualified producer timestamp.
        assert!(valid(
            &json!({"loads":[{"timestamp":fresh}]}),
            NativeIdFormat::VllmChatId
        ));
        assert!(!valid(
            &json!({"loads":[{"timestamp":stale}]}),
            NativeIdFormat::VllmChatId
        ));
        assert!(sgl(
            &json!({"loads":[{"timestamp":stale,"frozen":true,"snapshot_generation":3,"heartbeat_timestamp":fresh}]})
        ));
        assert!(!sgl(
            &json!({"loads":[{"timestamp":stale,"frozen":false,"snapshot_generation":3,"heartbeat_timestamp":0}]})
        ));
    }

    #[test]
    fn bounded_future_clock_skew_does_not_extend_the_stale_window() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        let slight_future = now + 0.01;
        let too_far_future = now + 0.2;
        let stale = now - 2.7;
        assert!(valid(
            &json!({"loads":[{"timestamp":slight_future}]}),
            NativeIdFormat::VllmChatId
        ));
        assert!(!valid(
            &json!({"loads":[{"timestamp":too_far_future}]}),
            NativeIdFormat::VllmChatId
        ));
        assert!(sgl(
            &json!({"loads":[{"timestamp":slight_future,"frozen":false,"snapshot_generation":1,"heartbeat_timestamp":0}]})
        ));
        assert!(sgl(
            &json!({"loads":[{"timestamp":stale,"frozen":true,"snapshot_generation":1,"heartbeat_timestamp":slight_future}]})
        ));
        assert!(!sgl(
            &json!({"loads":[{"timestamp":stale,"frozen":true,"snapshot_generation":1,"heartbeat_timestamp":too_far_future}]})
        ));
        assert!(!sgl(
            &json!({"loads":[{"timestamp":stale,"frozen":false,"snapshot_generation":1,"heartbeat_timestamp":0}]})
        ));
    }
}

fn native_incarnation(value: &Value, expected_dp: usize) -> Option<String> {
    let rows = value.get("loads")?.as_array()?;
    if expected_dp == 0 || rows.len() != expected_dp {
        return None;
    }
    let mut identities = Vec::with_capacity(rows.len());
    let mut uuids = BTreeSet::new();
    for row in rows {
        let rank = usize::try_from(row.get("dp_rank")?.as_u64()?).ok()?;
        let incarnation = row.get("native_incarnation")?.as_str()?;
        if incarnation.len() != 32
            || !incarnation
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            || incarnation.as_bytes()[12] != b'4'
            || !(b'8'..=b'b').contains(&incarnation.as_bytes()[16])
            || !uuids.insert(incarnation)
        {
            return None;
        }
        identities.push((rank, incarnation));
    }
    identities.sort_unstable_by_key(|(rank, _)| *rank);
    if identities
        .iter()
        .enumerate()
        .any(|(expected, (rank, _))| expected != *rank)
    {
        return None;
    }
    // Each scheduler has its own startup UUID. The sorted complete DP vector
    // identifies this endpoint; any scheduler restart changes the shared hint domain.
    let encoded = serde_json::to_vec(&identities).ok()?;
    let mut digest = Sha256::new();
    digest.update(b"dynamo-native-dp-incarnation-v1\0");
    digest.update(encoded);
    Some(format!("{:x}", digest.finalize()))
}

#[cfg(test)]
mod incarnation_tests {
    use super::native_incarnation;
    use serde_json::json;

    #[test]
    fn complete_dp_incarnation_is_order_independent_and_restart_sensitive() {
        let a = "00000000000040008000000000000001";
        let b = "00000000000040008000000000000002";
        let c = "00000000000040008000000000000003";
        let first = json!({"loads":[{"dp_rank":0,"native_incarnation":a},{"dp_rank":1,"native_incarnation":b}]});
        let reversed = json!({"loads":[{"dp_rank":1,"native_incarnation":b},{"dp_rank":0,"native_incarnation":a}]});
        let restarted = json!({"loads":[{"dp_rank":0,"native_incarnation":a},{"dp_rank":1,"native_incarnation":c}]});
        assert_eq!(
            native_incarnation(&first, 2),
            native_incarnation(&reversed, 2)
        );
        assert_ne!(
            native_incarnation(&first, 2),
            native_incarnation(&restarted, 2)
        );
        assert!(native_incarnation(&first, 2).is_some());
        assert!(native_incarnation(&first, 3).is_none());
        assert!(native_incarnation(&json!({"loads":[{"dp_rank":0,"native_incarnation":a},{"dp_rank":0,"native_incarnation":b}]}), 2).is_none());
        assert!(native_incarnation(&json!({"loads":[{"dp_rank":0,"native_incarnation":a},{"dp_rank":1,"native_incarnation":a}]}), 2).is_none());
        assert!(native_incarnation(&json!({"loads":[{"native_incarnation":a}]}), 1).is_none());
        assert!(native_incarnation(&json!({"loads":[{"dp_rank":0,"native_incarnation":"00000000000000008000000000000001"}]}), 1).is_none());
    }
}

#[cfg(test)]
mod exact_id_tests {
    use super::{ExactLoadSnapshot, NativeIdFormat, NativeIdMatch, PendingNative};
    use std::collections::BTreeSet;

    #[test]
    fn compact_sgl_and_vllm_ids_reject_noncanonical_indices() {
        let sgl = NativeIdFormat::SglangPairedRid;
        assert_eq!(
            sgl.match_id("owned", 1, "owned"),
            Some(NativeIdMatch::Choice(0, None))
        );
        assert_eq!(
            sgl.match_id("owned", 3, "owned_0:prefill"),
            Some(NativeIdMatch::Prefill)
        );
        assert_eq!(
            sgl.match_id("owned", 3, "owned_0:sample:2"),
            Some(NativeIdMatch::Choice(2, None))
        );
        assert!(sgl.match_id("owned", 3, "owned_0:sample:02").is_none());
        assert!(
            sgl.match_id("owned", 3, "owned_0:sample:4294967296")
                .is_none()
        );

        let vllm = NativeIdFormat::VllmChatId;
        assert_eq!(
            vllm.match_id("owned", 1, "chatcmpl-owned-a1b2c3d4"),
            Some(NativeIdMatch::Choice(0, Some("a1b2c3d4".into())))
        );
        assert_eq!(
            vllm.match_id("owned", 3, "2_chatcmpl-owned-a1b2c3d4"),
            Some(NativeIdMatch::Choice(2, Some("a1b2c3d4".into())))
        );
        assert!(
            vllm.match_id("owned", 3, "02_chatcmpl-owned-a1b2c3d4")
                .is_none()
        );
        assert!(
            vllm.match_id("owned", 3, "4294967296_chatcmpl-owned-a1b2c3d4")
                .is_none()
        );
        assert!(
            vllm.match_id("owned", 3, "2_chatcmpl-owned-badsalt")
                .is_none()
        );
        assert!(
            vllm.match_id("owned", 3, "2_chatcmpl-owned-A1B2C3D4")
                .is_none()
        );
    }

    #[test]
    fn vllm_salt_conflict_rejects_snapshot_without_rewriting_accepted_choices() {
        let mut pending = PendingNative {
            worker_url: "http://worker".into(),
            n: 2,
            incarnation: None,
            accepted_choices: BTreeSet::new(),
            vllm_salt: None,
            live_prefill: false,
            restarted: false,
        };
        let sample = |ids: Vec<&str>, incarnation: &str| ExactLoadSnapshot {
            native_total: ids.len() as u64,
            ids: ids.into_iter().map(str::to_owned).collect(),
            incarnation: incarnation.into(),
            sampled_at: tokio::time::Instant::now(),
        };
        let first = sample(vec!["0_chatcmpl-owned-a1b2c3d4"], "incarnation1");
        assert!(pending.reconcile(&first, NativeIdFormat::VllmChatId, "owned"));
        assert_eq!(pending.pending_weight(), 1);
        let mut trial = pending.clone();
        let conflict = sample(vec!["1_chatcmpl-owned-ffffffff"], "incarnation1");
        assert!(!trial.reconcile(&conflict, NativeIdFormat::VllmChatId, "owned"));
        assert_eq!(pending.pending_weight(), 1);
        assert!(pending.reconcile(
            &sample(vec![], "incarnation1"),
            NativeIdFormat::VllmChatId,
            "owned"
        ));
        assert_eq!(pending.pending_weight(), 1);
        assert!(pending.reconcile(
            &sample(vec![], "incarnation2"),
            NativeIdFormat::VllmChatId,
            "owned"
        ));
        assert_eq!(pending.pending_weight(), 2);
        assert!(pending.reconcile(
            &sample(vec!["0_chatcmpl-owned-a1b2c3d4"], "incarnation2"),
            NativeIdFormat::VllmChatId,
            "owned"
        ));
        assert_eq!(pending.pending_weight(), 2);
    }

    #[test]
    fn sgl_prefill_is_only_live_coverage_until_sample_is_accepted() {
        let mut pending = PendingNative {
            worker_url: "http://worker".into(),
            n: 3,
            incarnation: None,
            accepted_choices: BTreeSet::new(),
            vllm_salt: None,
            live_prefill: false,
            restarted: false,
        };
        let sample = |ids: Vec<&str>| ExactLoadSnapshot {
            native_total: ids.len() as u64,
            ids: ids.into_iter().map(str::to_owned).collect(),
            incarnation: "same-incarnation".into(),
            sampled_at: tokio::time::Instant::now(),
        };
        assert!(pending.reconcile(
            &sample(vec!["owned_0:prefill"]),
            NativeIdFormat::SglangPairedRid,
            "owned"
        ));
        assert_eq!(pending.pending_weight(), 2);
        assert!(pending.reconcile(&sample(vec![]), NativeIdFormat::SglangPairedRid, "owned"));
        assert_eq!(pending.pending_weight(), 3);
        assert!(pending.reconcile(
            &sample(vec!["owned_0:sample:0"]),
            NativeIdFormat::SglangPairedRid,
            "owned"
        ));
        assert_eq!(pending.pending_weight(), 2);
        assert!(pending.reconcile(&sample(vec![]), NativeIdFormat::SglangPairedRid, "owned"));
        assert_eq!(pending.pending_weight(), 2);
    }
}

fn complete_json_choices(value: &Value, expected: usize) -> bool {
    let Some(choices) = value.get("choices").and_then(Value::as_array) else {
        return false;
    };
    let indices: Option<BTreeSet<_>> = choices
        .iter()
        .map(|choice| choice.get("index")?.as_u64())
        .collect();
    choices.len() == expected
        && indices.is_some_and(|indices| {
            indices.len() == expected && indices.iter().all(|index| *index < expected as u64)
        })
}

fn record_choice_terminals(value: &Value, expected: usize, completed: &mut BTreeSet<u64>) -> bool {
    let Some(choices) = value.get("choices").and_then(Value::as_array) else {
        return false;
    };
    let mut seen = BTreeSet::new();
    for choice in choices {
        let Some(index) = choice.get("index").and_then(Value::as_u64) else {
            return false;
        };
        if index >= expected as u64 || completed.contains(&index) || !seen.insert(index) {
            return false;
        }
        if let Some(reason) = choice
            .get("finish_reason")
            .filter(|reason| !reason.is_null())
        {
            if !reason.as_str().is_some_and(|reason| {
                matches!(
                    reason,
                    "stop" | "length" | "tool_calls" | "content_filter" | "function_call"
                )
            }) {
                return false;
            }
            completed.insert(index);
        }
    }
    true
}

fn native_load_parts(value: &Value) -> Option<(usize, usize)> {
    let rows = value.get("loads")?.as_array()?;
    if rows.is_empty() {
        return None;
    }
    rows.iter()
        .try_fold((0usize, 0usize), |(running, waiting), row| {
            Some((
                running
                    .checked_add(usize::try_from(row.get("num_running_reqs")?.as_u64()?).ok()?)?,
                waiting
                    .checked_add(usize::try_from(row.get("num_waiting_reqs")?.as_u64()?).ok()?)?,
            ))
        })
}

fn valid_terminal(value: &Value) -> bool {
    value
        .get("choices")
        .and_then(Value::as_array)
        .is_some_and(|choices| {
            !choices.is_empty()
                && choices.iter().all(|choice| {
                    choice
                        .get("finish_reason")
                        .and_then(Value::as_str)
                        .is_some_and(|r| {
                            matches!(
                                r,
                                "stop"
                                    | "length"
                                    | "tool_calls"
                                    | "content_filter"
                                    | "function_call"
                            )
                        })
                })
        })
}

fn event_boundary(bytes: &[u8]) -> Option<(usize, usize)> {
    for i in 0..bytes.len() {
        if bytes[i..].starts_with(b"\n\n") {
            return Some((i, 2));
        }
        if bytes[i..].starts_with(b"\r\n\r\n") {
            return Some((i, 4));
        }
    }
    None
}

#[async_trait]
impl StreamingEngine for NativeHttpEngine {
    async fn handle_chat(
        &self,
        req: SingleIn<NvCreateChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> {
        // Absence is an integration error; never fall back to lossy DTO serialization.
        let body = req
            .get::<Value>(NATIVE_HTTP_BODY)
            .map_err(|_| invalid("native_http_body was not preserved at HTTP boundary"))?;
        let (_, context) = req.into_parts();
        let ctx = context.context();
        let raw = self.forward_chat((*body).clone(), ctx.clone()).await?;
        let output = raw.map(|item| {
            let Some(mut value) = item.data else {
                return Annotated {
                    data: None,
                    id: item.id,
                    event: item.event,
                    comment: item.comment,
                    error: item.error,
                };
            };
            // Dynamo aggregates chunks for nonstreaming callers. Convert the
            // native message to one delta without synthesizing a finish reason.
            if let Some(choices) = value.get_mut("choices").and_then(Value::as_array_mut) {
                for choice in choices {
                    if let Some(message) = choice.as_object_mut().and_then(|c| c.remove("message"))
                    {
                        choice["delta"] = message;
                    }
                }
            }
            match serde_json::from_value(value) {
                Ok(data) => Annotated::from_data(data),
                Err(_) => Annotated::from_error(
                    "native response cannot be represented by Dynamo Chat DTO",
                ),
            }
        });
        Ok(ResponseStream::new(Box::pin(output), ctx))
    }

    async fn handle_completion(
        &self,
        _: SingleIn<NvCreateCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateCompletionResponse>>, Error> {
        Err(invalid(
            "native HTTP engine supports Chat Completions; legacy text completions are unsupported",
        ))
    }
}

#[cfg(test)]
#[path = "native_http_ha_tests.rs"]
mod native_http_ha_tests;

#[cfg(test)]
#[path = "native_cache_terminal_usage_tests.rs"]
mod native_cache_terminal_usage_tests;

#[cfg(test)]
#[path = "native_cache_http_wire_tests.rs"]
mod native_cache_http_wire_tests;

#[cfg(test)]
#[path = "native_cache_stale_http_tests.rs"]
mod native_cache_stale_http_tests;

#[cfg(test)]
#[path = "native_cache_startup_dispatch_tests.rs"]
mod native_cache_startup_dispatch_tests;

#[cfg(test)]
mod count_total_budget_tests {
    use super::*;
    use crate::cache_authority::frontend_cache_hook::{NativeIdField, QualifiedWorker};
    use crate::cache_authority::native_cache_frontend::FrontendWorkerBinding;
    use crate::cache_authority::{EtcdStore, cache_auth};
    use axum::{body::Body, http::Response};
    use serde_json::json;
    use tokio::sync::Notify;

    const CPU_KEY: [u8; 32] = [b'H'; 32];

    fn valid_count_reply(request: &Value) -> Value {
        if request.get(cache_auth::PROOF_FIELD).is_none() {
            return json!({"count":7});
        }
        cache_auth::verify_signature(
            &CPU_KEY,
            &request[cache_auth::PROOF_FIELD],
            &cache_auth::payload_for("render", request).unwrap(),
        )
        .unwrap();
        let mut claims = request[cache_auth::PROOF_FIELD]["claims"].clone();
        let tokens = json!((0..7).collect::<Vec<u32>>());
        let mut reply = json!({"tokens":tokens,"count":7,"max_model_len":1_048_576,
            "prompt_cache_provenance":{"native_origin_uuid":claims["native_incarnation"],
                "runtime_profile_digest":claims["runtime_profile_digest"],
                "namespace_digest":claims["namespace_digest"],"full_prompt_digest":"4".repeat(64),
                "token_ids_digest":cache_auth::digest(&tokens).unwrap(),"num_prompt_tokens":7,"endpoints":[]}});
        claims["full_render_digest"] =
            reply["prompt_cache_provenance"]["full_prompt_digest"].clone();
        claims["token_ids_digest"] = reply["prompt_cache_provenance"]["token_ids_digest"].clone();
        let digest = cache_auth::digest(&reply).unwrap();
        for key in ["operation_id", "nonce"] {
            claims[key] = cache_auth::digest(&json!([
                "native-response",
                request[cache_auth::PROOF_FIELD]["claims"][key],
                digest
            ]))
            .unwrap()
            .into();
        }
        reply[cache_auth::PROOF_FIELD] = cache_auth::sign(&CPU_KEY, &claims, &reply).unwrap();
        reply
    }

    #[tokio::test]
    async fn count_total_deadline_bounds_failover_and_stops_after_success() {
        let hits = Arc::new((0..14).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>());
        let mode = Arc::new(AtomicUsize::new(0));
        let first_entered = Arc::new(Notify::new());
        let release_first = Arc::new(Notify::new());
        let decode_entered = Arc::new(Notify::new());
        let signed_starts = Arc::new(Mutex::new(Vec::new()));
        let mut urls = Vec::new();
        let mut servers = Vec::new();
        for index in 0..hits.len() {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            urls.push(format!("http://{}", listener.local_addr().unwrap()));
            let (hits, mode, entered, release, decoding) = (
                hits.clone(),
                mode.clone(),
                first_entered.clone(),
                release_first.clone(),
                decode_entered.clone(),
            );
            let starts = signed_starts.clone();
            let router =
                axum::Router::new().fallback(move |axum::Json(request): axum::Json<Value>| {
                    let (hits, mode, entered, release, decoding) = (
                        hits.clone(),
                        mode.clone(),
                        entered.clone(),
                        release.clone(),
                        decoding.clone(),
                    );
                    let starts = starts.clone();
                    async move {
                        hits[index].fetch_add(1, Ordering::SeqCst);
                        if let Some(proof) = request.get(cache_auth::PROOF_FIELD) {
                            cache_auth::verify_signature(
                                &CPU_KEY,
                                proof,
                                &cache_auth::payload_for("render", &request).unwrap(),
                            )
                            .unwrap();
                            starts
                                .lock()
                                .unwrap()
                                .push(proof["claims"]["request_started_at_ms"].as_u64().unwrap());
                        }
                        match mode.load(Ordering::SeqCst) {
                            1 => Response::new(Body::from(valid_count_reply(&request).to_string())),
                            2 => Response::builder().status(422).body(Body::empty()).unwrap(),
                            3 => Response::builder().status(503).body(Body::from(
                                json!({"error":"native explicit cache unavailable: native backend disconnected"}).to_string(),
                            )).unwrap(),
                            _ if index == 0 => {
                                entered.notify_one();
                                release.notified().await;
                                Response::builder().status(503).body(Body::empty()).unwrap()
                            }
                            _ => Response::new(Body::from_stream(async_stream::stream! {
                                // Headers succeed; JSON decoding, not send(), then stalls.
                                decoding.notify_one();
                                std::future::pending::<()>().await;
                                yield Ok::<_, std::convert::Infallible>(bytes::Bytes::new());
                            })),
                        }
                    }
                });
            servers.push(tokio::spawn(async move {
                axum::serve(listener, router).await.unwrap()
            }));
        }
        let engine = || {
            Arc::new(
                NativeHttpEngine::new(NativeHttpEngineConfig {
                    model: "glm-5.3-flash".into(),
                    qualification_tier: None,
                    workers: urls
                        .iter()
                        .map(|url| NativeWorkerConfig {
                            url: url.clone(),
                            tier: NativeTier::Shared,
                            max_inflight: None,
                            load_path: Some("/v1/loads?include=core".into()),
                            max_input_tokens: None,
                            capability_source: None,
                            active_request_ids_source: Some(
                                "CPU Count deadline fixture only".into(),
                            ),
                            native_dp_size: Some(1),
                            native_id_format: NativeIdFormat::VllmChatId,
                            identity: None,
                            thinking_budget_qualification: None,
                            forced_zero_qualification: None,
                        })
                        .collect(),
                })
                .unwrap(),
            )
        };
        let body = json!({"model":"glm-5.3-flash","messages":[{"role":"user","content":"count"}]});
        let signed_engine = Arc::try_unwrap(engine()).ok().unwrap();
        let bindings = urls
            .iter()
            .map(|url| FrontendWorkerBinding {
                worker: QualifiedWorker {
                    worker_url: url.clone(),
                    native_origin_uuid: "00000000000040008000000000000001".into(),
                    runtime_profile_digest: "2".repeat(64),
                    namespace_digest: "3".repeat(64),
                    native_clock_domain: "cpu-count-only".into(),
                    required_tiers: BTreeSet::from(["l1".into()]),
                    native_id_field: NativeIdField::RequestId,
                    rank_tier_clocks: BTreeMap::from([(
                        0,
                        BTreeMap::from([("l1".into(), "cpu-count-only".into())]),
                    )]),
                },
                model: "glm-5.3-flash".into(),
                source_digest: "1".repeat(64),
                kv_compat_digest: "5".repeat(64),
                count_path: "/tokenize".into(),
                tenant: "cpu-count-only".into(),
                default_cache_salt: "cpu-count-only".into(),
                secret: CPU_KEY,
                shared_backend: None,
            })
            .collect();
        // The established client is lazy and Count performs no etcd operation.
        // No etcd service, cache reservation or native generation is started.
        let store = EtcdStore::connect(vec![urls[0].clone()]).await.unwrap();
        let frontend = Arc::new(
            NativeCacheFrontend::new(signed_engine.native_http_client(), store, bindings)
                .unwrap()
                .use_cpu_fixture_clock()
                .with_cleanup_registry(signed_engine.cleanup_registry()),
        );
        let signed_engine = Arc::new(signed_engine.with_cache_frontend(frontend).unwrap());
        let ctx = dynamo_runtime::pipeline::Context::with_id_and_metadata(
            body.clone(),
            "count-budget-cpu".into(),
            BTreeMap::new(),
        )
        .context();
        // Keep the executor runnable so real loopback IO cannot auto-advance
        // paused time. Only explicit advances consume the tested Count budget.
        let keep_clock = || {
            tokio::spawn(async {
                loop {
                    tokio::task::yield_now().await;
                }
            })
        };
        let clock_keeper = keep_clock();
        tokio::time::pause();
        let began = tokio::time::Instant::now();
        let (count_engine, count_body, count_ctx) = (engine(), body.clone(), ctx.clone());
        let mut count = tokio::spawn(async move {
            count_engine
                .count(&count_body, &count_ctx, began + Duration::from_secs(30))
                .await
        });
        tokio::select! {
            _ = first_entered.notified() => {},
            result = &mut count => panic!("ordinary Count ended before first send: {result:?}"),
        }
        eprintln!("COUNT_CAUSAL_STAGE ordinary_first_send");
        tokio::time::advance(Duration::from_secs(20)).await;
        release_first.notify_one();
        tokio::select! {
            _ = decode_entered.notified() => {},
            result = &mut count => panic!("ordinary Count ended before fallback body: {result:?}"),
        }
        eprintln!("COUNT_CAUSAL_STAGE ordinary_fallback_body");
        tokio::time::advance(Duration::from_millis(9_999)).await;
        assert!(!count.is_finished());
        tokio::time::advance(Duration::from_millis(1)).await;
        // Advance the timer wheel explicitly; unrelated loopback/tonic tasks
        // must not determine paused-time auto-advance or the measured budget.
        tokio::time::advance(Duration::from_millis(1)).await;
        let error = count.await.unwrap().unwrap_err();
        assert!(error.downcast_ref::<DynamoError>().is_some());
        let ordinary_elapsed = tokio::time::Instant::now() - began;
        // Tokio's timer wheel rounds up to a millisecond; the deadline itself
        // stays exactly 30s, including the pre-launch monotonic fence.
        assert!(ordinary_elapsed >= Duration::from_secs(30));
        assert!(
            ordinary_elapsed <= Duration::from_millis(30_001),
            "serial counters must share one deadline, observed {ordinary_elapsed:?}"
        );
        eprintln!("COUNT_CAUSAL_STAGE ordinary_deadline");
        let calls = || {
            hits.iter()
                .map(|hit| hit.load(Ordering::SeqCst))
                .collect::<Vec<_>>()
        };
        assert_eq!(calls(), [vec![1, 1], vec![0; 12]].concat());
        tokio::time::advance(Duration::from_secs(510)).await;
        tokio::task::yield_now().await;
        assert_eq!(
            calls(),
            [vec![1, 1], vec![0; 12]].concat(),
            "expired traversal must not launch another worker"
        );

        mode.store(1, Ordering::SeqCst);
        assert_eq!(
            engine()
                .count(
                    &body,
                    &ctx,
                    tokio::time::Instant::now() + Duration::from_secs(30)
                )
                .await
                .unwrap(),
            7
        );
        assert_eq!(
            calls(),
            [vec![2, 1], vec![0; 12]].concat(),
            "first successful Count ends traversal"
        );
        mode.store(2, Ordering::SeqCst);
        assert!(
            engine()
                .count(
                    &body,
                    &ctx,
                    tokio::time::Instant::now() + Duration::from_secs(30)
                )
                .await
                .unwrap_err()
                .downcast_ref::<DynamoError>()
                .is_some()
        );
        assert_eq!(
            calls(),
            [vec![3, 1], vec![0; 12]].concat(),
            "canonical rejection is not retried"
        );
        let before_signed = calls();
        eprintln!("COUNT_CAUSAL_STAGE ordinary_success_and_rejection");
        mode.store(0, Ordering::SeqCst);
        let mut signed_body = body.clone();
        signed_body["request_id"] = "owned-signed-count".into();
        let request = Arc::new(CacheRequest::new(
            ctx.id().into(),
            Instant::now(),
            BTreeMap::new(),
            BTreeSet::new(),
        ));
        let signed_began = tokio::time::Instant::now();
        let signed_deadline = signed_began + Duration::from_secs(30);
        let (candidate, signed_input, signed_ctx, signed_request) = (
            signed_engine.clone(),
            signed_body.clone(),
            ctx.clone(),
            request.clone(),
        );
        let mut signed = tokio::spawn(async move {
            candidate
                .signed_count(&signed_input, &signed_ctx, &signed_request, signed_deadline)
                .await
        });
        tokio::select! {
            _ = first_entered.notified() => {},
            result = &mut signed => panic!("signed Count ended before first send: {}", result.unwrap().err().unwrap()),
        }
        eprintln!("COUNT_CAUSAL_STAGE signed_first_send");
        tokio::time::advance(Duration::from_secs(20)).await;
        release_first.notify_one();
        tokio::select! {
            _ = decode_entered.notified() => {},
            result = &mut signed => panic!("signed Count ended before fallback body: {}", result.unwrap().err().unwrap()),
        }
        eprintln!("COUNT_CAUSAL_STAGE signed_fallback_body");
        tokio::time::advance(Duration::from_secs(10)).await;
        tokio::time::advance(Duration::from_millis(1)).await;
        assert!(
            signed
                .await
                .unwrap()
                .err()
                .unwrap()
                .downcast_ref::<DynamoError>()
                .is_some()
        );
        let signed_elapsed = tokio::time::Instant::now() - signed_began;
        assert!(signed_elapsed >= Duration::from_secs(30));
        assert!(
            signed_elapsed <= Duration::from_millis(30_001),
            "signed counters must share the same deadline, observed {signed_elapsed:?}"
        );
        assert_eq!(
            calls(),
            [
                vec![before_signed[0] + 1, before_signed[1] + 1],
                vec![0; 12]
            ]
            .concat()
        );
        let starts = signed_starts.lock().unwrap();
        assert_eq!(starts.len(), 2);
        assert_eq!(
            starts[0], starts[1],
            "failover must preserve the original qualified request start"
        );
        drop(starts);
        let after_signed = calls();
        // A different selected worker must not acquire a second Count budget.
        assert!(
            signed_engine
                .render_count_until(&urls[2], &signed_body, &ctx, &request, signed_deadline)
                .await
                .err()
                .unwrap()
                .downcast_ref::<DynamoError>()
                .is_some()
        );
        assert_eq!(calls(), after_signed);
        mode.store(1, Ordering::SeqCst);
        let rendered = signed_engine
            .signed_count(
                &signed_body,
                &ctx,
                &request,
                tokio::time::Instant::now() + Duration::from_secs(30),
            )
            .await
            .unwrap();
        assert_eq!(rendered.prompt_tokens, 7);
        assert_eq!(
            calls().iter().sum::<usize>(),
            after_signed.iter().sum::<usize>() + 1,
            "first signed success ends traversal and retains validated token provenance"
        );
        // Actual saved Qwen sequence: native tokenize503, then a disconnected
        // failover counter. Exercise the real signed Count/HTTP parser path.
        while servers.len() > 1 {
            let server = servers.pop().unwrap();
            server.abort();
            let _ = server.await;
        }
        signed_engine.counter_cursor.store(0, Ordering::SeqCst);
        mode.store(3, Ordering::SeqCst);
        let error = signed_engine.signed_count(
            &signed_body, &ctx, &request,
            tokio::time::Instant::now() + Duration::from_secs(30),
        ).await.err().expect("all counters are unavailable");
        assert!(matches!(error.downcast_ref::<crate::cache_authority::Error>(),
            Some(crate::cache_authority::Error::NativeHttp { status: 503, .. })));
        let public = crate::http::service::ErrorMessage::from_anyhow(
            error, "Native token count failed",
        );
        assert_eq!(public.0, axum::http::StatusCode::SERVICE_UNAVAILABLE);
        let before_cancel = calls();
        ctx.stop();
        assert!(
            engine()
                .count(
                    &body,
                    &ctx,
                    tokio::time::Instant::now() + Duration::from_secs(30)
                )
                .await
                .unwrap_err()
                .to_string()
                .contains("request cancelled")
        );
        assert!(
            signed_engine
                .render_count_until(&urls[2], &signed_body, &ctx, &request, signed_deadline)
                .await
                .err()
                .unwrap()
                .to_string()
                .contains("request cancelled")
        );
        assert_eq!(
            calls(),
            before_cancel,
            "cancellation wins even when the Count deadline is ready"
        );
        eprintln!(
            "COUNT_CAUSAL_CPU ordinary_total={ordinary_elapsed:?} signed_total={signed_elapsed:?} workers=14 expired_recount_calls=0 generation_calls=0 old_matrix=0"
        );
        clock_keeper.abort();
        let _ = clock_keeper.await;
        for server in servers {
            server.abort();
            let _ = server.await;
        }
    }
}
