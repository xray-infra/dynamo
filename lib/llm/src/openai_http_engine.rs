// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenAI-compatible HTTP transport for the upstream in-process text engine.
//! Public handlers, protocol conversion, aggregation and usage belong to Dynamo.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::{bail, ensure};
use async_stream::stream;
use async_trait::async_trait;
use dynamo_runtime::engine::{AsyncEngineContextProvider, ResponseStream};
use dynamo_runtime::pipeline::{Error, ManyOut, SingleIn};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::StreamExt;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::engines::StreamingEngine;
use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
};
use crate::protocols::openai::completions::{
    NvCreateCompletionRequest, NvCreateCompletionResponse,
};

const MAX_EVENT_BYTES: usize = 8 * 1024 * 1024;

pub struct OpenAiHttpEngine {
    client: reqwest::Client,
    workers: Vec<reqwest::Url>,
    next: AtomicUsize,
}

impl OpenAiHttpEngine {
    fn worker_index(&self, body: &Value) -> usize {
        let Some(key) = body
            .get("prompt_cache_key")
            .and_then(Value::as_str)
            .filter(|key| !key.is_empty())
        else {
            return self.next.fetch_add(1, Ordering::Relaxed) % self.workers.len();
        };
        // Rendezvous affinity is a routing hint, not evidence of resident KV.
        // Length framing separates model, namespace, key and worker identity;
        // scores are stable across frontend restart and worker-list reordering.
        let model = body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let salt = body
            .get("cache_salt")
            .and_then(Value::as_str)
            .unwrap_or_default();
        self.workers
            .iter()
            .enumerate()
            .max_by_key(|(_, worker)| {
                let mut hash = Sha256::new();
                for value in [model, salt, key, worker.as_str()] {
                    hash.update((value.len() as u64).to_be_bytes());
                    hash.update(value.as_bytes());
                }
                let digest = hash.finalize();
                u64::from_be_bytes(digest[..8].try_into().unwrap())
            })
            .unwrap()
            .0
    }

    pub fn new(workers: Vec<String>) -> anyhow::Result<Self> {
        ensure!(
            !workers.is_empty() && workers.len() <= 64,
            "configure 1..64 worker origins"
        );
        let workers = workers
            .into_iter()
            .map(|origin| {
                let url = reqwest::Url::parse(&origin)?;
                ensure!(
                    matches!(url.scheme(), "http" | "https")
                        && url.host_str().is_some()
                        && url.username().is_empty()
                        && url.password().is_none()
                        && url.path() == "/"
                        && url.query().is_none()
                        && url.fragment().is_none(),
                    "worker must be an HTTP origin without credentials, path or query"
                );
                Ok(url.join("v1/chat/completions")?)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(600))
            .build()?;
        Ok(Self {
            client,
            workers,
            next: AtomicUsize::new(0),
        })
    }
}

fn worker_request(mut request: NvCreateChatCompletionRequest) -> anyhow::Result<Value> {
    // Dynamo owns canonical/legacy salt precedence; HTTP workers own the
    // top-level cache_salt field. Preserve this isolation boundary explicitly.
    let cache_salt =
        crate::protocols::common::extensions::request_cache_salt(&request).map(str::to_owned);
    if request
        .unsupported_fields
        .get("prompt_cache_retention")
        .is_some_and(|value| !value.is_null())
    {
        return Err(crate::http::service::error::invalid_argument(
            "prompt_cache_retention is not supported by this HTTP backend; automatic prefix caching remains available",
        ).into());
    }
    request.normalize_reasoning_template_args()?;
    let mut body = serde_json::to_value(request)?;
    if let Some(cache_salt) = cache_salt {
        body["cache_salt"] = json!(cache_salt);
    }
    // Dynamo owns canonical chat_template_args; OpenAI HTTP workers consume
    // chat_template_kwargs. Normalize once at this transport boundary.
    if let Some(arguments) = body.as_object_mut().unwrap().remove("chat_template_args") {
        body["chat_template_kwargs"] = arguments;
    }
    body["stream"] = json!(true);
    body["stream_options"] = json!({"include_usage": true});
    Ok(body)
}

fn decode_event(data: &str) -> anyhow::Result<NvCreateChatCompletionStreamResponse> {
    let chunk: NvCreateChatCompletionStreamResponse = serde_json::from_str(data)?;
    // Required identity comes from the backend, never a synthesized template.
    ensure!(
        !chunk.inner.id.is_empty() && !chunk.inner.model.is_empty(),
        "missing backend identity"
    );
    Ok(chunk)
}

#[async_trait]
impl StreamingEngine for OpenAiHttpEngine {
    async fn handle_chat(
        &self,
        req: SingleIn<NvCreateChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> {
        let (request, context) = req.into_parts();
        let ctx = context.context();
        // Obtain native usage for upstream unary aggregation and Messages,
        // independent of the public client's stream_options.
        let body = worker_request(request)?;
        let choices = body.get("n").and_then(Value::as_u64).unwrap_or(1) as usize;
        let worker = &self.workers[self.worker_index(&body)];
        let response = tokio::select! {
            _ = ctx.stopped() => bail!("request cancelled before native HTTP headers"),
            response = self.client.post(worker.clone()).json(&body).send() => response?,
        };
        ensure!(
            response.status() == reqwest::StatusCode::OK,
            "backend HTTP status {}",
            response.status()
        );
        ensure!(
            response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .is_some_and(|v| v.starts_with("text/event-stream")),
            "backend did not return an SSE stream"
        );
        let stream_context = ctx.clone();
        let output = stream! {
            let mut bytes = response.bytes_stream();
            let mut buffer = Vec::new();
            let mut completed = BTreeSet::new();
            loop {
                let part = tokio::select! {
                    _ = stream_context.stopped() => {
                        yield Annotated::from_error("backend request cancelled");
                        return;
                    },
                    part = bytes.next() => part,
                };
                let Some(part) = part else {
                    yield Annotated::from_error("backend SSE closed before DONE");
                    return;
                };
                let part = match part {
                    Ok(part) => part,
                    Err(_) => { yield Annotated::from_error("backend SSE transport failed"); return; }
                };
                buffer.extend_from_slice(&part);
                loop {
                    let boundary = buffer.windows(2).position(|v| v == b"\n\n").map(|p| (p, 2))
                        .into_iter()
                        .chain(buffer.windows(4).position(|v| v == b"\r\n\r\n").map(|p| (p, 4)))
                        .min_by_key(|(position, _)| *position);
                    let Some((end, separator)) = boundary else { break; };
                    if end > MAX_EVENT_BYTES {
                        yield Annotated::from_error("backend SSE event too large"); return;
                    }
                    let frame = match std::str::from_utf8(&buffer[..end]) {
                        Ok(frame) => frame,
                        Err(_) => { yield Annotated::from_error("backend SSE is not UTF-8"); return; }
                    };
                    let data = frame.lines().filter_map(|line| line.strip_prefix("data:")
                        .map(|v| v.strip_prefix(' ').unwrap_or(v))).collect::<Vec<_>>().join("\n");
                    if data == "[DONE]" {
                        if completed.len() != choices {
                            yield Annotated::from_error("backend SSE ended without every choice terminal");
                        }
                        return;
                    }
                    if !data.is_empty() {
                        let chunk = match decode_event(&data) {
                            Ok(chunk) => chunk,
                            Err(_) => { yield Annotated::from_error("invalid backend completion chunk"); return; }
                        };
                        for choice in &chunk.inner.choices {
                            if choice.index as usize >= choices || (choice.finish_reason.is_some() && !completed.insert(choice.index)) {
                                yield Annotated::from_error("invalid backend choice terminal"); return;
                            }
                        }
                        yield Annotated::from_data(chunk);
                    }
                    buffer.drain(..end + separator);
                }
                if buffer.len() > MAX_EVENT_BYTES {
                    yield Annotated::from_error("backend SSE event too large"); return;
                }
            }
        };
        Ok(ResponseStream::new(Box::pin(output), ctx))
    }

    async fn handle_completion(
        &self,
        _: SingleIn<NvCreateCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateCompletionResponse>>, Error> {
        bail!("HTTP text adapter supports Chat Completions, not legacy completions")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_request_maps_canonical_and_legacy_template_controls() {
        for extra in [
            json!({"chat_template_args": {"enable_thinking": false, "test_custom": 7}}),
            json!({"chat_template_kwargs": {"enable_thinking": false, "test_custom": 7}}),
            json!({"thinking": {"type": "disabled"}}),
        ] {
            let mut raw =
                json!({"model": "glm-5.3-flash", "messages": [{"role": "user", "content": "OK"}]});
            raw.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            let body = worker_request(serde_json::from_value(raw).unwrap()).unwrap();
            assert!(body.get("chat_template_args").is_none());
            assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
            if extra.get("thinking").is_none() {
                assert_eq!(body["chat_template_kwargs"]["test_custom"], 7);
            }
            assert_eq!(body["stream"], true);
            assert_eq!(body["stream_options"]["include_usage"], true);
        }
    }

    #[test]
    fn worker_request_preserves_canonical_cache_salt_and_legacy_precedence() {
        for (extra, expected) in [
            (json!({}), None),
            (json!({"cache_salt": "legacy"}), Some("legacy")),
            (
                json!({"nvext": {"cache_salt": "canonical"}}),
                Some("canonical"),
            ),
            (
                json!({"cache_salt": "legacy", "nvext": {"cache_salt": "canonical"}}),
                Some("canonical"),
            ),
            (
                json!({"cache_salt": "legacy", "nvext": {"cache_salt": ""}}),
                Some("legacy"),
            ),
        ] {
            let mut raw = json!({"model": "glm-5.3-flash",
                "messages": [{"role": "user", "content": "OK"}]});
            raw.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            let body = worker_request(serde_json::from_value(raw).unwrap()).unwrap();
            assert_eq!(body.get("cache_salt").and_then(Value::as_str), expected);
        }
    }

    #[test]
    fn explicit_cache_key_has_stable_affinity_without_changing_worker_wire() {
        let engine = OpenAiHttpEngine::new(vec![
            "http://127.0.0.1:28100".into(),
            "http://127.0.0.1:28101".into(),
        ])
        .unwrap();
        let raw = json!({"model":"glm-5.3-flash", "messages":[{"role":"user","content":"OK"}],
            "prompt_cache_key":"prefix-key", "nvext":{"cache_salt":"tenant-A"}});
        let body = worker_request(serde_json::from_value(raw).unwrap()).unwrap();
        assert_eq!(body["prompt_cache_key"], "prefix-key");
        let first = engine.worker_index(&body);
        for _ in 0..16 {
            assert_eq!(engine.worker_index(&body), first);
        }
        let no_key = json!({"model":"glm-5.3-flash"});
        assert_ne!(engine.worker_index(&no_key), engine.worker_index(&no_key));
    }

    #[test]
    fn native_usage_only_chunk_retains_backend_envelope() {
        let raw = json!({"id":"chatcmpl-native", "object":"chat.completion.chunk",
            "created":7, "model":"glm-5.3-flash", "choices":[],
            "usage":{"prompt_tokens":100,"completion_tokens":7,"total_tokens":107,
                "prompt_tokens_details":{"cached_tokens":60}}});
        let chunk = decode_event(&raw.to_string()).unwrap();
        let roundtrip = serde_json::to_value(chunk).unwrap();
        for key in ["id", "object", "created", "model", "usage"] {
            assert_eq!(roundtrip[key], raw[key]);
        }
        for key in ["id", "object", "created", "model"] {
            let mut invalid = raw.clone();
            invalid.as_object_mut().unwrap().remove(key);
            assert!(decode_event(&invalid.to_string()).is_err());
        }
        assert!(decode_event(&json!({"choices":[], "usage":raw["usage"]}).to_string()).is_err());
    }
}
