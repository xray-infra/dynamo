// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Producer -> real completion DTO -> protocol converter regression gates.
//! CPU fixtures only; not an HTTP handler, native receipt or live-worker test.
use super::*;
use crate::cache_authority::cache_accounting::VerifiedCacheAccounting;
use crate::protocols::anthropic::stream_converter::AnthropicStreamConverter;
use crate::protocols::openai::responses::{
    ResponseParams, stream_converter::ResponseStreamConverter,
};
use axum::response::{IntoResponse, sse::Event};
use http_body_util::BodyExt;

#[path = "protocols/anthropic/native_messages.rs"]
mod native_messages;

fn native_chunk(tool: bool) -> Value {
    let delta = if tool {
        json!({"role":"assistant","tool_calls":[{"index":0,"id":"call_native_cache",
            "type":"function","function":{"name":"lookup","arguments":"{\"service\":\"demo\"}"}}]})
    } else {
        json!({"role":"assistant","content":"OK"})
    };
    json!({"id":"chatcmpl-native-cache","object":"chat.completion.chunk",
        "created":1700000000,"model":"glm-5.3-flash","system_fingerprint":"native-fixture",
        "service_tier":"default",
        "choices":[{"index":0,"delta":delta,"finish_reason":if tool {"tool_calls"} else {"stop"},"logprobs":null}],
        "usage":{"prompt_tokens":100,"completion_tokens":7,"total_tokens":107,
            "completion_tokens_details":{"reasoning_tokens":2},"cached_tokens":99},
        "nvext":{"must_not_replay":true},"llm_metrics":{"must_not_replay":true},
        "tool_call_completion":[{"choice_index":0,"tool_index":0,"complete":true}],
        "opaque_native":{"must_not_replay":true}})
}

fn report(hot: bool, hour: bool) -> VerifiedCacheAccounting {
    VerifiedCacheAccounting {
        input_tokens: 40,
        cache_read_input_tokens: if hot { 60 } else { 0 },
        cache_creation_5m_input_tokens: if hot || hour { 0 } else { 60 },
        cache_creation_1h_input_tokens: if hot || !hour { 0 } else { 60 },
    }
}

#[test]
fn native_cache_terminal_usage_preserves_envelope() {
    for tool in [false, true] {
        for hot in [false, true] {
            for hour in [false, true] {
                let native = native_chunk(tool);
                let report = report(hot, hour);
                let terminal =
                    verified_ttl_terminal_usage(100, Some(&native), Some(&report)).unwrap();
                serde_json::from_value::<NvCreateChatCompletionStreamResponse>(terminal.clone())
                    .expect("terminal usage must be a complete chunk");
                assert_eq!(terminal["choices"], json!([]));
                for field in [
                    "id",
                    "object",
                    "created",
                    "model",
                    "system_fingerprint",
                    "service_tier",
                ] {
                    assert_eq!(terminal[field], native[field], "{field}");
                }
                for field in [
                    "nvext",
                    "llm_metrics",
                    "tool_call_completion",
                    "opaque_native",
                ] {
                    assert!(terminal.get(field).is_none(), "replayed {field}");
                }
                assert_eq!(terminal["usage"]["completion_tokens"], 7);
                assert_eq!(
                    terminal["usage"]["completion_tokens_details"]["reasoning_tokens"],
                    2
                );
                assert!(terminal["usage"].get("cached_tokens").is_none());
                let authoritative = native_messages::usage(terminal.get("usage")).unwrap();
                assert_eq!(authoritative["input_tokens"], 40);
                assert_eq!(authoritative["output_tokens"], 7);
                assert_eq!(
                    authoritative["cache_read_input_tokens"],
                    if hot { 60 } else { 0 }
                );
                assert_eq!(
                    authoritative["cache_creation_input_tokens"],
                    if hot { 0 } else { 60 }
                );
                assert_eq!(
                    authoritative["cache_creation"]["ephemeral_1h_input_tokens"],
                    report.cache_creation_1h_input_tokens
                );
                assert_eq!(
                    authoritative["cache_creation"]["ephemeral_5m_input_tokens"],
                    report.cache_creation_5m_input_tokens
                );
            }
        }
    }
}

async fn event_values(events: Vec<Result<Event, Error>>) -> Vec<Value> {
    let body = axum::response::sse::Sse::new(futures::stream::iter(
        events
            .into_iter()
            .map(|event| event.map_err(axum::Error::new)),
    ))
    .into_response()
    .into_body()
    .collect()
    .await
    .unwrap()
    .to_bytes();
    std::str::from_utf8(&body)
        .unwrap()
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|data| serde_json::from_str(data).unwrap())
        .collect()
}

#[tokio::test]
async fn native_cache_terminal_usage_preserves_protocol_endings() {
    for tool in [false, true] {
        let native = native_chunk(tool);
        let terminal =
            verified_ttl_terminal_usage(100, Some(&native), Some(&report(false, false))).unwrap();
        let typed: NvCreateChatCompletionStreamResponse = serde_json::from_value(terminal).unwrap();
        // Do not carry fixture-only internal metadata into the converter.
        let mut content = native;
        for field in [
            "nvext",
            "llm_metrics",
            "tool_call_completion",
            "opaque_native",
            "usage",
        ] {
            content.as_object_mut().unwrap().remove(field);
        }
        let content: NvCreateChatCompletionStreamResponse =
            serde_json::from_value(content).unwrap();
        let mut messages = AnthropicStreamConverter::new("glm-5.3-flash".into(), 999);
        let mut events = messages.emit_start_events();
        events.extend(messages.process_chunk(&content));
        events.extend(messages.process_chunk(&typed));
        events.extend(messages.emit_end_events());
        let values = event_values(events).await;
        assert!(values.iter().all(|value| value["type"] != "error"));
        assert_eq!(
            values
                .iter()
                .filter(|value| value["type"] == "message_delta")
                .count(),
            1
        );
        assert_eq!(values.last().unwrap()["type"], "message_stop");
        let end = values
            .iter()
            .find(|value| value["type"] == "message_delta")
            .unwrap();
        assert_eq!(
            end["delta"]["stop_reason"],
            if tool { "tool_use" } else { "end_turn" }
        );
        // This baseline converter consumes the typed counters; native TTL
        // projection above is a separate seam, not handler integration.
        assert_eq!(end["usage"]["input_tokens"], 100);
        assert_eq!(end["usage"]["output_tokens"], 7);
        if tool {
            assert_eq!(
                values
                    .iter()
                    .filter(|value| value["type"] == "content_block_start"
                        && value["content_block"]["type"] == "tool_use")
                    .count(),
                1
            );
        }
        let mut responses =
            ResponseStreamConverter::new("glm-5.3-flash".into(), ResponseParams::default());
        let mut events = responses.emit_start_events();
        events.extend(responses.process_chunk(&content));
        events.extend(responses.process_chunk(&typed));
        events.extend(responses.emit_end_events());
        let values = event_values(events).await;
        assert!(
            values
                .iter()
                .all(|value| value["type"] != "response.failed")
        );
        assert_eq!(
            values
                .iter()
                .filter(|value| value["type"] == "response.completed")
                .count(),
            1
        );
        let end = values
            .iter()
            .find(|value| value["type"] == "response.completed")
            .unwrap();
        assert_eq!(end["response"]["usage"]["input_tokens"], 100);
        assert_eq!(end["response"]["usage"]["output_tokens"], 7);
    }
}

#[test]
fn native_cache_terminal_usage_does_not_relax_missing_id_validation() {
    let mut valid = native_terminal_usage_template(&native_chunk(false));
    assert!(serde_json::from_value::<NvCreateChatCompletionStreamResponse>(valid.clone()).is_ok());
    valid.as_object_mut().unwrap().remove("id");
    let error = serde_json::from_value::<NvCreateChatCompletionStreamResponse>(valid).unwrap_err();
    assert!(error.to_string().contains("missing field `id`"));
}

#[test]
fn native_cache_terminal_usage_rejects_missing_or_malformed_envelope() {
    for field in ["id", "object", "created", "model"] {
        let mut missing = native_chunk(false);
        missing.as_object_mut().unwrap().remove(field);
        assert!(
            verified_ttl_terminal_usage(100, Some(&missing), Some(&report(false, false))).is_err(),
            "missing {field}"
        );
        for value in [Value::Null, json!(false), json!([]), json!({})] {
            let mut malformed = native_chunk(false);
            malformed[field] = value;
            assert!(
                verified_ttl_terminal_usage(100, Some(&malformed), Some(&report(false, false)))
                    .is_err(),
                "malformed {field}"
            );
        }
    }
    let mut overflow = native_chunk(false);
    overflow["created"] = json!(u64::MAX);
    assert!(verified_ttl_terminal_usage(100, Some(&overflow), None).is_err());
}

#[test]
fn native_cache_terminal_usage_supports_missing_optional_identity_fields() {
    let mut native = native_chunk(false);
    for field in ["system_fingerprint", "service_tier"] {
        native.as_object_mut().unwrap().remove(field);
    }
    let terminal =
        verified_ttl_terminal_usage(100, Some(&native), Some(&report(false, false))).unwrap();
    assert!(terminal.get("system_fingerprint").is_none());
    assert!(terminal.get("service_tier").is_none());
    assert!(serde_json::from_value::<NvCreateChatCompletionStreamResponse>(terminal).is_ok());
}

#[test]
fn native_cache_terminal_usage_unknown_is_not_zero_billing() {
    let native = native_chunk(false);
    let unknown = verified_ttl_terminal_usage(100, Some(&native), None).unwrap();
    assert!(unknown["usage"][VERIFIED_CACHE_ACCOUNTING_FIELD].is_null());
    let usage = native_messages::usage(unknown.get("usage")).unwrap();
    assert_eq!(usage, json!({"input_tokens":100,"output_tokens":7}));
    assert!(serde_json::from_value::<NvCreateChatCompletionStreamResponse>(unknown).is_ok());
}

#[test]
fn native_cache_terminal_usage_records_identity_with_usage_and_null_invalidates() {
    let native = native_chunk(true);
    let mut recorded = None;
    record_native_usage_chunk(&mut recorded, &native);
    assert_eq!(recorded.as_ref().unwrap()["id"], native["id"]);
    assert_eq!(recorded.as_ref().unwrap()["choices"], json!([]));
    assert!(
        recorded
            .as_ref()
            .unwrap()
            .get("tool_call_completion")
            .is_none()
    );
    let before = recorded.clone();
    let mut trailing = native.clone();
    trailing.as_object_mut().unwrap().remove("usage");
    trailing["id"] = json!("do-not-combine-with-earlier-usage");
    record_native_usage_chunk(&mut recorded, &trailing);
    assert_eq!(recorded, before);
    let terminal =
        verified_ttl_terminal_usage(100, recorded.as_ref(), Some(&report(true, false))).unwrap();
    assert_eq!(terminal["id"], native["id"]);
    trailing["usage"] = Value::Null;
    record_native_usage_chunk(&mut recorded, &trailing);
    assert!(
        verified_ttl_terminal_usage(100, recorded.as_ref(), Some(&report(true, false))).is_err()
    );
}

#[test]
fn native_cache_terminal_usage_rejects_inconsistent_verified_accounting() {
    let native = native_chunk(false);
    let mut invalid = report(false, false);
    invalid.input_tokens += 1;
    assert!(verified_ttl_terminal_usage(100, Some(&native), Some(&invalid)).is_err());
    assert!(verified_ttl_terminal_usage(99, Some(&native), Some(&report(false, false))).is_err());
    assert!(verified_ttl_terminal_usage(100, None, Some(&report(false, false))).is_err());
}
