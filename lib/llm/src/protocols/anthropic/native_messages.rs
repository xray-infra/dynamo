// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lossless text/tool Messages boundary for native HTTP workers.
use crate::cache_authority::cache_accounting::VerifiedCacheAccounting;
use anyhow::{Result, bail, ensure};
use serde_json::{Value, json};

/// In-process Engine output only. The Engine strips any worker-supplied value
/// before adding its verified report; every public serializer removes this key.
pub(crate) const VERIFIED_CACHE_ACCOUNTING: &str = "_dynamo_verified_cache_accounting";

pub use crate::native_cache_messages::native_messages_to_chat;

fn text<'a>(value: Option<&'a Value>, path: &str) -> Result<&'a str> {
    value
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("{path}: expected string"))
}

/// Native raw usage preserves both flat engine aliases and OpenAI nested aliases.
pub(crate) fn usage(raw: Option<&Value>) -> Result<Value> {
    let raw = raw.unwrap_or(&Value::Null);
    let total = raw
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            anyhow::anyhow!("usage_unavailable: native input count missing or invalid")
        })?;
    let output = raw
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            anyhow::anyhow!("usage_unavailable: native output count missing or invalid")
        })?;
    if let Some(report) = raw.get(VERIFIED_CACHE_ACCOUNTING) {
        if report.is_null() {
            // Unknown TTL accounting is not a cache miss or a known zero write.
            return Ok(json!({"input_tokens":total,"output_tokens":output}));
        }
        let report: VerifiedCacheAccounting = serde_json::from_value(report.clone())?;
        report.validate(total)?;
        let creation = report
            .cache_creation_5m_input_tokens
            .checked_add(report.cache_creation_1h_input_tokens)
            .ok_or_else(|| anyhow::anyhow!("verified cache creation count overflow"))?;
        return Ok(
            json!({"input_tokens":report.input_tokens,"output_tokens":output,
            "cache_read_input_tokens":report.cache_read_input_tokens,
            "cache_creation_input_tokens":creation,
            "cache_creation":{"ephemeral_5m_input_tokens":report.cache_creation_5m_input_tokens,
                "ephemeral_1h_input_tokens":report.cache_creation_1h_input_tokens}}),
        );
    }
    let read = raw
        .get("cache_read_input_tokens")
        .or_else(|| raw.get("cached_tokens"))
        .or_else(|| raw.pointer("/prompt_tokens_details/cached_tokens"))
        .and_then(Value::as_u64);
    let write = raw
        .get("cache_creation_input_tokens")
        .or_else(|| raw.get("cache_creation_tokens"))
        .and_then(Value::as_u64);
    let mut out = json!({"input_tokens":total.saturating_sub(read.unwrap_or(0)).saturating_sub(write.unwrap_or(0)),"output_tokens":output});
    if let Some(r) = read {
        out["cache_read_input_tokens"] = json!(r);
    }
    if let Some(w) = write {
        out["cache_creation_input_tokens"] = json!(w);
    }
    Ok(out)
}

/// Convert complete native Chat JSON. Missing finish and invalid successful tool arguments fail.
pub fn native_chat_response_to_messages(raw: &Value, model: &str) -> Result<Value> {
    ensure!(raw.get("error").is_none(), "native generation error");
    let c = raw
        .pointer("/choices/0")
        .ok_or_else(|| anyhow::anyhow!("native response has no choice"))?;
    let finish = text(c.get("finish_reason"), "finish_reason")?;
    let stop = match finish {
        "stop" => "end_turn",
        "tool_calls" | "function_call" => "tool_use",
        "length" => "max_tokens",
        _ => bail!("unsupported native finish_reason {finish}"),
    };
    let m = c
        .get("message")
        .ok_or_else(|| anyhow::anyhow!("native response has no message"))?;
    let mut content = Vec::new();
    if let Some(t) = m
        .get("reasoning_content")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        content.push(json!({"type":"thinking","thinking":t,"signature":"erased"}));
    }
    if let Some(t) = m.get("content").filter(|v| !v.is_null()) {
        ensure!(t.is_string(), "native nontext response unsupported");
        content.push(json!({"type":"text","text":t}));
    }
    if let Some(t) = m.get("refusal").filter(|v| !v.is_null()) {
        ensure!(t.is_string(), "native refusal must be text");
        content.push(json!({"type":"text","text":t}));
    }
    if matches!(finish, "tool_calls" | "function_call") {
        ensure!(
            m.get("tool_calls")
                .and_then(Value::as_array)
                .is_some_and(|ts| !ts.is_empty()),
            "native tool finish has no tool calls"
        );
    }
    if let Some(ts) = m.get("tool_calls").filter(|value| !value.is_null()) {
        for t in ts
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("tool_calls: expected array"))?
        {
            let id = text(t.get("id"), "tool.id")?;
            let name = text(t.pointer("/function/name"), "tool.name")?;
            ensure!(
                !id.is_empty() && !name.is_empty(),
                "invalid native tool identity"
            );
            let args = text(t.pointer("/function/arguments"), "tool.arguments")?;
            let parsed = serde_json::from_str::<Value>(args);
            if finish == "length" && !parsed.as_ref().is_ok_and(Value::is_object) {
                continue;
            }
            let input = parsed?;
            ensure!(input.is_object(), "tool arguments must be an object");
            content.push(json!({"type":"tool_use","id":id,"name":name,"input":input}));
        }
    }
    Ok(
        json!({"id":format!("msg_{}",uuid::Uuid::new_v4().simple()),"type":"message","role":"assistant","model":model,"content":content,"stop_reason":stop,"stop_sequence":null,"usage":usage(raw.get("usage"))?}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_messages_optional_tool_calls_null_preserves_tool_guards() {
        let response = json!({"choices":[{"finish_reason":"stop",
            "message":{"content":"native text","tool_calls":null}}],
            "usage":{"prompt_tokens":4935,"completion_tokens":4,"reasoning_tokens":0}});
        let null = native_chat_response_to_messages(&response, "test-model").unwrap();
        let mut absent = response.clone();
        absent["choices"][0]["message"]
            .as_object_mut()
            .unwrap()
            .remove("tool_calls");
        let absent = native_chat_response_to_messages(&absent, "test-model").unwrap();
        assert_eq!(
            null["content"],
            json!([{"type":"text","text":"native text"}])
        );
        assert_eq!(null["content"], absent["content"]);
        assert_eq!(null["stop_reason"], "end_turn");
        assert_eq!(
            null["usage"],
            json!({"input_tokens":4935,"output_tokens":4})
        );

        let mut valid = response.clone();
        valid["choices"][0]["finish_reason"] = json!("tool_calls");
        valid["choices"][0]["message"]["tool_calls"] = json!([
            {"id":"call_native_1","type":"function","function":{"name":"lookup","arguments":"{\"key\":\"value\"}"}}
        ]);
        let out = native_chat_response_to_messages(&valid, "test-model").unwrap();
        assert_eq!(
            out["content"][1],
            json!({"type":"tool_use","id":"call_native_1","name":"lookup","input":{"key":"value"}})
        );
        assert_eq!(out["stop_reason"], "tool_use");
        for bad in [json!({}), json!("invalid"), json!(false), json!(7)] {
            let mut raw = response.clone();
            raw["choices"][0]["message"]["tool_calls"] = bad;
            assert!(native_chat_response_to_messages(&raw, "test-model").is_err());
        }
        for bad in [
            Value::Null,
            json!([]),
            json!([{}]),
            json!([{"id":"","function":{"name":"lookup","arguments":"{}"}}]),
            json!([{"id":"call","function":{"name":"","arguments":"{}"}}]),
            json!([{"id":"call","function":{"name":"lookup","arguments":"[]"}}]),
            json!([{"id":"call","function":{"name":"lookup","arguments":"{"}}]),
        ] {
            let mut raw = valid.clone();
            raw["choices"][0]["message"]["tool_calls"] = bad;
            assert!(native_chat_response_to_messages(&raw, "test-model").is_err());
        }
    }
    fn verified_usage(report: Value) -> Value {
        json!({"prompt_tokens":100,"completion_tokens":7,
            "cached_tokens":99,"cache_creation_tokens":99,
            "_dynamo_verified_cache_accounting":report})
    }
    #[test]
    fn native_verified_cache_billing_disjoint_actual_counts() {
        let verified = VerifiedCacheAccounting {
            input_tokens: 10,
            cache_read_input_tokens: 40,
            cache_creation_5m_input_tokens: 20,
            cache_creation_1h_input_tokens: 30,
        };
        let raw = verified_usage(serde_json::to_value(verified).unwrap());
        let out = usage(Some(&raw)).unwrap();
        assert_eq!(
            out,
            json!({"input_tokens":10,"output_tokens":7,"cache_read_input_tokens":40,
            "cache_creation_input_tokens":50,"cache_creation":{
                "ephemeral_5m_input_tokens":20,"ephemeral_1h_input_tokens":30}})
        );
        assert!(out.get(VERIFIED_CACHE_ACCOUNTING).is_none());
    }
    #[test]
    fn native_verified_cache_billing_unknown_omits_alias_claims() {
        let out = usage(Some(&verified_usage(Value::Null))).unwrap();
        assert_eq!(out, json!({"input_tokens":100,"output_tokens":7}));
        assert!(out.get("cache_read_input_tokens").is_none());
        assert!(out.get("cache_creation_input_tokens").is_none());
        assert!(out.get("cache_creation").is_none());
    }
    #[test]
    fn native_verified_cache_billing_rejects_overlap_overflow_and_invalid() {
        for report in [
            json!({"input_tokens":10,"cache_read_input_tokens":90,
                "cache_creation_5m_input_tokens":20,"cache_creation_1h_input_tokens":0}),
            json!({"input_tokens":0,"cache_read_input_tokens":0,
                "cache_creation_5m_input_tokens":u64::MAX,"cache_creation_1h_input_tokens":1}),
            json!({"input_tokens":0,"cache_read_input_tokens":-1,
                "cache_creation_5m_input_tokens":0,"cache_creation_1h_input_tokens":0}),
            json!({"input_tokens":100}),
        ] {
            assert!(usage(Some(&verified_usage(report))).is_err());
        }
    }
    fn missing_counter_errors(marker: Option<Value>) {
        for field in ["prompt_tokens", "completion_tokens"] {
            let mut raw = json!({"prompt_tokens":100,"completion_tokens":7});
            if let Some(marker) = &marker {
                raw[VERIFIED_CACHE_ACCOUNTING] = marker.clone();
            }
            raw.as_object_mut().unwrap().remove(field);
            let response = json!({"choices":[{"finish_reason":"stop",
                "message":{"content":"valid content"}}],"usage":raw});
            let error = native_chat_response_to_messages(&response, "test-model").unwrap_err();
            assert!(error.to_string().starts_with("usage_unavailable:"));
        }
    }
    #[test]
    fn native_verified_cache_billing_missing_counter_plain() {
        missing_counter_errors(None);
    }
    #[test]
    fn native_verified_cache_billing_missing_counter_some() {
        missing_counter_errors(Some(json!({"input_tokens":100,"cache_read_input_tokens":0,
            "cache_creation_5m_input_tokens":0,"cache_creation_1h_input_tokens":0})));
    }
    #[test]
    fn native_verified_cache_billing_missing_counter_unknown() {
        missing_counter_errors(Some(Value::Null));
    }
    #[test]
    fn native_streaming_messages_requests_native_usage() {
        let mut raw = request();
        raw["stream"] = json!(true);
        let chat = native_messages_to_chat(&raw).unwrap();
        assert_eq!(chat["stream_options"], json!({"include_usage":true}));
        raw["stream"] = json!(false);
        assert!(
            native_messages_to_chat(&raw)
                .unwrap()
                .get("stream_options")
                .is_none()
        );
        raw.as_object_mut().unwrap().remove("stream");
        assert!(
            native_messages_to_chat(&raw)
                .unwrap()
                .get("stream_options")
                .is_none()
        );
    }
    fn request() -> Value {
        json!({"model":"glm-5.3-flash","max_tokens":800,"messages":[{"role":"user","content":"hi"}]})
    }
    #[test]
    fn native_history_preserves_leading_thinking_and_rejects_interleaving() {
        let mut r = request();
        r["messages"] = json!([{"role":"assistant","content":[{"type":"thinking","thinking":"first"},{"type":"thinking","thinking":"second"},{"type":"tool_use","id":"id","name":"Read","input":{}}]}]);
        let c = native_messages_to_chat(&r).unwrap();
        assert_eq!(c["messages"][0]["reasoning_content"], "firstsecond");
        assert_eq!(c["messages"][0]["tool_calls"][0]["id"], "id");
        r["messages"][0]["content"]
            .as_array_mut()
            .unwrap()
            .push(json!({"type":"thinking","thinking":"after"}));
        assert!(
            native_messages_to_chat(&r)
                .unwrap_err()
                .to_string()
                .contains("interleaved thinking")
        );
    }
    #[test]
    fn native_request_preserves_budget_strict_namespace_and_identity() {
        let mut r = request();
        r["thinking"] = json!({"type":"enabled","budget_tokens":32});
        r["output_config"] = json!({"effort":"high"});
        r["cache_salt"] = json!("s1");
        r["tools"] = json!([{"name":"Read","strict":false,"input_schema":{"type":"object"}}]);
        let c = native_messages_to_chat(&r).unwrap();
        assert_eq!(c["thinking_token_budget"], 32);
        assert_eq!(c["max_tokens"], 800);
        assert_eq!(c["reasoning_effort"], "high");
        assert_eq!(c["tools"][0]["function"]["strict"], false);
        assert_eq!(c["cache_salt"], "s1");
        assert!(c["tools"][0].get("cache_control").is_none());
    }
    #[test]
    fn native_disabled_is_none_and_invalid_contracts_fail() {
        let mut r = request();
        r["thinking"] = json!({"type":"disabled"});
        assert_eq!(
            native_messages_to_chat(&r).unwrap()["reasoning_effort"],
            "none"
        );
        for (k, v) in [
            ("top_k", json!(5)),
            ("unknown", json!(false)),
            ("cache_control", json!({"type":"ephemeral","ttl":"2m"})),
            ("thinking", json!({"type":"enabled","budget_tokens":true})),
            ("tools", json!([{"type":"web_search","name":"web"}])),
        ] {
            let mut r = request();
            r[k] = v;
            assert!(native_messages_to_chat(&r).is_err(), "{k}");
        }
        for block in [
            json!({"type":"image","source":{}}),
            json!({"type":"tool_result","tool_use_id":"t1","is_error":"false"}),
            json!({"type":"server_tool_use","id":"s","name":"web","input":{}}),
        ] {
            let mut r = request();
            r["messages"][0]["content"] = json!([block]);
            assert!(native_messages_to_chat(&r).is_err());
        }
    }
    #[test]
    fn native_tool_result_error_status_preserves_content_and_history_identity() {
        for content in [
            json!("local failure"),
            json!([{"type":"text","text":"local failure"}]),
        ] {
            for is_error in [false, true] {
                let raw = json!({"model":"qwen3.8-flash-next","max_tokens":64,
                    "tools":[{"name":"lookup","input_schema":{"type":"object"}}],
                    "messages":[{"role":"assistant","content":[{"type":"tool_use","id":"call_original","name":"lookup","input":{}}]},
                        {"role":"user","content":[{"type":"tool_result","tool_use_id":"call_original","content":content,"is_error":is_error}]}]});
                let chat = native_messages_to_chat(&raw).unwrap();
                assert_eq!(chat["messages"][0]["tool_calls"][0]["id"], "call_original");
                assert_eq!(chat["messages"][1]["tool_call_id"], "call_original");
                assert_eq!(chat["messages"][1]["role"], "tool");
                if is_error {
                    assert!(
                        chat["messages"][1]["content"]
                            .to_string()
                            .contains("Tool execution error:")
                    );
                    assert!(
                        chat["messages"][1]["content"]
                            .to_string()
                            .contains("local failure")
                    );
                } else {
                    assert_eq!(chat["messages"][1]["content"], content);
                }
                crate::native_contract::ModelContract::for_model("qwen3.8-flash-next")
                    .unwrap()
                    .normalize(chat)
                    .unwrap();
                let mut invalid = raw.clone();
                invalid["messages"][1]["content"][0]["is_error"] = json!(1);
                assert!(native_messages_to_chat(&invalid).is_err());
            }
        }
    }
    #[test]
    fn raw_response_identity_usage_and_terminal_are_honest() {
        let mut r = json!({"choices":[{"finish_reason":"tool_calls","message":{"tool_calls":[{"id":"call_orig","function":{"name":"Read","arguments":"{}"}}]}}],"usage":{"prompt_tokens":20,"completion_tokens":4,"cached_tokens":16}});
        let m = native_chat_response_to_messages(&r, "m").unwrap();
        assert_eq!(m["content"][0]["id"], "call_orig");
        assert_eq!(m["usage"]["input_tokens"], 4);
        assert_eq!(m["usage"]["cache_read_input_tokens"], 16);
        assert!(m["usage"].get("cache_creation_input_tokens").is_none());
        r["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json!("{");
        assert!(native_chat_response_to_messages(&r, "m").is_err());
        r["choices"][0]["finish_reason"] = json!("length");
        let m = native_chat_response_to_messages(&r, "m").unwrap();
        assert_eq!(m["stop_reason"], "max_tokens");
        assert_eq!(m["content"], json!([]));
        r["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json!("{}");
        let m = native_chat_response_to_messages(&r, "m").unwrap();
        assert_eq!(m["stop_reason"], "max_tokens");
        assert_eq!(m["content"][0]["id"], "call_orig");
        r["choices"][0]["finish_reason"] = Value::Null;
        assert!(native_chat_response_to_messages(&r, "m").is_err());
    }
}
