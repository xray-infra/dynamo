// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lossless text/tool Messages boundary for native HTTP workers.
use anyhow::{Result, bail, ensure};
use serde_json::{Map, Value, json};

fn object<'a>(v: &'a Value, path: &str, keys: &[&str]) -> Result<&'a Map<String, Value>> {
    let o = v
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("{path}: expected object"))?;
    for k in o.keys() {
        ensure!(keys.contains(&k.as_str()), "{path}.{k}: unsupported field");
    }
    Ok(o)
}
fn text<'a>(v: Option<&'a Value>, path: &str) -> Result<&'a str> {
    v.and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("{path}: expected string"))
}
fn cache(v: Option<&Value>, path: &str) -> Result<()> {
    if let Some(v) = v {
        let o = object(v, path, &["type", "ttl"])?;
        ensure!(
            text(o.get("type"), path)? == "ephemeral",
            "{path}: only automatic ephemeral prefix hints are supported"
        );
        if let Some(ttl) = o.get("ttl") {
            ensure!(
                ["5m", "1h"].contains(&text(Some(ttl), path)?),
                "{path}: unsupported TTL"
            );
        }
    }
    Ok(())
}
fn text_blocks(v: &Value, path: &str) -> Result<Value> {
    if v.is_string() {
        return Ok(v.clone());
    }
    let a = v
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("{path}: expected text or blocks"))?;
    let mut out = Vec::new();
    for (i, b) in a.iter().enumerate() {
        let p = format!("{path}[{i}]");
        let o = object(b, &p, &["type", "text", "cache_control"])?;
        ensure!(
            text(o.get("type"), &p)? == "text",
            "{p}: only text is supported"
        );
        text(o.get("text"), &p)?;
        cache(o.get("cache_control"), &p)?;
        out.push(b.clone());
    }
    Ok(Value::Array(out))
}

/// Convert validated raw Messages JSON into the one canonical native Chat body.
/// Model-specific reasoning normalization and schema meta-validation follow this boundary.
pub fn native_messages_to_chat(raw: &Value) -> Result<Value> {
    let r = object(
        raw,
        "request",
        &[
            "model",
            "max_tokens",
            "messages",
            "system",
            "temperature",
            "top_p",
            "stop_sequences",
            "stream",
            "metadata",
            "tools",
            "tool_choice",
            "thinking",
            "output_config",
            "cache_control",
            "prompt_cache_key",
            "cache_salt",
        ],
    )?;
    let model = text(r.get("model"), "model")?;
    ensure!(!model.is_empty(), "model: must not be empty");
    let limit = r
        .get("max_tokens")
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow::anyhow!("max_tokens: expected positive integer"))?;
    ensure!(
        limit > 0 && limit <= u32::MAX as u64,
        "max_tokens: out of range"
    );
    cache(r.get("cache_control"), "cache_control")?;
    let mut out = json!({"model":model,"max_tokens":limit,"messages":[],"stream":false});
    for k in ["temperature", "top_p"] {
        if let Some(v) = r.get(k) {
            ensure!(v.is_number(), "{k}: expected number");
            out[k] = v.clone();
        }
    }
    if let Some(v) = r.get("stream") {
        ensure!(v.is_boolean(), "stream: expected boolean");
        out["stream"] = v.clone();
    }
    // Messages usage must come from the worker, including its final stream
    // chunk; the public Messages request has no stream_options control.
    if out["stream"] == true {
        out["stream_options"] = json!({"include_usage":true});
    }
    if let Some(v) = r.get("stop_sequences") {
        ensure!(
            v.as_array().is_some_and(|a| a.iter().all(Value::is_string)),
            "stop_sequences: expected string array"
        );
        out["stop"] = v.clone();
    }
    if let Some(v) = r.get("metadata") {
        object(v, "metadata", &["user_id"])?;
        if let Some(id) = v.get("user_id") {
            text(Some(id), "metadata.user_id")?;
            out["user"] = id.clone();
        }
    }
    for k in ["cache_control", "prompt_cache_key", "cache_salt"] {
        if let Some(v) = r.get(k) {
            if k != "cache_control" {
                let s = text(Some(v), k)?;
                ensure!(
                    !s.is_empty() && s.len() <= 1024,
                    "{k}: expected 1..1024 UTF-8 bytes"
                );
            }
            out[k] = v.clone();
        }
    }
    if let (Some(a), Some(b)) = (r.get("cache_salt"), r.get("prompt_cache_key")) {
        ensure!(a == b, "cache_salt and prompt_cache_key conflict");
    }
    if let Some(t) = r.get("thinking") {
        let t = object(t, "thinking", &["type", "budget_tokens"])?;
        match text(t.get("type"), "thinking.type")? {
            "disabled" => {
                ensure!(
                    !t.contains_key("budget_tokens"),
                    "disabled thinking cannot carry a budget"
                );
                out["reasoning_effort"] = json!("none");
            }
            "enabled" => {
                let b = t
                    .get("budget_tokens")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| {
                        anyhow::anyhow!("thinking.budget_tokens: expected positive integer")
                    })?;
                ensure!(
                    b > 0 && b < limit && b <= u32::MAX as u64,
                    "thinking.budget_tokens: must be positive and smaller than max_tokens"
                );
                out["thinking_token_budget"] = json!(b);
                out["chat_template_kwargs"] = json!({"enable_thinking":true});
            }
            "adaptive" => {
                ensure!(
                    !t.contains_key("budget_tokens"),
                    "adaptive thinking cannot carry a budget"
                );
                out["chat_template_kwargs"] = json!({"enable_thinking":true});
            }
            _ => bail!("thinking.type: unsupported value"),
        }
    }
    if let Some(c) = r.get("output_config") {
        let c = object(c, "output_config", &["effort"])?;
        if let Some(e) = c.get("effort") {
            let e = text(Some(e), "output_config.effort")?;
            ensure!(
                ["minimal", "low", "medium", "high", "xhigh", "max", "none"].contains(&e),
                "output_config.effort: unsupported value"
            );
            if let Some(old) = out.get("reasoning_effort") {
                ensure!(old == e, "thinking and output_config.effort conflict");
            } else {
                out["reasoning_effort"] = json!(e);
            }
        }
    }
    let mut messages = Vec::new();
    if let Some(s) = r.get("system") {
        messages.push(json!({"role":"system","content":text_blocks(s,"system")?}));
    }
    let ms = r
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("messages: expected array"))?;
    ensure!(!ms.is_empty(), "messages: must not be empty");
    for (i, m) in ms.iter().enumerate() {
        let p = format!("messages[{i}]");
        let m = object(m, &p, &["role", "content"])?;
        let role = text(m.get("role"), &p)?;
        ensure!(
            ["user", "assistant", "system"].contains(&role),
            "{p}.role: unsupported role"
        );
        let c = m
            .get("content")
            .ok_or_else(|| anyhow::anyhow!("{p}.content: required"))?;
        if c.is_string() {
            messages.push(json!({"role":role,"content":c}));
            continue;
        }
        let bs = c
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("{p}.content: expected text or blocks"))?;
        let mut parts = Vec::new();
        let mut calls = Vec::new();
        let mut reasoning = String::new();
        for (j, b) in bs.iter().enumerate() {
            let bp = format!("{p}.content[{j}]");
            let kind = text(b.get("type"), &bp)?;
            match kind {
                "text" => {
                    text_blocks(&json!([b]), &bp)?;
                    parts.push(b.clone());
                }
                "tool_use" => {
                    ensure!(
                        role == "assistant",
                        "{bp}: tool_use requires assistant role"
                    );
                    let b = object(b, &bp, &["type", "id", "name", "input", "cache_control"])?;
                    cache(b.get("cache_control"), &bp)?;
                    let id = text(b.get("id"), &bp)?;
                    let name = text(b.get("name"), &bp)?;
                    ensure!(
                        !id.is_empty() && !name.is_empty(),
                        "{bp}: empty tool identity"
                    );
                    let input = b
                        .get("input")
                        .ok_or_else(|| anyhow::anyhow!("{bp}.input: required"))?;
                    ensure!(input.is_object(), "{bp}.input: expected object");
                    let mut call = json!({"id":id,"type":"function","function":{"name":name,"arguments":serde_json::to_string(input)?}});
                    if let Some(marker) = b.get("cache_control") {
                        call["cache_control"] = marker.clone();
                    }
                    calls.push(call);
                }
                "tool_result" => {
                    ensure!(role == "user", "{bp}: tool_result requires user role");
                    let b = object(
                        b,
                        &bp,
                        &[
                            "type",
                            "tool_use_id",
                            "content",
                            "is_error",
                            "cache_control",
                        ],
                    )?;
                    cache(b.get("cache_control"), &bp)?;
                    let is_error = b
                        .get("is_error")
                        .map(|value| {
                            value
                                .as_bool()
                                .ok_or_else(|| anyhow::anyhow!("{bp}.is_error: expected boolean"))
                        })
                        .transpose()?
                        .unwrap_or(false);
                    if !parts.is_empty() {
                        messages.push(json!({"role":role,"content":std::mem::take(&mut parts)}));
                    }
                    let id = text(b.get("tool_use_id"), &bp)?;
                    ensure!(!id.is_empty(), "{bp}: empty tool identity");
                    let mut content = text_blocks(b.get("content").unwrap_or(&json!("")), &bp)?;
                    if is_error {
                        content = match content {
                            Value::String(text) => json!(format!("Tool execution error:\n{text}")),
                            Value::Array(mut blocks) => {
                                blocks.insert(
                                    0,
                                    json!({"type":"text","text":"Tool execution error:"}),
                                );
                                Value::Array(blocks)
                            }
                            _ => unreachable!("text_blocks validated tool result content"),
                        };
                    }
                    let mut msg = json!({"role":"tool","tool_call_id":id,"content":content});
                    if let Some(marker) = b.get("cache_control") {
                        msg["cache_control"] = marker.clone();
                    }
                    messages.push(msg);
                }
                "thinking" => {
                    ensure!(
                        parts.is_empty(),
                        "{bp}: thinking after text is unsupported by the native HTTP worker"
                    );
                    ensure!(
                        calls.is_empty(),
                        "{bp}: interleaved thinking after tool_use is unsupported by the native HTTP worker"
                    );
                    ensure!(
                        role == "assistant",
                        "{bp}: thinking requires assistant role"
                    );
                    let b = object(b, &bp, &["type", "thinking", "signature"])?;
                    if let Some(s) = b.get("signature") {
                        text(Some(s), &bp)?;
                    }
                    reasoning.push_str(text(b.get("thinking"), &bp)?);
                }
                _ => bail!("{bp}: unsupported content block {kind}"),
            }
        }
        if !parts.is_empty() || !calls.is_empty() || !reasoning.is_empty() {
            let mut msg = json!({"role":role,"content":parts});
            if !calls.is_empty() {
                msg["tool_calls"] = json!(calls);
            }
            if !reasoning.is_empty() {
                msg["reasoning_content"] = json!(reasoning);
            }
            messages.push(msg);
        }
    }
    out["messages"] = json!(messages);
    if let Some(ts) = r.get("tools") {
        let ts = ts
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("tools: expected array"))?;
        let mut tools = Vec::new();
        for (i, t) in ts.iter().enumerate() {
            let p = format!("tools[{i}]");
            let t = object(
                t,
                &p,
                &[
                    "type",
                    "name",
                    "description",
                    "input_schema",
                    "strict",
                    "cache_control",
                ],
            )?;
            if let Some(kind) = t.get("type") {
                ensure!(kind == "custom", "{p}.type: unsupported server tool");
            }
            let name = text(t.get("name"), &p)?;
            ensure!(!name.is_empty(), "{p}: empty name");
            let schema = t
                .get("input_schema")
                .ok_or_else(|| anyhow::anyhow!("{p}.input_schema: required"))?;
            ensure!(
                schema.is_object() || schema.is_boolean(),
                "{p}.input_schema: expected schema"
            );
            let mut tool = json!({"type":"function","function":{"name":name,"parameters":schema}});
            if let Some(d) = t.get("description") {
                text(Some(d), &p)?;
                tool["function"]["description"] = d.clone();
            }
            if let Some(s) = t.get("strict") {
                ensure!(s.is_boolean(), "{p}.strict: expected boolean");
                tool["function"]["strict"] = s.clone();
            }
            cache(t.get("cache_control"), &p)?;
            if let Some(c) = t.get("cache_control") {
                tool["cache_control"] = c.clone();
            }
            tools.push(tool);
        }
        out["tools"] = json!(tools);
    }
    if let Some(tc) = r.get("tool_choice") {
        let tc = object(
            tc,
            "tool_choice",
            &["type", "name", "disable_parallel_tool_use"],
        )?;
        let t = text(tc.get("type"), "tool_choice.type")?;
        out["tool_choice"] = match t {
            "auto" => json!("auto"),
            "any" => json!("required"),
            "none" => json!("none"),
            "tool" => {
                let n = text(tc.get("name"), "tool_choice.name")?;
                ensure!(!n.is_empty(), "tool_choice.name: empty");
                json!({"type":"function","function":{"name":n}})
            }
            _ => bail!("tool_choice.type: unsupported"),
        };
        ensure!(
            t == "tool" || !tc.contains_key("name"),
            "tool_choice.name: only valid with type tool"
        );
        if let Some(d) = tc.get("disable_parallel_tool_use") {
            let d = d
                .as_bool()
                .ok_or_else(|| anyhow::anyhow!("disable_parallel_tool_use: expected boolean"))?;
            out["parallel_tool_calls"] = json!(!d);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_result_ingress(content: Value, is_error: Value) -> Value {
        json!({"model":"qwen3.8-flash-next","max_tokens":64,
            "tools":[{"name":"lookup","input_schema":{"type":"object"}}],
            "messages":[{"role":"user","content":"call lookup"},
                {"role":"assistant","content":[{"type":"tool_use","id":"call_original","name":"lookup","input":{}}]},
                {"role":"user","content":[{"type":"tool_result","tool_use_id":"call_original","content":content,"is_error":is_error}]}]})
    }

    #[test]
    fn native_messages_ingress_tool_result_error_and_plain() {
        for content in [
            json!("local failure"),
            json!([{"type":"text","text":"local failure"}]),
        ] {
            for is_error in [false, true] {
                for marked in [false, true] {
                    let mut raw = tool_result_ingress(content.clone(), json!(is_error));
                    if marked {
                        raw["messages"][2]["content"][0]["cache_control"] =
                            json!({"type":"ephemeral","ttl":"5m"});
                    }
                    // The public Messages handler calls this exact planning
                    // factory before dispatch, even for an ordinary request.
                    let plan = crate::native_cache_plan::messages_cache_plan(
                        &raw,
                        crate::native_cache_instruction_roles::Family::Qwen,
                        || uuid::Uuid::new_v4().simple().to_string(),
                    )
                    .unwrap();
                    let result = &plan.body["messages"][2];
                    assert_eq!(result["role"], "tool");
                    assert_eq!(result["tool_call_id"], "call_original");
                    let expected = if is_error {
                        if let Some(text) = content.as_str() {
                            json!(format!("Tool execution error:\n{text}"))
                        } else {
                            json!([{"type":"text","text":"Tool execution error:"},{"type":"text","text":"local failure"}])
                        }
                    } else {
                        content.clone()
                    };
                    assert_eq!(result["content"], expected);
                    crate::native_cache_plan::validate_message_order(
                        &plan.body,
                        &plan.message_order,
                    )
                    .unwrap();
                    if expected.is_string() {
                        assert_eq!(
                            plan.message_order[2]["parts"][0]["content_end_byte"],
                            expected.as_str().unwrap().len()
                        );
                    } else {
                        assert_eq!(
                            plan.message_order[2]["parts"].as_array().unwrap().len(),
                            expected.as_array().unwrap().len()
                        );
                    }
                    assert_eq!(plan.writes.len(), usize::from(marked));
                    if marked {
                        assert_eq!(plan.writes[0]["message_index"], 2);
                        if let Some(text) = expected.as_str() {
                            assert_eq!(plan.writes[0]["content_end_byte"], text.len());
                        } else {
                            assert_eq!(
                                plan.writes[0]["content_block_index"],
                                expected.as_array().unwrap().len() - 1
                            );
                        }
                    }
                    crate::native_contract::ModelContract::for_model("qwen3.8-flash-next")
                        .unwrap()
                        .normalize(plan.body)
                        .unwrap();
                    // Omission has the same ordinary result semantics as false.
                    if !is_error {
                        raw["messages"][2]["content"][0]
                            .as_object_mut()
                            .unwrap()
                            .remove("is_error");
                        let omitted = crate::native_cache_plan::messages_cache_plan(
                            &raw,
                            crate::native_cache_instruction_roles::Family::Qwen,
                            || uuid::Uuid::new_v4().simple().to_string(),
                        )
                        .unwrap();
                        assert_eq!(omitted.body["messages"][2]["content"], expected);
                    }
                }
            }
        }
    }

    #[test]
    fn native_messages_ingress_tool_result_error_type_rejected() {
        for invalid in [
            Value::Null,
            json!("true"),
            json!(0),
            json!(1),
            json!({}),
            json!([]),
        ] {
            let raw = tool_result_ingress(json!("local failure"), invalid);
            let error = crate::native_cache_plan::messages_cache_plan(
                &raw,
                crate::native_cache_instruction_roles::Family::Qwen,
                || uuid::Uuid::new_v4().simple().to_string(),
            )
            .unwrap_err();
            assert!(error.to_string().contains("is_error: expected boolean"));
        }
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
    fn native_request_preserves_budget_strict_cache_and_identity() {
        let mut r = request();
        r["thinking"] = json!({"type":"enabled","budget_tokens":32});
        r["output_config"] = json!({"effort":"high"});
        r["cache_salt"] = json!("s1");
        r["tools"] = json!([{"name":"Read","strict":false,"input_schema":{"type":"object"},"cache_control":{"type":"ephemeral"}}]);
        let c = native_messages_to_chat(&r).unwrap();
        assert_eq!(c["thinking_token_budget"], 32);
        assert_eq!(c["max_tokens"], 800);
        assert_eq!(c["reasoning_effort"], "high");
        assert_eq!(c["tools"][0]["function"]["strict"], false);
        assert_eq!(c["cache_salt"], "s1");
        assert_eq!(
            c["tools"][0]["cache_control"],
            r["tools"][0]["cache_control"]
        );
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
}
