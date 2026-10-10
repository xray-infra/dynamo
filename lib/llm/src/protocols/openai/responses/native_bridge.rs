// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stateless, text-only Responses requests for native HTTP engines.
//! Validate before typed decoding so unsupported fields cannot disappear.

use super::ResponsesConversionError;
use super::{NvCreateResponse, ResponseParams};
use serde_json::{Value, json};

/// Echo only the stateless native request that was validated by the raw bridge.
pub(crate) fn native_response_params(
    raw: &Value,
) -> Result<ResponseParams, ResponsesConversionError> {
    let request: NvCreateResponse =
        serde_json::from_value(raw.clone()).map_err(|error| invalid(error.to_string()))?;
    Ok(ResponseParams {
        model: request.inner.model,
        temperature: request.inner.temperature,
        top_p: request.inner.top_p,
        max_output_tokens: request.inner.max_output_tokens,
        parallel_tool_calls: request.inner.parallel_tool_calls,
        store: request.inner.store,
        metadata: request.inner.metadata,
        tools: request.inner.tools,
        tool_choice: request.inner.tool_choice,
        instructions: request.inner.instructions,
        reasoning: request.inner.reasoning,
        text: request.inner.text,
        service_tier: request.inner.service_tier,
        include: request.inner.include,
        truncation: request.inner.truncation,
        presence_penalty: raw
            .get("presence_penalty")
            .and_then(Value::as_f64)
            .map(|v| v as f32),
        frequency_penalty: raw
            .get("frequency_penalty")
            .and_then(Value::as_f64)
            .map(|v| v as f32),
        prompt_cache_key: request.inner.prompt_cache_key,
        prompt_cache_retention: request.inner.prompt_cache_retention,
        safety_identifier: request.inner.safety_identifier,
    })
}

fn invalid(message: impl Into<String>) -> ResponsesConversionError {
    ResponsesConversionError::InvalidArgument(message.into())
}
fn unsupported(message: impl Into<String>) -> ResponsesConversionError {
    ResponsesConversionError::UnsupportedContent(message.into())
}
fn keys(value: &Value, allowed: &[&str], location: &str) -> Result<(), ResponsesConversionError> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid(format!("{location} must be an object")))?;
    if let Some(key) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(unsupported(format!("Unsupported {location} field: {key}")));
    }
    Ok(())
}
fn text_content(value: &Value) -> Result<Value, ResponsesConversionError> {
    if value.is_string() {
        return Ok(value.clone());
    }
    let parts = value
        .as_array()
        .ok_or_else(|| invalid("message content must be text or an array"))?;
    let mut result = Vec::new();
    for part in parts {
        keys(
            part,
            &["type", "text", "cache_control", "annotations", "logprobs"],
            "content",
        )?;
        if !matches!(
            part["type"].as_str(),
            Some("input_text" | "output_text" | "text")
        ) {
            return Err(unsupported(
                "Only text content is supported by native Responses",
            ));
        }
        let text = part["text"]
            .as_str()
            .ok_or_else(|| invalid("content text must be a string"))?;
        if part
            .get("annotations")
            .is_some_and(|v| !v.as_array().is_some_and(Vec::is_empty))
            || part
                .get("logprobs")
                .is_some_and(|v| !v.is_null() && !v.as_array().is_some_and(Vec::is_empty))
        {
            return Err(unsupported("Input annotations/logprobs are unsupported"));
        }
        let mut converted = json!({"type":"text", "text":text});
        if let Some(cache) = part.get("cache_control") {
            converted["cache_control"] = cache.clone();
        }
        result.push(converted);
    }
    Ok(Value::Array(result))
}

/// Build the exact canonical Chat JSON that count and generation must share.
/// Model-specific reasoning/cache/schema checks belong to the native contract.
pub(crate) fn native_response_to_chat(raw: Value) -> Result<Value, ResponsesConversionError> {
    keys(
        &raw,
        &[
            "model",
            "input",
            "instructions",
            "temperature",
            "top_p",
            "max_output_tokens",
            "stream",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "reasoning",
            "text",
            "store",
            "background",
            "truncation",
            "metadata",
            "prompt_cache_key",
            "prompt_cache_retention",
            "cache_salt",
            "chat_template_kwargs",
            "chat_template_args",
            "thinking_token_budget",
            "reasoning_budget",
            "presence_penalty",
            "frequency_penalty",
            "safety_identifier",
            "user",
            "previous_response_id",
            "conversation",
            "include",
            "service_tier",
            "nvext",
        ],
        "Responses",
    )?;
    for field in ["previous_response_id", "conversation"] {
        if raw.get(field).is_some_and(|v| !v.is_null()) {
            return Err(unsupported(format!(
                "{field} requires persistent Responses state"
            )));
        }
    }
    for field in ["store", "background"] {
        if raw.get(field).is_some_and(|v| *v != Value::Bool(false)) {
            return Err(unsupported(format!("{field} must be false")));
        }
    }
    if raw.get("truncation").is_some_and(|v| v != "disabled") {
        return Err(unsupported("Only truncation=disabled is supported"));
    }
    if raw
        .get("include")
        .is_some_and(|v| !v.as_array().is_some_and(Vec::is_empty))
    {
        return Err(unsupported("Responses include is unsupported"));
    }
    if raw
        .get("service_tier")
        .is_some_and(|v| v != "auto" && v != "default")
    {
        return Err(unsupported("Unsupported service_tier"));
    }
    if raw.get("nvext").is_some_and(|v| !v.is_null()) {
        return Err(unsupported("Native Responses nvext is unsupported"));
    }
    let mut chat = json!({"model":raw["model"], "messages":[]});
    let mut messages = Vec::new();
    if let Some(instructions) = raw.get("instructions") {
        if !instructions.is_string() {
            return Err(invalid("instructions must be text"));
        }
        messages.push(json!({"role":"system","content":instructions}));
    }
    match raw.get("input") {
        Some(Value::String(text)) => messages.push(json!({"role":"user","content":text})),
        Some(Value::Array(items)) => {
            for item in items {
                let kind = match item.get("type") {
                    None => "message",
                    Some(Value::String(kind)) => kind.as_str(),
                    _ => return Err(invalid("input item type must be text")),
                };
                match kind {
                    "message" => {
                        keys(
                            item,
                            &["type", "role", "content", "id", "status", "phase"],
                            "input message",
                        )?;
                        let role = item["role"]
                            .as_str()
                            .ok_or_else(|| invalid("message role must be text"))?;
                        if !matches!(role, "system" | "developer" | "user" | "assistant") {
                            return Err(unsupported("Unsupported message role"));
                        }
                        messages
                            .push(json!({"role":if role == "developer" { "system" } else { role },"content":text_content(&item["content"])?}));
                    }
                    "function_call" => {
                        keys(
                            item,
                            &["type", "id", "call_id", "name", "arguments", "status"],
                            "function_call",
                        )?;
                        for field in ["call_id", "name", "arguments"] {
                            if !item[field].is_string() {
                                return Err(invalid(format!("function_call {field} must be text")));
                            }
                        }
                        if item["call_id"].as_str().unwrap().trim().is_empty()
                            || item["name"].as_str().unwrap().trim().is_empty()
                            || !serde_json::from_str::<Value>(item["arguments"].as_str().unwrap())
                                .is_ok_and(|value| value.is_object())
                        {
                            return Err(invalid(
                                "function_call requires an ID, name, and JSON object arguments",
                            ));
                        }
                        messages.push(json!({"role":"assistant","content":null,"tool_calls":[{"id":item["call_id"],"type":"function","function":{"name":item["name"],"arguments":item["arguments"]}}]}));
                    }
                    "function_call_output" => {
                        keys(
                            item,
                            &["type", "id", "call_id", "output", "status"],
                            "function_call_output",
                        )?;
                        if !item["call_id"]
                            .as_str()
                            .is_some_and(|id| !id.trim().is_empty())
                        {
                            return Err(invalid("function_call_output call_id must be text"));
                        }
                        messages.push(json!({"role":"tool","tool_call_id":item["call_id"],"content":text_content(&item["output"])?}));
                    }
                    other => return Err(unsupported(format!("Unsupported input item: {other}"))),
                }
            }
        }
        _ => return Err(invalid("input must be text or an item array")),
    }
    // Multiple Responses items can describe one assistant turn.
    let mut turns: Vec<Value> = Vec::new();
    for message in messages {
        if message["role"] == "assistant"
            && turns.last().is_some_and(|last| last["role"] == "assistant")
        {
            let last = turns.last_mut().unwrap();
            if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
                if last.get("tool_calls").is_none() {
                    last["tool_calls"] = json!([]);
                }
                last["tool_calls"]
                    .as_array_mut()
                    .unwrap()
                    .extend(calls.iter().cloned());
            }
            if !message["content"].is_null() {
                let mut parts = content_parts(&last["content"]);
                parts.extend(content_parts(&message["content"]));
                last["content"] = Value::Array(parts);
            }
        } else {
            turns.push(message);
        }
    }
    chat["messages"] = Value::Array(turns);
    for field in [
        "temperature",
        "top_p",
        "stream",
        "parallel_tool_calls",
        "metadata",
        "prompt_cache_key",
        "prompt_cache_retention",
        "cache_salt",
        "chat_template_kwargs",
        "chat_template_args",
        "thinking_token_budget",
        "reasoning_budget",
        "presence_penalty",
        "frequency_penalty",
        "safety_identifier",
        "user",
    ] {
        if let Some(value) = raw.get(field) {
            chat[field] = value.clone();
        }
    }
    // Responses requires final native accounting. Its public request has no
    // stream_options field with which a client could request this explicitly.
    if chat.get("stream").and_then(Value::as_bool) == Some(true) {
        chat["stream_options"] = json!({"include_usage":true});
    }
    if let Some(value) = raw.get("max_output_tokens") {
        chat["max_tokens"] = value.clone();
    }
    if let Some(reasoning) = raw.get("reasoning") {
        keys(reasoning, &["effort", "summary"], "reasoning")?;
        if let Some(effort) = reasoning.get("effort") {
            chat["reasoning_effort"] = effort.clone();
        }
        if reasoning.get("summary").is_some_and(|v| v != "auto") {
            return Err(unsupported("Only reasoning.summary=auto is supported"));
        }
    }
    if let Some(tools) = raw.get("tools") {
        let tools = tools
            .as_array()
            .ok_or_else(|| invalid("tools must be an array"))?;
        let mut converted = Vec::new();
        for tool in tools {
            keys(
                tool,
                &["type", "name", "description", "parameters", "strict"],
                "tool",
            )?;
            if tool["type"] != "function" {
                return Err(unsupported("Only function tools are supported"));
            }
            let mut function = tool.clone();
            function.as_object_mut().unwrap().remove("type");
            converted.push(json!({"type":"function","function":function}));
        }
        chat["tools"] = Value::Array(converted);
    }
    if let Some(choice) = raw.get("tool_choice") {
        chat["tool_choice"] = if choice.is_string() {
            choice.clone()
        } else {
            keys(choice, &["type", "name"], "tool_choice")?;
            if choice["type"] != "function" {
                return Err(unsupported("Only function tool_choice is supported"));
            }
            json!({"type":"function","function":{"name":choice["name"]}})
        };
    }
    if let Some(text) = raw.get("text") {
        keys(text, &["format"], "text")?;
        let format = &text["format"];
        chat["response_format"] = match format["type"].as_str() {
            Some("text" | "json_object") => {
                keys(format, &["type"], "text.format")?;
                format.clone()
            }
            Some("json_schema") => {
                keys(
                    format,
                    &["type", "name", "description", "schema", "strict"],
                    "text.format",
                )?;
                let mut schema = format.clone();
                schema.as_object_mut().unwrap().remove("type");
                json!({"type":"json_schema","json_schema":schema})
            }
            _ => return Err(unsupported("Unsupported text.format")),
        };
    }
    Ok(chat)
}

fn content_parts(content: &Value) -> Vec<Value> {
    match content {
        Value::Array(parts) => parts.clone(),
        Value::String(text) => vec![json!({"type":"text","text":text})],
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_streaming_response_requests_native_usage() {
        let mut raw = json!({"model":"qwen3.8-flash-next","input":"hello","stream":true});
        let chat = native_response_to_chat(raw.clone()).unwrap();
        assert_eq!(chat["stream_options"], json!({"include_usage":true}));
        raw["stream"] = json!(false);
        assert!(
            native_response_to_chat(raw.clone())
                .unwrap()
                .get("stream_options")
                .is_none()
        );
        raw.as_object_mut().unwrap().remove("stream");
        assert!(
            native_response_to_chat(raw)
                .unwrap()
                .get("stream_options")
                .is_none()
        );
    }
    #[test]
    fn native_echo_params_keep_explicit_none_and_metadata() {
        let raw = json!({"model":"qwen","input":"hello","reasoning":{"effort":"none","summary":"auto"},"metadata":{"request":"cpu"},"store":false,"background":false});
        let params = native_response_params(&raw).unwrap();
        assert_eq!(
            serde_json::to_value(params.reasoning).unwrap()["effort"],
            "none"
        );
        assert_eq!(params.metadata.unwrap()["request"], "cpu");
        assert_eq!(params.store, Some(false));
    }
    #[test]
    fn native_preserves_none_budget_salt_and_tool_ids() {
        let chat = native_response_to_chat(json!({"model":"qwen","input":[{"type":"function_call","call_id":"original","name":"f","arguments":"{}"},{"type":"function_call_output","call_id":"original","output":"ok"}],"reasoning":{"effort":"none","summary":"auto"},"max_output_tokens":800,"thinking_token_budget":32,"cache_salt":"s","stream":false,"tools":[{"type":"function","name":"f","parameters":{"type":"object"},"strict":true}]})).unwrap();
        assert_eq!(chat["reasoning_effort"], "none");
        assert_eq!(chat["thinking_token_budget"], 32);
        assert_eq!(chat["cache_salt"], "s");
        assert_eq!(chat["max_tokens"], 800);
        assert_eq!(chat["stream"], false);
        assert_eq!(chat["messages"][0]["tool_calls"][0]["id"], "original");
        assert_eq!(chat["messages"][1]["tool_call_id"], "original");
        assert_eq!(chat["tools"][0]["function"]["strict"], true);
    }
    #[test]
    fn native_rejects_semantics_that_would_disappear() {
        for field in ["previous_response_id", "conversation", "unknown"] {
            let mut request = json!({"model":"qwen","input":"hi"});
            request[field] = json!("x");
            assert!(native_response_to_chat(request).is_err());
        }
        for item in [
            json!({"type":"item_reference","id":"x"}),
            json!({"role":"system","content":[{"type":"input_image","image_url":"x"}]}),
            json!({"type":"reasoning","encrypted_content":"x"}),
        ] {
            assert!(native_response_to_chat(json!({"model":"qwen","input":[item]})).is_err());
        }
        assert!(
            native_response_to_chat(
                json!({"model":"qwen","input":"hi","reasoning":{"summary":"detailed"}})
            )
            .is_err()
        );
    }
}
