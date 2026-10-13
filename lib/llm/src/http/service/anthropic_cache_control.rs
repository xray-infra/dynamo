// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Anthropic `cache_control` validation and source-to-renderer coordinate lowering.

use dynamo_renderer::{PromptBoundaryPosition, PromptBoundaryTarget};
use serde_json::Value;

pub(crate) const CACHE_CONTROL_CONTEXT_KEY: &str = "dynamo.llm.anthropic_cache_control";

#[derive(Clone, Debug)]
pub(crate) struct CacheControlTarget {
    pub target: PromptBoundaryTarget,
    pub duration_seconds: u64,
}

fn directive(value: &Value, location: &str) -> Result<u64, String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{location}.cache_control must be an object"))?;
    if object.get("type").and_then(Value::as_str) != Some("ephemeral") {
        return Err(format!(
            "{location}.cache_control.type must be \"ephemeral\""
        ));
    }
    let ttl = object.get("ttl").and_then(Value::as_str).unwrap_or("5m");
    match ttl {
        "5m" => Ok(300),
        "1h" => Ok(3600),
        _ => Err(format!(
            "{location}.cache_control.ttl must be \"5m\" or \"1h\""
        )),
    }
}

fn has_cache_control(raw: &Value) -> bool {
    raw.get("cache_control").is_some()
        || raw
            .get("tools")
            .and_then(Value::as_array)
            .is_some_and(|tools| tools.iter().any(|tool| tool.get("cache_control").is_some()))
        || raw
            .get("system")
            .and_then(Value::as_array)
            .is_some_and(|blocks| {
                blocks
                    .iter()
                    .any(|block| block.get("cache_control").is_some())
            })
        || raw
            .get("messages")
            .and_then(Value::as_array)
            .is_some_and(|messages| {
                messages.iter().any(|message| {
                    message.get("cache_control").is_some()
                        || message
                            .get("content")
                            .and_then(Value::as_array)
                            .is_some_and(|blocks| {
                                blocks.iter().any(|block| {
                                    block.get("cache_control").is_some()
                                        || block
                                            .get("content")
                                            .and_then(Value::as_array)
                                            .is_some_and(|nested| {
                                                nested
                                                    .iter()
                                                    .any(|item| item.get("cache_control").is_some())
                                            })
                                })
                            })
                })
            })
}

fn push_target(
    targets: &mut Vec<CacheControlTarget>,
    id: String,
    position: PromptBoundaryPosition,
    control: &Value,
    location: &str,
) -> Result<(), String> {
    targets.push(CacheControlTarget {
        target: PromptBoundaryTarget::new(id, position),
        duration_seconds: directive(control, location)?,
    });
    Ok(())
}

/// Translate raw Anthropic coordinates to the exact normalized values supplied to the renderer.
///
/// This intentionally runs against the raw request because the typed Anthropic protocol folds
/// system blocks and drops per-block source coordinates during Messages-to-Chat conversion.
pub(crate) fn lower_cache_control(raw: &Value) -> Result<Vec<CacheControlTarget>, String> {
    if !has_cache_control(raw) {
        return Ok(Vec::new());
    }
    if raw.get("cache_control").is_some() {
        return Err("top-level cache_control is not supported".to_string());
    }

    let mut targets = Vec::new();

    if let Some(tools) = raw.get("tools").and_then(Value::as_array) {
        for (tool_index, tool) in tools.iter().enumerate() {
            let location = format!("tools[{tool_index}]");
            if !matches!(
                tool.get("type").and_then(Value::as_str),
                None | Some("custom")
            ) || !tool.get("input_schema").is_some_and(Value::is_object)
            {
                return Err(format!(
                    "{location}: cache_control cannot be mapped through an unsupported tool shape"
                ));
            }
            if let Some(control) = tool.get("cache_control") {
                push_target(
                    &mut targets,
                    format!("anthropic-cache-control-tools-{tool_index}"),
                    PromptBoundaryPosition::AfterTool { tool_index },
                    control,
                    &location,
                )?;
            }
        }
    }

    let mut normalized_message_index = 0usize;
    if let Some(system) = raw.get("system") {
        match system {
            Value::String(_) => normalized_message_index = 1,
            Value::Array(blocks) => {
                let mut byte_offset = 0usize;
                for (block_index, block) in blocks.iter().enumerate() {
                    let location = format!("system[{block_index}]");
                    if block.get("type").and_then(Value::as_str) != Some("text") {
                        return Err(format!(
                            "{location}: cache_control cannot be mapped through a non-text system block"
                        ));
                    }
                    let text = block.get("text").and_then(Value::as_str).ok_or_else(|| {
                        format!("{location}.text must be a string for cache_control")
                    })?;
                    if block_index > 0 {
                        byte_offset += 1; // Messages-to-Chat joins system blocks with '\n'.
                    }
                    byte_offset += text.len();
                    if let Some(control) = block.get("cache_control") {
                        push_target(
                            &mut targets,
                            format!("anthropic-cache-control-system-{block_index}"),
                            PromptBoundaryPosition::MessageContentByteOffset {
                                message_index: 0,
                                byte_offset,
                            },
                            control,
                            &location,
                        )?;
                    }
                }
                normalized_message_index = 1;
            }
            _ => return Err("system must be a string or an array of text blocks".to_string()),
        }
    }

    let messages = raw
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| "messages must be an array".to_string())?;
    for (source_message_index, message) in messages.iter().enumerate() {
        if message.get("cache_control").is_some() {
            return Err(format!(
                "messages[{source_message_index}].cache_control is not supported; cache_control must annotate a content block"
            ));
        }
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("messages[{source_message_index}].role must be a string"))?;
        let Some(blocks) = message.get("content").and_then(Value::as_array) else {
            // Plain text always converts to exactly one normalized message and cannot own a
            // block-level directive.
            normalized_message_index += 1;
            continue;
        };

        match role {
            "system" => {
                let target_message_index = normalized_message_index;
                let mut byte_offset = 0usize;
                let mut text_count = 0usize;
                for (block_index, block) in blocks.iter().enumerate() {
                    let location =
                        format!("messages[{source_message_index}].content[{block_index}]");
                    if block.get("type").and_then(Value::as_str) != Some("text") {
                        return Err(format!(
                            "{location}: cache_control cannot be mapped through a skipped system block"
                        ));
                    }
                    let text = block.get("text").and_then(Value::as_str).ok_or_else(|| {
                        format!("{location}.text must be a string for cache_control")
                    })?;
                    if text_count > 0 {
                        byte_offset += 1;
                    }
                    text_count += 1;
                    byte_offset += text.len();
                    if let Some(control) = block.get("cache_control") {
                        push_target(
                            &mut targets,
                            format!(
                                "anthropic-cache-control-messages-{source_message_index}-{block_index}"
                            ),
                            PromptBoundaryPosition::MessageContentByteOffset {
                                message_index: target_message_index,
                                byte_offset,
                            },
                            control,
                            &location,
                        )?;
                    }
                }
                normalized_message_index += 1;
            }
            "assistant" => {
                let target_message_index = normalized_message_index;
                let mut text_offset = 0usize;
                let mut tool_call_index = 0usize;
                for (block_index, block) in blocks.iter().enumerate() {
                    let location =
                        format!("messages[{source_message_index}].content[{block_index}]");
                    match block.get("type").and_then(Value::as_str) {
                        Some("text") => {
                            let text =
                                block.get("text").and_then(Value::as_str).ok_or_else(|| {
                                    format!("{location}.text must be a string for cache_control")
                                })?;
                            text_offset += text.len();
                            if let Some(control) = block.get("cache_control") {
                                push_target(
                                    &mut targets,
                                    format!(
                                        "anthropic-cache-control-messages-{source_message_index}-{block_index}"
                                    ),
                                    PromptBoundaryPosition::MessageContentByteOffset {
                                        message_index: target_message_index,
                                        byte_offset: text_offset,
                                    },
                                    control,
                                    &location,
                                )?;
                            }
                        }
                        Some("tool_use") => {
                            if let Some(control) = block.get("cache_control") {
                                push_target(
                                    &mut targets,
                                    format!(
                                        "anthropic-cache-control-messages-{source_message_index}-{block_index}"
                                    ),
                                    PromptBoundaryPosition::AfterMessageToolCall {
                                        message_index: target_message_index,
                                        tool_call_index,
                                    },
                                    control,
                                    &location,
                                )?;
                            }
                            tool_call_index += 1;
                        }
                        Some("thinking") | Some("redacted_thinking") => {
                            return Err(format!(
                                "{location}: cache_control cannot be mapped through thinking content"
                            ));
                        }
                        Some("image") => {
                            return Err(format!(
                                "{location}: cache_control cannot be mapped through media content"
                            ));
                        }
                        _ => {
                            return Err(format!(
                                "{location}: cache_control cannot be mapped through a skipped or unsupported assistant block"
                            ));
                        }
                    }
                }
                normalized_message_index += 1;
            }
            "user" => {
                let mut pending_text_offset = 0usize;
                let mut pending_text = false;
                for (block_index, block) in blocks.iter().enumerate() {
                    let location =
                        format!("messages[{source_message_index}].content[{block_index}]");
                    match block.get("type").and_then(Value::as_str) {
                        Some("text") => {
                            let text =
                                block.get("text").and_then(Value::as_str).ok_or_else(|| {
                                    format!("{location}.text must be a string for cache_control")
                                })?;
                            pending_text = true;
                            pending_text_offset += text.len();
                            if let Some(control) = block.get("cache_control") {
                                push_target(
                                    &mut targets,
                                    format!(
                                        "anthropic-cache-control-messages-{source_message_index}-{block_index}"
                                    ),
                                    PromptBoundaryPosition::MessageContentByteOffset {
                                        message_index: normalized_message_index,
                                        byte_offset: pending_text_offset,
                                    },
                                    control,
                                    &location,
                                )?;
                            }
                        }
                        Some("tool_result") => {
                            if pending_text {
                                normalized_message_index += 1;
                                pending_text = false;
                                pending_text_offset = 0;
                            }
                            if block
                                .get("content")
                                .and_then(Value::as_array)
                                .is_some_and(|items| {
                                    items.iter().any(|item| {
                                        item.get("type").and_then(Value::as_str) != Some("text")
                                            || item.get("cache_control").is_some()
                                    })
                                })
                            {
                                return Err(format!(
                                    "{location}: cache_control cannot be mapped through media, nested, or unsupported tool_result content"
                                ));
                            }
                            if let Some(control) = block.get("cache_control") {
                                push_target(
                                    &mut targets,
                                    format!(
                                        "anthropic-cache-control-messages-{source_message_index}-{block_index}"
                                    ),
                                    PromptBoundaryPosition::AfterMessage {
                                        message_index: normalized_message_index,
                                    },
                                    control,
                                    &location,
                                )?;
                            }
                            normalized_message_index += 1;
                        }
                        Some("image") => {
                            return Err(format!(
                                "{location}: cache_control cannot be mapped through media content"
                            ));
                        }
                        _ => {
                            return Err(format!(
                                "{location}: cache_control cannot be mapped through a skipped or unsupported user block"
                            ));
                        }
                    }
                }
                if pending_text {
                    normalized_message_index += 1;
                }
            }
            _ => {
                return Err(format!(
                    "messages[{source_message_index}].role is unsupported"
                ));
            }
        }
    }

    if targets.len() > 4 {
        return Err("cache_control supports at most 4 breakpoints".to_string());
    }
    let mut saw_five_minutes = false;
    for target in &targets {
        match target.duration_seconds {
            300 => saw_five_minutes = true,
            3600 if saw_five_minutes => {
                return Err("1h cache_control breakpoints must precede 5m breakpoints".to_string());
            }
            _ => {}
        }
    }
    Ok(targets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn lowers_tools_system_text_tool_calls_and_tool_result_coordinates() {
        let raw = json!({
            "tools": [{"name":"lookup", "input_schema":{}, "cache_control":{"type":"ephemeral","ttl":"1h"}}],
            "system": [
                {"type":"text", "text":"one"},
                {"type":"text", "text":"two", "cache_control":{"type":"ephemeral","ttl":"1h"}}
            ],
            "messages": [
                {"role":"assistant", "content":[
                    {"type":"tool_use", "id":"a", "name":"lookup", "input":{}},
                    {"type":"tool_use", "id":"b", "name":"lookup", "input":{}, "cache_control":{"type":"ephemeral"}}
                ]},
                {"role":"user", "content":[
                    {"type":"tool_result", "tool_use_id":"b", "content":"ok", "cache_control":{"type":"ephemeral"}}
                ]}
            ]
        });
        let targets = lower_cache_control(&raw).unwrap();
        assert_eq!(targets.len(), 4);
        assert_eq!(targets[0].duration_seconds, 3600);
        assert_eq!(
            targets[1].target.position,
            PromptBoundaryPosition::MessageContentByteOffset {
                message_index: 0,
                byte_offset: 7
            }
        );
        assert_eq!(
            targets[2].target.position,
            PromptBoundaryPosition::AfterMessageToolCall {
                message_index: 1,
                tool_call_index: 1
            }
        );
        assert_eq!(
            targets[3].target.position,
            PromptBoundaryPosition::AfterMessage { message_index: 2 }
        );
    }

    #[test]
    fn rejects_media_skipped_nested_and_ttl_order() {
        let media = json!({"messages":[{"role":"user","content":[
            {"type":"image","source":{}},
            {"type":"text","text":"x","cache_control":{"type":"ephemeral"}}
        ]}]});
        assert!(lower_cache_control(&media).unwrap_err().contains("media"));

        let skipped = json!({"messages":[{"role":"assistant","content":[
            {"type":"thinking","thinking":"x"},
            {"type":"text","text":"x","cache_control":{"type":"ephemeral"}}
        ]}]});
        assert!(
            lower_cache_control(&skipped)
                .unwrap_err()
                .contains("thinking")
        );

        let nested = json!({"messages":[{"role":"user","content":[{
            "type":"tool_result", "tool_use_id":"a", "content":[{
                "type":"text", "text":"x", "cache_control":{"type":"ephemeral"}
            }]
        }]}]});
        assert!(lower_cache_control(&nested).unwrap_err().contains("nested"));

        let misplaced = json!({"messages":[{
            "role":"user", "content":"x", "cache_control":{"type":"ephemeral"}
        }]});
        assert!(
            lower_cache_control(&misplaced)
                .unwrap_err()
                .contains("content block")
        );

        let invalid_ttl = json!({"messages":[{"role":"user","content":[
            {"type":"text","text":"x","cache_control":{"type":"ephemeral","ttl":"10m"}}
        ]}]});
        assert!(
            lower_cache_control(&invalid_ttl)
                .unwrap_err()
                .contains("5m")
        );

        let misordered = json!({"messages":[{"role":"user","content":[
            {"type":"text","text":"a","cache_control":{"type":"ephemeral","ttl":"5m"}},
            {"type":"text","text":"b","cache_control":{"type":"ephemeral","ttl":"1h"}}
        ]}]});
        assert!(
            lower_cache_control(&misordered)
                .unwrap_err()
                .contains("must precede")
        );
    }
}
