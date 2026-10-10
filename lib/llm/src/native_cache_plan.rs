//! Private planning only. A plan does not qualify TTL retention or authorize a hit.
use crate::{
    native_cache_instruction_roles::{Family, normalize_native_instruction_roles},
    native_cache_messages::native_messages_to_chat,
};
use anyhow::{Result, bail, ensure};
use serde_json::{Value, json};

#[derive(Clone, Debug)]
struct Block {
    selector: Value,
    ttl: Option<u64>,
    eligible: bool,
    run: Option<&'static str>,
}
#[derive(Debug)]
pub struct CachePlan {
    pub body: Value,
    pub writes: Vec<Value>,
    pub reads: Vec<Value>,
    /// Server-generated source order within each canonical native role wrapper.
    pub message_order: Vec<Value>,
}

fn ttl(marker: Option<&Value>) -> Result<Option<u64>> {
    marker
        .map(|v| {
            let o = v
                .as_object()
                .ok_or_else(|| anyhow::anyhow!("cache_control must be an object"))?;
            ensure!(
                o.keys().all(|k| ["type", "ttl"].contains(&k.as_str())),
                "unsupported cache_control field"
            );
            ensure!(
                o.get("type").and_then(Value::as_str) == Some("ephemeral"),
                "unsupported cache type"
            );
            match o.get("ttl").map(Value::as_str) {
                None | Some(Some("5m")) => Ok(300_000),
                Some(Some("1h")) => Ok(3_600_000),
                _ => bail!("unsupported cache TTL"),
            }
        })
        .transpose()
}
fn block(
    selector: Value,
    marker: Option<&Value>,
    eligible: bool,
    run: Option<&'static str>,
) -> Result<Block> {
    let ttl = ttl(marker)?;
    ensure!(
        eligible || ttl.is_none(),
        "empty text/thinking/sub-content blocks cannot carry cache_control"
    );
    Ok(Block {
        selector,
        ttl,
        eligible,
        run,
    })
}
fn content_end(message: usize, content: &Value) -> Value {
    match content {
        Value::String(s) => {
            json!({"source":"messages","message_index":message,"content_end_byte":s.len()})
        }
        Value::Array(a) if !a.is_empty() => {
            json!({"source":"messages","message_index":message,"content_block_index":a.len()-1})
        }
        _ => json!({"source":"messages","message_index":message}),
    }
}
fn plain_blocks(content: &Value, message: usize, blocks: &mut Vec<Block>) -> Result<()> {
    if let Some(s) = content.as_str() {
        blocks.push(block(
            content_end(message, content),
            None,
            !s.is_empty(),
            None,
        )?);
    } else if let Some(a) = content.as_array() {
        for (i, b) in a.iter().enumerate() {
            let s = b.get("text").and_then(Value::as_str).unwrap_or("");
            blocks.push(block(
                json!({"source":"messages","message_index":message,"content_block_index":i}),
                b.get("cache_control"),
                !s.is_empty(),
                None,
            )?);
        }
    }
    Ok(())
}
fn content_order(content: &Value) -> Vec<Value> {
    match content {
        Value::String(s) => vec![json!({"kind":"content","content_end_byte":s.len()})],
        Value::Array(a) => (0..a.len())
            .map(|i| json!({"kind":"content","content_block_index":i}))
            .collect(),
        _ => vec![],
    }
}
fn strip_metadata(body: &mut Value) {
    body.as_object_mut().unwrap().remove("cache_control");
    if let Some(ts) = body.get_mut("tools").and_then(Value::as_array_mut) {
        for t in ts {
            t.as_object_mut().unwrap().remove("cache_control");
        }
    }
    for m in body["messages"].as_array_mut().unwrap() {
        m.as_object_mut().unwrap().remove("cache_control");
        for field in ["content", "tool_calls"] {
            if let Some(bs) = m.get_mut(field).and_then(Value::as_array_mut) {
                for b in bs {
                    if let Some(o) = b.as_object_mut() {
                        o.remove("cache_control");
                    }
                }
            }
        }
    }
}

/// Accept public Messages fields, then generate authority-owned endpoint identities.
/// `endpoint_id` is a callback from the internal authority, never a client field.
pub fn messages_cache_plan(
    raw: &Value,
    family: Family,
    endpoint_id: impl FnMut() -> String,
) -> Result<CachePlan> {
    let body = native_messages_to_chat(raw)?;
    let mut blocks = Vec::new();
    let mut message_order = Vec::new();
    if let Some(ts) = raw.get("tools").and_then(Value::as_array) {
        for (i, t) in ts.iter().enumerate() {
            blocks.push(block(
                json!({"source":"tools","tool_index":i}),
                t.get("cache_control"),
                true,
                None,
            )?);
        }
    }
    let mut mi = 0;
    if let Some(s) = raw.get("system") {
        plain_blocks(s, mi, &mut blocks)?;
        message_order.push(json!({"message_index":mi,"parts":content_order(s)}));
        mi += 1;
    }
    for m in raw["messages"].as_array().unwrap() {
        let c = &m["content"];
        if c.is_string() {
            plain_blocks(c, mi, &mut blocks)?;
            message_order.push(json!({"message_index":mi,"parts":content_order(c)}));
            mi += 1;
            continue;
        }
        let mut text_index = 0;
        let mut call_index = 0;
        let mut pending = false;
        let mut parts = Vec::new();
        let mut reasoning_end_byte = 0;
        for b in c.as_array().unwrap() {
            match b["type"].as_str().unwrap() {
                "text" => {
                    let s = b["text"].as_str().unwrap();
                    blocks.push(block(json!({"source":"messages","message_index":mi,"content_block_index":text_index}), b.get("cache_control"), !s.is_empty(), None)?);
                    parts.push(json!({"kind":"content","content_block_index":text_index}));
                    text_index += 1;
                    pending = true;
                }
                "tool_use" => {
                    blocks.push(block(json!({"source":"messages","message_index":mi,"tool_call_index":call_index}), b.get("cache_control"), true, Some("tool_use"))?);
                    parts.push(json!({"kind":"tool_call","tool_call_index":call_index}));
                    call_index += 1;
                    pending = true;
                }
                "tool_result" => {
                    if text_index > 0 {
                        message_order
                            .push(json!({"message_index":mi,"parts":std::mem::take(&mut parts)}));
                        mi += 1;
                        text_index = 0;
                        pending = false;
                    }
                    let raw_content = b.get("content").cloned().unwrap_or(json!(""));
                    ensure!(
                        !raw_content
                            .as_array()
                            .is_some_and(|a| a.iter().any(|b| b.get("cache_control").is_some())),
                        "tool_result sub-content cannot carry cache_control; mark the result block"
                    );
                    // The public converter includes the model-visible error status.
                    // Cache selectors must describe that exact canonical content.
                    let content = body["messages"]
                        .get(mi)
                        .and_then(|message| message.get("content"))
                        .ok_or_else(|| anyhow::anyhow!("tool_result canonical content missing"))?;
                    blocks.push(block(
                        content_end(mi, content),
                        b.get("cache_control"),
                        true,
                        Some("tool_result"),
                    )?);
                    message_order.push(json!({"message_index":mi,"parts":content_order(content)}));
                    mi += 1;
                }
                "thinking" => {
                    pending |= !b["thinking"].as_str().unwrap().is_empty();
                    let start = reasoning_end_byte;
                    reasoning_end_byte += b["thinking"].as_str().unwrap().len();
                    parts.push(json!({"kind":"reasoning","content_start_byte":start,"content_end_byte":reasoning_end_byte}));
                    blocks.push(block(
                        json!({"source":"messages","message_index":mi}),
                        None,
                        false,
                        None,
                    )?);
                }
                _ => unreachable!(),
            }
        }
        if pending {
            message_order.push(json!({"message_index":mi,"parts":parts}));
            mi += 1;
        }
    }
    finish_plan(
        body,
        blocks,
        message_order,
        family,
        ttl(raw.get("cache_control"))?,
        endpoint_id,
    )
}

/// Plan public Chat markers using the same breakpoints, lifetime rules and
/// native renderer coordinates as Messages. Sampling fields stay unchanged.
pub fn chat_cache_plan(
    raw: &Value,
    family: Family,
    endpoint_id: impl FnMut() -> String,
) -> Result<CachePlan> {
    let body = raw.clone();
    ensure!(body.is_object(), "Chat request must be an object");
    let marked = body.get("cache_control").is_some()
        || body
            .get("tools")
            .and_then(Value::as_array)
            .is_some_and(|tools| tools.iter().any(|t| t.get("cache_control").is_some()))
        || body
            .get("messages")
            .and_then(Value::as_array)
            .is_some_and(|messages| {
                messages.iter().any(|m| {
                    m.get("cache_control").is_some()
                        || ["content", "tool_calls"].iter().any(|field| {
                            m.get(field).and_then(Value::as_array).is_some_and(|items| {
                                items.iter().any(|item| item.get("cache_control").is_some())
                            })
                        })
                })
            });
    if !marked {
        return Ok(CachePlan {
            body,
            writes: vec![],
            reads: vec![],
            message_order: vec![],
        });
    }
    let mut blocks = Vec::new();
    let mut message_order = Vec::new();
    if let Some(tools) = body.get("tools").and_then(Value::as_array) {
        for (index, tool) in tools.iter().enumerate() {
            blocks.push(block(
                json!({"source":"tools","tool_index":index}),
                tool.get("cache_control"),
                true,
                None,
            )?);
        }
    }
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("messages must be an array"))?;
    for (index, message) in messages.iter().enumerate() {
        ensure!(message.is_object(), "message must be an object");
        let first = blocks.len();
        let content = &message["content"];
        plain_blocks(content, index, &mut blocks)?;
        if message["role"] == "tool" {
            for block in &mut blocks[first..] {
                block.run = Some("tool_result");
            }
        }
        let mut parts = content_order(content);
        if let Some(reasoning) = message
            .get("reasoning_content")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            parts.insert(0, json!({"kind":"reasoning","content_start_byte":0,"content_end_byte":reasoning.len()}));
        }
        if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
            for (call_index, call) in calls.iter().enumerate() {
                blocks.push(block(
                    json!({"source":"messages","message_index":index,"tool_call_index":call_index}),
                    call.get("cache_control"),
                    true,
                    Some("tool_use"),
                )?);
                parts.push(json!({"kind":"tool_call","tool_call_index":call_index}));
            }
        }
        if let Some(retain) = ttl(message.get("cache_control"))? {
            let last = blocks[first..]
                .iter_mut()
                .rev()
                .find(|b| b.eligible)
                .ok_or_else(|| {
                    anyhow::anyhow!("message cache_control requires a nonempty breakpoint")
                })?;
            ensure!(
                last.ttl.is_none_or(|current| current == retain),
                "message and block cache TTL conflict"
            );
            last.ttl = Some(retain);
        }
        message_order.push(json!({"message_index":index,"parts":parts}));
    }
    finish_plan(
        body,
        blocks,
        message_order,
        family,
        ttl(raw.get("cache_control"))?,
        endpoint_id,
    )
}

fn finish_plan(
    mut body: Value,
    mut blocks: Vec<Block>,
    message_order: Vec<Value>,
    family: Family,
    automatic: Option<u64>,
    mut endpoint_id: impl FnMut() -> String,
) -> Result<CachePlan> {
    if let Some(ttl) = automatic {
        let last = blocks
            .iter_mut()
            .rev()
            .find(|b| b.eligible)
            .ok_or_else(|| {
                anyhow::anyhow!("cache_control requires an eligible nonempty breakpoint")
            })?;
        if let Some(explicit) = last.ttl {
            ensure!(
                explicit == ttl,
                "automatic and final explicit cache TTL conflict"
            );
        } else {
            last.ttl = Some(ttl);
        }
    }
    let marked: Vec<_> = blocks
        .iter()
        .enumerate()
        .filter(|(_, b)| b.ttl.is_some())
        .collect();
    ensure!(
        marked.len() <= 4,
        "at most four cache breakpoints are allowed"
    );
    let mut previous = 3_600_000;
    for (_, b) in &marked {
        let current = b.ttl.unwrap();
        ensure!(
            current <= previous,
            "one-hour cache breakpoint must precede five-minute breakpoints"
        );
        previous = current;
    }
    let mut writes = Vec::new();
    let mut reads = Vec::new();
    for (end, b) in marked {
        let mut selector = b.selector.clone();
        selector["ttl_ms"] = json!(b.ttl.unwrap());
        writes.push(selector);
        // Exactly 20 positions including the marked block, with consecutive
        // parallel tool runs collapsed into one position. Never write these.
        let mut cursor = end as isize;
        for _ in 0..20 {
            if cursor < 0 {
                break;
            }
            let position = &blocks[cursor as usize];
            if position.eligible && !reads.contains(&position.selector) {
                reads.push(position.selector.clone());
            }
            let run = position.run;
            cursor -= 1;
            if let Some(run) = run {
                while cursor >= 0 && blocks[cursor as usize].run == Some(run) {
                    cursor -= 1;
                }
            }
        }
    }
    strip_metadata(&mut body);
    normalize_cache_selectors(&mut body, &mut writes, &mut reads, family)?;
    validate_message_order(&body, &message_order)?;
    // Coordinate identity is shared, but producer reservation identity is not.
    // A read may refresh an old 1h hold while a new 5m write targets the same
    // physical prefix. Distinct namespaces keep those logical holds independent.
    let mut identities: Vec<(Value, String)> = Vec::new();
    for (namespace, selectors) in [("write", &mut writes), ("read", &mut reads)] {
        for selector in selectors.iter_mut() {
            let mut key = selector.clone();
            key.as_object_mut().unwrap().remove("ttl_ms");
            let id = if let Some((_, id)) = identities.iter().find(|(old, _)| old == &key) {
                id.clone()
            } else {
                let id = endpoint_id();
                ensure!(
                    !id.is_empty() && !identities.iter().any(|(_, old)| old == &id),
                    "authority endpoint IDs must be unique and nonempty"
                );
                identities.push((key, id.clone()));
                id
            };
            selector["endpoint_id"] = json!(format!("{namespace}:{id}"));
        }
    }
    Ok(CachePlan {
        body,
        writes,
        reads,
        message_order,
    })
}

fn text_and_boundary(content: &Value, selector: &Value) -> Result<(String, usize)> {
    if let Some(s) = content.as_str() {
        let end = selector
            .get("content_end_byte")
            .and_then(Value::as_u64)
            .unwrap_or(s.len() as u64) as usize;
        ensure!(
            end <= s.len() && s.is_char_boundary(end),
            "invalid UTF-8 cache boundary"
        );
        return Ok((s.into(), end));
    }
    let a = content
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("instruction content must be text"))?;
    let index = selector
        .get("content_block_index")
        .and_then(Value::as_u64)
        .unwrap_or(a.len().saturating_sub(1) as u64) as usize;
    ensure!(index < a.len(), "invalid instruction content boundary");
    let mut text = String::new();
    let mut end = 0;
    for (i, b) in a.iter().enumerate() {
        text.push_str(
            b["text"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("instruction content must be text"))?,
        );
        if i == index {
            end = text.len();
        }
    }
    Ok((text, end))
}
pub(crate) fn remap_selector(selector: &mut Value, body: &Value, family: Family) -> Result<()> {
    if matches!(family, Family::DeepSeek | Family::GlmFlash) || selector["source"] != "messages" {
        return Ok(());
    }
    let messages = body["messages"].as_array().unwrap();
    let leading = messages
        .iter()
        .take_while(|m| matches!(m["role"].as_str(), Some("system" | "developer")))
        .count();
    let merge = leading > 1 && messages[..leading].iter().any(|m| m["role"] == "developer");
    let old = selector["message_index"].as_u64().unwrap() as usize;
    ensure!(
        old < messages.len(),
        "cache boundary message index out of range"
    );
    if merge && old < leading {
        let mut offset = "The following instruction messages retain their original order and priority. System instructions take precedence over developer instructions.\n\n".len();
        for (i, m) in messages[..leading].iter().enumerate() {
            offset += format!("[Instruction {}: {}]\n", i + 1, m["role"].as_str().unwrap()).len();
            let (text, end) = text_and_boundary(
                &m["content"],
                if i == old { selector } else { &Value::Null },
            )?;
            if i == old {
                selector["message_index"] = json!(0);
                selector
                    .as_object_mut()
                    .unwrap()
                    .remove("content_block_index");
                selector["content_end_byte"] = json!(offset + end);
                return Ok(());
            }
            offset += text.len() + 2;
        }
    }
    if old >= leading && messages[old]["role"] == "developer" {
        let (_, end) = text_and_boundary(&messages[old]["content"], selector)?;
        selector
            .as_object_mut()
            .unwrap()
            .remove("content_block_index");
        selector["content_end_byte"] = json!("[Instruction role: developer]\nSystem instructions take precedence over this developer instruction.\n\n".len()+end);
    }
    if merge && old >= leading {
        selector["message_index"] = json!(old - leading + 1);
    }
    Ok(())
}
/// Called at the actual NativeContract role-normalization boundary before
/// canonical input is consumed by either native Count or generation.
pub fn normalize_cache_selectors(
    body: &mut Value,
    writes: &mut [Value],
    reads: &mut [Value],
    family: Family,
) -> Result<()> {
    for selector in writes.iter_mut().chain(reads.iter_mut()) {
        remap_selector(selector, body, family)?;
    }
    normalize_native_instruction_roles(
        body.as_object_mut()
            .ok_or_else(|| anyhow::anyhow!("native body must be an object"))?,
        family,
    )
}

/// Add only server-generated selectors after public parsing and role normalization.
/// The returned token input has identical content in Count and generation bodies.
pub fn native_render_body(plan: &CachePlan) -> Value {
    let mut body = plan.body.clone();
    body["prompt_cache_blocks"] = json!(plan.writes);
    body["prompt_cache_read_candidates"] = json!(plan.reads);
    if !plan.writes.is_empty() {
        body["prompt_cache_message_order"] = json!(plan.message_order);
    }
    body
}

/// Validate source order against the actual canonical Chat fields. No text or
/// tool-call payload is duplicated in metadata; every source field is consumed
/// exactly once. Native serializers must keep one role wrapper and use this
/// sequence for both ordinary full render and marked render in cache mode.
pub fn validate_message_order(body: &Value, order: &[Value]) -> Result<()> {
    let messages = body["messages"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("messages missing"))?;
    ensure!(
        order.len() == messages.len(),
        "ordered message coverage differs from canonical Chat"
    );
    for (expected, mapping) in order.iter().enumerate() {
        ensure!(
            mapping["message_index"].as_u64() == Some(expected as u64),
            "ordered message index must match canonical Chat"
        );
        let message = &messages[expected];
        let parts = mapping["parts"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("ordered parts must be an array"))?;
        let content = &message["content"];
        let content_count = if content.is_string() {
            1
        } else {
            content.as_array().map_or(0, Vec::len)
        };
        let calls_count = message
            .get("tool_calls")
            .and_then(Value::as_array)
            .map_or(0, Vec::len);
        let mut content_seen = vec![false; content_count];
        let mut calls_seen = vec![false; calls_count];
        let reasoning = message
            .get("reasoning_content")
            .and_then(Value::as_str)
            .unwrap_or("");
        let mut reasoning_end = 0;
        for part in parts {
            match part["kind"].as_str() {
                Some("content") => {
                    let index = if content.is_string() {
                        ensure!(
                            part["content_end_byte"].as_u64()
                                == Some(content.as_str().unwrap().len() as u64),
                            "ordered string boundary differs"
                        );
                        0
                    } else {
                        part["content_block_index"]
                            .as_u64()
                            .ok_or_else(|| anyhow::anyhow!("ordered block index missing"))?
                            as usize
                    };
                    ensure!(
                        index < content_seen.len() && !content_seen[index],
                        "ordered content missing or duplicated"
                    );
                    content_seen[index] = true;
                }
                Some("tool_call") => {
                    let index = part["tool_call_index"]
                        .as_u64()
                        .ok_or_else(|| anyhow::anyhow!("ordered call index missing"))?
                        as usize;
                    ensure!(
                        index < calls_seen.len() && !calls_seen[index],
                        "ordered tool call missing or duplicated"
                    );
                    calls_seen[index] = true;
                }
                Some("reasoning") => {
                    let start = part["content_start_byte"]
                        .as_u64()
                        .ok_or_else(|| anyhow::anyhow!("ordered reasoning start missing"))?
                        as usize;
                    let end = part["content_end_byte"]
                        .as_u64()
                        .ok_or_else(|| anyhow::anyhow!("ordered reasoning end missing"))?
                        as usize;
                    ensure!(
                        start == reasoning_end
                            && end >= start
                            && end <= reasoning.len()
                            && reasoning.is_char_boundary(start)
                            && reasoning.is_char_boundary(end),
                        "ordered reasoning boundary differs"
                    );
                    reasoning_end = end;
                }
                _ => bail!("unknown ordered part"),
            }
        }
        ensure!(
            content_seen.iter().all(|x| *x)
                && calls_seen.iter().all(|x| *x)
                && reasoning_end == reasoning.len(),
            "ordered source coverage incomplete"
        );
    }
    Ok(())
}
