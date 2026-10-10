use serde_json::{Map, Value, json};
#[derive(Clone, Copy)]
pub enum Family {
    DeepSeek,
    GlmFlash,
    Glm,
    Qwen,
}
fn invalid(s: &str) -> anyhow::Error {
    anyhow::anyhow!("{s}")
}
pub fn normalize_native_instruction_roles(
    body: &mut Map<String, Value>,
    family: Family,
) -> Result<(), anyhow::Error> {
    if matches!(family, Family::DeepSeek | Family::GlmFlash) {
        return Ok(());
    }
    let messages = body
        .get_mut("messages")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| invalid("messages must be an array"))?;
    let leading = messages
        .iter()
        .take_while(|message| {
            matches!(
                message.get("role").and_then(Value::as_str),
                Some("system" | "developer")
            )
        })
        .count();
    for message in &mut messages[leading..] {
        if message.get("role").and_then(Value::as_str) == Some("developer") {
            let text = instruction_text(message.get("content"))?;
            message["role"] = json!("system");
            message["content"] = json!(format!(
                "[Instruction role: developer]\nSystem instructions take precedence over this developer instruction.\n\n{text}"
            ));
        }
    }
    if !messages[..leading]
        .iter()
        .any(|message| message.get("role").and_then(Value::as_str) == Some("developer"))
    {
        return Ok(());
    }
    if leading == 1 {
        messages[0]["role"] = json!("system");
        return Ok(());
    }
    let mut content = String::from(
        "The following instruction messages retain their original order and priority. System instructions take precedence over developer instructions.\n\n",
    );
    for (index, message) in messages[..leading].iter().enumerate() {
        let message = message
            .as_object()
            .ok_or_else(|| invalid("instruction message must be an object"))?;
        if message
            .keys()
            .any(|key| !["role", "content"].contains(&key.as_str()))
        {
            return Err(invalid(
                "leading instruction metadata cannot be merged without losing its semantics",
            ));
        }
        let role = message.get("role").and_then(Value::as_str).unwrap();
        let text = instruction_text(message.get("content"))?;
        content.push_str(&format!("[Instruction {}: {role}]\n", index + 1));
        content.push_str(&text);
        content.push_str("\n\n");
    }
    messages.splice(..leading, [json!({"role":"system","content":content})]);
    Ok(())
}

fn instruction_text(content: Option<&Value>) -> Result<String, anyhow::Error> {
    match content {
        Some(Value::String(text)) => Ok(text.clone()),
        Some(Value::Array(blocks)) => {
            let mut text = String::new();
            for block in blocks {
                let object = block
                    .as_object()
                    .ok_or_else(|| invalid("instruction blocks must be text objects"))?;
                if object
                    .keys()
                    .any(|key| !["type", "text"].contains(&key.as_str()))
                    || object.get("type").and_then(Value::as_str) != Some("text")
                {
                    return Err(invalid(
                        "instruction block semantics cannot be merged without loss",
                    ));
                }
                text.push_str(
                    object
                        .get("text")
                        .and_then(Value::as_str)
                        .ok_or_else(|| invalid("instruction text must be a string"))?,
                );
            }
            Ok(text)
        }
        _ => Err(invalid("merged instruction messages require text content")),
    }
}
