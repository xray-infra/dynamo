//! Native function-tool identities, selection policy and opt-in strict schemas.
//!
//! Schemas are compiled before tokenization/generation. External references are
//! deliberately unavailable; only references inside the supplied schema work.

use std::collections::{BTreeMap, BTreeSet};

use jsonschema::{PatternOptions, Validator};
use serde_json::Value;

const MAX_BUFFERED_TOOL_BYTES: usize = 8 * 1024 * 1024;

pub struct StrictToolSchemas {
    declared: BTreeSet<String>,
    strict: BTreeMap<String, Validator>,
    choice: ToolChoice,
    parallel: bool,
}

enum ToolChoice {
    Auto,
    None,
    Required,
    Named(String),
}

impl StrictToolSchemas {
    pub fn compile(body: &Value) -> Result<Self, &'static str> {
        let mut declared = BTreeSet::new();
        let mut strict = BTreeMap::new();
        if let Some(tools) = body.get("tools").and_then(Value::as_array) {
            for tool in tools {
                let function = tool.get("function").ok_or("tool function is missing")?;
                let name = function
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|name| !name.is_empty())
                    .ok_or("tool function name is missing")?;
                if !declared.insert(name.to_owned()) {
                    return Err("duplicate tool function name");
                }
                if function.get("strict") == Some(&Value::Bool(true)) {
                    let schema = function
                        .get("parameters")
                        .ok_or("strict tool requires parameters JSON Schema")?;
                    // The existing request contract checks the meta-schema. The
                    // real instance validator is separate and never fetches a
                    // user-controlled URL or local file.
                    let validator = jsonschema::options()
                        .offline()
                        .with_pattern_options(PatternOptions::regex())
                        .build(schema)
                        .map_err(|_| "strict tool JSON Schema cannot be compiled locally")?;
                    strict.insert(name.to_owned(), validator);
                }
            }
        }
        let choice = match body.get("tool_choice") {
            None => ToolChoice::Auto,
            Some(Value::String(mode)) if mode == "auto" => ToolChoice::Auto,
            Some(Value::String(mode)) if mode == "none" => ToolChoice::None,
            Some(Value::String(mode)) if mode == "required" && !declared.is_empty() => {
                ToolChoice::Required
            }
            Some(value) if value.get("type").and_then(Value::as_str) == Some("function") => {
                let name = value
                    .pointer("/function/name")
                    .and_then(Value::as_str)
                    .filter(|name| declared.contains(*name))
                    .ok_or("tool_choice must select a declared function")?;
                ToolChoice::Named(name.into())
            }
            _ => return Err("invalid tool_choice"),
        };
        let parallel = body
            .get("parallel_tool_calls")
            .map(|value| value.as_bool().ok_or("parallel_tool_calls must be boolean"))
            .transpose()?
            .unwrap_or(true);
        Ok(Self {
            declared,
            strict,
            choice,
            parallel,
        })
    }

    fn has_tool_contract(&self) -> bool {
        !self.declared.is_empty() || !matches!(&self.choice, ToolChoice::Auto) || !self.parallel
    }

    fn validate_finish(&self, reason: &str, names: &[&str]) -> Result<(), &'static str> {
        if !self.parallel && names.len() > 1 {
            return Err("native parallel tool calls exceed parallel_tool_calls: false");
        }
        match &self.choice {
            ToolChoice::None if !names.is_empty() => {
                return Err("native tool call violates tool_choice: none");
            }
            ToolChoice::Named(name) if names.iter().any(|called| *called != name.as_str()) => {
                return Err("native tool call violates named tool_choice");
            }
            ToolChoice::Required | ToolChoice::Named(_)
                if names.is_empty() && reason != "length" =>
            {
                return Err("native successful finish omitted a required tool call");
            }
            _ => {}
        }
        if reason == "tool_calls" && names.is_empty() {
            return Err("native tool finish has no tool calls");
        }
        if !names.is_empty() && !matches!(reason, "tool_calls" | "function_call" | "length") {
            return Err("native tool calls have an invalid finish reason");
        }
        Ok(())
    }

    fn validate_call(&self, name: &str, arguments: &str) -> Result<(), &'static str> {
        if !self.declared.contains(name) {
            return Err("native tool call names an undeclared function");
        }
        if let Some(validator) = self.strict.get(name) {
            let parsed: Value = serde_json::from_str(arguments)
                .map_err(|_| "native strict tool arguments are incomplete JSON")?;
            if !parsed.is_object() {
                return Err("native strict tool arguments must be a JSON object");
            }
            if !validator.is_valid(&parsed) {
                return Err("native strict tool arguments violate their JSON Schema");
            }
        }
        Ok(())
    }

    pub fn validate_json(&self, value: &Value) -> Result<(), &'static str> {
        for choice in value
            .get("choices")
            .and_then(Value::as_array)
            .ok_or("native response choices are missing")?
        {
            let message = choice
                .get("message")
                .ok_or("native response message is missing")?;
            let reason = choice
                .get("finish_reason")
                .and_then(Value::as_str)
                .ok_or("native response finish reason is missing")?;
            let calls = message
                .get("tool_calls")
                .filter(|value| !value.is_null())
                .map(|value| value.as_array().ok_or("native tool calls must be an array"))
                .transpose()?;
            let mut names = Vec::new();
            let mut ids = BTreeSet::new();
            if let Some(calls) = calls {
                for call in calls {
                    let function = call
                        .get("function")
                        .ok_or("native tool function is missing")?;
                    let name = function
                        .get("name")
                        .and_then(Value::as_str)
                        .filter(|name| !name.is_empty())
                        .ok_or("native tool function name is missing")?;
                    if !self.declared.contains(name) {
                        return Err("native tool call names an undeclared function");
                    }
                    names.push(name);
                    let id = call
                        .get("id")
                        .and_then(Value::as_str)
                        .filter(|id| !id.is_empty())
                        .ok_or("native tool call ID is missing")?;
                    if !ids.insert(id) {
                        return Err("native tool call IDs must be unique");
                    }
                    let arguments = function
                        .get("arguments")
                        .and_then(Value::as_str)
                        .ok_or("native tool arguments are missing")?;
                    // A native budget limit reports an unfinished call. Its
                    // prefix is not a successful schema-conforming tool result.
                    if reason != "length" {
                        self.validate_call(name, arguments)?;
                    }
                }
            }
            self.validate_finish(reason, &names)?;
        }
        Ok(())
    }
}

#[derive(Default)]
struct PendingCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
    unknown_frames: Vec<Value>,
}

/// Streams declared tool argument fragments while retaining their complete
/// arguments for final validation. Unknown names, terminal markers, and usage
/// wait until their binding or final validation is complete.
pub struct StrictStreamBuffer {
    schemas: StrictToolSchemas,
    calls: BTreeMap<(u64, u64), PendingCall>,
    terminal_reasons: BTreeMap<u64, String>,
    terminal_frames: Vec<Value>,
    usage_frames: Vec<Value>,
    buffered_bytes: usize,
}

impl StrictStreamBuffer {
    pub fn new(schemas: StrictToolSchemas) -> Self {
        Self {
            schemas,
            calls: BTreeMap::new(),
            terminal_reasons: BTreeMap::new(),
            terminal_frames: Vec::new(),
            usage_frames: Vec::new(),
            buffered_bytes: 0,
        }
    }

    fn buffer_terminal(&mut self, frame: Value) -> Result<(), &'static str> {
        self.buffered_bytes = self
            .buffered_bytes
            .checked_add(frame.to_string().len())
            .ok_or("native strict tool buffer exceeded limit")?;
        if self.buffered_bytes > MAX_BUFFERED_TOOL_BYTES {
            return Err("native strict tool buffer exceeded limit");
        }
        self.terminal_frames.push(frame);
        Ok(())
    }

    pub fn push(&mut self, value: Value) -> Result<Vec<Value>, &'static str> {
        if !self.schemas.has_tool_contract() {
            return Ok(vec![value]);
        }
        let Some(choices) = value.get("choices").and_then(Value::as_array) else {
            return Err("native strict stream choices are missing");
        };
        let mut immediate = value.clone();
        if value.get("usage").is_some_and(|usage| !usage.is_null()) {
            let mut usage_only = value.clone();
            usage_only["choices"] = Value::Array(Vec::new());
            self.buffered_bytes = self
                .buffered_bytes
                .checked_add(usage_only.to_string().len())
                .ok_or("native strict tool buffer exceeded limit")?;
            if self.buffered_bytes > MAX_BUFFERED_TOOL_BYTES {
                return Err("native strict tool buffer exceeded limit");
            }
            self.usage_frames.push(usage_only);
            if let Some(object) = immediate.as_object_mut() {
                object.remove("usage");
            }
        }
        let mut immediate_choices = Vec::new();
        let mut immediate_tools = Vec::new();
        for choice in choices {
            let choice_index = choice
                .get("index")
                .and_then(Value::as_u64)
                .ok_or("native strict stream choice index is missing")?;
            let tool_calls = choice
                .get("delta")
                .and_then(|delta| delta.get("tool_calls"))
                .and_then(Value::as_array)
                .filter(|calls| !calls.is_empty());
            if let Some(tool_calls) = tool_calls {
                for call in tool_calls {
                    let index = call
                        .get("index")
                        .and_then(Value::as_u64)
                        .ok_or("native tool delta index is missing")?;
                    let pending = self.calls.entry((choice_index, index)).or_default();
                    if let Some(id) = call.get("id").filter(|v| !v.is_null()) {
                        let id = id
                            .as_str()
                            .filter(|id| !id.is_empty())
                            .ok_or("native tool delta ID is invalid")?;
                        if pending.id.as_deref().is_some_and(|old| old != id) {
                            return Err("native tool delta ID changed");
                        }
                        pending.id = Some(id.to_owned());
                    }
                    if let Some(function) = call.get("function") {
                        if let Some(name) = function.get("name").filter(|v| !v.is_null()) {
                            let name = name
                                .as_str()
                                .filter(|name| !name.is_empty())
                                .ok_or("native tool delta name is invalid")?;
                            if pending.name.as_deref().is_some_and(|old| old != name) {
                                return Err("native tool delta name changed");
                            }
                            pending.name = Some(name.to_owned());
                        }
                        if let Some(args) = function.get("arguments").filter(|v| !v.is_null()) {
                            let args = args
                                .as_str()
                                .ok_or("native tool delta arguments are invalid")?;
                            pending.arguments.push_str(args);
                            self.buffered_bytes = self
                                .buffered_bytes
                                .checked_add(args.len())
                                .ok_or("native strict tool buffer exceeded limit")?;
                            if self.buffered_bytes > MAX_BUFFERED_TOOL_BYTES {
                                return Err("native strict tool buffer exceeded limit");
                            }
                        }
                    }
                    let name = pending.name.as_deref();
                    if name.is_some_and(|name| !self.schemas.declared.contains(name)) {
                        return Err("native tool call names an undeclared function");
                    }
                    let mut tool_only = value.clone();
                    tool_only["choices"] = Value::Array(vec![choice.clone()]);
                    tool_only["choices"][0]["delta"] = serde_json::json!({"tool_calls": [call]});
                    tool_only["choices"][0]["finish_reason"] = Value::Null;
                    if let Some(object) = tool_only.as_object_mut() {
                        object.remove("usage");
                    }
                    if name.is_some() {
                        let unknown = std::mem::take(&mut pending.unknown_frames);
                        for frame in &unknown {
                            self.buffered_bytes -= frame.to_string().len();
                        }
                        immediate_tools.extend(unknown);
                        immediate_tools.push(tool_only);
                    } else {
                        self.buffered_bytes = self
                            .buffered_bytes
                            .checked_add(tool_only.to_string().len())
                            .ok_or("native strict tool buffer exceeded limit")?;
                        if self.buffered_bytes > MAX_BUFFERED_TOOL_BYTES {
                            return Err("native strict tool buffer exceeded limit");
                        }
                        pending.unknown_frames.push(tool_only);
                    }
                }
            }
            if choice
                .get("finish_reason")
                .is_some_and(|reason| !reason.is_null())
            {
                let reason = choice["finish_reason"]
                    .as_str()
                    .ok_or("native strict stream finish reason is invalid")?;
                if self
                    .terminal_reasons
                    .insert(choice_index, reason.to_owned())
                    .is_some()
                {
                    return Err("native strict stream has repeated finish reason");
                }
                let mut terminal = value.clone();
                terminal["choices"] = Value::Array(vec![choice.clone()]);
                terminal["choices"][0]["delta"] = serde_json::json!({});
                if let Some(object) = terminal.as_object_mut() {
                    object.remove("usage");
                }
                self.buffer_terminal(terminal)?;
            }
            let mut outgoing = choice.clone();
            if let Some(delta) = outgoing.get_mut("delta").and_then(Value::as_object_mut) {
                delta.remove("tool_calls");
            }
            outgoing["finish_reason"] = Value::Null;
            let has_delta = outgoing
                .get("delta")
                .and_then(Value::as_object)
                .is_some_and(|delta| delta.values().any(|v| !v.is_null()));
            if has_delta {
                immediate_choices.push(outgoing);
            }
        }
        immediate["choices"] = Value::Array(immediate_choices);
        let mut frames = Vec::new();
        if !immediate["choices"].as_array().is_some_and(Vec::is_empty) {
            frames.push(immediate);
        }
        frames.extend(immediate_tools);
        Ok(frames)
    }

    pub fn finish(&mut self) -> Result<Vec<Value>, &'static str> {
        if !self.schemas.has_tool_contract() {
            return Ok(Vec::new());
        }
        let mut choices: BTreeMap<u64, Vec<&str>> = BTreeMap::new();
        let mut ids: BTreeMap<u64, BTreeSet<&str>> = BTreeMap::new();
        for ((choice_index, _), pending) in &self.calls {
            let name = pending
                .name
                .as_deref()
                .ok_or("native tool function name is missing")?;
            if !self.schemas.declared.contains(name) {
                return Err("native tool call names an undeclared function");
            }
            choices.entry(*choice_index).or_default().push(name);
            if !self.terminal_reasons.contains_key(choice_index) {
                return Err("native strict tool call has no terminal finish reason");
            }
            let _id = pending
                .id
                .as_deref()
                .ok_or("native strict tool call ID is missing")?;
            if !ids.entry(*choice_index).or_default().insert(_id) {
                return Err("native tool call IDs must be unique");
            }
            if self.terminal_reasons.get(choice_index).map(String::as_str) != Some("length") {
                self.schemas.validate_call(name, &pending.arguments)?;
            }
        }
        for (index, reason) in &self.terminal_reasons {
            self.schemas
                .validate_finish(reason, choices.get(index).map(Vec::as_slice).unwrap_or(&[]))?;
        }
        let mut frames = std::mem::take(&mut self.terminal_frames);
        frames.extend(std::mem::take(&mut self.usage_frames));
        Ok(frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request() -> Value {
        json!({"tools":[{"type":"function","function":{"name":"lookup","strict":true,
            "parameters":{"type":"object","properties":{"q":{"type":"string"}},
            "required":["q"],"additionalProperties":false}}}]})
    }

    #[test]
    fn strict_budget_length_preserves_unfinished_calls_only() {
        let schemas = StrictToolSchemas::compile(&request()).unwrap();
        let reply = |reason: &str| {
            json!({"choices":[{
                "finish_reason":reason,"message":{"tool_calls":[{
                    "id":"call_1","function":{"name":"lookup","arguments":"{\"q\":"}
                }]}
            }]})
        };
        assert!(schemas.validate_json(&reply("length")).is_ok());
        assert!(schemas.validate_json(&reply("tool_calls")).is_err());
        assert!(schemas.validate_json(&reply("stop")).is_err());
        let mut unknown = reply("length");
        unknown["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = json!("foreign");
        assert!(schemas.validate_json(&unknown).is_err());

        for reason in ["length", "tool_calls", "stop"] {
            let mut stream =
                StrictStreamBuffer::new(StrictToolSchemas::compile(&request()).unwrap());
            let prefix = stream
                .push(json!({"choices":[{"index":0,"delta":{"tool_calls":[{
                "index":0,"id":"call_1","function":{"name":"lookup","arguments":"{\"q\":"}
            }]},"finish_reason":null}]}))
                .unwrap();
            assert_eq!(prefix.len(), 1);
            assert!(prefix[0]["choices"][0]["finish_reason"].is_null());
            assert!(
                stream
                    .push(
                        json!({"choices":[{"index":0,"delta":{},"finish_reason":reason}],
                "usage":{"completion_tokens":1}})
                    )
                    .unwrap()
                    .is_empty()
            );
            if reason == "length" {
                let terminal = stream.finish().unwrap();
                assert_eq!(terminal[0]["choices"][0]["finish_reason"], "length");
                assert_eq!(terminal[1]["usage"]["completion_tokens"], 1);
            } else {
                assert!(stream.finish().is_err());
            }
        }
        // A limited choice cannot excuse a different successful choice's
        // malformed tool arguments.
        let mut mixed = StrictStreamBuffer::new(StrictToolSchemas::compile(&request()).unwrap());
        mixed.push(json!({"choices":[
            {"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1",
                "function":{"name":"lookup","arguments":"{\"q\":"}}]},"finish_reason":"length"},
            {"index":1,"delta":{"tool_calls":[{"index":0,"id":"call_2",
                "function":{"name":"lookup","arguments":"{\"q\":3}"}}]},"finish_reason":"tool_calls"}
        ]})).unwrap();
        assert!(mixed.finish().is_err());
    }

    #[test]
    fn strict_json_validates_instances_and_rejects_external_refs() {
        let schemas = StrictToolSchemas::compile(&request()).unwrap();
        let reply = |args: &str| {
            json!({"choices":[{"finish_reason":"tool_calls",
            "message":{"tool_calls":[{"id":"call_1","function":{"name":"lookup","arguments":args}}]}}]})
        };
        assert!(schemas.validate_json(&reply(r#"{"q":"ok"}"#)).is_ok());
        assert!(schemas.validate_json(&reply(r#"{"q":3}"#)).is_err());
        assert!(
            schemas
                .validate_json(&reply(r#"{"q":"ok","extra":1}"#))
                .is_err()
        );
        assert!(schemas.validate_json(&reply(r#"{"q":"#)).is_err());
        assert!(schemas.validate_json(&reply("[]")).is_err());
        assert!(schemas.validate_json(&reply("{}")).is_err());
        let mut truncated = reply(r#"{"q":"ok"}"#);
        truncated["choices"][0]["finish_reason"] = Value::String("length".into());
        assert!(schemas.validate_json(&truncated).is_ok());
        let mut no_args = reply(r#"{"q":"ok"}"#);
        no_args["choices"][0]["message"]["tool_calls"][0]["function"]
            .as_object_mut()
            .unwrap()
            .remove("arguments");
        assert!(schemas.validate_json(&no_args).is_err());
        let mut remote = request();
        remote["tools"][0]["function"]["parameters"] =
            json!({"$ref":"https://example.invalid/schema.json"});
        assert!(StrictToolSchemas::compile(&remote).is_err());
        let mut missing = request();
        missing["tools"][0]["function"]
            .as_object_mut()
            .unwrap()
            .remove("parameters");
        assert!(StrictToolSchemas::compile(&missing).is_err());
    }

    #[test]
    fn strict_stream_emits_fragments_before_validated_terminal() {
        let mut guard = StrictStreamBuffer::new(StrictToolSchemas::compile(&request()).unwrap());
        let first = json!({"choices":[{"index":0,"delta":{"content":"hello",
            "tool_calls":[{"index":0,"id":"call_1","function":{"name":"lookup","arguments":"{\"q\":"}}]},
            "finish_reason":null}]});
        let emitted = guard.push(first).unwrap();
        assert_eq!(emitted.len(), 2);
        assert_eq!(emitted[0]["choices"][0]["delta"]["content"], "hello");
        assert!(
            emitted[0]["choices"][0]["delta"]
                .get("tool_calls")
                .is_none()
        );
        assert_eq!(
            emitted[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
            "{\"q\":"
        );
        assert!(
            emitted
                .iter()
                .all(|frame| frame["choices"][0]["finish_reason"].is_null())
        );
        let next = guard
            .push(
                json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,
            "function":{"arguments":"\"ok\"}"}}]},"finish_reason":null}]}),
            )
            .unwrap();
        assert_eq!(next.len(), 1);
        assert_eq!(
            next[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
            "\"ok\"}"
        );
        assert!(
            guard
                .push(json!({"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}))
                .unwrap()
                .is_empty()
        );
        let final_frames = guard.finish().unwrap();
        assert_eq!(final_frames.len(), 1);
        assert_eq!(final_frames[0]["choices"][0]["finish_reason"], "tool_calls");

        // Prefixes are observable before completion; invalid JSON or schema
        // can never release a completion marker or usage.
        for arguments in [r#"{"q":3}"#, r#"{"q":"#] {
            let mut invalid =
                StrictStreamBuffer::new(StrictToolSchemas::compile(&request()).unwrap());
            let prefix = invalid
                .push(json!({"choices":[{"index":0,"delta":{"tool_calls":[
                {"index":0,"id":"call_1","function":{"name":"lookup","arguments":arguments}}
            ]},"finish_reason":"tool_calls"}],"usage":{"completion_tokens":2}}))
                .unwrap();
            assert_eq!(prefix.len(), 1);
            assert!(prefix[0]["choices"][0]["finish_reason"].is_null());
            assert!(prefix[0].get("usage").is_none());
            assert!(invalid.finish().is_err());
        }

        let mut unknown = StrictStreamBuffer::new(StrictToolSchemas::compile(&request()).unwrap());
        assert!(
            unknown
                .push(json!({"choices":[{"index":0,"delta":{"tool_calls":[
            {"index":0,"id":"call_1","function":{"arguments":"{\"q\":"}}
        ]},"finish_reason":null}]}))
                .unwrap()
                .is_empty()
        );
        let bound = unknown
            .push(json!({"choices":[{"index":0,"delta":{"tool_calls":[
            {"index":0,"function":{"name":"lookup","arguments":"\"ok\"}"}}
        ]},"finish_reason":null}]}))
            .unwrap();
        assert_eq!(bound.len(), 2);
        assert_eq!(
            bound[0]["choices"][0]["delta"]["tool_calls"][0]["id"],
            "call_1"
        );
        assert!(unknown.finish().is_err()); // No terminal marker yet.
    }

    #[test]
    fn strict_stream_schema_violation_never_releases_terminal() {
        let mut guard = StrictStreamBuffer::new(StrictToolSchemas::compile(&request()).unwrap());
        assert!(
            guard
                .push(
                    json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,
            "id":"call_1","function":{"name":"lookup","arguments":"{\"q\":3}"}}]},
            "finish_reason":"tool_calls"}]})
                )
                .unwrap()
                .len()
                == 1
        );
        assert!(guard.finish().is_err());
    }

    #[test]
    fn strict_stream_multiple_tools_and_truncation() {
        let mut guard = StrictStreamBuffer::new(StrictToolSchemas::compile(&request()).unwrap());
        let chunk = json!({"choices":[{"index":0,"delta":{"tool_calls":[
            {"index":0,"id":"call_1","function":{"name":"lookup","arguments":"{\"q\":\"one\"}"}},
            {"index":1,"id":"call_2","function":{"name":"lookup","arguments":"{\"q\":\"two\"}"}}
        ]},"finish_reason":null}]});
        let initial = guard.push(chunk.clone()).unwrap();
        assert_eq!(initial.len(), 2);
        assert_eq!(
            initial[0]["choices"][0]["delta"]["tool_calls"][0]["id"],
            "call_1"
        );
        assert_eq!(
            initial[1]["choices"][0]["delta"]["tool_calls"][0]["id"],
            "call_2"
        );
        assert!(
            guard
                .push(json!({"choices":[{"index":0,"delta":{},
            "finish_reason":"tool_calls"}]}))
                .unwrap()
                .is_empty()
        );
        let released = guard.finish().unwrap();
        assert_eq!(released.len(), 1);
        assert_eq!(released[0]["choices"][0]["finish_reason"], "tool_calls");

        let mut truncated =
            StrictStreamBuffer::new(StrictToolSchemas::compile(&request()).unwrap());
        truncated.push(chunk).unwrap();
        truncated
            .push(json!({"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}))
            .unwrap();
        let released = truncated.finish().unwrap();
        assert_eq!(released.len(), 1);
        assert_eq!(released[0]["choices"][0]["finish_reason"], "length");

        let mut incomplete =
            StrictStreamBuffer::new(StrictToolSchemas::compile(&request()).unwrap());
        incomplete
            .push(json!({"choices":[{"index":0,"delta":{"tool_calls":[
            {"index":0,"id":"call_1","function":{"name":"lookup","arguments":"{\"q\":"}}]},
            "finish_reason":"length"}]}))
            .unwrap();
        let terminal = incomplete.finish().unwrap();
        assert_eq!(terminal[0]["choices"][0]["finish_reason"], "length");

        let mut missing = StrictStreamBuffer::new(StrictToolSchemas::compile(&request()).unwrap());
        missing
            .push(json!({"choices":[{"index":0,"delta":{"tool_calls":[
            {"index":0,"id":"call_1","function":{"name":"lookup"}}]},
            "finish_reason":"tool_calls"}]}))
            .unwrap();
        assert!(missing.finish().is_err());
    }

    #[test]
    fn mixed_nonstrict_partial_arguments_keep_native_length_semantics() {
        let mut body = request();
        body["tools"]
            .as_array_mut()
            .unwrap()
            .push(json!({"type":"function",
            "function":{"name":"freeform","strict":false,"parameters":{"type":"object"}}}));
        let schemas = StrictToolSchemas::compile(&body).unwrap();
        let partial = json!({"choices":[{"finish_reason":"length","message":{"tool_calls":[
            {"id":"call_2","function":{"name":"freeform","arguments":"{\"q\":"}}]}}]});
        assert!(schemas.validate_json(&partial).is_ok());
        let mut stream = StrictStreamBuffer::new(StrictToolSchemas::compile(&body).unwrap());
        stream
            .push(json!({"choices":[{"index":0,"delta":{"tool_calls":[
            {"index":0,"id":"call_2","function":{"name":"freeform","arguments":"{\"q\":"}}]},
            "finish_reason":"length"}]}))
            .unwrap();
        assert!(stream.finish().is_ok());
    }

    #[test]
    fn nonstrict_only_leaves_native_partial_arguments_untouched() {
        let schemas = StrictToolSchemas::compile(&serde_json::json!({"tools":[{"type":"function",
            "function":{"name":"freeform","strict":false}}]}))
        .unwrap();
        assert!(schemas
            .validate_json(&serde_json::json!({"choices":[{"finish_reason":"length",
            "message":{"tool_calls":[{"id":"call_1","function":{"name":"freeform","arguments":"{\"q\":"}}]}}]}))
            .is_ok());
    }

    #[test]
    fn ordinary_tool_policy_identity_and_best_effort_arguments() {
        let mut body = request();
        body["tools"][0]["function"]
            .as_object_mut()
            .unwrap()
            .remove("strict");
        let reply = |calls: Value, reason: &str| {
            json!({"choices":[{"finish_reason":reason,
            "message":{"tool_calls":calls}}]})
        };
        let call = json!({"id":"call_1","function":{"name":"lookup","arguments":"{"}});
        let schemas = StrictToolSchemas::compile(&body).unwrap();
        assert!(
            schemas
                .validate_json(&reply(json!([call.clone()]), "tool_calls"))
                .is_ok()
        );
        for field in ["id", "function"] {
            let mut invalid = call.clone();
            invalid.as_object_mut().unwrap().remove(field);
            assert!(
                schemas
                    .validate_json(&reply(json!([invalid]), "tool_calls"))
                    .is_err()
            );
        }
        assert!(
            schemas
                .validate_json(&reply(json!([call.clone(), call.clone()]), "tool_calls"))
                .is_err()
        );
        body["tool_choice"] = json!("none");
        assert!(
            StrictToolSchemas::compile(&body)
                .unwrap()
                .validate_json(&reply(json!([call.clone()]), "tool_calls"))
                .is_err()
        );
        body["tool_choice"] = json!("required");
        let required = StrictToolSchemas::compile(&body).unwrap();
        assert!(required.validate_json(&reply(json!([]), "stop")).is_err());
        assert!(required.validate_json(&reply(json!([]), "length")).is_ok());
        body["tool_choice"] = json!({"type":"function","function":{"name":"lookup"}});
        body["parallel_tool_calls"] = json!(false);
        let mut second = call.clone();
        second["id"] = json!("call_2");
        assert!(
            StrictToolSchemas::compile(&body)
                .unwrap()
                .validate_json(&reply(json!([call, second]), "tool_calls"))
                .is_err()
        );
        body["parallel_tool_calls"] = json!(true);
        body["tools"]
            .as_array_mut()
            .unwrap()
            .push(json!({"type":"function",
            "function":{"name":"second","strict":false,"parameters":{"type":"object"}}}));
        let wrong_name = json!({"id":"call_other","function":{"name":"second","arguments":"{}"}});
        assert!(
            StrictToolSchemas::compile(&body)
                .unwrap()
                .validate_json(&reply(json!([wrong_name]), "tool_calls"))
                .is_err()
        );
        body["tool_choice"]["function"]["name"] = json!("foreign");
        assert!(StrictToolSchemas::compile(&body).is_err());
    }

    #[test]
    fn ordinary_stream_policy_and_stable_ids_gate_success() {
        let mut body = request();
        body["tools"][0]["function"]["strict"] = json!(false);
        body["parallel_tool_calls"] = json!(false);
        let mut stream = StrictStreamBuffer::new(StrictToolSchemas::compile(&body).unwrap());
        stream.push(json!({"choices":[{"index":0,"delta":{"tool_calls":[
            {"index":0,"id":"a","function":{"name":"lookup","arguments":"{}"}},
            {"index":1,"id":"b","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":"tool_calls"}]})).unwrap();
        assert!(stream.finish().is_err());
        let mut stream = StrictStreamBuffer::new(StrictToolSchemas::compile(&body).unwrap());
        stream.push(json!({"choices":[{"index":0,"delta":{"tool_calls":[
            {"index":0,"id":"a","function":{"name":"lookup","arguments":"{"}}]},"finish_reason":null}]})).unwrap();
        assert!(
            stream
                .push(json!({"choices":[{"index":0,"delta":{"tool_calls":[
            {"index":0,"id":"b","function":{"arguments":"}"}}]},"finish_reason":null}]}))
                .is_err()
        );
        body["tool_choice"] = json!("required");
        let mut stream = StrictStreamBuffer::new(StrictToolSchemas::compile(&body).unwrap());
        stream
            .push(
                json!({"choices":[{"index":0,"delta":{"content":"text"},"finish_reason":"stop"}]}),
            )
            .unwrap();
        assert!(stream.finish().is_err());
    }

    #[test]
    fn strict_numeric_decimal_multiple_and_bounds_factory() {
        for (multiple, valid, invalid) in [
            (json!(0.5), json!(3.5), json!(3.25)),
            (json!(0.1), json!(0.3), json!(0.31)),
            (json!(0.1), json!(-0.3), json!(-0.31)),
        ] {
            let mut body = request();
            body["tools"][0]["function"]["parameters"] = json!({"type":"object",
                "properties":{"n":{"type":"number","multipleOf":multiple,"minimum":-4,"exclusiveMaximum":4}},
                "required":["n"],"additionalProperties":false});
            crate::native_contract::validate_chat_strict_tools(&body).unwrap();
            let schemas = StrictToolSchemas::compile(&body).unwrap();
            let reply = |n: Value| {
                json!({"choices":[{"finish_reason":"tool_calls","message":{"tool_calls":[
                {"id":"call_numeric","function":{"name":"lookup","arguments":json!({"n":n}).to_string()}}]}}]})
            };
            assert!(
                schemas.validate_json(&reply(valid.clone())).is_ok(),
                "valid {valid} multiple {multiple}"
            );
            assert!(
                schemas.validate_json(&reply(invalid.clone())).is_err(),
                "invalid {invalid} multiple {multiple}"
            );
            assert!(schemas.validate_json(&reply(json!(4))).is_err());
            assert!(schemas.validate_json(&reply(json!(-4.5))).is_err());
        }
        // Parse the exact public JSON spelling. Converting these operands to
        // f64 would erase the one-unit boundary and the narrow decimal window.
        for (property, values) in [
            (
                json!({"type":"integer","maximum":1e30}),
                vec![
                    ("1000000000000000000000000000000", true),
                    ("1000000000000000000000000000001", false),
                ],
            ),
            (
                json!({"type":"number","minimum":1e-8,"maximum":1.00000001e-8}),
                vec![
                    ("1e-8", true),
                    ("1.000000005e-8", true),
                    ("1.00000001e-8", true),
                    ("0.99999999e-8", false),
                    ("1.00000002e-8", false),
                ],
            ),
        ] {
            let mut body = request();
            body["tools"][0]["function"]["parameters"] = json!({"type":"object",
                "properties":{"n":property},"required":["n"],"additionalProperties":false});
            crate::native_contract::validate_chat_strict_tools(&body).unwrap();
            let schemas = StrictToolSchemas::compile(&body).unwrap();
            for (spelling, expected) in values {
                let args = format!("{{\"n\":{spelling}}}");
                let parsed: Value = serde_json::from_str(&args).unwrap();
                let reply = json!({"choices":[{"finish_reason":"tool_calls","message":{"tool_calls":[
                    {"id":"call_exact","function":{"name":"lookup","arguments":parsed.to_string()}}]}}]});
                assert_eq!(
                    schemas.validate_json(&reply).is_ok(),
                    expected,
                    "exact operand {spelling}"
                );
            }
        }
        let mut body = request();
        body["model"] = json!("qwen3.8-flash-next");
        body["messages"] = json!([{"role":"user","content":"call lookup"}]);
        body["tools"][0]["function"]["parameters"] = serde_json::from_str(
            r#"{"type":"object","properties":{"n":{"type":"number","minimum":0.10000000000000000001,"maximum":0.10000000000000000001}},"required":["n"],"additionalProperties":false}"#,
        ).unwrap();
        crate::native_contract::validate_chat_strict_tools(&body).unwrap();
        let canonical = crate::native_contract::ModelContract::for_model("qwen3.8-flash-next")
            .unwrap()
            .normalize(body)
            .unwrap();
        // This is the generation transport's actual serialization and body
        // factory. Building the request sends no HTTP request.
        let transport = reqwest::Client::new()
            .post("http://127.0.0.1:1/v1/chat/completions")
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(serde_json::to_vec(&canonical).unwrap())
            .build()
            .unwrap();
        let forwarded: Value =
            serde_json::from_slice(transport.body().unwrap().as_bytes().unwrap()).unwrap();
        for field in ["minimum", "maximum"] {
            assert_eq!(
                forwarded["tools"][0]["function"]["parameters"]["properties"]["n"][field]
                    .to_string(),
                "0.10000000000000000001"
            );
        }
        let schemas = StrictToolSchemas::compile(&forwarded).unwrap();
        for (spelling, expected) in [
            ("0.10000000000000000001", true),
            ("0.1", false),
            ("0.10000000000000000002", false),
        ] {
            let reply = json!({"choices":[{"finish_reason":"tool_calls","message":{"tool_calls":[
                {"id":"call_long_decimal","function":{"name":"lookup",
                    "arguments":format!("{{\"n\":{spelling}}}")}}]}}]});
            assert_eq!(
                schemas.validate_json(&reply).is_ok(),
                expected,
                "exact long decimal {spelling}"
            );
        }
    }

    #[test]
    fn messages_strict_schema_optional_properties_survive_factory() {
        let schema = json!({"type":"object","properties":{"n":{"type":"number","multipleOf":0.5},
            "optional":{"type":"string"}},"required":["n"],"additionalProperties":false});
        let message = json!({"model":"qwen3.8-flash-next","max_tokens":64,
            "messages":[{"role":"user","content":"call lookup"}],
            "tools":[{"name":"lookup","strict":true,"input_schema":schema}],
            "tool_choice":{"type":"tool","name":"lookup","disable_parallel_tool_use":true}});
        let chat = crate::protocols::anthropic::native_messages_to_chat(&message).unwrap();
        assert_eq!(chat["tools"][0]["function"]["strict"], true);
        assert_eq!(chat["tools"][0]["function"]["parameters"], schema);
        assert_eq!(chat["parallel_tool_calls"], false);
        assert_eq!(chat["tool_choice"]["function"]["name"], "lookup");
        let normalized = crate::native_contract::ModelContract::for_model("qwen3.8-flash-next")
            .unwrap()
            .normalize(chat.clone())
            .unwrap();
        assert!(crate::native_contract::validate_chat_strict_tools(&chat).is_err());
        let schemas = StrictToolSchemas::compile(&normalized).unwrap();
        assert!(
            schemas
                .validate_json(
                    &json!({"choices":[{"finish_reason":"tool_calls","message":{"tool_calls":[
            {"id":"call_1","function":{"name":"lookup","arguments":"{\"n\":3.5}"}}]}}]})
                )
                .is_ok()
        );
    }

    #[test]
    fn mixed_nonstrict_delta_is_immediate_and_unknown_name_waits() {
        let mut body = request();
        body["tools"]
            .as_array_mut()
            .unwrap()
            .push(json!({"type":"function",
            "function":{"name":"freeform","strict":false,"parameters":{"type":"object"}}}));
        let mut stream = StrictStreamBuffer::new(StrictToolSchemas::compile(&body).unwrap());
        let early = stream
            .push(json!({"choices":[{"index":0,"delta":{"tool_calls":[
            {"index":0,"id":"call_2","function":{"arguments":"{\"q\":"}}]},
            "finish_reason":null}]}))
            .unwrap();
        assert!(early.is_empty());
        let classified = stream
            .push(json!({"choices":[{"index":0,"delta":{"tool_calls":[
            {"index":0,"function":{"name":"freeform","arguments":"1"}}]},
            "finish_reason":null}]}))
            .unwrap();
        assert_eq!(classified.len(), 2);
        assert_eq!(
            classified[0]["choices"][0]["delta"]["tool_calls"][0]["id"],
            "call_2"
        );
        assert_eq!(
            classified[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            "freeform"
        );
        stream
            .push(json!({"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}))
            .unwrap();
        let final_frames = stream.finish().unwrap();
        assert!(
            final_frames
                .iter()
                .all(|frame| frame["choices"][0]["delta"].get("tool_calls").is_none())
        );
    }
}
