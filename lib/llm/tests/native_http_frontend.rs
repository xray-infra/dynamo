// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Actual Dynamo public HTTP handlers -> direct NativeHttpEngine -> CPU fake worker.
//! No stand-in gateway, production address, tokenizer model, or GPU is involved.
use axum::{
    Json, Router,
    body::Body,
    extract::State,
    response::Response,
    routing::{get, post as axum_post},
};
use dynamo_llm::{
    engines::StreamingEngineAdapter,
    http::service::service_v2::HttpService,
    model_card::ModelDeploymentCard,
    native_contract::{
        ForcedZeroThinkingField, ForcedZeroThinkingQualification, NativeTier, NativeWorkerIdentity,
        ThinkingBudgetField, ThinkingBudgetQualification,
    },
    native_http_engine::{
        NativeHttpEngine, NativeHttpEngineConfig, NativeIdFormat, NativeWorkerConfig,
    },
};
use futures::StreamExt;
use serde_json::{Value, json};
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
use tokio_util::sync::CancellationToken;

const GF: &str = "glm-5.3-flash";
const GLM: &str = "zai-org/GLM-5.3";
const QWEN: &str = "qwen3.8-flash-next";
const DS: &str = "deepseek-ai/DeepSeek-V4.1-Flash";

#[derive(Clone)]
struct FakeState {
    model: &'static str,
    count: u64,
    mode: &'static str,
    count_calls: Arc<Mutex<Vec<Value>>>,
    generation_calls: Arc<Mutex<Vec<Value>>>,
    calls: Arc<AtomicUsize>,
    load_calls: Arc<AtomicUsize>,
    native_incarnation: String,
}
struct FakeWorker {
    url: String,
    state: FakeState,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for FakeWorker {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn generation(State(s): State<FakeState>, Json(body): Json<Value>) -> Response {
    s.calls.fetch_add(1, Ordering::AcqRel);
    s.generation_calls.lock().unwrap().push(body.clone());
    let tools =
        matches!(s.mode, "tool" | "length" | "valid_length") || s.mode.starts_with("strict_");
    let args = match s.mode {
        "length" => "{\"answer\":",
        "strict_invalid" => "{\"answer\":17}",
        "strict_mixed_partial" => "{\"q\":",
        _ => "{\"answer\":\"🙂\"}",
    };
    let tool_name = match s.mode {
        "strict_undeclared" => "surprise",
        "strict_mixed_partial" => "freeform",
        _ => "answer",
    };
    let finish = if matches!(
        s.mode,
        "length" | "valid_length" | "strict_valid_length" | "strict_mixed_partial"
    ) {
        "length"
    } else if tools {
        "tool_calls"
    } else {
        "stop"
    };
    let message = if tools {
        json!({"role":"assistant","content":null,"tool_calls":[{"id":"call-native-17","type":"function","function":{"name":tool_name,"arguments":args}}]})
    } else {
        json!({"role":"assistant","content":"ok🙂"})
    };
    let usage = json!({"prompt_tokens":s.count,"completion_tokens":4,"total_tokens":s.count+4,
        "cached_tokens":17,"completion_tokens_details":{"reasoning_tokens":2},"opaque_native":{"false":false,"null":null}});
    if !body["stream"].as_bool().unwrap_or(false) {
        return Response::builder()
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "id":"native-id","model":s.model,"object":"chat.completion","created":1,
                    "choices":[{"index":0,"message":message,"finish_reason":finish}],"usage":usage
                })
                .to_string(),
            ))
            .unwrap();
    }
    let delta = if tools {
        json!({"tool_calls":[{"index":0,"id":"call-native-17","type":"function","function":{"name":tool_name,"arguments":args}}]})
    } else {
        json!({"role":"assistant","content":"ok🙂"})
    };
    let first =
        json!({"id":"native-id","model":s.model,"object":"chat.completion.chunk","created":1,
        "choices":[{"index":0,"delta":delta,"finish_reason":null}]})
        .to_string();
    let mut last = json!({"id":"native-id","model":s.model,"object":"chat.completion.chunk","created":1,
        "choices":[{"index":0,"delta":{},"finish_reason":finish}]});
    if s.mode != "usage_contract" || body["stream_options"]["include_usage"] == true {
        last["usage"] = usage;
    }
    let last = last.to_string();
    let wire = match s.mode {
        "malformed" => "data: {bad}\n\ndata: [DONE]\n\n".to_owned(),
        "partial" => format!("data: {first}\n\ndata: [DONE]\n\n"),
        "strict_fragments" | "strict_late_error" => {
            let prefix = json!({"id":"native-id","model":s.model,"object":"chat.completion.chunk","created":1,
                "choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-native-17","type":"function","function":{"name":"answer","arguments":"{\"answer\":"}}]},"finish_reason":null}]}).to_string();
            let suffix = json!({"id":"native-id","model":s.model,"object":"chat.completion.chunk","created":1,
                "choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"🙂\"}"}}]},"finish_reason":null}]}).to_string();
            let end = if s.mode == "strict_late_error" {
                "data: {\"error\":{\"message\":\"native failed after finish\"}}\n\n"
            } else {
                "data: [DONE]\n\n"
            };
            format!("data: {prefix}\n\ndata: {suffix}\n\ndata: {last}\n\n{end}")
        }
        _ => format!(
            ": heartbeat\r\n\r\ndata: {first}\r\n\r\ndata: {last}\r\n\r\ndata: [DONE]\r\n\r\n"
        ),
    };
    let mode = s.mode;
    let chunks = async_stream::stream! {
        if mode == "hold" {
            yield Ok::<_, std::io::Error>(bytes::Bytes::from(format!("data: {first}\n\n")));
            std::future::pending::<()>().await;
        } else {
            for bytes in wire.as_bytes().chunks(7) {
                yield Ok::<_, std::io::Error>(bytes::Bytes::copy_from_slice(bytes));
            }
        }
    };
    Response::builder()
        .header("content-type", "text/event-stream")
        .body(Body::from_stream(chunks))
        .unwrap()
}

#[tokio::test]
async fn public_native_stream_usage_is_requested_and_not_assumed() {
    let native = fake(GF, 100, "usage_contract").await;
    let f = frontend(GF, vec![worker(&native, NativeTier::Shared)]).await;
    for (path, body) in requests(GF, true)
        .into_iter()
        .filter(|(path, _)| *path != "/v1/chat/completions")
    {
        let reply = post(&f, path, body).await;
        assert!(reply.status().is_success());
        let wire = reply.text().await.unwrap();
        let parsed = events(&wire);
        let generated = native
            .state
            .generation_calls
            .lock()
            .unwrap()
            .last()
            .unwrap()
            .clone();
        assert_eq!(generated["stream_options"]["include_usage"], true);
        assert!(
            native
                .state
                .count_calls
                .lock()
                .unwrap()
                .last()
                .unwrap()
                .get("stream_options")
                .is_none()
        );
        if path == "/v1/responses" {
            let terminal = parsed
                .iter()
                .find(|event| event["type"] == "response.completed")
                .unwrap();
            assert_eq!(terminal["response"]["usage"]["output_tokens"], 4);
        } else {
            let terminal = parsed
                .iter()
                .find(|event| event["type"] == "message_delta")
                .unwrap();
            assert_eq!(terminal["usage"]["output_tokens"], 4);
        }
    }
}

async fn fake(model: &'static str, count: u64, mode: &'static str) -> FakeWorker {
    let state = FakeState {
        model,
        count,
        mode,
        count_calls: Default::default(),
        generation_calls: Default::default(),
        calls: Default::default(),
        load_calls: Default::default(),
        native_incarnation: uuid::Uuid::new_v4().simple().to_string(),
    };
    let app = Router::new()
        .route(
            "/tokenize",
            axum_post(
                |State(s): State<FakeState>, Json(v): Json<Value>| async move {
                    s.count_calls.lock().unwrap().push(v);
                    Json(json!({"count":s.count,"max_model_len":1048576}))
                },
            ),
        )
        .route(
            "/v1/tokenize",
            axum_post(
                |State(s): State<FakeState>, Json(v): Json<Value>| async move {
                    s.count_calls.lock().unwrap().push(v);
                    Json(json!({"count":s.count,"max_model_len":1048576}))
                },
            ),
        )
        .route(
            "/v1/loads",
            get(|State(s):State<FakeState>| async move {
                s.load_calls.fetch_add(1,Ordering::AcqRel);
                let timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64();
                let mut row = json!({"num_running_reqs":0,"num_waiting_reqs":0,"active_request_ids":[],
                    "native_incarnation":s.native_incarnation,"dp_rank":0,"timestamp":timestamp,
                    "frozen":false,"heartbeat_timestamp":0.0,"snapshot_generation":1});
                if s.mode == "load_ids_missing" { row.as_object_mut().unwrap().remove("active_request_ids"); }
                Json(json!({"loads":[row]}))
            }),
        )
        .route("/v1/chat/completions", axum_post(generation))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    FakeWorker { url, state, task }
}
fn worker(server: &FakeWorker, tier: NativeTier) -> NativeWorkerConfig {
    let mut config = NativeWorkerConfig {
        url: server.url.clone(),
        tier,
        max_inflight: Some(1),
        load_path: Some("/v1/loads?include=core".into()),
        max_input_tokens: Some(1_048_576),
        capability_source: Some("CPU fake worker fixture capability".into()),
        active_request_ids_source: Some(
            "CPU fake worker same-snapshot IDs/rank/UUID4/time fixture; not GPU qualification"
                .into(),
        ),
        native_dp_size: Some(1),
        native_id_format: if server.state.model == GF {
            NativeIdFormat::VllmChatId
        } else {
            NativeIdFormat::SglangPairedRid
        },
        identity: None,
        thinking_budget_qualification: None,
        forced_zero_qualification: None,
    };
    if server.state.model == GF {
        qualify_cap_worker(&mut config, GF);
    }
    config
}
struct Frontend {
    url: String,
    engine: Arc<NativeHttpEngine>,
    task: tokio::task::JoinHandle<()>,
    cancel: CancellationToken,
}
impl Drop for Frontend {
    fn drop(&mut self) {
        self.cancel.cancel();
        self.task.abort();
    }
}
async fn frontend(model: &str, workers: Vec<NativeWorkerConfig>) -> Frontend {
    frontend_with_token(model, workers, None).await
}
async fn frontend_with_token(
    model: &str,
    mut workers: Vec<NativeWorkerConfig>,
    token: Option<Arc<str>>,
) -> Frontend {
    if model == "deepseek-ai/DeepSeek-V4.1-Flash" {
        for worker in &mut workers {
            worker.max_inflight = None;
            worker.native_dp_size = Some(1);
        }
    }
    let engine = NativeHttpEngine::new(NativeHttpEngineConfig {
        model: model.into(),
        workers,
        qualification_tier: None,
        qualified_model_identity: if model == "deepseek-ai/DeepSeek-V4.1-Flash" {
            Some("CPU fixture model/renderer/template identity".into())
        } else {
            None
        },
    })
    .unwrap();
    frontend_from_engine(model, engine, token).await
}
async fn frontend_from_engine(
    model: &str,
    engine: NativeHttpEngine,
    token: Option<Arc<str>>,
) -> Frontend {
    let engine = Arc::new(engine);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let cancel = CancellationToken::new();
    let service = HttpService::builder()
        .host("127.0.0.1".to_owned())
        .port(port)
        .enable_chat_endpoints(true)
        .enable_cmpl_endpoints(false)
        .enable_embeddings_endpoints(false)
        .enable_responses_endpoints(true)
        .enable_anthropic_endpoints(true)
        .enable_admin_api(false)
        .native_http_engine(Some(engine.clone()))
        .native_http_bearer_token(token)
        .cancel_token(Some(cancel.clone()))
        .build()
        .unwrap();
    let card = ModelDeploymentCard::with_name_only(model);
    service
        .model_manager()
        .add_chat_completions_model(
            model,
            card.mdcsum(),
            Arc::new(StreamingEngineAdapter::new(engine.clone())),
        )
        .unwrap();
    let stop = cancel.clone();
    let task = tokio::spawn(async move {
        service.run_with_listener(stop, listener).await.unwrap();
    });
    Frontend {
        url: format!("http://127.0.0.1:{port}"),
        engine,
        task,
        cancel,
    }
}
async fn post(f: &Frontend, path: &str, body: Value) -> reqwest::Response {
    reqwest::Client::new()
        .post(format!("{}{path}", f.url))
        .json(&body)
        .send()
        .await
        .unwrap()
}
fn chat(model: &str, stream: bool) -> Value {
    json!({"model":model,"messages":[{"role":"user","content":"hello🙂"}],"max_tokens":800,"stream":stream})
}
fn responses(model: &str, stream: bool) -> Value {
    json!({"model":model,"input":"hello🙂","max_output_tokens":800,"stream":stream,"store":false})
}
fn messages(model: &str, stream: bool) -> Value {
    json!({"model":model,"messages":[{"role":"user","content":"hello🙂"}],"max_tokens":800,"stream":stream})
}
fn requests(model: &str, stream: bool) -> [(&'static str, Value); 3] {
    [
        ("/v1/chat/completions", chat(model, stream)),
        ("/v1/responses", responses(model, stream)),
        ("/v1/messages", messages(model, stream)),
    ]
}
fn events(wire: &str) -> Vec<Value> {
    wire.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect()
}

#[tokio::test]
async fn public_json_protocols_preserve_native_usage_and_content() {
    let native = fake(GF, 100, "good").await;
    let f = frontend(GF, vec![worker(&native, NativeTier::Shared)]).await;
    for (path, body) in requests(GF, false) {
        let reply = post(&f, path, body).await;
        assert!(reply.status().is_success(), "{path}: {}", reply.status());
        let value: Value = reply.json().await.unwrap();
        match path {
            "/v1/chat/completions" => {
                assert_eq!(value["choices"][0]["message"]["content"], "ok🙂");
                assert_eq!(
                    value["usage"]["opaque_native"],
                    json!({"false":false,"null":null})
                );
                assert_eq!(value["usage"]["cached_tokens"], 17);
            }
            "/v1/responses" => {
                assert_eq!(value["status"], "completed");
                assert_eq!(value["usage"]["input_tokens"], 100);
                assert_eq!(value["usage"]["output_tokens"], 4);
                assert_eq!(value["usage"]["input_tokens_details"]["cached_tokens"], 17);
                assert_eq!(value["output"][0]["content"][0]["text"], "ok🙂");
            }
            _ => {
                assert_eq!(value["content"][0]["text"], "ok🙂");
                assert_eq!(value["stop_reason"], "end_turn");
                assert_eq!(value["usage"]["cache_read_input_tokens"], 17);
                assert_eq!(value["usage"]["output_tokens"], 4);
            }
        }
        assert_eq!(f.engine.inflight(), vec![0]);
    }
    assert_eq!(native.state.calls.load(Ordering::Acquire), 3);
}

#[tokio::test]
async fn public_fragmented_sse_protocols_have_real_usage_and_terminal() {
    let native = fake(GF, 100, "good").await;
    let f = frontend(GF, vec![worker(&native, NativeTier::Shared)]).await;
    for (path, body) in requests(GF, true) {
        let reply = post(&f, path, body).await;
        assert!(reply.status().is_success());
        let wire = reply.text().await.unwrap();
        assert!(wire.contains("ok🙂"), "{path}: {wire}");
        let parsed = events(&wire);
        match path {
            "/v1/chat/completions" => {
                assert!(wire.contains("[DONE]"));
                assert!(parsed.iter().any(|e| e["usage"]["cached_tokens"] == 17));
            }
            "/v1/responses" => {
                let final_event = parsed
                    .iter()
                    .find(|e| e["type"] == "response.completed")
                    .expect(&wire);
                assert_eq!(final_event["response"]["usage"]["output_tokens"], 4);
                assert_eq!(
                    final_event["response"]["usage"]["input_tokens_details"]["cached_tokens"],
                    17
                );
            }
            _ => {
                assert!(parsed.iter().any(|e| e["type"] == "message_stop"));
                assert!(
                    parsed
                        .iter()
                        .any(|e| e["type"] == "message_delta" && e["usage"]["output_tokens"] == 4)
                );
            }
        }
        assert_eq!(f.engine.inflight(), vec![0]);
    }
}

#[tokio::test]
async fn public_chat_preserves_native_fields_and_count_generation_agreement() {
    let native = fake(GF, 100, "good").await;
    let f = frontend(GF, vec![worker(&native, NativeTier::Shared)]).await;
    let mut body = chat(GF, false);
    body.as_object_mut().unwrap().remove("max_tokens");
    body["max_completion_tokens"] = json!(800);
    body["thinking_token_budget"] = json!(32);
    body["reasoning_effort"] = json!("medium");
    body["reasoning_budget"] = json!(0);
    body["prompt_cache_key"] = json!("isolated-S1");
    body["chat_template_kwargs"] =
        json!({"native_extension":{"false":false,"null":null,"中文":[1.23456789]}});
    let reply = post(&f, "/v1/chat/completions", body.clone()).await;
    let status = reply.status();
    let wire = reply.text().await.unwrap();
    assert!(status.is_success(), "{status}: {wire}");
    let generated = native.state.generation_calls.lock().unwrap()[0].clone();
    assert_eq!(generated["thinking_token_budget"], 32);
    assert_eq!(generated["reasoning_budget"], 0);
    assert_eq!(generated["cache_salt"], "isolated-S1");
    assert_eq!(generated["max_tokens"], 800);
    assert_eq!(
        generated["chat_template_kwargs"]["native_extension"],
        body["chat_template_kwargs"]["native_extension"]
    );
    assert_eq!(
        generated["chat_template_kwargs"]["reasoning_effort"],
        "high"
    );
    let count = native.state.count_calls.lock().unwrap()[0].clone();
    let mut stripped = generated;
    for key in ["max_tokens", "stream", "stream_options"] {
        stripped.as_object_mut().unwrap().remove(key);
    }
    assert_eq!(stripped, count);
}

#[tokio::test]
async fn public_count_endpoints_use_exact_native_count_without_generation() {
    let native = fake(GF, 12345, "good").await;
    let f = frontend(GF, vec![worker(&native, NativeTier::Shared)]).await;
    for (path, body) in [
        ("/v1/responses/input_tokens", responses(GF, false)),
        ("/v1/messages/count_tokens", messages(GF, false)),
    ] {
        let reply = post(&f, path, body).await;
        assert!(reply.status().is_success(), "{path}");
        assert_eq!(reply.json::<Value>().await.unwrap()["input_tokens"], 12345);
    }
    assert_eq!(native.state.count_calls.lock().unwrap().len(), 2);
    assert_eq!(native.state.calls.load(Ordering::Acquire), 0);
}

#[tokio::test]
async fn public_model_card_reports_current_native_contract() {
    let native = fake(GF, 100, "good").await;
    let f = frontend(GF, vec![worker(&native, NativeTier::Shared)]).await;
    let reply = reqwest::get(format!("{}/v1/models", f.url)).await.unwrap();
    assert!(reply.status().is_success());
    let value: Value = reply.json().await.unwrap();
    let card = &value["data"][0];
    assert_eq!(card["id"], GF);
    assert_eq!(card["owned_by"], "zai-org");
    assert_eq!(card["context_window"], 1048576);
    assert_eq!(card["max_output_tokens"], 131072);
    assert_eq!(
        card["gateway_capabilities"]["reasoning"]["disabled_supported"],
        false
    );
    assert_eq!(
        card["gateway_capabilities"]["reasoning"]["fixed_budget_supported"],
        true
    );
    assert_eq!(
        card["gateway_capabilities"]["prompt_cache"]["ttl_guaranteed"],
        false
    );
    assert_eq!(native.state.calls.load(Ordering::Acquire), 0);
}

#[tokio::test]
async fn public_invalid_contracts_fail_before_native_count_or_generation() {
    let native = fake(GF, 100, "good").await;
    let f = frontend(GF, vec![worker(&native, NativeTier::Shared)]).await;
    let mut bodies = Vec::new();
    for (key, value) in [
        ("max_tokens", json!(true)),
        ("reasoning_effort", json!("none")),
        ("thinking_token_budget", json!(800)),
        ("prompt_cache_retention", json!("24h")),
        (
            "tools",
            json!([{"type":"function","function":{"name":"bad","parameters":{"type":17}}}]),
        ),
    ] {
        let mut body = chat(GF, false);
        body[key] = value;
        bodies.push(("/v1/chat/completions", body));
    }
    let mut conflict = chat(GF, false);
    conflict["max_completion_tokens"] = json!(801);
    bodies.push(("/v1/chat/completions", conflict));
    let mut unknown = chat(GF, false);
    unknown["unrecognized_root_extension"] = json!({"false":false,"null":null});
    bodies.push(("/v1/chat/completions", unknown));
    let mut stateful = responses(GF, false);
    stateful["previous_response_id"] = json!("resp-old");
    bodies.push(("/v1/responses", stateful));
    let mut nontext = messages(GF, false);
    nontext["messages"][0]["content"] =
        json!([{"type":"image","source":{"type":"url","url":"https://invalid.example/image"}}]);
    bodies.push(("/v1/messages", nontext));
    for (path, body) in bodies {
        let reply = post(&f, path, body.clone()).await;
        assert!(
            reply.status().is_client_error(),
            "{path} {body}: {}",
            reply.status()
        );
    }
    assert!(native.state.count_calls.lock().unwrap().is_empty());
    assert_eq!(native.state.calls.load(Ordering::Acquire), 0);
    assert_eq!(f.engine.inflight(), vec![0]);
}

#[tokio::test]
async fn public_output_budget_flips_tier_and_over_context_never_generates() {
    let short = fake(QWEN, 16000, "good").await;
    let long = fake(QWEN, 16000, "good").await;
    let f = frontend(
        QWEN,
        vec![
            worker(&short, NativeTier::Short),
            worker(&long, NativeTier::Long),
        ],
    )
    .await;
    let mut small = chat(QWEN, false);
    small["max_tokens"] = json!(384);
    small["reasoning_effort"] = json!("none");
    assert!(
        post(&f, "/v1/chat/completions", small)
            .await
            .status()
            .is_success()
    );
    let mut large = chat(QWEN, false);
    large["reasoning_effort"] = json!("none");
    assert!(
        post(&f, "/v1/chat/completions", large)
            .await
            .status()
            .is_success()
    );
    assert_eq!(short.state.calls.load(Ordering::Acquire), 1);
    assert_eq!(long.state.calls.load(Ordering::Acquire), 1);
    assert_eq!(f.engine.inflight(), vec![0, 0]);
    let over = fake(QWEN, 1048576, "good").await;
    let blocked = frontend(QWEN, vec![worker(&over, NativeTier::Long)]).await;
    assert!(
        !post(&blocked, "/v1/chat/completions", chat(QWEN, false))
            .await
            .status()
            .is_success()
    );
    assert_eq!(over.state.calls.load(Ordering::Acquire), 0);
    assert_eq!(blocked.engine.inflight(), vec![0]);
}

#[tokio::test]
async fn public_tool_ids_and_length_are_preserved_across_protocols() {
    for mode in ["tool", "length", "valid_length"] {
        let native = fake(GF, 100, mode).await;
        let f = frontend(GF, vec![worker(&native, NativeTier::Shared)]).await;
        for streaming in [false, true] {
            for (path, body) in requests(GF, streaming) {
                let reply = post(&f, path, body).await;
                assert!(reply.status().is_success(), "{path} {mode}");
                let wire = reply.text().await.unwrap();
                // Messages requires an executable JSON object for tool input;
                // neither unary nor SSE can fabricate it from truncated args.
                if !(path == "/v1/messages" && mode == "length") {
                    assert!(wire.contains("call-native-17"), "{path} {mode}: {wire}");
                } else if streaming {
                    let parsed = events(&wire);
                    assert!(
                        parsed.iter().any(|event| event["type"] == "message_delta"
                            && event["delta"]["stop_reason"] == "max_tokens"),
                        "{wire}"
                    );
                    assert!(
                        parsed.iter().any(|event| event["type"] == "message_stop"),
                        "{wire}"
                    );
                    assert!(
                        !parsed
                            .iter()
                            .any(|event| event["content_block"]["type"] == "tool_use"),
                        "{wire}"
                    );
                    assert!(!wire.contains("input_json_delta"), "{wire}");
                } else {
                    let value: Value = serde_json::from_str(&wire).unwrap();
                    assert_eq!(value["stop_reason"], "max_tokens");
                    assert!(
                        !value["content"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .any(|block| block["type"] == "tool_use")
                    );
                }
                if path == "/v1/responses" && matches!(mode, "length" | "valid_length") {
                    assert!(wire.contains("incomplete"), "{wire}");
                    assert!(!wire.contains("response.completed"), "{wire}");
                }
                if path == "/v1/messages" && matches!(mode, "length" | "valid_length") {
                    assert!(wire.contains("max_tokens"), "{wire}");
                }
            }
        }
        assert_eq!(f.engine.inflight(), vec![0]);
    }
}

#[tokio::test]
async fn public_malformed_or_partial_native_sse_has_no_fake_success() {
    for mode in ["malformed", "partial"] {
        let native = fake(GF, 100, mode).await;
        let f = frontend(GF, vec![worker(&native, NativeTier::Shared)]).await;
        for (path, body) in requests(GF, true) {
            let reply = post(&f, path, body).await;
            let wire = reply.text().await.unwrap();
            assert!(
                wire.contains("error") || wire.contains("failed"),
                "{path} {mode}: {wire}"
            );
            assert!(!wire.contains("response.completed"), "{wire}");
            assert!(!wire.contains("message_stop"), "{wire}");
        }
        assert_eq!(native.state.calls.load(Ordering::Acquire), 3);
        assert_eq!(f.engine.inflight(), vec![0]);
    }
}

#[tokio::test]
async fn public_client_disconnect_releases_native_reservation() {
    let native = fake(GF, 100, "hold").await;
    let f = frontend(GF, vec![worker(&native, NativeTier::Shared)]).await;
    let response = post(&f, "/v1/chat/completions", chat(GF, true)).await;
    let mut stream = response.bytes_stream();
    assert!(
        tokio::time::timeout(Duration::from_secs(2), stream.next())
            .await
            .unwrap()
            .unwrap()
            .is_ok()
    );
    assert_eq!(f.engine.inflight(), vec![1]);
    drop(stream);
    tokio::time::timeout(Duration::from_secs(2), async {
        while f.engine.inflight() != vec![0] {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("public client drop must cancel native reservation");
    assert_eq!(native.state.calls.load(Ordering::Acquire), 1);
}

#[tokio::test]
async fn public_deepseek_missing_fence_is_closed_before_native_work() {
    let native = fake(DS, 100, "good").await;
    let f = frontend(DS, vec![worker(&native, NativeTier::Short)]).await;
    let mut body = chat(DS, false);
    body["max_tokens"] = json!(1);
    body["reasoning_effort"] = json!("none");
    assert!(
        !post(&f, "/v1/chat/completions", body)
            .await
            .status()
            .is_success()
    );
    assert!(native.state.count_calls.lock().unwrap().is_empty());
    assert_eq!(native.state.calls.load(Ordering::Acquire), 0);
    assert_eq!(f.engine.inflight(), vec![0]);
}

#[tokio::test]
async fn public_qwen_effort_and_disabled_thinking_agree_across_protocols() {
    let native = fake(QWEN, 100, "good").await;
    let f = frontend(QWEN, vec![worker(&native, NativeTier::Short)]).await;
    for effort in ["none", "high"] {
        for (path, mut body) in requests(QWEN, false) {
            match path {
                "/v1/chat/completions" => body["reasoning_effort"] = json!(effort),
                "/v1/responses" => body["reasoning"] = json!({"effort":effort}),
                _ if effort == "none" => body["thinking"] = json!({"type":"disabled"}),
                _ => body["output_config"] = json!({"effort":effort}),
            }
            let reply = post(&f, path, body).await;
            assert!(
                reply.status().is_success(),
                "{path} {effort}: {}",
                reply.status()
            );
            let captured = native
                .state
                .generation_calls
                .lock()
                .unwrap()
                .last()
                .unwrap()
                .clone();
            assert_eq!(
                captured["chat_template_kwargs"]["enable_thinking"],
                effort != "none"
            );
            if effort == "high" {
                assert_eq!(
                    captured["chat_template_kwargs"]["reasoning_effort"],
                    "xhigh"
                );
            }
        }
    }
    assert_eq!(native.state.calls.load(Ordering::Acquire), 6);
}

#[tokio::test]
async fn public_messages_fixed_budget_and_namespace_are_native_controls() {
    let native = fake(GF, 100, "good").await;
    let f = frontend(GF, vec![worker(&native, NativeTier::Shared)]).await;
    let mut body = messages(GF, false);
    body["thinking"] = json!({"type":"enabled","budget_tokens":32});
    body["prompt_cache_key"] = json!("messages-S1");
    body["messages"][0]["content"] = json!([{"type":"text","text":"hello🙂"}]);
    let reply = post(&f, "/v1/messages", body).await;
    assert!(reply.status().is_success());
    let captured = native.state.generation_calls.lock().unwrap()[0].clone();
    assert_eq!(captured["thinking_token_budget"], 32);
    assert_eq!(captured["max_tokens"], 800);
    assert_eq!(captured["cache_salt"], "messages-S1");
    assert_eq!(captured["chat_template_kwargs"]["enable_thinking"], true);
    assert!(
        captured["messages"][0]["content"]
            .to_string()
            .contains("hello🙂")
    );
    assert_eq!(native.state.calls.load(Ordering::Acquire), 1);
}

#[tokio::test]
async fn public_messages_unqualified_cache_lifetime_and_breakpoints_fail_before_native() {
    let native = fake(GF, 100, "good").await;
    let f = frontend(GF, vec![worker(&native, NativeTier::Shared)]).await;
    let marker = json!({"type":"ephemeral"});
    let mut variants = Vec::new();
    let mut top = messages(GF, false);
    top["cache_control"] = marker.clone();
    variants.push(top);
    let mut explicit = messages(GF, false);
    explicit["cache_control"] = json!({"type":"ephemeral","ttl":"5m"});
    variants.push(explicit);
    let mut system = messages(GF, false);
    system["system"] = json!([{"type":"text","text":"instruction","cache_control":marker}]);
    variants.push(system);
    let mut content = messages(GF, false);
    content["messages"][0]["content"] =
        json!([{"type":"text","text":"hello","cache_control":marker}]);
    variants.push(content);
    let mut tool = messages(GF, false);
    tool["tools"] =
        json!([{"name":"Read","input_schema":{"type":"object"},"cache_control":marker}]);
    variants.push(tool);
    let mut use_block = messages(GF, false);
    use_block["messages"] = json!([{"role":"assistant","content":[{"type":"tool_use","id":"call1","name":"Read","input":{},"cache_control":marker}]}]);
    variants.push(use_block);
    let mut result_block = messages(GF, false);
    result_block["messages"] = json!([{"role":"user","content":[{"type":"tool_result","tool_use_id":"call1","content":"ok","cache_control":marker}]}]);
    variants.push(result_block);
    for body in variants {
        for path in ["/v1/messages", "/v1/messages/count_tokens"] {
            let response = post(&f, path, body.clone()).await;
            assert_eq!(response.status().as_u16(), 400, "{path}");
            let error: Value = response.json().await.unwrap();
            assert_eq!(error["type"], "error");
            assert_eq!(error["error"]["type"], "invalid_request_error");
            assert!(
                error["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("cache_control")
            );
        }
    }
    assert!(native.state.count_calls.lock().unwrap().is_empty());
    assert!(native.state.generation_calls.lock().unwrap().is_empty());
    assert_eq!(native.state.calls.load(Ordering::Acquire), 0);
}

#[tokio::test]
async fn public_native_health_distinguishes_readiness_from_liveness() {
    for (model, tier, status) in [(GF, NativeTier::Shared, 200), (DS, NativeTier::Short, 503)] {
        let native = fake(model, 100, "good").await;
        let f = frontend(model, vec![worker(&native, tier)]).await;
        let reply = reqwest::get(format!("{}/health", f.url)).await.unwrap();
        assert_eq!(reply.status().as_u16(), status);
        let health: Value = reply.json().await.unwrap();
        assert_eq!(health["transport"], "native_http");
        assert_eq!(health["model"], model);
        assert_eq!(health["inflight"], json!([0]));
        let live = reqwest::get(format!("{}/live", f.url)).await.unwrap();
        assert!(live.status().is_success());
        assert!(native.state.count_calls.lock().unwrap().is_empty());
        assert_eq!(native.state.calls.load(Ordering::Acquire), 0);
    }
}

#[tokio::test]
async fn public_bearer_auth_covers_generation_count_and_model_routes() {
    let native = fake(GF, 100, "good").await;
    let f = frontend_with_token(
        GF,
        vec![worker(&native, NativeTier::Shared)],
        Some(Arc::from("local-fixture-only")),
    )
    .await;
    let client = reqwest::Client::new();
    for wrong in [None, Some("wrong-fixture-token")] {
        for (path, body) in requests(GF, false).into_iter().chain([
            ("/v1/responses/input_tokens", responses(GF, false)),
            ("/v1/messages/count_tokens", messages(GF, false)),
        ]) {
            let mut request = client.post(format!("{}{path}", f.url)).json(&body);
            if let Some(token) = wrong {
                request = request.bearer_auth(token);
            }
            let reply = request.send().await.unwrap();
            assert_eq!(reply.status().as_u16(), 401, "{path}");
        }
        let mut request = client.get(format!("{}/v1/models", f.url));
        if let Some(token) = wrong {
            request = request.bearer_auth(token);
        }
        assert_eq!(request.send().await.unwrap().status().as_u16(), 401);
    }
    assert!(native.state.count_calls.lock().unwrap().is_empty());
    assert_eq!(native.state.calls.load(Ordering::Acquire), 0);
    let allowed = client
        .post(format!("{}/v1/chat/completions", f.url))
        .bearer_auth("local-fixture-only")
        .json(&chat(GF, false))
        .send()
        .await
        .unwrap();
    assert!(allowed.status().is_success());
    assert_eq!(native.state.calls.load(Ordering::Acquire), 1);
    assert!(
        reqwest::get(format!("{}/health", f.url))
            .await
            .unwrap()
            .status()
            .is_success()
    );
}

fn strict_requests(streaming: bool) -> [(&'static str, Value); 3] {
    let schema = json!({"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false});
    requests(GF, streaming).map(|(path,mut body)| {
        body["tools"] = match path {
            "/v1/chat/completions" => json!([{"type":"function","function":{"name":"answer","parameters":schema,"strict":true}}]),
            "/v1/responses" => json!([{"type":"function","name":"answer","parameters":schema,"strict":true}]),
            _ => json!([{"name":"answer","input_schema":schema,"strict":true}]),
        };
        (path,body)
    })
}

fn assert_strict_success(path: &str, streaming: bool, length: bool, wire: &str) {
    assert!(wire.contains("call-native-17"), "{path}: {wire}");
    if streaming {
        let parsed = events(wire);
        match path {
            "/v1/chat/completions" => {
                assert!(wire.contains("[DONE]"));
                assert!(parsed.iter().any(|e| e["usage"]["cached_tokens"] == 17));
                assert!(parsed.iter().any(|e| e["choices"][0]["finish_reason"]
                    == if length { "length" } else { "tool_calls" }));
            }
            "/v1/responses" => {
                let expected = if length {
                    "response.incomplete"
                } else {
                    "response.completed"
                };
                let terminal = parsed.iter().find(|e| e["type"] == expected).expect(wire);
                assert_eq!(terminal["response"]["usage"]["output_tokens"], 4);
                assert_eq!(
                    terminal["response"]["usage"]["input_tokens_details"]["cached_tokens"],
                    17
                );
                if length {
                    assert!(!wire.contains("response.completed"));
                }
            }
            _ => {
                assert!(parsed.iter().any(|e| e["type"] == "message_stop"));
                assert!(
                    parsed
                        .iter()
                        .any(|e| e["content_block"]["type"] == "tool_use"
                            && e["content_block"]["id"] == "call-native-17")
                );
                assert!(
                    parsed
                        .iter()
                        .any(|e| e["type"] == "message_delta" && e["usage"]["output_tokens"] == 4)
                );
            }
        }
    } else {
        let parsed: Value = serde_json::from_str(wire).unwrap();
        match path {
            "/v1/chat/completions" => {
                assert_eq!(
                    parsed["choices"][0]["message"]["tool_calls"][0]["id"],
                    "call-native-17"
                );
                assert_eq!(parsed["usage"]["cached_tokens"], 17);
            }
            "/v1/responses" => {
                assert_eq!(
                    parsed["status"],
                    if length { "incomplete" } else { "completed" }
                );
                assert_eq!(parsed["output"][0]["call_id"], "call-native-17");
                assert_eq!(parsed["usage"]["output_tokens"], 4);
            }
            _ => {
                assert_eq!(parsed["content"][0]["type"], "tool_use");
                assert_eq!(parsed["content"][0]["id"], "call-native-17");
                assert_eq!(parsed["content"][0]["input"], json!({"answer":"🙂"}));
                assert_eq!(parsed["usage"]["cache_read_input_tokens"], 17);
            }
        }
    }
}

fn assert_public_error_envelope(path: &str, value: &Value) {
    let error = value
        .get("error")
        .and_then(Value::as_object)
        .expect("public error must be nested under error");
    assert!(
        error
            .get("message")
            .and_then(Value::as_str)
            .is_some_and(|message| !message.is_empty())
    );
    assert!(
        error
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|kind| !kind.is_empty())
    );
    assert!(
        value.get("message").is_none(),
        "internal flat error must not cross the public boundary: {value}"
    );
    if path == "/v1/messages" {
        assert_eq!(value["type"], "error");
    }
}

#[tokio::test]
async fn public_strict_schema_compiles_before_native_count_and_generation() {
    let native = fake(GF, 100, "strict_tool").await;
    let f = frontend(GF, vec![worker(&native, NativeTier::Shared)]).await;
    let (_, mut body) = strict_requests(false).into_iter().next().unwrap();
    // Syntactically valid meta-schema, but the strict validator cannot compile
    // this invalid regular expression. Failure must precede native dispatch.
    body["tools"][0]["function"]["parameters"]["properties"]["answer"]["pattern"] = json!("[");
    let reply = post(&f, "/v1/chat/completions", body).await;
    assert!(reply.status().is_client_error());
    let error: Value = reply.json().await.unwrap();
    assert_public_error_envelope("/v1/chat/completions", &error);
    assert!(native.state.count_calls.lock().unwrap().is_empty());
    assert_eq!(native.state.calls.load(Ordering::Acquire), 0);
    assert_eq!(f.engine.inflight(), vec![0]);
}

#[tokio::test]
async fn public_strict_valid_json_and_fragmented_sse_keep_tool_identity() {
    for streaming in [false, true] {
        let native = fake(
            GF,
            100,
            if streaming {
                "strict_fragments"
            } else {
                "strict_tool"
            },
        )
        .await;
        let f = frontend(GF, vec![worker(&native, NativeTier::Shared)]).await;
        for (path, body) in strict_requests(streaming) {
            let reply = post(&f, path, body).await;
            let status = reply.status();
            let wire = reply.text().await.unwrap();
            assert!(status.is_success(), "{path}: {status} {wire}");
            assert_strict_success(path, streaming, false, &wire);
            let generated = native
                .state
                .generation_calls
                .lock()
                .unwrap()
                .last()
                .unwrap()
                .clone();
            assert_eq!(generated["tools"][0]["function"]["strict"], true);
        }
        assert_eq!(native.state.calls.load(Ordering::Acquire), 3);
        assert_eq!(f.engine.inflight(), vec![0]);
    }
}

#[tokio::test]
async fn public_strict_invalid_instance_and_undeclared_tool_fail_closed() {
    for (mode, streaming) in [("strict_invalid", false), ("strict_undeclared", true)] {
        let native = fake(GF, 100, mode).await;
        let f = frontend(GF, vec![worker(&native, NativeTier::Shared)]).await;
        for (path, body) in strict_requests(streaming) {
            let reply = post(&f, path, body).await;
            let status = reply.status();
            let wire = reply.text().await.unwrap();
            if streaming {
                assert!(
                    wire.contains("error") || wire.contains("failed"),
                    "{path}: {wire}"
                );
                assert!(!wire.contains("response.completed"));
                assert!(!wire.contains("message_stop"));
            } else {
                assert!(!status.is_success(), "{path}: {wire}");
                assert_public_error_envelope(path, &serde_json::from_str::<Value>(&wire).unwrap());
            }
            assert!(
                !wire.contains("call-native-17"),
                "{path}: strict tool leaked: {wire}"
            );
            assert!(!wire.contains("input_json_delta"), "{path}: {wire}");
            assert!(!wire.contains("\"tool_use\""), "{path}: {wire}");
        }
        assert_eq!(native.state.calls.load(Ordering::Acquire), 3);
        assert_eq!(f.engine.inflight(), vec![0]);
    }
}

#[tokio::test]
async fn public_strict_fragmented_args_and_late_native_error_never_emit_tool() {
    let native = fake(GF, 100, "strict_late_error").await;
    let f = frontend(GF, vec![worker(&native, NativeTier::Shared)]).await;
    for (path, body) in strict_requests(true) {
        let reply = post(&f, path, body).await;
        let wire = reply.text().await.unwrap();
        assert!(
            wire.contains("error") || wire.contains("failed"),
            "{path}: {wire}"
        );
        for forbidden in [
            "call-native-17",
            "input_json_delta",
            "\"tool_use\"",
            "response.completed",
            "message_stop",
        ] {
            assert!(
                !wire.contains(forbidden),
                "{path}: leaked {forbidden}: {wire}"
            );
        }
    }
    assert_eq!(native.state.calls.load(Ordering::Acquire), 3);
    assert_eq!(f.engine.inflight(), vec![0]);
}

#[tokio::test]
async fn public_strict_valid_length_keeps_id_and_responses_incomplete() {
    let native = fake(GF, 100, "strict_valid_length").await;
    let f = frontend(GF, vec![worker(&native, NativeTier::Shared)]).await;
    for streaming in [false, true] {
        for (path, body) in strict_requests(streaming)
            .into_iter()
            .filter(|(path, _)| *path == "/v1/responses" || (*path == "/v1/messages" && streaming))
        {
            let reply = post(&f, path, body).await;
            let status = reply.status();
            let wire = reply.text().await.unwrap();
            assert!(status.is_success(), "{path}: {status} {wire}");
            assert_strict_success(path, streaming, true, &wire);
            if path == "/v1/messages" {
                assert!(wire.contains("max_tokens"), "{wire}");
            }
        }
    }
    assert_eq!(native.state.calls.load(Ordering::Acquire), 3);
    assert_eq!(f.engine.inflight(), vec![0]);
    // Declaring an unused strict tool must not turn a nonstrict tool's partial
    // length result into failure. This fixture does not claim low-latency deltas.
    let mixed = fake(GF, 100, "strict_mixed_partial").await;
    let mixed_frontend = frontend(GF, vec![worker(&mixed, NativeTier::Shared)]).await;
    let (_, mut body) = strict_requests(true)
        .into_iter()
        .find(|(path, _)| *path == "/v1/responses")
        .unwrap();
    body["tools"].as_array_mut().unwrap().push(json!({"type":"function","name":"freeform","parameters":{"type":"object","properties":{"q":{"type":"string"}}},"strict":false}));
    let reply = post(&mixed_frontend, "/v1/responses", body).await;
    let status = reply.status();
    let wire = reply.text().await.unwrap();
    assert!(status.is_success(), "{status}: {wire}");
    assert_strict_success("/v1/responses", true, true, &wire);
    assert!(wire.contains("freeform"));
    assert!(!wire.contains("response.failed"), "{wire}");
    assert_eq!(mixed.state.calls.load(Ordering::Acquire), 1);
    assert_eq!(mixed_frontend.engine.inflight(), vec![0]);
}

// This grant is a CPU fixture to exercise the already tested authority check,
// not an etcd, native short-prefix, or production HA qualification.
struct CapAuthorityStore;
#[async_trait::async_trait]
impl dynamo_llm::native_fencing::FenceStore for CapAuthorityStore {
    async fn acquire(
        &self,
        _: &str,
        _: &str,
    ) -> Result<
        Option<dynamo_llm::native_fencing::LeaseGrant>,
        dynamo_llm::native_fencing::FenceError,
    > {
        Ok(Some(dynamo_llm::native_fencing::LeaseGrant {
            id: 1,
            epoch: 1,
        }))
    }
    async fn keep_alive(
        &self,
        _: dynamo_llm::native_fencing::LeaseGrant,
    ) -> Result<(), dynamo_llm::native_fencing::FenceError> {
        Ok(())
    }
    async fn read_short_gate(
        &self,
    ) -> Result<dynamo_llm::native_fencing::GateState, dynamo_llm::native_fencing::FenceError> {
        Ok(dynamo_llm::native_fencing::GateState {
            enabled: false,
            commit_id: "CPU cap fixture; no short traffic".into(),
            revision: 1,
            mod_revision: 1,
        })
    }
}
struct CapUnusedPrefixStore;
#[async_trait::async_trait]
impl dynamo_llm::native_fencing::prefix_records::PrefixStore for CapUnusedPrefixStore {
    async fn snapshot(
        &self,
    ) -> Result<
        dynamo_llm::native_fencing::prefix_records::PrefixSnapshot,
        dynamo_llm::native_fencing::prefix_records::PrefixError,
    > {
        Ok(dynamo_llm::native_fencing::prefix_records::PrefixSnapshot {
            revision: 1,
            rows: Vec::new(),
        })
    }
    async fn watch_from(
        &self,
        _: i64,
    ) -> Result<
        dynamo_llm::native_fencing::prefix_records::PrefixStream,
        dynamo_llm::native_fencing::prefix_records::PrefixError,
    > {
        Ok(Box::pin(futures::stream::pending()))
    }
    async fn put(
        &self,
        _: dynamo_llm::native_fencing::prefix_records::PrefixRow,
        _: u64,
    ) -> Result<(), dynamo_llm::native_fencing::prefix_records::PrefixError> {
        Ok(())
    }
}
async fn cap_ds_authority(engine: NativeHttpEngine) -> NativeHttpEngine {
    let fence = dynamo_llm::native_fencing::NativeFence::new(
        dynamo_llm::native_fencing::FenceConfig {
            election_key: "/CPU-public-cap/authority-only".into(),
            holder: "CPU-public-cap".into(),
            // No requests are sent to these closed IDs; this fixture selects long.
            short_workers: (0..12)
                .map(|id| format!("http://127.0.0.1:{}", 49000 + id))
                .collect(),
            max_effective_load: 2,
        },
        Arc::new(CapAuthorityStore),
        Arc::new(CapUnusedPrefixStore),
    )
    .unwrap();
    fence.refresh_authority().await.unwrap();
    engine.with_fence(fence)
}

fn qualify_cap_worker(worker: &mut NativeWorkerConfig, model: &str) {
    // Deliberately synthetic CPU evidence, never a native GPU qualification.
    let identity = NativeWorkerIdentity {
        model: model.into(),
        model_identity: "CPU fixture model/renderer/template identity".into(),
        source_sha256: "a".repeat(64),
        template_sha256: "b".repeat(64),
    };
    worker.identity = Some(identity.clone());
    worker.thinking_budget_qualification = Some(ThinkingBudgetQualification {
        identity,
        evidence_sha256: "c".repeat(64),
        native_field: ThinkingBudgetField::ThinkingTokenBudget,
    });
    if model == DS {
        worker.max_inflight = None;
        worker.native_dp_size = Some(1);
    }
}
async fn cap_frontend(model: &str, workers: Vec<NativeWorkerConfig>) -> Frontend {
    let engine = NativeHttpEngine::new(NativeHttpEngineConfig {
        model: model.into(),
        workers,
        qualification_tier: None,
        qualified_model_identity: if model == DS {
            Some("CPU fixture model/renderer/template identity".into())
        } else {
            None
        },
    })
    .unwrap();
    let engine = if model == DS {
        cap_ds_authority(engine).await
    } else {
        engine
    };
    frontend_from_engine(model, engine, None).await
}
async fn cap_card(f: &Frontend) -> Value {
    let reply = reqwest::get(format!("{}/v1/models", f.url)).await.unwrap();
    assert!(reply.status().is_success());
    reply.json::<Value>().await.unwrap()["data"][0].clone()
}

#[tokio::test]
async fn public_model_input_card_uses_qualified_long_capacity_without_short_minimum() {
    let short = fake(QWEN, 100, "good").await;
    let long = fake(QWEN, 100, "good").await;
    let smaller_long = fake(QWEN, 100, "good").await;
    let mut short_config = worker(&short, NativeTier::Short);
    short_config.max_input_tokens = Some(16_384);
    let mut long_config = worker(&long, NativeTier::Long);
    long_config.max_input_tokens = Some(1_048_569);
    let mut smaller_config = worker(&smaller_long, NativeTier::Long);
    smaller_config.max_input_tokens = Some(262_144);
    let f = frontend(
        QWEN,
        vec![short_config.clone(), smaller_config, long_config],
    )
    .await;
    let card = cap_card(&f).await;
    assert_eq!(card["max_input_tokens"], 1_048_569);
    assert_eq!(card["context_window"], 1_048_576);
    assert_eq!(card["max_output_tokens"], 262_144);
    assert_eq!(card["id"], QWEN);
    assert_eq!(
        card["gateway_capabilities"]["prompt_cache"]["ttl_guaranteed"],
        false
    );
    assert_eq!(
        card["gateway_capabilities"]["prompt_cache"]["block_breakpoints_supported"],
        false
    );
    let mut unknown_long = worker(&long, NativeTier::Long);
    unknown_long.max_input_tokens = None;
    unknown_long.capability_source = None;
    let unqualified = frontend(QWEN, vec![short_config, unknown_long]).await;
    assert!(
        cap_card(&unqualified)
            .await
            .get("max_input_tokens")
            .is_none()
    );
    for server in [&short, &long, &smaller_long] {
        assert_eq!(server.state.calls.load(Ordering::Acquire), 0);
        assert!(server.state.count_calls.lock().unwrap().is_empty());
    }
}
fn cap_body(model: &str) -> Value {
    let mut body = chat(model, false);
    body["thinking_token_budget"] = json!(32);
    body
}
fn assert_cap_native(server: &FakeWorker) {
    let generated = server
        .state
        .generation_calls
        .lock()
        .unwrap()
        .last()
        .unwrap()
        .clone();
    assert_eq!(generated["thinking_token_budget"], 32);
    assert_eq!(generated["max_tokens"], 800);
    assert_eq!(generated["chat_template_kwargs"]["enable_thinking"], true);
}

#[tokio::test]
async fn public_cap_qualified_four_models_keep_budget_and_card_agreement() {
    for model in [GLM, GF, DS, QWEN] {
        let count = if model == DS { 65500 } else { 100 };
        let first = fake(model, count, "good").await;
        let second = fake(model, count, "good").await;
        let mut first_config = worker(
            &first,
            if model == GF {
                NativeTier::Shared
            } else {
                NativeTier::Short
            },
        );
        qualify_cap_worker(&mut first_config, model);
        let mut workers = vec![first_config];
        if model != GF {
            let mut long = worker(&second, NativeTier::Long);
            qualify_cap_worker(&mut long, model);
            workers.push(long);
        }
        let f = cap_frontend(model, workers).await;
        assert_eq!(
            cap_card(&f).await["gateway_capabilities"]["reasoning"]["fixed_budget_supported"],
            true
        );
        assert!(f.engine.contract().supports_fixed_thinking_budget());
        let reply = post(&f, "/v1/chat/completions", cap_body(model)).await;
        let status = reply.status();
        let wire = reply.text().await.unwrap();
        assert!(status.is_success(), "{model}: {status} {wire}");
        let generated_server = if model == DS { &second } else { &first };
        assert_cap_native(generated_server);
        let count_body = first
            .state
            .count_calls
            .lock()
            .unwrap()
            .last()
            .unwrap()
            .clone();
        assert_eq!(count_body["thinking_token_budget"], 32);
        assert!(count_body.get("max_tokens").is_none());
        let mut generated = generated_server
            .state
            .generation_calls
            .lock()
            .unwrap()
            .last()
            .unwrap()
            .clone();
        for field in ["max_tokens", "stream", "stream_options"] {
            generated.as_object_mut().unwrap().remove(field);
        }
        assert_eq!(
            generated, count_body,
            "{model}: counted body must match dispatched body"
        );
        if model == GLM || model == GF {
            let prior_count = first.state.count_calls.lock().unwrap().len()
                + second.state.count_calls.lock().unwrap().len();
            let prior_gen = first.state.calls.load(Ordering::Acquire)
                + second.state.calls.load(Ordering::Acquire);
            let mut disabled = chat(model, false);
            disabled["reasoning_effort"] = json!("none");
            assert!(
                post(&f, "/v1/chat/completions", disabled)
                    .await
                    .status()
                    .is_client_error()
            );
            assert_eq!(
                first.state.count_calls.lock().unwrap().len()
                    + second.state.count_calls.lock().unwrap().len(),
                prior_count
            );
            assert_eq!(
                first.state.calls.load(Ordering::Acquire)
                    + second.state.calls.load(Ordering::Acquire),
                prior_gen
            );
        }
    }
}

#[tokio::test]
async fn public_cap_missing_receipt_or_tier_coverage_rejects_before_native_work() {
    for (model, partial_coverage) in [(GF, false), (GLM, true)] {
        let native = fake(model, 100, "good").await;
        let mut configuration = worker(
            &native,
            if model == GF {
                NativeTier::Shared
            } else {
                NativeTier::Short
            },
        );
        configuration.identity = None;
        configuration.thinking_budget_qualification = None;
        if partial_coverage {
            qualify_cap_worker(&mut configuration, model);
        }
        let f = cap_frontend(model, vec![configuration]).await;
        assert_eq!(
            cap_card(&f).await["gateway_capabilities"]["reasoning"]["fixed_budget_supported"],
            false
        );
        assert!(!f.engine.contract().supports_fixed_thinking_budget());
        let reply = post(&f, "/v1/chat/completions", cap_body(model)).await;
        assert!(reply.status().is_client_error());
        assert_public_error_envelope(
            "/v1/chat/completions",
            &reply.json::<Value>().await.unwrap(),
        );
        assert!(native.state.count_calls.lock().unwrap().is_empty());
        assert_eq!(native.state.calls.load(Ordering::Acquire), 0);
    }
}

#[tokio::test]
async fn public_cap_mixed_workers_and_connect_retry_never_use_unqualified_candidate() {
    for connection_failure in [false, true] {
        let short = fake(GLM, 100, "good").await;
        let long = fake(GLM, 100, "good").await;
        let unqualified = fake(GLM, 100, "good").await;
        let mut short_config = worker(&short, NativeTier::Short);
        qualify_cap_worker(&mut short_config, GLM);
        if connection_failure {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            short_config.url = format!("http://{}", listener.local_addr().unwrap());
            drop(listener); // A real refused TCP connection, not an HTTP error.
        }
        let mut long_config = worker(&long, NativeTier::Long);
        qualify_cap_worker(&mut long_config, GLM);
        let unqualified_config = worker(&unqualified, NativeTier::Short);
        let f = cap_frontend(GLM, vec![unqualified_config, short_config, long_config]).await;
        assert_eq!(
            cap_card(&f).await["gateway_capabilities"]["reasoning"]["fixed_budget_supported"],
            true
        );
        let reply = post(&f, "/v1/chat/completions", cap_body(GLM)).await;
        if connection_failure {
            assert!(!reply.status().is_success());
        } else {
            assert!(reply.status().is_success());
            assert_cap_native(&short);
        }
        assert!(unqualified.state.count_calls.lock().unwrap().is_empty());
        assert_eq!(unqualified.state.calls.load(Ordering::Acquire), 0);
        assert_eq!(long.state.calls.load(Ordering::Acquire), 0);
        assert_eq!(f.engine.inflight(), vec![0, 0, 0]);
    }
}

#[tokio::test]
async fn public_cap_messages_and_responses_fixed_budget_match_qualified_card() {
    let native = fake(GF, 100, "good").await;
    let f = cap_frontend(GF, vec![worker(&native, NativeTier::Shared)]).await;
    assert_eq!(
        cap_card(&f).await["gateway_capabilities"]["reasoning"]["fixed_budget_supported"],
        true
    );
    let mut messages_body = messages(GF, false);
    messages_body["thinking"] = json!({"type":"enabled","budget_tokens":32});
    let mut responses_body = responses(GF, false);
    responses_body["thinking_token_budget"] = json!(32);
    for (path, body) in [
        ("/v1/messages", messages_body),
        ("/v1/responses", responses_body),
    ] {
        let reply = post(&f, path, body).await;
        let status = reply.status();
        let wire = reply.text().await.unwrap();
        assert!(status.is_success(), "{path}: {status} {wire}");
        assert_cap_native(&native);
        assert_eq!(
            native.state.count_calls.lock().unwrap().last().unwrap()["thinking_token_budget"],
            32
        );
    }
    assert_eq!(native.state.calls.load(Ordering::Acquire), 2);
}

#[tokio::test]
async fn public_exact_load_missing_metadata_fails_closed_without_generation() {
    let native = fake(GF, 100, "load_ids_missing").await;
    let f = frontend(GF, vec![worker(&native, NativeTier::Shared)]).await;
    let reply = post(&f, "/v1/chat/completions", chat(GF, false)).await;
    let status = reply.status();
    let value: Value = reply.json().await.unwrap();
    assert!(
        !status.is_success(),
        "metadata absence must not become native load zero: {value}"
    );
    assert_eq!(status.as_u16(), 503, "{value}");
    assert_public_error_envelope("/v1/chat/completions", &value);
    assert!(
        native.state.load_calls.load(Ordering::Acquire) > 0,
        "public request must query the real fixture HTTP load endpoint"
    );
    assert_eq!(native.state.calls.load(Ordering::Acquire), 0);
    assert_eq!(f.engine.inflight(), vec![0]);
}

fn qualify_forced_zero_worker(worker: &mut NativeWorkerConfig, model: &str) {
    let identity = worker
        .identity
        .get_or_insert_with(|| NativeWorkerIdentity {
            model: model.into(),
            model_identity: "CPU fixture model/renderer/template identity".into(),
            source_sha256: "a".repeat(64),
            template_sha256: "b".repeat(64),
        })
        .clone();
    worker.forced_zero_qualification = Some(ForcedZeroThinkingQualification {
        identity,
        evidence_sha256: "d".repeat(64),
        native_field: ForcedZeroThinkingField::ForcedZeroThinking,
    });
}
fn forced_zero_request(model: &str, path: &str) -> Value {
    let literal = "CPU literal <think>保留正文</think> <tool_call>非控制文本</tool_call>";
    let schema = json!({"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false});
    if path == "/v1/chat/completions" {
        let mut body = chat(model, false);
        body["reasoning_effort"] = json!("none");
        body["messages"] = json!([
            {"role":"user","content":literal},
            {"role":"assistant","content":null,"tool_calls":[{"id":"history-call-CPU","type":"function","function":{"name":"answer","arguments":"{\"answer\":\"<think>literal</think>\"}"}}]},
            {"role":"tool","tool_call_id":"history-call-CPU","content":literal}
        ]);
        body["tools"] = json!([{"type":"function","function":{"name":"answer","description":literal,"parameters":schema,"strict":false}}]);
        body
    } else {
        let mut body = messages(model, false);
        body["thinking"] = json!({"type":"disabled"});
        body["messages"] = json!([
            {"role":"user","content":literal},
            {"role":"assistant","content":[{"type":"tool_use","id":"history-call-CPU","name":"answer","input":{"answer":"<think>literal</think>"}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"history-call-CPU","content":literal}]}
        ]);
        body["tools"] =
            json!([{"name":"answer","description":literal,"input_schema":schema,"strict":false}]);
        body
    }
}
fn assert_forced_zero_capture(native: &FakeWorker, counter: &FakeWorker) {
    let generated = native
        .state
        .generation_calls
        .lock()
        .unwrap()
        .last()
        .unwrap()
        .clone();
    let count = counter
        .state
        .count_calls
        .lock()
        .unwrap()
        .last()
        .unwrap()
        .clone();
    assert_eq!(generated["forced_zero_thinking"], true);
    assert_eq!(count["forced_zero_thinking"], true);
    assert!(generated.get("reasoning_effort").is_none());
    assert!(
        generated["chat_template_kwargs"]
            .get("reasoning_effort")
            .is_none()
    );
    assert_ne!(generated["chat_template_kwargs"]["enable_thinking"], false);
    assert_ne!(generated["chat_template_kwargs"]["thinking"], false);
    assert_eq!(generated["max_tokens"], 800);
    assert!(
        generated["messages"][0]["content"]
            .to_string()
            .contains("<think>保留正文</think>")
    );
    assert_eq!(
        generated["messages"][1]["tool_calls"][0]["id"],
        "history-call-CPU"
    );
    assert_eq!(generated["messages"][2]["tool_call_id"], "history-call-CPU");
    assert!(
        generated["tools"][0]["function"]["description"]
            .as_str()
            .unwrap()
            .contains("<tool_call>非控制文本</tool_call>")
    );
    let mut stripped = generated;
    for field in ["max_tokens", "stream", "stream_options"] {
        stripped.as_object_mut().unwrap().remove(field);
    }
    assert_eq!(stripped, count);
}

#[tokio::test]
async fn public_forced_zero_qualified_glm_families_card_and_standard_ingress_agree() {
    for model in [GF, GLM] {
        let short = fake(model, 100, "good").await;
        let long = fake(model, 100, "good").await;
        let mut short_config = worker(
            &short,
            if model == GF {
                NativeTier::Shared
            } else {
                NativeTier::Short
            },
        );
        qualify_forced_zero_worker(&mut short_config, model);
        let mut configurations = vec![short_config];
        if model == GLM {
            let mut long_config = worker(&long, NativeTier::Long);
            qualify_forced_zero_worker(&mut long_config, model);
            configurations.push(long_config);
        }
        let f = cap_frontend(model, configurations).await;
        let card = cap_card(&f).await;
        assert_eq!(
            card["gateway_capabilities"]["reasoning"]["disabled_supported"],
            true
        );
        assert!(
            card["gateway_capabilities"]["reasoning"]["efforts"]
                .as_array()
                .unwrap()
                .iter()
                .any(|e| e == "none")
        );
        for path in ["/v1/chat/completions", "/v1/messages"] {
            let reply = post(&f, path, forced_zero_request(model, path)).await;
            let status = reply.status();
            let wire = reply.text().await.unwrap();
            assert!(status.is_success(), "{model} {path}: {status} {wire}");
            assert_forced_zero_capture(
                &short,
                if model == GLM && path == "/v1/messages" {
                    &long
                } else {
                    &short
                },
            );
        }
        assert_eq!(short.state.calls.load(Ordering::Acquire), 2);
        assert_eq!(long.state.calls.load(Ordering::Acquire), 0);
    }
}

#[tokio::test]
async fn public_forced_zero_candidate_filter_coverage_and_private_fields_fail_closed() {
    for dead_qualified in [false, true] {
        let short = fake(GLM, 100, "good").await;
        let long = fake(GLM, 100, "good").await;
        let unqualified = fake(GLM, 100, "good").await;
        let mut short_config = worker(&short, NativeTier::Short);
        qualify_forced_zero_worker(&mut short_config, GLM);
        let mut long_config = worker(&long, NativeTier::Long);
        qualify_forced_zero_worker(&mut long_config, GLM);
        if dead_qualified {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            short_config.url = format!("http://{}", listener.local_addr().unwrap());
            drop(listener);
        }
        let f = cap_frontend(
            GLM,
            vec![
                worker(&unqualified, NativeTier::Short),
                short_config,
                long_config,
            ],
        )
        .await;
        assert_eq!(
            cap_card(&f).await["gateway_capabilities"]["reasoning"]["disabled_supported"],
            true
        );
        let reply = post(
            &f,
            "/v1/chat/completions",
            forced_zero_request(GLM, "/v1/chat/completions"),
        )
        .await;
        if dead_qualified {
            assert!(!reply.status().is_success());
        } else {
            assert!(reply.status().is_success());
            assert_forced_zero_capture(&short, &short);
        }
        assert!(unqualified.state.count_calls.lock().unwrap().is_empty());
        assert_eq!(unqualified.state.calls.load(Ordering::Acquire), 0);
        assert_eq!(long.state.calls.load(Ordering::Acquire), 0);
    }

    let partial = fake(GLM, 100, "good").await;
    let mut partial_worker = worker(&partial, NativeTier::Short);
    qualify_forced_zero_worker(&mut partial_worker, GLM);
    let partial_frontend = cap_frontend(GLM, vec![partial_worker]).await;
    assert_eq!(
        cap_card(&partial_frontend).await["gateway_capabilities"]["reasoning"]["disabled_supported"],
        false
    );
    let reply = post(
        &partial_frontend,
        "/v1/chat/completions",
        forced_zero_request(GLM, "/v1/chat/completions"),
    )
    .await;
    assert_eq!(reply.status().as_u16(), 400);
    let error: Value = reply.json().await.unwrap();
    assert_public_error_envelope("/v1/chat/completions", &error);
    assert_eq!(error["error"]["type"], "invalid_request_error");
    assert!(partial.state.count_calls.lock().unwrap().is_empty());
    assert_eq!(partial.state.calls.load(Ordering::Acquire), 0);

    let qualified = fake(GF, 100, "good").await;
    let mut config = worker(&qualified, NativeTier::Shared);
    qualify_forced_zero_worker(&mut config, GF);
    let qualified_frontend = cap_frontend(GF, vec![config]).await;
    assert_eq!(
        cap_card(&qualified_frontend).await["gateway_capabilities"]["reasoning"]["disabled_supported"],
        true
    );
    for (key, nested) in [
        ("forced_zero_thinking", false),
        ("_forced_zero_thinking_closed", false),
        ("_forced_zero_thinking_closed", true),
        ("forced_zero_thinking", true),
    ] {
        let mut body = chat(GF, false);
        if nested {
            body["chat_template_kwargs"] = json!({key:true});
        } else {
            body[key] = json!(true);
        }
        let reply = post(&qualified_frontend, "/v1/chat/completions", body).await;
        assert!(reply.status().is_client_error(), "{key}, nested={nested}");
        assert_public_error_envelope(
            "/v1/chat/completions",
            &reply.json::<Value>().await.unwrap(),
        );
    }
    assert!(qualified.state.count_calls.lock().unwrap().is_empty());
    assert_eq!(qualified.state.calls.load(Ordering::Acquire), 0);
}

#[tokio::test]
async fn public_qualification_scope_short_budget_works_without_promoting_model_card() {
    let qualified = fake(QWEN, 100, "good").await;
    let unqualified = fake(QWEN, 100, "good").await;
    let mut qualified_worker = worker(&qualified, NativeTier::Short);
    qualify_cap_worker(&mut qualified_worker, QWEN);
    let engine = NativeHttpEngine::new(NativeHttpEngineConfig {
        model: QWEN.into(),
        workers: vec![worker(&unqualified, NativeTier::Short), qualified_worker],
        qualified_model_identity: None,
        qualification_tier: Some(NativeTier::Short),
    })
    .unwrap();
    let f = frontend_from_engine(QWEN, engine, None).await;
    let card = cap_card(&f).await;
    assert_eq!(
        card["gateway_capabilities"]["reasoning"]["fixed_budget_supported"],
        false
    );
    assert_eq!(card["qualification"]["scope"], "single_tier");
    assert_eq!(card["qualification"]["tier"], "short");
    assert_eq!(card["qualification"]["full_model_qualified"], false);
    assert!(f.engine.contract().supports_fixed_thinking_budget());

    let mut body = cap_body(QWEN);
    body["max_tokens"] = json!(128);
    let reply = post(&f, "/v1/chat/completions", body.clone()).await;
    let status = reply.status();
    let wire = reply.text().await.unwrap();
    assert!(status.is_success(), "{status}: {wire}");
    let captured = qualified.state.generation_calls.lock().unwrap()[0].clone();
    let counted = qualified.state.count_calls.lock().unwrap()[0].clone();
    assert_eq!(captured["thinking_token_budget"], 32);
    assert_eq!(captured["max_tokens"], 128);
    assert_eq!(counted["thinking_token_budget"], 32);
    let mut stripped = captured;
    for field in ["max_tokens", "stream", "stream_options"] {
        stripped.as_object_mut().unwrap().remove(field);
    }
    assert_eq!(stripped, counted);
    assert!(unqualified.state.count_calls.lock().unwrap().is_empty());
    assert_eq!(unqualified.state.calls.load(Ordering::Acquire), 0);

    // The same prompt count (100) plus 16,300 requested output exceeds the
    // actual Qwen short threshold 16,384. The private scope cannot route long.
    body["max_tokens"] = json!(16300);
    let reply = post(&f, "/v1/chat/completions", body).await;
    assert_eq!(reply.status().as_u16(), 400);
    assert_public_error_envelope(
        "/v1/chat/completions",
        &reply.json::<Value>().await.unwrap(),
    );
    assert_eq!(
        qualified.state.calls.load(Ordering::Acquire),
        1,
        "cross-tier rejection must add no generation"
    );
    assert_eq!(unqualified.state.calls.load(Ordering::Acquire), 0);
    assert!(unqualified.state.count_calls.lock().unwrap().is_empty());
    assert_eq!(f.engine.inflight(), vec![0, 0]);
}
