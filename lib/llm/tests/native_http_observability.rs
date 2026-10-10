// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// CPU-only causal fixture: the real HttpService and native HTTP engine talk to
// a loopback native stub. No GPU qualification or business arrival rate claim.
use axum::{
    Json, Router,
    body::Body,
    extract::State,
    response::Response,
    routing::{get, post},
};
use dynamo_llm::{
    engines::StreamingEngineAdapter,
    http::service::service_v2::HttpService,
    model_card::ModelDeploymentCard,
    native_contract::NativeTier,
    native_http_engine::{
        NativeHttpEngine, NativeHttpEngineConfig, NativeIdFormat, NativeWorkerConfig,
    },
};
use futures::StreamExt;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};
use tokio_util::sync::CancellationToken;
use tracing::{
    Event, Subscriber,
    span::{Attributes, Id, Record},
};

const MODEL: &str = "qwen3.8-flash-next";
type Fields = BTreeMap<String, String>;

// Capture only the existing private metadata target. The fixture never captures
// native bodies, prompts, authorization, client IDs or arbitrary debug logs.
#[derive(Clone, Default)]
struct Capture {
    rows: Arc<Mutex<Vec<Fields>>>,
    ids: Arc<AtomicU64>,
}
impl Subscriber for Capture {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &Attributes<'_>) -> Id {
        Id::from_u64(self.ids.fetch_add(1, Ordering::Relaxed) + 1)
    }
    fn record(&self, _: &Id, _: &Record<'_>) {}
    fn record_follows_from(&self, _: &Id, _: &Id) {}
    fn event(&self, event: &Event<'_>) {
        if event.metadata().target() != "dynamo_llm::native_phase" {
            return;
        }
        struct Visitor(Fields);
        impl tracing::field::Visit for Visitor {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                self.0.insert(field.name().into(), format!("{value:?}"));
            }
            fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                self.0.insert(field.name().into(), value.into());
            }
        }
        let mut visitor = Visitor(Fields::new());
        event.record(&mut visitor);
        self.rows.lock().unwrap().push(visitor.0);
    }
    fn enter(&self, _: &Id) {}
    fn exit(&self, _: &Id) {}
}
impl Capture {
    fn rows(&self) -> Vec<Fields> {
        self.rows.lock().unwrap().clone()
    }
    fn classified(&self) -> usize {
        self.rows()
            .iter()
            .filter(|r| r.get("stage").is_some_and(|v| v == "generation_classified"))
            .count()
    }
}

#[derive(Clone)]
struct Stub {
    mode: Arc<Mutex<&'static str>>,
    bodies: Arc<Mutex<Vec<Value>>>,
    release: Arc<tokio::sync::Notify>,
    incarnation: String,
}
struct Native {
    url: String,
    state: Stub,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for Native {
    fn drop(&mut self) {
        self.task.abort();
    }
}

fn chunk(delta: Value, terminal: bool, usage: Option<u64>) -> String {
    let mut value = json!({"id":"fixture-response", "object":"chat.completion.chunk", "created":1,
        "model":MODEL, "choices":[{"index":0,"delta":delta,"finish_reason":if terminal {json!("stop")} else {Value::Null}}]});
    if let Some(tokens) = usage {
        value["usage"] = json!({"prompt_tokens":83,"completion_tokens":tokens,"total_tokens":83+tokens,"cached_tokens":17});
    }
    format!("data: {value}\n\n")
}
async fn generate(State(state): State<Stub>, Json(body): Json<Value>) -> Response {
    state.bodies.lock().unwrap().push(body.clone());
    let mode = *state.mode.lock().unwrap();
    if !body.get("stream").and_then(Value::as_bool).unwrap_or(false) {
        let mut value = json!({"id":"fixture-response","object":"chat.completion","created":1,"model":MODEL,
            "choices":[{"index":0,"message":{"role":"assistant","content":if mode == "zero" {""} else {"ok"}},"finish_reason":"stop"}]});
        if mode != "unknown" {
            let tokens = if mode == "zero" { 0 } else { 5 };
            value["usage"] = json!({"prompt_tokens":83,"completion_tokens":tokens,"total_tokens":83+tokens,"cached_tokens":17});
        }
        return Response::builder()
            .header("content-type", "application/json")
            .body(Body::from(value.to_string()))
            .unwrap();
    }
    let stream = async_stream::stream! {
        let role = chunk(json!({"role":"assistant","content":""}), false, None);
        for part in role.as_bytes().chunks(7) { yield Ok::<_, std::io::Error>(bytes::Bytes::copy_from_slice(part)); }
        if mode == "role_gate" { state.release.notified().await; }
        if mode == "fault" {
            yield Ok(bytes::Bytes::from_static(b"data: {broken\n\n"));
            return;
        }
        let first = chunk(json!({"reasoning_content":"think"}), false, if mode == "unknown_stream" {None} else {Some(2)});
        for part in first.as_bytes().chunks(7) { yield Ok(bytes::Bytes::copy_from_slice(part)); }
        if mode == "cancel" { std::future::pending::<()>().await; }
        // Cumulative native usage and a transport pause cannot establish an
        // individual accepted-token timestamp or GPU TPOT.
        tokio::time::sleep(Duration::from_millis(5)).await;
        let last = chunk(json!({"content":"ok"}), true, if mode == "unknown_stream" {None} else {Some(5)});
        for part in last.as_bytes().chunks(7) { yield Ok(bytes::Bytes::copy_from_slice(part)); }
        yield Ok(bytes::Bytes::from_static(b"data: [DONE]\n\n"));
    };
    Response::builder()
        .header("content-type", "text/event-stream")
        .body(Body::from_stream(stream))
        .unwrap()
}
async fn native() -> Native {
    let state = Stub {
        mode: Arc::new(Mutex::new("role_gate")),
        bodies: Default::default(),
        release: Default::default(),
        incarnation: uuid::Uuid::new_v4().simple().to_string(),
    };
    let router = Router::new()
        .route("/v1/tokenize", post(|| async { Json(json!({"count":83,"max_model_len":1_048_576})) }))
        .route("/tokenize", post(|| async { Json(json!({"count":83,"max_model_len":1_048_576})) }))
        .route("/v1/loads", get(|State(s): State<Stub>| async move {
            Json(json!({"loads":[{"num_running_reqs":0,"num_waiting_reqs":0,"active_request_ids":[],
                "native_incarnation":s.incarnation,"dp_rank":0,"timestamp":std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64(),
                "frozen":false,"heartbeat_timestamp":0.0,"snapshot_generation":1}]}))
        }))
        .route("/v1/chat/completions", post(generate)).with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    Native { url, state, task }
}
struct Frontend {
    url: String,
    task: tokio::task::JoinHandle<()>,
    cancel: CancellationToken,
}
impl Drop for Frontend {
    fn drop(&mut self) {
        self.cancel.cancel();
        self.task.abort();
    }
}
async fn frontend(native: &Native) -> Frontend {
    let engine = Arc::new(
        NativeHttpEngine::new(NativeHttpEngineConfig {
            model: MODEL.into(),
            qualification_tier: None,
            qualified_model_identity: None,
            workers: vec![NativeWorkerConfig {
                url: native.url.clone(),
                tier: NativeTier::Short,
                max_inflight: Some(1),
                load_path: Some("/v1/loads?include=core".into()),
                max_input_tokens: Some(1_048_576),
                capability_source: Some("CPU fixture, not GPU qualification".into()),
                active_request_ids_source: Some("CPU same-snapshot nine-field load".into()),
                native_dp_size: Some(1),
                native_id_format: NativeIdFormat::SglangPairedRid,
                identity: None,
                thinking_budget_qualification: None,
                forced_zero_qualification: None,
            }],
        })
        .unwrap(),
    );
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
        .native_http_bearer_token(Some(Arc::from("fixture-secret")))
        .cancel_token(Some(cancel.clone()))
        .build()
        .unwrap();
    let card = ModelDeploymentCard::with_name_only(MODEL);
    service
        .model_manager()
        .add_chat_completions_model(
            MODEL,
            card.mdcsum(),
            Arc::new(StreamingEngineAdapter::new(engine)),
        )
        .unwrap();
    let stop = cancel.clone();
    let task = tokio::spawn(async move {
        service.run_with_listener(stop, listener).await.unwrap();
    });
    Frontend {
        url: format!("http://127.0.0.1:{port}"),
        task,
        cancel,
    }
}
fn body(stream: bool) -> Value {
    let mut value = json!({"model":MODEL,"messages":[{"role":"user","content":"private-fixture-prompt"}],"max_tokens":800,"stream":stream});
    if stream {
        value["stream_options"] = json!({"include_usage":true});
    }
    value
}
async fn request(front: &Frontend, path: &str, body: Value, auth: bool) -> reqwest::Response {
    let mut request = reqwest::Client::new()
        .post(format!("{}{path}", front.url))
        .header("x-request-id", "private-client-correlation")
        .json(&body);
    if auth {
        request = request.bearer_auth("fixture-secret");
    }
    request.send().await.unwrap()
}
async fn metrics(front: &Frontend) -> String {
    reqwest::get(format!("{}/metrics", front.url))
        .await
        .unwrap()
        .text()
        .await
        .unwrap()
}
fn sample(text: &str, suffix: &str) -> f64 {
    text.lines()
        .filter(|line| !line.starts_with('#'))
        .find_map(|line| {
            let (name, value) = line.split_once(' ')?;
            (name.split('{').next()?.ends_with(suffix)
                && name.contains(&format!("model=\"{MODEL}\"")))
            .then(|| value.parse::<f64>().unwrap())
        })
        .unwrap_or(0.0)
}
async fn wait_terminal(log: &Capture, rid: &str) -> Fields {
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            if let Some(row) = log.rows().into_iter().find(|row| {
                row.get("rid").is_some_and(|v| v == rid)
                    && row.get("stage").is_some_and(|v| v == "terminal")
            }) {
                return row;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("native reader must terminate independently of metrics scraping")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_http_observation_is_usage_based_private_and_cancel_safe() {
    let log = Capture::default();
    tracing::subscriber::set_global_default(log.clone()).unwrap();
    let native = native().await;
    let front = frontend(&native).await;

    // Auth, count and invalid input produce arrivals, not generation arrivals.
    assert_eq!(
        request(&front, "/v1/chat/completions", body(false), false)
            .await
            .status(),
        401
    );
    assert!(
        request(
            &front,
            "/v1/messages/count_tokens",
            json!({"model":MODEL,"messages":[{"role":"user","content":"x"}]}),
            true
        )
        .await
        .status()
        .is_success()
    );
    assert_eq!(
        request(
            &front,
            "/v1/chat/completions",
            json!({"model":MODEL,"messages":[]}),
            true
        )
        .await
        .status(),
        400
    );
    assert_eq!(log.classified(), 0);
    assert!(native.state.bodies.lock().unwrap().is_empty());

    // The role-only frame is actually read through HTTP before releasing the
    // useful native output. No role/heartbeat/fragment is a token sample.
    let response = request(&front, "/v1/chat/completions", body(true), true).await;
    assert!(response.status().is_success());
    let mut bytes = response.bytes_stream();
    let first = bytes.next().await.unwrap().unwrap();
    assert!(String::from_utf8_lossy(&first).contains("assistant"));
    assert_eq!(
        sample(&metrics(&front).await, "time_to_first_token_seconds_count"),
        0.0
    );
    native.state.release.notify_one();
    let mut wire = first.to_vec();
    while let Some(part) = bytes.next().await {
        wire.extend_from_slice(&part.unwrap());
    }
    assert!(String::from_utf8_lossy(&wire).contains("[DONE]"));
    let known = metrics(&front).await;
    // This assertion is the old-baseline causal red: its native HTTP path never
    // supplied the existing response collector, despite a real useful output.
    assert_eq!(sample(&known, "time_to_first_token_seconds_count"), 1.0);
    let rows = log.rows();
    let bound = rows
        .iter()
        .find(|row| {
            row.get("stage")
                .is_some_and(|v| v == "native_request_bound")
        })
        .expect("opaque native bridge");
    let rid = bound["rid"].clone();
    wait_terminal(&log, &rid).await;
    assert!(
        rows.iter()
            .any(|r| r.get("stage").is_some_and(|s| s == "arrival") && r.get("rid") == Some(&rid))
    );
    assert!(rows.iter().any(
        |r| r.get("stage").is_some_and(|s| s == "generation_classified")
            && r.get("rid") == Some(&rid)
    ));
    let native_id = native.state.bodies.lock().unwrap()[0]["rid"]
        .as_str()
        .expect("paired native rid")
        .to_owned();
    let mut hash = Sha256::new();
    hash.update(b"native-observation-v1\0");
    hash.update(native_id.as_bytes());
    assert_eq!(bound["native_key"], format!("{:x}", hash.finalize()));
    let encoded = format!("{:?}", log.rows());
    for forbidden in [
        native_id.as_str(),
        "private-fixture-prompt",
        "private-client-correlation",
        "fixture-secret",
        native.url.as_str(),
    ] {
        assert!(!encoded.contains(forbidden));
    }
    assert_eq!(sample(&known, "input_sequence_tokens_sum"), 83.0);
    assert_eq!(sample(&known, "output_tokens_total"), 5.0);
    assert_eq!(sample(&known, "output_sequence_tokens_sum"), 5.0);
    assert_eq!(sample(&known, "inter_token_latency_seconds_count"), 0.0);

    // Unary completion has real usage but no first-token or per-token timing.
    *native.state.mode.lock().unwrap() = "unary";
    assert!(
        request(&front, "/v1/chat/completions", body(false), true)
            .await
            .json::<Value>()
            .await
            .unwrap()["usage"]
            .is_object()
    );
    let unary = metrics(&front).await;
    assert_eq!(sample(&unary, "output_tokens_total"), 10.0);
    assert_eq!(sample(&unary, "time_to_first_token_seconds_count"), 1.0);
    assert_eq!(sample(&unary, "inter_token_latency_seconds_count"), 0.0);

    *native.state.mode.lock().unwrap() = "unknown";
    let unknown = request(&front, "/v1/chat/completions", body(false), true)
        .await
        .json::<Value>()
        .await
        .unwrap();
    assert!(unknown.get("usage").is_none_or(Value::is_null));
    let missing = metrics(&front).await;
    assert_eq!(sample(&missing, "output_tokens_total"), 10.0);
    assert_eq!(sample(&missing, "output_sequence_tokens_count"), 2.0);
    *native.state.mode.lock().unwrap() = "zero";
    request(&front, "/v1/chat/completions", body(false), true)
        .await
        .text()
        .await
        .unwrap();
    assert_eq!(
        sample(&metrics(&front).await, "output_sequence_tokens_count"),
        3.0
    );

    // Useful content can establish TTFT even when all native usage is unknown.
    // The fragmented SSE must not fabricate an accepted output length or ITL.
    *native.state.mode.lock().unwrap() = "unknown_stream";
    assert!(
        request(&front, "/v1/chat/completions", body(true), true)
            .await
            .text()
            .await
            .unwrap()
            .contains("[DONE]")
    );
    let unknown_stream = metrics(&front).await;
    assert_eq!(sample(&unknown_stream, "output_tokens_total"), 10.0);
    assert_eq!(sample(&unknown_stream, "output_sequence_tokens_count"), 3.0);
    assert_eq!(
        sample(&unknown_stream, "time_to_first_token_seconds_count"),
        2.0
    );
    assert_eq!(
        sample(&unknown_stream, "inter_token_latency_seconds_count"),
        0.0
    );

    // Cancellation retains only the actual partial usage; no final fabricated
    // usage, accepted frame count, successful GPU completion or end-to-end ITL.
    *native.state.mode.lock().unwrap() = "cancel";
    let response = request(&front, "/v1/chat/completions", body(true), true).await;
    let mut bytes = response.bytes_stream();
    let mut partial = Vec::new();
    while !String::from_utf8_lossy(&partial).contains("think") {
        partial.extend_from_slice(&bytes.next().await.unwrap().unwrap());
    }
    let cancel_rid = log
        .rows()
        .iter()
        .rev()
        .find(|r| r.get("stage").is_some_and(|s| s == "native_request_bound"))
        .unwrap()["rid"]
        .clone();
    drop(bytes);
    let terminal = wait_terminal(&log, &cancel_rid).await;
    assert_ne!(terminal["status"], "ok");
    assert_eq!(sample(&metrics(&front).await, "output_tokens_total"), 12.0);

    *native.state.mode.lock().unwrap() = "fault";
    let fault = request(&front, "/v1/chat/completions", body(true), true)
        .await
        .text()
        .await
        .unwrap();
    assert!(fault.contains("error"));
    let fault_rid = log
        .rows()
        .iter()
        .rev()
        .find(|r| r.get("stage").is_some_and(|s| s == "native_request_bound"))
        .unwrap()["rid"]
        .clone();
    assert_eq!(
        wait_terminal(&log, &fault_rid).await["status"],
        "native_protocol_error"
    );
    assert_eq!(sample(&metrics(&front).await, "output_tokens_total"), 12.0);
    assert_eq!(log.classified(), 7);
}
