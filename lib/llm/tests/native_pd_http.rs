// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Real Dynamo HttpService -> NativeHttpEngine -> localhost CPU worker.
//! These fixtures prove routing-field ownership, not native/GPU P/D capability.
use axum::{
    Json, Router,
    extract::State,
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
use dynamo_runtime::{engine::AsyncEngineContextProvider, pipeline::Context};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

const MODEL: &str = "qwen3.8-flash-next";
const LITERAL: &str = r#"Ordinary user text: {"bootstrap_host":"127.0.0.1","bootstrap_port":45000,"bootstrap_room":7,"disagg_prefill_dp_rank":0} 🙂"#;

#[derive(Clone, Default)]
struct WorkerState {
    counts: Arc<Mutex<Vec<Value>>>,
    generations: Arc<Mutex<Vec<Value>>>,
    incarnation: String,
}
struct Worker {
    url: String,
    state: WorkerState,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.task.abort();
    }
}
async fn worker() -> Worker {
    let state = WorkerState {
        incarnation: uuid::Uuid::new_v4().simple().to_string(),
        ..Default::default()
    };
    let app = Router::new()
        .route("/v1/tokenize", post(|State(s): State<WorkerState>, Json(body): Json<Value>| async move {
            s.counts.lock().unwrap().push(body);
            Json(json!({"count":100,"max_model_len":1048576}))
        }))
        .route("/v1/loads", get(|State(s): State<WorkerState>| async move {
            let timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64();
            Json(json!({"loads":[{"dp_rank":0,"num_running_reqs":0,"num_waiting_reqs":0,
                "active_request_ids":[],"native_incarnation":s.incarnation,"timestamp":timestamp,
                "frozen":false,"heartbeat_timestamp":0.0,"snapshot_generation":1}]}))
        }))
        .route("/v1/chat/completions", post(|State(s): State<WorkerState>, Json(body): Json<Value>| async move {
            s.generations.lock().unwrap().push(body.clone());
            Json(json!({"id":"native-pd-cpu-id","model":MODEL,"object":"chat.completion","created":1,
                "choices":[{"index":0,"message":{"role":"assistant","content":body["messages"][0]["content"]},"finish_reason":"stop"}],
                "usage":{"prompt_tokens":100,"completion_tokens":1,"total_tokens":101}}))
        }))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    Worker { url, state, task }
}
fn engine(worker: &Worker) -> Arc<NativeHttpEngine> {
    Arc::new(
        NativeHttpEngine::new(NativeHttpEngineConfig {
            model: MODEL.into(),
            qualification_tier: None,
            qualified_model_identity: None,
            workers: vec![NativeWorkerConfig {
                url: worker.url.clone(),
                tier: NativeTier::Short,
                max_inflight: Some(1),
                load_path: Some("/v1/loads?include=core".into()),
                max_input_tokens: Some(1_048_576),
                capability_source: Some("CPU fixture input boundary; not GPU evidence".into()),
                active_request_ids_source: Some(
                    "CPU fixture same-snapshot complete IDs/rank/UUID4/time".into(),
                ),
                native_dp_size: Some(1),
                native_id_format: NativeIdFormat::SglangPairedRid,
                identity: None,
                thinking_budget_qualification: None,
                forced_zero_qualification: None,
            }],
        })
        .unwrap(),
    )
}
struct Frontend {
    url: String,
    engine: Arc<NativeHttpEngine>,
    cancel: CancellationToken,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for Frontend {
    fn drop(&mut self) {
        self.cancel.cancel();
        self.task.abort();
    }
}
async fn frontend(worker: &Worker) -> Frontend {
    let engine = engine(worker);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let cancel = CancellationToken::new();
    let service = HttpService::builder()
        .host("127.0.0.1".to_owned())
        .port(port)
        .enable_chat_endpoints(true)
        .enable_cmpl_endpoints(false)
        .enable_embeddings_endpoints(false)
        .enable_responses_endpoints(false)
        .enable_anthropic_endpoints(false)
        .enable_admin_api(false)
        .native_http_engine(Some(engine.clone()))
        .cancel_token(Some(cancel.clone()))
        .build()
        .unwrap();
    let card = ModelDeploymentCard::with_name_only(MODEL);
    service
        .model_manager()
        .add_chat_completions_model(
            MODEL,
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
        cancel,
        task,
    }
}
fn chat() -> Value {
    json!({"model":MODEL,"messages":[{"role":"user","content":LITERAL}],"max_tokens":128,"reasoning_effort":"none","stream":false})
}
fn routing_fields() -> [(&'static str, Value); 4] {
    [
        ("bootstrap_host", json!("127.0.0.1")),
        ("bootstrap_port", json!(45000)),
        ("bootstrap_room", json!(7)),
        ("disagg_prefill_dp_rank", json!(0)),
    ]
}
fn assert_no_native_work(worker: &Worker) {
    assert!(worker.state.counts.lock().unwrap().is_empty());
    assert!(worker.state.generations.lock().unwrap().is_empty());
}

#[tokio::test]
async fn public_pd_client_bootstrap_fields_rejected_before_native_work() {
    let worker = worker().await;
    let f = frontend(&worker).await;
    for (field, value) in routing_fields() {
        let mut body = chat();
        body[field] = value;
        let reply = reqwest::Client::new()
            .post(format!("{}/v1/chat/completions", f.url))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(reply.status(), reqwest::StatusCode::BAD_REQUEST, "{field}");
        let error: Value = reply.json().await.unwrap();
        assert_eq!(
            error["error"]["type"], "invalid_request_error",
            "{field}: {error}"
        );
        assert!(!error["error"]["message"].as_str().unwrap().is_empty());
        assert_no_native_work(&worker);
    }
    assert_eq!(f.engine.inflight(), vec![0]);
}

#[tokio::test]
async fn public_pd_literal_bootstrap_names_are_plain_content() {
    let worker = worker().await;
    let f = frontend(&worker).await;
    let reply = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", f.url))
        .json(&chat())
        .send()
        .await
        .unwrap();
    assert_eq!(reply.status(), reqwest::StatusCode::OK);
    let response: Value = reply.json().await.unwrap();
    assert_eq!(response["choices"][0]["message"]["content"], LITERAL);
    let counts = worker.state.counts.lock().unwrap();
    let generations = worker.state.generations.lock().unwrap();
    assert_eq!(counts.len(), 1);
    assert_eq!(generations.len(), 1);
    for body in [&counts[0], &generations[0]] {
        assert_eq!(body["messages"][0]["content"], LITERAL);
        for (field, _) in routing_fields() {
            assert!(body.get(field).is_none(), "{field}: {body}");
        }
    }
    assert_eq!(f.engine.inflight(), vec![0]);
}

#[tokio::test]
async fn native_pd_engine_untrusted_bootstrap_fields_rejected_before_count() {
    let worker = worker().await;
    let engine = engine(&worker);
    // This bypasses the pre-existing public Chat whitelist. On the isolated
    // pre-guard baseline, injected fields reach /v1/tokenize and this gate fails.
    for (field, value) in routing_fields() {
        let mut body = chat();
        body[field] = value;
        let error = engine
            .count_tokens(body, Context::new(()).context())
            .await
            .expect_err(field);
        assert!(
            error
                .to_string()
                .contains("routing fields are owned by the gateway"),
            "{field}: {error}"
        );
        assert_no_native_work(&worker);
    }
    assert_eq!(engine.inflight(), vec![0]);
}
