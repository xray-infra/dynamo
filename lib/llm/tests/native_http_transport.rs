// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::{
    Json, Router,
    body::Body,
    extract::State,
    response::Response,
    routing::{get, post},
};
use dynamo_llm::{
    engines::StreamingEngine,
    native_contract::NativeTier,
    native_http_engine::{
        NATIVE_HTTP_BODY, NativeHttpEngine, NativeHttpEngineConfig, NativeIdFormat,
        NativeWorkerConfig,
    },
};
use dynamo_runtime::{
    engine::{AsyncEngineContext, AsyncEngineContextProvider},
    pipeline::Context,
};
use futures::StreamExt;
use serde_json::{Value, json};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Duration;

const MODEL: &str = "qwen3.8-flash-next";

#[derive(Clone)]
struct Fake {
    count: u64,
    load: u64,
    mode: &'static str,
    generation_calls: Arc<AtomicUsize>,
    counts: Arc<Mutex<Vec<Value>>>,
    generations: Arc<Mutex<Vec<Value>>>,
    load_override: Arc<Mutex<Option<Value>>>,
    incarnation: String,
}

struct Server {
    url: String,
    state: Fake,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn fake(count: u64, load: u64, mode: &'static str) -> Server {
    let state = Fake {
        count,
        load,
        mode,
        generation_calls: Arc::new(AtomicUsize::new(0)),
        counts: Default::default(),
        generations: Default::default(),
        load_override: Default::default(),
        incarnation: uuid::Uuid::new_v4().simple().to_string(),
    };
    let app = Router::new().route("/v1/tokenize", post(|State(s): State<Fake>, Json(v): Json<Value>| async move {
        s.counts.lock().unwrap().push(v); Json(json!({"count":s.count,"max_model_len":1048576}))
    })).route("/v1/loads", get(|State(s):State<Fake>| async move {
        if let Some(value) = s.load_override.lock().unwrap().clone() { return Json(value); }
        if s.mode == "bad_load" { return Json(json!({"loads":[{"dp_rank":0,"frozen":false,"heartbeat_timestamp":0.0,"snapshot_generation":1,"num_running_reqs":false,"num_waiting_reqs":0}]})); }
        Json(json!({"loads":[{"dp_rank":0,"frozen":false,"heartbeat_timestamp":0.0,"snapshot_generation":1,"num_running_reqs":s.load,"num_waiting_reqs":0,"active_request_ids":[],"native_incarnation":s.incarnation,"timestamp":std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64()}]}))
    })).route("/v1/chat/completions", post(|State(s): State<Fake>, Json(v):Json<Value>| async move {
        s.generation_calls.fetch_add(1, Ordering::AcqRel); s.generations.lock().unwrap().push(v.clone());
        if !v["stream"].as_bool().unwrap_or(false) {
            return Response::builder().header("content-type","application/json").body(Body::from(json!({
                "id":"native-id","model":v["model"],"object":"chat.completion","created":1,
                "choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
                "usage":{"prompt_tokens":s.count,"completion_tokens":1,"total_tokens":s.count+1,"cached_tokens":17,"opaque_native":{"field":false}}
            }).to_string())).unwrap();
        }
        let terminal = json!({"id":"native-id","model":v["model"],"object":"chat.completion.chunk","created":1,
            "choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}],"usage":{"cached_tokens":17}}).to_string();
        let content = json!({"id":"native-id","model":v["model"],"object":"chat.completion.chunk","created":1,
            "choices":[{"index":0,"delta":{"content":"🙂"},"finish_reason":null}]}).to_string();
        let wire = match s.mode {
            "malformed" => "data: {bad}\n\ndata: [DONE]\n\n".to_string(),
            "partial" => format!("data: {content}\n\ndata: [DONE]\n\n"),
            "no_done" => format!("data: {terminal}\n\n"),
            "after_terminal" => format!("data: {terminal}\n\ndata: {content}\n\ndata: [DONE]\n\n"),
            "n2" => {
                let mut second: Value = serde_json::from_str(&terminal).unwrap();
                second["choices"][0]["index"] = 1.into();
                format!("data: {content}\n\ndata: {terminal}\n\ndata: {second}\n\ndata: [DONE]\n\n")
            },
            _ => format!(": heartbeat\r\n\r\ndata: {content}\r\n\r\ndata: {terminal}\r\n\r\ndata: [DONE]\r\n\r\n"),
        };
        let mode = s.mode;
        let chunks = async_stream::stream! {
            if mode == "hold" {
                yield Ok::<_, std::io::Error>(bytes::Bytes::from(format!("data: {content}\n\n")));
                std::future::pending::<()>().await;
            } else {
                // Deliberately split UTF-8 and CRLF boundaries in separate reads.
                for chunk in wire.as_bytes().chunks(7) { yield Ok::<_, std::io::Error>(bytes::Bytes::copy_from_slice(chunk)); }
            }
        };
        Response::builder().header("content-type", "text/event-stream").body(Body::from_stream(chunks)).unwrap()
    })).with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    Server { url, state, task }
}

fn worker(server: &Server, tier: NativeTier) -> NativeWorkerConfig {
    NativeWorkerConfig {
        url: server.url.clone(),
        identity: None,
        thinking_budget_qualification: None,
        forced_zero_qualification: None,
        tier,
        max_inflight: Some(1),
        load_path: Some("/v1/loads?include=core".into()),
        max_input_tokens: Some(1_048_576),
        capability_source: Some("CPU fake worker fixture native input capability".into()),
        active_request_ids_source: Some("CPU fake same-snapshot native request IDs".into()),
        native_dp_size: Some(1),
        native_id_format: NativeIdFormat::SglangPairedRid,
    }
}

fn engine(workers: Vec<NativeWorkerConfig>) -> NativeHttpEngine {
    NativeHttpEngine::new(NativeHttpEngineConfig {
        model: MODEL.into(),
        workers,
        qualified_model_identity: None,
        qualification_tier: None,
    })
    .unwrap()
}

fn body(stream: bool) -> Value {
    json!({"model":MODEL,"messages":[{"role":"user","content":"test"}],"max_tokens":800,"stream":stream,"reasoning_effort":"none","cache_salt":"transport-test"})
}

fn ctx() -> Arc<dyn AsyncEngineContext> {
    Context::new(()).context()
}

#[tokio::test]
async fn raw_json_keeps_native_usage_and_canonical_generation_count_agreement() {
    let server = fake(100, 0, "good").await;
    let engine = engine(vec![worker(&server, NativeTier::Short)]);
    let mut output = engine.forward_chat(body(false), ctx()).await.unwrap();
    assert_eq!(engine.inflight(), vec![1]);
    let raw = output.next().await.unwrap().data.unwrap();
    assert_eq!(raw["usage"]["cached_tokens"], 17);
    assert_eq!(raw["usage"]["opaque_native"]["field"], false);
    assert!(output.next().await.is_none());
    assert_eq!(engine.inflight(), vec![0]);
    let count = server.state.counts.lock().unwrap()[0].clone();
    let mut generated = server.state.generations.lock().unwrap()[0].clone();
    for field in [
        "max_tokens",
        "max_completion_tokens",
        "max_output_tokens",
        "stream",
        "stream_options",
    ] {
        generated.as_object_mut().unwrap().remove(field);
    }
    assert_eq!(count, generated);
    assert_eq!(count["cache_salt"], "transport-test");
    assert_eq!(count["chat_template_kwargs"]["enable_thinking"], false);
}

#[tokio::test]
async fn tier_filter_precedes_native_load_and_output_budget_flips_tier() {
    let short = fake(16000, 0, "good").await;
    let long_busy = fake(16000, 20, "good").await;
    let long_free = fake(16000, 0, "good").await;
    let engine = engine(vec![
        worker(&short, NativeTier::Short),
        worker(&long_busy, NativeTier::Long),
        worker(&long_free, NativeTier::Long),
    ]);
    let raw = engine
        .forward_chat(body(false), ctx())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(raw[0].error.is_none());
    assert_eq!(short.state.generation_calls.load(Ordering::Acquire), 0);
    assert_eq!(long_busy.state.generation_calls.load(Ordering::Acquire), 0);
    assert_eq!(long_free.state.generation_calls.load(Ordering::Acquire), 1);
    assert_eq!(engine.inflight(), vec![0, 0, 0]);
}

#[tokio::test]
async fn fragmented_sse_preserves_unicode_usage_and_terminal() {
    let server = fake(100, 0, "good").await;
    let engine = engine(vec![worker(&server, NativeTier::Short)]);
    let output = engine
        .forward_chat(body(true), ctx())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert_eq!(output.len(), 2);
    assert!(output.iter().all(|p| p.error.is_none()));
    assert_eq!(
        output[0].data.as_ref().unwrap()["choices"][0]["delta"]["content"],
        "🙂"
    );
    assert_eq!(
        output[1].data.as_ref().unwrap()["usage"]["cached_tokens"],
        17
    );
    assert_eq!(engine.inflight(), vec![0]);
}

#[tokio::test]
async fn malformed_and_partial_streams_fail_without_replay_or_fake_terminal() {
    for mode in ["malformed", "partial", "no_done", "after_terminal"] {
        let server = fake(100, 0, mode).await;
        let engine = engine(vec![worker(&server, NativeTier::Short)]);
        let output = engine
            .forward_chat(body(true), ctx())
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(output.last().unwrap().error.is_some(), "{mode}");
        assert_eq!(
            server.state.generation_calls.load(Ordering::Acquire),
            1,
            "{mode}"
        );
        assert_eq!(engine.inflight(), vec![0], "{mode}");
    }
}

#[tokio::test]
async fn drop_and_explicit_cancel_release_reservation_even_before_first_poll() {
    let server = fake(100, 0, "hold").await;
    let engine = engine(vec![worker(&server, NativeTier::Short)]);
    let output = engine.forward_chat(body(true), ctx()).await.unwrap();
    assert_eq!(engine.inflight(), vec![1]);
    let saturated = engine.forward_chat(body(false), ctx()).await.unwrap_err();
    assert!(dynamo_runtime::error::match_error_chain(
        saturated.as_ref(),
        &[dynamo_runtime::error::ErrorType::ResourceExhausted],
        &[]
    ));
    assert_eq!(server.state.generation_calls.load(Ordering::Acquire), 1);
    drop(output);
    assert_eq!(engine.inflight(), vec![0]);
    let context = ctx();
    let mut output = engine
        .forward_chat(body(true), context.clone())
        .await
        .unwrap();
    assert!(output.next().await.unwrap().data.is_some());
    context.stop();
    assert!(
        tokio::time::timeout(std::time::Duration::from_secs(1), output.next())
            .await
            .unwrap()
            .is_none()
    );
    assert_eq!(engine.inflight(), vec![0]);
}

#[tokio::test]
async fn over_context_and_missing_raw_context_do_not_generate() {
    let server = fake(1048576, 0, "good").await;
    let engine = engine(vec![worker(&server, NativeTier::Long)]);
    assert!(engine.forward_chat(body(false), ctx()).await.is_err());
    assert_eq!(server.state.generation_calls.load(Ordering::Acquire), 0);
    let typed: dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest =
        serde_json::from_value(body(false)).unwrap();
    assert!(engine.handle_chat(Context::new(typed)).await.is_err());
    // The typed trait path must also use the preserved body when provided.
    let typed: dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest =
        serde_json::from_value(body(false)).unwrap();
    let mut request = Context::new(typed);
    request.insert(NATIVE_HTTP_BODY, body(false));
    assert!(engine.handle_chat(request).await.is_err()); // Native count exceeds context.
    assert_eq!(server.state.generation_calls.load(Ordering::Acquire), 0);
}

#[tokio::test]
async fn native_input_boundary_filters_before_dispatch_and_short_can_select_long() {
    let short = fake(800, 0, "good").await;
    let long = fake(800, 0, "good").await;
    let mut constrained = worker(&short, NativeTier::Short);
    constrained.max_input_tokens = Some(799);
    let engine = engine(vec![constrained, worker(&long, NativeTier::Long)]);
    let output = engine
        .forward_chat(body(false), ctx())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(output[0].error.is_none());
    assert_eq!(short.state.generation_calls.load(Ordering::Acquire), 0);
    assert_eq!(long.state.generation_calls.load(Ordering::Acquire), 1);
    drop(engine);
    let constrained_long = fake(16000, 0, "good").await;
    let mut constrained = worker(&constrained_long, NativeTier::Long);
    constrained.max_input_tokens = Some(15999);
    let engine = NativeHttpEngine::new(NativeHttpEngineConfig {
        model: MODEL.into(),
        workers: vec![constrained],
        qualified_model_identity: None,
        qualification_tier: None,
    })
    .unwrap();
    assert!(engine.forward_chat(body(false), ctx()).await.is_err());
    assert_eq!(
        constrained_long
            .state
            .generation_calls
            .load(Ordering::Acquire),
        0
    );
    assert_eq!(engine.inflight(), vec![0]);
}

#[tokio::test]
async fn deepseek_missing_authority_is_fail_closed_before_native_count_or_generation() {
    let server = fake(100, 0, "good").await;
    let mut ds_worker = worker(&server, NativeTier::Short);
    ds_worker.max_inflight = None;
    let engine = NativeHttpEngine::new(NativeHttpEngineConfig {
        model: "deepseek-ai/DeepSeek-V4.1-Flash".into(),
        workers: vec![ds_worker],
        qualified_model_identity: Some("CPU fixture identity only".into()),
        qualification_tier: None,
    })
    .unwrap();
    let mut body = body(false);
    body["model"] = "deepseek-ai/DeepSeek-V4.1-Flash".into();
    assert!(engine.forward_chat(body, ctx()).await.is_err());
    assert!(server.state.counts.lock().unwrap().is_empty());
    assert_eq!(server.state.generation_calls.load(Ordering::Acquire), 0);
}

#[tokio::test]
async fn invalid_native_load_fails_unavailable_without_dispatch() {
    let server = fake(100, 0, "bad_load").await;
    let engine = engine(vec![worker(&server, NativeTier::Short)]);
    let error = engine.forward_chat(body(false), ctx()).await.unwrap_err();
    assert!(dynamo_runtime::error::match_error_chain(
        error.as_ref(),
        &[dynamo_runtime::error::ErrorType::Unavailable],
        &[]
    ));
    assert_eq!(server.state.generation_calls.load(Ordering::Acquire), 0);
    assert_eq!(engine.inflight(), vec![0]);
}

#[tokio::test]
async fn n_rejects_invalid_types_before_count_and_each_native_choice_needs_terminal() {
    let server = fake(100, 0, "n2").await;
    let engine = engine(vec![worker(&server, NativeTier::Short)]);
    for invalid in [
        json!(true),
        json!(0),
        json!(-1),
        json!(1.5),
        Value::Null,
        json!(u32::MAX as u64 + 1),
    ] {
        let mut request = body(true);
        request["n"] = invalid;
        assert!(engine.forward_chat(request, ctx()).await.is_err());
    }
    assert!(server.state.counts.lock().unwrap().is_empty());
    assert_eq!(server.state.generation_calls.load(Ordering::Acquire), 0);
    let mut request = body(true);
    request["n"] = 2.into();
    let chunks = engine
        .forward_chat(request, ctx())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert_eq!(chunks.len(), 3);
    assert!(chunks.iter().all(|chunk| chunk.error.is_none()));
    assert_eq!(chunks[1].data.as_ref().unwrap()["choices"][0]["index"], 0);
    assert_eq!(chunks[2].data.as_ref().unwrap()["choices"][0]["index"], 1);
    assert_eq!(engine.inflight(), vec![0]);
}

struct IdentityTransportStore;
#[async_trait::async_trait]
impl dynamo_llm::native_fencing::FenceStore for IdentityTransportStore {
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
            enabled: true,
            commit_id: "cpu-transport-gate-identity".into(),
            revision: 2,
            mod_revision: 1,
        })
    }
}

struct IdentityPrefixStore;
#[async_trait::async_trait]
impl dynamo_llm::native_fencing::prefix_records::PrefixStore for IdentityPrefixStore {
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

async fn ds_identity_engine(
    servers: &[Server],
) -> (
    NativeHttpEngine,
    Arc<dynamo_llm::native_fencing::NativeFence>,
) {
    let fence = dynamo_llm::native_fencing::NativeFence::new(
        dynamo_llm::native_fencing::FenceConfig {
            election_key: "/cpu-transport/native-identity".into(),
            holder: "cpu-test".into(),
            short_workers: servers.iter().map(|server| server.url.clone()).collect(),
            max_effective_load: 2,
        },
        Arc::new(IdentityTransportStore),
        Arc::new(IdentityPrefixStore),
    )
    .unwrap();
    fence.refresh_authority().await.unwrap();
    fence.start_prefixes();
    tokio::time::timeout(Duration::from_secs(1), async {
        while !fence.prefixes_ready() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let workers = servers
        .iter()
        .map(|server| {
            let mut config = worker(server, NativeTier::Short);
            config.max_inflight = None;
            config
        })
        .collect();
    let engine = NativeHttpEngine::new(NativeHttpEngineConfig {
        model: "deepseek-ai/DeepSeek-V4.1-Flash".into(),
        workers,
        qualified_model_identity: Some("CPU fixture identity only".into()),
        qualification_tier: None,
    })
    .unwrap()
    .with_fence(fence.clone());
    (engine, fence)
}

fn ds_body(n: u32) -> Value {
    let mut request = body(true);
    request["model"] = "deepseek-ai/DeepSeek-V4.1-Flash".into();
    request["n"] = n.into();
    request
}

fn identity_snapshot(ids: Value, running: usize, timestamp: f64, incarnation: &str) -> Value {
    json!({"loads":[{"dp_rank":0,"frozen":false,"heartbeat_timestamp":0.0,"snapshot_generation":1,"num_running_reqs":running,"num_waiting_reqs":0,"active_request_ids":ids,"timestamp":timestamp,"native_incarnation":incarnation}]})
}

fn wall_time() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

#[tokio::test]
async fn ds_owned_rid_exact_partial_samples_release_weight_without_prefix_inference() {
    let mut servers = Vec::new();
    for _ in 0..12 {
        servers.push(fake(100, 0, "hold").await);
    }
    let (engine, fence) = ds_identity_engine(&servers).await;
    let output = engine.forward_chat(ds_body(2), ctx()).await.unwrap();
    let selected = servers
        .iter()
        .find(|server| server.state.generation_calls.load(Ordering::Acquire) == 1)
        .unwrap();
    let native_body = selected.state.generations.lock().unwrap()[0].clone();
    let rid = native_body["rid"].as_str().unwrap();
    assert_eq!(rid.len(), 32);
    assert!(rid.bytes().all(|byte| byte.is_ascii_hexdigit()));
    assert_eq!(native_body["n"], 2);
    assert_eq!(fence.effective_load(&selected.url), Some(2)); // Headers do not remove pending weight.
    *selected.state.load_override.lock().unwrap() = Some(identity_snapshot(
        json!([format!("{rid}_0:sample:0")]),
        1,
        wall_time(),
        &selected.state.incarnation,
    ));
    engine.refresh_short_loads(&ctx()).await.unwrap();
    assert_eq!(fence.effective_load(&selected.url), Some(2)); // native1 + pending1, no double count.
    *selected.state.load_override.lock().unwrap() = Some(identity_snapshot(
        json!([format!("{rid}_0:sample:00")]),
        1,
        wall_time(),
        &selected.state.incarnation,
    ));
    engine.refresh_short_loads(&ctx()).await.unwrap();
    assert_eq!(fence.effective_load(&selected.url), Some(3)); // Prefix / noncanonical index cannot acknowledge.
    *selected.state.load_override.lock().unwrap() = Some(identity_snapshot(
        json!([format!("{rid}_0:sample:0"), format!("{rid}_0:sample:1")]),
        2,
        wall_time(),
        &selected.state.incarnation,
    ));
    engine.refresh_short_loads(&ctx()).await.unwrap();
    assert_eq!(fence.effective_load(&selected.url), Some(2));
    assert_eq!(fence.reservation_count(), 1); // Native acceptance keeps drain lifecycle entry.
    drop(output);
    assert_eq!(fence.reservation_count(), 0);
    assert!(engine.inflight().iter().all(|count| *count == 0));
}

#[tokio::test]
async fn ds_missing_duplicate_mismatched_or_stale_snapshot_ids_fail_before_generation() {
    let mut servers = Vec::new();
    for _ in 0..12 {
        servers.push(fake(100, 0, "hold").await);
    }
    let (engine, _) = ds_identity_engine(&servers).await;
    let cases = [
        identity_snapshot(Value::Null, 0, wall_time(), &servers[0].state.incarnation),
        identity_snapshot(
            json!(["duplicate", "duplicate"]),
            2,
            wall_time(),
            &servers[0].state.incarnation,
        ),
        identity_snapshot(json!([]), 1, wall_time(), &servers[0].state.incarnation),
        identity_snapshot(
            json!([]),
            0,
            wall_time() - 10.0,
            &servers[0].state.incarnation,
        ),
        identity_snapshot(
            Value::Null, // Producer reports IDs null if the complete MsgPack snapshot overflows.
            1,
            wall_time(),
            &servers[0].state.incarnation,
        ),
        identity_snapshot(json!([]), 0, wall_time(), ""),
    ];
    for snapshot in cases {
        *servers[0].state.load_override.lock().unwrap() = Some(snapshot);
        assert!(engine.forward_chat(ds_body(1), ctx()).await.is_err());
        assert!(
            servers
                .iter()
                .all(|server| server.state.generation_calls.load(Ordering::Acquire) == 0)
        );
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum HaFault {
    CounterUnavailable,
    CounterRejected,
    CloseListenerAfterLoad,
    CloseAfterPartialBody,
    CloseAfterCompleteBody,
}

// Real TCP boundaries rather than a synthetic reqwest error. Every successful
// metadata response closes its connection, so a dead listener cannot be hidden
// behind a pooled HTTP connection when generation starts.
async fn ha_tcp_fault(fault: HaFault) -> (Server, Arc<AtomicUsize>) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let state = Fake {
        count: 100,
        load: 0,
        mode: "ha_tcp",
        generation_calls: Arc::new(AtomicUsize::new(0)),
        counts: Default::default(),
        generations: Default::default(),
        load_override: Default::default(),
        incarnation: uuid::Uuid::new_v4().simple().to_string(),
    };
    let received_body = Arc::new(AtomicUsize::new(0));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let recorded = state.clone();
    let received = received_body.clone();
    let task = tokio::spawn(async move {
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut header = Vec::new();
            while !header.ends_with(b"\r\n\r\n") {
                let mut byte = [0u8];
                if socket.read_exact(&mut byte).await.is_err() {
                    return;
                }
                header.push(byte[0]);
                assert!(header.len() <= 32 * 1024);
            }
            let header = String::from_utf8(header).unwrap();
            let path = header
                .lines()
                .next()
                .unwrap()
                .split_whitespace()
                .nth(1)
                .unwrap();
            let length = header
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
                .unwrap_or(0);
            let generation = path == "/v1/chat/completions";
            if generation {
                recorded.generation_calls.fetch_add(1, Ordering::AcqRel);
                let to_read = if fault == HaFault::CloseAfterPartialBody {
                    length.min(4096)
                } else {
                    length
                };
                let mut bytes = vec![0; to_read];
                socket.read_exact(&mut bytes).await.unwrap();
                received.store(bytes.len(), Ordering::Release);
                if to_read == length {
                    recorded
                        .generations
                        .lock()
                        .unwrap()
                        .push(serde_json::from_slice(&bytes).unwrap());
                }
                // No response headers: even complete body delivery does not
                // prove whether generation began, so this must not be replayed.
                drop(socket);
                return;
            }
            let mut bytes = vec![0; length];
            socket.read_exact(&mut bytes).await.unwrap();
            let (status, reply) = if path == "/v1/tokenize" {
                recorded
                    .counts
                    .lock()
                    .unwrap()
                    .push(serde_json::from_slice(&bytes).unwrap());
                let status = match fault {
                    HaFault::CounterUnavailable => "503 Service Unavailable",
                    HaFault::CounterRejected => "422 Unprocessable Entity",
                    _ => "200 OK",
                };
                (
                    status,
                    json!({"count":recorded.count,"max_model_len":1048576}),
                )
            } else {
                assert_eq!(path, "/v1/loads?include=core");
                let reply = identity_snapshot(json!([]), 0, wall_time(), &recorded.incarnation);
                if fault == HaFault::CloseListenerAfterLoad {
                    // Drop the listener before returning the load response.
                    // The next generation connection is provably not accepted.
                    drop(listener);
                    let body = reply.to_string();
                    socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).as_bytes()).await.unwrap();
                    socket.shutdown().await.unwrap();
                    return;
                }
                ("200 OK", reply)
            };
            let body = reply.to_string();
            socket.write_all(format!("HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).as_bytes()).await.unwrap();
            socket.shutdown().await.unwrap();
        }
    });
    (Server { url, state, task }, received_body)
}

#[tokio::test]
async fn ha_counter_unavailable_uses_other_counter_without_generation() {
    let (unavailable, _) = ha_tcp_fault(HaFault::CounterUnavailable).await;
    let available = fake(123, 0, "good").await;
    let engine = engine(vec![
        worker(&unavailable, NativeTier::Short),
        worker(&available, NativeTier::Short),
    ]);
    assert_eq!(engine.count_tokens(body(false), ctx()).await.unwrap(), 123);
    assert_eq!(unavailable.state.counts.lock().unwrap().len(), 1);
    assert_eq!(available.state.counts.lock().unwrap().len(), 1);
    assert_eq!(
        unavailable.state.generation_calls.load(Ordering::Acquire),
        0
    );
    assert_eq!(available.state.generation_calls.load(Ordering::Acquire), 0);
}

#[tokio::test]
async fn ha_counter_semantic_rejection_is_not_hidden_by_other_counter() {
    let (rejected, _) = ha_tcp_fault(HaFault::CounterRejected).await;
    let available = fake(123, 0, "good").await;
    let engine = engine(vec![
        worker(&rejected, NativeTier::Short),
        worker(&available, NativeTier::Short),
    ]);
    assert!(engine.count_tokens(body(false), ctx()).await.is_err());
    assert_eq!(rejected.state.counts.lock().unwrap().len(), 1);
    assert!(available.state.counts.lock().unwrap().is_empty());
    assert_eq!(available.state.generation_calls.load(Ordering::Acquire), 0);
}

#[tokio::test]
async fn ha_proven_connect_failure_switches_compatible_worker_once() {
    let (disappeared, received) = ha_tcp_fault(HaFault::CloseListenerAfterLoad).await;
    let available = fake(100, 1, "good").await;
    let engine = engine(vec![
        worker(&disappeared, NativeTier::Short),
        worker(&available, NativeTier::Short),
    ]);
    let output = engine
        .forward_chat(body(false), ctx())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert_eq!(output.len(), 1);
    assert!(output[0].data.is_some());
    assert_eq!(
        disappeared.state.generation_calls.load(Ordering::Acquire),
        0
    );
    assert_eq!(received.load(Ordering::Acquire), 0);
    assert_eq!(available.state.generation_calls.load(Ordering::Acquire), 1);
    assert_eq!(disappeared.state.counts.lock().unwrap().len(), 1);
    assert!(available.state.counts.lock().unwrap().is_empty()); // No repeated tokenize RPC.
    let count = disappeared.state.counts.lock().unwrap()[0].clone();
    let generated = available.state.generations.lock().unwrap()[0].clone();
    assert_eq!(dynamo_llm::native_contract::count_body(&generated), count);
    assert_eq!(engine.inflight(), vec![0, 0]);
}

#[tokio::test]
async fn ha_partial_or_complete_large_body_unknown_result_never_replays_generation() {
    for fault in [
        HaFault::CloseAfterPartialBody,
        HaFault::CloseAfterCompleteBody,
    ] {
        let (failed, received) = ha_tcp_fault(fault).await;
        let available = fake(100, 1, "good").await;
        let engine = engine(vec![
            worker(&failed, NativeTier::Short),
            worker(&available, NativeTier::Short),
        ]);
        let mut request = body(false);
        request["messages"][0]["content"] = "x".repeat(256 * 1024).into();
        let result = engine.forward_chat(request, ctx()).await;
        match result {
            Err(_) => {}
            Ok(output) => assert!(
                output
                    .collect::<Vec<_>>()
                    .await
                    .iter()
                    .any(|chunk| chunk.data.is_none())
            ),
        }
        let observed = received.load(Ordering::Acquire);
        if fault == HaFault::CloseAfterPartialBody {
            assert_eq!(observed, 4096);
        } else {
            assert!(observed > 256 * 1024);
        }
        assert_eq!(failed.state.generation_calls.load(Ordering::Acquire), 1);
        assert_eq!(available.state.generation_calls.load(Ordering::Acquire), 0);
        assert_eq!(engine.inflight(), vec![0, 0]);
    }
}

#[tokio::test]
async fn ha_stream_eof_after_native_output_never_dispatches_other_worker() {
    let failed = fake(100, 0, "no_done").await;
    let available = fake(100, 1, "good").await;
    let engine = engine(vec![
        worker(&failed, NativeTier::Short),
        worker(&available, NativeTier::Short),
    ]);
    let output = engine
        .forward_chat(body(true), ctx())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(output.iter().any(|chunk| chunk.data.is_some()));
    assert!(output.iter().any(|chunk| chunk.data.is_none()));
    assert_eq!(failed.state.generation_calls.load(Ordering::Acquire), 1);
    assert_eq!(available.state.generation_calls.load(Ordering::Acquire), 0);
    assert_eq!(engine.inflight(), vec![0, 0]);
}

#[tokio::test]
async fn ha_ds_short_connect_failover_releases_old_permit_and_preserves_exact_rid() {
    let (disappeared, received) = ha_tcp_fault(HaFault::CloseListenerAfterLoad).await;
    let mut servers = vec![disappeared];
    for rank in 0..11 {
        let server = fake(100, 1, "good").await;
        *server.state.load_override.lock().unwrap() = Some(identity_snapshot(
            json!([format!("other-native-request-{rank}")]),
            1,
            wall_time(),
            &server.state.incarnation,
        ));
        servers.push(server);
    }
    let (engine, fence) = ds_identity_engine(&servers).await;
    let output = engine
        .forward_chat(ds_body(1), ctx())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(output.iter().all(|chunk| chunk.data.is_some()));
    assert!(!output.is_empty());
    assert_eq!(servers[0].state.generation_calls.load(Ordering::Acquire), 0);
    assert_eq!(received.load(Ordering::Acquire), 0);
    let dispatched: Vec<_> = servers
        .iter()
        .skip(1)
        .filter(|server| server.state.generation_calls.load(Ordering::Acquire) != 0)
        .collect();
    assert_eq!(dispatched.len(), 1);
    assert_eq!(
        dispatched[0].state.generation_calls.load(Ordering::Acquire),
        1
    );
    let original_count = servers[0].state.counts.lock().unwrap()[0].clone();
    let native = dispatched[0].state.generations.lock().unwrap()[0].clone();
    assert_eq!(native["rid"], original_count["rid"]);
    assert_eq!(native["n"], 1);
    assert_eq!(
        dynamo_llm::native_contract::count_body(&native),
        original_count
    );
    assert!(
        servers
            .iter()
            .skip(1)
            .all(|server| server.state.counts.lock().unwrap().is_empty())
    );
    assert_eq!(fence.reservation_count(), 0);
    assert!(engine.inflight().iter().all(|count| *count == 0));
}

#[derive(Clone)]
struct ExactLoadState {
    count: u64,
    snapshot: Arc<Mutex<Value>>,
    generations: Arc<Mutex<Vec<Value>>>,
    release: tokio::sync::watch::Sender<bool>,
    incarnation: String,
}
struct ExactLoadServer {
    url: String,
    state: ExactLoadState,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for ExactLoadServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}
async fn exact_load_server(count: u64, unrelated: usize) -> ExactLoadServer {
    let incarnation = uuid::Uuid::new_v4().simple().to_string();
    let ids: Vec<_> = (0..unrelated)
        .map(|index| format!("external:{index}"))
        .collect();
    let (release, _) = tokio::sync::watch::channel(false);
    let state = ExactLoadState {
        count,
        snapshot: Arc::new(Mutex::new(identity_snapshot(
            json!(ids),
            unrelated,
            wall_time(),
            &incarnation,
        ))),
        generations: Default::default(),
        release,
        incarnation,
    };
    let app = Router::new()
        .route("/v1/tokenize", post(|State(s): State<ExactLoadState>, Json(_): Json<Value>| async move { Json(json!({"count":s.count,"max_model_len":1048576})) }))
        .route("/v1/loads", get(|State(s): State<ExactLoadState>| async move { Json(s.snapshot.lock().unwrap().clone()) }))
        .route("/v1/chat/completions", post(|State(s): State<ExactLoadState>, Json(request): Json<Value>| async move {
            s.generations.lock().unwrap().push(request.clone());
            let n = request.get("n").and_then(Value::as_u64).unwrap_or(1);
            let choices: Vec<_> = (0..n).map(|index| json!({"index":index,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"})).collect();
            let reply = json!({"id":"exact-fixture","object":"chat.completion","created":1,"model":request["model"],"choices":choices,"usage":{"prompt_tokens":s.count,"completion_tokens":n,"total_tokens":s.count+n}});
            let mut release = s.release.subscribe();
            let chunks = async_stream::stream! {
                // Real HTTP headers and incomplete JSON are delivered while
                // native IDs/counts may already contain this request. No model
                // output, role delta or finish time is used as acceptance proof.
                yield Ok::<_, std::io::Error>(bytes::Bytes::from_static(b" "));
                loop {
                    let released = *release.borrow_and_update();
                    if released { break; }
                    if release.changed().await.is_err() { return; }
                }
                yield Ok(bytes::Bytes::from(reply.to_string()));
            };
            Response::builder().header("content-type", "application/json").body(Body::from_stream(chunks)).unwrap()
        })).with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    ExactLoadServer { url, state, task }
}
fn exact_worker(server: &ExactLoadServer, tier: NativeTier) -> NativeWorkerConfig {
    NativeWorkerConfig {
        url: server.url.clone(),
        identity: None,
        thinking_budget_qualification: None,
        forced_zero_qualification: None,
        tier,
        max_inflight: None,
        load_path: Some("/v1/loads?include=core".into()),
        max_input_tokens: Some(1_048_576),
        capability_source: Some("CPU exact-load fixture only".into()),
        active_request_ids_source: Some("CPU exact scheduler snapshot fixture only".into()),
        native_dp_size: Some(1),
        native_id_format: NativeIdFormat::SglangPairedRid,
    }
}
async fn exact_engine(model: &str, workers: Vec<NativeWorkerConfig>) -> NativeHttpEngine {
    let ds = model == "deepseek-ai/DeepSeek-V4.1-Flash";
    let engine = NativeHttpEngine::new(NativeHttpEngineConfig {
        model: model.into(),
        workers,
        qualified_model_identity: ds.then(|| "CPU fixture identity only".into()),
        qualification_tier: None,
    })
    .unwrap();
    if !ds {
        return engine;
    }
    // Long-pool admission checks authority only. These short URLs are never
    // polled or dispatched by a long request; no short gate behavior is changed.
    let fence = dynamo_llm::native_fencing::NativeFence::new(
        dynamo_llm::native_fencing::FenceConfig {
            election_key: "/cpu-exact-load/long".into(),
            holder: "cpu-long".into(),
            short_workers: (0..12)
                .map(|i| format!("http://cpu-short-{i}.invalid"))
                .collect(),
            max_effective_load: 2,
        },
        Arc::new(IdentityTransportStore),
        Arc::new(IdentityPrefixStore),
    )
    .unwrap();
    fence.refresh_authority().await.unwrap();
    engine.with_fence(fence)
}
fn exact_body(model: &str, n: u32) -> Value {
    let mut request = body(false);
    request["model"] = model.into();
    request["n"] = n.into();
    request
}
fn exact_ids(server: &ExactLoadServer, ids: Vec<String>) {
    *server.state.snapshot.lock().unwrap() = identity_snapshot(
        json!(ids),
        ids.len(),
        wall_time(),
        &server.state.incarnation,
    );
}
fn exact_rid(server: &ExactLoadServer, index: usize) -> String {
    server.state.generations.lock().unwrap()[index]["rid"]
        .as_str()
        .expect("qualified native generation must carry owned RID")
        .to_owned()
}
async fn exact_dispatch(
    engine: &NativeHttpEngine,
    request: Value,
    context: Arc<dyn AsyncEngineContext>,
) -> dynamo_runtime::pipeline::ManyOut<dynamo_runtime::protocols::annotated::Annotated<Value>> {
    tokio::time::timeout(
        Duration::from_secs(3),
        engine.forward_chat(request, context),
    )
    .await
    .unwrap()
    .unwrap()
}

#[tokio::test]
async fn exact_load_nonstream_native_ids_absorb_local_charge_before_body_returns() {
    for (model, tier, count) in [
        ("deepseek-ai/DeepSeek-V4.1-Flash", NativeTier::Long, 70000),
        (MODEL, NativeTier::Short, 100),
    ] {
        let a = exact_load_server(count, 0).await;
        let b = exact_load_server(count, 3).await;
        let engine =
            exact_engine(model, vec![exact_worker(&a, tier), exact_worker(&b, tier)]).await;
        let first = exact_dispatch(&engine, exact_body(model, 1), ctx()).await;
        let rid1 = exact_rid(&a, 0);
        exact_ids(&a, vec![rid1.clone()]);
        let second = exact_dispatch(&engine, exact_body(model, 1), ctx()).await;
        let rid2 = exact_rid(&a, 1);
        assert_ne!(rid1, rid2);
        exact_ids(&a, vec![rid1, rid2]);
        // A actual2 < B3. The obsolete native2+local2 score would wrongly pick B.
        let third = exact_dispatch(&engine, exact_body(model, 1), ctx()).await;
        assert_eq!(a.state.generations.lock().unwrap().len(), 3, "{model}");
        assert!(b.state.generations.lock().unwrap().is_empty(), "{model}");
        assert_eq!(engine.inflight(), vec![3, 0]);
        // The accepted IDs can finish in the scheduler before the buffered
        // HTTP JSON returns. Same incarnation + absent accepted IDs must not
        // recreate their pending charges. Only the third request remains pending.
        exact_ids(&a, Vec::new());
        exact_ids(&b, vec!["external:0".into(), "external:1".into()]);
        let fourth = exact_dispatch(&engine, exact_body(model, 1), ctx()).await;
        assert_eq!(
            a.state.generations.lock().unwrap().len(),
            4,
            "{model}: disappeared accepted IDs cannot become pending again"
        );
        assert!(b.state.generations.lock().unwrap().is_empty());
        drop((first, second, third, fourth));
        assert_eq!(engine.inflight(), vec![0, 0]);
    }
}

#[tokio::test]
async fn exact_load_pending_parallel_weight_and_partial_exact_identity_are_preserved() {
    for (model, tier, count) in [
        (MODEL, NativeTier::Short, 100),
        ("deepseek-ai/DeepSeek-V4.1-Flash", NativeTier::Long, 70000),
    ] {
        let a = exact_load_server(count, 0).await;
        let b = exact_load_server(count, 3).await;
        let engine =
            exact_engine(model, vec![exact_worker(&a, tier), exact_worker(&b, tier)]).await;
        let pending = exact_dispatch(&engine, exact_body(model, 2), ctx()).await;
        exact_ids(&b, vec!["external:0".into()]);
        let next = exact_dispatch(&engine, exact_body(model, 1), ctx()).await;
        assert_eq!(
            a.state.generations.lock().unwrap().len(),
            1,
            "n2 pending must score2, not1: {model}"
        );
        assert_eq!(b.state.generations.lock().unwrap().len(), 1);
        drop((pending, next));
        assert_eq!(engine.inflight(), vec![0, 0]);
        for forged in [false, true] {
            let a = exact_load_server(count, 0).await;
            let b = exact_load_server(count, 5).await;
            let engine =
                exact_engine(model, vec![exact_worker(&a, tier), exact_worker(&b, tier)]).await;
            let pending = exact_dispatch(&engine, exact_body(model, 3), ctx()).await;
            let rid = exact_rid(&a, 0);
            let indices = if forged { ["00", "01"] } else { ["0", "1"] };
            exact_ids(
                &a,
                indices
                    .iter()
                    .map(|index| format!("{rid}_0:sample:{index}"))
                    .collect(),
            );
            exact_ids(&b, (0..4).map(|i| format!("external:{i}")).collect());
            let next = exact_dispatch(&engine, exact_body(model, 1), ctx()).await;
            // Exact partial: native2+pending1=3 < B4; forged: native2+pending3=5 > B4.
            assert_eq!(
                a.state.generations.lock().unwrap().len(),
                if forged { 1 } else { 2 },
                "{model}, forged={forged}"
            );
            assert_eq!(
                b.state.generations.lock().unwrap().len(),
                usize::from(forged)
            );
            drop((pending, next));
            assert_eq!(engine.inflight(), vec![0, 0]);
        }
    }
}

#[tokio::test]
async fn exact_load_incomplete_or_unqualified_snapshot_skips_worker() {
    for case in 0..8 {
        let bad = exact_load_server(100, 0).await;
        let good = exact_load_server(100, 10).await;
        let mut config = exact_worker(&bad, NativeTier::Short);
        let mut snapshot = identity_snapshot(json!([]), 0, wall_time(), &bad.state.incarnation);
        match case {
            0 => {
                snapshot["loads"][0]["active_request_ids"] = Value::Null;
            }
            1 => {
                snapshot["loads"][0]
                    .as_object_mut()
                    .unwrap()
                    .remove("active_request_ids");
            }
            2 => {
                snapshot = identity_snapshot(
                    json!(["duplicate", "duplicate"]),
                    2,
                    wall_time(),
                    &bad.state.incarnation,
                );
            }
            3 => {
                snapshot =
                    identity_snapshot(json!(["one"]), 2, wall_time(), &bad.state.incarnation);
            }
            4 => {
                config.native_dp_size = Some(2);
            } // actual DP1 response is partial for qualified DP2.
            5 => {
                snapshot["loads"][0]["native_incarnation"] = "not-a-scheduler-uuid".into();
            }
            6 => {
                snapshot["loads"][0]["timestamp"] = (wall_time() - 10.0).into();
            }
            7 => {
                snapshot["loads"][0]["timestamp"] = (wall_time() + 1000.0).into();
            }
            _ => unreachable!(),
        }
        *bad.state.snapshot.lock().unwrap() = snapshot;
        let engine =
            exact_engine(MODEL, vec![config, exact_worker(&good, NativeTier::Short)]).await;
        let output = exact_dispatch(&engine, exact_body(MODEL, 1), ctx()).await;
        assert!(
            bad.state.generations.lock().unwrap().is_empty(),
            "case {case}"
        );
        assert_eq!(
            good.state.generations.lock().unwrap().len(),
            1,
            "case {case}"
        );
        drop(output);
        assert_eq!(engine.inflight(), vec![0, 0]);
    }
}

#[tokio::test]
async fn exact_load_concurrent_poll_reserve_finish_and_cancel_leave_no_pending_charge() {
    let a = exact_load_server(100, 0).await;
    let b = exact_load_server(100, 100).await;
    let engine = exact_engine(
        MODEL,
        vec![
            exact_worker(&a, NativeTier::Short),
            exact_worker(&b, NativeTier::Short),
        ],
    )
    .await;
    let contexts: Vec<_> = (0..4).map(|_| ctx()).collect();
    let mut outputs =
        futures::future::join_all(contexts.iter().enumerate().map(|(index, context)| {
            exact_dispatch(
                &engine,
                exact_body(MODEL, if index == 1 { 2 } else { 1 }),
                context.clone(),
            )
        }))
        .await;
    assert_eq!(a.state.generations.lock().unwrap().len(), 4);
    let mut accepted = Vec::new();
    for request in a.state.generations.lock().unwrap().iter() {
        let rid = request["rid"].as_str().unwrap();
        if request["n"] == 2 {
            accepted.extend([format!("{rid}_0:sample:0"), format!("{rid}_0:sample:1")]);
        } else {
            accepted.push(rid.to_owned());
        }
    }
    assert_eq!(accepted.len(), 5);
    exact_ids(&a, accepted);
    let dropped = outputs.remove(0);
    let mut cancelled = outputs.remove(0);
    a.state.release.send(true).unwrap();
    let (probe, ()) = futures::join!(
        async {
            exact_dispatch(&engine, exact_body(MODEL, 1), ctx())
                .await
                .collect::<Vec<_>>()
                .await
        },
        async {
            drop(dropped);
            contexts[1].stop();
            assert!(
                tokio::time::timeout(Duration::from_secs(1), cancelled.next())
                    .await
                    .unwrap()
                    .is_none()
            );
            for output in outputs {
                assert!(
                    output
                        .collect::<Vec<_>>()
                        .await
                        .iter()
                        .all(|chunk| chunk.data.is_some())
                );
            }
        }
    );
    assert!(probe.iter().all(|chunk| chunk.data.is_some()));
    assert_eq!(engine.inflight(), vec![0, 0]);
    // A fresh idle sample must not retain ghost weights after cancellation,
    // completion and drop. A stale n2 cancellation charge would push this to B.
    exact_ids(&a, Vec::new());
    exact_ids(&b, vec!["external:0".into()]);
    let before = a.state.generations.lock().unwrap().len();
    let final_probe = exact_dispatch(&engine, exact_body(MODEL, 1), ctx())
        .await
        .collect::<Vec<_>>()
        .await;
    assert!(final_probe.iter().all(|chunk| chunk.data.is_some()));
    assert_eq!(a.state.generations.lock().unwrap().len(), before + 1);
    assert!(b.state.generations.lock().unwrap().is_empty());
    assert_eq!(engine.inflight(), vec![0, 0]);
}

#[tokio::test]
async fn exact_load_connect_failure_keeps_owned_rid_and_releases_previous_reservation() {
    let (failed, received) = ha_tcp_fault(HaFault::CloseListenerAfterLoad).await;
    let good = exact_load_server(100, 1).await;
    let engine = exact_engine(
        MODEL,
        vec![
            worker(&failed, NativeTier::Short),
            exact_worker(&good, NativeTier::Short),
        ],
    )
    .await;
    let output = exact_dispatch(&engine, exact_body(MODEL, 1), ctx()).await;
    let counted = failed.state.counts.lock().unwrap()[0].clone();
    let generated = good.state.generations.lock().unwrap()[0].clone();
    assert!(generated["rid"].is_string());
    assert_eq!(generated["rid"], counted["rid"]);
    assert_eq!(generated["n"], counted["n"]);
    assert_eq!(dynamo_llm::native_contract::count_body(&generated), counted);
    assert_eq!(failed.state.generation_calls.load(Ordering::Acquire), 0);
    assert_eq!(received.load(Ordering::Acquire), 0);
    exact_ids(&good, vec![generated["rid"].as_str().unwrap().to_owned()]);
    good.state.release.send(true).unwrap();
    assert!(
        output
            .collect::<Vec<_>>()
            .await
            .iter()
            .all(|chunk| chunk.data.is_some())
    );
    assert_eq!(engine.inflight(), vec![0, 0]);
}
