// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Actual public HTTP generation and signed physical cleanup, using only
//! synthetic localhost CPU cohorts. No native/GPU qualification is claimed.
use axum::{
    Json, Router,
    body::Body,
    extract::State,
    response::Response,
    routing::{get, post},
};
use dynamo_llm::{
    cache_authority::cache_auth,
    engines::StreamingEngineAdapter,
    http::service::service_v2::HttpService,
    model_card::ModelDeploymentCard,
    native_cleanup::{CleanupState, NativeCleanupRegistry},
    native_cleanup_cohort::{
        CleanupCohortApproval, CleanupCohortArtifact, CleanupCohortParticipant,
        InstalledCleanupCohort, install_cleanup_cohort,
    },
    native_contract::NativeTier,
    native_http_engine::{
        NativeHttpEngine, NativeHttpEngineConfig, NativeIdFormat, NativeWorkerConfig,
    },
    native_request_lifecycle::QualifiedRequestLifecycle,
};
use futures::StreamExt;
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    os::unix::fs::PermissionsExt,
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

const MODEL: &str = "qwen3.8-flash-next";
const SECRET: [u8; 32] = [0x63; 32];
const PARTICIPANTS: [&str; 3] = ["opaque-helper", "opaque-scheduler", "opaque-tokenizer"];
const CPU_TRACKING: &str = "Reference ID : CB00710F (CPU-only cleanup tracking stand-in)\nSystem time : 0 seconds slow of NTP time\nRoot delay : 0 seconds\nRoot dispersion : 0 seconds\nLeap status : Normal\n";
fn cpu_clock() {
    static CLOCK: OnceLock<tempfile::TempDir> = OnceLock::new();
    CLOCK.get_or_init(|| {
        let directory = tempfile::tempdir().unwrap();
        let tracking = directory.path().join("CPU-cleanup-tracking");
        std::fs::write(&tracking, CPU_TRACKING).unwrap();
        let executable = directory.path().join("chronyc");
        std::fs::write(
            &executable,
            b"#!/bin/sh\nexec /bin/cat \"$DYNAMO_CLEANUP_CPU_TRACKING\"\n",
        )
        .unwrap();
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
        let existing = std::env::var_os("PATH").unwrap_or_default();
        let path = std::env::join_paths(
            std::iter::once(directory.path().to_path_buf()).chain(std::env::split_paths(&existing)),
        )
        .unwrap();
        // Run this integration target with --test-threads=1. The command is an
        // explicit synthetic CPU tracking model; production ChronyClock still
        // qualifies the actual command, wall/monotonic sample, and freshness.
        unsafe {
            std::env::set_var("PATH", path);
            std::env::set_var("DYNAMO_CLEANUP_CPU_TRACKING", &tracking);
        }
        directory
    });
}
fn wall_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
        .try_into()
        .unwrap()
}
struct Wire {
    incarnation: String,
    hold_stream: bool,
    bad_ack: bool,
    registry: Mutex<Option<Arc<NativeCleanupRegistry>>>,
    counts: Mutex<Vec<Value>>,
    generations: Mutex<Vec<Value>>,
    controls: Mutex<Vec<Value>>,
    observed: Notify,
    release: Notify,
}
struct Worker {
    url: String,
    wire: Arc<Wire>,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.task.abort();
    }
}

fn installed(origin: &str, incarnation: &str) -> InstalledCleanupCohort {
    let participants: Vec<_> = PARTICIPANTS
        .into_iter()
        .map(|id| CleanupCohortParticipant {
            participant_id: id.into(),
            role: format!("synthetic-CPU-role-{id}"),
            process_birth: format!("synthetic-CPU-birth-{id}"),
            source_digest: "a".repeat(64),
            runtime_registration_digest: "b".repeat(64),
        })
        .collect();
    let ids: BTreeSet<_> = participants
        .iter()
        .map(|p| p.participant_id.clone())
        .collect();
    let artifact = CleanupCohortArtifact {
        schema_version: 1,
        native_origin: origin.into(),
        native_incarnation: incarnation.into(),
        model: MODEL.into(),
        source_digest: "c".repeat(64),
        runtime_profile_digest: "d".repeat(64),
        cleanup_cohort_digest: cache_auth::digest(&json!(ids)).unwrap(),
        participants,
    };
    let approval = CleanupCohortApproval {
        schema_version: 1,
        artifact_digest: artifact.artifact_digest().unwrap(),
    };
    install_cleanup_cohort(&approval, artifact.clone(), &artifact).unwrap()
}
async fn control(State(w): State<Arc<Wire>>, Json(packet): Json<Value>) -> Json<Value> {
    let mut body = packet;
    let proof = body
        .as_object_mut()
        .unwrap()
        .remove(cache_auth::PROOF_FIELD)
        .unwrap();
    let claims = cache_auth::verify_signature(&SECRET, &proof, &body).unwrap();
    assert!(matches!(
        claims["operation"].as_str(),
        Some("request-abort" | "request-status")
    ));
    assert_eq!(claims["native_request_id"], body["request_root"]);
    assert_eq!(claims["reservation_id"], "");
    assert_eq!(claims["authority_revision"], 0);
    w.controls.lock().unwrap().push(body.clone());
    w.observed.notify_one();
    if !w.bad_ack {
        w.release.notified().await;
    }
    let ids = if w.bad_ack {
        &PARTICIPANTS[..2]
    } else {
        &PARTICIPANTS[..]
    };
    let participants: Vec<_> = ids
        .iter()
        .map(|id| {
            json!({"participant_id":id,
        "ingress_closed":true,"tokenizer_drained":true,"scheduler_drained":true,
        "live_request_holds":0,"live_kv_holds":0,"live_meta_holds":0,"live_io_holds":0,
        "transport_drained":true,"abort_ack":true})
        })
        .collect();
    let mut reply = json!({"schema_version":1,"transaction_id":body["transaction_id"],"request_root":body["request_root"],
        "native_incarnation":body["expected_native_incarnation"],"cleanup_cohort_digest":body["expected_cleanup_cohort_digest"],
        "ordinary_body_digest":body["ordinary_body_digest"],"expected_choices":body["expected_choices"],
        "captured_at_ms":wall_ms(),"participants":participants,"terminal":true,"aborted":true});
    let event = cache_auth::digest(&reply).unwrap();
    let mut response_claims = claims.clone();
    for key in ["operation_id", "nonce"] {
        response_claims[key] = cache_auth::digest(&json!(["native-response", claims[key], event]))
            .unwrap()
            .into();
    }
    reply[cache_auth::PROOF_FIELD] = cache_auth::sign(&SECRET, &response_claims, &reply).unwrap();
    Json(reply)
}
async fn generation(State(w): State<Arc<Wire>>, Json(body): Json<Value>) -> Response {
    let registry = w.registry.lock().unwrap().clone().unwrap();
    assert_eq!(
        registry.states().len(),
        1,
        "registration must precede actual native POST"
    );
    assert!(
        registry
            .states()
            .values()
            .all(|s| *s == CleanupState::Running)
    );
    let payload = cache_auth::payload_for("request-generate", &body).unwrap();
    let claims =
        cache_auth::verify_signature(&SECRET, &body[cache_auth::PROOF_FIELD], &payload).unwrap();
    assert_eq!(claims["operation"], "request-generate");
    assert_eq!(body["n"], 2);
    assert_eq!(body["temperature"], json!(0.3));
    assert_eq!(body["prompt_cache_request_control"]["expected_choices"], 2);
    let canonical: Value =
        serde_json::from_str(body["prompt_cache_canonical_body"].as_str().unwrap()).unwrap();
    assert_eq!(canonical, cache_auth::ordinary_body_for(&body).unwrap());
    assert_eq!(canonical["messages"][0]["content"], "CPU cleanup body 🙂");
    w.generations.lock().unwrap().push(body);
    let first =
        json!({"id":"cpu-native-id","model":MODEL,"object":"chat.completion.chunk","created":1,
        "choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},"finish_reason":null},
                   {"index":1,"delta":{"role":"assistant","content":"ok"},"finish_reason":null}]})
        .to_string();
    let last = json!({"id":"cpu-native-id","model":MODEL,"object":"chat.completion.chunk","created":1,
        "choices":[{"index":0,"delta":{},"finish_reason":"stop"},{"index":1,"delta":{},"finish_reason":"stop"}],
        "usage":{"prompt_tokens":100,"completion_tokens":2,"total_tokens":102}}).to_string();
    let chunks = async_stream::stream! {
        yield Ok::<_,std::io::Error>(bytes::Bytes::from(format!("data: {first}\n\n")));
        if w.hold_stream { std::future::pending::<()>().await; }
        yield Ok::<_,std::io::Error>(bytes::Bytes::from(format!("data: {last}\n\ndata: [DONE]\n\n")));
    };
    Response::builder()
        .header("content-type", "text/event-stream")
        .body(Body::from_stream(chunks))
        .unwrap()
}
async fn worker(hold_stream: bool, bad_ack: bool) -> Worker {
    cpu_clock();
    let wire = Arc::new(Wire {
        incarnation: uuid::Uuid::new_v4().simple().to_string(),
        hold_stream,
        bad_ack,
        registry: Mutex::new(None),
        counts: Mutex::new(vec![]),
        generations: Mutex::new(vec![]),
        controls: Mutex::new(vec![]),
        observed: Notify::new(),
        release: Notify::new(),
    });
    let router = Router::new()
        .route("/v1/tokenize",post(|State(w):State<Arc<Wire>>,Json(body):Json<Value>| async move {
            w.counts.lock().unwrap().push(body); Json(json!({"count":100,"max_model_len":1048576}))
        }))
        .route("/v1/loads",get(|State(w):State<Arc<Wire>>| async move {
            Json(json!({"loads":[{"dp_rank":0,"num_running_reqs":0,"num_waiting_reqs":0,"active_request_ids":[],
                "native_incarnation":w.incarnation,"timestamp":SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64(),
                "frozen":false,"heartbeat_timestamp":0.0,"snapshot_generation":1}]}))
        }))
        .route("/v1/chat/completions",post(generation))
        .route("/v1/requests/abort",post(control)).route("/v1/requests/status",post(control))
        .with_state(wire.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    Worker { url, wire, task }
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
    let engine = NativeHttpEngine::new(NativeHttpEngineConfig {
        model: MODEL.into(),
        qualification_tier: None,
        qualified_model_identity: None,
        workers: vec![NativeWorkerConfig {
            url: worker.url.clone(),
            tier: NativeTier::Short,
            max_inflight: Some(1),
            load_path: Some("/v1/loads?include=core".into()),
            max_input_tokens: Some(1_048_576),
            capability_source: Some("synthetic CPU input boundary".into()),
            active_request_ids_source: Some(
                "synthetic CPU complete same-snapshot IDs/rank/UUID4/time".into(),
            ),
            native_dp_size: Some(1),
            native_id_format: NativeIdFormat::SglangPairedRid,
            identity: None,
            thinking_budget_qualification: None,
            forced_zero_qualification: None,
        }],
    })
    .unwrap();
    let registry = engine.cleanup_registry();
    *worker.wire.registry.lock().unwrap() = Some(registry.clone());
    let lifecycle = QualifiedRequestLifecycle::new(
        installed(&worker.url, &worker.wire.incarnation),
        SECRET,
        &worker.url,
        registry.as_ref().clone(),
    )
    .unwrap();
    let engine = Arc::new(engine.with_request_lifecycles(vec![lifecycle]).unwrap());
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
async fn public_request(f: &Frontend) -> reqwest::Response {
    let body = json!({"model":MODEL,"messages":[{"role":"user","content":"CPU cleanup body 🙂"}],
        "max_tokens":128,"reasoning_effort":"none","n":2,"temperature":0.3,"stream":true,"stream_options":{"include_usage":true}});
    let response = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", f.url))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    response
}
async fn trigger_public_finish(f: &Frontend, cancel: bool) {
    let response = public_request(f).await;
    if cancel {
        let mut body = response.bytes_stream();
        assert!(
            tokio::time::timeout(Duration::from_secs(2), body.next())
                .await
                .unwrap()
                .unwrap()
                .is_ok()
        );
        drop(body);
    } else {
        let body = response.text().await.unwrap();
        assert!(body.contains("[DONE]"), "{body}");
    }
}

#[tokio::test]
async fn public_cleanup_eof_and_body_drop_hold_admission_until_signed_physical_ack() {
    for cancel in [false, true] {
        let native = worker(cancel, false).await;
        let f = frontend(&native).await;
        trigger_public_finish(&f, cancel).await;
        tokio::time::timeout(Duration::from_secs(2), native.wire.observed.notified())
            .await
            .unwrap();
        let registry = f.engine.cleanup_registry();
        assert_eq!(f.engine.inflight(), vec![1], "cancel={cancel}");
        assert_eq!(registry.states().len(), 1);
        assert!(
            registry
                .states()
                .values()
                .all(|s| *s == CleanupState::Running)
        );
        assert_eq!(native.wire.counts.lock().unwrap().len(), 1);
        assert_eq!(native.wire.generations.lock().unwrap().len(), 1);
        let waiter_registry = registry.clone();
        let mut shutdown = tokio::spawn(async move {
            waiter_registry.wait_for_physical_zero().await;
        });
        assert!(
            tokio::time::timeout(Duration::from_millis(100), &mut shutdown)
                .await
                .is_err(),
            "SIGTERM waiter completed before physical ACK"
        );
        assert_eq!(f.engine.inflight(), vec![1]);
        native.wire.release.notify_one();
        tokio::time::timeout(Duration::from_secs(2), shutdown)
            .await
            .unwrap()
            .unwrap();
        assert!(registry.states().is_empty());
        assert_eq!(f.engine.inflight(), vec![0]);
        let controls = native.wire.controls.lock().unwrap();
        assert_eq!(controls.len(), 1);
        assert_eq!(controls[0]["expected_choices"], 2);
    }
}

#[tokio::test]
async fn public_cleanup_bad_signed_ack_stays_blocked_and_shutdown_pending() {
    let native = worker(false, true).await;
    let f = frontend(&native).await;
    trigger_public_finish(&f, false).await;
    let registry = f.engine.cleanup_registry();
    let blocked = tokio::time::timeout(Duration::from_secs(2), registry.await_quiescence())
        .await
        .unwrap()
        .unwrap_err();
    assert_eq!(blocked.entries.len(), 1);
    assert!(
        blocked
            .entries
            .values()
            .all(|s| matches!(s, CleanupState::Blocked(_)))
    );
    assert_eq!(f.engine.inflight(), vec![1]);
    assert_eq!(native.wire.generations.lock().unwrap().len(), 1);
    let waiter_registry = registry.clone();
    let mut shutdown = tokio::spawn(async move {
        waiter_registry.wait_for_physical_zero().await;
    });
    assert!(
        tokio::time::timeout(Duration::from_millis(100), &mut shutdown)
            .await
            .is_err()
    );
    assert_eq!(f.engine.inflight(), vec![1]);
    shutdown.abort();
}
