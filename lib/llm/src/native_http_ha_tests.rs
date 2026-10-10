// SPDX-License-Identifier: Apache-2.0
//! Causal HA tests through the production material/dispatch factory. All HTTP
//! origins, opaque owners, clock tracking and ACKs here are synthetic CPU data.
//! Run only this selector with --test-threads=1; no GPU qualification is claimed.
use super::*;
use crate::cache_authority::cache_auth;
use crate::native_cleanup_cohort::{
    CleanupCohortArtifact, CleanupCohortParticipant, OrdinaryCleanupBindingFiles,
    load_ordinary_cleanup_bindings,
};
use axum::{Json, Router, body::Body, extract::State, response::Response, routing::{get, post}};
use dynamo_runtime::pipeline::Context;
use std::{os::unix::fs::PermissionsExt, sync::OnceLock, time::{SystemTime, UNIX_EPOCH}};

const MODEL: &str = "qwen3.8-flash-next";
const SECRET: [u8; 32] = [0x63; 32];

fn cpu_clock() {
    static CLOCK: OnceLock<tempfile::TempDir> = OnceLock::new();
    CLOCK.get_or_init(|| {
        let dir = tempfile::tempdir().unwrap();
        let tracking = dir.path().join("tracking");
        std::fs::write(&tracking, "Reference ID : CB00710F (synthetic CPU HA)\nSystem time : 0 seconds slow of NTP time\nRoot delay : 0 seconds\nRoot dispersion : 0 seconds\nLeap status : Normal\n").unwrap();
        let exe = dir.path().join("chronyc");
        std::fs::write(&exe, b"#!/bin/sh\nexec /bin/cat \"$DYNAMO_HA_CPU_TRACKING\"\n").unwrap();
        std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o700)).unwrap();
        let old = std::env::var_os("PATH").unwrap_or_default();
        let path = std::env::join_paths(std::iter::once(dir.path().to_path_buf()).chain(std::env::split_paths(&old))).unwrap();
        unsafe { std::env::set_var("PATH", path); std::env::set_var("DYNAMO_HA_CPU_TRACKING", tracking); }
        dir
    });
}

fn wall_ms() -> u64 { SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis().try_into().unwrap() }

struct Wire {
    artifact: Mutex<CleanupCohortArtifact>,
    load_offline: AtomicUsize,
    cohort_offline: AtomicUsize,
    post_failure: AtomicUsize,
    prompt: AtomicUsize,
    discoveries: AtomicUsize,
    generations: Mutex<Vec<Value>>,
    controls: AtomicUsize,
}
struct Worker { url: String, wire: Arc<Wire>, task: tokio::task::JoinHandle<()> }
impl Drop for Worker { fn drop(&mut self) { self.task.abort(); } }

fn artifact(origin: &str) -> CleanupCohortArtifact {
    let participants = ["CPU-scheduler", "CPU-tokenizer"].into_iter().map(|id| CleanupCohortParticipant {
        participant_id: id.into(), role: format!("synthetic-{id}"), process_birth: format!("synthetic-birth-{id}"),
        source_digest: "a".repeat(64), runtime_registration_digest: "b".repeat(64),
    }).collect::<Vec<_>>();
    let ids = participants.iter().map(|p| p.participant_id.clone()).collect::<BTreeSet<_>>();
    CleanupCohortArtifact { schema_version: 1, model: MODEL.into(), native_origin: origin.into(),
        native_incarnation: uuid::Uuid::new_v4().simple().to_string(), source_digest: "c".repeat(64),
        runtime_profile_digest: "d".repeat(64), cleanup_cohort_digest: cache_auth::digest(&json!(ids)).unwrap(), participants }
}

fn unavailable() -> Response { Response::builder().status(503).body(Body::from("synthetic offline origin")).unwrap() }

async fn control(State(w): State<Arc<Wire>>, Json(mut body): Json<Value>) -> Json<Value> {
    let proof = body.as_object_mut().unwrap().remove(cache_auth::PROOF_FIELD).unwrap();
    let claims = cache_auth::verify_signature(&SECRET, &proof, &body).unwrap();
    w.controls.fetch_add(1, Ordering::SeqCst);
    let owners = w.artifact.lock().unwrap().participant_ids().into_iter().map(|id| json!({
        "participant_id":id,"ingress_closed":true,"tokenizer_drained":true,"scheduler_drained":true,
        "live_request_holds":0,"live_kv_holds":0,"live_meta_holds":0,"live_io_holds":0,
        "transport_drained":true,"abort_ack":true
    })).collect::<Vec<_>>();
    let mut reply = json!({"schema_version":1,"transaction_id":body["transaction_id"],"request_root":body["request_root"],
        "native_incarnation":body["expected_native_incarnation"],"cleanup_cohort_digest":body["expected_cleanup_cohort_digest"],
        "ordinary_body_digest":body["ordinary_body_digest"],"expected_choices":body["expected_choices"],
        "captured_at_ms":wall_ms(),"participants":owners,"terminal":true,"aborted":true});
    let event = cache_auth::digest(&reply).unwrap();
    let mut response_claims = claims.clone();
    for key in ["operation_id", "nonce"] { response_claims[key] = cache_auth::digest(&json!(["native-response",claims[key],event])).unwrap().into(); }
    reply[cache_auth::PROOF_FIELD] = cache_auth::sign(&SECRET, &response_claims, &reply).unwrap();
    Json(reply)
}

async fn worker() -> Worker {
    cpu_clock();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let wire = Arc::new(Wire { artifact: Mutex::new(artifact(&url)), load_offline: AtomicUsize::new(0),
        cohort_offline: AtomicUsize::new(0), post_failure: AtomicUsize::new(0), prompt: AtomicUsize::new(100),
        discoveries: AtomicUsize::new(0), generations: Mutex::new(vec![]), controls: AtomicUsize::new(0) });
    let router = Router::new()
        .route("/v1/tokenize", post(|State(w): State<Arc<Wire>>| async move {
            Json(json!({"count":w.prompt.load(Ordering::SeqCst),"max_model_len":1_048_576}))
        }))
        .route("/v1/loads", get(|State(w): State<Arc<Wire>>| async move {
            if w.load_offline.load(Ordering::SeqCst) != 0 { return unavailable(); }
            let reply = json!({"loads":[{"dp_rank":0,"num_running_reqs":0,"num_waiting_reqs":0,"active_request_ids":[],
                "native_incarnation":w.artifact.lock().unwrap().native_incarnation,
                "timestamp":SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64(),
                "frozen":false,"heartbeat_timestamp":0.0,"snapshot_generation":1}]});
            Response::new(Body::from(reply.to_string()))
        }))
        .route("/v1/requests/cleanup-cohort", get(|State(w): State<Arc<Wire>>| async move {
            w.discoveries.fetch_add(1, Ordering::SeqCst);
            if w.cohort_offline.load(Ordering::SeqCst) != 0 { return unavailable(); }
            Response::new(Body::from(serde_json::to_string(&*w.artifact.lock().unwrap()).unwrap()))
        }))
        .route("/v1/chat/completions", post(|State(w): State<Arc<Wire>>, Json(body): Json<Value>| async move {
            let payload = cache_auth::payload_for("request-generate", &body).unwrap();
            let claims = cache_auth::verify_signature(&SECRET, &body[cache_auth::PROOF_FIELD], &payload).unwrap();
            assert_eq!(claims["native_incarnation"], w.artifact.lock().unwrap().native_incarnation);
            w.generations.lock().unwrap().push(body);
            if w.post_failure.load(Ordering::SeqCst) != 0 { return unavailable(); }
            let prompt = w.prompt.load(Ordering::SeqCst);
            let reply = json!({"id":"CPU-HA","model":MODEL,"object":"chat.completion","created":1,
                "choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
                "usage":{"prompt_tokens":prompt,"completion_tokens":2,"total_tokens":prompt+2,"completion_tokens_details":{"reasoning_tokens":0}}});
            Response::new(Body::from(reply.to_string()))
        }))
        .route("/v1/requests/abort", post(control)).route("/v1/requests/status", post(control))
        .with_state(wire.clone());
    let task = tokio::spawn(async move { axum::serve(listener, router).await.unwrap(); });
    Worker { url, wire, task }
}

fn engine(workers: &[(&Worker, NativeTier)], scope: Option<NativeTier>) -> (NativeHttpEngine, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let key = dir.path().join("key");
    std::fs::write(&key, SECRET).unwrap();
    std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o400)).unwrap();
    let materials = load_ordinary_cleanup_bindings(&workers.iter().map(|(w,_)| OrdinaryCleanupBindingFiles {
        model: MODEL.into(), native_origin: w.url.clone(), existing_authority_secret_path: key.clone(),
    }).collect::<Vec<_>>()).unwrap();
    let engine = NativeHttpEngine::new(NativeHttpEngineConfig { model: MODEL.into(), qualification_tier: scope,
        workers: workers.iter().map(|(w,tier)| NativeWorkerConfig { url:w.url.clone(),tier:*tier,max_inflight:Some(1),
            load_path:Some("/v1/loads?include=core".into()), max_input_tokens:Some(1_048_576),
            capability_source:Some("synthetic CPU input boundary".into()),active_request_ids_source:Some("synthetic CPU same-snapshot IDs".into()),
            native_dp_size:Some(1), native_id_format:NativeIdFormat::SglangPairedRid, identity:None,
            thinking_budget_qualification:None, forced_zero_qualification:None }).collect() }).unwrap()
        .with_request_lifecycle_materials(materials.into_values().collect()).unwrap();
    (engine,dir)
}

async fn dispatch(engine: &NativeHttpEngine) -> Result<(), Error> {
    let mut output = engine.forward_chat(json!({"model":MODEL,"messages":[{"role":"user","content":"CPU HA"}],
        "max_tokens":8,"temperature":0.3,"stream":false}), Context::new(()).context()).await?;
    let mut n = 0;
    while let Some(frame) = output.next().await {
        assert!(frame.error.is_none(), "{frame:?}");
        assert_eq!(frame.data.as_ref().unwrap()["choices"][0]["message"]["content"], "ok");
        n += 1;
    }
    assert_eq!(n,1);
    Ok(())
}

async fn released(engine: &NativeHttpEngine) {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if engine.inflight().iter().all(|n| *n == 0) && engine.exact_load.lock().unwrap().is_empty() && engine.cleanup_registry().states().is_empty() { break; }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }).await.unwrap();
}
fn posts(w: &Worker) -> usize { w.wire.generations.lock().unwrap().len() }

#[tokio::test]
async fn qwen_ha_startup_and_prepost_discovery_reselection_use_current_cohort() {
    let a = worker().await; let b = worker().await; let l = worker().await;
    a.wire.cohort_offline.store(1, Ordering::SeqCst);
    let (engine,_keys) = engine(&[(&a,NativeTier::Short),(&b,NativeTier::Short),(&l,NativeTier::Long)],None);
    assert_eq!(a.wire.discoveries.load(Ordering::SeqCst),0,"startup never queries an offline origin");
    dispatch(&engine).await.unwrap(); released(&engine).await;
    assert_eq!((posts(&a),posts(&b),posts(&l)),(0,1,0));
    assert_eq!(a.wire.controls.load(Ordering::SeqCst),0,"no generation registration exists before failed discovery");
    // A normal recovery needs neither a new gateway nor a stale installed cohort.
    a.wire.artifact.lock().unwrap().native_incarnation = uuid::Uuid::new_v4().simple().to_string();
    a.wire.cohort_offline.store(0, Ordering::SeqCst);
    dispatch(&engine).await.unwrap(); released(&engine).await;
    assert_eq!((posts(&a),posts(&b)),(1,1));
}

#[tokio::test]
async fn qwen_ha_unavailable_origin_load_skips_to_sibling() {
    let a=worker().await; let b=worker().await;
    a.wire.load_offline.store(1, Ordering::SeqCst);
    let (engine,_keys)=engine(&[(&a,NativeTier::Short),(&b,NativeTier::Short)],None);
    dispatch(&engine).await.unwrap(); released(&engine).await;
    assert_eq!((posts(&a),posts(&b)),(0,1));
}

#[tokio::test]
async fn qwen_ha_all_short_offline_uses_existing_long_pool() {
    let a=worker().await; let b=worker().await; let l=worker().await;
    a.wire.load_offline.store(1, Ordering::SeqCst); b.wire.load_offline.store(1, Ordering::SeqCst);
    let (engine,_keys)=engine(&[(&a,NativeTier::Short),(&b,NativeTier::Short),(&l,NativeTier::Long)],None);
    dispatch(&engine).await.unwrap(); released(&engine).await;
    assert_eq!((posts(&a),posts(&b),posts(&l)),(0,0,1));
}

#[tokio::test]
async fn qwen_ha_private_short_scope_cannot_spill() {
    let a=worker().await;
    a.wire.load_offline.store(1, Ordering::SeqCst);
    let (engine,_keys)=engine(&[(&a,NativeTier::Short)],Some(NativeTier::Short));
    assert!(dispatch(&engine).await.is_err()); released(&engine).await;
    assert_eq!(posts(&a),0);
}

#[tokio::test]
async fn qwen_ha_long_request_never_falls_back_to_short_and_all_offline_is_bounded() {
    let a=worker().await; let l=worker().await;
    a.wire.prompt.store(20_000,Ordering::SeqCst); l.wire.prompt.store(20_000,Ordering::SeqCst);
    l.wire.load_offline.store(1,Ordering::SeqCst);
    let (engine,_keys)=engine(&[(&a,NativeTier::Short),(&l,NativeTier::Long)],None);
    assert!(tokio::time::timeout(Duration::from_secs(5),dispatch(&engine)).await.unwrap().is_err());
    released(&engine).await; assert_eq!((posts(&a),posts(&l)),(0,0));
    a.wire.prompt.store(100,Ordering::SeqCst); l.wire.prompt.store(100,Ordering::SeqCst);
    a.wire.load_offline.store(1,Ordering::SeqCst);
    assert!(tokio::time::timeout(Duration::from_secs(5),dispatch(&engine)).await.unwrap().is_err());
    released(&engine).await; assert_eq!((posts(&a),posts(&l)),(0,0));
}

#[tokio::test]
async fn qwen_ha_possible_post_failure_never_replays_generation() {
    let a=worker().await; let b=worker().await; let l=worker().await;
    a.wire.post_failure.store(1,Ordering::SeqCst);
    let (engine,_keys)=engine(&[(&a,NativeTier::Short),(&b,NativeTier::Short),(&l,NativeTier::Long)],None);
    assert!(dispatch(&engine).await.is_err()); released(&engine).await;
    assert_eq!((posts(&a),posts(&b),posts(&l)),(1,0,0));
}
