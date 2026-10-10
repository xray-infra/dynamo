//! Synthetic CPU HTTP/qualification material. No GPU or installed native proof.
use axum::{
    Json, Router,
    body::Body,
    extract::State,
    response::Response,
    routing::{get, post},
};
use dynamo_llm::{
    cache_authority::cache_auth,
    native_cleanup::NativeCleanupRegistry,
    native_cleanup_cohort::{
        CleanupCohortApproval, CleanupCohortArtifact, CleanupCohortParticipant,
        install_cleanup_cohort,
    },
    native_pd::{
        control::PdRole,
        dispatch::{PdDispatchError, PdDispatcher, RoomRange},
        qualification::{PdRoleArtifact, PdRoleMaterialFiles, VerifiedPdPair, load_pd_role},
    },
    native_request_lifecycle::QualifiedRequestLifecycle,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    os::unix::fs::PermissionsExt,
    path::Path,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

const SECRET: [u8; 32] = [13; 32];
const ROOT: &str = "1234567890abcdef1234567890abcdef";
const TXN: &str = "abcdef1234567890abcdef1234567890";
const NORMAL_TRACKING: &str = "Reference ID : CB00710F (CPU-only tracking stand-in)\nSystem time : 0 seconds slow of NTP time\nRoot delay : 0 seconds\nRoot dispersion : 0 seconds\nLeap status : Normal\n";
struct CpuClock {
    _directory: tempfile::TempDir,
    tracking: std::path::PathBuf,
    prepared: tokio::sync::OnceCell<Result<(), String>>,
}
fn cpu_clock() -> &'static CpuClock {
    static CLOCK: OnceLock<CpuClock> = OnceLock::new();
    CLOCK.get_or_init(|| {
        let directory = tempfile::tempdir().unwrap();
        let tracking = directory.path().join("CPU-tracking");
        std::fs::write(&tracking, NORMAL_TRACKING).unwrap();
        let executable = directory.path().join("chronyc");
        std::fs::write(
            &executable,
            b"#!/bin/sh\nexec /bin/cat \"$DYNAMO_PD_CPU_TRACKING\"\n",
        )
        .unwrap();
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
        let existing = std::env::var_os("PATH").unwrap_or_default();
        let path = std::env::join_paths(
            std::iter::once(directory.path().to_path_buf()).chain(std::env::split_paths(&existing)),
        )
        .unwrap();
        // This test binary is run with --test-threads=1. Initialize before any
        // clock-qualified request; this command stand-in is CPU evidence only.
        // Production still executes the real chronyc with no fallback/provider.
        unsafe {
            std::env::set_var("PATH", path);
            std::env::set_var("DYNAMO_PD_CPU_TRACKING", &tracking);
        }
        CpuClock {
            _directory: directory,
            tracking,
            prepared: tokio::sync::OnceCell::new(),
        }
    })
}
async fn prepare_cpu_clock() {
    use std::process::Stdio;
    use tokio::{io::AsyncReadExt, process::Command};
    let prepared = cpu_clock()
        .prepared
        .get_or_init(|| async {
            // The sealed raw diagnostic measured 0.420ms spawn and 266.658ms
            // stdout reading for the first CPU shim execution. Pay that fixture
            // startup cost once before any qualification. This is a raw command,
            // not a Clock.qualify call, retry, or source of ClockBounds. Every
            // subsequent qualification retains the production 100ms deadline.
            let mut child = Command::new("chronyc")
                .args(["-n", "tracking"])
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .kill_on_drop(true)
                .spawn()
                .map_err(|error| format!("raw CPU fixture spawn failed: {error}"))?;
            let mut stdout = child.stdout.take().unwrap();
            let (bytes, status) = tokio::time::timeout(Duration::from_secs(3), async {
                let mut bytes = Vec::new();
                (&mut stdout)
                    .take(16_385)
                    .read_to_end(&mut bytes)
                    .await
                    .map_err(|error| format!("raw CPU fixture read failed: {error}"))?;
                let status = child
                    .wait()
                    .await
                    .map_err(|error| format!("raw CPU fixture wait failed: {error}"))?;
                Ok::<_, String>((bytes, status))
            })
            .await
            .map_err(|_| "raw CPU fixture preparation exceeded 3s; no qualification attempted")??;
            if !status.success() || bytes != NORMAL_TRACKING.as_bytes() {
                return Err(format!(
                    "raw CPU fixture preparation invalid: status={status}, bytes={}",
                    bytes.len()
                ));
            }
            Ok(())
        })
        .await;
    // Cache failures too: fixture setup never retries a raw command and cannot
    // silently turn an invalid preparation into a qualified-clock attempt.
    prepared.as_ref().unwrap();
}
async fn qualified_start() -> u64 {
    prepare_cpu_clock().await;
    dynamo_llm::cache_authority::frontend_clock::ChronyClock
        .qualify()
        .await
        .unwrap()
        .request_start_lower_ms()
}
struct RestoreCpuTracking;
impl Drop for RestoreCpuTracking {
    fn drop(&mut self) {
        std::fs::write(&cpu_clock().tracking, NORMAL_TRACKING).unwrap();
    }
}
fn bad_cpu_tracking() {
    std::fs::write(
        &cpu_clock().tracking,
        NORMAL_TRACKING.replace("Leap status : Normal", "Leap status : Not synchronised"),
    )
    .unwrap();
}
fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}
#[derive(Default)]
struct Wire {
    d_arrived: AtomicBool,
    d_notify: Notify,
    p_release: Notify,
    p_fail: AtomicBool,
    p_empty: AtomicBool,
    hold: AtomicBool,
    transfer: AtomicBool,
    calls: AtomicUsize,
    controls: AtomicUsize,
    bodies: Mutex<Vec<Value>>,
    registry: Mutex<Option<NativeCleanupRegistry>>,
    incarnation: Mutex<String>,
    counts: Mutex<Vec<Value>>,
}
#[derive(Clone)]
struct Role {
    wire: Arc<Wire>,
    prefill: bool,
}
struct Server {
    origin: String,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}
async fn server(wire: Arc<Wire>, prefill: bool) -> Server {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let app = Router::new()
        .route("/v1/tokenize",post(|State(role):State<Role>,Json(body):Json<Value>| async move {
            role.wire.counts.lock().unwrap().push(body);
            Json(json!({"count":80,"max_model_len":1_048_576}))
        }))
        .route("/v1/loads",get(|State(role):State<Role>| async move {
            let timestamp=SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64();
            // Unfrozen scheduler snapshots have no immutable-publication
            // heartbeat. A positive heartbeat with frozen=false is invalid
            // under the actual native snapshot contract and must fence.
            Json(json!({"loads":[{"dp_rank":0,"frozen":false,"snapshot_generation":1,"heartbeat_timestamp":0.0,"timestamp":timestamp,"num_running_reqs":0,"num_waiting_reqs":0,"active_request_ids":[],"native_incarnation":role.wire.incarnation.lock().unwrap().clone()}]}))
        }))
        .route("/v1/chat/completions", post(generation))
        .route("/v1/requests/status", post(ordinary))
        .route("/v1/requests/abort", post(ordinary))
        .route("/v1/pd/transactions/status", post(pd))
        .route("/v1/pd/transactions/abort", post(pd))
        .with_state(Role { wire, prefill });
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    Server { origin, task }
}
async fn generation(State(role): State<Role>, Json(packet): Json<Value>) -> Response {
    let w = &role.wire;
    assert_eq!(
        w.registry.lock().unwrap().as_ref().unwrap().states().len(),
        4,
        "all ordinary and PD entries exist before either native POST"
    );
    let payload = cache_auth::payload_for("request-generate", &packet).unwrap();
    let claims =
        cache_auth::verify_signature(&SECRET, &packet[cache_auth::PROOF_FIELD], &payload).unwrap();
    assert_eq!(claims["operation"], "request-generate");
    assert_eq!(
        packet["prompt_cache_request_control"]["transaction_id"],
        packet["prompt_cache_pd_binding"]["control"]["transaction_id"]
    );
    assert_eq!(
        packet["prompt_cache_request_control"]["request_root"],
        packet["rid"]
    );
    assert_eq!(packet["max_tokens"], 77);
    assert_eq!(packet["temperature"], 0.3);
    assert_eq!(claims["native_request_id"], packet["rid"]);
    w.calls.fetch_add(1, Ordering::SeqCst);
    w.bodies.lock().unwrap().push(packet.clone());
    if role.prefill {
        while !w.d_arrived.load(Ordering::SeqCst) {
            let notified = w.d_notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if !w.d_arrived.load(Ordering::SeqCst) {
                notified.await;
            }
        }
        if w.p_fail.load(Ordering::SeqCst) || w.p_empty.load(Ordering::SeqCst) {
            let wire = w.clone();
            let body = async_stream::stream! {wire.p_release.notified().await;if wire.p_fail.load(Ordering::SeqCst) {yield Ok::<_,std::io::Error>(bytes::Bytes::from_static(b"{\"error\":\"P failed after D headers\"}"));}};
            return Response::builder().body(Body::from_stream(body)).unwrap();
        }
    } else {
        w.d_arrived.store(true, Ordering::SeqCst);
        w.d_notify.notify_waiters();
    }
    let n = packet.get("n").and_then(Value::as_u64).unwrap_or(1);
    let choices:Vec<_>=(0..n).map(|i|json!({"index":i,"message":{"role":"assistant","content":if role.prefill{"native P boundary"}else{"native D output"}},"finish_reason":"stop"})).collect();
    let completion = if role.prefill { 1 } else { 13 };
    Response::builder()
        .header("content-type","application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({"id":"synthetic-native-pd","model":packet["model"],"object":"chat.completion","created":1,"choices":choices,"usage":{"prompt_tokens":80,"completion_tokens":completion,"total_tokens":80+completion,"cached_tokens":17,"completion_tokens_details":{"reasoning_tokens":2}}})).unwrap(),
        ))
        .unwrap()
}
fn verified_packet(mut packet: Value) -> (Value, Value) {
    let proof = packet
        .as_object_mut()
        .unwrap()
        .remove(cache_auth::PROOF_FIELD)
        .unwrap();
    let claims = cache_auth::verify_signature(&SECRET, &proof, &packet).unwrap();
    (packet, claims)
}
fn signed(mut reply: Value, mut claims: Value) -> Json<Value> {
    let event = cache_auth::digest(&reply).unwrap();
    for key in ["operation_id", "nonce"] {
        claims[key] =
            json!(cache_auth::digest(&json!(["native-response", claims[key], event])).unwrap());
    }
    reply[cache_auth::PROOF_FIELD] = cache_auth::sign(&SECRET, &claims, &reply).unwrap();
    Json(reply)
}
async fn ordinary(State(role): State<Role>, Json(packet): Json<Value>) -> Json<Value> {
    let (c, claims) = verified_packet(packet);
    let hold = role.wire.hold.load(Ordering::SeqCst);
    role.wire.controls.fetch_add(1, Ordering::SeqCst);
    let participants:Vec<_>=["opaque-helper","opaque-scheduler","opaque-tokenizer"].into_iter().map(|id|json!({"participant_id":id,"ingress_closed":!hold,"tokenizer_drained":!hold,"scheduler_drained":!hold,"transport_drained":!hold,"live_request_holds":u64::from(hold),"live_kv_holds":0,"live_meta_holds":0,"live_io_holds":0,"abort_ack":!hold})).collect();
    signed(
        json!({"schema_version":1,"transaction_id":c["transaction_id"],"request_root":c["request_root"],"native_incarnation":c["expected_native_incarnation"],"cleanup_cohort_digest":c["expected_cleanup_cohort_digest"],"ordinary_body_digest":c["ordinary_body_digest"],"expected_choices":c["expected_choices"],"captured_at_ms":now(),"participants":participants,"terminal":!hold,"aborted":!hold}),
        claims,
    )
}
async fn pd(State(role): State<Role>, Json(packet): Json<Value>) -> Json<Value> {
    let (c, claims) = verified_packet(packet);
    let hold = role.wire.hold.load(Ordering::SeqCst);
    role.wire.controls.fetch_add(1, Ordering::SeqCst);
    let abort = claims["operation"] == "pd-abort" && !hold;
    let transfer = role.wire.transfer.load(Ordering::SeqCst);
    let ranks:Vec<_>=(0..8).map(|tp|json!({"tp_rank":tp,"dcp_rank":tp%4,"replica_rank":tp/4,"rooms":c["rooms"].as_array().unwrap().iter().map(|r|json!({"room":r["room"],"sample_rid":r["sample_rid"],"state":if hold{"transferring"}else{"finished"},"transfer_complete":transfer,"decode_committed":transfer,"live_kv_holds":u64::from(hold),"live_meta_holds":0,"live_io_holds":0,"abort_ack":abort,"transport_drained":!hold})).collect::<Vec<_>>()})).collect();
    signed(
        json!({"schema_version":1,"transaction_id":c["transaction_id"],"role":c["role"],"native_incarnation":c["expected_native_incarnation"],"request_root":c["request_root"],"rooms":c["rooms"],"captured_at_ms":now(),"qualified_rank_set_digest":c["expected_qualified_rank_set_digest"],"rank_states":ranks,"terminal":!hold,"aborted":abort}),
        claims,
    )
}
fn write(path: &Path, bytes: &[u8]) {
    std::fs::write(path, bytes).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o400)).unwrap();
}
fn document(path: &Path, value: &Value) -> Vec<u8> {
    let bytes = serde_json::to_vec(value).unwrap();
    write(path, &bytes);
    bytes
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn material(
    dir: &Path,
    p_origin: &str,
    d_origin: &str,
    bootstrap_port: u16,
) -> (
    PdRoleMaterialFiles,
    PdRoleMaterialFiles,
    PdRoleArtifact,
    PdRoleArtifact,
) {
    let native: Value = serde_json::from_str(include_str!(
        "../src/native_pd/role-qualification-v1.synthetic.json"
    ))
    .unwrap();
    let mut p: PdRoleArtifact = serde_json::from_value(native["artifact"].clone()).unwrap();
    let mut d: PdRoleArtifact = serde_json::from_value(native["peer_artifact"].clone()).unwrap();
    p.model = "zai-org/GLM-5.3".into();
    d.model = p.model.clone();
    let asset = dir.join("CPU-synthetic-source");
    write(
        &asset,
        b"CPU synthetic source bytes; not native/GPU qualification",
    );
    let hash = digest(&std::fs::read(&asset).unwrap());
    let secret = dir.join("secret");
    write(&secret, &SECRET);
    let sources = json!([{"path":"/synthetic-native/source.py","sha256":hash}]);
    let enc = json!([{"path":"/synthetic-native/tokenizer.json","sha256":hash}]);
    for (a, origin) in [(&mut p, p_origin), (&mut d, d_origin)] {
        a.native_origin = origin.into();
        for r in a.registrations.iter_mut() {
            r.native_rank_incarnation = uuid::Uuid::new_v4().simple().to_string();
        }
        a.native_incarnation = a.registrations[0].native_rank_incarnation.clone();
        if a.role == PdRole::Prefill {
            a.prefill_bootstrap_host = Some("127.0.0.1".into());
            a.prefill_bootstrap_port = Some(bootstrap_port);
            a.prefill_bootstrap_bind_host = Some("127.0.0.1".into());
            for r in &mut a.registrations {
                r.transport.bootstrap_registration.as_mut().unwrap().url =
                    format!("http://127.0.0.1:{bootstrap_port}");
            }
        }
        a.native_role_registration_digest =
            cache_auth::digest(&json!({"ranks":a.ranks,"registrations":a.registrations})).unwrap();
        for k in [
            "model_manifest_sha256",
            "config_sha256",
            "tokenizer_sha256",
            "position_sha256",
        ] {
            a.kv_compat_descriptor[k] = json!(hash);
        }
        assert!(
            cache_auth::canonical_bytes(&a.kv_compat_descriptor)
                .unwrap()
                .is_ascii()
        );
        a.kv_compat_digest = cache_auth::digest(&a.kv_compat_descriptor).unwrap();
        a.source_digest = cache_auth::digest(&sources).unwrap();
    }
    let mut profile = json!({"version":1,"family":"glm","sources":sources,"encoder_assets":enc,"weights_attestation_digest":p.weights_attestation_digest,"layout":{"page_size":64,"tp_size":8,"pp_size":1,"dcp_size":4,"device_capacity":128,"kv_compat_descriptor":p.kv_compat_descriptor}});
    let runtime = cache_auth::digest(&profile).unwrap();
    profile["runtime_profile_digest"] = json!(runtime);
    p.runtime_profile_digest = runtime.clone();
    d.runtime_profile_digest = runtime;
    let pp = dir.join("profile");
    let profile_bytes = document(&pp, &profile);
    let mp = dir.join("manifest");
    let manifest = document(
        &mp,
        &json!({"version":1,"family":"glm","profile_sha256":digest(&profile_bytes),"qualification_receipt_id":"synthetic-weights-controller-only","assets":[{"kind":"source","native_path":"/synthetic-native/source.py","installed_path":asset,"sha256":hash},{"kind":"encoder","native_path":"/synthetic-native/tokenizer.json","installed_path":asset,"sha256":hash}]}),
    );
    let mut files = Vec::new();
    for (name, a, peer) in [("prefill", &p, &d), ("decode", &d, &p)] {
        let dp = dir.join(format!("{name}-deployment"));
        document(
            &dp,
            &json!({"version":1,"model":a.model,"native_origin":a.native_origin,"weights_attestation_digest":a.weights_attestation_digest,"qualification_receipt_id":"synthetic-weights-controller-only","authority_secret_path":secret}),
        );
        let rp = dir.join(format!("{name}-receipt"));
        document(
            &rp,
            &json!({"schema_version":1,"qualification_receipt_id":"synthetic-role-controller-only","authority_secret_path":secret,"artifact":a,"artifact_digest":a.artifact_digest().unwrap(),"peer_artifact":peer,"peer_artifact_digest":peer.artifact_digest().unwrap()}),
        );
        files.push(PdRoleMaterialFiles {
            role_receipt_path: rp,
            deployment_path: dp,
            profile_path: pp.clone(),
            frontend_asset_manifest_path: mp.clone(),
            expected_frontend_asset_manifest_sha256: digest(&manifest),
            expected_artifact_digest: a.artifact_digest().unwrap(),
            expected_native_incarnation: a.native_incarnation.clone(),
        });
    }
    (files.remove(0), files.remove(0), p, d)
}
fn lifecycle(
    a: &PdRoleArtifact,
    registry: &NativeCleanupRegistry,
    wrong_profile: bool,
) -> Arc<QualifiedRequestLifecycle> {
    let participants: Vec<_> = ["opaque-helper", "opaque-scheduler", "opaque-tokenizer"]
        .into_iter()
        .map(|id| CleanupCohortParticipant {
            participant_id: id.into(),
            role: format!("CPU-role-{id}"),
            process_birth: format!("CPU-birth-{id}"),
            source_digest: "a".repeat(64),
            runtime_registration_digest: "b".repeat(64),
        })
        .collect();
    let ids: BTreeSet<_> = participants
        .iter()
        .map(|p| p.participant_id.clone())
        .collect();
    let c = CleanupCohortArtifact {
        schema_version: 1,
        native_origin: a.native_origin.clone(),
        native_incarnation: a.native_incarnation.clone(),
        model: a.model.clone(),
        source_digest: a.source_digest.clone(),
        runtime_profile_digest: if wrong_profile {
            "0".repeat(64)
        } else {
            a.runtime_profile_digest.clone()
        },
        cleanup_cohort_digest: cache_auth::digest(&json!(ids)).unwrap(),
        participants,
    };
    let approval = CleanupCohortApproval {
        schema_version: 1,
        artifact_digest: c.artifact_digest().unwrap(),
    };
    Arc::new(
        QualifiedRequestLifecycle::new(
            install_cleanup_cohort(&approval, c.clone(), &c).unwrap(),
            SECRET,
            &a.native_origin,
            registry.clone(),
        )
        .unwrap(),
    )
}
struct Fixture {
    dispatcher: Arc<PdDispatcher>,
    peer_dispatcher: PdDispatcher,
    pair: Arc<VerifiedPdPair>,
    wire: Arc<Wire>,
    registry: NativeCleanupRegistry,
    model: String,
    _servers: [Server; 2],
    _files: tempfile::TempDir,
    _bootstrap: tokio::net::TcpListener,
}
async fn fixture(wrong_profile: bool) -> Result<Fixture, String> {
    cpu_clock();
    let wire = Arc::new(Wire::default());
    wire.transfer.store(true, Ordering::SeqCst);
    let p = server(wire.clone(), true).await;
    let d = server(wire.clone(), false).await;
    let bootstrap = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let dir = tempfile::tempdir().unwrap();
    let (pf, df, pa, da) = material(
        dir.path(),
        &p.origin,
        &d.origin,
        bootstrap.local_addr().unwrap().port(),
    );
    let pair = Arc::new(VerifiedPdPair::from_roles(
        load_pd_role(&pf, &pa.model, &pa.native_origin, PdRole::Prefill)?,
        load_pd_role(&df, &da.model, &da.native_origin, PdRole::Decode)?,
    )?);
    let registry = NativeCleanupRegistry::new();
    *wire.registry.lock().unwrap() = Some(registry.clone());
    *wire.incarnation.lock().unwrap() = da.native_incarnation.clone();
    let p_lifecycle = lifecycle(&pa, &registry, wrong_profile);
    let d_lifecycle = lifecycle(&da, &registry, false);
    let dispatcher = Arc::new(
        PdDispatcher::new(pair.clone(), p_lifecycle.clone(), d_lifecycle.clone())
            .map_err(|e| e.to_string())?,
    );
    let peer_dispatcher =
        PdDispatcher::new(pair.clone(), p_lifecycle, d_lifecycle).map_err(|e| e.to_string())?;
    Ok(Fixture {
        dispatcher,
        peer_dispatcher,
        pair,
        wire,
        registry,
        model: pa.model,
        _servers: [p, d],
        _files: dir,
        _bootstrap: bootstrap,
    })
}
fn body(f: &Fixture, n: u32) -> Value {
    json!({"model":f.model,"rid":ROOT,"n":n,"messages":[{"role":"user","content":"CPU synthetic PD"}],"temperature":0.3,"max_tokens":77,"stream":false})
}
struct Hold(Arc<AtomicUsize>);
impl Drop for Hold {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[tokio::test]
async fn pd_dispatch_concurrent_roles_and_physical_transfer_before_terminal() {
    let f = fixture(false).await.unwrap();
    let drops = Arc::new(AtomicUsize::new(0));
    let plan = f
        .dispatcher
        .plan(
            body(&f, 3),
            ROOT,
            3,
            RoomRange::new(100, 3).unwrap(),
            qualified_start().await,
            TXN,
        )
        .await
        .unwrap();
    let mut generation = tokio::time::timeout(
        Duration::from_secs(3),
        plan.dispatch(Box::new(Hold(drops.clone())), CancellationToken::new()),
    )
    .await
    .unwrap()
    .unwrap();
    let response = generation
        .take_decode_response()
        .unwrap()
        .json::<Value>()
        .await
        .unwrap();
    assert_eq!(
        response["choices"][0]["message"]["content"],
        "native D output"
    );
    f.wire.transfer.store(false, Ordering::SeqCst);
    assert!(
        tokio::time::timeout(
            Duration::from_millis(150),
            generation.qualify_successful_terminal()
        )
        .await
        .is_err(),
        "P HTTP completion cannot substitute for native transfer receipt"
    );
    f.wire.transfer.store(true, Ordering::SeqCst);
    tokio::time::timeout(
        Duration::from_secs(2),
        generation.qualify_successful_terminal(),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert_eq!(f.wire.calls.load(Ordering::SeqCst), 2);
    let bodies = f.wire.bodies.lock().unwrap();
    assert_eq!(
        bodies[0]["prompt_cache_canonical_body"],
        bodies[1]["prompt_cache_canonical_body"]
    );
    assert_eq!(bodies[0]["bootstrap_room"], 100);
    assert_eq!(
        bodies[0]["prompt_cache_pd_binding"]["control"]["rooms"]
            .as_array()
            .unwrap()
            .len(),
        4
    );
    drop(bodies);
    drop(generation);
    tokio::time::timeout(Duration::from_secs(3), f.registry.wait_for_physical_zero())
        .await
        .unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}
#[tokio::test]
async fn pd_dispatch_p_failure_after_d_headers_and_empty_p_never_success() {
    for empty in [false, true] {
        let f = fixture(false).await.unwrap();
        f.wire.p_fail.store(!empty, Ordering::SeqCst);
        f.wire.p_empty.store(empty, Ordering::SeqCst);
        let plan = f
            .dispatcher
            .plan(
                body(&f, 1),
                ROOT,
                1,
                RoomRange::new(7, 1).unwrap(),
                qualified_start().await,
                TXN,
            )
            .await
            .unwrap();
        let mut generation = tokio::time::timeout(
            Duration::from_secs(3),
            plan.dispatch(Box::new(()), CancellationToken::new()),
        )
        .await
        .unwrap()
        .unwrap();
        let _d = generation.take_decode_response().unwrap();
        f.wire.p_release.notify_one();
        assert!(matches!(
            tokio::time::timeout(
                Duration::from_secs(2),
                generation.qualify_successful_terminal()
            )
            .await
            .unwrap(),
            Err(PdDispatchError::Prefill(_))
        ));
        assert_eq!(
            f.wire.calls.load(Ordering::SeqCst),
            2,
            "no unknown generation replay"
        );
        drop(generation);
        tokio::time::timeout(Duration::from_secs(3), f.registry.wait_for_physical_zero())
            .await
            .unwrap();
    }
}
#[tokio::test]
async fn pd_dispatch_cancel_retains_all_four_receipt_holds_and_shutdown() {
    let f = fixture(false).await.unwrap();
    f.wire.hold.store(true, Ordering::SeqCst);
    let drops = Arc::new(AtomicUsize::new(0));
    let cancel = CancellationToken::new();
    let plan = f
        .dispatcher
        .allocate_plan(body(&f, 3), ROOT, 3, qualified_start().await)
        .await
        .unwrap();
    assert_eq!(plan.room_range().base, 1);
    let mut generation = plan
        .dispatch(Box::new(Hold(drops.clone())), cancel.clone())
        .await
        .unwrap();
    cancel.cancel();
    assert!(matches!(
        generation.qualify_successful_terminal().await,
        Err(PdDispatchError::Cancelled)
    ));
    drop(generation);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert_eq!(f.registry.states().len(), 4);
    let retained = f
        .peer_dispatcher
        .allocate_plan(body(&f, 3), ROOT, 3, qualified_start().await)
        .await
        .unwrap();
    assert_eq!(
        retained.room_range().base,
        5,
        "blocked old physical hold still owns samples 1..3 and warmup 4"
    );
    assert!(
        tokio::time::timeout(
            Duration::from_millis(200),
            f.registry.wait_for_physical_zero()
        )
        .await
        .is_err()
    );
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    f.wire.hold.store(false, Ordering::SeqCst);
    tokio::time::timeout(Duration::from_secs(3), f.registry.wait_for_physical_zero())
        .await
        .unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    let recycled = f
        .peer_dispatcher
        .allocate_plan(body(&f, 3), ROOT, 3, qualified_start().await)
        .await
        .unwrap();
    assert_eq!(
        recycled.room_range().base,
        1,
        "only all four ACKs allow reuse"
    );
    drop(retained);
    drop(recycled);
}
#[tokio::test]
async fn pd_dispatch_shared_p_allocator_is_atomic_and_failed_plan_releases_interval() {
    let f = fixture(false).await.unwrap();
    assert!(matches!(
        f.dispatcher
            .allocate_plan(body(&f, u32::MAX), ROOT, u32::MAX, qualified_start().await)
            .await,
        Err(PdDispatchError::ResourceBound)
    ));
    let plans = futures::future::join_all((0..16).map(|j| {
        let dispatcher = if j % 2 == 0 {
            f.dispatcher.as_ref()
        } else {
            &f.peer_dispatcher
        };
        let mut b = body(&f, 3);
        let root = format!("{j:032x}");
        b["rid"] = json!(root);
        async move {
            dispatcher
                .allocate_plan(b, &root, 3, qualified_start().await)
                .await
                .unwrap()
        }
    }))
    .await;
    let bases: BTreeSet<_> = plans.iter().map(|plan| plan.room_range().base).collect();
    assert_eq!(bases, (0..16).map(|j| 1 + j * 4).collect());
    let next = f
        .peer_dispatcher
        .allocate_plan(body(&f, 3), ROOT, 3, qualified_start().await)
        .await
        .unwrap();
    assert_eq!(next.room_range().base, 65);
    drop(plans);
    let recycled = f
        .dispatcher
        .allocate_plan(body(&f, 3), ROOT, 3, qualified_start().await)
        .await
        .unwrap();
    assert_eq!(recycled.room_range().base, 1);
    assert_eq!(f.wire.calls.load(Ordering::SeqCst), 0);
}
#[tokio::test]
async fn pd_dispatch_private_metadata_profile_and_large_n_fail_before_posts() {
    assert!(fixture(true).await.is_err());
    let f = fixture(false).await.unwrap();
    for field in [
        "bootstrap_host",
        "prompt_cache_pd_binding",
        "prompt_cache_authority_proof",
        "prompt_cache_request_control",
    ] {
        let mut b = body(&f, 1);
        b[field] = json!("client");
        assert!(
            f.dispatcher
                .plan(
                    b,
                    ROOT,
                    1,
                    RoomRange::new(1, 1).unwrap(),
                    qualified_start().await,
                    TXN
                )
                .await
                .is_err()
        );
    }
    assert!(matches!(
        f.dispatcher
            .plan(
                body(&f, u32::MAX),
                ROOT,
                u32::MAX,
                RoomRange::new(1, u32::MAX).unwrap(),
                qualified_start().await,
                TXN
            )
            .await,
        Err(PdDispatchError::ResourceBound)
    ));
    assert_eq!(f.wire.calls.load(Ordering::SeqCst), 0);
    assert!(f.registry.states().is_empty());
}
#[test]
fn pd_dispatch_rooms_match_actual_native_helper_export_and_wrap() {
    let native: Value = serde_json::from_str(include_str!(
        "../../../../../native-glm/NATIVE-CHOICE-WIRE-EXAMPLES.json"
    ))
    .unwrap();
    for example in native["examples"].as_array().unwrap() {
        let input = &example["input"];
        let n = input["n"].as_u64().unwrap() as u32;
        let base = input["bootstrap_room"].as_i64().unwrap();
        let Some(expected) = example.get("rooms") else {
            continue;
        };
        let rooms = RoomRange::new(base, n).unwrap().bindings(ROOT).unwrap();
        let actual = serde_json::to_value(rooms).unwrap();
        let mut expected = expected.clone();
        for r in expected.as_array_mut().unwrap() {
            r["sample_rid"] = json!(r["sample_rid"].as_str().unwrap().replacen('R', ROOT, 1));
        }
        assert_eq!(actual, expected, "{}", example["name"]);
    }
    let range = RoomRange::new(i64::MAX - 2, 3).unwrap();
    assert_eq!(range.warmup, Some(1));
    assert!(RoomRange::new(i64::MAX - 1, 3).is_err());
    assert!(RoomRange::new(0, 1).is_err());
}

struct HttpFrontend {
    url: String,
    cancel: CancellationToken,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for HttpFrontend {
    fn drop(&mut self) {
        self.cancel.cancel();
        self.task.abort();
    }
}

#[tokio::test]
async fn pd_dispatch_actual_http_service_keeps_d_output_usage_and_one_count() {
    use dynamo_llm::{
        engines::StreamingEngineAdapter,
        http::service::service_v2::HttpService,
        model_card::ModelDeploymentCard,
        native_contract::NativeTier,
        native_http_engine::{
            NativeHttpEngine, NativeHttpEngineConfig, NativeIdFormat, NativeWorkerConfig,
        },
    };
    let f = fixture(false).await.unwrap();
    prepare_cpu_clock().await;
    let d = f.pair.decode().artifact();
    let p = f.pair.prefill().artifact();
    let engine = NativeHttpEngine::new(NativeHttpEngineConfig {
        model: f.model.clone(),
        qualification_tier: Some(NativeTier::Short),
        qualified_model_identity: None,
        workers: vec![NativeWorkerConfig {
            url: d.native_origin.clone(),
            tier: NativeTier::Short,
            max_inflight: Some(2),
            load_path: Some("/v1/loads?include=core".into()),
            max_input_tokens: Some(65_536),
            capability_source: Some("synthetic CPU PD decode input boundary".into()),
            active_request_ids_source: Some(
                "synthetic CPU same-snapshot IDs/rank/UUID4/frozen/heartbeat".into(),
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
    *f.wire.registry.lock().unwrap() = Some(registry.as_ref().clone());
    let dispatcher = PdDispatcher::new(
        f.pair.clone(),
        lifecycle(p, &registry, false),
        lifecycle(d, &registry, false),
    )
    .unwrap();
    let engine = Arc::new(engine.with_pd_dispatchers(vec![dispatcher]).unwrap());
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
    let card = ModelDeploymentCard::with_name_only(&f.model);
    service
        .model_manager()
        .add_chat_completions_model(
            &f.model,
            card.mdcsum(),
            Arc::new(StreamingEngineAdapter::new(engine.clone())),
        )
        .unwrap();
    let stop = cancel.clone();
    let task = tokio::spawn(async move {
        service.run_with_listener(stop, listener).await.unwrap();
    });
    let frontend = HttpFrontend {
        url: format!("http://127.0.0.1:{port}"),
        cancel,
        task,
    };
    let request = json!({"model":f.model,"messages":[{"role":"user","content":"CPU synthetic PD HttpService"}],"n":3,"temperature":0.3,"max_tokens":77,"stream":false});
    let response = tokio::time::timeout(
        Duration::from_secs(5),
        reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .build()
            .unwrap()
            .post(format!("{}/v1/chat/completions", frontend.url))
            .json(&request)
            .send(),
    )
    .await
    .unwrap()
    .unwrap();
    let status = response.status();
    let reply: Value = response.json().await.unwrap();
    assert_eq!(
        status,
        reqwest::StatusCode::OK,
        "{reply}; count_calls={}, pd_generation_calls={}, control_calls={}",
        f.wire.counts.lock().unwrap().len(),
        f.wire.calls.load(Ordering::SeqCst),
        f.wire.controls.load(Ordering::SeqCst)
    );
    assert_eq!(reply["choices"].as_array().unwrap().len(), 3);
    for c in reply["choices"].as_array().unwrap() {
        assert_eq!(c["message"]["content"], "native D output");
    }
    assert_eq!(reply["usage"]["prompt_tokens"], 80);
    assert_eq!(reply["usage"]["completion_tokens"], 13);
    assert_eq!(reply["usage"]["total_tokens"], 93);
    assert_eq!(
        f.wire.calls.load(Ordering::SeqCst),
        2,
        "exactly two role POSTs, no generation replay"
    );
    let bodies = f.wire.bodies.lock().unwrap();
    assert_eq!(bodies.len(), 2);
    assert_eq!(
        bodies[0]["prompt_cache_canonical_body"],
        bodies[1]["prompt_cache_canonical_body"]
    );
    assert_eq!(bodies[0]["n"], 3);
    assert_eq!(bodies[1]["max_tokens"], 77);
    assert_eq!(
        bodies[0]["prompt_cache_pd_binding"]["control"]["rooms"]
            .as_array()
            .unwrap()
            .len(),
        4
    );
    drop(bodies);
    let counts = f.wire.counts.lock().unwrap();
    assert_eq!(
        counts.len(),
        1,
        "P/D does not introduce a second native Count"
    );
    assert_eq!(counts[0]["n"], 3);
    assert!(counts[0].get("prompt_cache_pd_binding").is_none());
    drop(counts);
    tokio::time::timeout(Duration::from_secs(3), registry.wait_for_physical_zero())
        .await
        .unwrap();
    assert!(registry.states().is_empty());
    assert!(
        f.wire.controls.load(Ordering::SeqCst) >= 6,
        "two natural PD status proofs plus four cleanup controls"
    );
}

#[tokio::test]
async fn pd_dispatch_e28_clock_failure_closes_signing_without_release_or_new_window() {
    use dynamo_llm::{
        cache_authority::frontend_clock::ChronyClock,
        native_request_cleanup::ORDINARY_REQUEST_WINDOW_MS,
    };
    // Run this target with --test-threads=1. The CPU tracking command is an
    // explicit stand-in; wall/monotonic sampling and freshness are the real
    // ChronyClock code, and no production clock fallback is introduced.
    let f = fixture(false).await.unwrap();
    let _restore = RestoreCpuTracking;
    let start = qualified_start().await;
    bad_cpu_tracking();
    assert!(
        f.dispatcher
            .allocate_plan(body(&f, 1), ROOT, 1, start)
            .await
            .is_err()
    );
    assert_eq!(f.wire.calls.load(Ordering::SeqCst), 0);
    assert!(f.registry.states().is_empty());
    std::fs::write(&cpu_clock().tracking, NORMAL_TRACKING).unwrap();
    let clock = ChronyClock.qualify().await.unwrap();
    let expired_start = clock.lower_ms() - ORDINARY_REQUEST_WINDOW_MS;
    assert!(matches!(
        f.dispatcher
            .allocate_plan(body(&f, 1), ROOT, 1, expired_start)
            .await,
        Err(PdDispatchError::Deadline)
    ));
    let plan = f
        .dispatcher
        .allocate_plan(body(&f, 1), ROOT, 1, qualified_start().await)
        .await
        .unwrap();
    let drops = Arc::new(AtomicUsize::new(0));
    bad_cpu_tracking();
    assert!(
        plan.dispatch(Box::new(Hold(drops.clone())), CancellationToken::new())
            .await
            .is_err()
    );
    assert_eq!(
        drops.load(Ordering::SeqCst),
        1,
        "pre-registration signer rejection owns no native resources"
    );
    assert_eq!(f.wire.calls.load(Ordering::SeqCst), 0);
    assert!(f.registry.states().is_empty());
    std::fs::write(&cpu_clock().tracking, NORMAL_TRACKING).unwrap();
    let plan = f
        .dispatcher
        .allocate_plan(body(&f, 1), ROOT, 1, qualified_start().await)
        .await
        .unwrap();
    assert_eq!(
        plan.room_range().base,
        1,
        "failed clock-qualified plans do not leak rooms"
    );
    let held = Arc::new(AtomicUsize::new(0));
    let mut generation = plan
        .dispatch(Box::new(Hold(held.clone())), CancellationToken::new())
        .await
        .unwrap();
    generation
        .take_decode_response()
        .unwrap()
        .json::<Value>()
        .await
        .unwrap();
    generation
        .prefill_monitor
        .wait_http_success()
        .await
        .unwrap();
    let ordinary = generation.ordinary_recovery.clone();
    let pd = generation.pd_recovery.clone();
    bad_cpu_tracking();
    assert!(generation.qualify_successful_terminal().await.is_err());
    assert_eq!(
        f.wire.controls.load(Ordering::SeqCst),
        0,
        "unqualified control claims are never posted"
    );
    drop(generation);
    let blocked = tokio::time::timeout(Duration::from_secs(2), f.registry.await_quiescence())
        .await
        .unwrap()
        .unwrap_err();
    assert_eq!(blocked.entries.len(), 4);
    assert_eq!(held.load(Ordering::SeqCst), 0);
    std::fs::write(&cpu_clock().tracking, NORMAL_TRACKING).unwrap();
    for recovery in ordinary {
        recovery.retry_blocked(&f.registry).unwrap();
    }
    for recovery in pd {
        recovery.retry_blocked(&f.registry).unwrap();
    }
    tokio::time::timeout(Duration::from_secs(3), f.registry.wait_for_physical_zero())
        .await
        .unwrap();
    assert_eq!(
        held.load(Ordering::SeqCst),
        1,
        "fresh clock alone cannot release; four verified native ACKs do"
    );
}

#[tokio::test]
async fn pd_dispatch_cpu_clock_command_stage_diagnostic() {
    use std::process::Stdio;
    use tokio::{io::AsyncReadExt, process::Command, time::Instant};
    // One raw command in a fresh filtered test process, after the identical
    // fixture startup used by E28. This does NOT qualify a clock, return bounds,
    // retry a failed qualification, or change the production 100ms limit.
    let f = fixture(false).await.unwrap();
    let path = cpu_clock()._directory.path().join("chronyc");
    let before = Instant::now();
    let mut child = Command::new("chronyc")
        .args(["-n", "tracking"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let spawned = Instant::now();
    eprintln!(
        "CPU chronyc raw diagnostic: command={path:?} argv=[-n,tracking] spawn_ms={:.3}",
        spawned.duration_since(before).as_secs_f64() * 1000.0
    );
    let mut stdout = child.stdout.take().unwrap();
    let result=tokio::time::timeout(Duration::from_secs(3),async {
        let mut bytes=Vec::new();(&mut stdout).take(16_385).read_to_end(&mut bytes).await.unwrap();
        let eof=Instant::now();
        eprintln!("CPU chronyc raw diagnostic: stdout_eof_ms={:.3} read_after_spawn_ms={:.3} bytes={} normal_tracking={}",eof.duration_since(before).as_secs_f64()*1000.0,eof.duration_since(spawned).as_secs_f64()*1000.0,bytes.len(),bytes==NORMAL_TRACKING.as_bytes());
        let status=child.wait().await.unwrap();
        eprintln!("CPU chronyc raw diagnostic: exit_ms={:.3} wait_after_eof_ms={:.3} status={status}",before.elapsed().as_secs_f64()*1000.0,eof.elapsed().as_secs_f64()*1000.0);
        (bytes,status)
    }).await.expect("raw diagnostic exceeded 3s; no clock qualification was attempted");
    assert!(result.1.success());
    assert_eq!(result.0, NORMAL_TRACKING.as_bytes());
    assert_eq!(f.wire.calls.load(Ordering::SeqCst), 0);
    assert_eq!(f.wire.controls.load(Ordering::SeqCst), 0);
}
