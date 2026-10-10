//! New stale-read gate over actual NativeHttpEngine, etcd, and the actual GF
//! Core/four-rank CPU control implementation. No model forward or GPU occurs.
use super::*;
use crate::cache_authority::cpu_http_marker_fixture::{
    revoke_seed_marker, seed_marker, seed_marker_expiring,
};
use crate::cache_authority::frontend_cache_hook::{NativeIdField, QualifiedWorker};
use crate::cache_authority::native_cache_frontend::{FrontendWorkerBinding, NativeCacheFrontend};
use crate::cache_authority::{EtcdStore, Prefix, cache_auth};
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    routing::{get, post},
};
use dynamo_runtime::pipeline::Context;
use serde_json::json;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

const KEY: [u8; 32] = [b'H'; 32];
const INC: &str = "00000000000040008000000000000002";

struct NativeBridge {
    child: Child,
    input: ChildStdin,
    output: Lines<BufReader<ChildStdout>>,
}
impl NativeBridge {
    async fn call(&mut self, message: Value) -> Value {
        let bytes = serde_json::to_vec(&message).unwrap();
        self.input.write_all(&bytes).await.unwrap();
        self.input.write_all(b"\n").await.unwrap();
        self.input.flush().await.unwrap();
        let line = self
            .output
            .next_line()
            .await
            .unwrap()
            .expect("actual native CPU bridge closed");
        let result: Value = serde_json::from_str(&line).unwrap();
        assert!(
            result.get("error").is_none(),
            "actual native CPU bridge error: {result}"
        );
        result
    }
}
#[derive(Clone)]
struct Fixture {
    bridge: Arc<tokio::sync::Mutex<NativeBridge>>,
    store: EtcdStore,
    marker_lease: i64,
    tokens: Value,
    metadata: Value,
    counter_only: bool,
    calls: Arc<Mutex<Vec<Value>>>,
    reserve_number: Arc<AtomicUsize>,
    last_generation: Arc<Mutex<Option<Value>>>,
    source_unavailable: bool,
}
fn response(request: &Value, mut payload: Value) -> Value {
    let mut claims = request[cache_auth::PROOF_FIELD]["claims"].clone();
    claims["full_render_digest"] = payload["prompt_cache_provenance"]["full_prompt_digest"].clone();
    claims["token_ids_digest"] = payload["prompt_cache_provenance"]["token_ids_digest"].clone();
    let digest = cache_auth::digest(&payload).unwrap();
    for key in ["operation_id", "nonce"] {
        claims[key] = cache_auth::digest(&json!([
            "native-response",
            request[cache_auth::PROOF_FIELD]["claims"][key],
            digest
        ]))
        .unwrap()
        .into();
    }
    payload[cache_auth::PROOF_FIELD] = cache_auth::sign(&KEY, &claims, &payload).unwrap();
    payload
}
async fn count(State(s): State<Fixture>, Json(body): Json<Value>) -> Json<Value> {
    if s.counter_only {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let payload = cache_auth::payload_for("render", &body).unwrap();
    cache_auth::verify_signature(&KEY, &body[cache_auth::PROOF_FIELD], &payload).unwrap();
    s.calls.lock().unwrap().push(
        json!({"operation":if s.counter_only{"counter_count"}else{"selected_count"},"body":body}),
    );
    let tokens = s.tokens.as_array().unwrap();
    let prefix = cache_auth::digest(&Value::Array(tokens[..6144].to_vec())).unwrap();
    let mut endpoints = Vec::new();
    for p in body["prompt_cache_blocks"]
        .as_array()
        .unwrap()
        .iter()
        .chain(body["prompt_cache_read_candidates"].as_array().unwrap())
    {
        endpoints.push(json!({"endpoint_id":p["endpoint_id"],"native_mark_tokens":6144,"native_page_tokens":6144,
            "native_end_byte":10,"native_mark_prefix_digest":prefix,"native_page_prefix_digest":prefix,
            "native_mark_render_prefix_digest":"6".repeat(64)}));
    }
    let c = &body[cache_auth::PROOF_FIELD]["claims"];
    Json(response(
        &body,
        json!({"count":tokens.len(),"tokens":s.tokens,"max_model_len":1_048_576,
        "prompt_cache_provenance":{"native_origin_uuid":c["native_incarnation"],"runtime_profile_digest":c["runtime_profile_digest"],
            "namespace_digest":c["namespace_digest"],"full_prompt_digest":"4".repeat(64),
            "token_ids_digest":cache_auth::digest(&s.tokens).unwrap(),"num_prompt_tokens":tokens.len(),"endpoints":endpoints}}),
    ))
}
async fn loads(State(s): State<Fixture>) -> Json<Value> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    let ids = if s.counter_only {
        json!(["CPU-occupied-request"])
    } else {
        json!([])
    };
    Json(
        json!({"loads":[{"dp_rank":0,"num_running_reqs":if s.counter_only{1}else{0},"num_waiting_reqs":0,
        "active_request_ids":ids,"timestamp":now,"native_incarnation":INC}]}),
    )
}
async fn reserve(State(s): State<Fixture>, Json(body): Json<Value>) -> (StatusCode, Json<Value>) {
    assert!(!s.counter_only);
    let number = s.reserve_number.fetch_add(1, Ordering::SeqCst);
    if number == 0 && s.source_unavailable {
        assert_eq!(body["prompt_cache_read_source_expires_at_ms"], 1_000_003);
        s.bridge
            .lock()
            .await
            .call(json!({"operation":"advance_clock","seconds":0.004}))
            .await;
    }
    let result = s
        .bridge
        .lock()
        .await
        .call(json!({"operation":"control","path":"/v1/cache/reservations","body":body}))
        .await;
    s.calls
        .lock()
        .unwrap()
        .push(json!({"operation":"reserve","body":body,"response":result}));
    let status = StatusCode::from_u16(result["status"].as_u64().unwrap() as u16).unwrap();
    if !status.is_success() {
        eprintln!("actual native reserve rejected: {}", result["body"]);
    }
    if number == 0 && s.source_unavailable {
        assert_eq!(
            status,
            StatusCode::CONFLICT,
            "actual expired native source must reject before ACK"
        );
        assert_eq!(
            result["body"]["error"]["type"],
            "cache_read_source_unavailable"
        );
        assert!(result["body"].get("reservation_ack").is_none());
    } else if number == 0 && status.is_success() {
        assert_eq!(body["prompt_cache_read_max_tokens"], 6144);
        // The actual native four-rank ACK now exists. Expire the logical read
        // before it returns to the Frontend, exercising its real reprepare path.
        revoke_seed_marker(&s.store, s.marker_lease).await.unwrap();
    }
    (status, Json(result["body"].clone()))
}
async fn cancel(State(s): State<Fixture>, Json(body): Json<Value>) -> (StatusCode, Json<Value>) {
    let result = s
        .bridge
        .lock()
        .await
        .call(json!({"operation":"control","path":"/v1/cache/reservations/cancel","body":body}))
        .await;
    s.calls
        .lock()
        .unwrap()
        .push(json!({"operation":"cancel","body":body,"response":result}));
    (
        StatusCode::from_u16(result["status"].as_u64().unwrap() as u16).unwrap(),
        Json(result["body"].clone()),
    )
}
async fn receipts(State(s): State<Fixture>, Json(body): Json<Value>) -> (StatusCode, Json<Value>) {
    let result = s
        .bridge
        .lock()
        .await
        .call(json!({"operation":"control","path":"/v1/cache/reservations/receipts","body":body}))
        .await;
    s.calls
        .lock()
        .unwrap()
        .push(json!({"operation":"receipts","body":body,"response":result}));
    (
        StatusCode::from_u16(result["status"].as_u64().unwrap() as u16).unwrap(),
        Json(result["body"].clone()),
    )
}
async fn generate(State(s): State<Fixture>, Json(body): Json<Value>) -> (StatusCode, Json<Value>) {
    assert_eq!(body["prompt_cache_read_max_tokens"], 0);
    assert!(body.get("prompt_cache_read_commit_point").is_none());
    *s.last_generation.lock().unwrap() = Some(body.clone());
    let result = s
        .bridge
        .lock()
        .await
        .call(json!({"operation":"generate","body":body}))
        .await;
    s.calls
        .lock()
        .unwrap()
        .push(json!({"operation":"generate","body":body,"response":result}));
    (
        StatusCode::from_u16(result["status"].as_u64().unwrap() as u16).unwrap(),
        Json(result["body"].clone()),
    )
}
fn router(state: Fixture) -> Router {
    Router::new()
        .route("/tokenize", post(count))
        .route("/v1/loads", get(loads))
        .route("/v1/cache/reservations", post(reserve))
        .route("/v1/cache/reservations/cancel", post(cancel))
        .route("/v1/cache/reservations/receipts", post(receipts))
        .route("/v1/chat/completions", post(generate))
        .with_state(state)
}
fn worker(url: &str) -> NativeWorkerConfig {
    NativeWorkerConfig {
        url: url.into(),
        tier: NativeTier::Shared,
        max_inflight: None,
        load_path: Some("/v1/loads?include=core".into()),
        max_input_tokens: None,
        capability_source: None,
        active_request_ids_source: Some("CPU actual same-snapshot fixture".into()),
        native_dp_size: Some(1),
        native_id_format: NativeIdFormat::VllmChatId,
        identity: None,
        thinking_budget_qualification: None,
        forced_zero_qualification: None,
    }
}
fn qualified(url: &str, metadata: &Value) -> QualifiedWorker {
    let mut clocks: BTreeMap<u32, BTreeMap<String, String>> = BTreeMap::new();
    for row in metadata["rank_tier_clocks"].as_array().unwrap() {
        clocks
            .entry(row["rank_id"].as_u64().unwrap() as u32)
            .or_default()
            .insert(
                row["tier"].as_str().unwrap().into(),
                row["native_clock_domain"].as_str().unwrap().into(),
            );
    }
    QualifiedWorker {
        worker_url: url.into(),
        native_origin_uuid: metadata["native_incarnation"].as_str().unwrap().into(),
        runtime_profile_digest: metadata["runtime_profile_digest"].as_str().unwrap().into(),
        namespace_digest: metadata["namespace_digest"].as_str().unwrap().into(),
        native_clock_domain: metadata["native_clock_domain"].as_str().unwrap().into(),
        required_tiers: metadata["required_tiers"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().into())
            .collect(),
        native_id_field: NativeIdField::RequestId,
        rank_tier_clocks: clocks,
    }
}
fn decode_hex(s: &str) -> [u8; 32] {
    let mut b = [0; 32];
    for (i, p) in s.as_bytes().chunks_exact(2).enumerate() {
        b[i] = u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap();
    }
    b
}

#[tokio::test]
async fn actual_engine_stale_read_cancel_reprepare_native_ledger_http_gate() {
    run_native_read_reprepare(false).await;
}

#[tokio::test]
async fn actual_engine_signed_source_unavailable_pre_ack_http_gate() {
    run_native_read_reprepare(true).await;
}

async fn run_native_read_reprepare(source_unavailable: bool) {
    let store = EtcdStore::connect(vec![std::env::var("CACHE_AUTHORITY_TEST_ETCD").unwrap()])
        .await
        .unwrap();
    let a = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let b = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url_a = format!("http://{}", a.local_addr().unwrap());
    let url_b = format!("http://{}", b.local_addr().unwrap());
    let namespace = cache_auth::namespace_for("cpu-stale", None, &"5".repeat(64)).unwrap();
    let mut child = Command::new(std::env::var("CACHE_AUTHORITY_TEST_PYTHON").unwrap())
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/native_stale_cpu_bridge.py"
        ))
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(
            std::fs::File::create(format!(
                "{}.gf-stale.stderr",
                std::env::var("CACHE_AUTHORITY_TEST_NATIVE_PROOF").unwrap()
            ))
            .unwrap(),
        )
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let child_pid = child.id().unwrap();
    let input = child.stdin.take().unwrap();
    let output = BufReader::new(child.stdout.take().unwrap()).lines();
    let bridge = Arc::new(tokio::sync::Mutex::new(NativeBridge {
        child,
        input,
        output,
    }));
    let initial=bridge.lock().await.call(json!({"operation":"init","identity":{"native_origin":url_b,"native_origin_uuid":INC,
        "source_digest":"1".repeat(64),"runtime_profile_digest":"2".repeat(64),"namespace_digest":namespace,"rank_ids":[0,1,2,3]}})).await;
    assert_eq!(initial["tokens"].as_array().unwrap().len(), 7000);
    let metadata = initial["metadata"].clone();
    let tokens = initial["tokens"].clone();
    let prefix_digest =
        cache_auth::digest(&Value::Array(tokens.as_array().unwrap()[..6144].to_vec())).unwrap();
    let q = qualified(&url_b, &metadata);
    let prefix = Prefix {
        tenant: "cpu-stale".into(),
        namespace: namespace.clone(),
        runtime: q.registry_runtime(),
        digest: decode_hex(&prefix_digest),
        marked_endpoint_digest: decode_hex(&prefix_digest),
        marked_render_digest: [0x66; 32],
        marked_tokens: 6144,
        tokens: 6144,
    };
    let marker_lease = if source_unavailable {
        seed_marker_expiring(&store, prefix, &url_b, INC, 1_000_002, 1_000_003)
            .await
            .unwrap()
    } else {
        seed_marker(&store, prefix, &url_b, INC, 1000002)
            .await
            .unwrap()
    };
    let calls = Arc::new(Mutex::new(Vec::new()));
    let last_generation = Arc::new(Mutex::new(None));
    let state = Fixture {
        bridge: bridge.clone(),
        store: store.clone(),
        marker_lease,
        tokens,
        metadata: metadata.clone(),
        counter_only: false,
        calls: calls.clone(),
        reserve_number: Arc::new(AtomicUsize::new(0)),
        last_generation: last_generation.clone(),
        source_unavailable,
    };
    let mut counter = state.clone();
    counter.counter_only = true;
    let server_a = tokio::spawn(async move { axum::serve(a, router(counter)).await.unwrap() });
    let server_b = tokio::spawn(async move { axum::serve(b, router(state)).await.unwrap() });
    let engine = NativeHttpEngine::new(NativeHttpEngineConfig {
        model: "glm-5.3-flash".into(),
        workers: vec![worker(&url_a), worker(&url_b)],
        qualification_tier: None,
    })
    .unwrap();
    let bindings = [&url_a, &url_b]
        .into_iter()
        .map(|url| FrontendWorkerBinding {
            worker: qualified(url, &metadata),
            model: "glm-5.3-flash".into(),
            source_digest: "1".repeat(64),
            kv_compat_digest: "5".repeat(64),
            count_path: "/tokenize".into(),
            tenant: "cpu-stale".into(),
            default_cache_salt: "cpu-stale".into(),
            secret: KEY,
            shared_backend: None,
        })
        .collect();
    let frontend = Arc::new(
        NativeCacheFrontend::new(engine.native_http_client(), store, bindings)
            .unwrap()
            .use_cpu_fixture_clock()
            .with_cleanup_registry(engine.cleanup_registry()),
    );
    let engine = engine.with_cache_frontend(frontend).unwrap();
    let body = json!({"model":"glm-5.3-flash","messages":[{"role":"user","content":"CPU fixed native encoder fixture"}],"max_tokens":16,"stream":false});
    let context =
        Context::with_id_and_metadata(body.clone(), "owned-stale-http-id".into(), BTreeMap::new())
            .context();
    let cache = MessagesCacheContext {
        origin: HttpIngressOrigin {
            request_id: "owned-stale-http-id".into(),
            started_at: Instant::now().checked_sub(Duration::from_secs(12)).unwrap(),
        },
        plan: crate::native_cache_plan::CachePlan {
            body: body.clone(),
            writes: vec![json!({"endpoint_id":"write:cpu","ttl_ms":300000})],
            reads: vec![json!({"endpoint_id":"read:cpu"})],
            message_order: vec![],
        },
    };
    let forwarded = engine
        .forward_messages_with_cache_count(body, context, cache)
        .await;
    if let Err(error) = &forwarded {
        std::fs::write(std::env::var("CACHE_AUTHORITY_TEST_NATIVE_PROOF").unwrap(),
            serde_json::to_vec_pretty(&json!({"passed":false,"error":error.to_string(),"http_records":calls.lock().unwrap().clone(),"native_child_pid":child_pid})).unwrap()).unwrap();
        let mut child = bridge.lock().await;
        child.child.kill().await.unwrap();
        child.child.wait().await.unwrap();
    }
    let (count, mut stream) = forwarded.unwrap();
    assert_eq!(count, 7000);
    let terminal = stream.next().await.unwrap();
    std::fs::write(std::env::var("CACHE_AUTHORITY_TEST_NATIVE_PROOF").unwrap(),
        serde_json::to_vec_pretty(&json!({"passed":false,"http_records":calls.lock().unwrap().clone(),"native_child_pid":child_pid})).unwrap()).unwrap();
    let output = terminal.into_data().unwrap().unwrap();
    assert_eq!(
        output["choices"][0]["message"]["content"],
        "CPU admission only"
    );
    assert_ne!(
        output["usage"][VERIFIED_CACHE_ACCOUNTING_FIELD]["cache_read_input_tokens"], 6999,
        "worker forged the Engine's verified accounting bridge"
    );
    assert!(
        output["choices"][0]["message"]
            .get(VERIFIED_CACHE_ACCOUNTING_FIELD)
            .is_none()
    );
    assert!(stream.next().await.is_none());
    let records = calls.lock().unwrap().clone();
    assert_eq!(
        records
            .iter()
            .map(|r| r["operation"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec![
            "counter_count",
            "selected_count",
            "reserve",
            "cancel",
            "reserve",
            "generate",
            "receipts",
            "cancel"
        ]
    );
    let first_start =
        records[0]["body"][cache_auth::PROOF_FIELD]["claims"]["request_started_at_ms"].clone();
    let native_id =
        records[0]["body"][cache_auth::PROOF_FIELD]["claims"]["native_request_id"].clone();
    for record in &records {
        let claims = &record["body"][cache_auth::PROOF_FIELD]["claims"];
        assert_eq!(
            claims["request_started_at_ms"], first_start,
            "request origin drift: {}",
            record["operation"]
        );
        assert_eq!(
            claims["native_request_id"], native_id,
            "native ID drift: {}",
            record["operation"]
        );
    }
    let reserves = records
        .iter()
        .filter(|r| r["operation"] == "reserve")
        .collect::<Vec<_>>();
    let query = records
        .iter()
        .find(|r| r["operation"] == "receipts")
        .unwrap();
    let events = query["response"]["body"]["receipts"].as_array().unwrap();
    assert!(
        events
            .iter()
            .any(|event| event["event_kind"] == "write_commit"),
        "nonzero generation must query actual signed write_commit, not an empty receipt list"
    );
    let cancelled = records.last().unwrap();
    assert_eq!(cancelled["response"]["status"], 200);
    assert_eq!(cancelled["response"]["body"]["cancelled"], true);
    assert_eq!(reserves[0]["body"]["prompt_cache_read_max_tokens"], 6144);
    assert_eq!(reserves[1]["body"]["prompt_cache_read_max_tokens"], 0);
    assert!(
        reserves[1]["body"]["authority_revision"].as_i64().unwrap()
            > reserves[0]["body"]["authority_revision"].as_i64().unwrap()
    );
    assert_ne!(
        reserves[1]["body"]["reservation_id"],
        reserves[0]["body"]["reservation_id"]
    );
    let gen_body = last_generation.lock().unwrap().clone().unwrap();
    assert_eq!(
        gen_body["authority_revision"],
        reserves[1]["body"]["authority_revision"]
    );
    // A fresh signed operation and higher reservation revision must not permit
    // a second generation with an executed native ID, even after cancellation.
    let mut replay = reserves[1]["body"].clone();
    let mut claims = replay[cache_auth::PROOF_FIELD]["claims"].clone();
    replay["reservation_id"] = json!("cpu-illegal-second-execution");
    replay["authority_revision"] = json!(gen_body["authority_revision"].as_i64().unwrap() + 100);
    claims["reservation_id"] = replay["reservation_id"].clone();
    claims["authority_revision"] = replay["authority_revision"].clone();
    claims["operation_id"] = json!("cpu-fresh-second-execution-op");
    claims["nonce"] = json!("cpu-fresh-second-execution-nonce");
    replay
        .as_object_mut()
        .unwrap()
        .remove(cache_auth::PROOF_FIELD);
    let payload = cache_auth::payload_for("reserve", &replay).unwrap();
    replay[cache_auth::PROOF_FIELD] = cache_auth::sign(&KEY, &claims, &payload).unwrap();
    let rejected = bridge
        .lock()
        .await
        .call(json!({"operation":"control","path":"/v1/cache/reservations","body":replay}))
        .await;
    assert!(
        rejected["status"].as_u64().unwrap() >= 400,
        "actual native ledger allowed executed native ID with a fresh revision"
    );
    let fence = bridge
        .lock()
        .await
        .call(json!({"operation":"fence","native_request_id":native_id}))
        .await;
    assert_eq!(fence["consumed"], true);
    assert_eq!(fence["revision"], gen_body["authority_revision"]);
    let proof = json!({"scope":"actual NativeHttpEngine/etcd/GF Core and four-rank CPU ledgers; synthetic signed Count encoder/startup identity; no GPU or model forward", "scenario":if source_unavailable {"signed sourceUnavailable409 before ACK"} else {"logical source revoked after ACK"},
        "native_child_pid":child_pid,"initial_native_fixture":initial,"http_records":records,"fresh_revision_native_rejection":rejected,"native_execution_fence":fence,
        "generation_posts":1,"native_model_forwards":0,"counter_recounts":2,"preserved_start":first_start,"preserved_native_id":native_id});
    std::fs::write(
        std::env::var("CACHE_AUTHORITY_TEST_NATIVE_PROOF").unwrap(),
        serde_json::to_vec_pretty(&proof).unwrap(),
    )
    .unwrap();
    server_a.abort();
    server_b.abort();
    let mut bridge = bridge.lock().await;
    bridge.child.kill().await.unwrap();
    let _ = bridge.child.wait().await.unwrap();
    eprintln!(
        "actual stale-read HTTP/native ledger gate PASS: same origin/RID across two Counts; cancel->higher revision/read0/new ACK->one generation; fresh revision after execution rejected; CPU child {child_pid} closed"
    );
}
