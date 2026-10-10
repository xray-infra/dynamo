//! Actual NativeHttpEngine HTTP dispatch gate. CPU/mock only; no live native
//! generation, real weights, GPU retention, or Chrony qualification is claimed.
use super::*;
use crate::cache_authority::frontend_cache_hook::{NativeIdField, QualifiedWorker};
use crate::cache_authority::native_cache_frontend::{FrontendWorkerBinding, NativeCacheFrontend};
use crate::cache_authority::{EtcdStore, cache_auth};
use axum::{
    Json, Router,
    extract::State,
    routing::{get, post},
};
use dynamo_runtime::pipeline::Context;
use serde_json::json;

const CPU_KEY: [u8; 32] = [b'H'; 32];
const CPU_INC: &str = "00000000000040008000000000000001";

#[derive(Clone)]
struct CpuHttpState {
    calls: Arc<Mutex<Vec<String>>>,
}

fn signed_response(request: &Value, mut payload: Value, operation: &str) -> Value {
    let mut claims = request[cache_auth::PROOF_FIELD]["claims"].clone();
    claims["operation"] = operation.into();
    if operation == "render" {
        claims["full_render_digest"] =
            payload["prompt_cache_provenance"]["full_prompt_digest"].clone();
        claims["token_ids_digest"] = payload["prompt_cache_provenance"]["token_ids_digest"].clone();
    }
    let digest = cache_auth::digest(&payload).unwrap();
    for field in ["operation_id", "nonce"] {
        claims[field] = cache_auth::digest(&json!([
            "native-response",
            request[cache_auth::PROOF_FIELD]["claims"][field],
            digest
        ]))
        .unwrap()
        .into();
    }
    payload[cache_auth::PROOF_FIELD] = cache_auth::sign(&CPU_KEY, &claims, &payload).unwrap();
    payload
}

async fn count(State(state): State<CpuHttpState>, Json(body): Json<Value>) -> Json<Value> {
    state.calls.lock().unwrap().push("count".into());
    let claims = &body[cache_auth::PROOF_FIELD]["claims"];
    let protected = cache_auth::payload_for("render", &body).unwrap();
    cache_auth::verify_signature(&CPU_KEY, &body[cache_auth::PROOF_FIELD], &protected).unwrap();
    let tokens = json!((0..10).collect::<Vec<u32>>());
    let mut endpoints = Vec::new();
    for selector in body["prompt_cache_blocks"]
        .as_array()
        .unwrap()
        .iter()
        .chain(body["prompt_cache_read_candidates"].as_array().unwrap())
    {
        endpoints.push(json!({"endpoint_id":selector["endpoint_id"], "native_mark_tokens":8, "native_page_tokens":0,
            "native_end_byte":10, "native_mark_prefix_digest":cache_auth::digest(&json!((0..8).collect::<Vec<u32>>())).unwrap(),
            "native_page_prefix_digest":cache_auth::digest(&json!([])).unwrap(), "native_mark_render_prefix_digest":"6".repeat(64)}));
    }
    let result = json!({"tokens":tokens, "count":10, "max_model_len":1_048_576,
        "prompt_cache_provenance":{"native_origin_uuid":claims["native_incarnation"],
            "runtime_profile_digest":claims["runtime_profile_digest"], "namespace_digest":claims["namespace_digest"],
            "full_prompt_digest":"4".repeat(64), "token_ids_digest":cache_auth::digest(&tokens).unwrap(),
            "num_prompt_tokens":10, "endpoints":endpoints}});
    Json(signed_response(&body, result, "render"))
}

async fn reserve(State(state): State<CpuHttpState>, Json(body): Json<Value>) -> Json<Value> {
    state.calls.lock().unwrap().push("reserve".into());
    let protected = cache_auth::payload_for("reserve", &body).unwrap();
    cache_auth::verify_signature(&CPU_KEY, &body[cache_auth::PROOF_FIELD], &protected).unwrap();
    let claims = &body[cache_auth::PROOF_FIELD]["claims"];
    assert_eq!(body["token_ids"], json!((0..10).collect::<Vec<u32>>()));
    let points = body["prompt_cache_commit_points"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            assert_eq!(p["native_page_tokens"], 0);
            assert_eq!(p["retain_for_ms"], 0);
            json!({"endpoint_id":p["endpoint_id"], "tier_acks":[]})
        })
        .collect::<Vec<_>>();
    Json(signed_response(
        &body,
        json!({"native_request_id":body["native_request_id"],
        "reservation_ack":{"reservation_id":body["reservation_id"], "authority_revision":body["authority_revision"],
            "native_origin_uuid":claims["native_incarnation"], "source_digest":claims["source_digest"],
            "runtime_profile_digest":claims["runtime_profile_digest"], "namespace_digest":claims["namespace_digest"],
            "rank_acks":[{"rank_id":0, "point_acks":points}]}}),
        "receipts",
    ))
}

async fn loads(State(state): State<CpuHttpState>) -> Json<Value> {
    state.calls.lock().unwrap().push("loads".into());
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    Json(
        json!({"loads":[{"dp_rank":0, "num_running_reqs":0, "num_waiting_reqs":0,
        "active_request_ids":[], "native_incarnation":CPU_INC, "timestamp":now}]}),
    )
}

async fn generate(State(state): State<CpuHttpState>, Json(body): Json<Value>) -> Json<Value> {
    state.calls.lock().unwrap().push("generate".into());
    let payload = cache_auth::payload_for("generate", &body).unwrap();
    cache_auth::verify_signature(&CPU_KEY, &body[cache_auth::PROOF_FIELD], &payload).unwrap();
    assert!(
        body["prompt_cache_commit_points"]
            .as_array()
            .unwrap()
            .iter()
            .all(|p| p["no_pin"] == true)
    );
    assert_eq!(body["prompt_cache_read_max_tokens"], 0);
    assert!(body["cache_request_started_at_ms"].as_u64().unwrap() < 1_000_000 - 10_000);
    Json(
        json!({"id":"cpu-generation", "choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"OK"}}],
        "usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11},
        "prompt_cache_authority_proof":{"private":"must not reach public output"}}),
    )
}

async fn receipts(State(state): State<CpuHttpState>, Json(body): Json<Value>) -> Json<Value> {
    state.calls.lock().unwrap().push("receipts".into());
    let payload = cache_auth::payload_for("receipts", &body).unwrap();
    cache_auth::verify_signature(&CPU_KEY, &body[cache_auth::PROOF_FIELD], &payload).unwrap();
    Json(json!({"native_request_id":body["native_request_id"],"receipts":[]}))
}
async fn cancel(State(state): State<CpuHttpState>, Json(body): Json<Value>) -> Json<Value> {
    state.calls.lock().unwrap().push("cancel".into());
    let payload = cache_auth::payload_for("cancel", &body).unwrap();
    cache_auth::verify_signature(&CPU_KEY, &body[cache_auth::PROOF_FIELD], &payload).unwrap();
    Json(signed_response(
        &body,
        json!({"native_request_id":body["native_request_id"],"reservation_id":body["reservation_id"],
        "authority_revision":body["authority_revision"],"cancelled":true,"rank_results":[{"rank_id":0,"cancelled":true}]}),
        "receipts",
    ))
}

fn engine(url: &str) -> NativeHttpEngine {
    NativeHttpEngine::new(NativeHttpEngineConfig {
        model: "glm-5.3-flash".into(),
        qualification_tier: None,
        workers: vec![NativeWorkerConfig {
            url: url.into(),
            tier: NativeTier::Shared,
            max_inflight: None,
            load_path: Some("/v1/loads?include=core".into()),
            max_input_tokens: None,
            capability_source: None,
            active_request_ids_source: Some("CPU mock exact IDs".into()),
            native_dp_size: Some(1),
            native_id_format: NativeIdFormat::VllmChatId,
            identity: None,
            thinking_budget_qualification: None,
            forced_zero_qualification: None,
        }],
    })
    .unwrap()
}
fn cache_context(request_id: &str) -> MessagesCacheContext {
    MessagesCacheContext {
        origin: HttpIngressOrigin {
            request_id: request_id.into(),
            started_at: Instant::now().checked_sub(Duration::from_secs(12)).unwrap(),
        },
        plan: crate::native_cache_plan::CachePlan {
            body: json!({}),
            writes: vec![json!({"endpoint_id":"write:cpu","ttl_ms":300_000})],
            reads: vec![json!({"endpoint_id":"read:cpu"})],
            message_order: vec![],
        },
    }
}

#[tokio::test]
async fn actual_engine_same_port_zero_page_cache_http_gate() {
    let endpoint = std::env::var("CACHE_AUTHORITY_TEST_ETCD")
        .expect("disposable owned etcd endpoint required");
    let calls = Arc::new(Mutex::new(Vec::new()));
    let state = CpuHttpState {
        calls: calls.clone(),
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let router = Router::new()
        .route("/tokenize", post(count))
        .route("/v1/loads", get(loads))
        .route("/v1/cache/reservations", post(reserve))
        .route("/v1/chat/completions", post(generate))
        .route("/v1/cache/reservations/receipts", post(receipts))
        .route("/v1/cache/reservations/cancel", post(cancel))
        .with_state(state);
    let server = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    let plain_engine = engine(&url);
    let body = json!({"model":"glm-5.3-flash","messages":[{"role":"user","content":"ten CPU mock tokens"}],"max_tokens":16,"stream":false});
    let context =
        Context::with_id_and_metadata(body.clone(), "owned-http-id".into(), BTreeMap::new())
            .context();
    let rejected = plain_engine
        .forward_messages_with_cache_count(
            body.clone(),
            context.clone(),
            cache_context("owned-http-id"),
        )
        .await;
    assert!(rejected.is_err());
    assert!(
        calls.lock().unwrap().is_empty(),
        "unqualified TTL must fail before native side effects"
    );
    let binding = FrontendWorkerBinding {
        worker: QualifiedWorker {
            worker_url: url.clone(),
            native_origin_uuid: CPU_INC.into(),
            runtime_profile_digest: "2".repeat(64),
            namespace_digest: "3".repeat(64),
            native_clock_domain: "cpu-monotonic".into(),
            required_tiers: BTreeSet::from(["l1".into()]),
            native_id_field: NativeIdField::RequestId,
            rank_tier_clocks: BTreeMap::from([(
                0,
                BTreeMap::from([("l1".into(), "cpu-monotonic".into())]),
            )]),
        },
        model: "glm-5.3-flash".into(),
        source_digest: "1".repeat(64),
        kv_compat_digest: "5".repeat(64),
        count_path: "/tokenize".into(),
        tenant: "cpu-test".into(),
        default_cache_salt: "cpu-test".into(),
        secret: CPU_KEY,
        shared_backend: None,
    };
    let store = EtcdStore::connect(vec![endpoint]).await.unwrap();
    let frontend =
        NativeCacheFrontend::new(plain_engine.native_http_client(), store, vec![binding])
            .unwrap()
            .use_cpu_fixture_clock()
            .with_cleanup_registry(plain_engine.cleanup_registry());
    let engine = plain_engine
        .with_cache_frontend(Arc::new(frontend))
        .unwrap();
    let (count, mut stream) = engine
        .forward_messages_with_cache_count(body, context, cache_context("owned-http-id"))
        .await
        .unwrap();
    assert_eq!(count, 10);
    let output = stream.next().await.unwrap().into_data().unwrap().unwrap();
    assert_eq!(output["choices"][0]["message"]["content"], "OK");
    assert!(output.get("prompt_cache_authority_proof").is_none());
    assert!(stream.next().await.is_none());
    assert_eq!(
        *calls.lock().unwrap(),
        vec![
            "count", "loads", "reserve", "generate", "receipts", "cancel"
        ]
    );
    server.abort();
    eprintln!(
        "actual NativeHttpEngine Count -> reserve ACK -> one same-port mock generation -> receipts -> authenticated cancel PASS; zero live native generations"
    );
}
