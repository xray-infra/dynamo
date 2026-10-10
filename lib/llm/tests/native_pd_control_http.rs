use axum::{Json, Router, body::Body, extract::State, response::Response, routing::post};
use dynamo_llm::cache_authority::cache_auth;
use dynamo_llm::cache_authority::frontend_clock::{ChronyClock, ClockBounds};
use dynamo_llm::native_cleanup::{
    CleanupFailure, CleanupReceipt, CleanupState, NativeCleanupRegistry,
};
use dynamo_llm::native_pd::control::{
    ControlRequest, ExpectedTransaction, PdRole, PhysicalRank, RoomBinding,
};
use dynamo_llm::native_pd::transport::{
    NativePdTransport, PdClaimProvider, PdTransportError, QualifiedClaims,
};
use dynamo_llm::native_request_cleanup::{ExpectedRequest, RequestControl};
use dynamo_llm::native_request_transport::{
    NativeRequestTransport, RequestClaimProvider, RequestTransportError,
};
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const SECRET: [u8; 32] = [42; 32]; // CPU fixture only, never a production authority.
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}
fn expected(role: PdRole) -> Arc<ExpectedTransaction> {
    let ranks: BTreeSet<_> = (0..8)
        .map(|tp_rank| PhysicalRank {
            tp_rank,
            dcp_rank: tp_rank % 4,
            replica_rank: tp_rank / 4,
        })
        .collect();
    let tuples: Vec<_> = ranks
        .iter()
        .map(|r| [r.tp_rank, r.dcp_rank, r.replica_rank])
        .collect();
    let digest = cache_auth::digest(&json!(tuples)).unwrap();
    Arc::new(ExpectedTransaction {
        request: ControlRequest {
            schema_version: 1,
            transaction_id: "cpu-fixture-tx".into(),
            request_root: "cpu-fixture-root".into(),
            expected_native_incarnation: "11111111111141118111111111111111".into(),
            expected_qualified_rank_set_digest: digest,
            role,
            rooms: vec![
                RoomBinding {
                    room: 17,
                    sample_rid: "cpu-root:sample:0".into(),
                },
                RoomBinding {
                    room: 18,
                    sample_rid: "cpu-root:sample:1".into(),
                },
            ],
        },
        ranks,
    })
}
struct FixtureClaims {
    origin: String,
    expected: Arc<ExpectedTransaction>,
    nonce: AtomicUsize,
    started_at_ms: u64,
}
#[async_trait::async_trait]
impl PdClaimProvider for FixtureClaims {
    async fn clock_bounds(&self) -> Result<ClockBounds, String> {
        // Integration execution requires a declared CPU tracking-model chronyc
        // stand-in or a genuinely qualified command. No unchecked wall fallback.
        ChronyClock.qualify().await.map_err(|e| e.to_string())
    }
    async fn claims(&self, operation: &str) -> Result<QualifiedClaims, String> {
        let digest = cache_auth::digest(
            &serde_json::to_value(&self.expected.request).map_err(|e| e.to_string())?,
        )?;
        let clock = self.clock_bounds().await?;
        let now = clock.lower_ms();
        let n = self.nonce.fetch_add(1, Ordering::Relaxed);
        Ok(QualifiedClaims {
            claims: json!({"version":1,"operation":operation,"operation_id":format!("cpu-operation-{n}"),"nonce":format!("cpu-nonce-{n}"),"model":"CPU fixture only","native_origin":self.origin,"native_incarnation":self.expected.request.expected_native_incarnation,"source_digest":"a".repeat(64),"runtime_profile_digest":"b".repeat(64),"reservation_id":"","authority_revision":0,"native_request_id":self.expected.request.request_root,"namespace_digest":"","full_render_digest":"","token_ids_digest":"","request_started_at_ms":self.started_at_ms,"issued_at_ms":now,"deadline_ms":self.started_at_ms+2000,"retain_for_ms":0,"payload_digest":digest}),
            clock,
        })
    }
}
#[derive(Clone, Copy)]
enum Mode {
    Good,
    Tampered,
    PartialRank,
    WrongRoom,
    TerminalHolds,
    MissingAck,
    Stale,
    WrongIncarnation,
    Hold,
    Slow,
    Oversize,
    Unavailable,
}
#[derive(Clone)]
struct Worker {
    mode: Arc<Mutex<Mode>>,
    calls: Arc<AtomicUsize>,
    generation_calls: Arc<AtomicUsize>,
}
struct Server {
    origin: String,
    worker: Worker,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}
async fn fixture(mode: Mode) -> Server {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let worker = Worker {
        mode: Arc::new(Mutex::new(mode)),
        calls: Arc::new(AtomicUsize::new(0)),
        generation_calls: Arc::new(AtomicUsize::new(0)),
    };
    let app=Router::new().route("/v1/pd/transactions/status",post(receipt)).route("/v1/pd/transactions/abort",post(receipt))
      .route("/v1/requests/status",post(request_receipt)).route("/v1/requests/abort",post(request_receipt))
      .route("/cpu-fixture/generation",post(|State(s):State<Worker>|async move {
          s.generation_calls.fetch_add(1,Ordering::Release);
          let body=async_stream::stream! { yield Ok::<_,std::io::Error>(bytes::Bytes::from_static(b"CPU fixture in progress")); std::future::pending::<()>().await; };
          Response::builder().body(Body::from_stream(body)).unwrap()
      })).with_state(worker.clone());
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    Server {
        origin,
        worker,
        task,
    }
}
async fn receipt(
    State(worker): State<Worker>,
    uri: axum::http::Uri,
    Json(mut packet): Json<Value>,
) -> Response {
    worker.calls.fetch_add(1, Ordering::Release);
    let mode = *worker.mode.lock().unwrap();
    if matches!(mode, Mode::Unavailable) {
        return Response::builder().status(503).body(Body::empty()).unwrap();
    }
    if matches!(mode, Mode::Slow) {
        let bytes = async_stream::stream! {yield Ok::<_,std::io::Error>(bytes::Bytes::from_static(b" "));std::future::pending::<()>().await;};
        return Response::builder().body(Body::from_stream(bytes)).unwrap();
    }
    if matches!(mode, Mode::Oversize) {
        let chunks = futures::stream::iter([
            Ok::<_, std::io::Error>(bytes::Bytes::from(vec![b' '; 8 * 1024 * 1024])),
            Ok(bytes::Bytes::from_static(b"x")),
        ]);
        return Response::builder().body(Body::from_stream(chunks)).unwrap();
    }
    let proof = packet
        .as_object_mut()
        .unwrap()
        .remove(cache_auth::PROOF_FIELD)
        .unwrap();
    let mut claims = cache_auth::verify_signature(&SECRET, &proof, &packet).unwrap();
    let request: ControlRequest = serde_json::from_value(packet).unwrap();
    assert_eq!(
        claims["operation"],
        if uri.path().ends_with("abort") {
            "pd-abort"
        } else {
            "pd-status"
        }
    );
    let hold = matches!(mode, Mode::Hold);
    let aborted = uri.path().ends_with("abort") && !hold;
    let mut rank_states = Vec::new();
    for tp in 0..8 {
        rank_states.push(json!({"tp_rank":tp,"dcp_rank":tp%4,"replica_rank":tp/4,"rooms":request.rooms.iter().map(|r|json!({"room":r.room,"sample_rid":r.sample_rid,"state":if hold{"transferring"}else{"finished"},"transfer_complete":!hold,"decode_committed":!hold,"live_kv_holds":u64::from(hold),"live_meta_holds":0,"live_io_holds":0,"abort_ack":aborted,"transport_drained":!hold})).collect::<Vec<_>>() }));
    }
    let mut payload = json!({"schema_version":1,"transaction_id":request.transaction_id,"role":request.role,"native_incarnation":request.expected_native_incarnation,"request_root":request.request_root,"rooms":request.rooms,"captured_at_ms":now_ms(),"qualified_rank_set_digest":request.expected_qualified_rank_set_digest,"rank_states":rank_states,"terminal":!hold,"aborted":aborted});
    match mode {
        Mode::PartialRank => {
            payload["rank_states"].as_array_mut().unwrap().pop();
        }
        Mode::WrongRoom => {
            payload["rank_states"][7]["rooms"][1]["room"] = 19.into();
        }
        Mode::TerminalHolds => {
            payload["rank_states"][7]["rooms"][1]["live_io_holds"] = 1.into();
        }
        Mode::MissingAck => {
            payload["rank_states"][7]["rooms"][1]["abort_ack"] = false.into();
        }
        Mode::Stale => {
            payload["captured_at_ms"] = (now_ms() - 10000).into();
        }
        Mode::WrongIncarnation => {
            payload["native_incarnation"] = "22222222222242228222222222222222".into();
        }
        _ => {}
    }
    let event = cache_auth::digest(&payload).unwrap();
    for key in ["operation_id", "nonce"] {
        claims[key] = cache_auth::digest(&json!(["native-response", claims[key], event]))
            .unwrap()
            .into();
    }
    let proof = cache_auth::sign(&SECRET, &claims, &payload).unwrap();
    payload[cache_auth::PROOF_FIELD] = proof;
    if matches!(mode, Mode::Tampered) {
        payload["rank_states"][0]["rooms"][0]["live_kv_holds"] = 9.into();
    }
    Response::builder()
        .header("content-type", "application/json")
        .body(Body::from(payload.to_string()))
        .unwrap()
}
fn claims(server: &Server, expected: Arc<ExpectedTransaction>) -> Arc<FixtureClaims> {
    Arc::new(FixtureClaims {
        origin: server.origin.clone(),
        expected,
        nonce: AtomicUsize::new(0),
        started_at_ms: now_ms().saturating_sub(100),
    })
}
fn transport(server: &Server, timeout: Duration) -> Arc<NativePdTransport> {
    Arc::new(NativePdTransport::new(&server.origin, SECRET, timeout).unwrap())
}

#[tokio::test]
async fn pd_control_http_full_eight_rank_natural_and_abort_receipts_are_distinct_role_bound() {
    let server = fixture(Mode::Good).await;
    let transport = transport(&server, Duration::from_secs(1));
    let p = expected(PdRole::Prefill);
    let d = expected(PdRole::Decode);
    assert_ne!(p.cleanup_id(), d.cleanup_id());
    let provider = claims(&server, p.clone());
    let natural = transport.status(&p, provider.as_ref()).await.unwrap();
    assert!(natural.physical_terminal());
    assert!(!natural.response().aborted);
    assert_eq!(natural.response().rank_states.len(), 8);
    let aborted = transport.abort(&p, provider.as_ref()).await.unwrap();
    assert!(aborted.response().aborted);
    assert_eq!(aborted.cleanup_id(), p.cleanup_id());
    assert!(
        aborted
            .response()
            .rank_states
            .iter()
            .all(|rank| rank.rooms.iter().all(|room| room.abort_ack))
    );
    assert_eq!(server.worker.calls.load(Ordering::Acquire), 2);
}

#[tokio::test]
async fn pd_control_http_rejects_signed_partial_room_holds_ack_clock_identity_and_tamper() {
    for mode in [
        Mode::Tampered,
        Mode::PartialRank,
        Mode::WrongRoom,
        Mode::TerminalHolds,
        Mode::MissingAck,
        Mode::Stale,
        Mode::WrongIncarnation,
    ] {
        let server = fixture(mode).await;
        let expected = expected(PdRole::Decode);
        let provider = claims(&server, expected.clone());
        let transport = transport(&server, Duration::from_secs(1));
        assert!(matches!(
            transport.abort(&expected, provider.as_ref()).await,
            Err(PdTransportError::InvalidReceipt(_))
        ));
        assert_eq!(server.worker.calls.load(Ordering::Acquire), 1);
    }
}

#[tokio::test]
async fn pd_control_http_real_body_deadline_unknown_and_size_status_errors_never_ack() {
    for (mode, error) in [
        (Mode::Slow, PdTransportError::Unknown { timed_out: true }),
        (Mode::Oversize, PdTransportError::ResponseTooLarge),
        (Mode::Unavailable, PdTransportError::HttpStatus(503)),
    ] {
        let server = fixture(mode).await;
        let expected = expected(PdRole::Decode);
        let provider = claims(&server, expected.clone());
        let transport = transport(
            &server,
            if matches!(mode, Mode::Slow) {
                Duration::from_millis(40)
            } else {
                Duration::from_secs(1)
            },
        );
        let result = transport.abort(&expected, provider.as_ref()).await;
        assert_eq!(result.err().unwrap(), error);
        assert_eq!(server.worker.calls.load(Ordering::Acquire), 1);
    }
    // The private control request is bounded before signing or native POST,
    // including inputs that would otherwise create a large room set.
    let server = fixture(Mode::Good).await;
    let original = expected(PdRole::Decode);
    let mut request = original.request.clone();
    request.rooms = (0..30000)
        .map(|room| RoomBinding {
            room,
            sample_rid: format!("cpu-{room:08}-{}", "x".repeat(280)),
        })
        .collect();
    let oversized = Arc::new(ExpectedTransaction {
        request,
        ranks: original.ranks.clone(),
    });
    let provider = claims(&server, oversized.clone());
    let transport = transport(&server, Duration::from_secs(1));
    let result = transport.abort(&oversized, provider.as_ref()).await;
    assert_eq!(result.err().unwrap(), PdTransportError::RequestTooLarge);
    assert_eq!(server.worker.calls.load(Ordering::Acquire), 0);
}

struct DisconnectTrigger(Option<tokio::sync::oneshot::Sender<()>>);
impl Drop for DisconnectTrigger {
    fn drop(&mut self) {
        if let Some(sender) = self.0.take() {
            let _ = sender.send(());
        }
    }
}
#[tokio::test]
async fn pd_control_http_disconnect_cleanup_shutdown_waits_physical_rank_ack() {
    let server = fixture(Mode::Hold).await;
    let expected = expected(PdRole::Decode);
    let id = expected.cleanup_id();
    let provider = claims(&server, expected.clone());
    let transport = transport(&server, Duration::from_millis(200));
    let registry = NativeCleanupRegistry::new();
    let (sender, receiver) = tokio::sync::oneshot::channel();
    registry
        .register(id.clone(), async move {
            let _ = receiver.await;
            let receipt = transport
                .cleanup(
                    expected,
                    provider,
                    tokio::time::Instant::now() + Duration::from_secs(2),
                    Duration::from_millis(5),
                )
                .await?;
            CleanupReceipt::from_pd(receipt)
        })
        .unwrap();
    // Registration precedes the real TCP fixture request. The dummy generation
    // endpoint is explicitly CPU-only and does not claim a native P/D wire shape.
    let response = reqwest::Client::new()
        .post(format!("{}/cpu-fixture/generation", server.origin))
        .send()
        .await
        .unwrap();
    let trigger = DisconnectTrigger(Some(sender));
    drop(response);
    drop(trigger);
    tokio::time::timeout(Duration::from_secs(1), async {
        while server.worker.calls.load(Ordering::Acquire) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(registry.states().get(&id), Some(&CleanupState::Running));
    assert!(
        tokio::time::timeout(Duration::from_millis(40), registry.wait_for_physical_zero())
            .await
            .is_err()
    );
    *server.worker.mode.lock().unwrap() = Mode::Good;
    tokio::time::timeout(Duration::from_secs(1), registry.wait_for_physical_zero())
        .await
        .unwrap();
    assert!(registry.states().is_empty());
    assert_eq!(server.worker.generation_calls.load(Ordering::Acquire), 1);
}

#[tokio::test]
async fn pd_control_http_blocked_stays_sealed_until_exact_identity_verified_retry() {
    let server = fixture(Mode::Hold).await;
    let expected = expected(PdRole::Prefill);
    let id = expected.cleanup_id();
    let provider = claims(&server, expected.clone());
    let transport = transport(&server, Duration::from_millis(100));
    let registry = NativeCleanupRegistry::new();
    let e = expected.clone();
    let c = provider.clone();
    let t = transport.clone();
    registry
        .register(id.clone(), async move {
            CleanupReceipt::from_pd(
                t.cleanup(
                    e,
                    c,
                    tokio::time::Instant::now() + Duration::from_millis(40),
                    Duration::from_millis(5),
                )
                .await?,
            )
        })
        .unwrap();
    assert!(registry.await_quiescence().await.is_err());
    assert!(matches!(
        registry.states().get(&id),
        Some(CleanupState::Blocked(CleanupFailure::Pd(
            PdTransportError::Deadline
        )))
    ));
    assert!(
        tokio::time::timeout(Duration::from_millis(20), registry.wait_for_physical_zero())
            .await
            .is_err()
    );
    *server.worker.mode.lock().unwrap() = Mode::Good;
    registry
        .retry_blocked(id, async move {
            CleanupReceipt::from_pd(transport.status(&expected, provider.as_ref()).await?)
        })
        .unwrap();
    tokio::time::timeout(Duration::from_secs(1), registry.wait_for_physical_zero())
        .await
        .unwrap();
    assert!(registry.states().is_empty());
}

#[test]
fn pd_control_no_runtime_registration_retains_typed_blocked() {
    let registry = NativeCleanupRegistry::new();
    assert_eq!(
        registry.register("native:cpu-fixture".into(), async {
            Err(CleanupFailure::Verification("CPU fixture".into()))
        }),
        Err(CleanupFailure::NoRuntime)
    );
    assert_eq!(
        registry.states().get("native:cpu-fixture"),
        Some(&CleanupState::Blocked(CleanupFailure::NoRuntime))
    );
}

fn request_expected() -> Arc<ExpectedRequest> {
    let participants: BTreeSet<String> = (0..8).map(|i| format!("cpu-participant-{i}")).collect();
    Arc::new(ExpectedRequest {
        control: RequestControl {
            schema_version: 1,
            transaction_id: "2".repeat(32),
            request_root: "3".repeat(32),
            expected_native_incarnation: "11111111111141118111111111111111".into(),
            expected_cleanup_cohort_digest: cache_auth::digest(&json!(participants)).unwrap(),
            ordinary_body_digest: cache_auth::digest(&json!(
                "CPU fixture exact ordinary JSON only"
            ))
            .unwrap(),
            expected_choices: 2,
        },
        participants,
    })
}
struct RequestFixtureClaims {
    origin: String,
    expected: Arc<ExpectedRequest>,
    started: u64,
    nonce: AtomicUsize,
}
#[async_trait::async_trait]
impl RequestClaimProvider for RequestFixtureClaims {
    async fn clock_bounds(&self) -> Result<ClockBounds, String> {
        ChronyClock.qualify().await.map_err(|e| e.to_string())
    }
    async fn claims(&self, operation: &str) -> Result<QualifiedClaims, String> {
        let digest = cache_auth::digest(
            &serde_json::to_value(&self.expected.control).map_err(|e| e.to_string())?,
        )?;
        let clock = self.clock_bounds().await?;
        let n = self.nonce.fetch_add(1, Ordering::Relaxed);
        Ok(QualifiedClaims {
            claims: json!({"version":1,"operation":operation,"operation_id":format!("cpu-request-operation-{n}"),"nonce":format!("cpu-request-nonce-{n}"),"model":"CPU fixture only","native_origin":self.origin,"native_incarnation":self.expected.control.expected_native_incarnation,"source_digest":"a".repeat(64),"runtime_profile_digest":"b".repeat(64),"reservation_id":"","authority_revision":0,"native_request_id":self.expected.control.request_root,"namespace_digest":"","full_render_digest":"","token_ids_digest":"","request_started_at_ms":self.started,"issued_at_ms":clock.lower_ms(),"deadline_ms":self.started+2000,"retain_for_ms":0,"payload_digest":digest}),
            clock,
        })
    }
}
fn request_claims(server: &Server, expected: Arc<ExpectedRequest>) -> Arc<RequestFixtureClaims> {
    Arc::new(RequestFixtureClaims {
        origin: server.origin.clone(),
        expected,
        started: now_ms().saturating_sub(100),
        nonce: AtomicUsize::new(0),
    })
}
async fn request_receipt(
    State(worker): State<Worker>,
    uri: axum::http::Uri,
    Json(mut packet): Json<Value>,
) -> Response {
    worker.calls.fetch_add(1, Ordering::Release);
    let mode = *worker.mode.lock().unwrap();
    if matches!(mode, Mode::Slow) {
        let body = async_stream::stream! { yield Ok::<_,std::io::Error>(bytes::Bytes::from_static(b" ")); std::future::pending::<()>().await; };
        return Response::builder().body(Body::from_stream(body)).unwrap();
    }
    let proof = packet
        .as_object_mut()
        .unwrap()
        .remove(cache_auth::PROOF_FIELD)
        .unwrap();
    let mut claims = cache_auth::verify_signature(&SECRET, &proof, &packet).unwrap();
    let request: RequestControl = serde_json::from_value(packet).unwrap();
    assert_eq!(
        claims["operation"],
        if uri.path().ends_with("abort") {
            "request-abort"
        } else {
            "request-status"
        }
    );
    let hold = matches!(mode, Mode::Hold);
    let aborted = uri.path().ends_with("abort") && !hold;
    let participants: Vec<_> = (0..8).map(|i| json!({"participant_id":format!("cpu-participant-{i}"),"ingress_closed":!hold,"tokenizer_drained":!hold,"scheduler_drained":!hold,"live_request_holds":u64::from(hold),"live_kv_holds":0,"live_meta_holds":0,"live_io_holds":0,"transport_drained":!hold,"abort_ack":aborted})).collect();
    let mut payload = json!({"schema_version":1,"transaction_id":request.transaction_id,"request_root":request.request_root,"native_incarnation":request.expected_native_incarnation,"cleanup_cohort_digest":request.expected_cleanup_cohort_digest,"ordinary_body_digest":request.ordinary_body_digest,"expected_choices":request.expected_choices,"captured_at_ms":now_ms(),"participants":participants,"terminal":!hold,"aborted":aborted});
    match mode {
        Mode::PartialRank => {
            payload["participants"].as_array_mut().unwrap().pop();
        }
        Mode::WrongRoom => payload["participants"][7]["ingress_closed"] = false.into(),
        Mode::TerminalHolds => payload["participants"][7]["live_request_holds"] = 1.into(),
        Mode::MissingAck => payload["participants"][7]["abort_ack"] = false.into(),
        Mode::Stale => payload["captured_at_ms"] = (now_ms() - 10000).into(),
        Mode::WrongIncarnation => payload["native_incarnation"] = "wrong-birth".into(),
        _ => {}
    }
    let digest = cache_auth::digest(&payload).unwrap();
    for field in ["operation_id", "nonce"] {
        claims[field] = cache_auth::digest(&json!(["native-response", claims[field], digest]))
            .unwrap()
            .into();
    }
    payload[cache_auth::PROOF_FIELD] = cache_auth::sign(&SECRET, &claims, &payload).unwrap();
    if matches!(mode, Mode::Tampered) {
        payload["expected_choices"] = 1.into();
    }
    Response::builder()
        .header("content-type", "application/json")
        .body(Body::from(payload.to_string()))
        .unwrap()
}

#[tokio::test]
async fn request_control_http_signed_cohort_rejects_missing_participants_live_holds_and_unprocessed_abort()
 {
    let server = fixture(Mode::Good).await;
    let expected = request_expected();
    let provider = request_claims(&server, expected.clone());
    let transport =
        NativeRequestTransport::new(&server.origin, SECRET, Duration::from_secs(1)).unwrap();
    let natural = transport
        .status(&expected, provider.as_ref())
        .await
        .unwrap();
    assert!(natural.terminal && !natural.aborted);
    assert_eq!(natural.participants.len(), 8);
    let aborted = transport.abort(&expected, provider.as_ref()).await.unwrap();
    assert!(aborted.terminal && aborted.aborted);
    for mode in [
        Mode::PartialRank,
        Mode::WrongRoom,
        Mode::TerminalHolds,
        Mode::MissingAck,
        Mode::Stale,
        Mode::WrongIncarnation,
        Mode::Tampered,
    ] {
        *server.worker.mode.lock().unwrap() = mode;
        assert!(matches!(
            transport.abort(&expected, provider.as_ref()).await,
            Err(RequestTransportError::Verification(_))
        ));
    }
    assert_eq!(server.worker.calls.load(Ordering::Acquire), 9);
}

#[tokio::test]
async fn request_control_http_drop_cleanup_registered_before_tcp_dispatch_blocks_shutdown_until_physical_terminal()
 {
    let server = fixture(Mode::Hold).await;
    let expected = request_expected();
    let id = expected.cleanup_id();
    let provider = request_claims(&server, expected.clone());
    let transport =
        NativeRequestTransport::new(&server.origin, SECRET, Duration::from_millis(200)).unwrap();
    let registry = NativeCleanupRegistry::new();
    let (sender, receiver) = tokio::sync::oneshot::channel();
    registry
        .register(id.clone(), async move {
            let _ = receiver.await;
            CleanupReceipt::from_request(
                transport
                    .cleanup(
                        expected,
                        provider,
                        tokio::time::Instant::now() + Duration::from_secs(2),
                        Duration::from_millis(5),
                    )
                    .await?,
            )
        })
        .unwrap();
    let response = reqwest::Client::new()
        .post(format!("{}/cpu-fixture/generation", server.origin))
        .send()
        .await
        .unwrap();
    drop(response);
    drop(DisconnectTrigger(Some(sender)));
    tokio::time::timeout(Duration::from_secs(1), async {
        while server.worker.calls.load(Ordering::Acquire) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(registry.states().get(&id), Some(&CleanupState::Running));
    assert!(
        tokio::time::timeout(Duration::from_millis(40), registry.wait_for_physical_zero())
            .await
            .is_err()
    );
    *server.worker.mode.lock().unwrap() = Mode::Good;
    tokio::time::timeout(Duration::from_secs(1), registry.wait_for_physical_zero())
        .await
        .unwrap();
    assert!(registry.states().is_empty());
    assert_eq!(server.worker.generation_calls.load(Ordering::Acquire), 1);
}

#[tokio::test]
async fn request_control_http_unknown_body_retains_blocked_and_wrong_origin_sends_nothing() {
    let server = fixture(Mode::Slow).await;
    let expected = request_expected();
    let id = expected.cleanup_id();
    let provider = request_claims(&server, expected.clone());
    let transport =
        NativeRequestTransport::new(&server.origin, SECRET, Duration::from_millis(40)).unwrap();
    let wrong = RequestFixtureClaims {
        origin: "http://127.0.0.1:1".into(),
        expected: expected.clone(),
        started: provider.started,
        nonce: AtomicUsize::new(0),
    };
    assert!(matches!(
        transport.status(&expected, &wrong).await,
        Err(RequestTransportError::Control(
            PdTransportError::InvalidRequest(_)
        ))
    ));
    assert_eq!(server.worker.calls.load(Ordering::Acquire), 0);
    let registry = NativeCleanupRegistry::new();
    registry
        .register(id.clone(), async move {
            // One actual body timeout. A status/abort transport error provides no
            // opaque terminal proof and cannot remove the registered lifecycle.
            transport.status(&expected, provider.as_ref()).await?;
            Err(CleanupFailure::Verification(
                "nonterminal status cannot acknowledge shutdown".into(),
            ))
        })
        .unwrap();
    assert!(registry.await_quiescence().await.is_err());
    assert!(matches!(
        registry.states().get(&id),
        Some(CleanupState::Blocked(CleanupFailure::Request(
            RequestTransportError::Control(PdTransportError::Unknown { timed_out: true })
        )))
    ));
    assert!(
        tokio::time::timeout(Duration::from_millis(20), registry.wait_for_physical_zero())
            .await
            .is_err()
    );
    assert_eq!(server.worker.calls.load(Ordering::Acquire), 1);
    assert_eq!(server.worker.generation_calls.load(Ordering::Acquire), 0);
}
