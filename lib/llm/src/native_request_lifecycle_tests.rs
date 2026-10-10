//! Synthetic CPU HTTP receipts exercise production registration/signing and
//! opaque physical verification; they are not native/GPU cohort qualification.
use super::*;
use crate::native_cleanup::CleanupState;
use crate::native_cleanup_cohort::{
    CleanupCohortArtifact, CleanupCohortParticipant, install_cleanup_cohort,
};
use axum::{Json, Router, extract::State, routing::post};
use std::sync::{
    Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use tokio::sync::Notify;

const SECRET: [u8; 32] = [0x63; 32];
const MODEL: &str = "synthetic-request-lifecycle-model";
// CPU fixture epoch source only. Production signing always qualifies Chrony.
fn cpu_fixture_wall_ms() -> Result<u64, std::num::TryFromIntError> {
    u64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("CPU fixture wall clock before epoch")
            .as_millis(),
    )
}
const CPU_TRACKING: &str = "Reference ID : CB00710F (CPU model)\nSystem time : 0 seconds slow of NTP time\nRoot delay : 0 seconds\nRoot dispersion : 0 seconds\nLeap status : Normal\n";

fn with_cpu_tracking(mut lifecycle: QualifiedRequestLifecycle) -> QualifiedRequestLifecycle {
    lifecycle.clock.tracking_model = Some(Arc::new(TrackingClockModel {
        wall_at_start_ns: u128::from(cpu_fixture_wall_ms().unwrap()) * 1_000_000,
        started_at: Instant::now(),
        tracking: CPU_TRACKING.into(),
        observation_lag: Duration::ZERO,
    }));
    lifecycle
}

#[tokio::test]
async fn request_lifecycle_clock_fresh_claims_keep_original_upper_deadline() {
    let origin = "http://127.0.0.1:49999";
    let lifecycle = with_cpu_tracking(
        QualifiedRequestLifecycle::new(
            installed(origin),
            SECRET,
            origin,
            NativeCleanupRegistry::new(),
        )
        .unwrap(),
    );
    let initial = lifecycle.clock.qualify().await.unwrap();
    let start = initial.request_start_lower_ms();
    let deadline = start + ORDINARY_REQUEST_WINDOW_MS;
    let claims = RequestClaims {
        cohort: lifecycle.cohort.clone(),
        origin: origin.into(),
        request_root: "4".repeat(32),
        started_at_ms: start,
        deadline_ms: deadline,
        control_digest: cache_auth::digest(&json!({"CPU":"already prepared actual control"}))
            .unwrap(),
        clock: lifecycle.clock.clone(),
    };
    let mut previous_issue = start;
    for operation in ["request-generate", "request-status", "request-abort"] {
        let qualified = claims
            .for_payload(operation, &claims.control_digest)
            .await
            .unwrap();
        assert_eq!(qualified.claims["issued_at_ms"], qualified.clock.lower_ms());
        assert_eq!(qualified.claims["request_started_at_ms"], start);
        assert_eq!(qualified.claims["deadline_ms"], deadline);
        assert!(qualified.clock.upper_ms() < deadline);
        assert!(qualified.clock.lower_ms() >= previous_issue);
        qualified.clock.ensure_fresh().unwrap();
        previous_issue = qualified.clock.lower_ms();
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    // A lower estimate can still look admissible when its upper bound crosses
    // the deadline. Such a claim is rejected rather than extending the window.
    let current = lifecycle.clock.qualify().await.unwrap();
    let expired = RequestClaims {
        deadline_ms: current.upper_ms(),
        ..claims
    };
    assert!(
        expired
            .for_payload("request-abort", &expired.control_digest)
            .await
            .is_err()
    );
    let future = RequestClaims {
        started_at_ms: current.upper_ms() + 10_000,
        deadline_ms: current.upper_ms() + 10_000 + ORDINARY_REQUEST_WINDOW_MS,
        ..expired
    };
    assert!(
        future
            .for_payload("request-status", &future.control_digest)
            .await
            .is_err()
    );
}

#[tokio::test]
async fn request_lifecycle_clock_unqualified_tracking_never_registers_or_falls_back() {
    let origin = "http://127.0.0.1:49999";
    for (tracking, lag) in [
        (
            CPU_TRACKING.replace("Normal", "Not synchronised"),
            Duration::ZERO,
        ),
        (
            CPU_TRACKING.replace("System time : 0 ", "System time : 0.2 "),
            Duration::ZERO,
        ),
        (CPU_TRACKING.into(), Duration::from_millis(101)),
    ] {
        let registry = NativeCleanupRegistry::new();
        let mut lifecycle =
            QualifiedRequestLifecycle::new(installed(origin), SECRET, origin, registry.clone())
                .unwrap();
        let now = cpu_fixture_wall_ms().unwrap();
        lifecycle.clock.tracking_model = Some(Arc::new(TrackingClockModel {
            wall_at_start_ns: u128::from(now) * 1_000_000,
            started_at: Instant::now(),
            tracking,
            observation_lag: lag,
        }));
        let rid = "5".repeat(32);
        let drops = Arc::new(AtomicUsize::new(0));
        let result = lifecycle
            .prepare_generation(
                body(&rid, 1),
                &rid,
                1,
                now - 100,
                Box::new(Hold(drops.clone())),
            )
            .await;
        assert!(matches!(result, Err(CleanupFailure::Verification(_))));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert!(registry.states().is_empty());
    }
}

fn installed(origin: &str) -> InstalledCleanupCohort {
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
    let ids: std::collections::BTreeSet<_> = participants
        .iter()
        .map(|p| p.participant_id.clone())
        .collect();
    let artifact = CleanupCohortArtifact {
        schema_version: 1,
        native_origin: origin.into(),
        native_incarnation: "1234567890abcdef1234567890abcdef".into(),
        model: MODEL.into(),
        source_digest: "c".repeat(64),
        runtime_profile_digest: "d".repeat(64),
        cleanup_cohort_digest: cache_auth::digest(&json!(ids)).unwrap(),
        participants,
    };
    install_cleanup_cohort(artifact).unwrap()
}
fn body(rid: &str, n: u32) -> Value {
    json!({"model":MODEL,"rid":rid,"n":n,"messages":[{"role":"user","content":"CPU 测试"}],"temperature":0.3})
}
struct Hold(Arc<AtomicUsize>);
impl Drop for Hold {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
#[derive(Default)]
struct Wire {
    requests: Mutex<Vec<Value>>,
    bad_cohort: AtomicBool,
    delay: AtomicBool,
    observed: Notify,
    release: Notify,
}
async fn control(State(wire): State<Arc<Wire>>, Json(packet): Json<Value>) -> Json<Value> {
    let mut control = packet.clone();
    let proof = control
        .as_object_mut()
        .unwrap()
        .remove(cache_auth::PROOF_FIELD)
        .unwrap();
    let claims = cache_auth::verify_signature(&SECRET, &proof, &control).unwrap();
    wire.requests.lock().unwrap().push(claims.clone());
    wire.observed.notify_one();
    if wire.delay.load(Ordering::SeqCst) {
        wire.release.notified().await;
    }
    let mut ids = vec!["opaque-helper", "opaque-scheduler", "opaque-tokenizer"];
    if wire.bad_cohort.load(Ordering::SeqCst) {
        ids.pop();
    }
    let participants: Vec<_> = ids
        .into_iter()
        .map(|id| {
            json!({"participant_id":id,
        "ingress_closed":true,"tokenizer_drained":true,"scheduler_drained":true,
        "live_request_holds":0,"live_kv_holds":0,"live_meta_holds":0,"live_io_holds":0,
        "transport_drained":true,"abort_ack":true})
        })
        .collect();
    let mut reply = json!({"schema_version":1,"transaction_id":control["transaction_id"],
        "request_root":control["request_root"],"native_incarnation":control["expected_native_incarnation"],
        "cleanup_cohort_digest":control["expected_cleanup_cohort_digest"],
        "ordinary_body_digest":control["ordinary_body_digest"],"expected_choices":control["expected_choices"],
        "captured_at_ms":cpu_fixture_wall_ms().unwrap(),"participants":participants,"terminal":true,"aborted":true});
    let event = cache_auth::digest(&reply).unwrap();
    let mut response_claims = claims.clone();
    for key in ["operation_id", "nonce"] {
        response_claims[key] =
            json!(cache_auth::digest(&json!(["native-response", claims[key], event])).unwrap());
    }
    reply[cache_auth::PROOF_FIELD] = cache_auth::sign(&SECRET, &response_claims, &reply).unwrap();
    Json(reply)
}
async fn server(wire: Arc<Wire>) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let router = Router::new()
        .route("/v1/requests/abort", post(control))
        .route("/v1/requests/status", post(control))
        .with_state(wire);
    let task = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (origin, task)
}

#[tokio::test]
async fn ordinary_lifecycle_connect_failure_releases_only_bound_noncache_post() {
    // Bind then close an ephemeral local port: actual reqwest POST fails while
    // establishing a connection, with no native server or generation involved.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    let registry = NativeCleanupRegistry::new();
    let lifecycle =
        QualifiedRequestLifecycle::new(installed(&origin), SECRET, &origin, registry.clone())
            .unwrap();
    let client = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .retry(reqwest::retry::never())
        .timeout(Duration::from_secs(1))
        .build()
        .unwrap();
    let url = format!("{origin}/v1/chat/completions");
    let lifecycle = with_cpu_tracking(lifecycle);
    let rid = uuid::Uuid::new_v4().simple().to_string();
    let drops = Arc::new(AtomicUsize::new(0));
    let mut prepared = lifecycle
        .prepare_generation(
            body(&rid, 1),
            &rid,
            1,
            lifecycle
                .clock
                .qualify()
                .await
                .unwrap()
                .request_start_lower_ms(),
            Box::new(Hold(drops.clone())),
        )
        .await
        .unwrap();
    let wrong_url_error = client
        .post(format!("{origin}/wrong-path"))
        .send()
        .await
        .unwrap_err();
    assert!(wrong_url_error.is_connect());
    assert!(
        prepared
            .guard
            .not_dispatched(&wrong_url_error)
            .await
            .is_err()
    );
    let builder_error = client.post("invalid URL").send().await.unwrap_err();
    assert!(!builder_error.is_connect());
    assert!(prepared.guard.not_dispatched(&builder_error).await.is_err());
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert!(matches!(
        registry.states().get(prepared.guard.cleanup_id()),
        Some(CleanupState::Running)
    ));
    let actual_error = client
        .post(&url)
        .json(&prepared.body)
        .send()
        .await
        .unwrap_err();
    assert!(actual_error.is_connect());
    tokio::time::timeout(
        Duration::from_secs(2),
        prepared.guard.not_dispatched(&actual_error),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(registry.states().is_empty());
    assert!(prepared.guard.not_dispatched(&actual_error).await.is_err());

    // Cache preparation may already have native reservation effects. Even the
    // same true connect error cannot prove those resources were never acquired.
    let cached_rid = uuid::Uuid::new_v4().simple().to_string();
    // The original admission fence has one second left. Failed signed control
    // may retry, but cannot restart a 30-second cleanup window from guard drop.
    let cached_start = cpu_fixture_wall_ms().unwrap() - ORDINARY_REQUEST_WINDOW_MS + 1_000;
    let mut cached = lifecycle
        .prepare_cached_generation(
            body(&cached_rid, 1),
            &cached_rid,
            1,
            cached_start,
            Box::new(Hold(drops.clone())),
        )
        .await
        .unwrap();
    assert!(cached.guard.not_dispatched(&actual_error).await.is_err());
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    drop(cached.guard);
    let blocked = tokio::time::timeout(Duration::from_secs(2), registry.await_quiescence())
        .await
        .unwrap()
        .unwrap_err();
    assert!(matches!(
        blocked.entries.get(&cached.recovery.cleanup_id()),
        Some(CleanupState::Blocked(_))
    ));
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn ordinary_lifecycle_registers_before_post_and_keeps_hold_until_physical_ack() {
    for cached in [false, true] {
        let wire = Arc::new(Wire::default());
        wire.delay.store(true, Ordering::SeqCst);
        let (origin, server) = server(wire.clone()).await;
        let registry = NativeCleanupRegistry::new();
        let lifecycle =
            QualifiedRequestLifecycle::new(installed(&origin), SECRET, &origin, registry.clone())
                .unwrap();
        let lifecycle = with_cpu_tracking(lifecycle);
        let rid = uuid::Uuid::new_v4().simple().to_string();
        let start = lifecycle
            .clock
            .qualify()
            .await
            .unwrap()
            .request_start_lower_ms();
        let drops = Arc::new(AtomicUsize::new(0));
        let prepared = if cached {
            lifecycle
                .prepare_cached_generation(
                    body(&rid, u32::MAX),
                    &rid,
                    u32::MAX,
                    start,
                    Box::new(Hold(drops.clone())),
                )
                .await
        } else {
            lifecycle
                .prepare_generation(
                    body(&rid, u32::MAX),
                    &rid,
                    u32::MAX,
                    start,
                    Box::new(Hold(drops.clone())),
                )
                .await
        }
        .unwrap();
        assert_eq!(
            registry.states().get(prepared.guard.cleanup_id()),
            Some(&CleanupState::Running)
        );
        assert!(wire.requests.lock().unwrap().is_empty());
        assert_eq!(
            prepared.body["prompt_cache_request_control"]["expected_choices"],
            u32::MAX
        );
        assert!(
            prepared.body["prompt_cache_canonical_body"]
                .as_str()
                .unwrap()
                .contains("0.3")
        );
        if cached {
            assert!(prepared.body.get(cache_auth::PROOF_FIELD).is_none());
        } else {
            let payload = cache_auth::payload_for("request-generate", &prepared.body).unwrap();
            let claims = cache_auth::verify_signature(
                &SECRET,
                &prepared.body[cache_auth::PROOF_FIELD],
                &payload,
            )
            .unwrap();
            assert_eq!(claims.as_object().unwrap().len(), 20);
            assert_eq!(claims["request_started_at_ms"], start);
            assert_eq!(claims["deadline_ms"], start + ORDINARY_REQUEST_WINDOW_MS);
            assert_eq!(claims["reservation_id"], "");
            assert_eq!(claims["full_render_digest"], "");
        }
        if cached {
            drop(prepared.guard);
        } else {
            prepared.guard.finish();
        }
        tokio::time::timeout(Duration::from_secs(2), wire.observed.notified())
            .await
            .unwrap();
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert!(!registry.states().is_empty());
        wire.release.notify_one();
        tokio::time::timeout(Duration::from_secs(2), registry.await_quiescence())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        server.abort();
    }
}

#[tokio::test]
async fn ordinary_lifecycle_bad_receipt_stays_blocked_and_retry_preserves_birth() {
    let wire = Arc::new(Wire::default());
    wire.bad_cohort.store(true, Ordering::SeqCst);
    let (origin, server) = server(wire.clone()).await;
    let registry = NativeCleanupRegistry::new();
    let lifecycle =
        QualifiedRequestLifecycle::new(installed(&origin), SECRET, &origin, registry.clone())
            .unwrap();
    let lifecycle = with_cpu_tracking(lifecycle);
    let rid = uuid::Uuid::new_v4().simple().to_string();
    let start = lifecycle
        .clock
        .qualify()
        .await
        .unwrap()
        .request_start_lower_ms();
    let drops = Arc::new(AtomicUsize::new(0));
    let prepared = lifecycle
        .prepare_generation(body(&rid, 1), &rid, 1, start, Box::new(Hold(drops.clone())))
        .await
        .unwrap();
    let recovery = prepared.recovery.clone();
    drop(prepared.guard);
    let blocked = tokio::time::timeout(Duration::from_secs(2), registry.await_quiescence())
        .await
        .unwrap()
        .unwrap_err();
    assert!(matches!(
        blocked.entries.get(&recovery.cleanup_id()),
        Some(CleanupState::Blocked(_))
    ));
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert!(
        recovery
            .retry_blocked(&NativeCleanupRegistry::new())
            .is_err()
    );
    wire.bad_cohort.store(false, Ordering::SeqCst);
    recovery.retry_blocked(&registry).unwrap();
    tokio::time::timeout(Duration::from_secs(2), registry.await_quiescence())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    let claims = wire.requests.lock().unwrap();
    assert_eq!(claims.len(), 2);
    for claim in claims.iter() {
        assert_eq!(claim["request_started_at_ms"], start);
        assert_eq!(claim["deadline_ms"], start + ORDINARY_REQUEST_WINDOW_MS);
        assert_eq!(claim["native_request_id"], rid);
    }
    assert_ne!(claims[0]["operation_id"], claims[1]["operation_id"]);
    assert_ne!(claims[0]["nonce"], claims[1]["nonce"]);
    server.abort();
}

#[tokio::test]
async fn ordinary_lifecycle_rejects_unbound_origin_scope_and_expired_fence_before_dispatch() {
    let origin = "http://127.0.0.1:49999";
    assert!(
        QualifiedRequestLifecycle::new(
            installed(origin),
            SECRET,
            "http://127.0.0.1:49998",
            NativeCleanupRegistry::new()
        )
        .is_err()
    );
    assert!(
        QualifiedRequestLifecycle::new(
            installed(origin),
            SECRET,
            "http://user@127.0.0.1:49999",
            NativeCleanupRegistry::new()
        )
        .is_err()
    );
    let registry = NativeCleanupRegistry::new();
    let lifecycle = QualifiedRequestLifecycle::new(
        installed(origin),
        SECRET,
        &format!("{origin}/"),
        registry.clone(),
    )
    .unwrap();
    let lifecycle = with_cpu_tracking(lifecycle);
    let rid = uuid::Uuid::new_v4().simple().to_string();
    let now = lifecycle
        .clock
        .qualify()
        .await
        .unwrap()
        .request_start_lower_ms();
    for (mut b, n, start, cached) in [
        (body(&rid, 1), 1, now - ORDINARY_REQUEST_WINDOW_MS, false),
        (body(&rid, 2), 1, now, false),
        (body(&rid, 0), 0, now, false),
        (body(&rid, 1), 1, now + 10_000, false),
        (body(&rid, 1), 1, now, false),
    ] {
        if start == now && n == 1 && b["n"] == 1 {
            b[cache_auth::PROOF_FIELD] = json!({"untrusted":true});
        }
        let drops = Arc::new(AtomicUsize::new(0));
        assert!(
            lifecycle
                .prepare(
                    b,
                    &rid,
                    n,
                    start,
                    Box::new(Hold(drops.clone())),
                    cached,
                    None
                )
                .await
                .is_err()
        );
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert!(registry.states().is_empty());
    }
    registry.await_quiescence().await.unwrap();
    assert!(matches!(
        lifecycle
            .prepare_generation(body(&rid, 1), &rid, 1, now, Box::new(()))
            .await,
        Err(CleanupFailure::Closed)
    ));
}
