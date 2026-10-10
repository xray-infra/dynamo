//! One process-wide, shutdown-owned physical cleanup registry. Register the
//! lifecycle future before native POST; trigger it from completion/disconnect.
use crate::cache_authority::native_cache_frontend::VerifiedCacheCleanup;
use crate::native_pd::transport::{PdTransportError, VerifiedReceipt};
use crate::native_request_cleanup::VerifiedNativeTerminal;
use crate::native_request_lifecycle::VerifiedNoNativePost;
use crate::native_request_transport::RequestTransportError;
use futures::FutureExt;
use std::collections::BTreeMap;
use std::future::Future;
use std::sync::{Arc, Mutex};
use tokio::sync::Notify;
use tokio_util::task::TaskTracker;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CleanupFailure {
    #[error("native P/D cleanup: {0}")]
    Pd(#[from] PdTransportError),
    #[error("ordinary native cleanup: {0}")]
    Request(#[from] RequestTransportError),
    #[error("physical cleanup verification failed: {0}")]
    Verification(String),
    #[error("cleanup registry sealed")]
    Closed,
    #[error("duplicate or empty cleanup identity")]
    Duplicate,
    #[error("cleanup identity is not blocked")]
    NotBlocked,
    #[error("cleanup task panicked")]
    Panicked,
    #[error("cleanup registration has no live async runtime")]
    NoRuntime,
}

#[cfg(test)]
mod hold_tests {
    use super::*;
    use crate::cache_authority::cache_auth;
    use crate::native_request_cleanup::{ExpectedRequest, RequestControl, verify_terminal};
    use serde_json::{Value, json};
    use std::collections::BTreeSet;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    struct AdmissionHold(Arc<AtomicUsize>);
    impl Drop for AdmissionHold {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    // CPU-only fixture. Mint through the actual signed participant verifier,
    // never a bool constructor or unverified deserialized terminal status.
    fn verified_receipt(root: &str) -> CleanupReceipt {
        let secret = [42u8; 32];
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let participants: BTreeSet<String> = ["cpu-cleanup-owner".into()].into();
        let expected = ExpectedRequest {
            control: RequestControl {
                schema_version: 1,
                transaction_id: "2".repeat(32),
                request_root: root.into(),
                expected_native_incarnation: "cpu-fixture-worker-birth".into(),
                expected_cleanup_cohort_digest: cache_auth::digest(&json!(participants)).unwrap(),
                ordinary_body_digest: cache_auth::digest(&json!(
                    "CPU-only canonical request fixture"
                ))
                .unwrap(),
                expected_choices: 1,
            },
            participants,
        };
        let claims = json!({"version":1,"operation":"request-status","operation_id":"cpu-hold-operation","nonce":"cpu-hold-nonce","model":"CPU fixture only","native_origin":"http://127.0.0.1:1","native_incarnation":expected.control.expected_native_incarnation,"source_digest":"a".repeat(64),"runtime_profile_digest":"b".repeat(64),"reservation_id":"","authority_revision":0,"native_request_id":root,"namespace_digest":"","full_render_digest":"","token_ids_digest":"","request_started_at_ms":now,"issued_at_ms":now,"deadline_ms":now+1000,"retain_for_ms":0,"payload_digest":"0".repeat(64)});
        let mut packet = json!({"schema_version":1,"transaction_id":expected.control.transaction_id,"request_root":root,"native_incarnation":expected.control.expected_native_incarnation,"cleanup_cohort_digest":expected.control.expected_cleanup_cohort_digest,"ordinary_body_digest":expected.control.ordinary_body_digest,"expected_choices":1,"captured_at_ms":now,"participants":[{"participant_id":"cpu-cleanup-owner","ingress_closed":true,"tokenizer_drained":true,"scheduler_drained":true,"live_request_holds":0,"live_kv_holds":0,"live_meta_holds":0,"live_io_holds":0,"transport_drained":true,"abort_ack":false}],"terminal":true,"aborted":false});
        let event = cache_auth::digest(&packet).unwrap();
        let mut response_claims = claims.clone();
        for field in ["operation_id", "nonce"] {
            response_claims[field] = Value::String(
                cache_auth::digest(&json!(["native-response", claims[field], event])).unwrap(),
            );
        }
        packet[cache_auth::PROOF_FIELD] =
            cache_auth::sign(&secret, &response_claims, &packet).unwrap();
        let packet = crate::native_pd::transport::PreparedControlResponse::new(packet).unwrap();
        let instant = tokio::time::Instant::now();
        let clock=crate::cache_authority::frontend_clock::qualified_cpu_tracking_sample(
            "Reference ID : C0000201\nSystem time : 0 seconds fast of NTP time\nRoot dispersion : 0 seconds\nRoot delay : 0 seconds\nLeap status : Normal\n",
            u128::from(now)*1_000_000,u128::from(now)*1_000_000,instant,instant).unwrap();
        CleanupReceipt::from_request(
            verify_terminal(&secret, &claims, &expected, &packet, &clock).unwrap(),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn cleanup_hold_error_panic_and_wrong_identity_retain_until_verified_retry() {
        for panic_task in [false, true] {
            let receipt = verified_receipt(&"3".repeat(32));
            let id = receipt.id().to_owned();
            let drops = Arc::new(AtomicUsize::new(0));
            let registry = NativeCleanupRegistry::new();
            registry
                .register_with_hold(
                    id.clone(),
                    async move {
                        if panic_task {
                            panic!("CPU fixture cleanup panic");
                        }
                        Err(CleanupFailure::Verification(
                            "physical state unknown".into(),
                        ))
                    },
                    Box::new(AdmissionHold(drops.clone())),
                )
                .unwrap();
            assert!(registry.await_quiescence().await.is_err());
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            let wrong = verified_receipt(&"4".repeat(32));
            registry
                .retry_blocked(id.clone(), async move { Ok(wrong) })
                .unwrap();
            assert!(registry.await_quiescence().await.is_err());
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            registry
                .retry_blocked(id, async move { Ok(receipt) })
                .unwrap();
            registry.wait_for_physical_zero().await;
            assert_eq!(drops.load(Ordering::SeqCst), 1);
            assert!(registry.states().is_empty());
        }
    }

    #[test]
    fn cleanup_hold_no_runtime_retains_original_admission_until_verified_recovery() {
        let receipt = verified_receipt(&"3".repeat(32));
        let id = receipt.id().to_owned();
        let drops = Arc::new(AtomicUsize::new(0));
        let registry = NativeCleanupRegistry::new();
        assert_eq!(
            registry.register_with_hold(
                id.clone(),
                async { Err(CleanupFailure::Verification("unreachable".into())) },
                Box::new(AdmissionHold(drops.clone()))
            ),
            Err(CleanupFailure::NoRuntime)
        );
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            registry
                .retry_blocked(id, async move { Ok(receipt) })
                .unwrap();
            registry.wait_for_physical_zero().await;
        });
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}
/// No bool-only constructor. Every adapter consumes opaque, already verified
/// physical evidence and binds the exact registered identity.
pub struct CleanupReceipt {
    id: String,
    _evidence: Evidence,
}
enum Evidence {
    Pd(VerifiedReceipt),
    Cache(VerifiedCacheCleanup),
    Request(VerifiedNativeTerminal),
    NoNativePost(VerifiedNoNativePost),
}
impl CleanupReceipt {
    pub fn from_pd(receipt: VerifiedReceipt) -> Result<Self, CleanupFailure> {
        if !receipt.physical_terminal() {
            return Err(CleanupFailure::Pd(PdTransportError::Nonterminal));
        }
        let id = receipt.cleanup_id();
        Ok(Self {
            id,
            _evidence: Evidence::Pd(receipt),
        })
    }
    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn from_request(receipt: VerifiedNativeTerminal) -> Result<Self, CleanupFailure> {
        let id = receipt.cleanup_id().to_owned();
        if id.is_empty() || !receipt.status().terminal {
            return Err(CleanupFailure::Verification(
                "ordinary physical terminal identity missing".into(),
            ));
        }
        Ok(Self {
            id,
            _evidence: Evidence::Request(receipt),
        })
    }
    pub(crate) fn from_no_native_post(proof: VerifiedNoNativePost) -> Result<Self, CleanupFailure> {
        let id = proof.cleanup_id();
        if id.is_empty() {
            return Err(CleanupFailure::Verification(
                "undispatched native identity missing".into(),
            ));
        }
        Ok(Self {
            id,
            _evidence: Evidence::NoNativePost(proof),
        })
    }
    pub(crate) fn from_cache(receipt: VerifiedCacheCleanup) -> Result<Self, CleanupFailure> {
        if receipt.native_request_id().is_empty() {
            return Err(CleanupFailure::Verification(
                "cache receipt native identity missing".into(),
            ));
        }
        if let Some((reservation, revision)) = receipt.cancellation() {
            if reservation.is_empty() || revision <= 0 {
                return Err(CleanupFailure::Verification(
                    "cache cancellation scope missing".into(),
                ));
            }
        }
        let id = format!("cache:{}", receipt.native_request_id());
        Ok(Self {
            id,
            _evidence: Evidence::Cache(receipt),
        })
    }
    fn physical_terminal(&self) -> bool {
        match &self._evidence {
            Evidence::Pd(receipt) => receipt.physical_terminal(),
            // The cache leaf's opaque proof preserves either verified native
            // cancel/CAS evidence or its owned, provably-never-sent preparation
            // branch. The registry cannot mint either from an HTTP status.
            Evidence::Cache(receipt) => !receipt.native_request_id().is_empty(),
            Evidence::Request(receipt) => receipt.status().terminal,
            // This opaque proof is minted only from the bound actual connect
            // error. No native generation POST or cache side effect occurred.
            Evidence::NoNativePost(proof) => !proof.cleanup_id().is_empty(),
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CleanupState {
    Running,
    Blocked(CleanupFailure),
}
#[derive(Debug, thiserror::Error)]
#[error("physical native cleanup unresolved: {entries:?}")]
pub struct CleanupBlocked {
    pub entries: BTreeMap<String, CleanupState>,
}
struct State {
    sealed: bool,
    entries: BTreeMap<String, CleanupState>,
    // Held independently of the fallible task. A task returning Err or panic
    // must not release local admission while physical cleanup is unknown.
    holds: BTreeMap<String, Box<dyn Send + 'static>>,
}
struct Inner {
    tasks: TaskTracker,
    state: Mutex<State>,
    changed: Notify,
}
#[derive(Clone)]
pub struct NativeCleanupRegistry(Arc<Inner>);

/// An unsubmitted lifecycle task and its local hold. A group is admitted as one
/// operation before any participating native generation can be dispatched.
pub(crate) struct CleanupRegistration {
    id: String,
    future: futures::future::BoxFuture<'static, Result<CleanupReceipt, CleanupFailure>>,
    hold: Box<dyn Send + 'static>,
}
impl CleanupRegistration {
    pub(crate) fn new(
        id: String,
        future: impl Future<Output = Result<CleanupReceipt, CleanupFailure>> + Send + 'static,
        hold: Box<dyn Send + 'static>,
    ) -> Self {
        Self {
            id,
            future: Box::pin(future),
            hold,
        }
    }
}
impl Default for NativeCleanupRegistry {
    fn default() -> Self {
        Self::new()
    }
}
impl NativeCleanupRegistry {
    pub fn new() -> Self {
        Self(Arc::new(Inner {
            tasks: TaskTracker::new(),
            state: Mutex::new(State {
                sealed: false,
                entries: BTreeMap::new(),
                holds: BTreeMap::new(),
            }),
            changed: Notify::new(),
        }))
    }
    pub fn states(&self) -> BTreeMap<String, CleanupState> {
        self.0.state.lock().unwrap().entries.clone()
    }
    pub fn shares_registry(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    pub fn register(
        &self,
        id: String,
        future: impl Future<Output = Result<CleanupReceipt, CleanupFailure>> + Send + 'static,
    ) -> Result<(), CleanupFailure> {
        self.register_impl(id, future, None)
    }
    /// Register before dispatch and transfer the local admission hold. Once
    /// accepted, only matching opaque physical evidence releases the hold.
    /// Closed/Duplicate reject registration before native POST; the unaccepted
    /// hold drops locally. NoRuntime retains an accepted hold as Blocked.
    pub fn register_with_hold(
        &self,
        id: String,
        future: impl Future<Output = Result<CleanupReceipt, CleanupFailure>> + Send + 'static,
        hold: Box<dyn Send + 'static>,
    ) -> Result<(), CleanupFailure> {
        self.register_impl(id, future, Some(hold))
    }
    pub(crate) fn register_prepared(
        &self,
        registration: CleanupRegistration,
    ) -> Result<(), CleanupFailure> {
        self.register_impl(
            registration.id,
            registration.future,
            Some(registration.hold),
        )
    }
    /// Reject the entire unsubmitted group before inserting anything. Runtime
    /// absence is a local preparation failure here, not an accepted task's
    /// physical failure; ordinary single-registration semantics are unchanged.
    pub(crate) fn register_group(
        &self,
        registrations: Vec<CleanupRegistration>,
    ) -> Result<(), CleanupFailure> {
        let mut state = self.0.state.lock().unwrap();
        if state.sealed {
            return Err(CleanupFailure::Closed);
        }
        let mut ids = std::collections::BTreeSet::new();
        if registrations.is_empty()
            || registrations.iter().any(|registration| {
                registration.id.is_empty()
                    || state.entries.contains_key(&registration.id)
                    || !ids.insert(registration.id.clone())
            })
        {
            return Err(CleanupFailure::Duplicate);
        }
        let handle =
            tokio::runtime::Handle::try_current().map_err(|_| CleanupFailure::NoRuntime)?;
        for registration in registrations {
            let CleanupRegistration { id, future, hold } = registration;
            state.entries.insert(id.clone(), CleanupState::Running);
            state.holds.insert(id.clone(), hold);
            self.spawn(id, future, &handle);
        }
        Ok(())
    }
    fn register_impl(
        &self,
        id: String,
        future: impl Future<Output = Result<CleanupReceipt, CleanupFailure>> + Send + 'static,
        hold: Option<Box<dyn Send + 'static>>,
    ) -> Result<(), CleanupFailure> {
        let mut state = self.0.state.lock().unwrap();
        if state.sealed {
            return Err(CleanupFailure::Closed);
        }
        if id.is_empty() || state.entries.contains_key(&id) {
            return Err(CleanupFailure::Duplicate);
        }
        state.entries.insert(id.clone(), CleanupState::Running);
        if let Some(hold) = hold {
            state.holds.insert(id.clone(), hold);
        }
        let handle = match tokio::runtime::Handle::try_current() {
            Ok(handle) => handle,
            Err(_) => {
                state
                    .entries
                    .insert(id, CleanupState::Blocked(CleanupFailure::NoRuntime));
                self.0.changed.notify_waiters();
                return Err(CleanupFailure::NoRuntime);
            }
        };
        self.spawn(id, future, &handle);
        Ok(())
    }
    pub fn retry_blocked(
        &self,
        id: String,
        future: impl Future<Output = Result<CleanupReceipt, CleanupFailure>> + Send + 'static,
    ) -> Result<(), CleanupFailure> {
        let mut state = self.0.state.lock().unwrap();
        if !matches!(state.entries.get(&id), Some(CleanupState::Blocked(_))) {
            return Err(CleanupFailure::NotBlocked);
        }
        // Recovery may run after sealing, but cannot introduce another identity.
        state.entries.insert(id.clone(), CleanupState::Running);
        let handle = match tokio::runtime::Handle::try_current() {
            Ok(handle) => handle,
            Err(_) => {
                state
                    .entries
                    .insert(id, CleanupState::Blocked(CleanupFailure::NoRuntime));
                self.0.changed.notify_waiters();
                return Err(CleanupFailure::NoRuntime);
            }
        };
        self.spawn(id, future, &handle);
        Ok(())
    }
    fn spawn(
        &self,
        id: String,
        future: impl Future<Output = Result<CleanupReceipt, CleanupFailure>> + Send + 'static,
        handle: &tokio::runtime::Handle,
    ) {
        let inner = self.0.clone();
        self.0.tasks.spawn_on(
            async move {
                let result = std::panic::AssertUnwindSafe(future).catch_unwind().await;
                let mut verified_ordinary = None;
                let released_hold = {
                    let mut state = inner.state.lock().unwrap();
                    match result {
                        Ok(Ok(receipt)) if receipt.id == id && receipt.physical_terminal() => {
                            if let Evidence::Request(verified) = &receipt._evidence {
                                verified_ordinary = Some(verified.status().clone());
                            }
                            state.entries.remove(&id);
                            state.holds.remove(&id)
                        }
                        Ok(Ok(_)) => {
                            tracing::warn!(target:"dynamo_llm::native_cleanup",cleanup_id=%id,
                                error="receipt belongs to another cleanup identity",
                                "native physical cleanup blocked; admission hold retained");
                            state.entries.insert(
                                id,
                                CleanupState::Blocked(CleanupFailure::Verification(
                                    "receipt belongs to another cleanup identity".into(),
                                )),
                            );
                            None
                        }
                        Ok(Err(error)) => {
                            tracing::warn!(target:"dynamo_llm::native_cleanup",cleanup_id=%id,error=%error,
                                "native physical cleanup blocked; admission hold retained");
                            state.entries.insert(id, CleanupState::Blocked(error));
                            None
                        }
                        Err(_) => {
                            tracing::warn!(target:"dynamo_llm::native_cleanup",cleanup_id=%id,error="cleanup task panicked",
                                "native physical cleanup blocked; admission hold retained");
                            state
                                .entries
                                .insert(id, CleanupState::Blocked(CleanupFailure::Panicked));
                            None
                        }
                    }
                };
                // Destructors may lock admission state or reenter diagnostics.
                // Drop outside the registry mutex, before notifying shutdown.
                drop(released_hold);
                if let Some(status) = verified_ordinary {
                    tracing::info!(target: "dynamo_llm::native_cleanup",
                        request_root = %status.request_root,
                        native_incarnation = %status.native_incarnation,
                        cleanup_cohort_digest = %status.cleanup_cohort_digest,
                        expected_choices = status.expected_choices,
                        terminal = status.terminal,
                        aborted = status.aborted,
                        captured_at_ms = status.captured_at_ms,
                        participant_count = status.participants.len(),
                        participants = %serde_json::json!(status.participants),
                        "native ordinary physical cleanup verified; admission hold released");
                }
                inner.changed.notify_waiters();
            },
            handle,
        );
    }
    fn seal(&self) {
        let mut state = self.0.state.lock().unwrap();
        state.sealed = true;
        self.0.tasks.close();
    }
    /// Diagnostics only: an error does not acknowledge physical shutdown.
    pub async fn await_quiescence(&self) -> Result<(), CleanupBlocked> {
        self.seal();
        self.0.tasks.wait().await;
        let entries = self.states();
        if entries.is_empty() {
            Ok(())
        } else {
            Err(CleanupBlocked { entries })
        }
    }
    /// Natural shutdown cannot finish while a Running or Blocked physical
    /// lifecycle remains. Explicit typed-evidence retries can recover Blocked.
    pub async fn wait_for_physical_zero(&self) {
        self.seal();
        loop {
            let changed = self.0.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if self.0.state.lock().unwrap().entries.is_empty() {
                self.0.tasks.wait().await;
                return;
            }
            changed.await;
        }
    }
}
