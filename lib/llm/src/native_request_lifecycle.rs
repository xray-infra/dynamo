//! Qualified ordinary request admission and registry-owned physical cleanup.
//! Register synchronously before POST; caller finish/drop only triggers the
//! already tracked task and never acknowledges native resource release.
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::sync::oneshot;
use tokio::time::Instant;

use crate::cache_authority::cache_auth;
use crate::cache_authority::frontend_clock::{ChronyClock, ClockBounds};
use crate::native_cleanup::{
    CleanupFailure, CleanupReceipt, CleanupRegistration, NativeCleanupRegistry,
};
use crate::native_cleanup_cohort::InstalledCleanupCohort;
use crate::native_request_cleanup::{
    ExpectedRequest, ORDINARY_REQUEST_WINDOW_MS, PreparedGeneration, RequestControl,
};
use crate::native_request_transport::{
    NativeRequestTransport, QualifiedClaims, RequestClaimProvider, RequestTransportError,
};

const CLEANUP_LIMIT: Duration = Duration::from_secs(30);
const CONTROL_TIMEOUT: Duration = Duration::from_secs(5);
const CONTROL_POLL: Duration = Duration::from_millis(100);

pub struct QualifiedRequestLifecycle {
    cohort: Arc<InstalledCleanupCohort>,
    origin: String,
    secret: [u8; 32],
    transport: Arc<NativeRequestTransport>,
    registry: NativeCleanupRegistry,
    clock: RequestClock,
}

#[derive(Clone, Default)]
struct RequestClock {
    #[cfg(test)]
    tracking_model: Option<Arc<TrackingClockModel>>,
}
impl RequestClock {
    async fn qualify(&self) -> Result<ClockBounds, String> {
        #[cfg(test)]
        if let Some(model) = &self.tracking_model {
            return model.qualify();
        }
        ChronyClock
            .qualify()
            .await
            .map_err(|error| error.to_string())
    }
}

// Explicit CPU-only tracking model. Production has no injectable clock or
// configuration fallback: each qualification invokes actual Chrony tracking.
#[cfg(test)]
struct TrackingClockModel {
    wall_at_start_ns: u128,
    started_at: Instant,
    tracking: String,
    observation_lag: Duration,
}
#[cfg(test)]
impl TrackingClockModel {
    fn qualify(&self) -> Result<ClockBounds, String> {
        let now = Instant::now();
        let elapsed = now
            .checked_duration_since(self.started_at)
            .ok_or("CPU tracking model monotonic reversal")?;
        let wall = self
            .wall_at_start_ns
            .checked_add(elapsed.as_nanos())
            .ok_or("CPU tracking model wall overflow")?;
        let observed = now
            .checked_sub(self.observation_lag)
            .ok_or("CPU tracking model observation underflow")?;
        crate::cache_authority::frontend_clock::qualified_cpu_tracking_sample(
            &self.tracking,
            wall,
            wall,
            observed,
            observed,
        )
        .map_err(|error| error.to_string())
    }
}

pub struct PreparedRequestGeneration {
    pub body: Value,
    pub guard: RequestCleanupTrigger,
    pub recovery: RequestCleanupRecovery,
}

/// Prepared P/D body with no ordinary native ingress or cleanup task yet.
/// Dropping this stage releases only its local admission/resource hold.
pub(crate) struct PreparedPdGeneration {
    body: Value,
    registration: DeferredRequestRegistration,
}
impl PreparedPdGeneration {
    pub(crate) fn into_parts(self) -> (Value, DeferredRequestRegistration) {
        (self.body, self.registration)
    }
    fn arm(self) -> Result<PreparedRequestGeneration, CleanupFailure> {
        let (body, registration) = self.into_parts();
        let (guard, recovery) = registration.arm()?;
        Ok(PreparedRequestGeneration {
            body,
            guard,
            recovery,
        })
    }
}

pub(crate) struct DeferredRequestRegistration {
    context: Arc<CleanupContext>,
    registry: NativeCleanupRegistry,
    resource_hold: Box<dyn Send + 'static>,
    no_dispatch_url: Option<String>,
}
impl DeferredRequestRegistration {
    pub(crate) fn admission_fence(&self) -> Instant {
        self.context.admission_fence
    }

    /// The dispatcher calls this only after both bodies, native envelopes and
    /// pre-send checks are complete, immediately before generation dispatch.
    pub(crate) fn arm(
        self,
    ) -> Result<(RequestCleanupTrigger, RequestCleanupRecovery), CleanupFailure> {
        let registry = self.registry.clone();
        let (registration, guard, recovery) = self.into_registration()?;
        registry.register_prepared(registration)?;
        Ok((guard, recovery))
    }
    pub(crate) fn into_registration(
        self,
    ) -> Result<
        (
            CleanupRegistration,
            RequestCleanupTrigger,
            RequestCleanupRecovery,
        ),
        CleanupFailure,
    > {
        let Self {
            context,
            registry,
            resource_hold,
            no_dispatch_url,
        } = self;
        if Instant::now() >= context.admission_fence {
            return Err(CleanupFailure::Request(RequestTransportError::Deadline));
        }
        let cleanup_id = context.expected.cleanup_id();
        let (sender, receiver) = oneshot::channel();
        let (released_sender, released) = oneshot::channel();
        let task_context = context.clone();
        let registration = CleanupRegistration::new(
            cleanup_id.clone(),
            async move {
                let command = tokio::select! {
                    result = receiver => result.unwrap_or_else(|_| CleanupCommand::Finished(Instant::now())),
                    _ = tokio::time::sleep_until(task_context.admission_fence) => {
                        return Err(CleanupFailure::Request(RequestTransportError::Deadline));
                    }
                };
                match command {
                    CleanupCommand::Finished(caller_finished) => {
                        task_context.cleanup(caller_finished).await
                    }
                    CleanupCommand::NotDispatched(proof) => {
                        CleanupReceipt::from_no_native_post(proof)
                    }
                }
            },
            Box::new(RegistryHold {
                resource: Some(resource_hold),
                _context: context.clone(),
                released: Some(released_sender),
            }),
        );
        Ok((
            registration,
            RequestCleanupTrigger {
                cleanup_id,
                sender: Some(sender),
                released: Some(released),
                no_dispatch_url,
                expected: context.expected.clone(),
            },
            RequestCleanupRecovery(context, registry),
        ))
    }
}

/// Dropping a public body or local permit cannot construct a physical receipt.
pub struct RequestCleanupTrigger {
    cleanup_id: String,
    sender: Option<oneshot::Sender<CleanupCommand>>,
    released: Option<oneshot::Receiver<()>>,
    no_dispatch_url: Option<String>,
    expected: Arc<ExpectedRequest>,
}
enum CleanupCommand {
    Finished(Instant),
    NotDispatched(VerifiedNoNativePost),
}

/// Only this module can construct source evidence that the bound ordinary
/// generation POST failed during connection establishment before dispatch.
pub(crate) struct VerifiedNoNativePost {
    expected: Arc<ExpectedRequest>,
    _post_url: String,
}
impl VerifiedNoNativePost {
    pub(crate) fn cleanup_id(&self) -> String {
        self.expected.cleanup_id()
    }
}
impl RequestCleanupTrigger {
    pub fn cleanup_id(&self) -> &str {
        &self.cleanup_id
    }
    pub fn finish(mut self) {
        self.trigger();
    }
    /// Accept only the actual connection failure from this exact ordinary POST.
    /// The tracked task owns the command even if this caller is cancelled while
    /// waiting; return only after the registry drops its admission hold.
    pub async fn not_dispatched(&mut self, error: &reqwest::Error) -> Result<(), CleanupFailure> {
        let expected_url = self.no_dispatch_url.as_ref().ok_or_else(|| {
            CleanupFailure::Verification("cache generation has prior native side effects".into())
        })?;
        if !error.is_connect()
            || error.url().map(reqwest::Url::as_str) != Some(expected_url.as_str())
        {
            return Err(CleanupFailure::Verification(
                "failure does not prove the bound native POST was not dispatched".into(),
            ));
        }
        let sender = self.sender.take().ok_or_else(|| {
            CleanupFailure::Verification("request cleanup already triggered".into())
        })?;
        sender
            .send(CleanupCommand::NotDispatched(VerifiedNoNativePost {
                expected: self.expected.clone(),
                _post_url: expected_url.clone(),
            }))
            .map_err(|_| {
                CleanupFailure::Verification("tracked request cleanup task unavailable".into())
            })?;
        self.released
            .take()
            .ok_or_else(|| {
                CleanupFailure::Verification("request release acknowledgement unavailable".into())
            })?
            .await
            .map_err(|_| {
                CleanupFailure::Verification("request release acknowledgement lost".into())
            })
    }
    fn trigger(&mut self) {
        if let Some(sender) = self.sender.take() {
            let _ = sender.send(CleanupCommand::Finished(Instant::now()));
        }
    }
}
impl Drop for RequestCleanupTrigger {
    fn drop(&mut self) {
        self.trigger();
    }
}

/// Retains exact birth identity for an explicit retry of blocked cleanup. A
/// retry cannot replay generation or extend the original admission fence.
#[derive(Clone)]
pub struct RequestCleanupRecovery(Arc<CleanupContext>, NativeCleanupRegistry);
impl RequestCleanupRecovery {
    pub(crate) fn admission_fence(&self) -> Instant {
        self.0.admission_fence
    }

    pub fn cleanup_id(&self) -> String {
        self.0.expected.cleanup_id()
    }
    pub fn retry_blocked(&self, registry: &NativeCleanupRegistry) -> Result<(), CleanupFailure> {
        if !self.1.shares_registry(registry) {
            return Err(CleanupFailure::Verification(
                "request recovery belongs to another cleanup registry".into(),
            ));
        }
        let context = self.0.clone();
        registry.retry_blocked(self.cleanup_id(), async move {
            context.cleanup(Instant::now()).await
        })
    }
}

struct CleanupContext {
    expected: Arc<ExpectedRequest>,
    claims: Arc<RequestClaims>,
    transport: Arc<NativeRequestTransport>,
    admission_fence: Instant,
}
impl CleanupContext {
    async fn cleanup(&self, caller_finished: Instant) -> Result<CleanupReceipt, CleanupFailure> {
        let deadline = caller_finished
            .checked_add(CLEANUP_LIMIT)
            .ok_or_else(|| CleanupFailure::Verification("cleanup deadline overflow".into()))?
            .min(self.admission_fence);
        let receipt = self
            .transport
            .cleanup(
                self.expected.clone(),
                self.claims.clone(),
                deadline,
                CONTROL_POLL,
            )
            .await?;
        CleanupReceipt::from_request(receipt)
    }
}

// The registry keeps both resources and immutable request context after a
// fallible future ends. Only its exact opaque physical success removes this.
struct RegistryHold {
    resource: Option<Box<dyn Send + 'static>>,
    _context: Arc<CleanupContext>,
    released: Option<oneshot::Sender<()>>,
}
impl Drop for RegistryHold {
    fn drop(&mut self) {
        drop(self.resource.take());
        if let Some(sender) = self.released.take() {
            let _ = sender.send(());
        }
    }
}

struct RequestClaims {
    cohort: Arc<InstalledCleanupCohort>,
    origin: String,
    request_root: String,
    started_at_ms: u64,
    deadline_ms: u64,
    control_digest: String,
    clock: RequestClock,
}
impl RequestClaims {
    async fn for_payload(
        &self,
        operation: &str,
        payload_digest: &str,
    ) -> Result<QualifiedClaims, String> {
        if !matches!(
            operation,
            "request-generate" | "request-status" | "request-abort"
        ) {
            return Err("ordinary request operation unsupported".into());
        }
        let bounds = self.clock.qualify().await?;
        bounds.ensure_fresh().map_err(|error| error.to_string())?;
        let issued = bounds.lower_ms();
        if issued < self.started_at_ms || bounds.upper_ms() >= self.deadline_ms {
            return Err(
                "ordinary request original admission fence expired or clock reversed".into(),
            );
        }
        let artifact = self.cohort.artifact();
        let claims = json!({"version":1,"operation":operation,
            "operation_id":uuid::Uuid::new_v4().simple().to_string(),
            "nonce":uuid::Uuid::new_v4().simple().to_string(),
            "model":artifact.model,"native_origin":self.origin,
            "native_incarnation":artifact.native_incarnation,
            "source_digest":artifact.source_digest,
            "runtime_profile_digest":artifact.runtime_profile_digest,
            "reservation_id":"","authority_revision":0,
            "native_request_id":self.request_root,"namespace_digest":"",
            "full_render_digest":"","token_ids_digest":"",
            "request_started_at_ms":self.started_at_ms,"issued_at_ms":issued,
            "deadline_ms":self.deadline_ms,"retain_for_ms":0,"payload_digest":payload_digest});
        cache_auth::validate_claims(&claims)?;
        Ok(QualifiedClaims {
            claims,
            clock: bounds,
        })
    }
}
#[async_trait]
impl RequestClaimProvider for RequestClaims {
    async fn clock_bounds(&self) -> Result<ClockBounds, String> {
        let bounds = self.clock.qualify().await?;
        bounds.ensure_fresh().map_err(|error| error.to_string())?;
        Ok(bounds)
    }
    async fn claims(&self, operation: &str) -> Result<QualifiedClaims, String> {
        if !matches!(operation, "request-status" | "request-abort") {
            return Err("ordinary cleanup cannot issue generation claims".into());
        }
        self.for_payload(operation, &self.control_digest).await
    }
}

fn normalized_origin(origin: &str) -> Result<String, String> {
    let url = reqwest::Url::parse(origin).map_err(|_| "native worker origin invalid")?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || url.path() != "/"
    {
        return Err("native worker must be one HTTP origin without credentials or path".into());
    }
    Ok(url.origin().ascii_serialization())
}

impl QualifiedRequestLifecycle {
    pub fn worker_url(&self) -> &str {
        &self.origin
    }
    pub fn model(&self) -> &str {
        &self.cohort.artifact().model
    }
    pub fn registry(&self) -> &NativeCleanupRegistry {
        &self.registry
    }

    /// Capture the selected worker's current birth before registering/signing
    /// this request. Existing guards retain their own immutable old cohort.
    pub(crate) async fn refresh_current(
        &self,
        client: &reqwest::Client,
        timeout: Duration,
    ) -> Result<Arc<Self>, String> {
        let bytes = crate::native_cleanup_cohort::load_current_discovery(
            client, self.model(), &self.origin, timeout,
        ).await?;
        let actual = crate::native_cleanup_cohort::parse_current_artifact(self.model(), &bytes)?;
        if actual.model != self.model()
            || actual.native_origin.trim_end_matches('/') != self.origin
        {
            return Err("ordinary current runtime model/origin differs from selected worker".into());
        }
        let installed = crate::native_cleanup_cohort::install_cleanup_cohort(actual)?;
        Ok(Arc::new(Self {
            cohort: Arc::new(installed), origin: self.origin.clone(), secret: self.secret,
            transport: self.transport.clone(), registry: self.registry.clone(), clock: self.clock.clone(),
        }))
    }

    pub fn new(
        installed: InstalledCleanupCohort,
        secret: [u8; 32],
        worker_url: &str,
        registry: NativeCleanupRegistry,
    ) -> Result<Self, String> {
        let origin = normalized_origin(&installed.artifact().native_origin)?;
        if normalized_origin(worker_url)? != origin {
            return Err("qualified cleanup cohort belongs to another native worker origin".into());
        }
        let transport = NativeRequestTransport::new(&origin, secret, CONTROL_TIMEOUT)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            cohort: Arc::new(installed),
            origin,
            secret,
            transport: Arc::new(transport),
            registry,
            clock: RequestClock::default(),
        })
    }

    /// Ordinary requests receive their own request-generate envelope. This
    /// Qualified clock sampling is asynchronous; successful return proves
    /// synchronous registry registration precedes any native generation POST.
    pub async fn prepare_generation(
        &self,
        body: Value,
        owned_rid: &str,
        n: u32,
        start_ms: u64,
        resource_hold: Box<dyn Send + 'static>,
    ) -> Result<PreparedRequestGeneration, CleanupFailure> {
        self.prepare(body, owned_rid, n, start_ms, resource_hold, false, None)
            .await?
            .arm()
    }

    /// Attach clean server-only lifecycle control before the cache owner signs
    /// its existing generate/reservation envelope. This does not write or
    /// replace PROOF_FIELD, and is not native cache composition qualification.
    pub async fn prepare_cached_generation(
        &self,
        body: Value,
        owned_rid: &str,
        n: u32,
        start_ms: u64,
        resource_hold: Box<dyn Send + 'static>,
    ) -> Result<PreparedRequestGeneration, CleanupFailure> {
        self.prepare(body, owned_rid, n, start_ms, resource_hold, true, None)
            .await?
            .arm()
    }

    /// Cached ordinary admission is prepared locally while the cache owner
    /// reserves/signs. Register only after that succeeds and before native POST.
    pub(crate) async fn prepare_cached_generation_draft(
        &self,
        body: Value,
        owned_rid: &str,
        n: u32,
        start_ms: u64,
        resource_hold: Box<dyn Send + 'static>,
    ) -> Result<PreparedPdGeneration, CleanupFailure> {
        self.prepare(body, owned_rid, n, start_ms, resource_hold, true, None)
            .await
    }

    /// Join independently approved capabilities by their actual native boot and
    /// existing model key. Each capability retains its own complete runtime
    /// profile: generation/root cleanup use the ordinary profile, while room
    /// controls use the P/D role profile. No secret is exposed.
    pub fn validate_pd_role(
        &self,
        role: &crate::native_pd::qualification::VerifiedPdRole,
    ) -> Result<(), String> {
        let actual = self.cohort.artifact();
        let pd = role.artifact();
        if actual.model != pd.model
            || self.origin != normalized_origin(&pd.native_origin)?
            || actual.native_incarnation != pd.native_incarnation
            || actual.source_digest != pd.source_digest
            || self.secret.as_slice() != role.secret_bytes()
        {
            return Err("PD role and installed cleanup cohort worker identity differ".into());
        }
        Ok(())
    }

    /// Server-only qualified dispatcher entry. One generation envelope protects
    /// both lifecycle control and the complete qualified P/D binding; the
    /// ordinary transaction is the dispatcher's existing P/D transaction.
    pub(crate) async fn prepare_pd_generation(
        &self,
        body: Value,
        owned_rid: &str,
        n: u32,
        start_ms: u64,
        transaction_id: &str,
        own_rank_set_digest: &str,
        resource_hold: Box<dyn Send + 'static>,
    ) -> Result<PreparedPdGeneration, CleanupFailure> {
        self.prepare_pd_generation_body(
            body,
            owned_rid,
            n,
            start_ms,
            transaction_id,
            own_rank_set_digest,
            resource_hold,
            false,
        )
        .await
    }

    /// The P cache owner signs only after both typed lifecycle bindings exist.
    pub(crate) async fn prepare_pd_cached_generation(
        &self,
        body: Value,
        owned_rid: &str,
        n: u32,
        start_ms: u64,
        transaction_id: &str,
        own_rank_set_digest: &str,
        resource_hold: Box<dyn Send + 'static>,
    ) -> Result<PreparedPdGeneration, CleanupFailure> {
        self.prepare_pd_generation_body(
            body,
            owned_rid,
            n,
            start_ms,
            transaction_id,
            own_rank_set_digest,
            resource_hold,
            true,
        )
        .await
    }

    async fn prepare_pd_generation_body(
        &self,
        body: Value,
        owned_rid: &str,
        n: u32,
        start_ms: u64,
        transaction_id: &str,
        own_rank_set_digest: &str,
        resource_hold: Box<dyn Send + 'static>,
        cached: bool,
    ) -> Result<PreparedPdGeneration, CleanupFailure> {
        let binding: crate::native_pd::dispatch::GenerationBinding =
            serde_json::from_value(body.get("prompt_cache_pd_binding").cloned().ok_or_else(
                || CleanupFailure::Verification("qualified PD generation binding missing".into()),
            )?)
            .map_err(|error| CleanupFailure::Verification(error.to_string()))?;
        binding
            .validate_ordinary(
                owned_rid,
                n,
                transaction_id,
                &self.cohort.artifact().native_incarnation,
                &self.origin,
                own_rank_set_digest,
            )
            .map_err(CleanupFailure::Verification)?;
        if body
            .get("bootstrap_host")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
            || body
                .get("bootstrap_port")
                .and_then(Value::as_u64)
                .is_none_or(|port| port == 0 || port > u64::from(u16::MAX))
            || body
                .get("bootstrap_room")
                .and_then(Value::as_i64)
                .is_none_or(|room| room <= 0)
            || body.get("disagg_prefill_dp_rank").is_some()
        {
            return Err(CleanupFailure::Verification(
                "qualified DP1 bootstrap fields invalid".into(),
            ));
        }
        self.prepare(
            body,
            owned_rid,
            n,
            start_ms,
            resource_hold,
            cached,
            Some(transaction_id),
        )
        .await
    }

    async fn prepare(
        &self,
        body: Value,
        owned_rid: &str,
        n: u32,
        start_ms: u64,
        resource_hold: Box<dyn Send + 'static>,
        cached: bool,
        pd_transaction: Option<&str>,
    ) -> Result<PreparedPdGeneration, CleanupFailure> {
        let verify = |error: String| CleanupFailure::Verification(error);
        let fields = body
            .as_object()
            .ok_or_else(|| verify("native body must be object".into()))?;
        if fields.contains_key(cache_auth::PROOF_FIELD)
            || fields.contains_key("prompt_cache_request_control")
            || fields.contains_key("prompt_cache_canonical_body")
        {
            return Err(verify(
                "request lifecycle must precede native envelope signing".into(),
            ));
        }
        if !cached
            && fields.keys().any(|key| {
                (key.starts_with("prompt_cache_")
                    && key != "prompt_cache_key"
                    && !(pd_transaction.is_some() && key == "prompt_cache_pd_binding"))
                    || matches!(key.as_str(), "reservation_id" | "authority_revision")
            })
        {
            return Err(verify(
                "cache generation requires verified native lifecycle composition".into(),
            ));
        }
        if pd_transaction.is_none()
            && fields.keys().any(|key| {
                matches!(
                    key.as_str(),
                    "prompt_cache_pd_binding"
                        | "bootstrap_host"
                        | "bootstrap_port"
                        | "bootstrap_room"
                        | "disagg_prefill_dp_rank"
                )
            })
        {
            return Err(verify(
                "PD private fields require the qualified dispatcher entry".into(),
            ));
        }
        if body.get("model").and_then(Value::as_str) != Some(self.cohort.artifact().model.as_str())
            || body
                .get("rid")
                .or_else(|| body.get("request_id"))
                .and_then(Value::as_str)
                != Some(owned_rid)
            || (body.get("rid").is_some() && body.get("request_id").is_some())
            || body.get("n").map_or(Some(1), Value::as_u64) != Some(u64::from(n))
        {
            return Err(verify(
                "request lifecycle model/owned root/choice mismatch".into(),
            ));
        }
        let deadline_ms = start_ms
            .checked_add(ORDINARY_REQUEST_WINDOW_MS)
            .filter(|deadline| *deadline <= i64::MAX as u64)
            .ok_or_else(|| verify("ordinary request original fence overflow".into()))?;
        let canonical =
            serde_json::to_string(&cache_auth::ordinary_body_for(&body).map_err(verify)?)
                .map_err(|error| verify(error.to_string()))?;
        let expected = Arc::new(ExpectedRequest {
            control: RequestControl {
                schema_version: 1,
                transaction_id: pd_transaction
                    .map(str::to_owned)
                    .unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string()),
                request_root: owned_rid.into(),
                expected_native_incarnation: self.cohort.artifact().native_incarnation.clone(),
                expected_cleanup_cohort_digest: self.cohort.cleanup_cohort_digest().into(),
                ordinary_body_digest: cache_auth::digest(&Value::String(canonical))
                    .map_err(verify)?,
                expected_choices: n,
            },
            participants: self.cohort.participant_ids(),
        });
        expected.validate().map_err(verify)?;
        let control =
            serde_json::to_value(&expected.control).map_err(|error| verify(error.to_string()))?;
        let claims = Arc::new(RequestClaims {
            cohort: self.cohort.clone(),
            origin: self.origin.clone(),
            request_root: owned_rid.into(),
            started_at_ms: start_ms,
            deadline_ms,
            control_digest: cache_auth::digest(&control).map_err(verify)?,
            clock: self.clock.clone(),
        });
        let (body, bounds) = if cached {
            let mut unsigned = body;
            cache_auth::attach_canonical_body(&mut unsigned).map_err(verify)?;
            unsigned["prompt_cache_request_control"] = control;
            let bounds = self.clock.qualify().await.map_err(verify)?;
            bounds
                .ensure_fresh()
                .map_err(|error| verify(error.to_string()))?;
            if bounds.lower_ms() < start_ms || bounds.upper_ms() >= deadline_ms {
                return Err(verify(
                    "ordinary request original admission fence expired or clock reversed".into(),
                ));
            }
            (unsigned, bounds)
        } else {
            let prepared = PreparedGeneration::new(&expected, body).map_err(verify)?;
            let generation_claims = claims
                .for_payload("request-generate", prepared.payload_digest())
                .await
                .map_err(verify)?;
            let signed = prepared
                .sign(&self.secret, &generation_claims)
                .map_err(verify)?;
            (signed, generation_claims.clock)
        };
        bounds
            .ensure_fresh()
            .map_err(|error| verify(error.to_string()))?;
        let admission_fence = bounds
            .observed_at()
            .checked_add(Duration::from_millis(deadline_ms - bounds.upper_ms()))
            .ok_or_else(|| verify("ordinary request monotonic fence overflow".into()))?;
        if Instant::now() >= admission_fence {
            return Err(verify(
                "ordinary request admission fence elapsed during preparation".into(),
            ));
        }
        let context = Arc::new(CleanupContext {
            expected,
            claims,
            transport: self.transport.clone(),
            admission_fence,
        });
        Ok(PreparedPdGeneration {
            body,
            registration: DeferredRequestRegistration {
                context,
                registry: self.registry.clone(),
                resource_hold,
                no_dispatch_url: (!cached && pd_transaction.is_none())
                    .then(|| format!("{}/v1/chat/completions", self.origin)),
            },
        })
    }
}

#[cfg(test)]
#[path = "native_request_lifecycle_tests.rs"]
mod tests;

#[cfg(test)]
mod current_cohort_tests {
    //! CPU HTTP discovery and production envelope factories only. No generation
    //! is dispatched and no physical native cleanup receipt is manufactured.
    use super::*;
    use crate::native_cleanup::{CleanupState, NativeCleanupRegistry};
    use crate::native_cleanup_cohort::{CleanupCohortArtifact, CleanupCohortParticipant, install_cleanup_cohort};
    use axum::{Json, Router, http::StatusCode, routing::get};
    use std::{collections::BTreeSet, sync::Mutex};
    use tokio_util::sync::CancellationToken;

    const MODEL: &str = "qwen3.8-flash-next";
    const SECRET: [u8; 32] = [0x37; 32];

    fn artifact(origin: &str, birth: &str) -> CleanupCohortArtifact {
        let participants = (0..6).map(|index| CleanupCohortParticipant {
            participant_id: format!("CPU-{birth}-{index}"), role: format!("CPU-owner-{index}"),
            process_birth: format!("CPU-process-{birth}-{index}"),
            source_digest: "a".repeat(64), runtime_registration_digest: "b".repeat(64),
        }).collect::<Vec<_>>();
        let ids = participants.iter().map(|row| row.participant_id.clone()).collect::<BTreeSet<_>>();
        CleanupCohortArtifact {
            schema_version: 1, native_origin: origin.into(), native_incarnation: birth.into(),
            model: MODEL.into(), source_digest: "c".repeat(64), runtime_profile_digest: "d".repeat(64),
            cleanup_cohort_digest: cache_auth::digest(&json!(ids)).unwrap(), participants,
        }
    }

    fn client() -> reqwest::Client {
        reqwest::Client::builder().no_proxy().redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never()).connect_timeout(Duration::from_secs(1)).build().unwrap()
    }

    type Document = Arc<Mutex<(StatusCode, Value, Duration)>>;
    async fn server(birth: &str) -> (String, Document, CancellationToken, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let document = Arc::new(Mutex::new((StatusCode::OK, json!(artifact(&origin, birth)), Duration::ZERO)));
        let router = Router::new().route(crate::native_cleanup_cohort::CLEANUP_COHORT_METADATA_PATH, get({
            let document = document.clone();
            move || {
                let document = document.clone();
                async move {
                    let (status, value, delay) = document.lock().unwrap().clone();
                    tokio::time::sleep(delay).await;
                    (status, Json(value))
                }
            }
        }));
        let cancel = CancellationToken::new();
        let task = tokio::spawn({
            let cancel = cancel.clone();
            async move { axum::serve(listener, router).with_graceful_shutdown(cancel.cancelled_owned()).await.unwrap(); }
        });
        (origin, document, cancel, task)
    }

    fn initial(origin: &str, registry: NativeCleanupRegistry) -> QualifiedRequestLifecycle {
        let mut lifecycle = QualifiedRequestLifecycle::new(
            install_cleanup_cohort(artifact(origin, "CPU-startup-birth")).unwrap(), SECRET, origin, registry,
        ).unwrap();
        // Explicit CPU clock model through the maintained ClockBounds parser;
        // production refresh/signing continues to invoke real Chrony tracking.
        lifecycle.clock.tracking_model = Some(Arc::new(TrackingClockModel {
            wall_at_start_ns: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos(),
            started_at: Instant::now(), observation_lag: Duration::ZERO,
            tracking: "Reference ID : CB00710F (CPU model)\nSystem time : 0 seconds slow of NTP time\nRoot delay : 0 seconds\nRoot dispersion : 0 seconds\nLeap status : Normal\n".into(),
        }));
        lifecycle
    }

    async fn arm(lifecycle: &QualifiedRequestLifecycle) -> PreparedRequestGeneration {
        let rid = uuid::Uuid::new_v4().simple().to_string();
        lifecycle.prepare_generation(
            json!({"model":MODEL,"rid":rid,"n":2,"messages":[{"role":"user","content":"CPU envelope only"}]}),
            &rid, 2, lifecycle.clock.qualify().await.unwrap().request_start_lower_ms(), Box::new(()),
        ).await.unwrap()
    }

    fn generation_claims(prepared: &PreparedRequestGeneration) -> Value {
        let payload = cache_auth::payload_for("request-generate", &prepared.body).unwrap();
        cache_auth::verify_signature(&SECRET, &prepared.body[cache_auth::PROOF_FIELD], &payload).unwrap()
    }

    #[tokio::test]
    async fn native_request_current_cohort_refresh_new_birth_keeps_armed_identity() {
        let (origin, document, cancel, task) = server("CPU-current-one").await;
        let registry = NativeCleanupRegistry::new();
        let startup = initial(&origin, registry.clone());
        let first = startup.refresh_current(&client(), Duration::from_secs(1)).await.unwrap();
        let mut old = arm(&first).await;
        let old_control = old.body["prompt_cache_request_control"].clone();
        let old_claims = generation_claims(&old);
        assert_eq!(old_claims["native_incarnation"], "CPU-current-one");
        assert_eq!(old_control["expected_native_incarnation"], "CPU-current-one");
        assert_eq!(registry.states()[old.guard.cleanup_id()], CleanupState::Running);
        document.lock().unwrap().1 = json!(artifact(&origin, "CPU-current-two"));
        let second = startup.refresh_current(&client(), Duration::from_secs(1)).await.unwrap();
        let mut new = arm(&second).await;
        let new_claims = generation_claims(&new);
        assert_eq!(new_claims["native_incarnation"], "CPU-current-two");
        assert_eq!(new.body["prompt_cache_request_control"]["expected_native_incarnation"], "CPU-current-two");
        assert_ne!(old_control["expected_cleanup_cohort_digest"], new.body["prompt_cache_request_control"]["expected_cleanup_cohort_digest"]);
        assert!(!Arc::ptr_eq(&first.cohort, &second.cohort));
        assert!(first.registry().shares_registry(second.registry()));
        let cleanup_claims = old.recovery.0.claims.for_payload("request-abort", &old.recovery.0.claims.control_digest).await.unwrap();
        assert_eq!(cleanup_claims.claims["native_incarnation"], "CPU-current-one");
        assert_eq!(old.recovery.0.expected.control.expected_native_incarnation, "CPU-current-one");
        assert_eq!(old.body["prompt_cache_request_control"], old_control);
        assert_eq!(old.recovery.0.expected.participants, first.cohort.participant_ids());
        assert_eq!(new.recovery.0.expected.participants, second.cohort.participant_ids());
        assert_eq!(registry.states()[old.guard.cleanup_id()], CleanupState::Running);
        cancel.cancel();
        tokio::time::timeout(Duration::from_secs(2), task).await.unwrap().unwrap();
        // Actual closed-port POST connection errors prove neither CPU request
        // was dispatched; this releases local holds without a physical ACK.
        let no_post = client();
        for prepared in [&mut old, &mut new] {
            let error = no_post.post(format!("{origin}/v1/chat/completions"))
                .timeout(Duration::from_secs(1)).json(&prepared.body).send().await.unwrap_err();
            tokio::time::timeout(Duration::from_secs(2), prepared.guard.not_dispatched(&error)).await.unwrap().unwrap();
        }
    }

    #[tokio::test]
    async fn native_request_current_cohort_refresh_invalid_discovery_never_uses_startup() {
        let (origin, document, cancel, task) = server("CPU-current-one").await;
        let registry = NativeCleanupRegistry::new();
        let startup = initial(&origin, registry.clone());
        let http = client();
        for field in ["model", "native_origin", "participants"] {
            let mut wrong = json!(artifact(&origin, "CPU-current-two"));
            wrong[field] = match field {
                "model" => json!("another-model"),
                "native_origin" => json!("http://127.0.0.1:1"),
                _ => json!([]),
            };
            document.lock().unwrap().1 = wrong;
            assert!(startup.refresh_current(&http, Duration::from_secs(1)).await.is_err());
            assert!(registry.states().is_empty());
            assert_eq!(startup.cohort.artifact().native_incarnation, "CPU-startup-birth");
        }
        *document.lock().unwrap() = (StatusCode::SERVICE_UNAVAILABLE, json!(artifact(&origin, "CPU-current-two")), Duration::ZERO);
        assert!(startup.refresh_current(&http, Duration::from_secs(1)).await.is_err());
        assert!(registry.states().is_empty());
        cancel.cancel();
        tokio::time::timeout(Duration::from_secs(2), task).await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn native_request_current_cohort_refresh_http_deadline_is_bounded() {
        let (origin, document, cancel, task) = server("CPU-current-one").await;
        document.lock().unwrap().2 = Duration::from_millis(200);
        let registry = NativeCleanupRegistry::new();
        let startup = initial(&origin, registry.clone());
        let began = Instant::now();
        assert!(startup.refresh_current(&client(), Duration::from_millis(20)).await.is_err());
        assert!(began.elapsed() < Duration::from_secs(1));
        assert!(registry.states().is_empty());
        cancel.cancel();
        tokio::time::timeout(Duration::from_secs(2), task).await.unwrap().unwrap();
    }
}
