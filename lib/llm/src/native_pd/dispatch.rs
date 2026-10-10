//! Qualified concurrent P/D generation. HTTP completion never releases resources.
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex, OnceLock, Weak},
    time::Duration,
};

use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize, ser::SerializeSeq};
use serde_json::{Value, json};
use tokio::{
    sync::{oneshot, watch},
    task::JoinHandle,
    time::Instant,
};
use tokio_util::sync::CancellationToken;

use super::{
    control::{ControlRequest, ExpectedTransaction, PdRole, RoomBinding},
    qualification::{VerifiedPdPair, VerifiedPdRole},
    transport::{
        NativePdTransport, PdClaimProvider, PdTransportError, QualifiedClaims, encode_bounded,
        transient,
    },
};
use crate::{
    cache_authority::cache_auth,
    cache_authority::frontend_clock::{ChronyClock, ClockBounds},
    cache_authority::native_cache_frontend::{
        CacheGenerationGuard, CacheRequest, NativeCacheFrontend, RenderedCount,
    },
    native_cleanup::{CleanupFailure, CleanupReceipt, CleanupRegistration, NativeCleanupRegistry},
    native_request_cleanup::ORDINARY_REQUEST_WINDOW_MS,
    native_request_lifecycle::{
        QualifiedRequestLifecycle, RequestCleanupRecovery, RequestCleanupTrigger,
    },
};

const CONTROL_TIMEOUT: Duration = Duration::from_secs(5);
const CLEANUP_LIMIT: Duration = Duration::from_secs(30);
const POLL: Duration = Duration::from_millis(100);
const MAX_FRAME_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, thiserror::Error)]
pub enum PdDispatchError {
    #[error("invalid qualified P/D generation: {0}")]
    Invalid(String),
    #[error("P/D generation cancelled")]
    Cancelled,
    #[error("P/D original generation deadline elapsed")]
    Deadline,
    #[error("native P/D {role} generation result unknown: {message}")]
    Unknown { role: &'static str, message: String },
    #[error("native P/D {role} generation_post HTTP status {status}: {cause}")]
    HttpStatus { role: &'static str, status: u16, cause: String },
    #[error("native prefill response invalid: {0}")]
    Prefill(String),
    #[error("P/D private control exceeds the existing 8 MiB resource bound")]
    ResourceBound,
    #[error("native P bootstrap room address space exhausted")]
    RoomExhausted,
    #[error("P/D cleanup registration failed: {0}")]
    Cleanup(String),
    #[error("native P/D cache preparation: {0}")]
    Cache(String),
    #[error("native P/D cache preparation: {0}")]
    CacheSource(#[from] #[source] crate::cache_authority::Error),
    #[error(transparent)]
    Control(#[from] PdTransportError),
}
impl From<CleanupFailure> for PdDispatchError {
    fn from(value: CleanupFailure) -> Self {
        Self::Cleanup(value.to_string())
    }
}

/// Compact native scalar Chat room mapping; no allocation proportional to n.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoomRange {
    pub base: i64,
    pub n: u32,
    pub warmup: Option<i64>,
}

struct RoomAllocator {
    state: Mutex<RoomAllocatorState>,
}
struct RoomAllocatorState {
    next: Option<i64>,
    used: BTreeMap<i64, i64>,
}
impl Default for RoomAllocator {
    fn default() -> Self {
        // Native managers can retain frames across frontend restarts. Keep a
        // fresh server namespace and never reuse a released room in this one.
        let seed = (uuid::Uuid::new_v4().as_u128() as u64) & ((1u64 << 62) - 1);
        Self {
            state: Mutex::new(RoomAllocatorState {
                next: Some(seed as i64 + 1),
                used: BTreeMap::new(),
            }),
        }
    }
}
struct RoomLease {
    allocator: Arc<RoomAllocator>,
    range: RoomRange,
    intervals: [Option<(i64, i64)>; 2],
}
impl Drop for RoomLease {
    fn drop(&mut self) {
        let mut state = self.allocator.state.lock().unwrap();
        for (start, end) in self.intervals.into_iter().flatten() {
            assert_eq!(state.used.remove(&start), Some(end));
        }
    }
}
impl RoomAllocator {
    fn for_pair(pair: &VerifiedPdPair) -> Result<Arc<Self>, PdDispatchError> {
        static ALLOCATORS: OnceLock<Mutex<BTreeMap<String, Weak<RoomAllocator>>>> = OnceLock::new();
        let p = pair.prefill().artifact();
        // Different D partners contend for the same actual P listener. A new
        // P incarnation names a newly qualified physical bootstrap listener.
        let key = serde_json::to_string(&json!({"native_origin":p.native_origin,
            "native_incarnation":p.native_incarnation,"bootstrap_host":p.prefill_bootstrap_host,
            "bootstrap_port":p.prefill_bootstrap_port}))
        .map_err(|error| PdDispatchError::Invalid(error.to_string()))?;
        let mut allocators = ALLOCATORS.get_or_init(Default::default).lock().unwrap();
        allocators.retain(|_, allocator| allocator.strong_count() > 0);
        if let Some(allocator) = allocators.get(&key).and_then(Weak::upgrade) {
            return Ok(allocator);
        }
        let allocator = Arc::new(Self::default());
        allocators.insert(key, Arc::downgrade(&allocator));
        Ok(allocator)
    }
    fn allocate(self: &Arc<Self>, n: u32) -> Result<RoomLease, PdDispatchError> {
        if n == 0 {
            return Err(PdDispatchError::Invalid("positive choices required".into()));
        }
        let needed = i64::from(n) + if n > 1 { 1 } else { 0 };
        let mut state = self.state.lock().unwrap();
        let base = state.next.ok_or(PdDispatchError::RoomExhausted)?;
        let end = base
            .checked_add(needed - 1)
            .ok_or(PdDispatchError::RoomExhausted)?;
        let range = RoomRange::new(base, n)?;
        let intervals = [Some((base, end)), None];
        for (start, end) in intervals.into_iter().flatten() {
            assert!(state.used.insert(start, end).is_none());
        }
        state.next = end.checked_add(1);
        drop(state);
        Ok(RoomLease {
            allocator: self.clone(),
            range,
            intervals,
        })
    }
}
#[cfg(test)]
mod room_namespace_tests {
    use super::*;

    #[test]
    fn released_rooms_are_not_reused_and_namespace_exhaustion_is_closed() {
        let allocator = Arc::new(RoomAllocator::default());
        let first = allocator.allocate(2).unwrap();
        let base = first.range.base;
        assert!(base > 0 && base <= (1i64 << 62));
        assert_eq!(first.range.warmup, Some(base + 2));
        drop(first);
        let second = allocator.allocate(1).unwrap();
        assert_eq!(second.range.base, base + 3);
        drop(second);
        assert!(allocator.state.lock().unwrap().used.is_empty());

        let edge = Arc::new(RoomAllocator {
            state: Mutex::new(RoomAllocatorState {
                next: Some(i64::MAX - 1),
                used: BTreeMap::new(),
            }),
        });
        assert!(matches!(
            edge.allocate(2),
            Err(PdDispatchError::RoomExhausted)
        ));
        assert!(edge.state.lock().unwrap().used.is_empty());
        let last_two = edge.allocate(1).unwrap();
        assert_eq!(last_two.range.base, i64::MAX - 1);
        drop(last_two);
        let last = edge.allocate(1).unwrap();
        assert_eq!(last.range.base, i64::MAX);
        drop(last);
        assert!(matches!(
            edge.allocate(1),
            Err(PdDispatchError::RoomExhausted)
        ));
    }
}

impl RoomRange {
    pub fn new(base: i64, n: u32) -> Result<Self, PdDispatchError> {
        if base < 1 || n == 0 {
            return Err(PdDispatchError::Invalid(
                "positive room and choices required".into(),
            ));
        }
        let last = base.checked_add(i64::from(n) - 1).ok_or_else(|| {
            PdDispatchError::Invalid("sample room range exceeds signed i64".into())
        })?;
        let warmup = (n > 1).then(|| last.checked_add(1).unwrap_or(1));
        if warmup.is_some_and(|room| room >= base && room <= last) {
            return Err(PdDispatchError::Invalid(
                "warmup room overlaps samples".into(),
            ));
        }
        Ok(Self { base, n, warmup })
    }
    /// Bound the actual RID strings during encoding, before creating the Vec.
    pub fn bindings(self, root: &str) -> Result<Vec<RoomBinding>, PdDispatchError> {
        if !hex(root, 32) {
            return Err(PdDispatchError::Invalid(
                "owned root must be UUID hex32".into(),
            ));
        }
        struct Rows<'a>(RoomRange, &'a str);
        impl Serialize for Rows<'_> {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                let mut seq = serializer.serialize_seq(None)?;
                for j in 0..self.0.n {
                    seq.serialize_element(&RoomBinding {
                        room: self.0.base + i64::from(j),
                        sample_rid: if self.0.n == 1 {
                            self.1.into()
                        } else {
                            format!("{}_0:sample:{j}", self.1)
                        },
                    })?;
                }
                if let Some(room) = self.0.warmup {
                    seq.serialize_element(&RoomBinding {
                        room,
                        sample_rid: format!("{}_0:prefill", self.1),
                    })?;
                }
                seq.end()
            }
        }
        let bytes = encode_bounded(&Rows(self, root)).map_err(|e| match e {
            PdTransportError::RequestTooLarge => PdDispatchError::ResourceBound,
            other => other.into(),
        })?;
        serde_json::from_slice(&bytes).map_err(|e| PdDispatchError::Invalid(e.to_string()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PdGenerationTransport {
    pub prefill_incarnation: String,
    pub decode_incarnation: String,
    pub prefill_native_origin: String,
    pub decode_native_origin: String,
    pub kv_compat_digest: String,
    pub qualified_rank_set_digest: String,
    pub transport_epoch: String,
    pub session_epoch: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GenerationBinding {
    pub control: ControlRequest,
    pub transport: PdGenerationTransport,
    pub peer_artifact: super::qualification::PdRoleArtifact,
}
impl PdGenerationTransport {
    pub(crate) fn epoch(&self, txn: Option<&str>) -> Result<String, String> {
        let mut input = json!({"prefill_incarnation":self.prefill_incarnation,"decode_incarnation":self.decode_incarnation,"kv_compat_digest":self.kv_compat_digest,"qualified_rank_set_digest":self.qualified_rank_set_digest});
        if let Some(txn) = txn {
            input["transaction_id"] = json!(txn);
        }
        cache_auth::digest(&input)
    }
}
impl GenerationBinding {
    pub(crate) fn validate_ordinary(
        &self,
        root: &str,
        n: u32,
        txn: &str,
        inc: &str,
        origin: &str,
        own_rank_set_digest: &str,
    ) -> Result<(), String> {
        let c = &self.control;
        let t = &self.transport;
        let peer = &self.peer_artifact;
        peer.validate()?;
        let (peer_role, peer_inc, peer_origin) = match c.role {
            PdRole::Prefill => (
                PdRole::Decode,
                &t.decode_incarnation,
                &t.decode_native_origin,
            ),
            PdRole::Decode => (
                PdRole::Prefill,
                &t.prefill_incarnation,
                &t.prefill_native_origin,
            ),
        };
        if peer.role != peer_role
            || &peer.native_incarnation != peer_inc
            || &peer.native_origin != peer_origin
            || (peer.role == PdRole::Prefill
                && (peer.kv_compat_digest != t.kv_compat_digest
                    || peer.qualified_rank_set_digest != t.qualified_rank_set_digest))
        {
            return Err("P/D peer artifact differs from signed transport identity".into());
        }
        let (own_inc, own_origin) = match c.role {
            PdRole::Prefill => (&t.prefill_incarnation, &t.prefill_native_origin),
            PdRole::Decode => (&t.decode_incarnation, &t.decode_native_origin),
        };
        if c.schema_version != 1
            || !hex(txn, 32)
            || !hex(root, 32)
            || !hex(inc, 32)
            || c.transaction_id != txn
            || c.request_root != root
            || c.expected_native_incarnation != inc
            || own_inc != inc
            || own_origin != origin
            || !hex(&c.expected_qualified_rank_set_digest, 64)
            || c.expected_qualified_rank_set_digest != own_rank_set_digest
            || (c.role == PdRole::Prefill
                && c.expected_qualified_rank_set_digest != t.qualified_rank_set_digest)
            || !hex(&t.prefill_incarnation, 32)
            || !hex(&t.decode_incarnation, 32)
            || t.prefill_incarnation == t.decode_incarnation
            || t.prefill_native_origin.is_empty()
            || t.decode_native_origin.is_empty()
            || t.prefill_native_origin == t.decode_native_origin
            || !hex(&t.kv_compat_digest, 64)
            || !hex(&t.qualified_rank_set_digest, 64)
            || t.epoch(Some(txn))? != t.transport_epoch
            || t.epoch(None)? != t.session_epoch
        {
            return Err("P/D binding does not match ordinary role/transaction identity".into());
        }
        let base = c.rooms.first().ok_or("P/D room mapping empty")?.room;
        let expected = RoomRange::new(base, n)
            .map_err(|e| e.to_string())?
            .bindings(root)
            .map_err(|e| e.to_string())?;
        if c.rooms != expected {
            return Err("P/D rooms differ from exact native scalar choice mapping".into());
        }
        Ok(())
    }
}
fn hex(s: &str, n: usize) -> bool {
    s.len() == n
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

pub struct PdDispatcher {
    pair: Arc<VerifiedPdPair>,
    prefill: Arc<QualifiedRequestLifecycle>,
    decode: Arc<QualifiedRequestLifecycle>,
    client: reqwest::Client,
    rooms: Arc<RoomAllocator>,
}
impl PdDispatcher {
    pub fn new(
        pair: Arc<VerifiedPdPair>,
        prefill: Arc<QualifiedRequestLifecycle>,
        decode: Arc<QualifiedRequestLifecycle>,
    ) -> Result<Self, PdDispatchError> {
        prefill
            .validate_pd_role(pair.prefill())
            .map_err(PdDispatchError::Invalid)?;
        decode
            .validate_pd_role(pair.decode())
            .map_err(PdDispatchError::Invalid)?;
        if !prefill.registry().shares_registry(decode.registry()) {
            return Err(PdDispatchError::Invalid(
                "both roles must share the process cleanup registry".into(),
            ));
        }
        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .connect_timeout(CONTROL_TIMEOUT)
            .build()
            .map_err(|e| PdDispatchError::Invalid(e.to_string()))?;
        Ok(Self {
            rooms: RoomAllocator::for_pair(&pair)?,
            pair,
            prefill,
            decode,
            client,
        })
    }
    pub fn prefill_url(&self) -> &str {
        &self.pair.prefill().artifact().native_origin
    }
    pub fn decode_url(&self) -> &str {
        &self.pair.decode().artifact().native_origin
    }
    pub fn registry(&self) -> &NativeCleanupRegistry {
        self.prefill.registry()
    }
    pub fn native_model(&self) -> &str {
        &self.pair.prefill().artifact().model
    }
    /// Product entry: allocate an exact native room map and one server txn.
    /// A blocked physical cleanup retains this lease through the four holds.
    pub async fn allocate_plan(
        &self,
        body: Value,
        root: &str,
        n: u32,
        start_ms: u64,
    ) -> Result<PdPlan, PdDispatchError> {
        let lease = self.rooms.allocate(n)?;
        let txn = uuid::Uuid::new_v4().simple().to_string();
        let mut plan = self
            .plan(body, root, n, lease.range, start_ms, &txn)
            .await?;
        plan.room_lease = Some(lease);
        Ok(plan)
    }
    /// The Count capability authenticates the server's render selectors before
    /// they enter either role body. Client admission remains the ordinary path.
    pub async fn allocate_cached_plan(
        &self,
        body: Value,
        root: &str,
        n: u32,
        start_ms: u64,
        rendered: &RenderedCount,
    ) -> Result<PdPlan, PdDispatchError> {
        if rendered.worker_url.trim_end_matches('/') != self.prefill_url().trim_end_matches('/') {
            return Err(PdDispatchError::Invalid(
                "cached Count must belong to the actual prefill role".into(),
            ));
        }
        validate_pd_plan_body(&body, Some(rendered))?;
        let lease = self.rooms.allocate(n)?;
        let txn = uuid::Uuid::new_v4().simple().to_string();
        let mut plan = self
            .plan_with_rendered_count(body, root, n, lease.range, start_ms, &txn, Some(rendered))
            .await?;
        plan.room_lease = Some(lease);
        Ok(plan)
    }
    /// Explicit mapping for native wire qualification fixtures. Product callers
    /// use allocate_plan so caller admission resources include the room lease.
    pub async fn plan(
        &self,
        body: Value,
        root: &str,
        n: u32,
        rooms: RoomRange,
        start_ms: u64,
        transaction_id: &str,
    ) -> Result<PdPlan, PdDispatchError> {
        self.plan_with_rendered_count(body, root, n, rooms, start_ms, transaction_id, None)
            .await
    }

    async fn plan_with_rendered_count(
        &self,
        body: Value,
        root: &str,
        n: u32,
        rooms: RoomRange,
        start_ms: u64,
        transaction_id: &str,
        rendered: Option<&RenderedCount>,
    ) -> Result<PdPlan, PdDispatchError> {
        validate_pd_plan_body(&body, rendered)?;
        let p = self.pair.prefill().artifact();
        let d = self.pair.decode().artifact();
        if reqwest::Url::parse(&p.native_origin)
            .map_err(|error| PdDispatchError::Invalid(error.to_string()))?
            .port_or_known_default()
            == p.prefill_bootstrap_port
        {
            return Err(PdDispatchError::Invalid(
                "qualified P bootstrap listener cannot be its native HTTP listener".into(),
            ));
        }
        if !hex(root, 32)
            || !hex(transaction_id, 32)
            || rooms != RoomRange::new(rooms.base, n)?
            || body.get("model").and_then(Value::as_str) != Some(p.model.as_str())
            || body
                .get("rid")
                .or_else(|| body.get("request_id"))
                .and_then(Value::as_str)
                != Some(root)
            || (body.get("rid").is_some() && body.get("request_id").is_some())
            || body.get("n").map_or(Some(1), Value::as_u64) != Some(u64::from(n))
            || !hex(&p.native_incarnation, 32)
            || !hex(&d.native_incarnation, 32)
        {
            return Err(PdDispatchError::Invalid(
                "canonical model/root/n/role incarnation mismatch".into(),
            ));
        }
        let bindings = rooms.bindings(root)?;
        let deadline_ms = start_ms
            .checked_add(ORDINARY_REQUEST_WINDOW_MS)
            .filter(|v| *v <= i64::MAX as u64)
            .ok_or_else(|| PdDispatchError::Invalid("original start deadline overflow".into()))?;
        let mut transport = PdGenerationTransport {
            prefill_incarnation: p.native_incarnation.clone(),
            decode_incarnation: d.native_incarnation.clone(),
            prefill_native_origin: p.native_origin.clone(),
            decode_native_origin: d.native_origin.clone(),
            kv_compat_digest: p.kv_compat_digest.clone(),
            qualified_rank_set_digest: p.qualified_rank_set_digest.clone(),
            transport_epoch: String::new(),
            session_epoch: String::new(),
        };
        transport.transport_epoch = transport
            .epoch(Some(transaction_id))
            .map_err(PdDispatchError::Invalid)?;
        transport.session_epoch = transport.epoch(None).map_err(PdDispatchError::Invalid)?;
        let role_plan = |role: &VerifiedPdRole| -> Result<RoleDraft, PdDispatchError> {
            let a = role.artifact();
            let expected = Arc::new(ExpectedTransaction {
                request: ControlRequest {
                    schema_version: 1,
                    transaction_id: transaction_id.into(),
                    request_root: root.into(),
                    expected_native_incarnation: a.native_incarnation.clone(),
                    expected_qualified_rank_set_digest: a.qualified_rank_set_digest.clone(),
                    role: a.role,
                    rooms: bindings.clone(),
                },
                ranks: a.ranks.iter().cloned().collect(),
            });
            let binding = GenerationBinding {
                control: expected.request.clone(),
                transport: transport.clone(),
                peer_artifact: if a.role == PdRole::Prefill {
                    d.clone()
                } else {
                    p.clone()
                },
            };
            binding
                .validate_ordinary(
                    root,
                    n,
                    transaction_id,
                    &a.native_incarnation,
                    &a.native_origin,
                    &a.qualified_rank_set_digest,
                )
                .map_err(PdDispatchError::Invalid)?;
            encode_bounded(&binding)?;
            let mut unsigned = role_render_body(&body, a.role, rendered)?;
            unsigned["prompt_cache_pd_binding"] = serde_json::to_value(binding)
                .map_err(|e| PdDispatchError::Invalid(e.to_string()))?;
            unsigned["bootstrap_host"] =
                json!(p.prefill_bootstrap_host.as_deref().ok_or_else(|| {
                    PdDispatchError::Invalid("qualified P bootstrap host absent".into())
                })?);
            unsigned["bootstrap_port"] =
                json!(
                    p.prefill_bootstrap_port
                        .ok_or_else(|| PdDispatchError::Invalid(
                            "qualified P bootstrap port absent".into()
                        ))?
                );
            unsigned["bootstrap_room"] = json!(rooms.base);
            let secret: [u8; 32] = role
                .secret_bytes()
                .try_into()
                .map_err(|_| PdDispatchError::Invalid("qualified HMAC key size mismatch".into()))?;
            let rpc = Arc::new(NativePdTransport::new(
                &a.native_origin,
                secret,
                CONTROL_TIMEOUT,
            )?);
            let claims = Arc::new(PdClaims {
                model: a.model.clone(),
                origin: a.native_origin.clone(),
                incarnation: a.native_incarnation.clone(),
                source: a.source_digest.clone(),
                profile: a.runtime_profile_digest.clone(),
                root: root.into(),
                start_ms,
                deadline_ms,
                control_digest: cache_auth::digest(
                    &serde_json::to_value(&expected.request)
                        .map_err(|e| PdDispatchError::Invalid(e.to_string()))?,
                )
                .map_err(PdDispatchError::Invalid)?,
            });
            Ok(RoleDraft {
                body: unsigned,
                expected,
                claims,
                rpc,
            })
        };
        let prefill = role_plan(self.pair.prefill())?;
        let decode = role_plan(self.pair.decode())?;
        // Full exact room/body digests precede this fresh clock estimate.
        // Use its conservative upper bound to avoid extending the old fence.
        let clock = ChronyClock
            .qualify()
            .await
            .map_err(|e| PdDispatchError::Invalid(e.to_string()))?;
        clock
            .ensure_fresh()
            .map_err(|e| PdDispatchError::Invalid(e.to_string()))?;
        if clock.lower_ms() < start_ms || clock.upper_ms() >= deadline_ms {
            return Err(PdDispatchError::Deadline);
        }
        let admission_fence = clock
            .observed_at()
            .checked_add(Duration::from_millis(deadline_ms - clock.upper_ms()))
            .ok_or(PdDispatchError::Deadline)?;
        Ok(PdPlan {
            prefill: prefill.finish(admission_fence),
            decode: decode.finish(admission_fence),
            prefill_lifecycle: self.prefill.clone(),
            decode_lifecycle: self.decode.clone(),
            client: self.client.clone(),
            root: root.into(),
            n,
            start_ms,
            transaction_id: transaction_id.into(),
            stream: body.get("stream").and_then(Value::as_bool).unwrap_or(false),
            room_lease: None,
        })
    }
}
pub(crate) fn validate_pd_plan_body(
    body: &Value,
    rendered: Option<&RenderedCount>,
) -> Result<(), PdDispatchError> {
    if let Some(rendered) = rendered {
        rendered
            .validate_server_render_plan(&body)
            .map_err(|error| PdDispatchError::Invalid(error.to_string()))?;
    }
    let fields = body
        .as_object()
        .ok_or_else(|| PdDispatchError::Invalid("Chat body must be object".into()))?;
    if fields.keys().any(|key| {
        (key.starts_with("prompt_cache_")
            && key != "prompt_cache_key"
            && !(rendered.is_some()
                && matches!(
                    key.as_str(),
                    "prompt_cache_blocks"
                        | "prompt_cache_read_candidates"
                        | "prompt_cache_message_order"
                )))
            || matches!(
                key.as_str(),
                "reservation_id"
                    | "authority_revision"
                    | "request_started_at_ms"
                    | "retain_for_ms"
                    | "_verified_claims"
                    | "bootstrap_host"
                    | "bootstrap_port"
                    | "bootstrap_room"
                    | "disagg_prefill_dp_rank"
            )
    }) {
        return Err(PdDispatchError::Invalid(
            "client private P/D or authority metadata forbidden".into(),
        ));
    }
    Ok(())
}

fn role_render_body(
    body: &Value,
    role: PdRole,
    rendered: Option<&RenderedCount>,
) -> Result<Value, PdDispatchError> {
    if let Some(rendered) = rendered {
        rendered
            .validate_server_render_plan(body)
            .map_err(|error| PdDispatchError::Invalid(error.to_string()))?;
    }
    let mut projected = body.clone();
    if rendered.is_some() && role == PdRole::Decode {
        // Only P owns the cache renderer and reservation. D retains the same
        // public request and ordinary lifecycle, with no cache ledger fields.
        for key in [
            "prompt_cache_blocks",
            "prompt_cache_read_candidates",
            "prompt_cache_message_order",
        ] {
            projected
                .as_object_mut()
                .ok_or_else(|| PdDispatchError::Invalid("Chat body must be object".into()))?
                .remove(key);
        }
        if cache_auth::ordinary_body_for(body).map_err(PdDispatchError::Invalid)?
            != cache_auth::ordinary_body_for(&projected).map_err(PdDispatchError::Invalid)?
        {
            return Err(PdDispatchError::Invalid(
                "decode cache projection changed the ordinary request".into(),
            ));
        }
    }
    Ok(projected)
}

struct RolePlan {
    body: Value,
    context: Arc<PdContext>,
}
struct RoleDraft {
    body: Value,
    expected: Arc<ExpectedTransaction>,
    claims: Arc<PdClaims>,
    rpc: Arc<NativePdTransport>,
}
impl RoleDraft {
    fn finish(self, admission_fence: Instant) -> RolePlan {
        RolePlan {
            body: self.body,
            context: Arc::new(PdContext {
                expected: self.expected,
                claims: self.claims,
                rpc: self.rpc,
                admission_fence,
            }),
        }
    }
}
pub struct PdPlan {
    prefill: RolePlan,
    decode: RolePlan,
    prefill_lifecycle: Arc<QualifiedRequestLifecycle>,
    decode_lifecycle: Arc<QualifiedRequestLifecycle>,
    client: reqwest::Client,
    root: String,
    n: u32,
    start_ms: u64,
    transaction_id: String,
    stream: bool,
    room_lease: Option<RoomLease>,
}
impl PdPlan {
    pub fn room_range(&self) -> RoomRange {
        RoomRange::new(self.prefill.context.expected.request.rooms[0].room, self.n)
            .expect("validated native choice mapping")
    }
}

struct PdClaims {
    model: String,
    origin: String,
    incarnation: String,
    source: String,
    profile: String,
    root: String,
    start_ms: u64,
    deadline_ms: u64,
    control_digest: String,
}
#[async_trait]
impl PdClaimProvider for PdClaims {
    async fn clock_bounds(&self) -> Result<ClockBounds, String> {
        let clock = ChronyClock.qualify().await.map_err(|e| e.to_string())?;
        clock.ensure_fresh().map_err(|e| e.to_string())?;
        Ok(clock)
    }
    async fn claims(&self, operation: &str) -> Result<QualifiedClaims, String> {
        if !matches!(operation, "pd-status" | "pd-abort") {
            return Err("unsupported P/D control operation".into());
        }
        // control_digest was computed from the complete exact room payload
        // before this fresh qualification. No unsynchronized wall fallback.
        let clock = ChronyClock.qualify().await.map_err(|e| e.to_string())?;
        clock.ensure_fresh().map_err(|e| e.to_string())?;
        if clock.lower_ms() < self.start_ms || clock.upper_ms() >= self.deadline_ms {
            return Err("P/D original admission fence expired or reversed".into());
        }
        let now = clock.lower_ms();
        let claims = json!({"version":1,"operation":operation,"operation_id":uuid::Uuid::new_v4().simple().to_string(),"nonce":uuid::Uuid::new_v4().simple().to_string(),"model":self.model,"native_origin":self.origin,"native_incarnation":self.incarnation,"source_digest":self.source,"runtime_profile_digest":self.profile,"reservation_id":"","authority_revision":0,"native_request_id":self.root,"namespace_digest":"","full_render_digest":"","token_ids_digest":"","request_started_at_ms":self.start_ms,"issued_at_ms":now,"deadline_ms":self.deadline_ms,"retain_for_ms":0,"payload_digest":self.control_digest});
        cache_auth::validate_claims(&claims)?;
        clock.ensure_fresh().map_err(|e| e.to_string())?;
        Ok(QualifiedClaims { claims, clock })
    }
}
struct PdContext {
    expected: Arc<ExpectedTransaction>,
    claims: Arc<PdClaims>,
    rpc: Arc<NativePdTransport>,
    admission_fence: Instant,
}
pub struct PdCleanupTrigger {
    sender: Option<oneshot::Sender<Instant>>,
}
impl Drop for PdCleanupTrigger {
    fn drop(&mut self) {
        if let Some(sender) = self.sender.take() {
            let _ = sender.send(Instant::now());
        }
    }
}
#[derive(Clone)]
pub struct PdCleanupRecovery {
    context: Arc<PdContext>,
    registry: NativeCleanupRegistry,
}
impl PdCleanupRecovery {
    pub fn cleanup_id(&self) -> String {
        self.context.expected.cleanup_id()
    }
    pub fn retry_blocked(&self, registry: &NativeCleanupRegistry) -> Result<(), CleanupFailure> {
        if !registry.shares_registry(&self.registry) {
            return Err(CleanupFailure::Verification(
                "P/D recovery belongs to another registry".into(),
            ));
        }
        let context = self.context.clone();
        registry.retry_blocked(self.cleanup_id(), async move {
            context.cleanup(Instant::now()).await
        })
    }
}
impl PdContext {
    async fn cleanup(&self, finished: Instant) -> Result<CleanupReceipt, CleanupFailure> {
        let deadline = finished
            .checked_add(CLEANUP_LIMIT)
            .ok_or_else(|| CleanupFailure::Verification("cleanup deadline overflow".into()))?
            .min(self.admission_fence);
        CleanupReceipt::from_pd(
            self.rpc
                .cleanup(self.expected.clone(), self.claims.clone(), deadline, POLL)
                .await?,
        )
    }
    fn registration(
        self: &Arc<Self>,
        registry: &NativeCleanupRegistry,
        hold: Arc<Mutex<Box<dyn Send>>>,
    ) -> Result<(CleanupRegistration, PdCleanupTrigger, PdCleanupRecovery), CleanupFailure> {
        if Instant::now() >= self.admission_fence {
            return Err(CleanupFailure::Pd(PdTransportError::Deadline));
        }
        let (sender, receiver) = oneshot::channel();
        let context = self.clone();
        // Retain immutable context with the hold so a blocked operation remains
        // recoverable without rediscovering an incarnation or extending time.
        let registration = CleanupRegistration::new(
            self.expected.cleanup_id(),
            async move {
                let finished = tokio::select! {v=receiver=>v.unwrap_or_else(|_|Instant::now()),_=tokio::time::sleep_until(context.admission_fence)=>return Err(CleanupFailure::Pd(PdTransportError::Deadline))};
                context.cleanup(finished).await
            },
            Box::new((hold, self.clone())),
        );
        Ok((
            registration,
            PdCleanupTrigger {
                sender: Some(sender),
            },
            PdCleanupRecovery {
                context: self.clone(),
                registry: registry.clone(),
            },
        ))
    }
    async fn successful_transfer(&self) -> Result<(), PdDispatchError> {
        loop {
            match self.rpc.status(&self.expected, self.claims.as_ref()).await {
                Ok(receipt) => {
                    let r = receipt.response();
                    if r.aborted {
                        return Err(PdDispatchError::Prefill(
                            "native P/D transaction aborted before successful terminal".into(),
                        ));
                    }
                    let qualified = receipt.physical_terminal()
                        && r.rank_states.iter().all(|rank| {
                            rank.rooms.iter().all(|room| match r.role {
                                PdRole::Prefill => room.transfer_complete && room.transport_drained,
                                PdRole::Decode => room.decode_committed && room.transport_drained,
                            })
                        });
                    if qualified {
                        return Ok(());
                    }
                }
                Err(e) if transient(&e) => {}
                Err(e) => return Err(e.into()),
            }
            tokio::time::sleep(POLL).await;
        }
    }
}

#[derive(Clone)]
enum PrefillState {
    Pending,
    Complete,
    Failed(PdDispatchError),
}
pub struct PrefillMonitor {
    state: watch::Receiver<PrefillState>,
    task: JoinHandle<()>,
    cancel: CancellationToken,
}
impl PrefillMonitor {
    /// Remains pending after successful P HTTP completion; use in D's per-frame
    /// select to catch P failure, without manufacturing a D output frame.
    pub async fn failure(&mut self) -> PdDispatchError {
        loop {
            if let PrefillState::Failed(error) = self.state.borrow().clone() {
                return error;
            }
            if self.state.changed().await.is_err() {
                return PdDispatchError::Prefill("P monitor ended without state".into());
            }
        }
    }
    pub async fn wait_http_success(&mut self) -> Result<(), PdDispatchError> {
        loop {
            let state = self.state.borrow().clone();
            match state {
                PrefillState::Complete => return Ok(()),
                PrefillState::Failed(error) => return Err(error),
                PrefillState::Pending => {}
            }
            self.state
                .changed()
                .await
                .map_err(|_| PdDispatchError::Prefill("P monitor ended without terminal".into()))?;
        }
    }
}
impl Drop for PrefillMonitor {
    fn drop(&mut self) {
        self.cancel.cancel();
        self.task.abort();
    }
}

pub struct PdGeneration {
    decode_response: Option<reqwest::Response>,
    pub prefill_monitor: PrefillMonitor,
    _ordinary_guards: [RequestCleanupTrigger; 2],
    _pd_guards: [PdCleanupTrigger; 2],
    pub ordinary_recovery: [RequestCleanupRecovery; 2],
    pub pd_recovery: [PdCleanupRecovery; 2],
    prefill: Arc<PdContext>,
    decode: Arc<PdContext>,
    cancel: CancellationToken,
}
impl PdGeneration {
    pub fn take_decode_response(&mut self) -> Result<reqwest::Response, PdDispatchError> {
        self.decode_response
            .take()
            .ok_or_else(|| PdDispatchError::Invalid("D response already consumed".into()))
    }
    /// Parent must call after validating D JSON or all D choices + DONE, and
    /// before exposing a public successful terminal. HTTP P EOF is insufficient.
    pub async fn qualify_successful_terminal(&mut self) -> Result<(), PdDispatchError> {
        let fence = self
            .prefill
            .admission_fence
            .min(self.decode.admission_fence);
        tokio::select! {biased;
            _=self.cancel.cancelled()=>Err(PdDispatchError::Cancelled),
            _=tokio::time::sleep_until(fence)=>Err(PdDispatchError::Deadline),
            result=async {self.prefill_monitor.wait_http_success().await?;tokio::try_join!(self.prefill.successful_transfer(),self.decode.successful_transfer())?;Ok(())}=>result,
        }
    }
}
impl Drop for PdGeneration {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}
impl PdPlan {
    /// All four tracked entries share this resource. Cancellation or any partial
    /// registration failure triggers accepted entries, never a local release ACK.
    pub async fn dispatch(
        self,
        resource_hold: Box<dyn Send>,
        cancel: CancellationToken,
    ) -> Result<PdGeneration, PdDispatchError> {
        self.dispatch_inner(resource_hold, cancel, None)
            .await
            .map(|(generation, _)| generation)
    }

    pub async fn dispatch_cached(
        self,
        resource_hold: Box<dyn Send>,
        cancel: CancellationToken,
        frontend: Arc<NativeCacheFrontend>,
        request: &CacheRequest,
        rendered: RenderedCount,
        epoch: Option<u64>,
    ) -> Result<(PdGeneration, CacheGenerationGuard), PdDispatchError> {
        let (generation, guard) = self
            .dispatch_inner(
                resource_hold,
                cancel,
                Some((frontend, request, rendered, epoch)),
            )
            .await?;
        Ok((
            generation,
            guard.ok_or_else(|| PdDispatchError::Cache("cache guard missing".into()))?,
        ))
    }

    async fn dispatch_inner(
        self,
        resource_hold: Box<dyn Send>,
        cancel: CancellationToken,
        cache: Option<(
            Arc<NativeCacheFrontend>,
            &CacheRequest,
            RenderedCount,
            Option<u64>,
        )>,
    ) -> Result<(PdGeneration, Option<CacheGenerationGuard>), PdDispatchError> {
        // Follow caller cancellation without cancelling its wider task tree on
        // normal generation drop or after a local dispatch failure.
        let cancel = cancel.child_token();
        struct GenerationResources {
            _caller: Box<dyn Send>,
            _rooms: Option<RoomLease>,
        }
        let hold: Arc<Mutex<Box<dyn Send>>> = Arc::new(Mutex::new(Box::new(GenerationResources {
            _caller: resource_hold,
            _rooms: self.room_lease,
        })));
        let mut prefill_body = self.prefill.body;
        if let Some((_, _, rendered, _)) = &cache {
            prefill_body["cache_salt"] = rendered
                .cache_salt()
                .map_err(PdDispatchError::from)?
                .into();
            if rendered.request_started_at_ms() != self.start_ms {
                return Err(PdDispatchError::Cache(
                    "cache Count and P/D original origin differ".into(),
                ));
            }
        }
        // Both signed bodies are prepared locally before either ordinary
        // cleanup is armed. A rejected body has never created native ingress.
        let d = self
            .decode_lifecycle
            .prepare_pd_generation(
                self.decode.body,
                &self.root,
                self.n,
                self.start_ms,
                &self.transaction_id,
                &self.decode.context.expected.request.expected_qualified_rank_set_digest,
                Box::new(hold.clone()),
            )
            .await?;
        let (decode_body, d_registration) = d.into_parts();
        let p = if cache.is_some() {
            self.prefill_lifecycle
                .prepare_pd_cached_generation(
                    prefill_body,
                    &self.root,
                    self.n,
                    self.start_ms,
                    &self.transaction_id,
                    &self.prefill.context.expected.request.expected_qualified_rank_set_digest,
                    Box::new(hold.clone()),
                )
                .await?
        } else {
            self.prefill_lifecycle
                .prepare_pd_generation(
                    prefill_body,
                    &self.root,
                    self.n,
                    self.start_ms,
                    &self.transaction_id,
                    &self.prefill.context.expected.request.expected_qualified_rank_set_digest,
                    Box::new(hold.clone()),
                )
                .await?
        };
        let (prefill_body, p_registration) = p.into_parts();
        let fence = self
            .prefill
            .context
            .admission_fence
            .min(self.decode.context.admission_fence);
        let (prefill_body, cache_guard) = if let Some((frontend, request, rendered, epoch)) = cache
        {
            let prepared = frontend
                .prepare_generation(
                    &self.prefill.context.claims.origin,
                    prefill_body,
                    request,
                    rendered,
                    epoch,
                    fence,
                )
                .await
                .map_err(PdDispatchError::from)?;
            (prepared.body, Some(prepared.guard))
        } else {
            (prefill_body, None)
        };
        if cancel.is_cancelled() {
            return Err(PdDispatchError::Cancelled);
        }
        if Instant::now() >= fence {
            return Err(PdDispatchError::Deadline);
        }
        // All large ordinary JSON and registration preparation precede the
        // cache owner's final fresh HMAC. Cached P bytes are sealed by Begin.
        let decode_bytes = serde_json::to_vec(&decode_body)
            .map_err(|e| PdDispatchError::Invalid(e.to_string()))?;
        let ordinary_prefill_bytes = if cache_guard.is_none() {
            Some(
                serde_json::to_vec(&prefill_body)
                    .map_err(|e| PdDispatchError::Invalid(e.to_string()))?,
            )
        } else {
            None
        };
        let (ordinary_p, ordinary_p_guard, ordinary_p_recovery) =
            p_registration.into_registration()?;
        let (ordinary_d, ordinary_d_guard, ordinary_d_recovery) =
            d_registration.into_registration()?;
        let (pd_p, pguard, precovery) = self
            .prefill
            .context
            .registration(self.prefill_lifecycle.registry(), hold.clone())?;
        let (pd_d, dguard, drecovery) = self
            .decode
            .context
            .registration(self.decode_lifecycle.registry(), hold.clone())?;
        let prefill_bytes = if let Some(guard) = &cache_guard {
            guard
                .begin_generation()
                .await
                .map_err(PdDispatchError::from)?
        } else {
            ordinary_prefill_bytes.expect("ordinary P body encoded before registration")
        };
        if cancel.is_cancelled() {
            return Err(PdDispatchError::Cancelled);
        }
        if Instant::now() >= fence {
            return Err(PdDispatchError::Deadline);
        }
        // The existing registry lock accepts all four controls or none. Cache
        // reservation cleanup remains independently owned by cache_guard.
        self.prefill_lifecycle
            .registry()
            .register_group(vec![ordinary_p, ordinary_d, pd_p, pd_d])?;
        drop(hold);
        let monitor = start_prefill(
            self.client.clone(),
            format!("{}/v1/chat/completions", self.prefill.context.claims.origin),
            prefill_bytes,
            self.n,
            self.stream,
            fence,
            cancel.child_token(),
        );
        let mut monitor = monitor;
        let dpost = self
            .client
            .post(format!(
                "{}/v1/chat/completions",
                self.decode.context.claims.origin
            ))
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(decode_bytes)
            .send();
        let response = tokio::select! {biased;
            _=cancel.cancelled()=>return Err(PdDispatchError::Cancelled),
            _=tokio::time::sleep_until(fence)=>return Err(PdDispatchError::Deadline),
            error=monitor.failure()=>return Err(error),
            response=dpost=>response.map_err(|e|PdDispatchError::Unknown {role:"decode",message:e.to_string()})?,
        };
        if !response.status().is_success() {
            let failure = observe_generation_http_failure(response, "decode", fence).await;
            return Err(PdDispatchError::HttpStatus {
                role: "decode",
                status: failure.status,
                cause: failure.cause(),
            });
        }
        Ok((
            PdGeneration {
                decode_response: Some(response),
                prefill_monitor: monitor,
                _ordinary_guards: [ordinary_p_guard, ordinary_d_guard],
                _pd_guards: [pguard, dguard],
                ordinary_recovery: [ordinary_p_recovery, ordinary_d_recovery],
                pd_recovery: [precovery, drecovery],
                prefill: self.prefill.context,
                decode: self.decode.context,
                cancel,
            },
            cache_guard,
        ))
    }
}

const MAX_ERROR_BODY_BYTES: usize = 8 * 1024;
const ERROR_BODY_READ_LIMIT: Duration = Duration::from_millis(250);

pub(crate) struct GenerationHttpFailure {
    pub(crate) status: u16,
    body_state: &'static str,
    // Only the native error envelope's scalar diagnostic fields survive.
    // Request echoes, authority proofs and opaque values are never logged.
    error: Value,
}
impl GenerationHttpFailure {
    pub(crate) fn cause(&self) -> String {
        if self.error.is_null() {
            format!("native error body {}", self.body_state)
        } else {
            // This value has already passed the bounded diagnostic filter.
            format!("{}: {}", self.body_state, self.error)
        }
    }
}

fn native_error_fields(body: &[u8]) -> Option<Value> {
    let value: Value = serde_json::from_slice(body).ok()?;
    native_error_fields_from_value(&value)
}

/// Bounded, filtered native error fields shared with the frontend startup loader.
pub fn native_error_fields_from_value(value: &Value) -> Option<Value> {
    // The native admission middleware returns {"error": "reason"}. Route its
    // scalar through the same bounded diagnostic filter as object messages.
    if let Some(message) = value.get("error").and_then(Value::as_str) {
        return native_error_fields_from_value(&json!({"message": message}));
    }
    let error = value.get("error").unwrap_or(value);
    let mut fields = serde_json::Map::new();
    for name in ["message", "detail", "type", "code", "param"] {
        let Some(value) = error.get(name) else {
            continue;
        };
        let value = match value {
            Value::Array(errors) if name == "detail" => {
                let details = errors
                    .iter()
                    .take(2)
                    .filter_map(|error| {
                        let mut summary = serde_json::Map::new();
                        for (source, target) in [("msg", "message"), ("type", "type")] {
                            if let Some(value) = error.get(source).filter(|value| value.is_string())
                            {
                                summary.insert(target.into(), value.clone());
                            }
                        }
                        if let Some(location) = error.get("loc").and_then(Value::as_array) {
                            let location = location
                                .iter()
                                .take(16)
                                .filter_map(|part| match part {
                                    Value::String(part) if part.len() <= 64 => Some(part.clone()),
                                    Value::Number(part) => Some(part.to_string()),
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join(".");
                            summary.insert("param".into(), Value::String(location));
                        }
                        // Pydantic's input/context may contain the entire signed
                        // request. Reuse the scalar filter on only type/loc/msg.
                        native_error_fields_from_value(&Value::Object(summary))
                    })
                    .collect();
                Value::Array(details)
            }
            Value::String(text) => {
                if text.len() > MAX_ERROR_BODY_BYTES {
                    fields.insert(
                        name.into(),
                        Value::String("[field exceeds diagnostic byte limit]".into()),
                    );
                    continue;
                }
                let lower = text.to_ascii_lowercase();
                if [
                    "prompt_cache_",
                    "proof",
                    "secret",
                    "authorization",
                    "bearer",
                    "api_key",
                    "\"messages\"",
                ]
                .iter()
                .any(|marker| lower.contains(marker))
                {
                    Value::String("[redacted sensitive native error]".into())
                } else {
                    let mut safe = String::new();
                    let mut token = String::new();
                    let flush = |token: &mut String, safe: &mut String| {
                        if token.len() >= 32 {
                            safe.push_str("[redacted opaque value]");
                        } else {
                            safe.push_str(token);
                        }
                        token.clear();
                    };
                    for ch in text.chars() {
                        if ch.is_ascii_alphanumeric() || matches!(ch, '+' | '/' | '_' | '=' | '-') {
                            token.push(ch);
                        } else {
                            flush(&mut token, &mut safe);
                            safe.push(if ch.is_control() { ' ' } else { ch });
                        }
                    }
                    flush(&mut token, &mut safe);
                    if safe.len() > 512 {
                        let mut end = 512;
                        while !safe.is_char_boundary(end) {
                            end -= 1;
                        }
                        safe.truncate(end);
                        safe.push_str("[field truncated]");
                    }
                    Value::String(safe)
                }
            }
            Value::Number(_) | Value::Bool(_) | Value::Null => value.clone(),
            _ => continue,
        };
        fields.insert(name.into(), value);
    }
    (!fields.is_empty()).then_some(Value::Object(fields))
}

pub(crate) fn observe_generation_error_packet(
    value: &Value,
    role: &'static str,
    phase: &'static str,
) {
    let error = native_error_fields_from_value(value).unwrap_or(Value::Null);
    tracing::warn!(
        native_role = role,
        native_phase = phase,
        native_error_body_state = "parsed_error_packet_summary",
        native_error = %error,
        "native P/D generation error packet"
    );
}

async fn read_generation_http_failure(
    response: reqwest::Response,
    fence: Instant,
) -> GenerationHttpFailure {
    let status = response.status().as_u16();
    let deadline = fence.min(Instant::now() + ERROR_BODY_READ_LIMIT);
    let read = async {
        if Instant::now() >= deadline {
            return Err("read_deadline_elapsed");
        }
        let mut bytes = Vec::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            if Instant::now() >= deadline {
                return Err("read_deadline_elapsed");
            }
            let chunk = chunk.map_err(|_| "read_failed")?;
            if chunk.len() > MAX_ERROR_BODY_BYTES.saturating_sub(bytes.len()) {
                return Err("truncated_at_byte_limit");
            }
            bytes.extend_from_slice(&chunk);
        }
        if bytes.is_empty() {
            return Err("empty_body");
        }
        native_error_fields(&bytes).ok_or("non_json_or_no_error_fields")
    };
    let (body_state, error) = match tokio::time::timeout_at(deadline, read).await {
        Ok(Ok(error)) => ("complete_json_error_summary", error),
        Ok(Err(reason)) => (reason, Value::Null),
        Err(_) => ("read_deadline_elapsed", Value::Null),
    };
    GenerationHttpFailure {
        status,
        body_state,
        error,
    }
}

pub(crate) async fn observe_generation_http_failure(
    response: reqwest::Response,
    role: &'static str,
    fence: Instant,
) -> GenerationHttpFailure {
    struct Diagnostic {
        role: &'static str,
        origin: String,
        path: String,
        failure: GenerationHttpFailure,
    }
    impl Drop for Diagnostic {
        fn drop(&mut self) {
            tracing::warn!(
                native_role = self.role,
                native_origin = %self.origin,
                native_path = %self.path,
                native_phase = "generation_post",
                native_status = self.failure.status,
                native_error_body_state = self.failure.body_state,
                native_error = %self.failure.error,
                "native generation HTTP failure"
            );
        }
    }
    // Cancellation of the P monitor still records the known HTTP status.
    // This is a diagnostic guard only, with no resource/cleanup ownership.
    let mut diagnostic = Diagnostic {
        role,
        origin: response.url().origin().ascii_serialization(),
        path: response.url().path().to_owned(),
        failure: GenerationHttpFailure {
            status: response.status().as_u16(),
            body_state: "read_cancelled",
            error: Value::Null,
        },
    };
    diagnostic.failure = read_generation_http_failure(response, fence).await;
    GenerationHttpFailure {
        status: diagnostic.failure.status,
        body_state: diagnostic.failure.body_state,
        error: diagnostic.failure.error.clone(),
    }
}

fn start_prefill(
    client: reqwest::Client,
    url: String,
    body: Vec<u8>,
    n: u32,
    stream: bool,
    fence: Instant,
    cancel: CancellationToken,
) -> PrefillMonitor {
    let (sender, state) = watch::channel(PrefillState::Pending);
    let task_cancel = cancel.clone();
    let task = tokio::spawn(async move {
        let result = tokio::select! {biased;
            _=task_cancel.cancelled()=>Err(PdDispatchError::Cancelled),
            _=tokio::time::sleep_until(fence)=>Err(PdDispatchError::Deadline),
            result=async {let response=client.post(url).header(reqwest::header::CONTENT_TYPE,"application/json").body(body).send().await.map_err(|e|PdDispatchError::Unknown {role:"prefill",message:e.to_string()})?; consume_prefill(response,n,stream,fence).await}=>result,
        };
        sender.send_replace(match result {
            Ok(()) => PrefillState::Complete,
            Err(e) => PrefillState::Failed(e),
        });
        // Keep the channel alive after success so failure() is never spuriously
        // ready while D continues. Drop cancels this owned task.
        task_cancel.cancelled().await;
    });
    PrefillMonitor {
        state,
        task,
        cancel,
    }
}
fn finish_choices(
    value: &Value,
    n: u32,
    finished: &mut BTreeSet<u32>,
    full: bool,
) -> Result<(), PdDispatchError> {
    if value.get("error").is_some() {
        observe_generation_error_packet(
            value,
            "prefill",
            if full {
                "generation_json"
            } else {
                "generation_sse"
            },
        );
        return Err(PdDispatchError::Prefill("native error packet".into()));
    }
    let choices = value
        .get("choices")
        .and_then(Value::as_array)
        .ok_or_else(|| PdDispatchError::Prefill("missing choices".into()))?;
    let mut frame = BTreeSet::new();
    for choice in choices {
        let index = choice
            .get("index")
            .and_then(Value::as_u64)
            .and_then(|v| u32::try_from(v).ok())
            .filter(|v| *v < n)
            .ok_or_else(|| PdDispatchError::Prefill("invalid choice index".into()))?;
        if !frame.insert(index) {
            return Err(PdDispatchError::Prefill("duplicate choice index".into()));
        }
        match choice.get("finish_reason") {
            Some(Value::String(reason))
                if matches!(
                    reason.as_str(),
                    "stop" | "length" | "tool_calls" | "function_call" | "content_filter"
                ) =>
            {
                finished.insert(index);
            }
            None | Some(Value::Null) if !full => {}
            _ => {
                return Err(PdDispatchError::Prefill(
                    "invalid or missing finish reason".into(),
                ));
            }
        }
    }
    if full && finished.len() != n as usize {
        return Err(PdDispatchError::Prefill("incomplete choices".into()));
    }
    Ok(())
}
async fn consume_prefill(
    response: reqwest::Response,
    n: u32,
    stream: bool,
    fence: Instant,
) -> Result<(), PdDispatchError> {
    if !response.status().is_success() {
        let failure = observe_generation_http_failure(response, "prefill", fence).await;
        return Err(PdDispatchError::HttpStatus {
            role: "prefill",
            status: failure.status,
            cause: failure.cause(),
        });
    }
    let mut bytes = response.bytes_stream();
    let mut buffer = Vec::new();
    let mut finished = BTreeSet::new();
    let mut done = false;
    while let Some(chunk) = bytes.next().await {
        let chunk = chunk.map_err(|e| PdDispatchError::Unknown {
            role: "prefill",
            message: e.to_string(),
        })?;
        if buffer
            .len()
            .checked_add(chunk.len())
            .is_none_or(|v| v > MAX_FRAME_BYTES)
        {
            return Err(PdDispatchError::ResourceBound);
        }
        buffer.extend_from_slice(&chunk);
        if stream {
            while let Some(end) = buffer.iter().position(|b| *b == b'\n') {
                let line = buffer.drain(..=end).collect::<Vec<_>>();
                let text = std::str::from_utf8(&line)
                    .map_err(|_| PdDispatchError::Prefill("SSE not UTF-8".into()))?
                    .trim_end_matches(['\r', '\n']);
                if text.is_empty() || text.starts_with(':') {
                    continue;
                }
                let data = text
                    .strip_prefix("data:")
                    .map(str::trim_start)
                    .ok_or_else(|| PdDispatchError::Prefill("unexpected SSE field".into()))?;
                if done {
                    return Err(PdDispatchError::Prefill("SSE data after DONE".into()));
                }
                if data == "[DONE]" {
                    if finished.len() != n as usize {
                        return Err(PdDispatchError::Prefill(
                            "DONE before every P choice completed".into(),
                        ));
                    }
                    done = true;
                } else {
                    let value: Value = serde_json::from_str(data)
                        .map_err(|_| PdDispatchError::Prefill("invalid SSE JSON".into()))?;
                    finish_choices(&value, n, &mut finished, false)?;
                }
            }
        }
    }
    if stream {
        if !done || buffer.iter().any(|b| !b.is_ascii_whitespace()) {
            return Err(PdDispatchError::Prefill(
                "empty or truncated P stream".into(),
            ));
        }
    } else {
        let value: Value = serde_json::from_slice(&buffer)
            .map_err(|_| PdDispatchError::Prefill("empty or invalid P JSON".into()))?;
        finish_choices(&value, n, &mut finished, true)?;
    }
    Ok(())
}

#[cfg(test)]
mod generation_http_error_tests {
    use super::*;
    use std::io::Write;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tracing::instrument::WithSubscriber;

    #[derive(Clone)]
    struct LogWriter(Arc<Mutex<Vec<u8>>>);
    impl Write for LogWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    async fn error_response(body: Vec<u8>, delay: Duration) -> reqwest::Response {
        native_response(body, delay, 400).await
    }

    async fn native_response(body: Vec<u8>, delay: Duration, status: u16) -> reqwest::Response {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0; 4096];
            let count = socket.read(&mut request).await.unwrap();
            assert!(
                std::str::from_utf8(&request[..count])
                    .unwrap()
                    .starts_with("POST /")
            );
            let header = format!(
                "HTTP/1.1 {status} Native Response\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            socket.write_all(header.as_bytes()).await.unwrap();
            tokio::time::sleep(delay).await;
            // A diagnostic read deadline can legitimately close this response.
            let _ = socket.write_all(&body).await;
        });
        reqwest::Client::builder()
            .no_proxy()
            .build()
            .unwrap()
            .post(format!("http://{address}/v1/chat/completions"))
            .send()
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn native_pd_non2xx_retains_bounded_role_phase_diagnostics_without_authority_echo() {
        let logs = Arc::new(Mutex::new(Vec::new()));
        let writer = LogWriter(logs.clone());
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .without_time()
            .with_writer(move || writer.clone())
            .finish();
        async {
            let body = serde_json::to_vec(&json!({"error": {
                "message":"ordinary request context unavailable", "type":"invalid_request_error",
                "code":400, "param":null,
                "prompt_cache_authority_proof":{"mac":"opaque-proof-must-not-appear"},
                "authority_secret":"opaque-secret-must-not-appear"},
                "messages":[{"content":"request-echo-must-not-appear"}]})).unwrap();
            let response = error_response(body.clone(), Duration::ZERO).await;
            let status = observe_generation_http_failure(response, "decode", Instant::now()+Duration::from_secs(1)).await;
            assert_eq!(status.status, 400);
            let response = error_response(body, Duration::ZERO).await;
            let error = consume_prefill(response, 1, true, Instant::now()+Duration::from_secs(1)).await.unwrap_err();
            assert!(matches!(error, PdDispatchError::HttpStatus { role:"prefill", status:400, .. }));
            let response = error_response(br#"{"message":"unread cancelled body"}"#.to_vec(), Duration::from_millis(100)).await;
            tokio::select! {
                biased;
                _ = tokio::time::sleep(Duration::from_millis(10)) => {},
                _ = observe_generation_http_failure(response, "prefill", Instant::now()+Duration::from_secs(1)) => panic!("body unexpectedly ready"),
            }
        }.with_subscriber(subscriber).await;
        let logs = String::from_utf8(logs.lock().unwrap().clone()).unwrap();
        for field in [
            "decode",
            "prefill",
            "generation_post",
            "400",
            "complete_json_error_summary",
            "read_cancelled",
            "ordinary request context unavailable",
        ] {
            assert!(logs.contains(field), "missing diagnostic {field}: {logs}");
        }
        for forbidden in [
            "opaque-proof-must-not-appear",
            "opaque-secret-must-not-appear",
            "request-echo-must-not-appear",
        ] {
            assert!(!logs.contains(forbidden));
        }
        let safe = native_error_fields(br#"{"message":"prompt_cache_authority_proof echoed secret","type":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}"#).unwrap().to_string();
        assert!(!safe.contains("prompt_cache_authority_proof"));
        assert!(!safe.contains(&"a".repeat(64)));

        let response = error_response(
            b"not JSON: authority data must not be logged".to_vec(),
            Duration::ZERO,
        )
        .await;
        let failure =
            read_generation_http_failure(response, Instant::now() + Duration::from_secs(1)).await;
        assert_eq!(failure.body_state, "non_json_or_no_error_fields");
        assert_eq!(failure.error, Value::Null);
        let response = error_response(vec![b'x'; MAX_ERROR_BODY_BYTES + 1], Duration::ZERO).await;
        let failure =
            read_generation_http_failure(response, Instant::now() + Duration::from_secs(1)).await;
        assert_eq!(failure.body_state, "truncated_at_byte_limit");
        assert_eq!(failure.error, Value::Null);
        // Body is held beyond the *remaining original fence* but below 250ms.
        let response = error_response(
            br#"{"message":"late body"}"#.to_vec(),
            Duration::from_millis(100),
        )
        .await;
        let failure =
            read_generation_http_failure(response, Instant::now() + Duration::from_millis(10))
                .await;
        assert_eq!(failure.body_state, "read_deadline_elapsed");
        assert_eq!(failure.status, 400);
        assert_eq!(failure.error, Value::Null);
    }

    #[tokio::test]
    async fn native_pd_403_scalar_error_propagates_sanitized_rejection_cause() {
        let logs = Arc::new(Mutex::new(Vec::new()));
        let writer = LogWriter(logs.clone());
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .without_time()
            .with_writer(move || writer.clone())
            .finish();
        async {
            let response = native_response(
                br#"{"error":"cache authority signature/source/clock qualification failed"}"#.to_vec(),
                Duration::ZERO,
                403,
            )
            .await;
            let status = observe_generation_http_failure(
                response,
                "decode",
                Instant::now() + Duration::from_secs(1),
            )
            .await;
            assert_eq!(status.status, 403);
            assert!(status.cause().contains("cache authority signature/source/clock qualification failed"));
            // Exercise the production P response adapter and returned error
            // chain, not just the diagnostic observer.
            let response = native_response(
                br#"{"error":{"message":"native PD cannot replace the installed peer during this birth","detail":"proof secret must not escape"},"messages":["request echo"]}"#.to_vec(),
                Duration::ZERO,
                403,
            ).await;
            let error = consume_prefill(response, 1, true, Instant::now()+Duration::from_secs(1)).await.unwrap_err();
            assert!(matches!(error, PdDispatchError::HttpStatus { role:"prefill", status:403, .. }));
            let (status, payload) = crate::http::service::ErrorMessage::from_anyhow(error.into(), "Native worker request failed");
            assert_eq!(status.as_u16(), 500);
            let payload = serde_json::to_string(&payload.0).unwrap();
            assert!(payload.contains("cannot replace the installed peer during this birth"));
            assert!(payload.contains("generation_post"));
            assert!(!payload.contains("proof secret"));
            assert!(!payload.contains("request echo"));
        }
        .with_subscriber(subscriber)
        .await;
        let logs = String::from_utf8(logs.lock().unwrap().clone()).unwrap();
        for expected in [
            "decode",
            "generation_post",
            "403",
            "complete_json_error_summary",
            "cache authority signature/source/clock qualification failed",
        ] {
            assert!(logs.contains(expected), "missing diagnostic {expected}: {logs}");
        }
    }

    #[tokio::test]
    async fn native_pd_200_error_packet_is_failed_and_diagnosed_without_authority_echo() {
        let logs = Arc::new(Mutex::new(Vec::new()));
        let writer = LogWriter(logs.clone());
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .without_time()
            .with_writer(move || writer.clone())
            .finish();
        async {
            let packet = json!({"error":{"message":"native worker lifecycle missing", "type":"native_worker_error", "code":500,
                "prompt_cache_authority_proof":"opaque-proof-must-not-appear"}, "messages":"request-echo-must-not-appear"});
            let frame = format!("data: {packet}\n\n");
            let response = native_response(frame.into_bytes(), Duration::ZERO, 200).await;
            assert_eq!(response.status().as_u16(), 200);
            let error = consume_prefill(response, 1, true, Instant::now()+Duration::from_secs(1)).await.unwrap_err();
            assert!(matches!(error, PdDispatchError::Prefill(ref message) if message == "native error packet"));
            // D JSON/SSE call sites use this same parsed packet observer before
            // returning their Annotated error, never an output frame.
            observe_generation_error_packet(&packet, "decode", "generation_sse");
            observe_generation_error_packet(&packet, "decode", "generation_json");
        }.with_subscriber(subscriber).await;
        let logs = String::from_utf8(logs.lock().unwrap().clone()).unwrap();
        for field in [
            "prefill",
            "decode",
            "generation_sse",
            "generation_json",
            "native_worker_error",
            "native worker lifecycle missing",
        ] {
            assert!(logs.contains(field), "missing diagnostic {field}: {logs}");
        }
        assert!(!logs.contains("opaque-proof-must-not-appear"));
        assert!(!logs.contains("request-echo-must-not-appear"));
        assert!(!logs.contains("native_status=200")); // No invented status for parsed frames.
        assert!(
            native_error_fields_from_value(
                &json!({"error":{"message":"x".repeat(MAX_ERROR_BODY_BYTES+1)}})
            )
            .unwrap()
            .to_string()
            .contains("field exceeds diagnostic byte limit")
        );
    }
}
