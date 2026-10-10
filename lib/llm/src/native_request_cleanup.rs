//! Ordinary request terminal evidence. Public finish, absent IDs and abort HTTP
//! 200 do not establish closed ingress or released native physical resources.
use std::collections::BTreeSet;

use crate::cache_authority::cache_auth;
use crate::cache_authority::cache_auth::PrehashedPayload;
use crate::cache_authority::frontend_clock::ClockBounds;
use crate::native_pd::transport::{PreparedControlResponse, QualifiedClaims, current_bounds};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Count (30s), longest native POST (10800s), and bounded physical cleanup (30s).
/// This is an internal admission fence lifetime, not a new public generation limit.
pub const ORDINARY_REQUEST_WINDOW_MS: u64 = 10_860_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestControl {
    pub schema_version: u32,
    pub transaction_id: String,
    pub request_root: String,
    pub expected_native_incarnation: String,
    pub expected_cleanup_cohort_digest: String,
    pub ordinary_body_digest: String,
    pub expected_choices: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParticipantState {
    pub participant_id: String,
    pub ingress_closed: bool,
    pub tokenizer_drained: bool,
    pub scheduler_drained: bool,
    pub live_request_holds: u64,
    pub live_kv_holds: u64,
    pub live_meta_holds: u64,
    pub live_io_holds: u64,
    pub transport_drained: bool,
    pub abort_ack: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestStatus {
    pub schema_version: u32,
    pub transaction_id: String,
    pub request_root: String,
    pub native_incarnation: String,
    pub cleanup_cohort_digest: String,
    pub ordinary_body_digest: String,
    pub expected_choices: u32,
    pub captured_at_ms: u64,
    pub participants: Vec<ParticipantState>,
    pub terminal: bool,
    pub aborted: bool,
}

/// The complete runtime descriptor supplies this set, including actual auxiliary
/// roles. It is never manufactured from cache TP clocks or core DP metadata.
pub struct ExpectedRequest {
    pub control: RequestControl,
    pub participants: BTreeSet<String>,
}

impl ExpectedRequest {
    pub fn cleanup_id(&self) -> String {
        format!(
            "request:{}:{}:{}",
            self.control.transaction_id,
            self.control.request_root,
            self.control.expected_native_incarnation
        )
    }

    pub fn validate(&self) -> Result<(), String> {
        let c = &self.control;
        if c.schema_version != 1
            || !hex(&c.transaction_id, 32)
            || !hex(&c.request_root, 32)
            || !bounded_identity(&c.expected_native_incarnation)
            || !hex(&c.expected_cleanup_cohort_digest, 64)
            || !hex(&c.ordinary_body_digest, 64)
            || c.expected_choices == 0
            || self.participants.is_empty()
            || self.participants.iter().any(|p| !bounded_identity(p))
        {
            return Err("ordinary native lifecycle qualification incomplete".into());
        }
        if cache_auth::digest(
            &serde_json::to_value(&self.participants).map_err(|e| e.to_string())?,
        )? != c.expected_cleanup_cohort_digest
        {
            return Err("ordinary native cleanup cohort digest mismatch".into());
        }
        Ok(())
    }
}

fn bounded_identity(value: &str) -> bool {
    !value.is_empty() && value.len() <= 1024
}
fn hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn claim_identity(claims: &Value, expected: &ExpectedRequest) -> Result<(), String> {
    claim_control_identity(claims, &expected.control)
}
fn claim_control_identity(claims: &Value, control: &RequestControl) -> Result<(), String> {
    if claims["native_request_id"] != control.request_root
        || claims["native_incarnation"] != control.expected_native_incarnation
        || claims["authority_revision"] != 0
        || claims["retain_for_ms"] != 0
        || claims["reservation_id"] != ""
        || ["namespace_digest", "full_render_digest", "token_ids_digest"]
            .iter()
            .any(|k| claims[*k] != "")
    {
        return Err("ordinary native lifecycle claim identity mismatch".into());
    }
    let start = claims["request_started_at_ms"]
        .as_u64()
        .ok_or("ordinary request start missing")?;
    let deadline = claims["deadline_ms"]
        .as_u64()
        .ok_or("ordinary request deadline missing")?;
    if deadline < start || deadline - start > ORDINARY_REQUEST_WINDOW_MS {
        return Err("ordinary request window exceeds its admission fence".into());
    }
    Ok(())
}

/// Attach only after public field rejection, normalization, owned RID and Count.
/// The exact JSON string protects floats without inventing rendered-token hashes.
pub struct PreparedGeneration {
    body: Value,
    payload: PrehashedPayload,
    control: RequestControl,
}
impl PreparedGeneration {
    pub fn new(expected: &ExpectedRequest, mut body: Value) -> Result<Self, String> {
        expected.validate()?;
        if body.get("prompt_cache_request_control").is_some()
            || body.get(cache_auth::PROOF_FIELD).is_some()
            || body.get("prompt_cache_canonical_body").is_some()
        {
            return Err("ordinary lifecycle metadata already supplied".into());
        }
        let root = body.get("rid").or_else(|| body.get("request_id"));
        if root != Some(&Value::String(expected.control.request_root.clone()))
            || (body.get("rid").is_some() && body.get("request_id").is_some())
            || body.get("n").map_or(Some(1), Value::as_u64)
                != Some(u64::from(expected.control.expected_choices))
        {
            return Err("ordinary lifecycle root/choice mismatch".into());
        }
        cache_auth::attach_canonical_body(&mut body)?;
        if cache_auth::digest(&body["prompt_cache_canonical_body"])?
            != expected.control.ordinary_body_digest
        {
            return Err("ordinary lifecycle canonical body mismatch".into());
        }
        body["prompt_cache_request_control"] =
            serde_json::to_value(&expected.control).map_err(|e| e.to_string())?;
        let payload = cache_auth::payload_for("request-generate", &body)?;
        Ok(Self {
            body,
            payload: PrehashedPayload::new(&payload)?,
            control: expected.control.clone(),
        })
    }
    pub fn payload_digest(&self) -> &str {
        self.payload.digest()
    }
    pub fn sign(mut self, secret: &[u8], qualified: &QualifiedClaims) -> Result<Value, String> {
        claim_control_identity(&qualified.claims, &self.control)?;
        if qualified.claims["operation"] != "request-generate" {
            return Err("ordinary generation operation mismatch".into());
        }
        qualified.ensure_fresh()?;
        self.body[cache_auth::PROOF_FIELD] =
            cache_auth::sign_prehashed(secret, &qualified.claims, &self.payload)?;
        Ok(self.body)
    }
}

pub struct PreparedRequestControl {
    body: Value,
    payload: PrehashedPayload,
    control: RequestControl,
}
impl PreparedRequestControl {
    pub fn new(expected: &ExpectedRequest) -> Result<Self, String> {
        expected.validate()?;
        let body = serde_json::to_value(&expected.control).map_err(|e| e.to_string())?;
        crate::native_pd::transport::encode_bounded(&body).map_err(|e| e.to_string())?;
        Ok(Self {
            payload: PrehashedPayload::new(&body)?,
            body,
            control: expected.control.clone(),
        })
    }
    pub fn payload_digest(&self) -> &str {
        self.payload.digest()
    }
    pub fn sign(mut self, secret: &[u8], qualified: &QualifiedClaims) -> Result<Value, String> {
        claim_control_identity(&qualified.claims, &self.control)?;
        if !matches!(
            qualified.claims["operation"].as_str(),
            Some("request-status" | "request-abort")
        ) {
            return Err("ordinary native control operation mismatch".into());
        }
        qualified.ensure_fresh()?;
        self.body[cache_auth::PROOF_FIELD] =
            cache_auth::sign_prehashed(secret, &qualified.claims, &self.payload)?;
        Ok(self.body)
    }
}

/// Opaque physical terminal proof; only exact signed participant verification
/// constructs it. A parsed nonterminal status is deliberately not this type.
pub struct VerifiedNativeTerminal {
    status: RequestStatus,
    cleanup_id: String,
}
impl VerifiedNativeTerminal {
    pub fn cleanup_id(&self) -> &str {
        &self.cleanup_id
    }
    pub fn status(&self) -> &RequestStatus {
        &self.status
    }
}

pub fn verify_status(
    secret: &[u8],
    request_claims: &Value,
    expected: &ExpectedRequest,
    packet: &PreparedControlResponse,
    clock: &ClockBounds,
) -> Result<RequestStatus, String> {
    expected.validate()?;
    claim_identity(request_claims, expected)?;
    if !matches!(
        request_claims["operation"].as_str(),
        Some("request-status" | "request-abort")
    ) {
        return Err("ordinary native status operation mismatch".into());
    }
    packet.verify(secret, request_claims, clock)?;
    let status: RequestStatus =
        serde_json::from_value(packet.payload().clone()).map_err(|e| e.to_string())?;
    let (lower_ms, upper_ms) = current_bounds(clock)?;
    let c = &expected.control;
    if status.schema_version != c.schema_version
        || status.transaction_id != c.transaction_id
        || status.request_root != c.request_root
        || status.native_incarnation != c.expected_native_incarnation
        || status.cleanup_cohort_digest != c.expected_cleanup_cohort_digest
        || status.ordinary_body_digest != c.ordinary_body_digest
        || status.expected_choices != c.expected_choices
    {
        return Err("ordinary native status identity mismatch".into());
    }
    if status.captured_at_ms == 0
        || status.captured_at_ms > lower_ms.saturating_add(100)
        || upper_ms.saturating_sub(status.captured_at_ms) > 2500
    {
        return Err("ordinary native status capture stale or clock invalid".into());
    }
    let mut participants = BTreeSet::new();
    for p in &status.participants {
        if !bounded_identity(&p.participant_id) || !participants.insert(p.participant_id.clone()) {
            return Err("ordinary native cleanup participant duplicated or invalid".into());
        }
        if status.terminal
            && (!p.ingress_closed
                || !p.tokenizer_drained
                || !p.scheduler_drained
                || !p.transport_drained
                || p.live_request_holds != 0
                || p.live_kv_holds != 0
                || p.live_meta_holds != 0
                || p.live_io_holds != 0)
        {
            return Err("ordinary native terminal still owns request resources".into());
        }
        if status.aborted && (!status.terminal || !p.abort_ack) {
            return Err("ordinary native abort lacks processed physical acknowledgement".into());
        }
    }
    if participants != expected.participants {
        return Err("ordinary native cleanup cohort incomplete".into());
    }
    clock.ensure_fresh().map_err(|e| e.to_string())?;
    Ok(status)
}

pub fn verify_terminal(
    secret: &[u8],
    claims: &Value,
    expected: &ExpectedRequest,
    packet: &PreparedControlResponse,
    clock: &ClockBounds,
) -> Result<VerifiedNativeTerminal, String> {
    let status = verify_status(secret, claims, expected, packet, clock)?;
    if !status.terminal {
        return Err("ordinary native request is not physically terminal".into());
    }
    Ok(VerifiedNativeTerminal {
        status,
        cleanup_id: expected.cleanup_id(),
    })
}

#[cfg(test)]
mod clock_tests {
    use super::*;
    use crate::cache_authority::frontend_clock::qualified_cpu_tracking_sample;
    use serde_json::json;
    use tokio::time::{Duration, Instant};
    const NOW: u64 = 1_000_000;
    const SECRET: [u8; 32] = [42; 32];
    fn clock(at: Instant) -> ClockBounds {
        // Explicit CPU source/time model through the real parser, never a
        // constructor for unchecked production wall-clock bounds.
        qualified_cpu_tracking_sample("Reference ID : C0000201\nSystem time : 0 seconds fast of NTP time\nRoot dispersion : 0 seconds\nRoot delay : 0 seconds\nLeap status : Normal\n",u128::from(NOW)*1_000_000,u128::from(NOW)*1_000_000,at,at).unwrap()
    }
    fn fixture() -> (ExpectedRequest, Value, Value) {
        let participants: BTreeSet<String> = ["cpu-ordinary-owner".into()].into();
        let body = json!({"model":"CPU fixture only","rid":"3".repeat(32),"n":1,"messages":[{"role":"user","content":"CPU-only ordinary request"}]});
        let canonical =
            serde_json::to_string(&cache_auth::ordinary_body_for(&body).unwrap()).unwrap();
        let expected = ExpectedRequest {
            control: RequestControl {
                schema_version: 1,
                transaction_id: "2".repeat(32),
                request_root: "3".repeat(32),
                expected_native_incarnation: "cpu-fixture-native-birth".into(),
                expected_cleanup_cohort_digest: cache_auth::digest(&json!(participants)).unwrap(),
                ordinary_body_digest: cache_auth::digest(&Value::String(canonical)).unwrap(),
                expected_choices: 1,
            },
            participants,
        };
        let claims = json!({"version":1,"operation":"request-generate","operation_id":"cpu-clock-operation","nonce":"cpu-clock-nonce","model":"CPU fixture only","native_origin":"http://127.0.0.1:1","native_incarnation":expected.control.expected_native_incarnation,"source_digest":"a".repeat(64),"runtime_profile_digest":"b".repeat(64),"reservation_id":"","authority_revision":0,"native_request_id":expected.control.request_root,"namespace_digest":"","full_render_digest":"","token_ids_digest":"","request_started_at_ms":NOW-100,"issued_at_ms":NOW-1,"deadline_ms":NOW+1000,"retain_for_ms":0,"payload_digest":"0".repeat(64)});
        (expected, claims, body)
    }
    fn signed_receipt(
        expected: &ExpectedRequest,
        claims: &Value,
        captured: u64,
    ) -> PreparedControlResponse {
        let mut payload = json!({"schema_version":1,"transaction_id":expected.control.transaction_id,"request_root":expected.control.request_root,"native_incarnation":expected.control.expected_native_incarnation,"cleanup_cohort_digest":expected.control.expected_cleanup_cohort_digest,"ordinary_body_digest":expected.control.ordinary_body_digest,"expected_choices":1,"captured_at_ms":captured,"participants":[{"participant_id":"cpu-ordinary-owner","ingress_closed":true,"tokenizer_drained":true,"scheduler_drained":true,"live_request_holds":0,"live_kv_holds":0,"live_meta_holds":0,"live_io_holds":0,"transport_drained":true,"abort_ack":false}],"terminal":true,"aborted":false});
        let event = cache_auth::digest(&payload).unwrap();
        let mut response_claims = claims.clone();
        for field in ["operation_id", "nonce"] {
            response_claims[field] =
                cache_auth::digest(&json!(["native-response", claims[field], event]))
                    .unwrap()
                    .into();
        }
        payload[cache_auth::PROOF_FIELD] =
            cache_auth::sign(&SECRET, &response_claims, &payload).unwrap();
        PreparedControlResponse::new(payload).unwrap()
    }
    #[tokio::test(start_paused = true)]
    async fn e28_ordinary_prepared_generation_checks_fresh_clock_and_exact_body_digest() {
        let (expected, mut claims, body) = fixture();
        let prepared = PreparedGeneration::new(&expected, body.clone()).unwrap();
        claims["payload_digest"] = prepared.payload_digest().into();
        let stale = clock(Instant::now() - Duration::from_millis(101));
        assert!(
            prepared
                .sign(
                    &SECRET,
                    &QualifiedClaims {
                        claims: claims.clone(),
                        clock: stale
                    }
                )
                .unwrap_err()
                .contains("stale")
        );
        let prepared = PreparedGeneration::new(&expected, body.clone()).unwrap();
        let qualified = QualifiedClaims {
            claims: claims.clone(),
            clock: clock(Instant::now()),
        };
        let signed = prepared.sign(&SECRET, &qualified).unwrap();
        let payload = cache_auth::payload_for("request-generate", &signed).unwrap();
        assert!(
            cache_auth::verify_signature(&SECRET, &signed[cache_auth::PROOF_FIELD], &payload)
                .is_ok()
        );
        let mut changed = body;
        changed["messages"][0]["content"] = "different request".into();
        assert!(PreparedGeneration::new(&expected, changed).is_err());
        let prepared =
            PreparedGeneration::new(&expected, cache_auth::ordinary_body_for(&signed).unwrap())
                .unwrap();
        claims["payload_digest"] = "0".repeat(64).into();
        assert!(
            prepared
                .sign(
                    &SECRET,
                    &QualifiedClaims {
                        claims,
                        clock: clock(Instant::now())
                    }
                )
                .is_err()
        );
    }
    #[tokio::test(start_paused = true)]
    async fn e28_ordinary_receive_fresh_bounds_keep_late_terminal_valid_and_close_uncertain_edges()
    {
        let (expected, mut claims, _) = fixture();
        claims["operation"] = "request-status".into();
        claims["deadline_ms"] = (NOW - 1).into(); // original valid admission window
        claims["issued_at_ms"] = (NOW - 2).into();
        let prepared = signed_receipt(&expected, &claims, NOW);
        let fresh = clock(Instant::now());
        assert!(
            verify_terminal(&SECRET, &claims, &expected, &prepared, &fresh)
                .unwrap()
                .status()
                .terminal
        );
        let stale = clock(Instant::now() - Duration::from_millis(101));
        assert!(verify_terminal(&SECRET, &claims, &expected, &prepared, &stale).is_err());
        for captured in [NOW + 100, NOW - 2500] {
            let prepared = signed_receipt(&expected, &claims, captured);
            let fresh = clock(Instant::now());
            assert!(verify_terminal(&SECRET, &claims, &expected, &prepared, &fresh).is_err());
        }
    }
}
