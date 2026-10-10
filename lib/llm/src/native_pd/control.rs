//! Signed, physical native-role receipts. An HTTP 200 abort is not a drain ACK.
use std::collections::BTreeSet;

use super::transport::{PreparedControlResponse, QualifiedClaims, current_bounds};
use crate::cache_authority::cache_auth;
use crate::cache_authority::cache_auth::PrehashedPayload;
use crate::cache_authority::frontend_clock::ClockBounds;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PdRole {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoomBinding {
    pub room: i64,
    pub sample_rid: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicalRank {
    pub tp_rank: u32,
    pub dcp_rank: u32,
    pub replica_rank: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlRequest {
    pub schema_version: u32,
    pub transaction_id: String,
    pub request_root: String,
    pub expected_native_incarnation: String,
    pub expected_qualified_rank_set_digest: String,
    pub role: PdRole,
    pub rooms: Vec<RoomBinding>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoomState {
    pub room: i64,
    pub sample_rid: String,
    pub state: String,
    pub transfer_complete: bool,
    pub decode_committed: bool,
    pub live_kv_holds: u64,
    pub live_meta_holds: u64,
    pub live_io_holds: u64,
    pub abort_ack: bool,
    pub transport_drained: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RankState {
    pub tp_rank: u32,
    pub dcp_rank: u32,
    pub replica_rank: u32,
    pub rooms: Vec<RoomState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlResponse {
    pub schema_version: u32,
    pub transaction_id: String,
    pub role: PdRole,
    pub native_incarnation: String,
    pub request_root: String,
    pub rooms: Vec<RoomBinding>,
    pub captured_at_ms: u64,
    pub qualified_rank_set_digest: String,
    pub rank_states: Vec<RankState>,
    pub terminal: bool,
    pub aborted: bool,
}

/// The expected set is copied from the qualified native physical descriptor,
/// never manufactured from core DP size or a GPU count.
pub struct ExpectedTransaction {
    pub request: ControlRequest,
    pub ranks: BTreeSet<PhysicalRank>,
}

impl ExpectedTransaction {
    fn validate(&self) -> Result<BTreeSet<RoomBinding>, String> {
        let r = &self.request;
        if r.schema_version != 1
            || r.transaction_id.is_empty()
            || r.request_root.is_empty()
            || !is_hex(&r.expected_native_incarnation, 32)
            || !is_hex(&r.expected_qualified_rank_set_digest, 64)
            || self.ranks.is_empty()
        {
            return Err("unqualified native P/D transaction identity".into());
        }
        let mut rank_tuples = Vec::with_capacity(self.ranks.len());
        for rank in &self.ranks {
            rank_tuples.push([rank.tp_rank, rank.dcp_rank, rank.replica_rank]);
        }
        if cache_auth::digest(&serde_json::to_value(rank_tuples).map_err(|e| e.to_string())?)?
            != r.expected_qualified_rank_set_digest
        {
            return Err("native P/D qualified rank descriptor digest mismatch".into());
        }
        exact_rooms(&r.rooms)
    }
}

fn is_hex(s: &str, length: usize) -> bool {
    s.len() == length
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn exact_rooms(rooms: &[RoomBinding]) -> Result<BTreeSet<RoomBinding>, String> {
    let mut ids = BTreeSet::new();
    let mut rids = BTreeSet::new();
    let mut result = BTreeSet::new();
    for room in rooms {
        if room.room <= 0
            || room.sample_rid.is_empty()
            || room.sample_rid.len() > 1024
            || !ids.insert(room.room)
            || !rids.insert(room.sample_rid.clone())
        {
            return Err("duplicate or invalid native P/D room mapping".into());
        }
        result.insert(room.clone());
    }
    if result.is_empty() {
        return Err("native P/D room mapping missing".into());
    }
    Ok(result)
}

pub struct PreparedPdControl {
    payload: Value,
    digest: PrehashedPayload,
    request: ControlRequest,
}
impl PreparedPdControl {
    pub fn new(expected: &ExpectedTransaction) -> Result<Self, String> {
        expected.validate()?;
        let payload = serde_json::to_value(&expected.request).map_err(|e| e.to_string())?;
        super::transport::encode_bounded(&payload).map_err(|e| e.to_string())?;
        Ok(Self {
            digest: PrehashedPayload::new(&payload)?,
            payload,
            request: expected.request.clone(),
        })
    }
    pub fn payload_digest(&self) -> &str {
        self.digest.digest()
    }
    pub fn sign(mut self, secret: &[u8], qualified: &QualifiedClaims) -> Result<Value, String> {
        check_claim_identity(&qualified.claims, &self.request)?;
        qualified.ensure_fresh()?;
        self.payload[cache_auth::PROOF_FIELD] =
            cache_auth::sign_prehashed(secret, &qualified.claims, &self.digest)?;
        Ok(self.payload)
    }
}

fn check_claim_identity(claims: &Value, request: &ControlRequest) -> Result<(), String> {
    if !matches!(claims["operation"].as_str(), Some("pd-status" | "pd-abort"))
        || claims["native_request_id"] != request.request_root
        || claims["reservation_id"] != ""
        || claims["authority_revision"] != 0
        || claims["native_incarnation"] != request.expected_native_incarnation
    {
        return Err("native P/D signed claim identity mismatch".into());
    }
    Ok(())
}

pub fn verify_response(
    secret: &[u8],
    request_claims: &Value,
    expected: &ExpectedTransaction,
    packet: &PreparedControlResponse,
    clock: &ClockBounds,
) -> Result<ControlResponse, String> {
    let expected_rooms = expected.validate()?;
    check_claim_identity(request_claims, &expected.request)?;
    packet.verify(secret, request_claims, clock)?;
    let response: ControlResponse =
        serde_json::from_value(packet.payload().clone()).map_err(|e| e.to_string())?;
    let (lower_ms, upper_ms) = current_bounds(clock)?;
    let r = &expected.request;
    if response.schema_version != r.schema_version
        || response.transaction_id != r.transaction_id
        || response.role != r.role
        || response.native_incarnation != r.expected_native_incarnation
        || response.request_root != r.request_root
        || response.qualified_rank_set_digest != r.expected_qualified_rank_set_digest
        || exact_rooms(&response.rooms)? != expected_rooms
    {
        return Err("native P/D response transaction/rank/room identity mismatch".into());
    }
    // Same bounded inter-host skew as the native scheduler load contract. Never
    // restamp an old capture with the HTTP receive time or invent a heartbeat.
    if response.captured_at_ms == 0
        || response.captured_at_ms > lower_ms.saturating_add(100)
        || upper_ms.saturating_sub(response.captured_at_ms) > 2500
    {
        return Err("native P/D physical receipt stale or clock invalid".into());
    }
    let mut ranks = BTreeSet::new();
    for rank in &response.rank_states {
        if !ranks.insert(PhysicalRank {
            tp_rank: rank.tp_rank,
            dcp_rank: rank.dcp_rank,
            replica_rank: rank.replica_rank,
        }) {
            return Err("duplicate native P/D physical rank".into());
        }
        let bindings: Vec<_> = rank
            .rooms
            .iter()
            .map(|room| RoomBinding {
                room: room.room,
                sample_rid: room.sample_rid.clone(),
            })
            .collect();
        if exact_rooms(&bindings)? != expected_rooms {
            return Err("native P/D physical rank room mapping incomplete".into());
        }
        for room in &rank.rooms {
            if room.state.is_empty() || room.state.len() > 128 {
                return Err("native P/D physical state missing".into());
            }
            if response.terminal
                && (room.live_kv_holds != 0
                    || room.live_meta_holds != 0
                    || room.live_io_holds != 0
                    || !room.transport_drained)
            {
                return Err("native P/D terminal still owns physical resources".into());
            }
            if response.aborted && (!response.terminal || !room.abort_ack) {
                return Err("native P/D abort lacks terminal physical ACK".into());
            }
        }
    }
    if ranks != expected.ranks {
        return Err("native P/D physical rank set incomplete".into());
    }
    clock.ensure_fresh().map_err(|e| e.to_string())?;
    Ok(response)
}
