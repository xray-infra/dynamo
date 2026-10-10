// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Synthetic CPU control receipts exercise the production signer/verifier.
//! The rank descriptor is a test fixture, never native/GPU qualification.
use super::control::{
    ControlRequest, ExpectedTransaction, PdRole, PhysicalRank, PreparedPdControl, RoomBinding,
};
use crate::cache_authority::cache_auth;
use serde_json::{Value, json};
use std::collections::BTreeSet;

const SECRET: &[u8; 32] = b"synthetic-pd-cpu-test-secret-000";
const NOW: u64 = 100_000;

// Explicit CPU tracking model, not native clock qualification or a production
// wall-time fallback. The unchanged parser supplies its actual +/-1ms bounds.
fn clock_model(now_ms: u64) -> crate::cache_authority::frontend_clock::ClockBounds {
    let instant = tokio::time::Instant::now();
    clock_model_at(now_ms, instant)
}
fn clock_model_at(
    now_ms: u64,
    instant: tokio::time::Instant,
) -> crate::cache_authority::frontend_clock::ClockBounds {
    crate::cache_authority::frontend_clock::qualified_cpu_tracking_sample(
        "Reference ID : C0000201\nSystem time : 0 seconds fast of NTP time\nRoot dispersion : 0 seconds\nRoot delay : 0 seconds\nLeap status : Normal\n",
        u128::from(now_ms)*1_000_000,u128::from(now_ms)*1_000_000,instant,instant).unwrap()
}

#[tokio::test(start_paused = true)]
async fn e28_pd_signing_stale_clock_and_wrong_prehashed_digest_cannot_mint_proof() {
    let (expected, mut claims, _) = fixture("pd-status");
    let prepared = PreparedPdControl::new(&expected).unwrap();
    claims["payload_digest"] = prepared.payload_digest().into();
    let clock = clock_model_at(
        NOW,
        tokio::time::Instant::now() - std::time::Duration::from_millis(101),
    );
    claims["issued_at_ms"] = clock.lower_ms().into();
    assert!(
        prepared
            .sign(
                SECRET,
                &super::transport::QualifiedClaims {
                    claims: claims.clone(),
                    clock
                }
            )
            .unwrap_err()
            .contains("stale")
    );
    let prepared = PreparedPdControl::new(&expected).unwrap();
    claims["payload_digest"] = "0".repeat(64).into();
    let clock = clock_model(NOW);
    claims["issued_at_ms"] = clock.lower_ms().into();
    assert!(
        prepared
            .sign(SECRET, &super::transport::QualifiedClaims { claims, clock })
            .is_err()
    );
}

#[tokio::test(start_paused = true)]
async fn e28_pd_receive_uncertainty_bounds_accept_late_terminal_and_reject_stale_clock() {
    let (expected, claims, mut payload) = fixture("pd-status");
    // The old signed admission deadline has expired, but this freshly captured
    // signed physical zero is still allowed to release its owned lifecycle.
    payload["captured_at_ms"] = (NOW + 2000).into();
    let prepared =
        super::transport::PreparedControlResponse::new(packet(&claims, &payload)).unwrap();
    let clock = clock_model(NOW + 2000);
    assert!(
        super::control::verify_response(SECRET, &claims, &expected, &prepared, &clock)
            .unwrap()
            .terminal
    );
    let stale = clock_model_at(
        NOW + 2000,
        tokio::time::Instant::now() - std::time::Duration::from_millis(101),
    );
    assert!(
        super::control::verify_response(SECRET, &claims, &expected, &prepared, &stale)
            .unwrap_err()
            .contains("stale")
    );
    for captured in [NOW + 100, NOW - 2500] {
        payload["captured_at_ms"] = captured.into();
        let prepared =
            super::transport::PreparedControlResponse::new(packet(&claims, &payload)).unwrap();
        let clock = clock_model(NOW);
        assert!(
            super::control::verify_response(SECRET, &claims, &expected, &prepared, &clock)
                .unwrap_err()
                .contains("clock")
        );
    }
}
fn sign_cpu_model(
    secret: &[u8],
    claims: &Value,
    expected: &ExpectedTransaction,
) -> Result<Value, String> {
    let prepared = PreparedPdControl::new(expected)?;
    let mut claims = claims.clone();
    claims["payload_digest"] = prepared.payload_digest().into();
    let clock = clock_model(NOW);
    claims["issued_at_ms"] = clock.lower_ms().into();
    prepared.sign(secret, &super::transport::QualifiedClaims { claims, clock })
}
fn verify_cpu_model(
    secret: &[u8],
    claims: &Value,
    expected: &ExpectedTransaction,
    packet: &Value,
    now_ms: u64,
) -> Result<super::control::ControlResponse, String> {
    let prepared = super::transport::PreparedControlResponse::new(packet.clone())?;
    let clock = clock_model(now_ms);
    super::control::verify_response(secret, claims, expected, &prepared, &clock)
}

fn fixture(operation: &str) -> (ExpectedTransaction, Value, Value) {
    let ranks: BTreeSet<_> = (0..8)
        .map(|rank| PhysicalRank {
            tp_rank: rank,
            dcp_rank: rank,
            replica_rank: 0,
        })
        .collect();
    let tuples: Vec<_> = ranks
        .iter()
        .map(|r| [r.tp_rank, r.dcp_rank, r.replica_rank])
        .collect();
    let digest = cache_auth::digest(&json!(tuples)).unwrap();
    let rooms = vec![
        RoomBinding {
            room: 710,
            sample_rid: "synthetic-root-0".into(),
        },
        RoomBinding {
            room: 711,
            sample_rid: "synthetic-root-1".into(),
        },
    ];
    let request = ControlRequest {
        schema_version: 1,
        transaction_id: "synthetic-cpu-transaction".into(),
        request_root: "synthetic-root".into(),
        expected_native_incarnation: "0123456789abcdef0123456789abcdef".into(),
        expected_qualified_rank_set_digest: digest.clone(),
        role: PdRole::Decode,
        rooms: rooms.clone(),
    };
    let expected = ExpectedTransaction { request, ranks };
    let claims = json!({
        "version":1,"operation":operation,"operation_id":"synthetic-cpu-operation","nonce":"synthetic-cpu-nonce",
        "model":"synthetic-cpu-model","native_origin":"http://127.0.0.1:45000",
        "native_incarnation":expected.request.expected_native_incarnation,
        "source_digest":"a".repeat(64),"runtime_profile_digest":"b".repeat(64),
        "reservation_id":"","authority_revision":0,
        "native_request_id":expected.request.request_root,"namespace_digest":"",
        "full_render_digest":"","token_ids_digest":"",
        "request_started_at_ms":NOW-1000,"issued_at_ms":NOW-500,"deadline_ms":NOW+1000,
        "retain_for_ms":0,"payload_digest":"f".repeat(64)
    });
    let rank_states: Vec<_> = expected.ranks.iter().map(|rank| {
        let states: Vec<_> = rooms.iter().map(|room| json!({
            "room":room.room,"sample_rid":room.sample_rid,"state":"released",
            "transfer_complete":true,"decode_committed":true,
            "live_kv_holds":0,"live_meta_holds":0,"live_io_holds":0,
            "abort_ack":operation == "pd-abort","transport_drained":true
        })).collect();
        json!({"tp_rank":rank.tp_rank,"dcp_rank":rank.dcp_rank,"replica_rank":rank.replica_rank,"rooms":states})
    }).collect();
    let payload = json!({
        "schema_version":1,"transaction_id":expected.request.transaction_id,"role":"decode",
        "native_incarnation":expected.request.expected_native_incarnation,
        "request_root":expected.request.request_root,"rooms":rooms,"captured_at_ms":NOW,
        "qualified_rank_set_digest":digest,"rank_states":rank_states,
        "terminal":true,"aborted":operation == "pd-abort"
    });
    (expected, claims, payload)
}

fn packet(claims: &Value, payload: &Value) -> Value {
    let event_digest = cache_auth::digest(payload).unwrap();
    let mut response_claims = claims.clone();
    for key in ["operation_id", "nonce"] {
        response_claims[key] =
            cache_auth::digest(&json!(["native-response", claims[key], event_digest]))
                .unwrap()
                .into();
    }
    let mut packet = payload.clone();
    packet[cache_auth::PROOF_FIELD] = cache_auth::sign(SECRET, &response_claims, payload).unwrap();
    packet
}
fn reject(expected: &ExpectedTransaction, claims: &Value, payload: &Value) -> String {
    verify_cpu_model(SECRET, claims, expected, &packet(claims, payload), NOW).unwrap_err()
}

#[test]
fn native_pd_control_signed_full_eight_rank_sorted_digest() {
    let (expected, claims, payload) = fixture("pd-status");
    let signed = sign_cpu_model(SECRET, &claims, &expected).unwrap();
    let mut unsigned = signed.clone();
    let proof = unsigned
        .as_object_mut()
        .unwrap()
        .remove(cache_auth::PROOF_FIELD)
        .unwrap();
    let authenticated = cache_auth::verify_signature(SECRET, &proof, &unsigned).unwrap();
    assert_eq!(authenticated["native_request_id"], "synthetic-root");
    assert_eq!(unsigned, serde_json::to_value(&expected.request).unwrap());
    let response =
        verify_cpu_model(SECRET, &claims, &expected, &packet(&claims, &payload), NOW).unwrap();
    assert_eq!(response.rank_states.len(), 8);
    assert!(response.terminal);
    assert!(!response.aborted);
    // Receipt ordering does not replace the qualified sorted descriptor digest.
    let mut reversed = payload.clone();
    reversed["rank_states"].as_array_mut().unwrap().reverse();
    assert!(verify_cpu_model(SECRET, &claims, &expected, &packet(&claims, &reversed), NOW).is_ok());
    let mut wrong_digest = expected;
    wrong_digest.request.expected_qualified_rank_set_digest = "0".repeat(64);
    assert!(
        sign_cpu_model(SECRET, &claims, &wrong_digest)
            .unwrap_err()
            .contains("descriptor digest mismatch")
    );
}

#[test]
fn native_pd_control_partial_rank_duplicate_room_and_sample_fail_closed() {
    let (expected, claims, payload) = fixture("pd-status");
    let mut partial = payload.clone();
    partial["rank_states"].as_array_mut().unwrap().pop();
    assert!(reject(&expected, &claims, &partial).contains("rank set incomplete"));
    let mut duplicate_rank = payload.clone();
    duplicate_rank["rank_states"][7] = duplicate_rank["rank_states"][0].clone();
    assert!(
        reject(&expected, &claims, &duplicate_rank).contains("duplicate native P/D physical rank")
    );
    for field in ["room", "sample_rid"] {
        let mut duplicate = payload.clone();
        duplicate["rank_states"][0]["rooms"][1][field] =
            duplicate["rank_states"][0]["rooms"][0][field].clone();
        assert!(
            reject(&expected, &claims, &duplicate).contains("duplicate or invalid"),
            "{field}"
        );
        let mut invalid_request = ExpectedTransaction {
            request: expected.request.clone(),
            ranks: expected.ranks.clone(),
        };
        if field == "room" {
            invalid_request.request.rooms[1].room = invalid_request.request.rooms[0].room;
        } else {
            invalid_request.request.rooms[1].sample_rid =
                invalid_request.request.rooms[0].sample_rid.clone();
        }
        assert!(
            sign_cpu_model(SECRET, &claims, &invalid_request).is_err(),
            "{field}"
        );
    }
}

#[test]
fn native_pd_control_tampered_echo_and_signature_fail_closed() {
    let (expected, claims, payload) = fixture("pd-status");
    for field in [
        "transaction_id",
        "request_root",
        "native_incarnation",
        "qualified_rank_set_digest",
    ] {
        let mut changed = payload.clone();
        changed[field] = "tampered-echo".into();
        assert!(
            reject(&expected, &claims, &changed).contains("identity mismatch"),
            "{field}"
        );
    }
    let mut unsigned_tamper = packet(&claims, &payload);
    unsigned_tamper["terminal"] = false.into();
    assert!(verify_cpu_model(SECRET, &claims, &expected, &unsigned_tamper, NOW).is_err());
    let mut wrong_claims = claims.clone();
    wrong_claims["native_request_id"] = "different-root".into();
    assert!(sign_cpu_model(SECRET, &wrong_claims, &expected).is_err());
    assert!(
        verify_cpu_model(
            SECRET,
            &claims,
            &expected,
            &packet(&wrong_claims, &payload),
            NOW
        )
        .unwrap_err()
        .contains("claim binding mismatch")
    );
}

#[test]
fn native_pd_control_terminal_physical_holds_and_abort_ack() {
    let (expected, claims, payload) = fixture("pd-abort");
    assert!(verify_cpu_model(SECRET, &claims, &expected, &packet(&claims, &payload), NOW).is_ok());
    for field in ["live_kv_holds", "live_meta_holds", "live_io_holds"] {
        let mut held = payload.clone();
        held["rank_states"][7]["rooms"][1][field] = 1.into();
        assert!(
            reject(&expected, &claims, &held).contains("still owns physical resources"),
            "{field}"
        );
    }
    let mut not_drained = payload.clone();
    not_drained["rank_states"][7]["rooms"][1]["transport_drained"] = false.into();
    assert!(reject(&expected, &claims, &not_drained).contains("still owns physical resources"));
    let mut missing_ack = payload.clone();
    missing_ack["rank_states"][7]["rooms"][1]["abort_ack"] = false.into();
    assert!(reject(&expected, &claims, &missing_ack).contains("abort lacks terminal physical ACK"));
    let mut nonterminal_abort = payload.clone();
    nonterminal_abort["terminal"] = false.into();
    assert!(
        reject(&expected, &claims, &nonterminal_abort)
            .contains("abort lacks terminal physical ACK")
    );
    // Natural termination does not fabricate an abort acknowledgement.
    let (expected, claims, natural) = fixture("pd-status");
    assert_eq!(natural["rank_states"][0]["rooms"][0]["abort_ack"], false);
    let response =
        verify_cpu_model(SECRET, &claims, &expected, &packet(&claims, &natural), NOW).unwrap();
    assert!(response.terminal && !response.aborted);
}

#[tokio::test(start_paused = true)]
async fn native_pd_control_counter_types_and_capture_time_boundaries() {
    let (expected, claims, payload) = fixture("pd-status");
    for field in ["live_kv_holds", "live_meta_holds", "live_io_holds"] {
        for malformed in [json!(true), json!(-1)] {
            let mut changed = payload.clone();
            changed["rank_states"][0]["rooms"][0][field] = malformed;
            assert!(
                verify_cpu_model(SECRET, &claims, &expected, &packet(&claims, &changed), NOW)
                    .is_err(),
                "{field}"
            );
        }
    }
    for (captured, accepted) in [
        (NOW - 2499, true),
        (NOW - 2500, false),
        (NOW + 99, true),
        (NOW + 100, false),
        (0, false),
    ] {
        let mut changed = payload.clone();
        changed["captured_at_ms"] = captured.into();
        assert_eq!(
            verify_cpu_model(SECRET, &claims, &expected, &packet(&claims, &changed), NOW).is_ok(),
            accepted,
            "captured={captured}"
        );
    }
}
