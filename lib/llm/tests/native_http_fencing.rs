// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Changed-risk CPU tests with explicit mock control-plane stores. No etcd or
//! native worker transport qualification is claimed by these snapshot fixtures.

use std::collections::BTreeSet;
use std::sync::Arc;

use async_trait::async_trait;
use dynamo_llm::native_fencing::prefix_records::{
    PrefixError, PrefixRow, PrefixSnapshot, PrefixStore, PrefixStream,
};
use dynamo_llm::native_fencing::{
    ExpectedRequestIds, FenceConfig, FenceError, FenceStore, GateState, GenerationPermit,
    LeaseGrant, NativeFence, NativeRequestPhase, WorkerLoad, affinity_for_chat,
};
use tokio::time::Instant;

struct MockControlPlane;
#[async_trait]
impl FenceStore for MockControlPlane {
    async fn acquire(&self, _: &str, _: &str) -> Result<Option<LeaseGrant>, FenceError> {
        Ok(Some(LeaseGrant { id: 1, epoch: 1 }))
    }
    async fn keep_alive(&self, _: LeaseGrant) -> Result<(), FenceError> {
        Ok(())
    }
    async fn read_short_gate(&self) -> Result<GateState, FenceError> {
        Ok(GateState {
            enabled: true,
            commit_id: "explicit-cpu-mock-20261001".into(),
            revision: 2,
            mod_revision: 1,
        })
    }
}
#[async_trait]
impl PrefixStore for MockControlPlane {
    async fn snapshot(&self) -> Result<PrefixSnapshot, PrefixError> {
        Ok(PrefixSnapshot {
            revision: 1,
            rows: vec![],
        })
    }
    async fn watch_from(&self, _: i64) -> Result<PrefixStream, PrefixError> {
        Ok(Box::pin(futures::stream::pending()))
    }
    async fn put(&self, _: PrefixRow, _: u64) -> Result<(), PrefixError> {
        Ok(())
    }
}
fn samples(worker: &str, ids: &[&str], incarnation: char) -> Vec<WorkerLoad> {
    (0..12)
        .map(|index| {
            let name = format!("native-{index}");
            let selected = name == worker;
            WorkerLoad {
                worker: name,
                healthy: true,
                running: if selected { ids.len() } else { 0 },
                waiting: 0,
                active_request_ids: Some(if selected {
                    ids.iter().map(|id| (*id).to_owned()).collect()
                } else {
                    vec![]
                }),
                native_incarnation: if selected {
                    incarnation.to_string().repeat(64)
                } else {
                    "1".repeat(64)
                },
                sampled_at: Instant::now(),
            }
        })
        .collect()
}
async fn setup(n: u32) -> (Arc<NativeFence>, GenerationPermit) {
    let store = Arc::new(MockControlPlane);
    let fence = NativeFence::new(
        FenceConfig {
            election_key: "/cpu-mock/exact-load".into(),
            holder: "cpu-only".into(),
            short_workers: (0..12).map(|index| format!("native-{index}")).collect(),
            max_effective_load: 2,
        },
        store.clone(),
        store,
    )
    .unwrap();
    fence.refresh_authority().await.unwrap();
    fence.update_samples(samples("native-0", &[], '1')).unwrap();
    fence.start_prefixes();
    for _ in 0..100 {
        if fence.prefixes_ready() {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert!(fence.prefixes_ready());
    let affinity=affinity_for_chat(&serde_json::json!({"model":"cpu-fixture","messages":[{"role":"user","content":"exact occupancy"}],"cache_salt":"cpu-only"}),"explicit-cpu-model-native-template-identity").unwrap();
    let permit = fence
        .take_short_permit(
            "R",
            &affinity,
            ExpectedRequestIds::new("R", n).unwrap(),
            100,
            &BTreeSet::new(),
        )
        .await
        .unwrap();
    (fence, permit)
}

#[tokio::test(start_paused = true)]
async fn exact_load_prefill_gap_and_finished_choices_never_become_ghost_pending() {
    let (fence, permit) = setup(3).await;
    let worker = &permit.worker;
    for (ids, expected) in [
        (vec!["R_0:prefill"], 3),
        (vec![], 3),
        (vec!["R_0:sample:0", "R_0:sample:1"], 3),
        (vec!["R_0:sample:1"], 2),
        (vec![], 1),
        (vec!["R_0:sample:2"], 1),
        (vec![], 0),
    ] {
        fence.update_samples(samples(worker, &ids, '1')).unwrap();
        assert_eq!(
            fence.effective_load(worker),
            Some(expected),
            "snapshot {ids:?}"
        );
    }
    assert_eq!(
        fence.reservation_count(),
        1,
        "live response guard still exists but accepted finished work contributes no pending load"
    );
    drop(permit);
    assert_eq!(fence.reservation_count(), 0);
}

#[tokio::test(start_paused = true)]
async fn exact_load_scalar_acceptance_persists_without_hiding_foreign_work() {
    let (fence, permit) = setup(1).await;
    let worker = &permit.worker;
    fence.update_samples(samples(worker, &["R"], '1')).unwrap();
    assert_eq!(fence.effective_load(worker), Some(1));
    fence.update_samples(samples(worker, &[], '1')).unwrap();
    assert_eq!(fence.effective_load(worker), Some(0));
    fence
        .update_samples(samples(worker, &["foreign"], '1'))
        .unwrap();
    assert_eq!(fence.effective_load(worker), Some(1));
}

#[tokio::test(start_paused = true)]
async fn exact_load_restart_cannot_inherit_accepted_choices_or_match_old_reservation() {
    let (fence, permit) = setup(3).await;
    let worker = &permit.worker;
    fence
        .update_samples(samples(worker, &["R_0:sample:0", "R_0:sample:1"], '1'))
        .unwrap();
    fence.update_samples(samples(worker, &[], '1')).unwrap();
    assert_eq!(fence.effective_load(worker), Some(1));
    fence
        .update_samples(samples(worker, &["R_0:sample:0", "R_0:sample:1"], '2'))
        .unwrap();
    assert_eq!(
        fence.effective_load(worker),
        Some(5),
        "new native work must not absorb an old incarnation's reservation"
    );
    fence.update_samples(samples(worker, &[], '2')).unwrap();
    assert_eq!(fence.effective_load(worker), Some(3));
    fence.update_samples(samples(worker, &[], '1')).unwrap();
    assert_eq!(
        fence.effective_load(worker),
        Some(3),
        "cleared acceptance cannot revive across an observed restart"
    );
    drop(permit);
    assert_eq!(fence.effective_load("native-0"), Some(0));
}

#[tokio::test(start_paused = true)]
async fn exact_load_phase_grammar_is_compact_and_prefill_never_double_absorbs_choice_zero() {
    let scalar = ExpectedRequestIds::new("R", 1).unwrap();
    assert_eq!(scalar.phase("R"), Some(NativeRequestPhase::Sample(0)));
    assert_eq!(scalar.phase("R_0:prefill"), None);
    let parallel = ExpectedRequestIds::new("R", u32::MAX).unwrap();
    assert_eq!(
        parallel.phase("R_0:prefill"),
        Some(NativeRequestPhase::Prefill)
    );
    assert_eq!(
        parallel.phase("R_0:sample:4294967294"),
        Some(NativeRequestPhase::Sample(u32::MAX - 1))
    );
    for bad in [
        "R_0:sample:00",
        "R_0:sample:+1",
        "R_0:sample:4294967295",
        "R_0:sample:1:extra",
        "RR_0:sample:0",
    ] {
        assert_eq!(parallel.phase(bad), None);
    }
    let (fence, permit) = setup(3).await;
    let worker = &permit.worker;
    fence
        .update_samples(samples(worker, &["R_0:sample:0", "R_0:prefill"], '1'))
        .unwrap();
    assert_eq!(
        fence.effective_load(worker),
        Some(4),
        "two actual native rows plus two unaccepted choices; prefill may not permanently or doubly accept zero"
    );
    fence.update_samples(samples(worker, &[], '1')).unwrap();
    assert_eq!(fence.effective_load(worker), Some(2));
}
