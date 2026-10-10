// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CPU policy tests. MockStore does not constitute an etcd transport or native
//! worker qualification. The separately ignored test uses a real local etcd.
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use dynamo_llm::native_fencing::prefix_records::{
    EtcdPrefixRecords, PrefixError, PrefixRow, PrefixSnapshot, PrefixStore, PrefixStream,
};
use dynamo_llm::native_fencing::{
    EtcdFenceStore, ExpectedRequestIds, FenceConfig, FenceError, FenceStore, GateState,
    GenerationPermit, LeaseGrant, NativeAffinity, NativeFence, WorkerLoad, affinity_for_chat,
};
use tokio::time::Instant;

struct ServerLease {
    grant: LeaseGrant,
    expires: Instant,
}
struct MockState {
    authority: Option<ServerLease>,
    epoch: u64,
    available: bool,
    gate: GateState,
    gate_delay: Duration,
    prefix_rows: Vec<PrefixRow>,
}
struct MockStore(Mutex<MockState>);
impl MockStore {
    fn new() -> Arc<Self> {
        Arc::new(Self(Mutex::new(MockState {
            authority: None,
            epoch: 0,
            available: true,
            gate: GateState {
                enabled: true,
                commit_id: "cpu-mock-commit-001".into(),
                revision: 2,
                mod_revision: 1,
            },
            gate_delay: Duration::ZERO,
            prefix_rows: vec![],
        })))
    }
}
#[async_trait]
impl PrefixStore for MockStore {
    async fn snapshot(&self) -> Result<PrefixSnapshot, PrefixError> {
        let state = self.0.lock().unwrap();
        if !state.available {
            return Err(PrefixError("explicit CPU mock unavailable".into()));
        }
        Ok(PrefixSnapshot {
            revision: 1,
            rows: state.prefix_rows.clone(),
        })
    }
    async fn watch_from(&self, _revision: i64) -> Result<PrefixStream, PrefixError> {
        Ok(Box::pin(futures::stream::pending()))
    }
    async fn put(&self, row: PrefixRow, _ttl_ms: u64) -> Result<(), PrefixError> {
        let mut state = self.0.lock().unwrap();
        state.prefix_rows.retain(|old| old.key != row.key);
        state.prefix_rows.push(row);
        Ok(())
    }
}
#[async_trait]
impl FenceStore for MockStore {
    async fn acquire(&self, _key: &str, _holder: &str) -> Result<Option<LeaseGrant>, FenceError> {
        let mut state = self.0.lock().unwrap();
        if !state.available {
            return Err(FenceError::Store("mock quorum unavailable".into()));
        }
        if state
            .authority
            .as_ref()
            .is_some_and(|a| Instant::now() < a.expires)
        {
            return Ok(None);
        }
        state.epoch += 1;
        let grant = LeaseGrant {
            id: state.epoch as i64,
            epoch: state.epoch,
        };
        state.authority = Some(ServerLease {
            grant,
            expires: Instant::now() + Duration::from_secs(5),
        });
        Ok(Some(grant))
    }
    async fn keep_alive(&self, grant: LeaseGrant) -> Result<(), FenceError> {
        let mut state = self.0.lock().unwrap();
        if !state.available {
            return Err(FenceError::Store("mock quorum unavailable".into()));
        }
        let authority = state.authority.as_mut().ok_or(FenceError::NotAuthority)?;
        if authority.grant.id != grant.id || Instant::now() >= authority.expires {
            return Err(FenceError::NotAuthority);
        }
        authority.expires = Instant::now() + Duration::from_secs(5);
        Ok(())
    }
    async fn read_short_gate(&self) -> Result<GateState, FenceError> {
        let (available, gate, delay) = {
            let state = self.0.lock().unwrap();
            (state.available, state.gate.clone(), state.gate_delay)
        };
        if !available {
            return Err(FenceError::Store("mock quorum unavailable".into()));
        }
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        Ok(gate)
    }
}
fn config(holder: &str) -> FenceConfig {
    FenceConfig {
        holder: holder.into(),
        election_key: "/cpu-test/native-authority".into(),
        short_workers: (0..12).map(|n| format!("native-{n}")).collect(),
        max_effective_load: 1,
    }
}
fn samples() -> Vec<WorkerLoad> {
    (0..12)
        .map(|n| WorkerLoad {
            worker: format!("native-{n}"),
            healthy: true,
            running: 0,
            waiting: 0,
            active_request_ids: Some(vec![]),
            native_incarnation: "1".repeat(64),
            sampled_at: Instant::now(),
        })
        .collect()
}
async fn active() -> (Arc<NativeFence>, Arc<MockStore>) {
    let store = MockStore::new();
    let fence = NativeFence::new(config("node65"), store.clone(), store.clone()).unwrap();
    fence.refresh_authority().await.unwrap();
    fence.update_samples(samples()).unwrap();
    seed_warm(&fence).await;
    (fence, store)
}
async fn reserve(fence: &Arc<NativeFence>, request: &str) -> Result<GenerationPermit, FenceError> {
    fence
        .take_short_permit(
            request,
            &affinity("prefix"),
            ExpectedRequestIds::new(request, 1)?,
            100,
            &BTreeSet::new(),
        )
        .await
}
fn affinity(prefix: &str) -> NativeAffinity {
    affinity_for_chat(&serde_json::json!({"model":"deepseek-ai/DeepSeek-V4.1-Flash", "messages":[{"role":"user","content":prefix}], "cache_salt":"cpu-fake-policy"}), "explicit-cpu-model-renderer-native-identity").unwrap()
}
async fn seed_warm(fence: &Arc<NativeFence>) {
    ready_prefixes(fence).await;
    let seed = reserve(fence, "seed-warm").await.unwrap();
    seed.complete_success(100, Some(95));
}
async fn ready_prefixes(fence: &Arc<NativeFence>) {
    fence.start_prefixes();
    for _ in 0..20 {
        if fence.prefixes_ready() {
            return;
        }
        tokio::task::yield_now().await;
    }
    panic!("explicit CPU prefix snapshot not delivered");
}

#[tokio::test(start_paused = true)]
async fn authority_65_to_64_then_reverse_without_double_active() {
    for (first, second) in [("node65", "node64"), ("node64", "node65")] {
        let store = MockStore::new();
        let a = NativeFence::new(config(first), store.clone(), store.clone()).unwrap();
        let b = NativeFence::new(config(second), store.clone(), store.clone()).unwrap();
        a.refresh_authority().await.unwrap();
        assert_eq!(b.refresh_authority().await, Err(FenceError::NotAuthority));
        assert_eq!(a.check_generation().unwrap(), 1);
        tokio::time::advance(Duration::from_secs(3)).await;
        assert_eq!(a.check_generation(), Err(FenceError::NotAuthority));
        assert_eq!(b.refresh_authority().await, Err(FenceError::NotAuthority));
        tokio::time::advance(Duration::from_secs(2)).await;
        b.refresh_authority().await.unwrap();
        assert_eq!(b.check_generation().unwrap(), 2);
        assert_eq!(a.check_generation(), Err(FenceError::NotAuthority));
        assert_eq!(a.refresh_authority().await, Err(FenceError::NotAuthority));
    }
}

#[tokio::test(start_paused = true)]
async fn quorum_failure_immediately_fences_next_generation() {
    let (fence, store) = active().await;
    store.0.lock().unwrap().available = false;
    assert!(matches!(
        reserve(&fence, "r1").await,
        Err(FenceError::Store(_))
    ));
    assert!(fence.refresh_authority().await.is_err());
    assert_eq!(fence.check_generation(), Err(FenceError::NotAuthority));
    assert_eq!(fence.reservation_count(), 0);
}

#[tokio::test(start_paused = true)]
async fn short_maintenance_drains_existing_permit_and_does_not_block_long() {
    let (fence, store) = active().await;
    let permit = reserve(&fence, "existing").await.unwrap();
    store.0.lock().unwrap().gate.enabled = false;
    assert!(matches!(
        reserve(&fence, "new").await,
        Err(FenceError::ShortDisabled)
    ));
    assert!(
        fence.check_generation().is_ok(),
        "long admission checks authority, not the short gate"
    );
    assert_eq!(
        fence.reservation_count(),
        1,
        "already admitted native work remains visible for draining"
    );
    drop(permit);
    assert_eq!(fence.reservation_count(), 0);
}

#[tokio::test(start_paused = true)]
async fn invalid_gate_revision_and_commit_fail_closed() {
    let (fence, store) = active().await;
    for gate in [
        GateState {
            enabled: true,
            commit_id: "😃😃😃😃".into(),
            revision: 2,
            mod_revision: 1,
        },
        GateState {
            enabled: true,
            commit_id: "short".into(),
            revision: 2,
            mod_revision: 1,
        },
        GateState {
            enabled: true,
            commit_id: "cpu-mock-commit-001".into(),
            revision: 1,
            mod_revision: 2,
        },
        GateState {
            enabled: true,
            commit_id: "cpu-mock-commit-001".into(),
            revision: 2,
            mod_revision: 0,
        },
    ] {
        store.0.lock().unwrap().gate = gate;
        assert!(matches!(
            reserve(&fence, "r").await,
            Err(FenceError::InvalidGate)
        ));
    }
    assert_eq!(fence.reservation_count(), 0);
}

#[tokio::test(start_paused = true)]
async fn full_healthy_fresh_worker_allowlist_is_required() {
    let (fence, _) = active().await;
    for bad in [
        samples()[..11].to_vec(),
        {
            let mut s = samples();
            s[11].worker = "alien".into();
            s
        },
        {
            let mut s = samples();
            s[11].worker = "native-0".into();
            s
        },
        {
            let mut s = samples();
            s.push(s[0].clone());
            s
        },
        {
            let mut s = samples();
            s[11].healthy = false;
            s
        },
        {
            let mut s = samples();
            s[11].sampled_at += Duration::from_millis(1);
            s
        },
    ] {
        assert_eq!(
            fence.update_samples(bad),
            Err(FenceError::IncompleteSamples)
        );
        assert!(!fence.is_ready());
        assert!(matches!(
            reserve(&fence, "r").await,
            Err(FenceError::IncompleteSamples)
        ));
    }
    fence.update_samples(samples()).unwrap();
    tokio::time::advance(Duration::from_millis(2501)).await;
    assert!(!fence.is_ready());
    assert!(matches!(
        reserve(&fence, "r").await,
        Err(FenceError::IncompleteSamples)
    ));
}

#[tokio::test(start_paused = true)]
async fn reservation_atomic_capacity_duplicate_and_drop_release() {
    let (fence, _) = active().await;
    let permit = reserve(&fence, "one").await.unwrap();
    assert_eq!(fence.effective_load("native-0"), Some(1));
    assert!(matches!(
        reserve(&fence, "two").await,
        Err(FenceError::Saturated)
    ));
    assert!(matches!(
        reserve(&fence, "one").await,
        Err(FenceError::DuplicateRequest)
    ));
    drop(permit);
    assert_eq!(fence.effective_load("native-0"), Some(0));
    assert!(reserve(&fence, "two").await.is_ok());
    assert_eq!(fence.reservation_count(), 0);
}

#[tokio::test(start_paused = true)]
async fn later_native_load_without_request_ids_cannot_clear_reservation() {
    let (fence, _) = active().await;
    let permit = reserve(&fence, "one").await.unwrap();
    tokio::time::advance(Duration::from_millis(1)).await;
    let mut native = samples();
    native[0].running = 1;
    native[0].active_request_ids = Some(vec!["foreign-native".into()]);
    fence.update_samples(native).unwrap();
    assert_eq!(fence.effective_load("native-0"), Some(2));
    drop(permit);
    assert_eq!(
        fence.effective_load("native-0"),
        Some(1),
        "native running sample still conservatively reserves capacity"
    );
}

#[tokio::test(start_paused = true)]
async fn sample_after_reservation_before_post_does_not_remove_its_charge() {
    let (fence, _) = active().await;
    let permit = reserve(&fence, "reserved-not-posted").await.unwrap();
    tokio::time::advance(Duration::from_millis(1)).await;
    fence.update_samples(samples()).unwrap();
    assert_eq!(fence.effective_load("native-0"), Some(1));
    assert!(matches!(
        reserve(&fence, "second").await,
        Err(FenceError::Saturated)
    ));
    drop(permit);
}

#[tokio::test(start_paused = true)]
async fn foreign_native_load_plus_local_reservation_respects_capacity_two() {
    let store = MockStore::new();
    let mut configuration = config("node65");
    configuration.max_effective_load = 2;
    let fence = NativeFence::new(configuration, store.clone(), store).unwrap();
    fence.refresh_authority().await.unwrap();
    fence.update_samples(samples()).unwrap();
    seed_warm(&fence).await;
    let permit = reserve(&fence, "local-not-posted").await.unwrap();
    tokio::time::advance(Duration::from_millis(1)).await;
    let mut native = samples();
    native[0].running = 1;
    native[0].active_request_ids = Some(vec!["foreign-native".into()]);
    fence.update_samples(native).unwrap();
    assert_eq!(fence.effective_load("native-0"), Some(2));
    assert!(matches!(
        reserve(&fence, "second-local").await,
        Err(FenceError::Saturated)
    ));
    drop(permit);
}

#[tokio::test(start_paused = true)]
async fn ids_nonstream_exact_native_receipt_restores_capacity_two() {
    let store = MockStore::new();
    let mut configuration = config("node65");
    configuration.max_effective_load = 2;
    let fence = NativeFence::new(configuration, store.clone(), store).unwrap();
    fence.refresh_authority().await.unwrap();
    fence.update_samples(samples()).unwrap();
    seed_warm(&fence).await;
    let permit = reserve(&fence, "native-generating").await.unwrap();
    tokio::time::advance(Duration::from_millis(1)).await;
    // No JSON response/content has arrived. Foreign IDs do not absorb pending.
    let mut native = samples();
    native[0].running = 1;
    native[0].active_request_ids = Some(vec!["native-generating-other".into()]);
    fence.update_samples(native).unwrap();
    assert_eq!(fence.effective_load("native-0"), Some(2));
    assert!(matches!(
        reserve(&fence, "second").await,
        Err(FenceError::Saturated)
    ));
    // A later native snapshot contains the exact owned request ID. No response
    // headers, first-content event or time-based acceptance inference is needed.
    tokio::time::advance(Duration::from_millis(1)).await;
    let mut native = samples();
    native[0].running = 1;
    native[0].active_request_ids = Some(vec!["native-generating".into()]);
    fence.update_samples(native).unwrap();
    assert_eq!(fence.effective_load("native-0"), Some(1));
    let second = reserve(&fence, "second").await.unwrap();
    assert_eq!(fence.effective_load("native-0"), Some(2));
    drop(second);
    drop(permit);
}

#[tokio::test(start_paused = true)]
async fn ids_prefix_collision_and_new_timestamp_do_not_absorb_pending() {
    let (fence, _) = active().await;
    tokio::time::advance(Duration::from_millis(1)).await;
    let permit = reserve(&fence, "pending").await.unwrap();
    tokio::time::advance(Duration::from_millis(1)).await;
    let mut native = samples();
    native[0].running = 1;
    native[0].active_request_ids = Some(vec!["pending_0".into()]);
    fence.update_samples(native).unwrap();
    assert_eq!(fence.effective_load("native-0"), Some(2));
    assert!(matches!(
        reserve(&fence, "second").await,
        Err(FenceError::Saturated)
    ));
    drop(permit);
}

#[test]
fn ids_compact_descriptor_exact_grammar_and_u32_domain() {
    let one = ExpectedRequestIds::new("R", 1).unwrap();
    assert!(one.matches("R"));
    for bad in ["R_0", "R_0:prefill", "R_0:sample:0", "R-other"] {
        assert!(!one.matches(bad));
    }
    let parallel = ExpectedRequestIds::new("R", 3).unwrap();
    for good in ["R_0:prefill", "R_0:sample:0", "R_0:sample:2"] {
        assert!(parallel.matches(good));
    }
    for bad in [
        "R",
        "R_0",
        "R_1:sample:0",
        "RR_0:sample:0",
        "R_0:sample:3",
        "R_0:sample:00",
        "R_0:sample:+1",
        "R_0:sample:1x",
        "R_0:sample:1:extra",
    ] {
        assert!(!parallel.matches(bad));
    }
    let huge = ExpectedRequestIds::new("R", u32::MAX).unwrap();
    assert!(huge.matches("R_0:sample:4294967294"));
    assert!(!huge.matches("R_0:sample:4294967295"));
    assert!(!huge.matches("R_0:sample:4294967296"));
    assert!(ExpectedRequestIds::new("R", 0).is_err());
}

#[tokio::test(start_paused = true)]
async fn ids_missing_null_duplicate_mismatch_oversize_fence_all_short_workers() {
    let (fence, _) = active().await;
    for bad in [
        {
            let mut s = samples();
            s[0].active_request_ids = None;
            s
        },
        {
            let mut s = samples();
            s[0].running = 1;
            s
        },
        {
            let mut s = samples();
            s[0].running = 2;
            s[0].active_request_ids = Some(vec!["R".into(), "R".into()]);
            s
        },
        {
            let mut s = samples();
            s[0].running = 1;
            s[0].active_request_ids = Some(vec!["".into()]);
            s
        },
        {
            let mut s = samples();
            s[0].running = 1;
            // Qualified producer marks a full MsgPack SHM snapshot overflow
            // as unavailable IDs; the gateway must not guess another size cap.
            s[0].active_request_ids = None;
            s
        },
    ] {
        assert_eq!(
            fence.update_samples(bad),
            Err(FenceError::IncompleteSamples)
        );
        assert!(matches!(
            reserve(&fence, "blocked").await,
            Err(FenceError::IncompleteSamples)
        ));
    }
}

#[tokio::test(start_paused = true)]
async fn ids_parallel_prefill_and_samples_keep_exact_weight() {
    let (fence, _) = active().await;
    let permit = fence
        .take_short_permit(
            "R",
            &affinity("parallel-cold"),
            ExpectedRequestIds::new("R", 3).unwrap(),
            100,
            &BTreeSet::new(),
        )
        .await
        .unwrap();
    assert_eq!(fence.effective_load(&permit.worker), Some(3));
    let worker = permit.worker.clone();
    for ids in [
        vec!["R_0:prefill"],
        vec!["R_0:sample:0"],
        vec!["R_0:sample:0", "R_0:sample:1", "R_0:sample:2"],
    ] {
        let mut native = samples();
        let row = native.iter_mut().find(|s| s.worker == worker).unwrap();
        row.running = ids.len();
        row.active_request_ids = Some(ids.into_iter().map(String::from).collect());
        fence.update_samples(native).unwrap();
        assert_eq!(fence.effective_load(&worker), Some(3));
    }
    drop(permit);
    assert_eq!(fence.reservation_count(), 0);
}

#[tokio::test(start_paused = true)]
async fn ids_parallel_impossible_simultaneous_prefill_and_all_samples_fences() {
    let (fence, _) = active().await;
    let permit = fence
        .take_short_permit(
            "R",
            &affinity("parallel-cold"),
            ExpectedRequestIds::new("R", 2).unwrap(),
            100,
            &BTreeSet::new(),
        )
        .await
        .unwrap();
    let mut native = samples();
    let row = native
        .iter_mut()
        .find(|s| s.worker == permit.worker)
        .unwrap();
    row.running = 3;
    row.active_request_ids = Some(vec![
        "R_0:prefill".into(),
        "R_0:sample:0".into(),
        "R_0:sample:1".into(),
    ]);
    assert_eq!(
        fence.update_samples(native),
        Err(FenceError::IncompleteSamples)
    );
    assert!(fence.effective_load(&permit.worker).is_none());
}

#[tokio::test(start_paused = true)]
async fn ids_cold_c32_does_not_become_whole_pool_hard_cap_24() {
    let store = MockStore::new();
    let mut configuration = config("node65");
    configuration.max_effective_load = 2;
    let fence = NativeFence::new(configuration, store.clone(), store).unwrap();
    fence.refresh_authority().await.unwrap();
    fence.update_samples(samples()).unwrap();
    ready_prefixes(&fence).await;
    let mut permits = Vec::new();
    for j in 0..32 {
        let rid = format!("cold-{j}");
        let prefix = format!("unique-cold-prefix-{j}");
        permits.push(
            fence
                .take_short_permit(
                    &rid,
                    &affinity(&prefix),
                    ExpectedRequestIds::new(&rid, 1).unwrap(),
                    100,
                    &BTreeSet::new(),
                )
                .await
                .unwrap(),
        );
    }
    assert_eq!(fence.reservation_count(), 32);
    assert!(fence.effective_load("native-0").unwrap() > 2);
    drop(permits);
    assert_eq!(fence.reservation_count(), 0);
}

#[tokio::test(start_paused = true)]
async fn ids_early_drop_does_not_create_warm_affinity() {
    let (fence, _) = active().await;
    let first = fence
        .take_short_permit(
            "first",
            &affinity("never-completed"),
            ExpectedRequestIds::new("first", 1).unwrap(),
            100,
            &BTreeSet::new(),
        )
        .await
        .unwrap();
    drop(first);
    let mut native = samples();
    native[0].running = 2;
    native[0].active_request_ids = Some(vec!["foreign-a".into(), "foreign-b".into()]);
    fence.update_samples(native).unwrap();
    let second = fence
        .take_short_permit(
            "second",
            &affinity("never-completed"),
            ExpectedRequestIds::new("second", 1).unwrap(),
            100,
            &BTreeSet::new(),
        )
        .await
        .unwrap();
    assert_ne!(
        second.worker, "native-0",
        "no warm primary on cancellation; use cold least loaded"
    );
}

#[test]
fn ids_affinity_append_stable_canonical_and_isolated_domains() {
    let body = serde_json::json!({"model":"ds","messages":[{"role":"user","content":"first"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],"cache_salt":"tenant-a"});
    let original = affinity_for_chat(&body, "weights-image-renderer-template-A").unwrap();
    let mut grown = body.clone();
    grown["messages"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({"role":"assistant","content":"second"}));
    grown["rid"] = serde_json::json!("owned-new-id");
    grown["n"] = serde_json::json!(u32::MAX);
    grown["temperature"] = serde_json::json!(0.9);
    let extended = affinity_for_chat(&grown, "weights-image-renderer-template-A").unwrap();
    assert_eq!(original.namespace, extended.namespace);
    assert_eq!(original.checkpoints[0], extended.checkpoints[0]);
    assert_eq!(extended.checkpoints.len(), 2);
    for field in ["cache_salt", "tools"] {
        let mut isolated = body.clone();
        isolated[field] = serde_json::json!("different");
        assert_ne!(
            original.namespace,
            affinity_for_chat(&isolated, "weights-image-renderer-template-A")
                .unwrap()
                .namespace
        );
    }
    assert_ne!(
        original.namespace,
        affinity_for_chat(&body, "weights-image-renderer-template-B")
            .unwrap()
            .namespace
    );
    let reversed = serde_json::from_str(r#"{"cache_salt":"tenant-a","tools":[{"function":{"parameters":{"type":"object"},"name":"f"},"type":"function"}],"messages":[{"content":"first","role":"user"}],"model":"ds"}"#).unwrap();
    assert_eq!(
        original.checkpoints,
        affinity_for_chat(&reversed, "weights-image-renderer-template-A")
            .unwrap()
            .checkpoints
    );
    let mut same_message_growth = body.clone();
    same_message_growth["messages"][0]["content"] = serde_json::json!("first longer");
    assert_ne!(
        original.checkpoints,
        affinity_for_chat(&same_message_growth, "weights-image-renderer-template-A")
            .unwrap()
            .checkpoints
    );
}

fn busy_native_zero() -> Vec<WorkerLoad> {
    let mut loads = samples();
    loads[0].running = 2;
    loads[0].active_request_ids = Some(vec!["foreign-a".into(), "foreign-b".into()]);
    loads
}

#[tokio::test(start_paused = true)]
async fn ids_completed_checkpoint_threshold_preserves_hot_set_without_cold_spill() {
    let (fence, _) = active().await;
    let mut grown = serde_json::json!({"model":"deepseek-ai/DeepSeek-V4.1-Flash", "messages":[{"role":"user","content":"prefix"},{"role":"assistant","content":"grown"}], "cache_salt":"cpu-fake-policy"});
    let hint = affinity_for_chat(&grown, "explicit-cpu-model-renderer-native-identity").unwrap();
    fence.update_samples(busy_native_zero()).unwrap();
    // Completed prior input 100/current input 300 >0.3 preserves warm primary.
    assert!(matches!(
        fence
            .take_short_permit(
                "hot",
                &hint,
                ExpectedRequestIds::new("hot", 1).unwrap(),
                300,
                &BTreeSet::new(),
            )
            .await,
        Err(FenceError::Saturated)
    ));
    // Tiny system/prefix 100/current 1000 does not label the new input warm.
    let cold = fence
        .take_short_permit(
            "cold",
            &hint,
            ExpectedRequestIds::new("cold", 1).unwrap(),
            1000,
            &BTreeSet::new(),
        )
        .await
        .unwrap();
    assert_ne!(cold.worker, "native-0");
    drop(cold);
    grown["messages"][0]["content"] = serde_json::json!("new unrelated");
    let unrelated =
        affinity_for_chat(&grown, "explicit-cpu-model-renderer-native-identity").unwrap();
    assert!(
        fence
            .take_short_permit(
                "unrelated",
                &unrelated,
                ExpectedRequestIds::new("unrelated", 1).unwrap(),
                100,
                &BTreeSet::new(),
            )
            .await
            .is_ok()
    );
}

#[tokio::test(start_paused = true)]
async fn ids_restart_clears_primary_after_bad_snapshot_and_blocks_old_completion() {
    let (fence, _) = active().await;
    let old = reserve(&fence, "in-flight-old").await.unwrap();
    let mut invalid = samples();
    invalid[0].active_request_ids = None;
    assert!(fence.update_samples(invalid).is_err());
    let mut restarted = busy_native_zero();
    restarted[0].native_incarnation = "2".repeat(64);
    fence.update_samples(restarted).unwrap();
    old.complete_success(100, Some(95));
    drop(old);
    let new = reserve(&fence, "after-restart").await.unwrap();
    assert_ne!(
        new.worker, "native-0",
        "old native identity must not restore phantom warm affinity"
    );
}

#[tokio::test(start_paused = true)]
async fn ids_new_authority_reads_shared_local_hit_and_rejects_restarted_incarnation() {
    let (first, store) = active().await;
    for _ in 0..20 {
        if !store.0.lock().unwrap().prefix_rows.is_empty() {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert_eq!(store.0.lock().unwrap().prefix_rows.len(), 1);
    first.stop();
    tokio::time::advance(Duration::from_secs(5)).await;
    let second = NativeFence::new(config("node64"), store.clone(), store.clone()).unwrap();
    second.refresh_authority().await.unwrap();
    second.update_samples(busy_native_zero()).unwrap();
    ready_prefixes(&second).await;
    assert!(
        matches!(
            reserve(&second, "shared-hot").await,
            Err(FenceError::Saturated)
        ),
        "real native >=90% hit record retains the warm set across authorities"
    );
    let mut restarted = busy_native_zero();
    restarted[0].native_incarnation = "2".repeat(64);
    second.update_samples(restarted).unwrap();
    assert_ne!(
        reserve(&second, "shared-restarted").await.unwrap().worker,
        "native-0"
    );
}

#[tokio::test(start_paused = true)]
async fn ids_missing_shared_metadata_fences_short_but_not_long_authority() {
    let store = MockStore::new();
    let fence = NativeFence::new(config("node65"), store.clone(), store).unwrap();
    fence.refresh_authority().await.unwrap();
    fence.update_samples(samples()).unwrap();
    assert!(!fence.is_ready());
    assert!(matches!(
        reserve(&fence, "no-prefix-snapshot").await,
        Err(FenceError::Store(_))
    ));
    assert!(fence.check_generation().is_ok());
}

#[tokio::test(start_paused = true)]
async fn ids_cacheless_success_is_primary_only_and_never_publishes_verified_replica() {
    let store = MockStore::new();
    let fence = NativeFence::new(config("node65"), store.clone(), store.clone()).unwrap();
    fence.refresh_authority().await.unwrap();
    fence.update_samples(samples()).unwrap();
    ready_prefixes(&fence).await;
    let permit = reserve(&fence, "no-cache-usage").await.unwrap();
    permit.complete_success(100, None);
    drop(permit);
    tokio::task::yield_now().await;
    assert!(store.0.lock().unwrap().prefix_rows.is_empty());
    fence.update_samples(busy_native_zero()).unwrap();
    assert!(matches!(
        reserve(&fence, "primary-full").await,
        Err(FenceError::Saturated)
    ));
}

#[tokio::test(start_paused = true)]
async fn ids_malformed_endpoint_incarnation_fences_short_pool() {
    let (fence, _) = active().await;
    for identity in [
        "",
        "native-image-v1",
        "00000000000040008000000000000001",
        &"G".repeat(64),
    ] {
        let mut invalid = samples();
        invalid[0].native_incarnation = identity.into();
        assert_eq!(
            fence.update_samples(invalid),
            Err(FenceError::IncompleteSamples)
        );
    }
}

#[tokio::test(start_paused = true)]
async fn ha_excluded_worker_is_not_selected_and_warm_set_does_not_cold_spill() {
    let (fence, _) = active().await;
    let excluded = BTreeSet::from(["native-0".into()]);
    let cold = fence
        .take_short_permit(
            "cold-retry",
            &affinity("ha-new-prefix"),
            ExpectedRequestIds::new("cold-retry", 1).unwrap(),
            100,
            &excluded,
        )
        .await
        .unwrap();
    assert_ne!(cold.worker, "native-0");
    assert!(
        fence.is_ready(),
        "selection exclusion does not invalidate the complete fresh pool"
    );
    drop(cold);
    assert!(
        matches!(
            fence
                .take_short_permit(
                    "warm-retry",
                    &affinity("prefix"),
                    ExpectedRequestIds::new("warm-retry", 1).unwrap(),
                    100,
                    &excluded
                )
                .await,
            Err(FenceError::Saturated)
        ),
        "excluded warm primary without verified replica must not become implicit cold overflow"
    );
    assert_eq!(fence.reservation_count(), 0);
}

#[tokio::test(start_paused = true)]
async fn ha_connect_retry_releases_weight_and_all_failed_workers_fail_closed() {
    let (fence, _) = active().await;
    let hint = affinity("ha-weighted-cold");
    let first = fence
        .take_short_permit(
            "R",
            &hint,
            ExpectedRequestIds::new("R", 3).unwrap(),
            100,
            &BTreeSet::new(),
        )
        .await
        .unwrap();
    let failed = first.worker.clone();
    assert_eq!(fence.effective_load(&failed), Some(3));
    drop(first); // Engine may do this only on proven pre-send connect error.
    assert_eq!(fence.effective_load(&failed), Some(0));
    let excluded = BTreeSet::from([failed.clone()]);
    let second = fence
        .take_short_permit(
            "R",
            &hint,
            ExpectedRequestIds::new("R", 3).unwrap(),
            100,
            &excluded,
        )
        .await
        .unwrap();
    assert_ne!(second.worker, failed);
    assert_eq!(fence.reservation_count(), 1);
    assert_eq!(fence.effective_load(&second.worker), Some(3));
    assert_eq!(fence.effective_load(&failed), Some(0));
    drop(second);
    let all_failed = config("unused").short_workers.into_iter().collect();
    assert!(matches!(
        fence
            .take_short_permit(
                "R",
                &hint,
                ExpectedRequestIds::new("R", 3).unwrap(),
                100,
                &all_failed
            )
            .await,
        Err(FenceError::Saturated)
    ));
    assert_eq!(fence.reservation_count(), 0);
    assert!(
        fence.is_ready(),
        "per-request exclusions do not mutate shared pool capacity or samples"
    );
    let next = fence
        .take_short_permit(
            "next",
            &hint,
            ExpectedRequestIds::new("next", 3).unwrap(),
            100,
            &BTreeSet::new(),
        )
        .await
        .unwrap();
    assert_eq!(next.worker, failed);
    assert_eq!(fence.effective_load(&next.worker), Some(3));
}

#[tokio::test(start_paused = true)]
async fn slow_gate_response_cannot_extend_local_authority() {
    let (fence, store) = active().await;
    store.0.lock().unwrap().gate_delay = Duration::from_secs(3);
    assert!(matches!(
        reserve(&fence, "one").await,
        Err(FenceError::NotAuthority)
    ));
    assert_eq!(fence.reservation_count(), 0);
}

#[tokio::test(start_paused = true)]
async fn stop_is_irreversible_and_never_deletes_shared_leader_key() {
    let (fence, store) = active().await;
    fence.stop();
    assert_eq!(fence.check_generation(), Err(FenceError::NotAuthority));
    assert_eq!(
        fence.refresh_authority().await,
        Err(FenceError::NotAuthority)
    );
    let other = NativeFence::new(config("node64"), store.clone(), store).unwrap();
    assert_eq!(
        other.refresh_authority().await,
        Err(FenceError::NotAuthority)
    );
    tokio::time::advance(Duration::from_secs(5)).await;
    other.refresh_authority().await.unwrap();
    assert_eq!(other.check_generation().unwrap(), 2);
}

#[tokio::test(start_paused = true)]
async fn background_lease_heartbeat_renews_and_stop_does_not_revive_it() {
    let store = MockStore::new();
    let a = NativeFence::new(config("node65"), store.clone(), store.clone()).unwrap();
    let b = NativeFence::new(config("node64"), store.clone(), store).unwrap();
    a.start();
    a.start(); // Idempotent: no second election/heartbeat loop.
    tokio::task::yield_now().await;
    assert!(a.check_generation().is_ok());
    for _ in 0..8 {
        tokio::time::advance(Duration::from_secs(1)).await;
        tokio::task::yield_now().await;
        assert!(a.check_generation().is_ok());
        assert_eq!(b.refresh_authority().await, Err(FenceError::NotAuthority));
    }
    a.stop();
    tokio::time::advance(Duration::from_secs(6)).await;
    tokio::task::yield_now().await;
    assert_eq!(a.check_generation(), Err(FenceError::NotAuthority));
    b.refresh_authority().await.unwrap();
    assert_eq!(b.check_generation().unwrap(), 2);
}

#[tokio::test(start_paused = true)]
async fn background_reservation_heartbeat_keeps_live_stream_visible() {
    let (fence, _) = active().await;
    let permit = reserve(&fence, "one").await.unwrap();
    tokio::task::yield_now().await;
    for _ in 0..8 {
        tokio::time::advance(Duration::from_secs(5)).await;
        tokio::task::yield_now().await;
        assert_eq!(fence.reservation_count(), 1);
    }
    drop(permit);
    assert_eq!(fence.reservation_count(), 0);
}

#[tokio::test(start_paused = true)]
async fn abandoned_reservation_expires_and_old_drop_cannot_release_new_one() {
    let (fence, _) = active().await;
    let old = reserve(&fence, "same").await.unwrap();
    // Do not schedule the heartbeat: model a stalled/cancelled reservation task.
    tokio::time::advance(Duration::from_secs(30)).await;
    assert_eq!(fence.reservation_count(), 0);
    // Old server lease expired; next attempt first fences it, then reacquires.
    assert!(fence.refresh_authority().await.is_err());
    fence.refresh_authority().await.unwrap();
    fence.update_samples(samples()).unwrap();
    let new = reserve(&fence, "same").await.unwrap();
    drop(old);
    assert_eq!(fence.reservation_count(), 1);
    drop(new);
    assert_eq!(fence.reservation_count(), 0);
}

#[test]
fn config_cannot_advertise_partial_pool_as_ready() {
    let store = MockStore::new();
    let mut partial = config("node65");
    partial.short_workers.pop();
    assert!(NativeFence::new(partial, store.clone(), store.clone()).is_err());
    let mut duplicate = config("node65");
    duplicate.short_workers[11] = duplicate.short_workers[0].clone();
    assert!(NativeFence::new(duplicate, store.clone(), store).is_err());
}

/// Explicit transport qualification only. Run against an isolated local etcd
/// created for this test, never the production cluster. The test creates its own
/// election key but requires the caller to prepare the fixed admission gate.
#[tokio::test]
#[ignore = "requires isolated CPU etcd and an explicitly configured short-admission gate"]
async fn real_local_etcd_transport_lease_and_linearizable_gate() {
    let endpoint =
        std::env::var("DYNAMO_TEST_LOCAL_ETCD").expect("isolated local etcd URL required");
    let parsed = url::Url::parse(&endpoint).unwrap();
    assert!(
        parsed.scheme() == "http"
            && matches!(parsed.host_str(), Some("127.0.0.1" | "localhost"))
            && parsed.username().is_empty()
            && parsed.password().is_none()
            && parsed.port().is_some()
            && parsed.path() == "/"
            && parsed.query().is_none()
            && parsed.fragment().is_none(),
        "etcd transport test requires an unambiguous loopback origin"
    );
    let mut control = etcd_client::Client::connect(vec![endpoint.clone()], None)
        .await
        .unwrap();
    let store = EtcdFenceStore::connect(vec![endpoint.clone()])
        .await
        .unwrap();
    let prefixes = EtcdPrefixRecords::connect(vec![endpoint]).await.unwrap();
    let mut a_config = config("node65");
    a_config.election_key = format!("/cpu-test/dynamo-fencing/{}", std::process::id());
    let mut b_config = a_config.clone();
    b_config.holder = "node64".into();
    let a = NativeFence::new(a_config, store.clone(), prefixes.clone()).unwrap();
    let b = NativeFence::new(b_config, store, prefixes).unwrap();
    a.refresh_authority().await.unwrap();
    a.update_samples(samples()).unwrap();
    ready_prefixes(&a).await;
    assert!(reserve(&a, "real-etcd").await.is_ok());
    control
        .put(
            dynamo_llm::native_fencing::SHORT_ADMISSION_KEY,
            r#"{"enabled":false,"commit_id":"local-cpu-fencing-disabled-20261001"}"#,
            None,
        )
        .await
        .unwrap();
    assert!(matches!(
        reserve(&a, "disabled").await,
        Err(FenceError::ShortDisabled)
    ));
    assert!(
        a.check_generation().is_ok(),
        "long still uses only authority"
    );
    control
        .put(
            dynamo_llm::native_fencing::SHORT_ADMISSION_KEY,
            r#"{"enabled":"true","commit_id":"local-cpu-fencing-bad-value-20261001"}"#,
            None,
        )
        .await
        .unwrap();
    assert!(matches!(
        reserve(&a, "bad").await,
        Err(FenceError::InvalidGate)
    ));
    control
        .delete(dynamo_llm::native_fencing::SHORT_ADMISSION_KEY, None)
        .await
        .unwrap();
    assert!(matches!(
        reserve(&a, "missing").await,
        Err(FenceError::InvalidGate)
    ));
    control
        .put(
            dynamo_llm::native_fencing::SHORT_ADMISSION_KEY,
            r#"{"enabled":true,"commit_id":"local-cpu-fencing-enabled-20261001"}"#,
            None,
        )
        .await
        .unwrap();
    assert!(b.refresh_authority().await.is_err());
    a.stop();
    assert!(a.check_generation().is_err());
    tokio::time::sleep(Duration::from_secs(6)).await;
    b.refresh_authority().await.unwrap();
    assert!(b.check_generation().is_ok());
    b.stop();
}
