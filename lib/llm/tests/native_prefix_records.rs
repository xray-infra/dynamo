// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[path = "../src/native_prefix_records.rs"]
mod records;

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use records::*;
use tokio_util::sync::CancellationToken;

const WORKER: &str = "http://worker1:30000";
const INCARNATION: &str = concat!(
    "0123456789abcdef",
    "0123456789abcdef",
    "0123456789abcdef",
    "0123456789abcdef"
);
fn prefix() -> String {
    format!("{}/{}", "a".repeat(64), "b".repeat(64))
}
fn allowed() -> BTreeSet<String> {
    [WORKER.to_string()].into()
}
fn row(cached: u64, confirmed: u64) -> PrefixRow {
    PrefixRow {
        key: record_key(&prefix(), WORKER).into_bytes(),
        value: serde_json::to_vec(&ReplicaRecord {
            worker: WORKER.into(),
            evidence: "local_hit".into(),
            confirmed_at_ms: confirmed,
            context_tokens: 100,
            native_incarnation: INCARNATION.into(),
            prompt_tokens: Some(100),
            cached_tokens: Some(cached),
        })
        .unwrap(),
    }
}

#[test]
fn published_record_schema_is_preserved_with_separate_namespace_and_checkpoint_key() {
    let r = row(90, 9_500);
    let verified = validate_record(&r, &allowed(), 10_000, 1_000).unwrap();
    assert_eq!(verified.age_ms, 500);
    assert_eq!(verified.evidence, ReplicaEvidence::LocalHit);
    assert_eq!(verified.prefix, prefix());
    let value: serde_json::Value = serde_json::from_slice(&r.value).unwrap();
    assert_eq!(value.as_object().unwrap().len(), 7);
    assert_eq!(verified.context_tokens, 100);
    assert_eq!(verified.native_incarnation, INCARNATION);
    assert!(!value.as_object().unwrap().contains_key("epoch"));
    assert!(!value.as_object().unwrap().contains_key("fingerprint"));
}

#[test]
fn evidence_rejects_missing_tokens_below_ninety_percent_and_overflow_tricks() {
    assert!(validate_record(&row(89, 10_000), &allowed(), 10_000, 1_000).is_none());
    let mut r = row(90, 10_000);
    let mut value: serde_json::Value = serde_json::from_slice(&r.value).unwrap();
    value.as_object_mut().unwrap().remove("cached_tokens");
    r.value = serde_json::to_vec(&value).unwrap();
    assert!(validate_record(&r, &allowed(), 10_000, 1_000).is_none());
    value["prompt_tokens"] = serde_json::json!(u64::MAX);
    value["cached_tokens"] = serde_json::json!(u64::MAX / 2);
    r.value = serde_json::to_vec(&value).unwrap();
    assert!(validate_record(&r, &allowed(), 10_000, 1_000).is_none());
    value["prompt_tokens"] = serde_json::json!(0);
    r.value = serde_json::to_vec(&value).unwrap();
    assert!(validate_record(&r, &allowed(), 10_000, 1_000).is_none());
}

#[test]
fn v2_requires_exact_single_input_count_and_does_not_substitute_aggregate_usage() {
    let mut r = row(90, 10_000);
    let mut value: serde_json::Value = serde_json::from_slice(&r.value).unwrap();
    value.as_object_mut().unwrap().remove("context_tokens");
    r.value = serde_json::to_vec(&value).unwrap();
    assert!(validate_record(&r, &allowed(), 10_000, 1_000).is_none());
    value["context_tokens"] = serde_json::json!(0);
    r.value = serde_json::to_vec(&value).unwrap();
    assert!(validate_record(&r, &allowed(), 10_000, 1_000).is_none());
    value["context_tokens"] = serde_json::json!(50);
    r.value = serde_json::to_vec(&value).unwrap();
    assert_eq!(
        validate_record(&r, &allowed(), 10_000, 1_000)
            .unwrap()
            .context_tokens,
        50
    );
}

#[test]
fn malformed_key_worker_allowlist_and_evidence_are_not_replica_proof() {
    let r = row(100, 10_000);
    assert!(validate_record(&r, &BTreeSet::new(), 10_000, 1_000).is_none());
    for key in [
        record_key("bad", WORKER),
        record_key(&"a".repeat(64), WORKER),
        record_key(&prefix(), "http://other:30000"),
        format!("{}/trailing", record_key(&prefix(), WORKER)),
        "other-root".into(),
    ] {
        let bad = PrefixRow {
            key: key.into_bytes(),
            value: r.value.clone(),
        };
        assert!(validate_record(&bad, &allowed(), 10_000, 1_000).is_none());
    }
    let mut value: serde_json::Value = serde_json::from_slice(&r.value).unwrap();
    value["evidence"] = serde_json::json!("reservation");
    let bad = PrefixRow {
        key: r.key,
        value: serde_json::to_vec(&value).unwrap(),
    };
    assert!(validate_record(&bad, &allowed(), 10_000, 1_000).is_none());
}

#[test]
fn v2_requires_derived_endpoint_incarnation_and_rejects_the_obsolete_record_root() {
    let mut r = row(90, 10_000);
    let mut value: serde_json::Value = serde_json::from_slice(&r.value).unwrap();
    value.as_object_mut().unwrap().remove("native_incarnation");
    r.value = serde_json::to_vec(&value).unwrap();
    assert!(validate_record(&r, &allowed(), 10_000, 1_000).is_none());
    for invalid in [
        "".to_string(),
        " ".to_string(),
        "ed142522-b0bd-41a6-83cb-c71c2a58b8bc".to_string(),
        "A".repeat(64),
        "g".repeat(64),
        "a".repeat(63),
        "a".repeat(65),
    ] {
        value["native_incarnation"] = serde_json::json!(invalid);
        r.value = serde_json::to_vec(&value).unwrap();
        assert!(validate_record(&r, &allowed(), 10_000, 1_000).is_none());
    }
    let mut obsolete = row(90, 10_000);
    obsolete.key = std::str::from_utf8(&obsolete.key)
        .unwrap()
        .replacen(REPLICA_ROOT, "/dsh/deepseek-v41/short-router/prefix/", 1)
        .into_bytes();
    assert!(validate_record(&obsolete, &allowed(), 10_000, 1_000).is_none());
}

#[test]
fn identical_message_checkpoint_in_different_cache_namespace_does_not_alias() {
    let first = row(90, 10_000);
    let other_prefix = format!("{}/{}", "c".repeat(64), "b".repeat(64));
    let second = PrefixRow {
        key: record_key(&other_prefix, WORKER).into_bytes(),
        value: first.value.clone(),
    };
    let mut cache = PrefixRecordCache::default();
    cache
        .apply(PrefixUpdate::Snapshot(PrefixSnapshot {
            revision: 1,
            rows: vec![first, second],
        }))
        .unwrap();
    let verified = cache.verified(&allowed(), 10_000, 1_000);
    assert_eq!(verified.len(), 2);
    assert_eq!(verified.iter().filter(|r| r.prefix == prefix()).count(), 1);
    assert_eq!(
        verified.iter().filter(|r| r.prefix == other_prefix).count(),
        1
    );
}

#[test]
fn unqualified_shared_l3_declaration_cannot_establish_a_replica() {
    let r = PrefixRow {
        key: record_key(&prefix(), WORKER).into_bytes(),
        value: serde_json::to_vec(&serde_json::json!({
            "worker": WORKER, "evidence": "shared_l3_presence", "confirmed_at_ms": 10_000,
            "context_tokens": 100,
            "native_incarnation": INCARNATION,
        }))
        .unwrap(),
    };
    assert!(validate_record(&r, &allowed(), 10_000, 1_000).is_none());
}

#[test]
fn replica_timestamp_boundaries_match_published_ttl_and_future_skew() {
    assert_eq!(valid_replica_age_ms(9_000, 10_000, 1_000), Some(1_000));
    assert_eq!(valid_replica_age_ms(8_999, 10_000, 1_000), None);
    assert_eq!(valid_replica_age_ms(0, 10_000, 1_000), None);
    assert_eq!(valid_replica_age_ms(15_000, 10_000, 1_000), Some(0));
    assert_eq!(valid_replica_age_ms(15_001, 10_000, 1_000), None);
}

#[test]
fn invalid_replacement_delete_disconnect_and_resnapshot_remove_former_evidence() {
    let r = row(90, 10_000);
    let mut cache = PrefixRecordCache::default();
    assert!(!cache.is_available());
    cache
        .apply(PrefixUpdate::Snapshot(PrefixSnapshot {
            revision: 1,
            rows: vec![r.clone()],
        }))
        .unwrap();
    assert_eq!(cache.verified(&allowed(), 10_000, 1_000).len(), 1);
    assert!(cache.is_available());
    cache
        .apply(PrefixUpdate::Events(PrefixBatch {
            revision: 2,
            events: vec![PrefixEvent::Put(row(89, 10_000))],
        }))
        .unwrap();
    assert!(cache.verified(&allowed(), 10_000, 1_000).is_empty());
    cache
        .apply(PrefixUpdate::Events(PrefixBatch {
            revision: 3,
            events: vec![PrefixEvent::Put(r.clone())],
        }))
        .unwrap();
    cache
        .apply(PrefixUpdate::Events(PrefixBatch {
            revision: 4,
            events: vec![PrefixEvent::Delete(r.key.clone())],
        }))
        .unwrap();
    assert!(cache.verified(&allowed(), 10_000, 1_000).is_empty());
    cache
        .apply(PrefixUpdate::Snapshot(PrefixSnapshot {
            revision: 5,
            rows: vec![r],
        }))
        .unwrap();
    cache
        .apply(PrefixUpdate::Unavailable("partition".into()))
        .unwrap();
    assert!(!cache.is_available());
    assert!(cache.verified(&allowed(), 10_000, 1_000).is_empty());
    cache
        .apply(PrefixUpdate::Snapshot(PrefixSnapshot {
            revision: 6,
            rows: vec![],
        }))
        .unwrap();
    assert!(cache.verified(&allowed(), 10_000, 1_000).is_empty());
    assert!(cache.is_available());
}

#[test]
fn records_expire_even_without_watch_events_and_revision_regression_is_rejected() {
    let mut cache = PrefixRecordCache::default();
    assert!(
        cache
            .apply(PrefixUpdate::Events(PrefixBatch {
                revision: 1,
                events: vec![]
            }))
            .is_err()
    );
    cache
        .apply(PrefixUpdate::Snapshot(PrefixSnapshot {
            revision: 10,
            rows: vec![row(90, 10_000)],
        }))
        .unwrap();
    assert!(cache.verified(&allowed(), 11_001, 1_000).is_empty());
    assert!(
        cache
            .apply(PrefixUpdate::Events(PrefixBatch {
                revision: 9,
                events: vec![]
            }))
            .is_err()
    );
    assert!(cache.verified(&allowed(), 10_000, 1_000).is_empty());
    cache
        .apply(PrefixUpdate::Snapshot(PrefixSnapshot {
            revision: 10,
            rows: vec![row(90, 10_000)],
        }))
        .unwrap();
    assert!(
        cache
            .apply(PrefixUpdate::Snapshot(PrefixSnapshot {
                revision: 9,
                rows: vec![row(90, 10_000)]
            }))
            .is_err()
    );
    assert!(cache.verified(&allowed(), 10_000, 1_000).is_empty());
}

struct FakeStore {
    requested_revision: AtomicI64,
    writes: Mutex<Vec<(PrefixRow, u64)>>,
}
impl Default for FakeStore {
    fn default() -> Self {
        Self {
            requested_revision: AtomicI64::new(0),
            writes: Mutex::new(vec![]),
        }
    }
}
#[async_trait]
impl PrefixStore for FakeStore {
    async fn snapshot(&self) -> Result<PrefixSnapshot, PrefixError> {
        Ok(PrefixSnapshot {
            revision: 7,
            rows: vec![row(90, 10_000)],
        })
    }
    async fn watch_from(&self, revision: i64) -> Result<PrefixStream, PrefixError> {
        self.requested_revision.store(revision, Ordering::SeqCst);
        let deleted = PrefixBatch {
            revision: 8,
            events: vec![PrefixEvent::Delete(row(90, 10_000).key)],
        };
        Ok(Box::pin(futures::stream::iter([
            Ok(deleted),
            Err(PrefixError("partition".into())),
        ])))
    }
    async fn put(&self, row: PrefixRow, ttl_ms: u64) -> Result<(), PrefixError> {
        self.writes.lock().unwrap().push((row, ttl_ms));
        Ok(())
    }
}

#[tokio::test]
async fn fake_transport_watch_starts_at_snapshot_plus_one_and_clears_on_gap() {
    let fake = Arc::new(FakeStore::default());
    let cancel = CancellationToken::new();
    let mut rx = subscribe(fake.clone(), cancel.clone());
    let mut cache = PrefixRecordCache::default();
    for expected in [1, 0, 0] {
        let update = tokio::time::timeout(Duration::from_secs(1), rx.recv())
            .await
            .unwrap()
            .unwrap();
        cache.apply(update).unwrap();
        assert_eq!(cache.verified(&allowed(), 10_000, 1_000).len(), expected);
    }
    assert_eq!(fake.requested_revision.load(Ordering::SeqCst), 8);
    cancel.cancel();
    assert!(
        tokio::time::timeout(Duration::from_secs(1), rx.recv())
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn only_allowed_native_ninety_percent_hit_is_published_with_original_ttl() {
    let fake = FakeStore::default();
    assert!(
        publish_local_hit(
            &fake,
            &allowed(),
            &prefix(),
            WORKER,
            100,
            89,
            100,
            INCARNATION,
            10_000
        )
        .await
        .is_err()
    );
    assert!(
        publish_local_hit(
            &fake,
            &BTreeSet::new(),
            &prefix(),
            WORKER,
            100,
            90,
            100,
            INCARNATION,
            10_000
        )
        .await
        .is_err()
    );
    publish_local_hit(
        &fake,
        &allowed(),
        &prefix(),
        WORKER,
        100,
        90,
        100,
        INCARNATION,
        10_000,
    )
    .await
    .unwrap();
    let writes = fake.writes.lock().unwrap();
    assert_eq!(writes.len(), 1);
    assert_eq!(writes[0].1, REPLICA_TTL_MS);
    assert_eq!(writes[0].0, row(90, 10_000));
}

/// New prefix snapshot/read-watch-gap/put/delete/lease-expiry gate only. This
/// never runs against production or native generation endpoints.
#[tokio::test]
#[ignore = "requires NATIVE_PREFIX_TEST_ETCD_ENDPOINT pointing to a local CPU etcd"]
async fn local_etcd_snapshot_watch_gap_delete_and_ttl() {
    let endpoint = std::env::var("NATIVE_PREFIX_TEST_ETCD_ENDPOINT").unwrap();
    let url = url::Url::parse(&endpoint).unwrap();
    assert!(matches!(
        url.host_str(),
        Some("127.0.0.1" | "localhost" | "::1")
    ));
    let store = EtcdPrefixRecords::connect(vec![endpoint.clone()])
        .await
        .unwrap();
    let unique = format!(
        "{:064x}/{:064x}",
        uuid::Uuid::new_v4().as_u128(),
        uuid::Uuid::new_v4().as_u128()
    );
    let key = record_key(&unique, WORKER);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    publish_local_hit(
        store.as_ref(),
        &allowed(),
        &unique,
        WORKER,
        100,
        95,
        100,
        INCARNATION,
        now,
    )
    .await
    .unwrap();
    let first = store.snapshot().await.unwrap();
    assert!(first.rows.iter().any(|r| r.key == key.as_bytes()));
    // Write after the read, before watch setup. Replay must still deliver it.
    let changed = ReplicaRecord {
        worker: WORKER.into(),
        evidence: "local_hit".into(),
        confirmed_at_ms: now,
        context_tokens: 100,
        native_incarnation: INCARNATION.into(),
        prompt_tokens: Some(100),
        cached_tokens: Some(96),
    };
    store
        .put(
            PrefixRow {
                key: key.as_bytes().to_vec(),
                value: serde_json::to_vec(&changed).unwrap(),
            },
            1_000,
        )
        .await
        .unwrap();
    let mut stream = store.watch_from(first.revision + 1).await.unwrap();
    tokio::time::timeout(Duration::from_secs(3), async {
        while let Some(batch) = stream.next().await {
            if batch
                .unwrap()
                .events
                .iter()
                .any(|e| matches!(e, PrefixEvent::Put(r) if r.key == key.as_bytes()))
            {
                return;
            }
        }
        panic!("watch ended before replaying read/watch-gap put");
    })
    .await
    .unwrap();
    let expiry = tokio::time::timeout(Duration::from_secs(5), async {
        while let Some(batch) = stream.next().await {
            if batch
                .unwrap()
                .events
                .iter()
                .any(|e| matches!(e, PrefixEvent::Delete(k) if k == key.as_bytes()))
            {
                return;
            }
        }
        panic!("watch ended before TTL delete");
    })
    .await;
    // Exact key cleanup also runs before reporting an expiry assertion failure.
    etcd_client::Client::connect([endpoint], None)
        .await
        .unwrap()
        .delete(key, None)
        .await
        .unwrap();
    expiry.unwrap();
}
