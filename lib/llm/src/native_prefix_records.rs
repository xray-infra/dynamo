// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared verified-replica declarations retaining the published DS usage fields
//! and requiring the exact single-input RPC count in the new affinity domain.
//! This transports observed local-hit evidence, not KV pages. There is no
//! qualified shared-L3 producer in this deployment, so L3 declarations cannot
//! establish replicas. Records contain neither KV fingerprints nor epochs.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use etcd_client::{Client, EventType, GetOptions, PutOptions, WatchOptions};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

pub const REPLICA_ROOT: &str = "/dsh/deepseek-v41/dynamo-short-router/v2/prefix/";
pub const REPLICA_TTL_MS: u64 = 300_000;
const MAX_FUTURE_SKEW_MS: u64 = 5_000;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ReplicaRecord {
    pub worker: String,
    pub evidence: String,
    pub confirmed_at_ms: u64,
    /// Exact count-RPC size of one input, separate from potentially aggregated
    /// native usage when the caller requested more than one completion.
    pub context_tokens: u64,
    /// Endpoint incarnation derived by the native transport from sorted
    /// (DP rank, actual scheduler UUID4) tuples with the
    /// `dynamo-native-dp-incarnation-v1\0` domain. This is SHA256 lowercase
    /// hex even for DP1; it is never an image/build fingerprint.
    pub native_incarnation: String,
    #[serde(default)]
    pub prompt_tokens: Option<u64>,
    #[serde(default)]
    pub cached_tokens: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplicaEvidence {
    LocalHit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedReplica {
    pub prefix: String,
    pub worker: String,
    pub confirmed_at_ms: u64,
    pub age_ms: u64,
    pub context_tokens: u64,
    pub native_incarnation: String,
    pub evidence: ReplicaEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixRow {
    pub key: Vec<u8>,
    pub value: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrefixEvent {
    Put(PrefixRow),
    Delete(Vec<u8>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixSnapshot {
    pub revision: i64,
    pub rows: Vec<PrefixRow>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixBatch {
    pub revision: i64,
    pub events: Vec<PrefixEvent>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrefixUpdate {
    Snapshot(PrefixSnapshot),
    Events(PrefixBatch),
    /// A gap cannot establish that a formerly known replica still exists.
    Unavailable(String),
}

#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
#[error("native prefix records: {0}")]
pub struct PrefixError(pub String);

pub type PrefixStream =
    std::pin::Pin<Box<dyn Stream<Item = Result<PrefixBatch, PrefixError>> + Send + 'static>>;

#[async_trait]
pub trait PrefixStore: Send + Sync {
    async fn snapshot(&self) -> Result<PrefixSnapshot, PrefixError>;
    async fn watch_from(&self, revision: i64) -> Result<PrefixStream, PrefixError>;
    async fn put(&self, row: PrefixRow, ttl_ms: u64) -> Result<(), PrefixError>;
}

pub struct EtcdPrefixRecords {
    client: Client,
}

impl EtcdPrefixRecords {
    pub async fn connect(endpoints: Vec<String>) -> Result<Arc<Self>, PrefixError> {
        if endpoints.is_empty() {
            return Err(PrefixError("empty etcd endpoints".into()));
        }
        let client = tokio::time::timeout(Duration::from_secs(5), Client::connect(endpoints, None))
            .await
            .map_err(|_| PrefixError("connect timeout".into()))?
            .map_err(|e| PrefixError(e.to_string()))?;
        Ok(Arc::new(Self { client }))
    }
}

#[async_trait]
impl PrefixStore for EtcdPrefixRecords {
    async fn snapshot(&self) -> Result<PrefixSnapshot, PrefixError> {
        let reply = tokio::time::timeout(
            Duration::from_secs(3),
            self.client
                .clone()
                .get(REPLICA_ROOT, Some(GetOptions::new().with_prefix())),
        )
        .await
        .map_err(|_| PrefixError("snapshot timeout".into()))?
        .map_err(|e| PrefixError(e.to_string()))?;
        let revision = reply.header().map(|h| h.revision()).unwrap_or(0);
        if revision <= 0 {
            return Err(PrefixError("invalid snapshot revision".into()));
        }
        Ok(PrefixSnapshot {
            revision,
            rows: reply
                .kvs()
                .iter()
                .map(|kv| PrefixRow {
                    key: kv.key().to_vec(),
                    value: kv.value().to_vec(),
                })
                .collect(),
        })
    }

    async fn watch_from(&self, revision: i64) -> Result<PrefixStream, PrefixError> {
        if revision <= 0 {
            return Err(PrefixError("invalid watch revision".into()));
        }
        let (watcher, mut stream) = tokio::time::timeout(
            Duration::from_secs(3),
            self.client.clone().watch(
                REPLICA_ROOT,
                Some(
                    WatchOptions::new()
                        .with_prefix()
                        .with_start_revision(revision),
                ),
            ),
        )
        .await
        .map_err(|_| PrefixError("watch setup timeout".into()))?
        .map_err(|e| PrefixError(e.to_string()))?;
        Ok(Box::pin(async_stream::try_stream! {
            // Keep the watch handle alive until the consumer cancels/drops it.
            let _watcher = watcher;
            while let Some(reply) = stream.message().await.map_err(|e| PrefixError(e.to_string()))? {
                if reply.canceled() || reply.compact_revision() > 0 {
                    Err(PrefixError(format!("watch canceled or compacted at {}", reply.compact_revision())))?;
                }
                let revision = reply.header().map(|h| h.revision()).unwrap_or(0);
                if revision <= 0 {
                    Err(PrefixError("invalid watch response revision".into()))?;
                }
                let events: Vec<_> = reply.events().iter().filter_map(|event| {
                    event.kv().map(|kv| match event.event_type() {
                        EventType::Put => PrefixEvent::Put(PrefixRow {
                            key: kv.key().to_vec(), value: kv.value().to_vec(),
                        }),
                        EventType::Delete => PrefixEvent::Delete(kv.key().to_vec()),
                    })
                }).collect();
                // Watch-created/progress headers can describe a newer store
                // revision before historical events have been delivered.
                // They do not advance the mirrored record revision.
                if events.is_empty() { continue; }
                yield PrefixBatch { revision, events };
            }
            Err(PrefixError("watch stream ended".into()))?;
        }))
    }

    async fn put(&self, row: PrefixRow, ttl_ms: u64) -> Result<(), PrefixError> {
        if ttl_ms == 0 || ttl_ms > REPLICA_TTL_MS {
            return Err(PrefixError("invalid replica TTL".into()));
        }
        let mut client = self.client.clone();
        let op = async {
            let seconds = ttl_ms.div_ceil(1000) as i64;
            let lease = client
                .lease_grant(seconds, None)
                .await
                .map_err(|e| PrefixError(e.to_string()))?;
            client
                .put(
                    row.key,
                    row.value,
                    Some(PutOptions::new().with_lease(lease.id())),
                )
                .await
                .map_err(|e| PrefixError(e.to_string()))?;
            Ok(())
        };
        tokio::time::timeout(Duration::from_secs(3), op)
            .await
            .map_err(|_| PrefixError("publish timeout".into()))?
    }
}

/// Snapshot+revision watch closes the read/watch race. A reconnect replaces the
/// entire mirror, preventing deleted records surviving reconnection indefinitely.
pub fn subscribe(
    store: Arc<dyn PrefixStore>,
    cancel: CancellationToken,
) -> mpsc::Receiver<PrefixUpdate> {
    let (tx, rx) = mpsc::channel(64);
    tokio::spawn(async move {
        loop {
            let session = async {
                let snapshot = store.snapshot().await?;
                let start = snapshot
                    .revision
                    .checked_add(1)
                    .ok_or_else(|| PrefixError("revision overflow".into()))?;
                // Establish the resumable watch before exposing the snapshot.
                let mut stream = store.watch_from(start).await?;
                tx.send(PrefixUpdate::Snapshot(snapshot))
                    .await
                    .map_err(|_| PrefixError("consumer closed".into()))?;
                while let Some(batch) = stream.next().await {
                    tx.send(PrefixUpdate::Events(batch?))
                        .await
                        .map_err(|_| PrefixError("consumer closed".into()))?;
                }
                Err::<(), _>(PrefixError("watch stream ended".into()))
            };
            let error = tokio::select! {
                _ = cancel.cancelled() => break,
                result = session => result.err(),
            };
            if tx.is_closed() {
                break;
            }
            if let Some(error) = error {
                tokio::select! {
                    _ = cancel.cancelled() => break,
                    result = tx.send(PrefixUpdate::Unavailable(error.0)) => {
                        if result.is_err() { break; }
                    }
                }
            }
            tokio::select! {
                _ = cancel.cancelled() => break,
                _ = tokio::time::sleep(Duration::from_secs(1)) => {}
            }
        }
    });
    rx
}

pub fn record_key(prefix: &str, worker: &str) -> String {
    format!(
        "{REPLICA_ROOT}{prefix}/replicas/{}",
        URL_SAFE_NO_PAD.encode(worker)
    )
}

pub fn valid_replica_age_ms(confirmed_at_ms: u64, now_ms: u64, ttl_ms: u64) -> Option<u64> {
    if confirmed_at_ms == 0 || confirmed_at_ms > now_ms.saturating_add(MAX_FUTURE_SKEW_MS) {
        return None;
    }
    let age = now_ms.saturating_sub(confirmed_at_ms);
    (age <= ttl_ms).then_some(age)
}

pub fn validate_record(
    row: &PrefixRow,
    allowed_workers: &BTreeSet<String>,
    now_ms: u64,
    ttl_ms: u64,
) -> Option<ValidatedReplica> {
    let record: ReplicaRecord = serde_json::from_slice(&row.value).ok()?;
    if !allowed_workers.contains(&record.worker)
        || record.context_tokens == 0
        || record.native_incarnation.len() != 64
        || !record
            .native_incarnation
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return None;
    }
    let key = std::str::from_utf8(&row.key).ok()?;
    let (prefix, _) = key.strip_prefix(REPLICA_ROOT)?.rsplit_once("/replicas/")?;
    let (namespace, checkpoint) = prefix.split_once('/')?;
    // Namespace separates complete cache domains (including salt); checkpoint
    // hashes bind ordered message boundaries without storing any context text.
    let hex64 = |part: &str| part.len() == 64 && part.bytes().all(|b| b.is_ascii_hexdigit());
    if !hex64(namespace) || !hex64(checkpoint) || key != record_key(prefix, &record.worker) {
        return None;
    }
    let evidence = match record.evidence.as_str() {
        "local_hit" => {
            let (prompt, cached) = record.prompt_tokens.zip(record.cached_tokens)?;
            if prompt == 0 || u128::from(cached) * 10 < u128::from(prompt) * 9 {
                return None;
            }
            ReplicaEvidence::LocalHit
        }
        _ => return None,
    };
    let age_ms = valid_replica_age_ms(record.confirmed_at_ms, now_ms, ttl_ms)?;
    Some(ValidatedReplica {
        prefix: prefix.into(),
        worker: record.worker,
        confirmed_at_ms: record.confirmed_at_ms,
        age_ms,
        context_tokens: record.context_tokens,
        native_incarnation: record.native_incarnation,
        evidence,
    })
}

/// Apply this mirror and the policy under the same scheduler mutex. A replacement
/// with invalid evidence must erase the previous usable declaration at that key.
#[derive(Default)]
pub struct PrefixRecordCache {
    rows: BTreeMap<Vec<u8>, PrefixRow>,
    revision: i64,
}

impl PrefixRecordCache {
    pub fn is_available(&self) -> bool {
        self.revision > 0
    }

    pub fn apply(&mut self, update: PrefixUpdate) -> Result<(), PrefixError> {
        match update {
            PrefixUpdate::Unavailable(_) => {
                self.rows.clear();
                self.revision = 0;
            }
            PrefixUpdate::Snapshot(snapshot) => {
                if snapshot.revision <= 0 || snapshot.revision < self.revision {
                    self.rows.clear();
                    self.revision = 0;
                    return Err(PrefixError("invalid snapshot revision".into()));
                }
                self.rows = snapshot
                    .rows
                    .into_iter()
                    .map(|r| (r.key.clone(), r))
                    .collect();
                self.revision = snapshot.revision;
            }
            PrefixUpdate::Events(batch) => {
                if self.revision == 0 || batch.revision < self.revision {
                    self.rows.clear();
                    self.revision = 0;
                    return Err(PrefixError(
                        "watch revision regressed or snapshot missing".into(),
                    ));
                }
                for event in batch.events {
                    match event {
                        PrefixEvent::Put(row) => {
                            self.rows.insert(row.key.clone(), row);
                        }
                        PrefixEvent::Delete(key) => {
                            self.rows.remove(&key);
                        }
                    }
                }
                self.revision = batch.revision;
            }
        }
        Ok(())
    }

    pub fn verified(
        &self,
        allowed: &BTreeSet<String>,
        now_ms: u64,
        ttl_ms: u64,
    ) -> Vec<ValidatedReplica> {
        self.rows
            .values()
            .filter_map(|row| validate_record(row, allowed, now_ms, ttl_ms))
            .collect()
    }
}

/// Call only after a successful native terminal result with native cache-token
/// evidence; a reservation, local estimate or error is not a cache declaration.
pub async fn publish_local_hit(
    store: &dyn PrefixStore,
    allowed_workers: &BTreeSet<String>,
    prefix: &str,
    worker: &str,
    prompt_tokens: u64,
    cached_tokens: u64,
    context_tokens: u64,
    native_incarnation: &str,
    confirmed_at_ms: u64,
) -> Result<(), PrefixError> {
    let record = ReplicaRecord {
        worker: worker.into(),
        evidence: "local_hit".into(),
        confirmed_at_ms,
        context_tokens,
        native_incarnation: native_incarnation.into(),
        prompt_tokens: Some(prompt_tokens),
        cached_tokens: Some(cached_tokens),
    };
    let row = PrefixRow {
        key: record_key(prefix, worker).into_bytes(),
        value: serde_json::to_vec(&record).map_err(|e| PrefixError(e.to_string()))?,
    };
    if validate_record(&row, allowed_workers, confirmed_at_ms, REPLICA_TTL_MS).is_none() {
        return Err(PrefixError(
            "completed result has invalid worker, affinity, exact count, incarnation or local-hit ratio".into(),
        ));
    }
    store.put(row, REPLICA_TTL_MS).await
}
