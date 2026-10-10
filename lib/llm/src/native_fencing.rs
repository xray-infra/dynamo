// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One in-process authority for direct native HTTP generation. The store issues
//! fencing epochs; local deadlines only shorten the store's five-second lease.
//! This does not make native workers validate epochs or transfer their KV cache.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use etcd_client::{Client, Compare, CompareOp, PutOptions, Txn, TxnOp};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

#[path = "native_balanced_policy.rs"]
mod balanced;
#[path = "native_prefix_records.rs"]
pub mod prefix_records;
pub use prefix_records::{EtcdPrefixRecords, PrefixStore};

pub const SHORT_ADMISSION_KEY: &str = "/dsh/deepseek-v41/short-admission/state";
const LOCAL_LEASE_VALIDITY: Duration = Duration::from_secs(3);
const LEASE_HEARTBEAT: Duration = Duration::from_secs(1);
const SAMPLE_TTL: Duration = Duration::from_millis(2500);
const RESERVATION_HEARTBEAT: Duration = Duration::from_secs(5);
const RESERVATION_TTL: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum FenceError {
    #[error("native authority is standby or locally expired")]
    NotAuthority,
    #[error("native control plane unavailable: {0}")]
    Store(String),
    #[error("short admission is disabled")]
    ShortDisabled,
    #[error("short admission value or revision is invalid")]
    InvalidGate,
    #[error("all configured short workers must have fresh healthy load samples")]
    IncompleteSamples,
    #[error("native worker has no reservation capacity")]
    Saturated,
    #[error("request already has a native reservation")]
    DuplicateRequest,
    #[error("invalid fencing configuration: {0}")]
    Config(String),
}

#[derive(Debug, Clone)]
pub struct FenceConfig {
    /// Must equal the election key used by every other scheduler for this pool.
    pub election_key: String,
    pub holder: String,
    pub short_workers: Vec<String>,
    /// Published DS warm-primary/verified-replica soft threshold. Cold selection
    /// is least loaded and is not rejected at this threshold.
    pub max_effective_load: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct LeaseGrant {
    pub id: i64,
    pub epoch: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GateState {
    pub enabled: bool,
    pub commit_id: String,
    pub revision: i64,
    pub mod_revision: i64,
}

impl GateState {
    pub fn validate(&self) -> Result<(), FenceError> {
        if self.commit_id.chars().count() < 16
            || self.mod_revision <= 0
            || self.mod_revision > self.revision
        {
            return Err(FenceError::InvalidGate);
        }
        Ok(())
    }
}

/// Production uses `EtcdFenceStore`; deterministic CPU tests use their own
/// explicitly named mock. Gate reads must be linearizable, never stale caches.
#[async_trait]
pub trait FenceStore: Send + Sync {
    async fn acquire(&self, key: &str, holder: &str) -> Result<Option<LeaseGrant>, FenceError>;
    async fn keep_alive(&self, grant: LeaseGrant) -> Result<(), FenceError>;
    async fn read_short_gate(&self) -> Result<GateState, FenceError>;
}

pub struct EtcdFenceStore {
    client: Client,
}

impl EtcdFenceStore {
    pub async fn connect(endpoints: Vec<String>) -> Result<Arc<Self>, FenceError> {
        if endpoints.is_empty() {
            return Err(FenceError::Config("etcd endpoints are empty".into()));
        }
        let mut client =
            tokio::time::timeout(Duration::from_secs(5), Client::connect(endpoints, None))
                .await
                .map_err(|_| FenceError::Store("connect timeout".into()))?
                .map_err(|e| FenceError::Store(e.to_string()))?;
        tokio::time::timeout(Duration::from_secs(3), client.status())
            .await
            .map_err(|_| FenceError::Store("status timeout".into()))?
            .map_err(|e| FenceError::Store(e.to_string()))?;
        Ok(Arc::new(Self { client }))
    }
}

#[async_trait]
impl FenceStore for EtcdFenceStore {
    async fn acquire(&self, key: &str, holder: &str) -> Result<Option<LeaseGrant>, FenceError> {
        let op = async {
            let mut client = self.client.clone();
            let id = client.lease_grant(5, None).await?.id();
            let reply = client
                .txn(
                    Txn::new()
                        .when(vec![Compare::version(key, CompareOp::Equal, 0)])
                        .and_then(vec![TxnOp::put(
                            key,
                            holder,
                            Some(PutOptions::new().with_lease(id)),
                        )]),
                )
                .await?;
            if !reply.succeeded() {
                return Ok(None);
            }
            let epoch = reply.header().map(|h| h.revision()).unwrap_or(0);
            if epoch <= 0 || id == 0 {
                return Err(FenceError::Store("invalid lease epoch".into()));
            }
            Ok(Some(LeaseGrant {
                id,
                epoch: epoch as u64,
            }))
        };
        tokio::time::timeout(Duration::from_secs(3), op)
            .await
            .map_err(|_| FenceError::Store("acquisition timeout".into()))?
    }

    async fn keep_alive(&self, grant: LeaseGrant) -> Result<(), FenceError> {
        let op = async {
            let mut client = self.client.clone();
            let (mut keeper, mut stream) = client.lease_keep_alive(grant.id).await?;
            keeper.keep_alive().await?;
            let reply = stream
                .message()
                .await?
                .ok_or_else(|| FenceError::Store("keepalive stream ended".into()))?;
            if reply.id() != grant.id || reply.ttl() <= 0 {
                return Err(FenceError::Store("lease expired".into()));
            }
            Ok(())
        };
        tokio::time::timeout(Duration::from_secs(2), op)
            .await
            .map_err(|_| FenceError::Store("keepalive timeout".into()))?
    }

    async fn read_short_gate(&self) -> Result<GateState, FenceError> {
        let op = async {
            // None uses etcd's default linearizable read; do not enable serializable.
            let reply = self.client.clone().get(SHORT_ADMISSION_KEY, None).await?;
            if reply.kvs().len() != 1 {
                return Err(FenceError::InvalidGate);
            }
            #[derive(Deserialize)]
            struct GateValue {
                enabled: bool,
                commit_id: String,
            }
            let row = &reply.kvs()[0];
            let value: GateValue =
                serde_json::from_slice(row.value()).map_err(|_| FenceError::InvalidGate)?;
            let gate = GateState {
                enabled: value.enabled,
                commit_id: value.commit_id,
                revision: reply.header().map(|h| h.revision()).unwrap_or(0),
                mod_revision: row.mod_revision(),
            };
            gate.validate()?;
            Ok(gate)
        };
        tokio::time::timeout(Duration::from_secs(3), op)
            .await
            .map_err(|_| FenceError::Store("gate read timeout".into()))?
    }
}

impl From<etcd_client::Error> for FenceError {
    fn from(error: etcd_client::Error) -> Self {
        Self::Store(error.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct WorkerLoad {
    pub worker: String,
    pub healthy: bool,
    pub running: usize,
    pub waiting: usize,
    /// Complete counted IDs from the same scheduler/SHM snapshot as the counts.
    /// None (including missing/null/oversized producer metadata) fences the pool.
    pub active_request_ids: Option<Vec<String>>,
    /// Derived endpoint identity: domain-separated SHA256 of the complete ordered
    /// DP ranks and their actual scheduler startup UUIDs. Not an image or epoch.
    pub native_incarnation: String,
    /// Must also include the producer scheduler timestamp's freshness. HTTP
    /// receive time alone cannot refresh an old SHM snapshot.
    pub sampled_at: Instant,
}

struct LocalLease {
    grant: LeaseGrant,
    valid_until: Instant,
}
struct Reservation {
    id: u64,
    worker: String,
    prefix: String,
    context_tokens: u64,
    native_incarnation: String,
    weight: usize,
    expected_native_ids: ExpectedRequestIds,
    // Sparse observed choices only; no allocation proportional to requested n.
    accepted_choices: BTreeSet<u32>,
    completed_successfully: bool,
    heartbeat: Instant,
}
#[derive(Default)]
struct State {
    lease: Option<LocalLease>,
    samples: BTreeMap<String, WorkerLoad>,
    reservations: BTreeMap<String, Reservation>,
    next_reservation: u64,
    policy: balanced::BalancedPolicy,
    prefix_records: prefix_records::PrefixRecordCache,
    worker_incarnations: BTreeMap<String, String>,
}

pub struct NativeFence {
    config: FenceConfig,
    store: Arc<dyn FenceStore>,
    prefixes: Arc<dyn PrefixStore>,
    state: Mutex<State>,
    refresh: tokio::sync::Mutex<()>,
    started: std::sync::atomic::AtomicBool,
    prefixes_started: std::sync::atomic::AtomicBool,
    stop: CancellationToken,
}

impl NativeFence {
    pub fn new(
        config: FenceConfig,
        store: Arc<dyn FenceStore>,
        prefixes: Arc<dyn PrefixStore>,
    ) -> Result<Arc<Self>, FenceError> {
        let workers: BTreeSet<_> = config.short_workers.iter().collect();
        if config.election_key.is_empty()
            || config.holder.is_empty()
            || workers.is_empty()
            || workers.len() != config.short_workers.len()
            || workers.iter().any(|w| w.trim().is_empty())
            || config.max_effective_load == 0
        {
            return Err(FenceError::Config(
                "holder/key and a nonempty distinct configured short worker set required".into(),
            ));
        }
        Ok(Arc::new(Self {
            config,
            store,
            prefixes,
            state: Mutex::new(State::default()),
            refresh: tokio::sync::Mutex::new(()),
            started: std::sync::atomic::AtomicBool::new(false),
            prefixes_started: std::sync::atomic::AtomicBool::new(false),
            stop: CancellationToken::new(),
        }))
    }

    /// One lease heartbeat attempt. Public for deterministic control-plane tests.
    /// Concurrent calls are serialized with the background heartbeat.
    pub async fn refresh_authority(&self) -> Result<(), FenceError> {
        let _refresh = self.refresh.lock().await;
        if self.stop.is_cancelled() {
            return Err(FenceError::NotAuthority);
        }
        let started = Instant::now();
        let current = self.state.lock().unwrap().lease.as_ref().map(|l| l.grant);
        let result = async {
            let grant = match current {
                Some(grant) => grant,
                None => self
                    .store
                    .acquire(&self.config.election_key, &self.config.holder)
                    .await?
                    .ok_or(FenceError::NotAuthority)?,
            };
            if grant.id == 0 || grant.epoch == 0 {
                return Err(FenceError::Store("invalid lease grant".into()));
            }
            self.store.keep_alive(grant).await?;
            if Instant::now() >= started + LOCAL_LEASE_VALIDITY {
                return Err(FenceError::NotAuthority);
            }
            Ok(LocalLease {
                grant,
                valid_until: started + LOCAL_LEASE_VALIDITY,
            })
        }
        .await;
        let mut state = self.state.lock().unwrap();
        if self.stop.is_cancelled() {
            state.lease = None;
            return Err(FenceError::NotAuthority);
        }
        match result {
            Ok(lease) => {
                state.lease = Some(lease);
                Ok(())
            }
            Err(error) => {
                state.lease = None;
                Err(error)
            }
        }
    }

    pub fn start(self: &Arc<Self>) {
        self.start_prefixes();
        if self.started.swap(true, std::sync::atomic::Ordering::AcqRel) {
            return;
        }
        let weak = Arc::downgrade(self);
        let stop = self.stop.clone();
        tokio::spawn(async move {
            loop {
                let Some(fence) = weak.upgrade() else {
                    return;
                };
                tokio::select! { _ = stop.cancelled() => return, _ = fence.refresh_authority() => {} }
                drop(fence);
                tokio::select! { _ = stop.cancelled() => return, _ = tokio::time::sleep(LEASE_HEARTBEAT) => {} }
            }
        });
    }

    pub fn start_prefixes(self: &Arc<Self>) {
        if self
            .prefixes_started
            .swap(true, std::sync::atomic::Ordering::AcqRel)
        {
            return;
        }
        let mut updates = prefix_records::subscribe(self.prefixes.clone(), self.stop.clone());
        let weak = Arc::downgrade(self);
        let stop = self.stop.clone();
        tokio::spawn(async move {
            loop {
                let update = tokio::select! { _ = stop.cancelled() => return, update = updates.recv() => update };
                let Some(update) = update else {
                    return;
                };
                let Some(fence) = weak.upgrade() else {
                    return;
                };
                let mut state = fence.state.lock().unwrap();
                let _ = state.prefix_records.apply(update);
                refresh_verified(&mut state, &fence.config);
            }
        });
    }

    pub fn prefixes_ready(&self) -> bool {
        self.state.lock().unwrap().prefix_records.is_available()
    }

    /// Self-fence immediately and stop renewing; the store's old lease expires
    /// naturally. No manual leader-key deletion or unconditional lease handoff.
    pub fn stop(&self) {
        self.stop.cancel();
        self.state.lock().unwrap().lease = None;
    }

    pub fn check_generation(&self) -> Result<u64, FenceError> {
        let state = self.state.lock().unwrap();
        valid_epoch(&state, Instant::now())
    }

    pub fn update_samples(&self, samples: Vec<WorkerLoad>) -> Result<(), FenceError> {
        let now = Instant::now();
        let sample_count = samples.len();
        let samples: BTreeMap<_, _> = samples.into_iter().map(|s| (s.worker.clone(), s)).collect();
        let mut state = self.state.lock().unwrap();
        if sample_count != self.config.short_workers.len()
            || samples.len() != self.config.short_workers.len()
            || self.config.short_workers.iter().any(|w| {
                samples.get(w).is_none_or(|s| {
                    !valid_sample(s)
                        || s.sampled_at > now
                        || now.duration_since(s.sampled_at) > SAMPLE_TTL
                })
            })
        {
            state.samples.clear();
            return Err(FenceError::IncompleteSamples);
        }
        if state.reservations.values().any(|r| {
            samples
                .get(&r.worker)
                .is_some_and(|s| matched_count(r, s) > r.weight)
        }) {
            state.samples.clear();
            return Err(FenceError::IncompleteSamples);
        }
        let changed: Vec<_> = samples
            .iter()
            .filter(|(worker, sample)| {
                state
                    .worker_incarnations
                    .get(*worker)
                    .is_some_and(|old| *old != sample.native_incarnation)
            })
            .map(|(worker, _)| worker.clone())
            .collect();
        for worker in changed {
            state.policy.clear_worker(&worker);
        }
        for reservation in state.reservations.values_mut() {
            let sample = &samples[&reservation.worker];
            if sample.native_incarnation != reservation.native_incarnation {
                reservation.accepted_choices.clear();
                continue;
            }
            for id in sample.active_request_ids.as_ref().unwrap() {
                if let Some(NativeRequestPhase::Sample(choice)) =
                    reservation.expected_native_ids.phase(id)
                {
                    reservation.accepted_choices.insert(choice);
                }
            }
        }
        state.worker_incarnations = samples
            .iter()
            .map(|(worker, sample)| (worker.clone(), sample.native_incarnation.clone()))
            .collect();
        state.samples = samples;
        refresh_verified(&mut state, &self.config);
        Ok(())
    }

    pub fn is_ready(&self) -> bool {
        let state = self.state.lock().unwrap();
        let now = Instant::now();
        valid_epoch(&state, now).is_ok()
            && samples_ready(&state, &self.config, now)
            && state.prefix_records.is_available()
    }

    pub fn effective_load(&self, worker: &str) -> Option<usize> {
        let mut state = self.state.lock().unwrap();
        let now = Instant::now();
        reap(&mut state, now);
        if !samples_ready(&state, &self.config, now) {
            return None;
        }
        load(&state, worker)
    }

    /// Exclusions are per-request confirmed pre-send connection failures. The
    /// caller must drop the previous permit before retrying the same owned RID;
    /// this never invalidates shared samples or relaxes a known warm set.
    pub async fn take_short_permit(
        self: &Arc<Self>,
        request_id: &str,
        affinity: &NativeAffinity,
        expected_native_ids: ExpectedRequestIds,
        context_tokens: u64,
        excluded_workers: &BTreeSet<String>,
    ) -> Result<GenerationPermit, FenceError> {
        if request_id.is_empty()
            || affinity.checkpoints.is_empty()
            || !valid_hash(&affinity.namespace)
            || affinity.checkpoints.iter().any(|key| {
                key.split_once('/').is_none_or(|(namespace, checkpoint)| {
                    namespace != affinity.namespace || !valid_hash(checkpoint)
                })
            })
            || context_tokens == 0
            || expected_native_ids.n == 0
            || expected_native_ids.rid != request_id
            || excluded_workers.len() > self.config.short_workers.len()
            || excluded_workers
                .iter()
                .any(|worker| !self.config.short_workers.contains(worker))
        {
            return Err(FenceError::Config(
                "owned request ID, isolated prefix and exact native fanout IDs required".into(),
            ));
        }
        let weight = usize::try_from(expected_native_ids.n)
            .map_err(|_| FenceError::Config("n does not fit target".into()))?;
        self.check_generation()?;
        let gate = self.store.read_short_gate().await?;
        gate.validate()?;
        if !gate.enabled {
            return Err(FenceError::ShortDisabled);
        }
        let now = Instant::now();
        let mut state = self.state.lock().unwrap();
        let epoch = valid_epoch(&state, now)?;
        reap(&mut state, now);
        if !samples_ready(&state, &self.config, now) {
            return Err(FenceError::IncompleteSamples);
        }
        if state.reservations.contains_key(request_id) {
            return Err(FenceError::DuplicateRequest);
        }
        if !state.prefix_records.is_available() {
            return Err(FenceError::Store(
                "shared prefix metadata unavailable".into(),
            ));
        }
        let loads: BTreeMap<_, _> = state
            .samples
            .keys()
            .filter_map(|worker| load(&state, worker).map(|value| (worker.clone(), value)))
            .collect();
        let prefix = affinity
            .checkpoints
            .iter()
            .rev()
            .find(|key| state.policy.has_affinity(key, now, context_tokens))
            .unwrap_or_else(|| affinity.checkpoints.last().unwrap());
        let worker = state
            .policy
            .select(
                prefix,
                &loads,
                now,
                self.config.max_effective_load,
                context_tokens,
                excluded_workers,
            )
            .ok_or(FenceError::Saturated)?;
        if state.samples.get(&worker).is_some_and(|sample| {
            sample
                .active_request_ids
                .as_ref()
                .into_iter()
                .flatten()
                .filter(|id| expected_native_ids.matches(id))
                .count()
                > weight
        }) {
            state.samples.clear();
            return Err(FenceError::IncompleteSamples);
        }
        state.next_reservation = state
            .next_reservation
            .checked_add(1)
            .ok_or_else(|| FenceError::Config("reservation sequence exhausted".into()))?;
        let reservation_id = state.next_reservation;
        let native_incarnation = state
            .samples
            .get(&worker)
            .unwrap()
            .native_incarnation
            .clone();
        state.reservations.insert(
            request_id.to_owned(),
            Reservation {
                id: reservation_id,
                worker: worker.clone(),
                prefix: affinity.checkpoints.last().unwrap().clone(),
                context_tokens,
                native_incarnation,
                weight,
                expected_native_ids,
                accepted_choices: BTreeSet::new(),
                completed_successfully: false,
                heartbeat: now,
            },
        );
        drop(state);
        let stop = CancellationToken::new();
        let weak = Arc::downgrade(self);
        let id = request_id.to_owned();
        let heartbeat_stop = stop.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! { _ = heartbeat_stop.cancelled() => return, _ = tokio::time::sleep(RESERVATION_HEARTBEAT) => {} }
                let Some(fence) = weak.upgrade() else {
                    return;
                };
                let mut state = fence.state.lock().unwrap();
                let Some(reservation) = state.reservations.get_mut(&id) else {
                    return;
                };
                if reservation.id != reservation_id {
                    return;
                }
                reservation.heartbeat = Instant::now();
            }
        });
        Ok(GenerationPermit {
            fence: self.clone(),
            request_id: request_id.into(),
            reservation_id,
            worker,
            epoch,
            gate_revision: gate.revision,
            stop,
        })
    }

    pub fn reservation_count(&self) -> usize {
        let mut state = self.state.lock().unwrap();
        reap(&mut state, Instant::now());
        state.reservations.len()
    }
}

fn valid_epoch(state: &State, now: Instant) -> Result<u64, FenceError> {
    state
        .lease
        .as_ref()
        .filter(|l| now < l.valid_until)
        .map(|l| l.grant.epoch)
        .ok_or(FenceError::NotAuthority)
}
fn samples_ready(state: &State, config: &FenceConfig, now: Instant) -> bool {
    state.samples.len() == config.short_workers.len()
        && config.short_workers.iter().all(|worker| {
            state.samples.get(worker).is_some_and(|s| {
                valid_sample(s)
                    && s.sampled_at <= now
                    && now.duration_since(s.sampled_at) <= SAMPLE_TTL
            })
        })
}
fn reap(state: &mut State, now: Instant) {
    state
        .reservations
        .retain(|_, r| now.duration_since(r.heartbeat) < RESERVATION_TTL);
}
fn load(state: &State, worker: &str) -> Option<usize> {
    let sample = state.samples.get(worker)?;
    let reserved = state
        .reservations
        .values()
        .filter(|r| r.worker == worker)
        .map(|r| {
            if sample.native_incarnation != r.native_incarnation {
                return r.weight;
            }
            let live_prefill = !r.accepted_choices.contains(&0)
                && sample.active_request_ids.as_ref().is_some_and(|ids| {
                    ids.iter().any(|id| {
                        r.expected_native_ids.phase(id) == Some(NativeRequestPhase::Prefill)
                    })
                });
            r.weight
                .saturating_sub(r.accepted_choices.len())
                .saturating_sub(usize::from(live_prefill))
        })
        .fold(0usize, usize::saturating_add);
    Some(
        sample
            .running
            .saturating_add(sample.waiting)
            .saturating_add(reserved),
    )
}

fn refresh_verified(state: &mut State, config: &FenceConfig) {
    let allowed = config.short_workers.iter().cloned().collect();
    let rows = state
        .prefix_records
        .verified(&allowed, unix_now_ms(), prefix_records::REPLICA_TTL_MS)
        .into_iter()
        .filter(|row| {
            state
                .samples
                .get(&row.worker)
                .is_some_and(|sample| sample.native_incarnation == row.native_incarnation)
        })
        .collect();
    state.policy.replace_verified(rows, Instant::now());
}

fn unix_now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

fn valid_sample(sample: &WorkerLoad) -> bool {
    let Some(ids) = &sample.active_request_ids else {
        return false;
    };
    sample.healthy
        && valid_hash(&sample.native_incarnation)
        && sample.running.checked_add(sample.waiting) == Some(ids.len())
        && ids.iter().all(|id| !id.is_empty())
        && ids.iter().collect::<BTreeSet<_>>().len() == ids.len()
}
fn matched_count(reservation: &Reservation, sample: &WorkerLoad) -> usize {
    if reservation.native_incarnation != sample.native_incarnation {
        return 0;
    }
    sample
        .active_request_ids
        .as_ref()
        .into_iter()
        .flatten()
        .filter(|id| reservation.expected_native_ids.matches(id))
        .count()
}

/// Compact paired-native scalar Chat identity grammar. Matching iterates only
/// actual snapshot IDs, never allocates an expected vector of n IDs.
#[derive(Debug, Clone)]
pub struct ExpectedRequestIds {
    pub rid: String,
    pub n: u32,
}

/// Exact native identity phase. Prefill is temporary occupancy and never a
/// persistently accepted completion choice, including choice zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeRequestPhase {
    Prefill,
    Sample(u32),
}

/// Request-derived routing hints. These hashes identify template/cache domains
/// and completed message boundaries; they do not assert native KV presence.
#[derive(Debug, Clone)]
pub struct NativeAffinity {
    pub namespace: String,
    pub checkpoints: Vec<String>,
}

pub fn affinity_for_chat(
    body: &Value,
    qualified_identity: &str,
) -> Result<NativeAffinity, FenceError> {
    if qualified_identity.is_empty() {
        return Err(FenceError::Config(
            "qualified native/model/template identity required".into(),
        ));
    }
    let object = body
        .as_object()
        .ok_or_else(|| FenceError::Config("affinity requires normalized Chat object".into()))?;
    let messages = object
        .get("messages")
        .and_then(Value::as_array)
        .filter(|m| !m.is_empty())
        .ok_or_else(|| FenceError::Config("affinity requires ordered nonempty messages".into()))?;
    let excluded: BTreeSet<&str> = [
        "messages",
        "rid",
        "n",
        "max_tokens",
        "max_completion_tokens",
        "max_output_tokens",
        "stream",
        "stream_options",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
        "seed",
        "stop",
        "logprobs",
        "top_logprobs",
        "logit_bias",
        "ignore_eos",
        "min_tokens",
    ]
    .into_iter()
    .collect();
    let mut metadata: serde_json::Map<String, Value> = object
        .iter()
        .filter(|(key, _)| !excluded.contains(key.as_str()))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    metadata.insert(
        "qualified_identity".into(),
        Value::String(qualified_identity.into()),
    );
    let namespace = format!(
        "{:x}",
        Sha256::digest(
            serde_json::to_vec(&canonical_json(&Value::Object(metadata)))
                .map_err(|e| FenceError::Config(e.to_string()))?
        )
    );
    let mut chain = Sha256::new();
    let mut checkpoints = Vec::with_capacity(messages.len());
    for message in messages {
        let encoded = serde_json::to_vec(&canonical_json(message))
            .map_err(|e| FenceError::Config(e.to_string()))?;
        chain.update((encoded.len() as u64).to_be_bytes());
        chain.update(encoded);
        checkpoints.push(format!("{namespace}/{:x}", chain.clone().finalize()));
    }
    Ok(NativeAffinity {
        namespace,
        checkpoints,
    })
}
fn valid_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
fn canonical_json(value: &Value) -> Value {
    match value {
        Value::Object(object) => Value::Object(
            object
                .iter()
                .map(|(key, value)| (key.clone(), canonical_json(value)))
                .collect::<BTreeMap<_, _>>()
                .into_iter()
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.iter().map(canonical_json).collect()),
        _ => value.clone(),
    }
}
impl ExpectedRequestIds {
    pub fn new(rid: impl Into<String>, n: u32) -> Result<Self, FenceError> {
        let rid = rid.into();
        if rid.is_empty() || n == 0 {
            return Err(FenceError::Config(
                "owned rid and positive u32 n required".into(),
            ));
        }
        Ok(Self { rid, n })
    }
    pub fn matches(&self, id: &str) -> bool {
        self.phase(id).is_some()
    }
    pub fn phase(&self, id: &str) -> Option<NativeRequestPhase> {
        if self.n == 1 {
            return (id == self.rid).then_some(NativeRequestPhase::Sample(0));
        }
        if self.n == 0 {
            return None;
        }
        let suffix = id.strip_prefix(self.rid.as_str())?;
        if suffix == "_0:prefill" {
            return Some(NativeRequestPhase::Prefill);
        }
        let Some(index) = suffix.strip_prefix("_0:sample:") else {
            return None;
        };
        if index.is_empty()
            || (index != "0" && index.starts_with('0'))
            || !index.bytes().all(|byte| byte.is_ascii_digit())
        {
            return None;
        }
        index
            .parse::<u32>()
            .ok()
            .filter(|choice| *choice < self.n)
            .map(NativeRequestPhase::Sample)
    }
}

/// Hold through the response stream lifetime; drop releases synchronously on
/// success, error, cancellation or disconnect, including before headers arrive.
pub struct GenerationPermit {
    fence: Arc<NativeFence>,
    request_id: String,
    reservation_id: u64,
    pub worker: String,
    pub epoch: u64,
    pub gate_revision: i64,
    stop: CancellationToken,
}

impl GenerationPermit {
    /// Call only on successful native DONE or complete validated JSON. Earlier
    /// content, headers, usage, disconnects and errors do not establish affinity.
    pub fn complete_success(&self, prompt_tokens: u64, cached_tokens: Option<u64>) {
        let now = Instant::now();
        let mut state = self.fence.state.lock().unwrap();
        let Some(reservation) = state.reservations.get_mut(&self.request_id) else {
            return;
        };
        if reservation.id != self.reservation_id || reservation.completed_successfully {
            return;
        }
        reservation.completed_successfully = true;
        let prefix = reservation.prefix.clone();
        let worker = reservation.worker.clone();
        let context_tokens = reservation.context_tokens;
        let native_incarnation = reservation.native_incarnation.clone();
        if state.worker_incarnations.get(&worker) != Some(&native_incarnation) {
            return;
        }
        state.policy.complete_success(
            &prefix,
            &worker,
            now,
            context_tokens,
            prompt_tokens,
            cached_tokens,
        );
        drop(state);
        if let Some(cached) = cached_tokens.filter(|cached| {
            prompt_tokens > 0
                && *cached <= prompt_tokens
                && u128::from(*cached) * 10 >= u128::from(prompt_tokens) * 9
        }) {
            let store = self.fence.prefixes.clone();
            let allowed = self.fence.config.short_workers.iter().cloned().collect();
            tokio::spawn(async move {
                let _ = prefix_records::publish_local_hit(
                    store.as_ref(),
                    &allowed,
                    &prefix,
                    &worker,
                    prompt_tokens,
                    cached,
                    context_tokens,
                    &native_incarnation,
                    unix_now_ms(),
                )
                .await;
            });
        }
    }
}

impl Drop for GenerationPermit {
    fn drop(&mut self) {
        self.stop.cancel();
        let mut state = self.fence.state.lock().unwrap();
        if state
            .reservations
            .get(&self.request_id)
            .is_some_and(|r| r.id == self.reservation_id)
        {
            state.reservations.remove(&self.request_id);
        }
    }
}
