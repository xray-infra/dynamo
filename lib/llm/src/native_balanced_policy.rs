// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Selector semantics from the published DS exact-07c8 balanced policy:
//! preserve an available primary, prefer a less loaded verified replica, reject
//! overflow from a known warm set, and leave cold least-load selection uncapped.
//! No KV presence is inferred from a load sample or a request ID.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;
use tokio::time::Instant;

const REPLICA_TTL: Duration = Duration::from_secs(300);

#[derive(Default)]
pub(super) struct BalancedPolicy {
    prefixes: BTreeMap<String, Prefix>,
}
#[derive(Default)]
struct Prefix {
    primary: Option<(String, Instant, u64)>,
    verified: BTreeMap<String, (Instant, u64)>,
}

impl BalancedPolicy {
    pub fn clear_worker(&mut self, worker: &str) {
        for entry in self.prefixes.values_mut() {
            if entry
                .primary
                .as_ref()
                .is_some_and(|(primary, _, _)| primary == worker)
            {
                entry.primary = None;
            }
            entry.verified.remove(worker);
        }
    }
    pub fn replace_verified(
        &mut self,
        rows: Vec<super::prefix_records::ValidatedReplica>,
        now: Instant,
    ) {
        for prefix in self.prefixes.values_mut() {
            prefix.verified.clear();
        }
        for row in rows {
            let at = now
                .checked_sub(Duration::from_millis(row.age_ms))
                .unwrap_or(now);
            self.prefixes
                .entry(row.prefix)
                .or_default()
                .verified
                .insert(row.worker, (at, row.context_tokens));
        }
    }
    pub fn has_affinity(&self, prefix: &str, now: Instant, context_tokens: u64) -> bool {
        self.prefixes.get(prefix).is_some_and(|entry| {
            entry
                .primary
                .as_ref()
                .is_some_and(|(_, at, tokens)| eligible(*at, *tokens, now, context_tokens))
                || entry
                    .verified
                    .values()
                    .any(|(at, tokens)| eligible(*at, *tokens, now, context_tokens))
        })
    }
    pub fn select(
        &self,
        prefix: &str,
        loads: &BTreeMap<String, usize>,
        now: Instant,
        warm_soft_threshold: usize,
        context_tokens: u64,
        excluded_workers: &BTreeSet<String>,
    ) -> Option<String> {
        let affinity = self.prefixes.get(prefix);
        let primary = affinity
            .and_then(|p| p.primary.as_ref())
            .filter(|(_, at, tokens)| eligible(*at, *tokens, now, context_tokens))
            .filter(|(worker, _, _)| loads.contains_key(worker));
        let replicas: Vec<_> = affinity
            .into_iter()
            .flat_map(|p| p.verified.iter())
            .filter(|(_, (at, tokens))| eligible(*at, *tokens, now, context_tokens))
            .filter(|(worker, _)| loads.contains_key(*worker))
            .collect();
        let available_primary = primary
            .filter(|(worker, _, _)| !excluded_workers.contains(worker))
            .and_then(|(worker, _, _)| {
                loads
                    .get(worker)
                    .filter(|load| **load < warm_soft_threshold)
                    .map(|load| (worker, *load))
            });
        let replica = replicas
            .iter()
            .filter(|(worker, _)| !excluded_workers.contains(*worker))
            .filter(|(worker, _)| primary.is_none_or(|(p, _, _)| p != *worker))
            .filter_map(|(worker, _)| {
                loads
                    .get(*worker)
                    .filter(|load| **load < warm_soft_threshold)
                    .map(|load| (*worker, *load))
            })
            .min_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(b.0)));
        match (available_primary, replica) {
            (Some((_, primary_load)), Some((worker, replica_load)))
                if replica_load < primary_load =>
            {
                Some(worker.clone())
            }
            (Some((worker, _)), _) => Some(worker.clone()),
            (None, Some((worker, _))) => Some(worker.clone()),
            (None, None) if primary.is_some() || !replicas.is_empty() => None,
            (None, None) => loads
                .iter()
                .filter(|(worker, _)| !excluded_workers.contains(*worker))
                .min_by(|a, b| a.1.cmp(b.1).then_with(|| a.0.cmp(b.0)))
                .map(|(worker, _)| worker.clone()),
        }
    }

    pub fn complete_success(
        &mut self,
        prefix: &str,
        worker: &str,
        now: Instant,
        context_tokens: u64,
        prompt_tokens: u64,
        cached_tokens: Option<u64>,
    ) {
        // A successful native completion establishes the primary. A verified
        // replica additionally requires actual native usage at >=90% cache hit.
        let entry = self.prefixes.entry(prefix.into()).or_default();
        entry.primary = Some((worker.into(), now, context_tokens));
        if cached_tokens.is_some_and(|cached| {
            prompt_tokens > 0
                && cached <= prompt_tokens
                && (cached as u128) * 10 >= (prompt_tokens as u128) * 9
        }) {
            entry.verified.insert(worker.into(), (now, context_tokens));
        }
    }
}

fn eligible(at: Instant, tokens: u64, now: Instant, current: u64) -> bool {
    at <= now
        && now.duration_since(at) <= REPLICA_TTL
        && tokens > 0
        && current > 0
        && (tokens as u128) * 10 > (current as u128) * 3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_selector_primary_tie_and_lower_verified_replica() {
        let now = Instant::now();
        let mut policy = BalancedPolicy::default();
        policy.complete_success("prefix", "replica", now, 100, 100, Some(95));
        policy.complete_success("prefix", "primary", now, 100, 100, None);
        let mut loads = BTreeMap::from([
            ("primary".into(), 1),
            ("replica".into(), 1),
            ("cold".into(), 0),
        ]);
        assert_eq!(
            policy.select("prefix", &loads, now, 2, 100, &BTreeSet::new()),
            Some("primary".into())
        );
        loads.insert("replica".into(), 0);
        assert_eq!(
            policy.select("prefix", &loads, now, 2, 100, &BTreeSet::new()),
            Some("replica".into())
        );
        loads.insert("primary".into(), 2);
        loads.insert("replica".into(), 2);
        assert_eq!(
            policy.select("prefix", &loads, now, 2, 100, &BTreeSet::new()),
            None,
            "known warm set may not spill to unrelated cold worker"
        );
    }

    #[test]
    fn ids_selector_strict_thirty_percent_eligibility_and_uncapped_cold() {
        let now = Instant::now();
        let mut policy = BalancedPolicy::default();
        policy.complete_success("prefix", "primary", now, 30, 30, None);
        assert!(!policy.has_affinity("prefix", now, 100));
        assert!(policy.has_affinity("prefix", now, 99));
        let loads = BTreeMap::from([("primary".into(), 32), ("cold".into(), 33)]);
        assert_eq!(
            policy.select("unknown", &loads, now, 2, 100, &BTreeSet::new()),
            Some("primary".into())
        );
        assert_eq!(
            policy.select("prefix", &loads, now, 2, 100, &BTreeSet::new()),
            Some("primary".into())
        );
        assert_eq!(
            policy.select("prefix", &loads, now, 2, 99, &BTreeSet::new()),
            None
        );
    }

    #[test]
    fn ids_selector_cache_ratio_uses_exact_u128_and_restart_clears_all_hints() {
        let now = Instant::now();
        let mut policy = BalancedPolicy::default();
        policy.complete_success(
            "prefix",
            "unverified",
            now,
            100,
            u64::MAX,
            Some(u64::MAX / 2),
        );
        policy.complete_success("prefix", "primary", now, 100, 100, None);
        let loads = BTreeMap::from([("primary".into(), 2), ("unverified".into(), 0)]);
        assert_eq!(
            policy.select("prefix", &loads, now, 2, 100, &BTreeSet::new()),
            None
        );
        policy.clear_worker("primary");
        assert_eq!(
            policy.select("prefix", &loads, now, 2, 100, &BTreeSet::new()),
            Some("unverified".into()),
            "after restart there is no warm proof; this is cold selection"
        );
    }
}
