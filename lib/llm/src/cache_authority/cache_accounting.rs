//! Accounting consumes only native coverage already checked by the parent
//! HMAC/all-rank verifier. Published prefix lengths are never fresh-write proof.
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct VerifiedCacheAccounting {
    pub input_tokens: u64,
    pub cache_read_input_tokens: u64,
    pub cache_creation_5m_input_tokens: u64,
    pub cache_creation_1h_input_tokens: u64,
}
impl VerifiedCacheAccounting {
    pub fn validate(&self, total_prompt_tokens: u64) -> Result<(), super::Error> {
        let sum = self
            .input_tokens
            .checked_add(self.cache_read_input_tokens)
            .and_then(|sum| sum.checked_add(self.cache_creation_5m_input_tokens))
            .and_then(|sum| sum.checked_add(self.cache_creation_1h_input_tokens));
        if sum != Some(total_prompt_tokens) {
            return Err(super::Error::Evidence(
                "verified cache accounting does not equal exact prompt count",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VerifiedEventCoverage {
    pub(crate) event_id: String,
    pub(crate) endpoint_id: String,
    pub(crate) ttl_ms: u64,
    pub(crate) read_tokens: u64,
    /// Explicit native fresh token intervals, half-open. Empty means native
    /// explicitly proved no fresh writes; omitted evidence never constructs this.
    pub(crate) fresh_write_ranges: Vec<(u64, u64)>,
    pub(crate) terminal_no_write: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct CoverageScope {
    pub(crate) endpoint_id: String,
    pub(crate) ttl_ms: u64,
    pub(crate) retained_tokens: u64,
    pub(crate) write: bool,
    pub(crate) read: bool,
}

pub(crate) struct CacheAccounting {
    total: u64,
    zero_plan: bool,
    scopes: BTreeMap<String, CoverageScope>,
    events: BTreeMap<String, VerifiedEventCoverage>,
    invalid: bool,
}

impl CacheAccounting {
    pub(crate) fn new(
        total_count: u64,
        zero_plan: bool,
        scopes: Vec<CoverageScope>,
    ) -> Result<Self, &'static str> {
        let mut map = BTreeMap::new();
        for scope in scopes {
            if scope.endpoint_id.is_empty()
                || !matches!(scope.ttl_ms, 300_000 | 3_600_000)
                || scope.retained_tokens > total_count
                || (!scope.write && !scope.read)
            {
                return Err("invalid verified accounting scope");
            }
            if map.insert(scope.endpoint_id.clone(), scope).is_some() {
                return Err("duplicate accounting endpoint");
            }
        }
        let zero_plan = zero_plan || map.is_empty();
        Ok(Self {
            total: total_count,
            zero_plan,
            scopes: map,
            events: BTreeMap::new(),
            invalid: false,
        })
    }

    pub(crate) fn observe(&mut self, coverage: VerifiedEventCoverage) -> Result<(), &'static str> {
        let valid = (|| {
            if self.zero_plan || coverage.event_id.is_empty() {
                return Err("no-pin plan cannot report cache coverage");
            }
            if let Some(previous) = self.events.get(&coverage.event_id) {
                return if previous == &coverage {
                    Ok(())
                } else {
                    Err("cache event ID changed coverage")
                };
            }
            let scope = self
                .scopes
                .get(&coverage.endpoint_id)
                .ok_or("cache coverage outside authorized endpoint scope")?;
            if coverage.ttl_ms != scope.ttl_ms
                || coverage.read_tokens > scope.retained_tokens
                || (!scope.read && coverage.read_tokens != 0)
                || (!scope.write && !coverage.fresh_write_ranges.is_empty())
            {
                return Err("cache coverage differs from verified native scope");
            }
            for &(start, end) in &coverage.fresh_write_ranges {
                if start >= end || end > scope.retained_tokens {
                    return Err("fresh native coverage outside retained page range");
                }
            }
            if coverage.terminal_no_write && (coverage.read_tokens!=0 || !coverage.fresh_write_ranges.is_empty()) {
                return Err("terminal no-write cannot report read or creation");
            }
            if self.events.values().filter(|e|e.endpoint_id==coverage.endpoint_id).any(|e|
                (coverage.terminal_no_write && !e.fresh_write_ranges.is_empty()) || (e.terminal_no_write && !coverage.fresh_write_ranges.is_empty())
                || (e.terminal_no_write && coverage.read_tokens!=0) || (coverage.terminal_no_write && e.read_tokens!=0)) {
                return Err("terminal no-write conflicts with completed cache coverage");
            }
            Ok(())
        })();
        if valid.is_err() {
            self.invalid = true;
            return valid;
        }
        self.events
            .entry(coverage.event_id.clone())
            .or_insert(coverage);
        Ok(())
    }

    pub(crate) fn finish(&self) -> Option<VerifiedCacheAccounting> {
        if self.invalid {
            return None;
        }
        if self.zero_plan {
            return Some(VerifiedCacheAccounting {
                input_tokens: self.total,
                cache_read_input_tokens: 0,
                cache_creation_5m_input_tokens: 0,
                cache_creation_1h_input_tokens: 0,
            });
        }
        if self.events.is_empty()
            || self.scopes.values().any(|scope| {
                !self
                    .events
                    .values()
                    .any(|event| event.endpoint_id == scope.endpoint_id)
            })
        {
            return None;
        }
        let read = self
            .events
            .values()
            .map(|event| event.read_tokens)
            .max()
            .unwrap_or(0);
        let mut five = Vec::new();
        let mut hour = Vec::new();
        for event in self.events.values() {
            for &(start, end) in &event.fresh_write_ranges {
                let start = start.max(read);
                if start < end {
                    if event.ttl_ms == 3_600_000 {
                        hour.push((start, end));
                    } else {
                        five.push((start, end));
                    }
                }
            }
        }
        let hour = union(hour);
        let five = union(five);
        let hour_count = length(&hour);
        let five_count = length(&five).checked_sub(intersection_length(&five, &hour))?;
        let input = self
            .total
            .checked_sub(read)?
            .checked_sub(hour_count)?
            .checked_sub(five_count)?;
        Some(VerifiedCacheAccounting {
            input_tokens: input,
            cache_read_input_tokens: read,
            cache_creation_5m_input_tokens: five_count,
            cache_creation_1h_input_tokens: hour_count,
        })
    }
}

fn union(mut ranges: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    ranges.sort_unstable();
    let mut merged: Vec<(u64, u64)> = Vec::new();
    for (start, end) in ranges {
        if let Some(last) = merged.last_mut() {
            if start <= last.1 {
                last.1 = last.1.max(end);
                continue;
            }
        }
        merged.push((start, end));
    }
    merged
}
fn length(ranges: &[(u64, u64)]) -> u64 {
    ranges.iter().map(|&(start, end)| end - start).sum()
}
fn intersection_length(a: &[(u64, u64)], b: &[(u64, u64)]) -> u64 {
    let (mut i, mut j, mut total) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        let start = a[i].0.max(b[j].0);
        let end = a[i].1.min(b[j].1);
        if start < end {
            total += end - start;
        }
        if a[i].1 <= b[j].1 {
            i += 1;
        } else {
            j += 1;
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    fn scope(id: &str, ttl: u64, write: bool, read: bool) -> CoverageScope {
        CoverageScope {
            endpoint_id: id.into(),
            ttl_ms: ttl,
            retained_tokens: 100,
            write,
            read,
        }
    }
    fn event(
        id: &str,
        endpoint: &str,
        ttl: u64,
        read: u64,
        ranges: Vec<(u64, u64)>,
    ) -> VerifiedEventCoverage {
        VerifiedEventCoverage {
            event_id: id.into(),
            endpoint_id: endpoint.into(),
            ttl_ms: ttl,
            read_tokens: read,
            fresh_write_ranges: ranges,
            terminal_no_write: false,
        }
    }
    #[test]
    fn accounting_overlap_read_and_idempotency() {
        let mut a = CacheAccounting::new(
            120,
            false,
            vec![
                scope("five", 300_000, true, false),
                scope("hour", 3_600_000, true, false),
                scope("read", 300_000, false, true),
            ],
        )
        .unwrap();
        let five = event("e1", "five", 300_000, 0, vec![(0, 60), (40, 80)]);
        a.observe(five.clone()).unwrap();
        a.observe(five).unwrap();
        a.observe(event("e2", "hour", 3_600_000, 0, vec![(50, 100)]))
            .unwrap();
        a.observe(event("e3", "read", 300_000, 30, vec![])).unwrap();
        assert_eq!(
            a.finish(),
            Some(VerifiedCacheAccounting {
                input_tokens: 20,
                cache_read_input_tokens: 30,
                cache_creation_5m_input_tokens: 20,
                cache_creation_1h_input_tokens: 50
            })
        );
        a.finish().unwrap().validate(120).unwrap();
        assert!(a.finish().unwrap().validate(121).is_err());
        assert!(
            VerifiedCacheAccounting {
                input_tokens: u64::MAX,
                cache_read_input_tokens: 1,
                cache_creation_5m_input_tokens: 0,
                cache_creation_1h_input_tokens: 0
            }
            .validate(0)
            .is_err()
        );
    }
    #[test]
    fn accounting_missing_proof_and_zero_plan() {
        let mut a = CacheAccounting::new(
            120,
            false,
            vec![
                scope("five", 300_000, true, false),
                scope("hour", 3_600_000, true, false),
            ],
        )
        .unwrap();
        assert_eq!(a.finish(), None);
        a.observe(event("e1", "five", 300_000, 0, vec![])).unwrap();
        assert_eq!(a.finish(), None);
        a.observe(event("e2", "hour", 3_600_000, 0, vec![]))
            .unwrap();
        assert_eq!(a.finish().unwrap().input_tokens, 120);
        let a = CacheAccounting::new(120, true, vec![]).unwrap();
        assert_eq!(
            a.finish(),
            Some(VerifiedCacheAccounting {
                input_tokens: 120,
                cache_read_input_tokens: 0,
                cache_creation_5m_input_tokens: 0,
                cache_creation_1h_input_tokens: 0
            })
        );
        assert_eq!(
            CacheAccounting::new(120, false, vec![]).unwrap().finish(),
            a.finish()
        );
        let mut read_required = CacheAccounting::new(
            120,
            false,
            vec![
                scope("write", 300_000, true, false),
                scope("read", 300_000, false, true),
            ],
        )
        .unwrap();
        read_required
            .observe(event("write", "write", 300_000, 0, vec![]))
            .unwrap();
        assert_eq!(read_required.finish(), None);
        read_required
            .observe(event("read", "read", 300_000, 0, vec![]))
            .unwrap();
        assert_eq!(read_required.finish().unwrap().input_tokens, 120);
    }
    #[test]
    fn accounting_rejects_unretained_or_changed_coverage() {
        let mut a =
            CacheAccounting::new(120, false, vec![scope("five", 300_000, true, false)]).unwrap();
        a.observe(event("e1", "five", 300_000, 0, vec![(0, 80)]))
            .unwrap();
        assert!(
            a.observe(event("e1", "five", 300_000, 0, vec![(0, 90)]))
                .is_err()
        );
        assert_eq!(a.finish(), None);
        let mut a =
            CacheAccounting::new(120, false, vec![scope("five", 300_000, true, false)]).unwrap();
        assert!(
            a.observe(event("e1", "five", 300_000, 0, vec![(80, 101)]))
                .is_err()
        );
        assert_eq!(a.finish(), None);
    }
}
