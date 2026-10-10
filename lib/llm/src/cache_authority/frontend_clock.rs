//! Fresh Frontend clock qualification for private cache authorization.
//!
//! https://chrony-project.org/doc/4.8/chronyc.html#tracking documents
//! |system offset| + root dispersion + root delay/2, assuming the selected
//! stratum-1 source is correct. Command latency and an extra millisecond are
//! included on EACH side. No synchronization, command, or wall-clock fallback.
use rust_decimal::{Decimal, prelude::ToPrimitive};
use std::{collections::BTreeMap, process::Stdio, time::{Duration, SystemTime, UNIX_EPOCH}};
use tokio::{io::AsyncReadExt, process::Command, time::{Instant, timeout}};

const MAX_ERROR_MS: u64 = 100;
const MAX_TRACKING_BYTES: usize = 16_384;
const MAX_AGE: Duration = Duration::from_millis(100);

#[derive(Debug, thiserror::Error)]
#[error("unqualified Frontend clock: {0}")]
pub struct ClockError(&'static str);

/// Validated current bounds at `observed_at`. At HTTP ingress, use the
/// request_start_lower_ms/request_started_at pair captured BEFORE chronyc.
/// A single-sided error <=100ms
/// means the complete interval can be up to 200ms wide.
#[derive(Clone, Debug)]
pub struct ClockBounds {
    lower_ms: u64,
    upper_ms: u64,
    error_ms: u64,
    observed_at: Instant,
    request_start_lower_ms: u64,
    request_started_at: Instant,
}
impl ClockBounds {
    #[cfg(test)]
    pub(crate) fn cpu_fixture(now:Instant)->Self {Self{lower_ms:1_000_000,upper_ms:1_000_002,error_ms:1,observed_at:now,request_start_lower_ms:1_000_000,request_started_at:now}}
    pub fn lower_ms(&self) -> u64 { self.lower_ms }
    pub fn upper_ms(&self) -> u64 { self.upper_ms }
    pub fn error_ms(&self) -> u64 { self.error_ms }
    pub fn observed_at(&self) -> Instant { self.observed_at }
    pub fn request_start_lower_ms(&self) -> u64 { self.request_start_lower_ms }
    pub fn request_started_at(&self) -> Instant { self.request_started_at }
    /// Never propagate an estimate indefinitely. The signing hook must obtain
    /// another estimate if work since qualification exceeded 100ms.
    pub fn ensure_fresh(&self) -> Result<(), ClockError> {
        self.ensure_fresh_at(Instant::now())
    }
    fn ensure_fresh_at(&self, now: Instant) -> Result<(), ClockError> {
        let age = now.checked_duration_since(self.observed_at)
            .ok_or(ClockError("monotonic reversal"))?;
        if age > MAX_AGE { return Err(ClockError("tracking estimate is stale")); }
        Ok(())
    }
}

/// Runs only the read-only `chronyc -n tracking` command. Missing chronyc,
/// a slow/failed command, bad output, and stepped clocks close cache use.
#[derive(Clone, Copy, Debug, Default)]
pub struct ChronyClock;

/// Explicit CPU tracking model for new signer tests. Production qualification
/// remains the read-only chronyc measurement below, with no wall-clock fallback.
#[cfg(test)]
pub(crate) fn qualified_cpu_tracking_sample(
    text: &str,
    wall_before: u128,
    wall_after: u128,
    mono_before: Instant,
    mono_after: Instant,
) -> Result<ClockBounds, ClockError> {
    qualify_sample(text, wall_before, wall_after, mono_before, mono_after)
}

impl ChronyClock {
    pub async fn qualify(&self) -> Result<ClockBounds, ClockError> {
        let mono_before = Instant::now();
        let wall_before = wall_ns()?;
        let mut child = Command::new("chronyc").args(["-n", "tracking"])
            .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null())
            .kill_on_drop(true).spawn().map_err(|_| ClockError("chronyc unavailable"))?;
        let mut stdout = child.stdout.take().ok_or(ClockError("missing tracking pipe"))?;
        let remaining = MAX_AGE.checked_sub(mono_before.elapsed())
            .ok_or(ClockError("command deadline exceeded"))?;
        let result = timeout(remaining, async {
            let mut bytes = Vec::new();
            (&mut stdout).take((MAX_TRACKING_BYTES + 1) as u64).read_to_end(&mut bytes)
                .await.map_err(|_| ClockError("tracking read failed"))?;
            if bytes.len() > MAX_TRACKING_BYTES { return Err(ClockError("oversized tracking output")); }
            let status = child.wait().await.map_err(|_| ClockError("chronyc wait failed"))?;
            if !status.success() { return Err(ClockError("chronyc failed")); }
            String::from_utf8(bytes).map_err(|_| ClockError("non-UTF8 tracking output"))
        }).await.map_err(|_| ClockError("command deadline exceeded"))??;
        let wall_after = wall_ns()?;
        let mono_after = Instant::now();
        qualify_sample(&result, wall_before, wall_after, mono_before, mono_after)
    }
}

fn wall_ns() -> Result<u128, ClockError> {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|n| n.as_nanos())
        .map_err(|_| ClockError("wall clock before epoch"))
}

fn tracking_error_ms(text: &str) -> Result<u64, ClockError> {
    if text.len() > MAX_TRACKING_BYTES { return Err(ClockError("oversized tracking output")); }
    let mut fields = BTreeMap::new();
    for line in text.lines() {
        if let Some((key, value)) = line.split_once(':') {
            if fields.insert(key.trim(), value.trim()).is_some() {
                return Err(ClockError("duplicate tracking field"));
            }
        }
    }
    if fields.get("Leap status") != Some(&"Normal") { return Err(ClockError("clock is not synchronized")); }
    let reference = fields.get("Reference ID").and_then(|v| v.split_whitespace().next())
        .ok_or(ClockError("missing external reference"))?;
    if reference.len() != 8 || !reference.bytes().all(|b| b.is_ascii_hexdigit())
        || reference.eq_ignore_ascii_case("00000000") || reference.eq_ignore_ascii_case("7F7F0101") {
        return Err(ClockError("clock has no qualified external reference"));
    }
    let seconds = |name| -> Result<Decimal, ClockError> {
        let value = fields.get(name).ok_or(ClockError("missing tracking field"))?;
        let mut words = value.split_whitespace();
        let raw = words.next().ok_or(ClockError("empty tracking field"))?;
        if words.next() != Some("seconds") || raw.len() > 128 {
            return Err(ClockError("invalid tracking field units"));
        }
        // Validate the mantissa with the exact maintained decimal parser before
        // its scientific parser, whose normal FromStr can otherwise round.
        let mut parts = raw.split(['e', 'E']);
        let base = parts.next().unwrap_or("");
        let unsigned = base.strip_prefix(['+', '-']).unwrap_or(base);
        let mut mantissa = unsigned.split('.');
        let integer = mantissa.next().unwrap_or("");
        if integer.is_empty() || !integer.bytes().all(|b| b.is_ascii_digit())
            || mantissa.next().is_some_and(|fraction| fraction.is_empty() || !fraction.bytes().all(|b| b.is_ascii_digit()))
            || mantissa.next().is_some() { return Err(ClockError("invalid tracking decimal")); }
        Decimal::from_str_exact(base).map_err(|_| ClockError("inexact tracking decimal"))?;
        let value = if let Some(exp) = parts.next() {
            if parts.next().is_some() || exp.is_empty() { return Err(ClockError("invalid decimal exponent")); }
            let digits = exp.strip_prefix(['+', '-']).unwrap_or(exp);
            if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) { return Err(ClockError("invalid decimal exponent")); }
            Decimal::from_scientific(raw).map_err(|_| ClockError("inexact tracking exponent"))?
        } else { Decimal::from_str_exact(raw).map_err(|_| ClockError("inexact tracking decimal"))? };
        // Leaves exact precision for division by two and accumulation. Values
        // outside this range cannot satisfy the 100ms final error gate anyway.
        if value.scale() > 24 || value.abs() > Decimal::ONE { return Err(ClockError("tracking value exceeds qualification range")); }
        Ok(value)
    };
    let dispersion = seconds("Root dispersion")?;
    let delay = seconds("Root delay")?;
    if dispersion < Decimal::ZERO || delay < Decimal::ZERO { return Err(ClockError("negative root uncertainty")); }
    let error = seconds("System time")?.abs().checked_add(dispersion)
        .and_then(|n| delay.checked_div(Decimal::from(2)).and_then(|d| n.checked_add(d)))
        .and_then(|n| n.checked_mul(Decimal::from(1000)))
        .ok_or(ClockError("tracking uncertainty overflow"))?;
    error.ceil().to_u64().ok_or(ClockError("tracking uncertainty overflow"))
}

fn qualify_sample(text: &str, wall_before: u128, wall_after: u128,
                  mono_before: Instant, mono_after: Instant) -> Result<ClockBounds, ClockError> {
    let elapsed = mono_after.checked_duration_since(mono_before).ok_or(ClockError("monotonic reversal"))?;
    if elapsed > MAX_AGE { return Err(ClockError("command deadline exceeded")); }
    let elapsed_ms = u64::try_from(elapsed.as_nanos().div_ceil(1_000_000)).map_err(|_| ClockError("elapsed overflow"))?;
    let wall_elapsed = wall_after.checked_sub(wall_before).ok_or(ClockError("wall clock stepped backwards"))?;
    if wall_elapsed.abs_diff(u128::from(elapsed_ms) * 1_000_000) > 2_000_000 {
        return Err(ClockError("wall clock stepped during qualification"));
    }
    let error_ms = tracking_error_ms(text)?.checked_add(elapsed_ms).and_then(|n| n.checked_add(1))
        .ok_or(ClockError("clock uncertainty overflow"))?;
    if error_ms > MAX_ERROR_MS { return Err(ClockError("clock uncertainty exceeds 100ms")); }
    // Ceil the center like Python. The extra 1ms already included in error_ms
    // also covers this upward rounding on the LOWER side (less than 1ms).
    let center = u64::try_from(wall_after.div_ceil(1_000_000)).map_err(|_| ClockError("wall clock overflow"))?;
    let start_center = u64::try_from(wall_before.div_ceil(1_000_000)).map_err(|_| ClockError("wall clock overflow"))?;
    Ok(ClockBounds { lower_ms: center.checked_sub(error_ms).ok_or(ClockError("lower clock bound overflow"))?,
        upper_ms: center.checked_add(error_ms).ok_or(ClockError("upper clock bound overflow"))?, error_ms,
        observed_at: mono_after,
        request_start_lower_ms: start_center.checked_sub(error_ms).ok_or(ClockError("request-start clock bound overflow"))?,
        request_started_at: mono_before })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tracking(offset: &str, delay: &str, dispersion: &str) -> String {
        format!("Reference ID : CB00710F (time.example)\nSystem time : {offset} seconds slow of NTP time\nRoot delay : {delay} seconds\nRoot dispersion : {dispersion} seconds\nLeap status : Normal\n")
    }
    #[test]
    fn exact_decimal_rounding_and_reference_gates() {
        assert_eq!(tracking_error_ms(&tracking("-0.000006523", "0.013639022", "0.001100737")).unwrap(), 8);
        assert_eq!(tracking_error_ms(&tracking("1e-6", "0", "0")).unwrap(), 1);
        assert_eq!(tracking_error_ms(&tracking("0.000999999999999999999999", "0", "0")).unwrap(), 1);
        for text in [tracking("nan", "0", "0"), tracking("0", "-0.1", "0"),
            tracking("0", "0", "0").replace("Normal", "Not synchronised"),
            tracking("0", "0", "0").replace("CB00710F", "7F7F0101"),
            tracking("0", "0", "0").replace("CB00710F", "00000000"),
            tracking("0", "0", "0") + "Root delay : 0 seconds\n",
            tracking("1e-29", "0", "0"), tracking("0_1", "0", "0"),
            tracking(".1", "0", "0"), tracking("1.", "0", "0")] {
            assert!(tracking_error_ms(&text).is_err(), "{text}");
        }
    }
    #[test]
    fn uncertainty_boundary_steps_and_freshness() {
        let before = Instant::now();
        let zero = tracking("0", "0", "0");
        let wall = 1_000_000_000_000u128;
        let bounds = qualify_sample(&tracking("0.098", "0", "0"), wall, wall + 1_000_000,
            before, before + Duration::from_millis(1)).unwrap();
        assert_eq!(bounds.error_ms(), 100);
        assert_eq!(bounds.upper_ms() - bounds.lower_ms(), 200);
        assert_eq!(bounds.request_start_lower_ms(), 999_900);
        assert_eq!(bounds.lower_ms(), 999_901);
        assert_eq!(bounds.request_started_at(), before);
        assert!(qualify_sample(&tracking("0.099", "0", "0"), wall, wall + 1_000_000,
            before, before + Duration::from_millis(1)).is_err());
        for after in [wall - 1, wall + 4_000_001] {
            assert!(qualify_sample(&zero, wall, after, before, before + Duration::from_millis(1)).is_err());
        }
        assert!(qualify_sample(&zero, wall, wall + 101_000_000, before, before + Duration::from_millis(101)).is_err());
        assert!(bounds.ensure_fresh_at(bounds.observed_at() + MAX_AGE).is_ok());
        assert!(bounds.ensure_fresh_at(bounds.observed_at() + MAX_AGE + Duration::from_nanos(1)).is_err());
        assert!(bounds.ensure_fresh_at(before).is_err());
        let fractional = qualify_sample(&zero, wall, wall + 1_000_001, before, before + Duration::from_nanos(1_000_001)).unwrap();
        assert_eq!(fractional.error_ms(), 3);
        assert_eq!(fractional.upper_ms() - fractional.lower_ms(), 6);
    }
}
