//! Native SGLang P/D routing inside the existing Dynamo frontend process.
//!
//! Routing fields are generated only after canonical validation and exact Count.
//! They must never come from a public Chat request or protocol conversion.

use serde_json::Value;

pub mod control;
pub mod dispatch;
pub mod qualification;
pub mod transport;

#[cfg(test)]
mod control_tests;

/// These fields are consumed by the pinned native Chat serving adapter and
/// forwarded to GenerateReqInput. Public callers do not own transport routing.
pub const NATIVE_ROUTING_FIELDS: &[&str] = &[
    "bootstrap_host",
    "bootstrap_port",
    "bootstrap_room",
    "disagg_prefill_dp_rank",
];

pub fn reject_client_routing(body: &Value) -> Result<(), &'static str> {
    let object = body.as_object().ok_or("native body must be an object")?;
    if NATIVE_ROUTING_FIELDS
        .iter()
        .any(|field| object.contains_key(*field))
    {
        return Err("native P/D routing fields are owned by the gateway");
    }
    Ok(())
}
