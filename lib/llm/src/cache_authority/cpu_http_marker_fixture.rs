//! Synthetic logical marker injection for the actual HTTP branch test only.
//! Native reserve/cancel/admission are separately executed by the actual GF
//! Core and four-rank CPU fixture; this seed does not claim native publication.
use super::*;

pub(crate) async fn seed_marker(
    store: &EtcdStore,
    prefix: Prefix,
    worker: &str,
    incarnation: &str,
    now_upper_ms: u64,
) -> Result<i64> {
    seed_marker_expiring(
        store,
        prefix,
        worker,
        incarnation,
        now_upper_ms,
        now_upper_ms + 300_000,
    )
    .await
}

pub(crate) async fn seed_marker_expiring(
    store: &EtcdStore,
    prefix: Prefix,
    worker: &str,
    incarnation: &str,
    now_upper_ms: u64,
    expires_at_ms: u64,
) -> Result<i64> {
    let reservation = format!("cpu-stale-seed-{}", uuid::Uuid::new_v4().simple());
    let key = format!("{ROOT}reservations/{reservation}");
    let response = bounded(store.client.clone().put(key, "CPU-only-seed", None)).await?;
    let revision = response
        .header()
        .ok_or(Error::Evidence("CPU seed revision missing"))?
        .revision();
    let lease = store
        .grant_origin_lease(expires_at_ms.checked_sub(now_upper_ms).unwrap())
        .await?
        .ok_or(Error::Evidence("CPU seed origin lease missing"))?;
    let evidence = NativeEvidence {
        request_id: "CPU-logical-seed".into(),
        native_request_id: "CPU-logical-seed-native".into(),
        prefix,
        worker: worker.into(),
        incarnation: incarnation.into(),
        receipt_id: reservation.clone(),
        kind: EventKind::WriteCommitted,
        observed_at_ms: now_upper_ms,
        request_started_at_ms: expires_at_ms.checked_sub(300_000).unwrap(),
        pinned_until_ms: expires_at_ms,
        committed_tokens: 6144,
    };
    if !store
        .publish_origin_marker(&evidence, &lease, &reservation, revision, now_upper_ms)
        .await?
    {
        return Err(Error::Evidence("CPU marker seed did not publish"));
    }
    Ok(lease.id)
}

pub(crate) async fn revoke_seed_marker(store: &EtcdStore, lease_id: i64) -> Result<()> {
    bounded(store.client.clone().lease_revoke(lease_id)).await?;
    Ok(())
}
