//! Exact origin expiry authorizes hits; etcd leases only bound cleanup.
//! A lease is granted before generation and never renewed. Markers merge per
//! worker incarnation; shorter refreshes cannot shorten a longer entry or grow
//! an unbounded ledger. The old value registry is a policy fixture only.
use super::*;
use etcd_client::GetOptions;
use tokio::time::Instant;

#[derive(Clone, Debug)]
pub(crate) struct OriginLease { pub id:i64 }
#[derive(Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
struct Marker { prefix:Prefix, pin:Pin,ttl_ms:u64 }
fn marker_root(prefix:&Prefix)->Result<String> {Ok(format!("{}markers/",prefix.key()?))}
impl EtcdStore {
    pub(crate) async fn grant_origin_lease(&self,remaining_ms:u64)->Result<Option<OriginLease>> {
        if remaining_ms==0 {return Ok(None)};
        // Cleanup may be later; a coarse etcd second cannot shorten a valid
        // entry. The exact persisted deadline remains the read fence.
        let seconds=remaining_ms.div_ceil(1000).saturating_add(2).max(5);
        let before=Instant::now();
        let reply=bounded(self.client.clone().lease_grant(seconds as i64,None)).await?;
        if before.elapsed()>Duration::from_secs(1) || reply.id()==0 || reply.ttl()!=seconds as i64 {
            if reply.id()!=0 {let _=bounded(self.client.clone().lease_revoke(reply.id())).await;}
            return Err(Error::Evidence("origin lease grant outside bounded deadline"));
        }
        Ok(Some(OriginLease{id:reply.id()}))
    }
    async fn lease_live(&self,id:i64)->Result<bool> {
        if id==0 {return Ok(false)};
        Ok(bounded(self.client.clone().lease_time_to_live(id,None)).await?.ttl()>0)
    }
    pub(crate) async fn publish_origin_marker(&self,e:&NativeEvidence,lease:&OriginLease,reservation:&str,revision:i64,now_upper_ms:u64)->Result<bool> {
        if e.pinned_until_ms<=now_upper_ms {return Ok(false)};
        if !self.lease_live(lease.id).await? {return Ok(false)};
        let mut hash=Sha256::new();hash.update((e.worker.len() as u64).to_be_bytes());hash.update(e.worker.as_bytes());hash.update(e.incarnation.as_bytes());
        let key=format!("{}{:x}",marker_root(&e.prefix)?,hash.finalize());
        let ttl_ms=e.pinned_until_ms.checked_sub(e.request_started_at_ms).ok_or(Error::Evidence("marker TTL invalid"))?;
        if !matches!(ttl_ms,300000|3600000) {return Err(Error::Evidence("marker TTL not protocol duration"));}
        let bytes=serde_json::to_vec(&Marker{prefix:e.prefix.clone(),pin:pin_from(e),ttl_ms}).map_err(|x|Error::Store(x.to_string()))?;
        let reservation_key=format!("{ROOT}reservations/{reservation}");
        for _ in 0..8 {
            let old=bounded(self.client.clone().get(key.as_str(),None)).await?;
            let existing=old.kvs().first();
            let new_ttl=bounded(self.client.clone().lease_time_to_live(lease.id,None)).await?.ttl();
            if new_ttl<=0 {return Ok(false)};
            if let Some(row)=existing {
                let marker:Marker=serde_json::from_slice(row.value()).map_err(|x|Error::Store(x.to_string()))?;
                if marker.prefix!=e.prefix || marker.pin.worker!=e.worker || marker.pin.incarnation!=e.incarnation {return Err(Error::Evidence("marker owner collision"));}
                if marker.pin.pinned_until_ms>=e.pinned_until_ms && marker.pin.pinned_until_ms>now_upper_ms {return Ok(true)};
            }
            let compare=if let Some(row)=existing {Compare::mod_revision(key.as_str(),CompareOp::Equal,row.mod_revision())}else{Compare::version(key.as_str(),CompareOp::Equal,0)};
            let reply=bounded(self.client.clone().txn(Txn::new().when(vec![
                Compare::mod_revision(reservation_key.clone(),CompareOp::Equal,revision),compare
            ]).and_then(vec![TxnOp::put(key.as_str(),bytes.clone(),Some(PutOptions::new().with_lease(lease.id)))]))).await?;
            // Expiration between check and put fails at the server; no new
            // lease or retry can resurrect that deadline.
            if reply.succeeded() {return Ok(true)};
        }
        Err(Error::Contention)
    }
    pub(crate) async fn lookup_origin_ttl(&self,prefix:&Prefix,worker:&str,incarnation:&str,now_upper_ms:u64)->Result<Option<(Ttl,u64)>> {
        let reply=bounded(self.client.clone().get(marker_root(prefix)?,Some(GetOptions::new().with_prefix().with_limit((MAX_PINS+1) as i64)))).await?;
        if reply.kvs().len()>MAX_PINS {return Err(Error::Evidence("too many live logical markers"))};
        for kv in reply.kvs() {
            let m:Marker=serde_json::from_slice(kv.value()).map_err(|e|Error::Store(e.to_string()))?;
            if m.prefix!=*prefix {return Err(Error::Evidence("leased marker prefix mismatch"))};
            if m.pin.worker!=worker || m.pin.incarnation!=incarnation {continue};
            if m.pin.pinned_until_ms<=now_upper_ms {continue};
            // Linearizable second read prevents accepting a removed/replaced
            // marker. Native request dispatch remains bound to its owner.
            let current=bounded(self.client.clone().get(kv.key(),None)).await?;
            if current.kvs().first().is_some_and(|v|v.mod_revision()==kv.mod_revision() && v.lease()==kv.lease()) {
                return match m.ttl_ms {300000=>Ok(Some((Ttl::FiveMinutes,m.pin.pinned_until_ms))),3600000=>Ok(Some((Ttl::OneHour,m.pin.pinned_until_ms))),_=>Err(Error::Evidence("marker stored invalid original TTL"))}
            }
        }
        Ok(None)
    }
    pub(crate) async fn lookup_origin_markers(&self,prefix:&Prefix,worker:&str,incarnation:&str,now_upper_ms:u64)->Result<bool> {
        Ok(self.lookup_origin_ttl(prefix,worker,incarnation,now_upper_ms).await?.is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    #[ignore="requires owned disposable loopback etcd fixture"]
    async fn real_etcd_origin_leases_do_not_renew_or_shorten_long_marker() {
        let store=EtcdStore::connect(vec![std::env::var("CACHE_AUTHORITY_TEST_ETCD").unwrap()]).await.unwrap();
        let other=store.clone();
        assert!(store.grant_origin_lease(0).await.unwrap().is_none());
        assert!(store.grant_origin_lease(100).await.unwrap().is_some());
        let long=store.grant_origin_lease(3600000).await.unwrap().unwrap();
        let short=store.grant_origin_lease(6000).await.unwrap().unwrap();
        let ttl=bounded(store.client.clone().lease_time_to_live(short.id,None)).await.unwrap().ttl();assert!(ttl>=6 && ttl<=8);
        let key=format!("{ROOT}reservations/owned-clock-fixture");
        let reply=bounded(store.client.clone().put(key.clone(),"CPU-only-reservation",None)).await.unwrap();let revision=reply.header().unwrap().revision();
        let prefix=Prefix{tenant:"owned-etcd-clock-fixture".into(),namespace:"ns".into(),runtime:[1;32],digest:[2;32],marked_endpoint_digest:[3;32],marked_render_digest:[4;32],marked_tokens:8,tokens:8};
        let mut e=NativeEvidence{request_id:"http-fixture".into(),native_request_id:"native-fixture".into(),prefix:prefix.clone(),worker:"cpu-fixture".into(),incarnation:"boot".into(),receipt_id:"long".into(),kind:EventKind::WriteCommitted,observed_at_ms:1,request_started_at_ms:0,pinned_until_ms:3600000,committed_tokens:8};
        // Synthetic event DTO tests etcd lifetime only; native qualification is
        // separately exercised by the actual native lock integration fixture.
        assert!(store.publish_origin_marker(&e,&long,"owned-clock-fixture",revision,1).await.unwrap());
        e.receipt_id="short".into();e.pinned_until_ms=300000;
        assert!(other.publish_origin_marker(&e,&short,"owned-clock-fixture",revision,1).await.unwrap());
        bounded(store.client.clone().lease_revoke(short.id)).await.unwrap();
        assert!(other.lookup_origin_markers(&prefix,"cpu-fixture","boot",1).await.unwrap());
        bounded(store.client.clone().lease_revoke(long.id)).await.unwrap();
        assert!(!other.lookup_origin_markers(&prefix,"cpu-fixture","boot",1).await.unwrap());
        assert!(!store.publish_origin_marker(&e,&short,"owned-clock-fixture",revision,1).await.unwrap());
    }

    #[tokio::test]
    #[ignore="requires owned disposable loopback etcd fixture"]
    async fn real_etcd_exact_millisecond_expiry_with_cleanup_lease() {
        let store=EtcdStore::connect(vec![std::env::var("CACHE_AUTHORITY_TEST_ETCD").unwrap()]).await.unwrap();
        let lease=store.grant_origin_lease(1).await.unwrap().unwrap();
        assert!(bounded(store.client.clone().lease_time_to_live(lease.id,None)).await.unwrap().ttl()>0);
        let reservation="exact-ms-origin-fixture";
        let r=bounded(store.client.clone().put(format!("{ROOT}reservations/{reservation}"),"cpu-origin",None)).await.unwrap();
        let revision=r.header().unwrap().revision();
        let prefix=Prefix{tenant:"owned-exact-ms".into(),namespace:"ns".into(),runtime:[1;32],digest:[2;32],marked_endpoint_digest:[3;32],marked_render_digest:[4;32],marked_tokens:8,tokens:8};
        let e=NativeEvidence{request_id:"http".into(),native_request_id:"native".into(),prefix:prefix.clone(),worker:"cpu".into(),incarnation:"boot".into(),receipt_id:"native-evidence-dto".into(),kind:EventKind::WriteCommitted,observed_at_ms:299999,request_started_at_ms:0,pinned_until_ms:300000,committed_tokens:8};
        assert!(store.publish_origin_marker(&e,&lease,reservation,revision,299999).await.unwrap());
        assert!(store.lookup_origin_ttl(&prefix,"cpu","boot",299999).await.unwrap().is_some());
        assert!(store.lookup_origin_ttl(&prefix,"cpu","boot",300000).await.unwrap().is_none());
        assert!(!store.publish_origin_marker(&e,&lease,reservation,revision,300000).await.unwrap());
        // The row still physically exists: only the exact logical expiry,
        // rather than coarse TTL, authorized the previous millisecond.
        assert!(!bounded(store.client.clone().get(marker_root(&prefix).unwrap(),Some(GetOptions::new().with_prefix()))).await.unwrap().kvs().is_empty());
        bounded(store.client.clone().lease_revoke(lease.id)).await.unwrap();
    }
}
