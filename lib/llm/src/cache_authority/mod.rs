//! Frontend-internal logical cache registry. Uses the existing etcd control
//! plane; it is not a service or an additional generation hop.
//!
//! A native adapter must verify event timestamps, complete prefix commit and a
//! non-evictable pin before calling this module. Usage.cached_tokens, HTTP
//! arrival time, successful generation and count results are not receipts.
use async_trait::async_trait;
use etcd_client::{Client, Compare, CompareOp, PutOptions, Txn, TxnOp};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::Duration;
#[path = "frontend_cache_hook.rs"]
pub mod frontend_cache_hook;
#[path = "../auth-rust/cache_auth.rs"]
pub mod cache_auth;
mod lease_authority;
#[path="frontend_clock.rs"]
pub mod frontend_clock;
pub mod native_cache_frontend;
pub mod frontend_qualification;
pub mod cache_accounting;
pub mod shared_cache_authority;
mod replicated_ds_promotion;

const ROOT: &str = "/dsh/anthropic-cache/v1/";
const MAX_PINS: usize = 32;

#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum Error {
    #[error("invalid native cache evidence: {0}")]
    Evidence(&'static str),
    #[error("cache registry unavailable: {0}")]
    Store(String),
    #[error("native cache operation HTTP {status}: {cause}")]
    NativeHttp { status: u16, cause: String },
    #[error("cache registry CAS contention")]
    Contention,
}
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Prefix {
    /// Trusted authenticated tenant or deployment tenant. Never request text.
    pub tenant: String,
    pub namespace: String,
    /// Digest of full weights/template/position/layout/backend runtime contract.
    pub runtime: [u8; 32],
    /// Digest of exact full-rendered token prefix, not a shortened chat render.
    pub digest: [u8; 32],
    /// Exact marked endpoint's full-rendered prefix. Two distinct logical
    /// markers sharing one physical page prefix are never the same entry.
    pub marked_endpoint_digest: [u8;32],
    /// SHA256 of the actual full renderer UTF-8 prefix at the raw marked byte.
    /// Different marks inside one BPE token remain distinct logical entries.
    pub marked_render_digest:[u8;32],
    pub marked_tokens: u64,
    pub tokens: u64,
}
impl Prefix {
    fn key(&self) -> Result<String> {
        if self.tenant.is_empty() || self.namespace.is_empty() || self.tokens == 0
            || self.tenant.len() > 1024 || self.namespace.len() > 1024
            || self.runtime == [0;32] || self.digest == [0;32]
            || self.marked_endpoint_digest==[0;32] || self.marked_render_digest==[0;32] || self.marked_tokens<self.tokens {
            return Err(Error::Evidence("invalid prefix identity"));
        }
        let mut h = Sha256::new();
        h.update(b"dynamo-anthropic-logical-prefix-v1\0");
        for s in [&self.tenant, &self.namespace] {
            h.update((s.len() as u64).to_be_bytes()); h.update(s.as_bytes());
        }
        h.update(self.runtime); h.update(self.digest); h.update(self.tokens.to_be_bytes());
        h.update(self.marked_endpoint_digest);h.update(self.marked_render_digest);h.update(self.marked_tokens.to_be_bytes());
        Ok(format!("{ROOT}{:x}", h.finalize()))
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Ttl { FiveMinutes, OneHour }
impl Ttl { fn ms(self) -> u64 { match self { Self::FiveMinutes => 300_000, Self::OneHour => 3_600_000 } } }
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum EventKind { WriteCommitted, ReadHit }

/// Internal native RPC evidence DTO. NOT a public request field. A production
/// verifier must bind every field to the dispatched server-owned request,
/// worker incarnation and full-render plan, and check native pin ownership.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeEvidence {
    pub request_id: String,
    pub native_request_id:String,
    pub prefix: Prefix,
    pub worker: String,
    pub incarnation: String,
    pub receipt_id: String,
    pub kind: EventKind,
    /// Authority-domain time this verified receipt was observed. Native actual
    /// event ordering and hold proof are validated separately in their own
    /// clock domain by the concrete parser; no cross-node conversion is implied.
    pub observed_at_ms: u64,
    /// Server-owned API request start, bound by verifier to dispatch context.
    /// Official lifetime starts here, not at the eventual native cache event.
    pub request_started_at_ms: u64,
    pub pinned_until_ms: u64,
    pub committed_tokens: u64,
}

#[async_trait]
pub trait NativeVerifier: Send + Sync {
    /// Reject missing actual events/pins, unqualified runtime, bad signatures
    /// or channel identity, incomplete ranks/pages, clock mapping uncertainty,
    /// wrong request/breakpoint or a revocable/evictable pin.
    async fn verify(&self, evidence: &NativeEvidence) -> Result<()>;
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Pin {
    pub worker: String,
    pub incarnation: String,
    pub receipt_id: String,
    pub pinned_until_ms: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg(test)]
#[serde(deny_unknown_fields)]
pub struct Record { prefix: Prefix, expires_at_ms: u64, pins: Vec<Pin> }
#[derive(Clone, Debug)]
#[cfg(test)]
pub struct Row { pub revision: i64, pub record: Record }

#[async_trait]
#[cfg(test)]
pub trait Store: Send + Sync {
    /// Linearizable read only. No local stale mirror can authorize a hit.
    async fn read(&self, key: &str) -> Result<Option<Row>>;
    /// revision=0 means create-if-absent. TTL is garbage collection only;
    /// exact logical expiry remains in the value and is checked on every read.
    async fn cas(&self, key: &str, revision: i64, record: &Record, gc_ttl_ms: u64) -> Result<bool>;
}

#[derive(Clone)]
pub struct EtcdStore { client: Client }
impl EtcdStore {
    pub async fn connect(endpoints: Vec<String>) -> Result<Self> {
        if endpoints.is_empty() { return Err(Error::Store("empty endpoints".into())); }
        let client = bounded(Client::connect(endpoints, None)).await?;
        Ok(Self { client })
    }
}
async fn bounded<T>(op: impl std::future::Future<Output=std::result::Result<T,etcd_client::Error>>) -> Result<T> {
    tokio::time::timeout(Duration::from_secs(1), op).await
        .map_err(|_|Error::Store("etcd timeout".into()))?
        .map_err(|e|Error::Store(e.to_string()))
}
#[async_trait]
#[cfg(test)]
impl Store for EtcdStore {
    async fn read(&self, key: &str) -> Result<Option<Row>> {
        let reply = bounded(self.client.clone().get(key, None)).await?;
        if reply.kvs().is_empty() { return Ok(None); }
        if reply.kvs().len()!=1 { return Err(Error::Store("multiple records".into())); }
        let kv = &reply.kvs()[0];
        if kv.mod_revision()<=0 { return Err(Error::Store("invalid revision".into())); }
        let record = serde_json::from_slice(kv.value()).map_err(|e|Error::Store(e.to_string()))?;
        Ok(Some(Row{ revision:kv.mod_revision(), record }))
    }
    async fn cas(&self, key: &str, revision: i64, record: &Record, gc_ttl_ms: u64) -> Result<bool> {
        if revision<0 || gc_ttl_ms==0 { return Err(Error::Evidence("invalid CAS/TTL")); }
        let bytes = serde_json::to_vec(record).map_err(|e|Error::Store(e.to_string()))?;
        let mut client = self.client.clone();
        let lease = bounded(client.lease_grant(gc_ttl_ms.div_ceil(1000) as i64, None)).await?.id();
        let compare = if revision==0 { Compare::version(key,CompareOp::Equal,0) }
            else { Compare::mod_revision(key,CompareOp::Equal,revision) };
        let reply = bounded(client.txn(Txn::new().when(vec![compare]).and_then(vec![
            TxnOp::put(key,bytes,Some(PutOptions::new().with_lease(lease)))
        ]))).await?;
        if !reply.succeeded() { bounded(client.lease_revoke(lease)).await?; }
        Ok(reply.succeeded())
    }
}

#[derive(Clone, Debug)]
pub struct ReadPermit { prefix: Prefix, authorized_until_ms: u64, request_started_at_ms: u64, request_id: String }
#[derive(Clone, Debug)]
#[cfg(test)]
pub struct Lookup { pub expires_at_ms: u64, pub pins: Vec<Pin>, pub permit: ReadPermit }

#[cfg(test)]
pub struct Authority<S,V> { store: S, verifier: V }
#[cfg(test)]
impl<S:Store,V:NativeVerifier> Authority<S,V> {
    pub fn new(store:S,verifier:V)->Self { Self{store,verifier} }
    /// Count endpoints never call publication. Lookup itself has no mutation.
    /// Adapter must qualify the selected pin's current worker incarnation.
    pub async fn lookup(&self,prefix:&Prefix,request_id:&str,request_started_at_ms:u64,now_ms:u64)->Result<Option<Lookup>> {
        if request_id.is_empty() || request_id.len()>1024 || request_started_at_ms>now_ms {return Err(Error::Evidence("invalid server-owned request context"));}
        let Some(row)=self.store.read(&prefix.key()?).await? else {return Ok(None)};
        validate_record(&row.record,prefix)?;
        if row.record.expires_at_ms<=now_ms {return Ok(None)};
        let pins:Vec<_>=row.record.pins.into_iter().filter(|p|p.pinned_until_ms>now_ms).collect();
        if pins.is_empty() {return Ok(None)};
        Ok(Some(Lookup{ expires_at_ms:row.record.expires_at_ms,pins,
            permit:ReadPermit{prefix:prefix.clone(),authorized_until_ms:row.record.expires_at_ms,request_started_at_ms,request_id:request_id.into()} }))
    }
    /// No evidence means no action, including cancellation/timeout. If actual
    /// pinned commit/hit occurred before cancellation it remains true; public
    /// billing still requires the caller's separate native-event adjudication.
    pub async fn publish(&self,e:&NativeEvidence,ttl:Ttl,permit:Option<&ReadPermit>,now_ms:u64)->Result<bool> {
        let key=e.prefix.key()?;
        if e.request_id.is_empty() || e.native_request_id.is_empty() || e.worker.is_empty() || e.incarnation.is_empty() || e.receipt_id.is_empty()
            || [e.request_id.len(),e.native_request_id.len(),e.worker.len(),e.incarnation.len(),e.receipt_id.len()].iter().any(|n|*n>1024)
            || e.committed_tokens!=e.prefix.tokens || e.observed_at_ms>now_ms
            || e.observed_at_ms<e.request_started_at_ms || e.pinned_until_ms<e.observed_at_ms {
            return Err(Error::Evidence("receipt identity/boundary/time mismatch"));
        }
        let expiry=e.request_started_at_ms.checked_add(ttl.ms()).ok_or(Error::Evidence("expiry overflow"))?;
        if e.pinned_until_ms<expiry {return Err(Error::Evidence("pin cannot fulfill TTL"))};
        match e.kind {
            EventKind::WriteCommitted if permit.is_some()=>return Err(Error::Evidence("write supplied read permit")),
            EventKind::ReadHit=>{
                let p=permit.ok_or(Error::Evidence("hit missing lookup permit"))?;
                if p.prefix!=e.prefix || e.request_id!=p.request_id || e.request_started_at_ms!=p.request_started_at_ms
                    || e.request_started_at_ms>=p.authorized_until_ms {
                    return Err(Error::Evidence("hit outside authorized lifetime"));
                }
            }, _=>{}
        }
        self.verifier.verify(e).await?;
        if expiry<=now_ms {return Ok(false)};
        for _ in 0..8 {
            let old=self.store.read(&key).await?;
            if let Some(row)=&old {validate_record(&row.record,&e.prefix)?;}
            let revision=old.as_ref().map_or(0,|r|r.revision);
            let mut record=old.map_or(Record{prefix:e.prefix.clone(),expires_at_ms:expiry,pins:vec![]},|r|r.record);
            record.expires_at_ms=record.expires_at_ms.max(expiry);
            record.pins.retain(|p|p.pinned_until_ms>now_ms);
            if let Some(pin)=record.pins.iter_mut().find(|p|p.worker==e.worker && p.incarnation==e.incarnation) {
                if e.pinned_until_ms>pin.pinned_until_ms {*pin=pin_from(e)}
            } else {
                if record.pins.len()>=MAX_PINS {return Err(Error::Evidence("too many pinned replicas"))};
                record.pins.push(pin_from(e));
            }
            // Native pin may be longer than the logical TTL. etcd GC must not
            // manufacture logical time or delete a concurrent longer refresh.
            let remaining=record.expires_at_ms-now_ms;
            if self.store.cas(&key,revision,&record,remaining).await? {return Ok(true)};
        }
        Err(Error::Contention)
    }
}
fn pin_from(e:&NativeEvidence)->Pin {Pin{worker:e.worker.clone(),incarnation:e.incarnation.clone(),receipt_id:e.receipt_id.clone(),pinned_until_ms:e.pinned_until_ms}}
#[cfg(test)]
fn validate_record(r:&Record,p:&Prefix)->Result<()> {
    if r.prefix!=*p || r.pins.is_empty() || r.pins.len()>MAX_PINS
        || r.pins.iter().any(|p|p.worker.is_empty()||p.incarnation.is_empty()||p.receipt_id.is_empty())
        || !r.pins.iter().any(|p|p.pinned_until_ms>=r.expires_at_ms) {
        return Err(Error::Store("corrupt/unsatisfied record".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{collections::HashMap,sync::{Arc,Mutex}};
    #[derive(Clone,Default)]
    struct Memory(Arc<Mutex<(i64,HashMap<String,Row>)>>);
    #[async_trait]
    impl Store for Memory {
        async fn read(&self,k:&str)->Result<Option<Row>> {Ok(self.0.lock().unwrap().1.get(k).cloned())}
        async fn cas(&self,k:&str,rev:i64,r:&Record,_ttl:u64)->Result<bool> {
            let mut s=self.0.lock().unwrap();
            if s.1.get(k).map_or(0,|r|r.revision)!=rev {return Ok(false)};
            s.0+=1; let revision=s.0;s.1.insert(k.into(),Row{revision,record:r.clone()});Ok(true)
        }
    }
    /// CPU fixture only. Never used as production verifier qualification.
    struct FixtureVerifier(bool);
    #[async_trait]
    impl NativeVerifier for FixtureVerifier {
        async fn verify(&self,_:&NativeEvidence)->Result<()> {
            if self.0 {Ok(())} else {Err(Error::Evidence("CPU verifier rejected missing actual pin"))}
        }
    }
    fn prefix()->Prefix {Prefix{tenant:"disposable-fixture-tenant".into(),namespace:"test".into(),runtime:[1;32],digest:[2;32],marked_endpoint_digest:[3;32],marked_render_digest:[4;32],marked_tokens:2100,tokens:2100}}
    fn receipt(start:u64,kind:EventKind)->NativeEvidence {NativeEvidence{
        request_id:format!("server-owned-{start}"),native_request_id:format!("native-body-{start}"),prefix:prefix(),worker:"cpu-native-fixture".into(),
        incarnation:"fixture-incarnation".into(),receipt_id:format!("fixture-pin-{start}"),kind,
        observed_at_ms:start+1,request_started_at_ms:start,pinned_until_ms:start+3_600_000,
        committed_tokens:2100,
    }}
    #[tokio::test] async fn ttl_starts_at_request_not_commit_and_long_prefill_expires() {
        let a=Authority::new(Memory::default(),FixtureVerifier(true));
        let mut e=receipt(0,EventKind::WriteCommitted);e.observed_at_ms=528_000;
        assert!(!a.publish(&e,Ttl::FiveMinutes,None,528_001).await.unwrap());
        assert!(a.lookup(&e.prefix,"fixture-read",528_001,528_001).await.unwrap().is_none());
        assert!(a.publish(&e,Ttl::OneHour,None,528_001).await.unwrap());
        assert_eq!(a.lookup(&e.prefix,"fixture-read",528_001,528_001).await.unwrap().unwrap().expires_at_ms,3_600_000);
    }
    #[tokio::test] async fn pin_and_exact_boundary_required_before_any_entry() {
        let store=Memory::default();let a=Authority::new(store.clone(),FixtureVerifier(true));
        let mut e=receipt(10,EventKind::WriteCommitted);e.pinned_until_ms=300_009;
        assert!(a.publish(&e,Ttl::FiveMinutes,None,11).await.is_err());
        e.pinned_until_ms=3_600_010;e.committed_tokens=2048;
        assert!(a.publish(&e,Ttl::FiveMinutes,None,11).await.is_err());
        assert!(store.0.lock().unwrap().1.is_empty());
    }
    #[tokio::test] async fn unqualified_native_verifier_fails_closed() {
        let store=Memory::default();let a=Authority::new(store.clone(),FixtureVerifier(false));
        assert!(a.publish(&receipt(0,EventKind::WriteCommitted),Ttl::FiveMinutes,None,1).await.is_err());
        assert!(store.0.lock().unwrap().1.is_empty());
    }
    #[tokio::test] async fn lookup_has_no_mutation_or_cancel_refresh() {
        let store=Memory::default();let a=Authority::new(store.clone(),FixtureVerifier(true));
        let e=receipt(0,EventKind::WriteCommitted);a.publish(&e,Ttl::FiveMinutes,None,1).await.unwrap();
        let before=store.0.lock().unwrap().0;
        let _cancelled_request=a.lookup(&e.prefix,"fixture-read",299_999,299_999).await.unwrap().unwrap();
        assert_eq!(store.0.lock().unwrap().0,before);
        assert!(a.lookup(&e.prefix,"fixture-read",300_000,300_000).await.unwrap().is_none());
    }
    #[tokio::test] async fn read_refresh_bound_to_original_lookup_request_and_event_order() {
        let a=Authority::new(Memory::default(),FixtureVerifier(true));
        a.publish(&receipt(0,EventKind::WriteCommitted),Ttl::FiveMinutes,None,1).await.unwrap();
        let p=a.lookup(&prefix(),"server-owned-250000",250_000,250_000).await.unwrap().unwrap().permit;
        assert!(a.lookup(&prefix(),"delayed-render-request",250_000,300_000).await.unwrap().is_none());
        assert!(a.lookup(&prefix(),"future-start",300_001,300_000).await.is_err());
        let mut e=receipt(250_000,EventKind::ReadHit);e.observed_at_ms=310_000;
        // Cache was authorized at request start; slow actual hit event is not
        // the TTL origin and must be accompanied by an actual retained pin.
        a.publish(&e,Ttl::FiveMinutes,Some(&p),310_001).await.unwrap();
        assert_eq!(a.lookup(&prefix(),"server-owned-250000",310_001,310_001).await.unwrap().unwrap().expires_at_ms,550_000);
        let mut wrong_id=e.clone();wrong_id.request_id="different-server-owned-request".into();
        assert!(a.publish(&wrong_id,Ttl::FiveMinutes,Some(&p),310_001).await.is_err());
        e.request_started_at_ms+=1;assert!(a.publish(&e,Ttl::FiveMinutes,Some(&p),310_001).await.is_err());
        e.request_started_at_ms=310_001;assert!(a.publish(&e,Ttl::FiveMinutes,None,310_001).await.is_err());
    }
    #[tokio::test] async fn tenant_runtime_namespace_boundary_isolate() {
        let a=Authority::new(Memory::default(),FixtureVerifier(true));
        a.publish(&receipt(0,EventKind::WriteCommitted),Ttl::FiveMinutes,None,1).await.unwrap();
        for p in [Prefix{tenant:"other".into(),..prefix()},Prefix{namespace:"other".into(),..prefix()},Prefix{runtime:[3;32],..prefix()},Prefix{tokens:2048,..prefix()}] {
            assert!(a.lookup(&p,"fixture-read",1,1).await.unwrap().is_none());
        }
    }
    #[tokio::test] async fn delayed_shorter_refresh_never_shortens_longer_commit() {
        let a=Authority::new(Memory::default(),FixtureVerifier(true));
        a.publish(&receipt(100,EventKind::WriteCommitted),Ttl::OneHour,None,101).await.unwrap();
        a.publish(&receipt(10,EventKind::WriteCommitted),Ttl::FiveMinutes,None,102).await.unwrap();
        assert_eq!(a.lookup(&prefix(),"server-owned-250000",102,102).await.unwrap().unwrap().expires_at_ms,3_600_100);
    }
    #[tokio::test] async fn missing_fields_future_clock_and_inverted_native_time_reject() {
        let a=Authority::new(Memory::default(),FixtureVerifier(true));
        for e in [NativeEvidence{receipt_id:"".into(),..receipt(10,EventKind::WriteCommitted)},NativeEvidence{observed_at_ms:9,..receipt(10,EventKind::WriteCommitted)},NativeEvidence{observed_at_ms:12,..receipt(10,EventKind::WriteCommitted)}] {
            assert!(a.publish(&e,Ttl::FiveMinutes,None,11).await.is_err());
        }
    }
    #[tokio::test]
    #[ignore = "requires isolated disposable loopback etcd fixture"]
    async fn real_etcd_two_frontends_cas_expiry_scope_and_no_count_mutation() {
        let url=std::env::var("CACHE_AUTHORITY_TEST_ETCD").expect("isolated etcd URL");
        let a=Authority::new(EtcdStore::connect(vec![url.clone()]).await.unwrap(),FixtureVerifier(true));
        let b=Authority::new(EtcdStore::connect(vec![url]).await.unwrap(),FixtureVerifier(true));
        let now=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64;
        let mut e=receipt(now,EventKind::WriteCommitted);
        e.prefix.tenant=format!("owned-etcd-test-{}-{now}",std::process::id());
        a.publish(&e,Ttl::FiveMinutes,None,now+1).await.unwrap();
        let key=e.prefix.key().unwrap();
        let row_a=a.store.read(&key).await.unwrap().unwrap();
        let row_b=b.store.read(&key).await.unwrap().unwrap();
        assert_eq!(row_a.revision,row_b.revision);
        // Simultaneous cross-instance stale-revision writes: exactly one wins.
        let (x,y)=tokio::join!(a.store.cas(&key,row_a.revision,&row_a.record,300_000),b.store.cas(&key,row_b.revision,&row_b.record,300_000));
        assert_ne!(x.unwrap(),y.unwrap());
        let mut later=e.clone();later.request_started_at_ms+=100;later.observed_at_ms+=100;later.pinned_until_ms+=100;later.receipt_id="newer-pin".into();
        let (x,y)=tokio::join!(a.publish(&later,Ttl::OneHour,None,now+101),b.publish(&e,Ttl::FiveMinutes,None,now+101));
        assert!(x.unwrap()&&y.unwrap());
        let seen=b.lookup(&e.prefix,"fixture-read",now+101,now+101).await.unwrap().unwrap();
        assert_eq!(seen.expires_at_ms,now+100+3_600_000);
        let before=a.store.read(&key).await.unwrap().unwrap().revision;
        let _=b.lookup(&e.prefix,"fixture-read",now+102,now+102).await.unwrap();
        assert_eq!(a.store.read(&key).await.unwrap().unwrap().revision,before);
        assert!(a.lookup(&e.prefix,"fixture-read",seen.expires_at_ms,seen.expires_at_ms).await.unwrap().is_none());
        assert!(b.lookup(&Prefix{namespace:"other".into(),..e.prefix.clone()},"fixture-read",now+102,now+102).await.unwrap().is_none());
        let permit=seen.permit;let mut hit=later.clone();hit.kind=EventKind::ReadHit;
        assert!(b.publish(&hit,Ttl::FiveMinutes,Some(&permit),now+102).await.is_err()); // request mismatch
        let valid_id="real-etcd-server-owned-read";
        let permit=b.lookup(&e.prefix,valid_id,now+102,now+102).await.unwrap().unwrap().permit;
        hit.request_id=valid_id.into();hit.request_started_at_ms=now+102;hit.observed_at_ms=now+103;hit.pinned_until_ms=now+3_600_102;
        b.publish(&hit,Ttl::OneHour,Some(&permit),now+104).await.unwrap();
        assert_eq!(a.lookup(&e.prefix,"observer",now+104,now+104).await.unwrap().unwrap().expires_at_ms,now+3_600_102);
        let unavailable=EtcdStore::connect(vec!["http://127.0.0.1:1".into()]).await;
        if let Ok(store)=unavailable {assert!(store.read(&key).await.is_err());}
    }
}

#[cfg(test)] pub(crate) mod cpu_http_marker_fixture;
