//! Concrete same-Frontend cache dispatch/receipt hook. Public input never
//! constructs these contexts or private native fields. Existing worker POST and
//! SSE/JSON reader invoke this module; no authority or generation service added.
use super::*;
use serde_json::{json,Value};
use std::collections::{BTreeMap,BTreeSet};
use tokio::time::Instant;
use super::shared_cache_authority::{SharedImportPermit,SharedBackendQualification,SharedRankRetain,VerifiedSharedRetain};
use super::frontend_clock::ClockBounds;
#[path="reservation_ack.rs"]
mod reservation_ack;
use reservation_ack::ReservationAck;
pub use reservation_ack::{accept_reservation_ack,signed_control_query,reprepare_after_cancel,cancel_dispatch};
pub(crate) use reservation_ack::signed_reservation_body;
pub(crate) use reservation_ack::{verify_receipt_query_response,verify_read_source_unavailable};

#[derive(Clone,Debug)]
pub struct QualifiedWorker {
    /// From the selected worker's actual transport response URL, not receipt JSON.
    pub worker_url:String,
    pub native_origin_uuid:String,
    pub runtime_profile_digest:String,
    pub namespace_digest:String,
    /// Trusted native reserve handshake's clock identity, bound to origin.
    pub native_clock_domain:String,
    pub required_tiers:BTreeSet<String>,
    pub native_id_field:NativeIdField,
    /// Exact qualified topology and clock domains from every reserve ACK.
    /// Empty layouts cannot enter the public signed publication path.
    pub rank_tier_clocks:BTreeMap<u32,BTreeMap<String,String>>,
}
#[derive(Clone,Copy,Debug)]
pub enum NativeIdField { Rid, RequestId }
impl NativeIdField {fn key(self)->&'static str {match self {Self::Rid=>"rid",Self::RequestId=>"request_id"}}}
impl QualifiedWorker {
    pub fn registry_runtime(&self)->[u8;32] {
        let mut h=Sha256::new();h.update(b"dynamo-native-profile-binding-v1\0");h.update(self.runtime_profile_digest.as_bytes());h.finalize().into()
    }
}

#[derive(Clone,Debug)]
pub struct RenderPoint {
    pub endpoint_id:String,
    pub native_end_byte:u64,
    pub native_mark_tokens:u64,
    pub native_page_tokens:u64,
    /// Exact page-prefix digest emitted by the canonical native full renderer.
    pub prefix:Prefix,
    pub ttl:Ttl,
}

#[derive(Clone,Debug)]
pub struct RequestOrigin { request_id:String, native_request_id:String, started_at_ms:u64, started:Instant }
impl RequestOrigin {
    pub(crate) fn request_started_at_ms(&self)->u64 {self.started_at_ms}
    pub(crate) fn operation_deadline(&self)->Result<Instant> {
        self.started.checked_add(Duration::from_millis(10_830_000)).ok_or(Error::Evidence("cache operation deadline overflow"))
    }
    /// Pass the existing HTTP boundary's captured server-owned context, never
    /// an incoming JSON request_started_at_ms. Wall time is logical registry
    /// time only; elapsed/retention deductions use this process's monotonic time.
    pub fn from_frontend_context(request_id:String,native_request_id:String,started_at_ms:u64,started:Instant)->Result<Self> {
        if request_id.is_empty() || request_id.len()>1024 || native_request_id.is_empty() || native_request_id.len()>1024 || started>Instant::now() {
            return Err(Error::Evidence("invalid Frontend request context"));
        }
        if started.elapsed()>=Duration::from_millis(10_830_000) {return Err(Error::Evidence("private cache operation window expired"));}
        Ok(Self{request_id,native_request_id,started_at_ms,started})
    }
    fn elapsed_ms(&self,now:Instant)->Result<u64> {
        let d=now.checked_duration_since(self.started).ok_or(Error::Evidence("Frontend monotonic reversal"))?;
        // Round deduction UP, so transport never receives an extra fractional
        // millisecond from the authority's own elapsed accounting.
        let ms=u64::try_from(d.as_nanos().div_ceil(1_000_000)).map_err(|_|Error::Evidence("elapsed overflow"))?;
        if ms>=10_830_000 {return Err(Error::Evidence("private cache operation window expired"));}
        Ok(ms)
    }
    fn authority_now_ms(&self,now:Instant)->Result<u64> {
        self.started_at_ms.checked_add(self.elapsed_ms(now)?).ok_or(Error::Evidence("authority clock overflow"))
    }
}

#[derive(Clone,Debug)]
pub struct Dispatch {
    origin:RequestOrigin,
    worker:QualifiedWorker,
    reservation_id:String,
    authority_revision:i64,
    points:BTreeMap<String,RenderPoint>,
    retain_for_ms:BTreeMap<String,u64>,
    origin_leases:BTreeMap<String,super::lease_authority::OriginLease>,
    signed_claims:Option<Value>,
    reserve_claims:Option<Value>,
    reserve_ack:Option<ReservationAck>,
    control_claims:Option<Value>,
    selected_read:Option<SelectedRead>,
    /// Existing /short fencing permit epoch when present; never a client epoch.
    pub generation_epoch:Option<u64>,
}
#[derive(Clone,Debug)]
pub struct SelectedRead { point:RenderPoint, permit:ReadPermit, shared_import:Option<SharedImportPermit> }
/// Actual read-only authority lookup on native full-rendered lookback endpoints.
/// No count mutation. A selected worker must still own the recorded incarnation.
pub async fn lookup_best_read(store:&EtcdStore,origin:&RequestOrigin,worker:&QualifiedWorker,lookback:&[RenderPoint],now_upper_ms:u64)->Result<Option<SelectedRead>> {
    if lookback.len()>80 {return Err(Error::Evidence("lookback outside protocol scope"));}
    let mut ordered:Vec<_>=lookback.iter().collect();
    ordered.sort_by_key(|p|std::cmp::Reverse((p.native_page_tokens,p.native_mark_tokens)));
    for p in ordered {
        if p.native_page_tokens==0 {continue};
        if p.prefix.tokens!=p.native_page_tokens || p.prefix.marked_tokens!=p.native_mark_tokens {return Err(Error::Evidence("lookback render boundary mismatch"));}
        if p.prefix.namespace!=worker.namespace_digest || p.prefix.runtime!=worker.registry_runtime() {return Err(Error::Evidence("lookback runtime/namespace mismatch"));}
        if let Some((ttl,source_expiry))=store.lookup_origin_ttl(&p.prefix,&worker.worker_url,&worker.native_origin_uuid,now_upper_ms).await? {
            let mut point=p.clone();point.ttl=ttl;
            return Ok(Some(SelectedRead{point,permit:ReadPermit{prefix:p.prefix.clone(),authorized_until_ms:source_expiry,request_started_at_ms:origin.started_at_ms,request_id:origin.request_id.clone()},shared_import:None}));
        }
    }
    Ok(None)
}

pub(crate) async fn lookup_shared_read(store:&EtcdStore,origin:&RequestOrigin,worker:&QualifiedWorker,lookback:&[RenderPoint],qualification:&SharedBackendQualification,clock:&ClockBounds)->Result<Option<SelectedRead>> {
    if lookback.len()>80 {return Err(Error::Evidence("shared lookback outside protocol scope"));}
    let mut ordered:Vec<_>=lookback.iter().collect();ordered.sort_by_key(|p|std::cmp::Reverse((p.native_page_tokens,p.native_mark_tokens)));
    for p in ordered {
        if p.native_page_tokens==0 {continue};
        if p.prefix.tokens!=p.native_page_tokens || p.prefix.marked_tokens!=p.native_mark_tokens || p.prefix.namespace!=worker.namespace_digest || p.prefix.runtime!=worker.registry_runtime() {return Err(Error::Evidence("shared actual render boundary/runtime mismatch"));}
        if let Some(import)=store.lookup_shared_marker(qualification,&p.prefix,clock).await? {
            let mut point=p.clone();point.ttl=import.ttl();
            return Ok(Some(SelectedRead{point,permit:ReadPermit{prefix:p.prefix.clone(),authorized_until_ms:import.source_expires_at_ms(),request_started_at_ms:origin.started_at_ms,request_id:origin.request_id.clone()},shared_import:Some(import)}));
        }
    }
    Ok(None)
}

#[derive(Clone,Debug,Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
struct ReservationRecord {
    request_id:String,native_request_id:String,worker_url:String,native_origin_uuid:String,
    runtime_profile_digest:String,namespace_digest:String,
    points:BTreeMap<String,ReservationPoint>,request_started_at_ms:u64,
    selected_read:Option<ReservationPoint>,
    selected_source_expires_at_ms:Option<u64>,
    selected_shared_import:Option<Value>,
    generation_epoch:Option<u64>,
}
#[derive(Clone,Debug,Serialize,Deserialize,PartialEq,Eq)]
#[serde(deny_unknown_fields)]
struct ReservationPoint { prefix:Prefix,native_end_byte:u64,native_mark_tokens:u64,native_page_tokens:u64,ttl_ms:u64 }
fn reservation_points(d:&BTreeMap<String,RenderPoint>)->BTreeMap<String,ReservationPoint> {
    d.iter().map(|(id,p)|(id.clone(),reservation_point(p))).collect()
}
fn reservation_point(p:&RenderPoint)->ReservationPoint {ReservationPoint{prefix:p.prefix.clone(),native_end_byte:p.native_end_byte,native_mark_tokens:p.native_mark_tokens,native_page_tokens:p.native_page_tokens,ttl_ms:p.ttl.ms()}}
fn selected_shared_companion(read:&Option<SelectedRead>)->Result<Option<Value>> {
    read.as_ref().and_then(|r|r.shared_import.as_ref().map(|s|(s,&r.point.endpoint_id))).map(|(s,id)|s.companion(id)).transpose()
}

impl EtcdStore {
    /// Durable admission before the first native generation POST. Cancelling
    /// a reservation never removes this model/native-RID execution fence.
    pub(crate) async fn claim_generation(&self,d:&Dispatch)->Result<()> {
        d.origin.elapsed_ms(Instant::now())?;
        let claims=d.signed_claims.as_ref().ok_or(Error::Evidence("generation lacks signed dispatch"))?;
        let digest=cache_auth::digest(&json!(["native-generation",claims["model"],d.origin.native_request_id])).map_err(Error::Store)?;
        let key=format!("{ROOT}executions/{digest}");
        let reservation=format!("{ROOT}reservations/{}",d.reservation_id);
        let current=bounded(self.client.clone().get(reservation.as_str(),None)).await?;
        let row=current.kvs().first().ok_or(Error::Evidence("generation reservation missing"))?;
        let receipt=serde_json::to_vec(&json!({"native_request_id":d.origin.native_request_id,"reservation_id":d.reservation_id,"authority_revision":d.authority_revision,
            "model":claims["model"],"native_origin":d.worker.worker_url,"operation_id":claims["operation_id"],"nonce":claims["nonce"]})).map_err(|e|Error::Store(e.to_string()))?;
        let result=bounded(self.client.clone().txn(Txn::new().when(vec![Compare::mod_revision(reservation,CompareOp::Equal,d.authority_revision),Compare::version(key.as_str(),CompareOp::Equal,0)])
            .and_then(vec![TxnOp::put(key,receipt,Some(PutOptions::new().with_lease(row.lease())))]))).await?;
        if !result.succeeded() {return Err(Error::Evidence("native request already dispatched or reservation cancelled"));}
        Ok(())
    }
    async fn reserve_dispatch(&self,record:&ReservationRecord,hold_ms:u64,window_remaining_ms:u64)->Result<(String,i64)> {
        if hold_ms>3_600_000 {return Err(Error::Evidence("invalid dispatch retention"));}
        let bytes=serde_json::to_vec(record).map_err(|e|Error::Store(e.to_string()))?;
        let mut digest=Sha256::new();digest.update(b"dynamo-cache-reservation-v1\0");digest.update(&bytes);
        let id=format!("{:x}",digest.finalize());let key=format!("{ROOT}reservations/{id}");
        let mut c=self.client.clone();
        // Reservation lifecycle alone is not a cache lifetime or hit authority.
        // Dispatch metadata must outlive the existing longest generation
        // timeout (10800s), including an already-expired/no-pin 5m plan.
        // Its lease is never cache eligibility and terminal cancel removes it.
        let lease=bounded(c.lease_grant(window_remaining_ms.div_ceil(1000).saturating_add(2).max(5) as i64,None)).await?.id();
        let response=bounded(c.txn(Txn::new().when(vec![Compare::version(key.as_str(),CompareOp::Equal,0)])
            .and_then(vec![TxnOp::put(key.as_str(),bytes,Some(PutOptions::new().with_lease(lease)))]))).await?;
        if !response.succeeded() {bounded(c.lease_revoke(lease)).await?;return Err(Error::Evidence("duplicate cache dispatch"));}
        let revision=response.header().map_or(0,|h|h.revision());
        if revision<=0 {return Err(Error::Store("invalid reservation revision".into()));}
        Ok((id,revision))
    }
    async fn confirm_dispatch(&self,d:&Dispatch)->Result<()> {
        let key=format!("{ROOT}reservations/{}",d.reservation_id);
        let response=bounded(self.client.clone().get(key,None)).await?;
        let kv=response.kvs().first().ok_or(Error::Evidence("cache reservation expired/missing"))?;
        if response.kvs().len()!=1 || kv.mod_revision()!=d.authority_revision {
            return Err(Error::Evidence("cache reservation revision changed"));
        }
        let r:ReservationRecord=serde_json::from_slice(kv.value()).map_err(|e|Error::Store(e.to_string()))?;
        if r.request_id!=d.origin.request_id || r.native_request_id!=d.origin.native_request_id || r.worker_url!=d.worker.worker_url
            || r.native_origin_uuid!=d.worker.native_origin_uuid || r.runtime_profile_digest!=d.worker.runtime_profile_digest
            || r.namespace_digest!=d.worker.namespace_digest || r.request_started_at_ms!=d.origin.started_at_ms
            || r.points!=reservation_points(&d.points) || r.generation_epoch!=d.generation_epoch
            || r.selected_read!=d.selected_read.as_ref().map(|s|reservation_point(&s.point)) {
            return Err(Error::Evidence("cache reservation binding changed"));
        }
        if r.selected_shared_import!=selected_shared_companion(&d.selected_read)? {return Err(Error::Evidence("selected shared import source changed"));}
        if r.selected_source_expires_at_ms!=d.selected_read.as_ref().map(|r|r.permit.authorized_until_ms) {return Err(Error::Evidence("selected original source expiry changed"));}
        Ok(())
    }
}

/// Called after actual native full rendering and selected-worker reservation,
/// before native generation. Uses the existing etcd client in this Frontend.
pub async fn prepare_dispatch(store:&EtcdStore,origin:RequestOrigin,worker:QualifiedWorker,points:Vec<RenderPoint>,selected_read:Option<SelectedRead>,generation_epoch:Option<u64>)->Result<Dispatch> {
    if points.len()>4 || worker.required_tiers.is_empty()
        || [&worker.worker_url,&worker.native_origin_uuid,&worker.runtime_profile_digest,&worker.namespace_digest,&worker.native_clock_domain].iter().any(|s|s.is_empty()) {
        return Err(Error::Evidence("unqualified worker/render plan"));
    }
    let elapsed=origin.elapsed_ms(Instant::now())?;
    let mut by_id=BTreeMap::new();let mut durations=BTreeMap::new();
    for point in points {
        if point.native_page_tokens>0 {point.prefix.key()?;}
        if point.endpoint_id.is_empty()
            || point.native_page_tokens>point.native_mark_tokens || point.prefix.tokens!=point.native_page_tokens
            || point.prefix.marked_tokens!=point.native_mark_tokens {
            return Err(Error::Evidence("invalid actual rendered endpoint"));
        }
        if point.prefix.namespace!=worker.namespace_digest || point.prefix.runtime!=worker.registry_runtime() {
            return Err(Error::Evidence("rendered prefix namespace/runtime not qualified worker"));
        }
        let remain=if point.native_page_tokens>0 {point.ttl.ms().saturating_sub(elapsed)}else{0};
        if by_id.insert(point.endpoint_id.clone(),point.clone()).is_some() {
            return Err(Error::Evidence("duplicate endpoint id"));
        }
        durations.insert(point.endpoint_id.clone(),remain);
    }
    if let Some(read)=&selected_read {
        if read.permit.request_id!=origin.request_id || read.permit.request_started_at_ms!=origin.started_at_ms {
            return Err(Error::Evidence("selected read belongs to another request"));
        }
        if by_id.get(&read.point.endpoint_id).is_some_and(|p|reservation_point(p)!=reservation_point(&read.point)) {
            return Err(Error::Evidence("read/write endpoint identity collides"));
        }
        durations.insert(read.point.endpoint_id.clone(),read.point.ttl.ms().saturating_sub(elapsed));
    }
    let mut origin_leases=BTreeMap::new();
    for (id,p) in by_id.iter().chain(selected_read.iter().map(|r|(&r.point.endpoint_id,&r.point))) {
        if p.native_page_tokens==0 {continue};
        let remaining=p.ttl.ms().saturating_sub(origin.elapsed_ms(Instant::now())?);
        if let Some(lease)=store.grant_origin_lease(remaining).await? {origin_leases.insert(id.clone(),lease);}
    }
    let record=ReservationRecord{request_id:origin.request_id.clone(),native_request_id:origin.native_request_id.clone(),worker_url:worker.worker_url.clone(),native_origin_uuid:worker.native_origin_uuid.clone(),runtime_profile_digest:worker.runtime_profile_digest.clone(),namespace_digest:worker.namespace_digest.clone(),points:reservation_points(&by_id),request_started_at_ms:origin.started_at_ms,selected_read:selected_read.as_ref().map(|s|reservation_point(&s.point)),selected_source_expires_at_ms:selected_read.as_ref().map(|r|r.permit.authorized_until_ms),selected_shared_import:selected_shared_companion(&selected_read)?,generation_epoch};
    let (reservation_id,authority_revision)=store.reserve_dispatch(&record,durations.values().copied().max().unwrap_or(0),10_830_000-elapsed).await?;
    Ok(Dispatch{origin,worker,reservation_id,authority_revision,points:by_id,retain_for_ms:durations,origin_leases,signed_claims:None,reserve_claims:None,reserve_ack:None,control_claims:None,selected_read,generation_epoch})
}
impl Dispatch {
    pub(crate) fn origin(&self)->RequestOrigin {self.origin.clone()}
    pub(crate) fn native_request_id(&self)->&str {&self.origin.native_request_id}
    /// Attach immediately before the SAME existing native generation POST. The
    /// native worker pins receive_monotonic + retain_for; network delay may only
    /// retain physical pages longer. Logical expiry never changes.
    fn attach_private_fields(&mut self,body:&mut Value)->Result<()> {
        if body.get(self.worker.native_id_field.key()).and_then(Value::as_str)!=Some(self.origin.native_request_id.as_str()) {
            return Err(Error::Evidence("canonical native body ID differs from admitted server ID"));
        }
        // The measured render selectors carry prompt order. Endpoint UUIDs are
        // lookup identities; sorting them must never reorder native write plans.
        let selectors=match body.get("prompt_cache_blocks") {
            Some(value)=>value.as_array().ok_or(Error::Evidence("native write selectors not an array"))?.as_slice(),
            None if self.points.is_empty()=>&[],
            None=>return Err(Error::Evidence("native write selectors missing")),
        };
        let mut seen=BTreeSet::new();let mut ordered=Vec::with_capacity(selectors.len());
        for selector in selectors {
            let id=selector["endpoint_id"].as_str().ok_or(Error::Evidence("native write selector ID missing"))?;
            if !seen.insert(id) || !self.points.contains_key(id) {return Err(Error::Evidence("native write selector identity differs from dispatch"));}
            ordered.push(id);
        }
        if seen!=self.points.keys().map(String::as_str).collect() {return Err(Error::Evidence("native write selectors omitted dispatched point"));}
        let elapsed=self.origin.elapsed_ms(Instant::now())?;
        let mut points=vec![];
        for id in ordered {
            let p=&self.points[id];
            // The acknowledged native EndpointPlan is immutable. Its physical
            // deadline was fixed by Reserve; Gen must not recompute its duration.
            let remaining=if self.reserve_ack.is_some() {
                *self.retain_for_ms.get(id).ok_or(Error::Evidence("acknowledged endpoint duration missing"))?
            }else{
                let remain=if self.origin_leases.contains_key(id) {p.ttl.ms().saturating_sub(elapsed)}else{0};
                self.retain_for_ms.insert(id.to_owned(),remain);remain
            };
            points.push(json!({"endpoint_id":id,"native_end_byte":p.native_end_byte,"native_mark_tokens":p.native_mark_tokens,"native_page_tokens":p.native_page_tokens,
                "native_page_prefix_digest":hex(&p.prefix.digest),"native_mark_prefix_digest":hex(&p.prefix.marked_endpoint_digest),"native_mark_render_prefix_digest":hex(&p.prefix.marked_render_digest),
                "ttl_ms":p.ttl.ms(),"expires_at_ms":self.origin.started_at_ms.checked_add(p.ttl.ms()).ok_or(Error::Evidence("expiry overflow"))?,
                "retain_for_ms":remaining,"no_pin":remaining==0,"required_tiers":self.worker.required_tiers}));
        }
        let map=body.as_object_mut().ok_or(Error::Evidence("native body not object"))?;
        for key in ["cache_request_started_at_ms","authority_revision","reservation_id","prompt_cache_commit_points","prompt_cache_read_commit_point","prompt_cache_read_max_tokens","prompt_cache_max_tokens","prompt_cache_shared_read","prompt_cache_read_source_expires_at_ms"] {
            if map.contains_key(key) {return Err(Error::Evidence("private cache fields were prepopulated"));}
        }
        map.insert("cache_request_started_at_ms".into(),self.origin.started_at_ms.into());
        map.insert("authority_revision".into(),self.authority_revision.into());
        map.insert("reservation_id".into(),self.reservation_id.clone().into());
        map.insert("prompt_cache_commit_points".into(),points.into());
        map.insert("prompt_cache_max_tokens".into(),self.points.values().map(|p|p.native_page_tokens).max().unwrap_or(0).into());
        let read_max=self.selected_read.as_ref().filter(|s|if self.reserve_ack.is_some() {
            self.retain_for_ms.get(&s.point.endpoint_id).is_some_and(|n|*n>0)
        }else{s.point.ttl.ms()>elapsed && self.origin_leases.contains_key(&s.point.endpoint_id)}).map_or(0,|s|s.point.native_page_tokens);
        map.insert("prompt_cache_read_max_tokens".into(),read_max.into());
        if let Some(s)=&self.selected_read {
            let remain=if self.reserve_ack.is_some() {
                *self.retain_for_ms.get(&s.point.endpoint_id).ok_or(Error::Evidence("acknowledged read duration missing"))?
            }else{
                let remain=if self.origin_leases.contains_key(&s.point.endpoint_id) {s.point.ttl.ms().saturating_sub(elapsed)}else{0};
                self.retain_for_ms.insert(s.point.endpoint_id.clone(),remain);remain
            };
            if remain==0 {return Ok(())};
            map.insert("prompt_cache_read_source_expires_at_ms".into(),s.permit.authorized_until_ms.into());
            map.insert("prompt_cache_read_commit_point".into(),json!({"endpoint_id":s.point.endpoint_id,"native_end_byte":s.point.native_end_byte,"native_mark_tokens":s.point.native_mark_tokens,"native_page_tokens":s.point.native_page_tokens,"native_page_prefix_digest":hex(&s.point.prefix.digest),"native_mark_prefix_digest":hex(&s.point.prefix.marked_endpoint_digest),"native_mark_render_prefix_digest":hex(&s.point.prefix.marked_render_digest),"ttl_ms":s.point.ttl.ms(),"expires_at_ms":self.origin.started_at_ms+s.point.ttl.ms(),"retain_for_ms":remain,"no_pin":false,"required_tiers":self.worker.required_tiers}));
            if let Some(import)=&s.shared_import {map.insert("prompt_cache_shared_read".into(),import.companion(&s.point.endpoint_id)?);}
        }
        Ok(())
    }
}

/// Values obtained from the qualified native renderer/profile and fresh clock
/// qualification. The controller must gate their actual weights/source proof.
/// This struct carries no default profile, guessed digest, or clock fallback.
#[derive(Clone)]
pub struct SigningContext {
    pub model:String,pub source_digest:String,pub full_render_digest:String,pub token_ids_digest:String,
    pub operation_id:String,pub nonce:String,
    pub now_lower_ms:u64,pub now_upper_ms:u64,
    pub clock_observed_at:Instant,
}
impl SigningContext {
    fn refreshed(&self,clock:&ClockBounds)->Result<Self> {
        clock.ensure_fresh().map_err(|_|Error::Evidence("clock qualification must be refreshed before signing"))?;
        Ok(Self{model:self.model.clone(),source_digest:self.source_digest.clone(),full_render_digest:self.full_render_digest.clone(),
            token_ids_digest:self.token_ids_digest.clone(),operation_id:self.operation_id.clone(),nonce:self.nonce.clone(),
            now_lower_ms:clock.lower_ms(),now_upper_ms:clock.upper_ms(),clock_observed_at:clock.observed_at()})
    }
    pub(crate) fn ensure_fresh(&self)->Result<()> {
        if Instant::now().checked_duration_since(self.clock_observed_at).ok_or(Error::Evidence("qualified clock monotonic reversal"))?>Duration::from_millis(100) {
            return Err(Error::Evidence("clock qualification must be refreshed before signing"));
        }
        Ok(())
    }
}

/// Final immediate send hook: revalidate a selected logical entry, cancel its
/// old immutable reservation if expired, prepare a new revision with read zero,
/// attach private scopes and sign the existing single generation POST body.
async fn validate_generation_dispatch(store:&EtcdStore,d:&mut Dispatch,c:&SigningContext)->Result<()> {
    if d.worker.rank_tier_clocks.is_empty() || c.now_upper_ms<c.now_lower_ms || c.now_upper_ms-c.now_lower_ms>200
        || c.now_lower_ms<d.origin.started_at_ms {return Err(Error::Evidence("clock/topology qualification missing"));}
    if let Some(reserve)=&d.reserve_claims {
        for (key,value) in [("model",&c.model),("source_digest",&c.source_digest),("full_render_digest",&c.full_render_digest),("token_ids_digest",&c.token_ids_digest)] {
            if reserve[key]!=value.as_str() {return Err(Error::Evidence("generation renderer/source changed after native reservation"));}
        }
    }
    let elapsed=d.origin.elapsed_ms(Instant::now())?;
    if d.reserve_ack.is_some() && d.points.iter().any(|(id,p)|d.retain_for_ms.get(id).is_some_and(|n|*n>0) && elapsed>=p.ttl.ms()) {
        return Err(Error::Evidence("write endpoint expired after native reserve"));
    }
    let clock_age=Instant::now().checked_duration_since(c.clock_observed_at).ok_or(Error::Evidence("qualified clock monotonic reversal"))?;
    if clock_age>Duration::from_millis(100) {return Err(Error::Evidence("clock qualification must be refreshed before signing"));}
    let lookup_upper=c.now_upper_ms.checked_add(u64::try_from(clock_age.as_nanos().div_ceil(1_000_000)).map_err(|_|Error::Evidence("clock elapsed overflow"))?).ok_or(Error::Evidence("clock overflow"))?;
    let stale=if let Some(r)=&d.selected_read {
        elapsed>=r.point.ttl.ms() || lookup_upper>=r.permit.authorized_until_ms || if let Some(import)=&r.shared_import {!store.confirm_shared_import_at_upper(import,lookup_upper).await?}else{!store.lookup_origin_markers(&r.point.prefix,&d.worker.worker_url,&d.worker.native_origin_uuid,lookup_upper).await?}
    }else{false};
    if stale {
        if d.reserve_ack.is_some() {return Err(Error::Evidence("selected read expired after native reserve: authenticated native cancel required before reprepare"));}
        let key=format!("{ROOT}reservations/{}",d.reservation_id);
        let reply=bounded(store.client.clone().txn(Txn::new().when(vec![Compare::mod_revision(key.as_str(),CompareOp::Equal,d.authority_revision)])
            .and_then(vec![TxnOp::delete(key,None)]))).await?;
        if !reply.succeeded() {return Err(Error::Evidence("cannot cancel stale immutable read reservation"));}
        *d=prepare_dispatch(store,d.origin.clone(),d.worker.clone(),d.points.values().cloned().collect(),None,d.generation_epoch).await?;
    }
    store.confirm_dispatch(d).await?;
    Ok(())
}

pub(crate) async fn signed_generation_body<F,Fut>(store:&EtcdStore,d:&mut Dispatch,body:&mut Value,secret:&[u8],c:&SigningContext,fresh_clock:F)->Result<()>
where F:FnOnce(&cache_auth::PrehashedPayload)->Fut,Fut:std::future::Future<Output=Result<ClockBounds>> {
    validate_generation_dispatch(store,d,c).await?;
    let mut candidate=body.clone();d.attach_private_fields(&mut candidate)?;
    if let Some(existing)=candidate.get("prompt_cache_canonical_body") {
        let expected=serde_json::to_string(&cache_auth::ordinary_body_for(&candidate).map_err(Error::Store)?)
            .map_err(|e|Error::Store(e.to_string()))?;
        if existing.as_str()!=Some(expected.as_str()) {
            return Err(Error::Evidence("existing canonical native body differs from generation body"));
        }
    }else{
        cache_auth::attach_canonical_body(&mut candidate).map_err(Error::Store)?;
    }
    if d.retain_for_ms.values().any(|n|*n>0) && d.reserve_ack.is_none() {return Err(Error::Evidence("generation requires actual all-rank reserve ACK"));}
    if body.get("model").and_then(Value::as_str)!=Some(c.model.as_str()) {return Err(Error::Evidence("native served model differs from trusted signing context"));}
    let ids=[body.get("rid"),body.get("request_id")].into_iter().flatten().count();
    if ids!=1 {return Err(Error::Evidence("exactly one canonical native request ID required"));}
    // Finish exact ordinary/canonical/private-body traversal before qualifying
    // the clock that will immediately precede the actual HMAC.
    let payload=cache_auth::payload_for("generate",&candidate).map_err(Error::Store)?;
    let prepared=cache_auth::PrehashedPayload::new(&payload).map_err(Error::Store)?;
    let clock=fresh_clock(&prepared).await?;
    let c=c.refreshed(&clock)?;
    let proof=generation_proof(d,secret,&c,&prepared)?;
    d.signed_claims=Some(proof["claims"].clone());
    candidate.as_object_mut().unwrap().insert(cache_auth::PROOF_FIELD.into(),proof);
    *body=candidate;
    Ok(())
}

fn generation_proof(d:&Dispatch,secret:&[u8],c:&SigningContext,prepared:&cache_auth::PrehashedPayload)->Result<Value> {
    if c.now_upper_ms<c.now_lower_ms || c.now_upper_ms-c.now_lower_ms>200 || c.now_lower_ms<d.origin.started_at_ms {
        return Err(Error::Evidence("clock/topology qualification missing"));
    }
    let clock_elapsed=Instant::now().checked_duration_since(c.clock_observed_at).ok_or(Error::Evidence("qualified clock monotonic reversal"))?;
    if clock_elapsed>Duration::from_millis(100) {return Err(Error::Evidence("clock qualification must be refreshed before signing"));}
    let advance=u64::try_from(clock_elapsed.as_nanos().div_ceil(1_000_000)).map_err(|_|Error::Evidence("clock elapsed overflow"))?;
    let issued=c.now_upper_ms.checked_add(advance).ok_or(Error::Evidence("qualified clock overflow"))?;
    // Keeping the ACKed plan does not renew its logical authorization. If a
    // write expired during preparation, the owned lifecycle cancels it instead.
    let elapsed=d.origin.elapsed_ms(Instant::now())?;
    for (id,point) in &d.points {
        if d.retain_for_ms.get(id).is_some_and(|n|*n>0)
            && (elapsed>=point.ttl.ms() || issued>=d.origin.started_at_ms.checked_add(point.ttl.ms()).ok_or(Error::Evidence("expiry overflow"))?) {
            return Err(Error::Evidence("write endpoint expired after native reserve"));
        }
    }
    // Expiry can cross while the large payload is being prepared. Never sign a
    // previously selected source using the earlier lookup's clock estimate.
    if let Some(read)=&d.selected_read {
        if d.origin.elapsed_ms(Instant::now())?>=read.point.ttl.ms() || issued>=read.permit.authorized_until_ms {
            return Err(Error::Evidence("selected read expired after native reserve: authenticated native cancel required before reprepare"));
        }
    }
    if d.origin.elapsed_ms(Instant::now())?>=10_830_000 {return Err(Error::Evidence("private cache operation window expired"));}
    let remaining=d.retain_for_ms.values().copied().max().unwrap_or(0);
    let deadline=c.now_lower_ms.checked_add(advance).and_then(|n|n.checked_add(1000)).ok_or(Error::Evidence("operation deadline overflow"))?.min(d.origin.started_at_ms.saturating_add(10_830_000));
    let claims=json!({"version":1,"operation":"generate","operation_id":c.operation_id,"nonce":c.nonce,
        "model":c.model,"native_origin":d.worker.worker_url,"native_incarnation":d.worker.native_origin_uuid,
        "source_digest":c.source_digest,"runtime_profile_digest":d.worker.runtime_profile_digest,
        "reservation_id":d.reservation_id,"authority_revision":d.authority_revision,"native_request_id":d.origin.native_request_id,
        "namespace_digest":d.worker.namespace_digest,"full_render_digest":c.full_render_digest,"token_ids_digest":c.token_ids_digest,
        "request_started_at_ms":d.origin.started_at_ms,"issued_at_ms":issued,"deadline_ms":deadline,
        "retain_for_ms":remaining,"payload_digest":prepared.digest()});
    c.ensure_fresh()?;
    cache_auth::sign_prehashed(secret,&claims,prepared).map_err(Error::Store)
}

/// The large, immutable body is hashed and encoded before the final clock.
/// Only the small authentication envelope is encoded after that sample.
struct GenerationPost {
    payload:cache_auth::PrehashedPayload,
    prefix:Vec<u8>,
    has_fields:bool,
}
impl GenerationPost {
    fn prepare(body:&Value)->Result<Self> {
        let proof_size=serde_json::to_vec(body.get(cache_auth::PROOF_FIELD).ok_or(Error::Evidence("prepared generation proof absent"))?).map_err(|e|Error::Store(e.to_string()))?.len();
        let mut unsigned=body.clone();
        let object=unsigned.as_object_mut().ok_or(Error::Evidence("generation body must be an object"))?;
        object.remove(cache_auth::PROOF_FIELD);
        let has_fields=!object.is_empty();
        let payload=cache_auth::PrehashedPayload::new(&cache_auth::payload_for("generate",&unsigned).map_err(Error::Store)?).map_err(Error::Store)?;
        let mut prefix=serde_json::to_vec(&unsigned).map_err(|e|Error::Store(e.to_string()))?;
        if prefix.pop()!=Some(b'}') {return Err(Error::Evidence("generation JSON object encoding invalid"));}
        // All final claim strings are unchanged; only bounded integer times can
        // grow. Reserve before sampling, so appending cannot copy the big body.
        prefix.reserve(proof_size.checked_add(512).ok_or(Error::Evidence("generation envelope size overflow"))?);
        Ok(Self{payload,prefix,has_fields})
    }
    fn finish(mut self,proof:&Value)->Result<(Vec<u8>,cache_auth::PrehashedPayload)> {
        let mut suffix=Vec::new();
        if self.has_fields {suffix.push(b',');}
        serde_json::to_writer(&mut suffix,cache_auth::PROOF_FIELD).map_err(|e|Error::Store(e.to_string()))?;
        suffix.push(b':');
        serde_json::to_writer(&mut suffix,proof).map_err(|e|Error::Store(e.to_string()))?;
        suffix.push(b'}');
        if suffix.len()>self.prefix.capacity()-self.prefix.len() {return Err(Error::Evidence("generation envelope exceeded prepared capacity"));}
        self.prefix.extend_from_slice(&suffix);
        Ok((self.prefix,self.payload))
    }
}

pub(crate) async fn signed_generation_post<F,Fut>(store:&EtcdStore,d:&mut Dispatch,body:&Value,secret:&[u8],c:&SigningContext,deadline:Instant,fresh_clock:F)->Result<(Vec<u8>,cache_auth::PrehashedPayload)>
where F:FnOnce()->Fut,Fut:std::future::Future<Output=Result<ClockBounds>> {
    let original=d.signed_claims.as_ref().ok_or(Error::Evidence("prepared generation claims absent"))?;
    let mut c=c.clone();
    c.operation_id=original["operation_id"].as_str().ok_or(Error::Evidence("generation operation ID absent"))?.into();
    c.nonce=original["nonce"].as_str().ok_or(Error::Evidence("generation nonce absent"))?.into();
    let original_payload_digest=original["payload_digest"].as_str().ok_or(Error::Evidence("generation payload digest absent"))?.to_owned();
    validate_generation_dispatch(store,d,&c).await?;
    store.claim_generation(d).await?;
    let post=GenerationPost::prepare(body)?;
    if post.payload.digest()!=original_payload_digest {return Err(Error::Evidence("prepared generation payload changed before POST"));}
    let clock=fresh_clock().await?;
    let c=c.refreshed(&clock)?;
    if Instant::now()>=deadline {return Err(Error::Evidence("native cache control deadline expired"));}
    let proof=generation_proof(d,secret,&c,&post.payload)?;
    d.signed_claims=Some(proof["claims"].clone());
    post.finish(&proof)
}

#[derive(Debug,Deserialize)]
struct Receipt {
    reservation_id:String,authority_revision:i64,native_origin_uuid:String,
    runtime_profile_digest:String,namespace_digest:String,endpoint_id:String,
    native_mark_tokens:u64,published_tokens:u64,actual_read_tokens:u64,
    request_started_at_ms:u64,rank_receipts:Vec<RankReceipt>,
}
#[derive(Debug,Deserialize)]
struct RankReceipt {
    rank_id:u32,pin_receipt_id:String,published_tokens:u64,actual_read_tokens:u64,tier_receipts:Vec<TierReceipt>,
    /// Absent evidence stays unknown. Published retained prefixes are not fresh writes.
    #[serde(default)] fresh_write_ranges:Option<Vec<FreshWriteRange>>,
}
#[derive(Debug,Deserialize)]
#[serde(deny_unknown_fields)]
struct FreshWriteRange {start_tokens:u64,end_tokens:u64}
#[derive(Debug,Deserialize)]
struct TierReceipt {
    tier:String,retained_tokens:u64,native_clock_domain:String,
    receive_monotonic_ns:u64,monotonic_event_ns:u64,
    retain_until_monotonic_ns:u64,ack_retain_for_ms:u64,
    #[serde(default)] pin_receipt_id:Option<String>,
    #[serde(default)] source_read_upper_ms:Option<u64>,
    #[serde(default)] source_read_clock_error_ms:Option<u64>,
    #[serde(default)] event_upper_ms:Option<u64>,
    #[serde(default)] event_clock_error_ms:Option<u64>,
    #[serde(default)] backend_namespace_digest:Option<String>,
    #[serde(default)] backend_runtime_digest:Option<String>,
    #[serde(default)] shared_object_digest:Option<String>,
    #[serde(default)] imported_shared_object_digest:Option<String>,
}
/// Concrete verifier is constructed ONLY after actual receipt parsing and
/// durable etcd reservation confirmation, never from public fields.
pub struct VerifiedReceipt { evidence:NativeEvidence }
#[async_trait]
impl NativeVerifier for VerifiedReceipt {
    async fn verify(&self,e:&NativeEvidence)->Result<()> {
        if e!=&self.evidence {return Err(Error::Evidence("receipt evidence was altered"));} Ok(())
    }
}

fn parse_receipt(d:&Dispatch,actual_response_worker_url:&str,actual_native_request_id:&str,raw:&Value,kind:EventKind,now:Instant)->Result<(VerifiedReceipt,Ttl)> {
    let r:Receipt=serde_json::from_value(raw.clone()).map_err(|error| {
        // serde's missing-field name comes from the Rust schema. Other data
        // errors can contain packet values, so only a constant label survives.
        let text=error.to_string();
        let missing=text.strip_prefix("missing field `").and_then(|s|s.strip_suffix('`'))
            .filter(|s|s.len()<=64 && s.bytes().all(|b|b.is_ascii_lowercase()||b.is_ascii_digit()||b==b'_'));
        tracing::warn!(target:"dynamo_llm::native_cache",stage="receipt_schema",native_request_id=%d.origin.native_request_id,
            missing_field=missing.unwrap_or("invalid_field_type"),"native cache receipt rejected");
        Error::Evidence("malformed actual native receipt")
    })?;
    let p=match kind {
        EventKind::WriteCommitted=>d.points.get(&r.endpoint_id),
        EventKind::ReadHit=>d.selected_read.as_ref().filter(|s|s.point.endpoint_id==r.endpoint_id).map(|s|&s.point),
    }.ok_or(Error::Evidence("receipt endpoint not dispatched for event kind"))?;
    for (field,matches) in [("response_worker",actual_response_worker_url==d.worker.worker_url),
        ("native_request_id",actual_native_request_id==d.origin.native_request_id),
        ("reservation_id",r.reservation_id==d.reservation_id),("authority_revision",r.authority_revision==d.authority_revision),
        ("native_origin_uuid",r.native_origin_uuid==d.worker.native_origin_uuid),("runtime_profile_digest",r.runtime_profile_digest==d.worker.runtime_profile_digest),
        ("namespace_digest",r.namespace_digest==d.worker.namespace_digest),("request_started_at_ms",r.request_started_at_ms==d.origin.started_at_ms),
        ("native_mark_tokens",r.native_mark_tokens==p.native_mark_tokens),("rank_tier_clocks",!d.worker.rank_tier_clocks.is_empty())] {
        if !matches {
            tracing::warn!(target:"dynamo_llm::native_cache",stage="receipt_binding",native_request_id=%d.origin.native_request_id,field,"native cache receipt rejected");
            return Err(Error::Evidence("native receipt dispatch binding mismatch"));
        }
    }
    if d.origin.elapsed_ms(now)? >=10_830_000 {return Err(Error::Evidence("private cache receipt window expired"));}
    let expiry=d.origin.started_at_ms.checked_add(p.ttl.ms()).ok_or(Error::Evidence("expiry overflow"))?;
    let written_ok=r.published_tokens>=p.native_page_tokens && r.published_tokens<=p.native_mark_tokens;
    if r.actual_read_tokens>d.selected_read.as_ref().map_or(0,|s|s.point.native_page_tokens) {
        return Err(Error::Evidence("native read exceeded dispatched logical scope"));
    }
    if (kind==EventKind::WriteCommitted && !written_ok) || (kind==EventKind::ReadHit && r.actual_read_tokens<p.native_page_tokens) {
        return Err(Error::Evidence("native cache event does not cover endpoint"));
    }
    let wanted=*d.retain_for_ms.get(&r.endpoint_id).ok_or(Error::Evidence("missing dispatched hold"))?;
    if wanted==0 {return Err(Error::Evidence("zero hold cannot publish native receipt"));}
    {
        let mut ranks=BTreeSet::new();let mut published=u64::MAX;let mut read=u64::MAX;
        for rank in &r.rank_receipts {
            if !ranks.insert(rank.rank_id) || rank.pin_receipt_id.is_empty() {return Err(Error::Evidence("duplicate/unpinned rank"));}
            let clocks=d.worker.rank_tier_clocks.get(&rank.rank_id).ok_or(Error::Evidence("unexpected native rank"))?;
            if (kind==EventKind::WriteCommitted && (rank.published_tokens<p.native_page_tokens || rank.published_tokens>p.native_mark_tokens))
                || (kind==EventKind::ReadHit && rank.actual_read_tokens<p.native_page_tokens)
                || rank.actual_read_tokens>d.selected_read.as_ref().map_or(0,|s|s.point.native_page_tokens) {
                return Err(Error::Evidence("rank cache event exceeds dispatched logical scope"));
            }
            let mut tiers=BTreeSet::new();
            for t in &rank.tier_receipts {
                qualified_event_time(t,d.origin.started_at_ms,expiry)?;
                if !tiers.insert(t.tier.clone()) || clocks.get(&t.tier)!=Some(&t.native_clock_domain)
                    || t.retained_tokens<p.native_page_tokens || t.ack_retain_for_ms<wanted || t.ack_retain_for_ms>p.ttl.ms()
                    || t.pin_receipt_id.as_ref().is_some_and(|id|id!=&rank.pin_receipt_id) {
                    return Err(Error::Evidence("rank tier lacks matching reserved pin/clock"));
                }
                if let Some(ack)=&d.reserve_ack {
                    let a=ack.rank_acks.iter().find(|a|a.rank_id==rank.rank_id)
                        .and_then(|a|a.point_acks.iter().find(|a|a.endpoint_id==r.endpoint_id))
                        .and_then(|a|a.tier_acks.iter().find(|a|a.tier==t.tier)).ok_or(Error::Evidence("committed tier lacks immutable reserve ACK"))?;
                    if a.native_clock_domain!=t.native_clock_domain || a.receive_monotonic_ns!=t.receive_monotonic_ns
                        || a.retain_until_monotonic_ns!=t.retain_until_monotonic_ns || a.ack_retain_for_ms!=t.ack_retain_for_ms {
                        return Err(Error::Evidence("committed tier reset reserved deadline/clock"));
                    }
                }
                let until=t.receive_monotonic_ns.checked_add(t.ack_retain_for_ms.checked_mul(1_000_000).ok_or(Error::Evidence("rank duration overflow"))?).ok_or(Error::Evidence("rank deadline overflow"))?;
                if until!=t.retain_until_monotonic_ns || t.monotonic_event_ns<t.receive_monotonic_ns || t.monotonic_event_ns>=until {return Err(Error::Evidence("rank event outside actual pin"));}
                if rank.actual_read_tokens>0 {
                    let read=d.selected_read.as_ref().ok_or(Error::Evidence("read event missing original source permission"))?;
                    let upper=t.source_read_upper_ms.ok_or(Error::Evidence("actual source read clock missing"))?;
                    let error=t.source_read_clock_error_ms.ok_or(Error::Evidence("actual source read clock qualification missing"))?;
                    if error>100 || upper<d.origin.started_at_ms || upper>=read.permit.authorized_until_ms {return Err(Error::Evidence("actual source read was not held before original expiry"));}
                }
            }
            if tiers!=d.worker.required_tiers || clocks.keys().cloned().collect::<BTreeSet<_>>()!=tiers {return Err(Error::Evidence("rank missing required tier"));}
            published=published.min(rank.published_tokens);read=read.min(rank.actual_read_tokens);
        }
        if ranks!=d.worker.rank_tier_clocks.keys().copied().collect() || published!=r.published_tokens || read!=r.actual_read_tokens {
            return Err(Error::Evidence("all-rank count minimum/topology mismatch"));
        }
    }
    let at=r.rank_receipts.iter().flat_map(|r|r.tier_receipts.iter()).filter_map(|t|t.event_upper_ms).max().ok_or(Error::Evidence("actual native event clock missing"))?;
    // These authority-domain bounds are derived from the verified duration
    // promise. Native bare monotonic/event wall values are NEVER translated.
    let evidence=NativeEvidence{request_id:d.origin.request_id.clone(),native_request_id:actual_native_request_id.into(),prefix:p.prefix.clone(),worker:d.worker.worker_url.clone(),incarnation:r.native_origin_uuid,
        // Registry identity names the authenticated aggregate event. Physical
        // pin ownership remains the actual per-rank receipt IDs checked above.
        receipt_id:cache_auth::digest(raw).map_err(Error::Store)?,kind,observed_at_ms:at,request_started_at_ms:d.origin.started_at_ms,pinned_until_ms:expiry,committed_tokens:p.native_page_tokens};
    Ok((VerifiedReceipt{evidence},p.ttl))
}
fn qualified_event_time(t:&TierReceipt,request_start:u64,expiry:u64)->Result<u64> {
    let upper=t.event_upper_ms.ok_or(Error::Evidence("actual native event clock missing"))?;
    let error=t.event_clock_error_ms.ok_or(Error::Evidence("actual native event clock error missing"))?;
    if error>100 || upper<request_start || upper>=expiry {return Err(Error::Evidence("actual native event outside original logical TTL"));}
    Ok(upper)
}
fn hex(bytes:&[u8;32])->String {bytes.iter().map(|b|format!("{b:02x}")).collect()}

/// Native worker emits this private envelope from its actual Req/input ID;
/// neither the HTTP phase UUID nor a supplied public client field is accepted.
fn authenticated_event_point<'a>(d:&'a Dispatch,actual_response_worker_url:&str,packet:&Value,secret:&[u8])->Result<&'a RenderPoint> {
    if actual_response_worker_url!=d.worker.worker_url {return Err(Error::Evidence("native event transport mismatch"));}
    if d.worker.rank_tier_clocks.is_empty() {return Err(Error::Evidence("native rank layout unqualified"));}
    if d.reserve_ack.is_none() {return Err(Error::Evidence("native packet lacks previously saved reserve ACK"));}
    let request_claims=d.signed_claims.as_ref().ok_or(Error::Evidence("dispatch lacks signed actual generation"))?;
    let payload=cache_auth::payload_for("receipts",packet).map_err(Error::Store)?;
    let proof=packet.get(cache_auth::PROOF_FIELD).ok_or(Error::Evidence("private native response unsigned"))?;
    let claims=cache_auth::verify_signature(secret,proof,&payload).map_err(Error::Store)?;
    if claims["operation"]!="receipts" {return Err(Error::Evidence("native response operation mismatch"));}
    for key in ["model","native_origin","native_incarnation","source_digest","runtime_profile_digest","reservation_id","authority_revision","native_request_id","namespace_digest","full_render_digest","token_ids_digest","request_started_at_ms","issued_at_ms","deadline_ms","retain_for_ms"] {
        if claims[key]!=request_claims[key] {
            tracing::warn!(target:"dynamo_llm::native_cache",stage="event_claims",native_request_id=%d.origin.native_request_id,field=key,"native cache receipt rejected");
            return Err(Error::Evidence("signed response differs from actual generation binding"));
        }
    }
    let event_digest=cache_auth::digest(&payload).map_err(Error::Store)?;
    for key in ["operation_id","nonce"] {
        let expected=cache_auth::digest(&json!(["native-response",request_claims[key],event_digest])).map_err(Error::Store)?;
        if claims[key]!=expected {
            tracing::warn!(target:"dynamo_llm::native_cache",stage="event_replay",native_request_id=%d.origin.native_request_id,field=key,"native cache receipt rejected");
            return Err(Error::Evidence("native response event replay binding mismatch"));
        }
    }
    let id=packet.get("native_request_id").and_then(Value::as_str).ok_or(Error::Evidence("native packet lacks actual Req ID"))?;
    if id!=d.origin.native_request_id {return Err(Error::Evidence("native event request ID mismatch"));}
    let receipt=packet.get("receipt").ok_or(Error::Evidence("native packet lacks private receipt"))?;
    let endpoint=receipt.get("endpoint_id").and_then(Value::as_str).ok_or(Error::Evidence("native packet lacks endpoint ID"))?;
    let p=d.points.get(endpoint).or_else(||d.selected_read.as_ref().filter(|s|s.point.endpoint_id==endpoint).map(|s|&s.point)).ok_or(Error::Evidence("native packet endpoint not dispatched"))?;
    if receipt.get("native_page_tokens").and_then(Value::as_u64)!=Some(p.native_page_tokens)
        || receipt.get("native_end_byte").and_then(Value::as_u64)!=Some(p.native_end_byte)
        || receipt.get("native_page_prefix_digest").and_then(Value::as_str)!=Some(hex(&p.prefix.digest).as_str())
        || receipt.get("native_mark_prefix_digest").and_then(Value::as_str)!=Some(hex(&p.prefix.marked_endpoint_digest).as_str()) {
        return Err(Error::Evidence("signed native physical/marked full-render binding mismatch"));
    }
    if receipt.get("native_mark_render_prefix_digest").and_then(Value::as_str)!=Some(hex(&p.prefix.marked_render_digest).as_str()) {return Err(Error::Evidence("native raw marked renderer prefix mismatch"));}
    let outer=packet.get("event_kind").and_then(Value::as_str).ok_or(Error::Evidence("native packet lacks outer event kind"))?;
    if receipt.get("event_kind").and_then(Value::as_str).is_some_and(|v|v!=outer) {return Err(Error::Evidence("native event kind echo mismatch"));}
    Ok(p)
}
async fn verify_native_packet(store:&EtcdStore,d:&Dispatch,actual_response_worker_url:&str,packet:&Value,secret:&[u8])->Result<VerifiedReceipt> {
    authenticated_event_point(d,actual_response_worker_url,packet,secret)?;
    let outer=packet["event_kind"].as_str().unwrap();
    let kind=match outer {
        "write_commit"=>EventKind::WriteCommitted,"read_refresh"=>EventKind::ReadHit,
        _=>return Err(Error::Evidence("native packet lacks actual cache event kind")),
    };
    store.confirm_dispatch(d).await?;
    Ok(parse_receipt(d,actual_response_worker_url,packet["native_request_id"].as_str().unwrap(),&packet["receipt"],kind,Instant::now())?.0)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TerminalRank {rank_id:u32,generation_terminal:bool,actual_read_tokens:u64,fresh_write_ranges:Vec<FreshWriteRange>}

fn terminal_zero_coverage(d:&Dispatch,p:&RenderPoint,packet:&Value)->Result<super::cache_accounting::VerifiedEventCoverage> {
    let r=&packet["receipt"];
    let mut allowed=BTreeSet::from(["reservation_id","authority_revision","native_origin_uuid","runtime_profile_digest","namespace_digest","request_started_at_ms","endpoint_id","native_mark_tokens","native_page_tokens","native_end_byte","native_page_prefix_digest","native_mark_prefix_digest","native_mark_render_prefix_digest","rank_receipts"]);
    if r.get("event_kind").is_some() {
        if r["event_kind"]!="terminal_no_write" {return Err(Error::Evidence("terminal no-write kind mismatch"));}
        allowed.insert("event_kind");
    }
    if r.as_object().ok_or(Error::Evidence("terminal no-write receipt is not an object"))?.keys().map(String::as_str).collect::<BTreeSet<_>>()!=allowed {
        return Err(Error::Evidence("terminal no-write exact receipt fields required"));
    }
    for (key,wanted) in [("reservation_id",json!(d.reservation_id)),("authority_revision",json!(d.authority_revision)),("native_origin_uuid",json!(d.worker.native_origin_uuid)),("runtime_profile_digest",json!(d.worker.runtime_profile_digest)),("namespace_digest",json!(d.worker.namespace_digest)),("request_started_at_ms",json!(d.origin.started_at_ms)),("native_mark_tokens",json!(p.native_mark_tokens))] {
        if r[key]!=wanted {return Err(Error::Evidence("terminal no-write immutable binding mismatch"));}
    }
    let ranks:Vec<TerminalRank>=serde_json::from_value(r["rank_receipts"].clone()).map_err(|_|Error::Evidence("terminal no-write rank proof malformed"))?;
    let is_write=d.points.contains_key(&p.endpoint_id);
    let cap=d.selected_read.as_ref().map(|r|r.point.native_page_tokens).unwrap_or(0);
    let mut seen=BTreeSet::new();
    for rank in ranks {
        if !seen.insert(rank.rank_id) || !rank.generation_terminal || !rank.fresh_write_ranges.is_empty()
            || rank.actual_read_tokens>cap || (!is_write && rank.actual_read_tokens!=0) {
            return Err(Error::Evidence("terminal no-write lacks complete zero-write outcome"));
        }
    }
    if seen!=d.worker.rank_tier_clocks.keys().copied().collect() || d.retain_for_ms.get(&p.endpoint_id).copied().unwrap_or(0)==0 {
        return Err(Error::Evidence("terminal no-write incomplete dispatched rank scope"));
    }
    Ok(super::cache_accounting::VerifiedEventCoverage{event_id:cache_auth::digest(&cache_auth::payload_for("receipts",packet).map_err(Error::Store)?).map_err(Error::Store)?,endpoint_id:p.endpoint_id.clone(),ttl_ms:p.ttl.ms(),read_tokens:0,fresh_write_ranges:Vec::new(),terminal_no_write:true})
}
async fn publish_verified_origin(store:&EtcdStore,d:&Dispatch,packet:&Value,verified:&VerifiedReceipt,now_upper_ms:u64)->Result<bool> {
    let endpoint=packet["receipt"]["endpoint_id"].as_str().ok_or(Error::Evidence("missing native endpoint"))?;
    let Some(lease)=d.origin_leases.get(endpoint) else {return Ok(false)};
    store.publish_origin_marker(&verified.evidence,lease,&d.reservation_id,d.authority_revision,now_upper_ms).await
}
#[cfg(test)]
async fn observe_native_packet(store:&EtcdStore,d:&Dispatch,url:&str,packet:&Value,secret:&[u8],now_upper_ms:u64)->Result<bool> {
    let verified=verify_native_packet(store,d,url,packet,secret).await?;
    publish_verified_origin(store,d,packet,&verified,now_upper_ms).await
}

/// Accounting is produced only after the same complete transport, HMAC,
/// immutable ACK, all-rank clock and held-page verifier used for publication.
pub(crate) async fn observe_native_packet_accounting(store:&EtcdStore,d:&Dispatch,url:&str,packet:&Value,secret:&[u8],clock:&ClockBounds,shared:Option<&SharedBackendQualification>)->Result<Option<super::cache_accounting::VerifiedEventCoverage>> {
    if packet["event_kind"]=="terminal_no_write" {
        let p=authenticated_event_point(d,url,packet,secret)?;
        let coverage=terminal_zero_coverage(d,p,packet)?;
        store.confirm_dispatch(d).await?;
        return Ok(Some(coverage));
    }
    let verified=verify_native_packet(store,d,url,packet,secret).await?;
    let r:Receipt=serde_json::from_value(packet["receipt"].clone()).map_err(|_|Error::Evidence("malformed accounting receipt"))?;
    let is_read=packet["event_kind"]=="read_refresh";
    let p=if is_read {d.selected_read.as_ref().map(|r|&r.point)}else{d.points.get(&r.endpoint_id)}.ok_or(Error::Evidence("accounting scope absent"))?;
    if let Some(qualification)=shared {
        let mut retained=Vec::new();let mut imported=BTreeMap::new();let mut read_clocks=BTreeMap::new();
        for rank in &r.rank_receipts {
            let tier=rank.tier_receipts.iter().find(|t|t.tier==qualification.tier()).ok_or(Error::Evidence("actual qualified shared tier missing"))?;
            let backend=tier.backend_namespace_digest.clone().ok_or(Error::Evidence("actual shared backend namespace missing"))?;
            if backend!=qualification.backend_namespace_digest() {return Err(Error::Evidence("actual shared backend qualification differs"));}
            if let Some(actual)=qualification.backend_runtime_digest() {
                if tier.backend_runtime_digest.as_deref()!=Some(actual) {return Err(Error::Evidence("actual GF shared backend runtime differs"));}
            }
            retained.push(SharedRankRetain{rank_id:rank.rank_id,backend_namespace_digest:backend,shared_object_digest:tier.shared_object_digest.clone().ok_or(Error::Evidence("actual durable shared object missing"))?,tier:tier.tier.clone(),retained_tokens:tier.retained_tokens,native_clock_domain:tier.native_clock_domain.clone(),receive_monotonic_ns:tier.receive_monotonic_ns,monotonic_event_ns:tier.monotonic_event_ns,retain_until_monotonic_ns:tier.retain_until_monotonic_ns,ack_retain_for_ms:tier.ack_retain_for_ms});
            if is_read && d.selected_read.as_ref().is_some_and(|r|r.shared_import.is_some()) {
                imported.insert(rank.rank_id,tier.imported_shared_object_digest.clone().ok_or(Error::Evidence("actual completed shared import missing"))?);
                read_clocks.insert(rank.rank_id,(tier.source_read_upper_ms.ok_or(Error::Evidence("actual shared read upper clock missing"))?,tier.source_read_clock_error_ms.ok_or(Error::Evidence("actual shared read clock error missing"))?));
            }
        }
        if is_read {if let Some(import)=d.selected_read.as_ref().and_then(|r|r.shared_import.as_ref()) {
            import.verify_source_read_clocks(qualification,d.origin.started_at_ms,&read_clocks)?;
            import.verify_destination_import(qualification,&verified.evidence,&imported,clock)?;
        }}
        let mut retained=VerifiedSharedRetain::from_verified_native(qualification,&verified.evidence,retained)?;
        if qualification.is_promotion_source() {retained.preserve_verified_source_packet(packet);}
        if let Some(lease)=d.origin_leases.get(&r.endpoint_id) {store.publish_shared_marker(qualification,&retained,lease,&d.reservation_id,d.authority_revision,clock).await?;}
    }else if is_read && d.selected_read.as_ref().is_some_and(|r|r.shared_import.is_some()) {return Err(Error::Evidence("selected shared backend qualification absent"));}
    publish_verified_origin(store,d,packet,&verified,clock.upper_ms()).await?;
    let mut intersection:Option<Vec<(u64,u64)>>=None;
    for rank in &r.rank_receipts {
        let Some(raw)=&rank.fresh_write_ranges else {return Ok(None)};
        if raw.len()>64 || (is_read && !raw.is_empty()) {return Err(Error::Evidence("fresh write evidence scope invalid"));}
        let mut ranges=Vec::with_capacity(raw.len());
        for range in raw {
            if range.start_tokens>=range.end_tokens || range.end_tokens>p.native_page_tokens || range.end_tokens>rank.published_tokens || range.start_tokens<rank.actual_read_tokens {
                return Err(Error::Evidence("fresh write outside actual held/completed page"));
            }
            ranges.push((range.start_tokens,range.end_tokens));
        }
        ranges.sort_unstable();
        let mut merged:Vec<(u64,u64)>=Vec::new();
        for (start,end) in ranges {if let Some(last)=merged.last_mut().filter(|last|start<=last.1) {last.1=last.1.max(end);}else{merged.push((start,end));}}
        intersection=Some(match intersection {
            None=>merged,
            Some(previous)=>{let mut next=Vec::new();for (a,b) in previous {for &(c,e) in &merged {let start=a.max(c);let end=b.min(e);if start<end {next.push((start,end));}}}next},
        });
    }
    Ok(Some(super::cache_accounting::VerifiedEventCoverage{event_id:cache_auth::digest(&cache_auth::payload_for("receipts",packet).map_err(Error::Store)?).map_err(Error::Store)?,endpoint_id:r.endpoint_id,ttl_ms:p.ttl.ms(),read_tokens:if is_read {r.actual_read_tokens}else{0},fresh_write_ranges:intersection.unwrap_or_default(),terminal_no_write:false}))
}

pub(crate) fn accounting_scopes(d:&Dispatch)->Vec<super::cache_accounting::CoverageScope> {
    let mut scopes=Vec::new();
    for (id,p) in &d.points {if d.retain_for_ms.get(id).copied().unwrap_or(0)>0 {scopes.push(super::cache_accounting::CoverageScope{endpoint_id:id.clone(),ttl_ms:p.ttl.ms(),retained_tokens:p.native_page_tokens,write:true,read:false});}}
    if let Some(read)=&d.selected_read {if d.retain_for_ms.get(&read.point.endpoint_id).copied().unwrap_or(0)>0 {scopes.push(super::cache_accounting::CoverageScope{endpoint_id:read.point.endpoint_id.clone(),ttl_ms:read.point.ttl.ms(),retained_tokens:read.point.native_page_tokens,write:false,read:true});}}
    scopes
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    #[ignore="requires owned disposable loopback etcd fixture"]
    async fn real_etcd_readonly_scope_keeps_independent_read_cap() {
        let store=EtcdStore::connect(vec![std::env::var("CACHE_AUTHORITY_TEST_ETCD").unwrap()]).await.unwrap();
        let d=dispatch("sgl");
        let mut point=d.points["endpoint"].clone();point.endpoint_id="read:readonly".into();
        let reservation="owned-readonly-marker";
        let row=bounded(store.client.clone().put(format!("{ROOT}reservations/{reservation}"),"CPU-marker",None)).await.unwrap();
        let lease=store.grant_origin_lease(300_000).await.unwrap().unwrap();
        let e=NativeEvidence{request_id:d.origin.request_id.clone(),native_request_id:d.origin.native_request_id.clone(),prefix:point.prefix.clone(),worker:d.worker.worker_url.clone(),incarnation:d.worker.native_origin_uuid.clone(),receipt_id:"CPU-readonly-marker-evidence".into(),kind:EventKind::WriteCommitted,observed_at_ms:1_000_001,request_started_at_ms:1_000_000,pinned_until_ms:1_300_000,committed_tokens:8};
        store.publish_origin_marker(&e,&lease,reservation,row.header().unwrap().revision(),1_000_001).await.unwrap();
        let selected=lookup_best_read(&store,&d.origin,&d.worker,&[point],1_000_002).await.unwrap();assert!(selected.is_some());
        let mut readonly=prepare_dispatch(&store,d.origin.clone(),d.worker.clone(),vec![],selected,None).await.unwrap();
        let mut body=json!({"rid":d.origin.native_request_id});readonly.attach_private_fields(&mut body).unwrap();
        assert_eq!(body["prompt_cache_commit_points"],json!([]));assert_eq!(body["prompt_cache_max_tokens"],0);
        assert_eq!(body["prompt_cache_read_max_tokens"],8);assert_eq!(body["prompt_cache_read_commit_point"]["endpoint_id"],"read:readonly");
        let mut miss=prepare_dispatch(&store,d.origin,d.worker,vec![],None,None).await.unwrap();
        let mut body=json!({"rid":miss.origin.native_request_id});miss.attach_private_fields(&mut body).unwrap();
        assert_eq!(body["prompt_cache_read_max_tokens"],0);assert_eq!(body["prompt_cache_max_tokens"],0);
    }
    #[tokio::test]
    #[ignore="requires owned disposable loopback etcd fixture"]
    async fn real_etcd_expired_cache_keeps_dispatch_cleanup_metadata() {
        let store=EtcdStore::connect(vec![std::env::var("CACHE_AUTHORITY_TEST_ETCD").unwrap()]).await.unwrap();
        let mut fixture=dispatch("sgl");
        fixture.origin.started=Instant::now()-Duration::from_secs(528);
        let d=prepare_dispatch(&store,fixture.origin,fixture.worker,fixture.points.into_values().collect(),None,None).await.unwrap();
        assert!(d.origin_leases.is_empty());
        assert!(d.retain_for_ms.values().all(|n|*n==0));
        let record=bounded(store.client.clone().get(format!("{ROOT}reservations/{}",d.reservation_id),None)).await.unwrap();
        let lease=record.kvs()[0].lease();
        let ttl=bounded(store.client.clone().lease_time_to_live(lease,None)).await.unwrap().ttl();
        assert!((10302..=10304).contains(&ttl));
        store.confirm_dispatch(&d).await.unwrap();
        bounded(store.client.clone().lease_revoke(lease)).await.unwrap();
    }
    #[tokio::test]
    #[ignore="requires owned disposable loopback etcd fixture"]
    async fn real_etcd_generation_once_across_reservation_revisions() {
        let store=EtcdStore::connect(vec![std::env::var("CACHE_AUTHORITY_TEST_ETCD").unwrap()]).await.unwrap();
        for already_dispatched in [false,true] {
            let mut fixture=dispatch("sgl");
            fixture.origin.native_request_id=format!("owned-once-{already_dispatched}");
            let mut d=prepare_dispatch(&store,fixture.origin.clone(),fixture.worker.clone(),fixture.points.values().cloned().collect(),None,None).await.unwrap();
            d.signed_claims=Some(json!({"model":"owned-CPU-model","operation_id":"first-op","nonce":"first-nonce"}));
            if already_dispatched {
                store.claim_generation(&d).await.unwrap();
                assert!(store.clone().claim_generation(&d).await.is_err());
            }
            // This is the etcd portion of an already verified native cancel.
            // The separate actual cancel packet tests verify native admission.
            let key=format!("{ROOT}reservations/{}",d.reservation_id);
            let deleted=bounded(store.client.clone().txn(Txn::new().when(vec![Compare::mod_revision(key.as_str(),CompareOp::Equal,d.authority_revision)])
                .and_then(vec![TxnOp::delete(key,None)]))).await.unwrap();assert!(deleted.succeeded());
            let mut next=prepare_dispatch(&store,fixture.origin,fixture.worker,fixture.points.into_values().collect(),None,None).await.unwrap();
            assert!(next.authority_revision>d.authority_revision);
            next.signed_claims=Some(json!({"model":"owned-CPU-model","operation_id":"fresh-op","nonce":"fresh-nonce"}));
            let fresh=store.clone().claim_generation(&next).await;
            assert_eq!(fresh.is_err(),already_dispatched);
        }
    }
    fn fixture(name:&str)->Value {let mut p:Value=serde_json::from_str(match name {
        "sgl"=>include_str!("fixtures/sgl.json"),"gf"=>include_str!("fixtures/gf.json"),
        "qwen"=>include_str!("fixtures/qwen.json"),"ds"=>include_str!("fixtures/ds.json"),_=>panic!("unknown fixture")
    }).unwrap();
        // Historical component fixtures are test inputs, not the public wire.
        // Obsolete floating monotonic diagnostics have no part in the current
        // integer-nanosecond receipt. Do not normalize them in production.
        p.as_object_mut().unwrap().remove("write_committed_monotonic");
        p.as_object_mut().unwrap().remove("pin_expires_monotonic");
        if name!="ds" {
            let mut tiers=p["tier_receipts"].clone();
            if p["actual_read_tokens"].as_u64().unwrap_or(0)>0 {for tier in tiers.as_array_mut().unwrap() {tier["source_read_upper_ms"]=json!(p["request_started_at_ms"].as_u64().unwrap()+3000);tier["source_read_clock_error_ms"]=json!(1);}}
            p["rank_receipts"]=json!([{"rank_id":0,"pin_receipt_id":p["pin_receipt_id"],"published_tokens":p["published_tokens"],"actual_read_tokens":p["actual_read_tokens"],"tier_receipts":tiers}]);
            // Deliberately synthetic qualified clock for historical component
            // unit inputs, not a native final signed wire/clock qualification.
            // Keep unrelated negative cases exercising their intended binding.
            let at=p["request_started_at_ms"].as_u64().unwrap()+3000;
            for rank in p["rank_receipts"].as_array_mut().unwrap() {
                for tier in rank["tier_receipts"].as_array_mut().unwrap() {
                    tier["event_upper_ms"]=json!(at);tier["event_clock_error_ms"]=json!(1);
                }
            }
        }
        p
    }
    fn prefix(tokens:u64)->Prefix {Prefix{tenant:"owned-cpu-tenant".into(),namespace:"cpu-qualified-only".into(),runtime:[1;32],digest:[2;32],marked_endpoint_digest:[3;32],marked_render_digest:[4;32],marked_tokens:tokens,tokens}}
    fn dispatch(family:&str)->Dispatch {
        // Independent expected configuration from each producer's CPU helper
        // fixture setup. Never derives authority bindings from a receipt under test.
        let (origin,profile,ns,clock,reservation,revision,endpoint,tokens,start,tiers)=match family {
            "sgl"=>("native","profile","nsdigest","native-boot","lease",1,"endpoint",8,1_000_000,vec!["L1","L2"]),
            "gf"=>("native-startup-incarnation","qualified-runtime","7c1664eaed158a43602dfeb189f54d93ba7077377fa8eeb842994db7f1c0e744","3d9c22de2a3c4c6f9daa7b954c65f669","authority-r1",1,"e",6144,1_000_000,vec!["gpu"]),
            "qwen"=>("native-process-uuid","runtime-digest","03ff83967116685d55f977b262e794e470a5c0514f3d55d003d59f77293588ed","boot-uuid:monotonic","r1",11,"e8",8,1_000_000_000,vec!["l1"]),
            _=>panic!("unknown family")
        };
        let o=RequestOrigin{request_id:"server-owned-cpu-context".into(),native_request_id:"native-body-cpu-context".into(),started_at_ms:start,started:Instant::now()-Duration::from_secs(3)};
        let required_tiers:BTreeSet<String>=tiers.into_iter().map(str::to_string).collect();
        let rank_tier_clocks=BTreeMap::from([(0,required_tiers.iter().map(|t|(t.clone(),clock.to_owned())).collect())]);
        let worker=QualifiedWorker{worker_url:format!("http://owned-{family}.invalid:1"),native_origin_uuid:origin.into(),runtime_profile_digest:profile.into(),namespace_digest:ns.into(),native_clock_domain:clock.into(),required_tiers,native_id_field:if family=="gf"{NativeIdField::RequestId}else{NativeIdField::Rid},rank_tier_clocks};
        let mut p=prefix(tokens);
        p.namespace=ns.into();p.runtime=worker.registry_runtime();
        if family=="qwen" {
            let h="29101ebf1ead685029508789eee51c4d4fff65524064d8e32a48efded7036982";
            for (i,b) in p.digest.iter_mut().enumerate() {*b=u8::from_str_radix(&h[2*i..2*i+2],16).unwrap();}
        }
        let point=RenderPoint{native_end_byte:10,endpoint_id:endpoint.into(),native_mark_tokens:tokens,native_page_tokens:tokens,prefix:p,ttl:Ttl::FiveMinutes};
        let selected_read=if family=="qwen" {let mut prefix=prefix(4);prefix.namespace=ns.into();prefix.runtime=worker.registry_runtime();let point=RenderPoint{native_end_byte:10,endpoint_id:"old-e4".into(),native_mark_tokens:4,native_page_tokens:4,prefix,ttl:Ttl::FiveMinutes};Some(SelectedRead{permit:ReadPermit{prefix:point.prefix.clone(),authorized_until_ms:start+300_000,request_started_at_ms:start,request_id:o.request_id.clone()},point,shared_import:None})}else{None};
        Dispatch{origin:o,worker,reservation_id:reservation.into(),authority_revision:revision,
            points:BTreeMap::from([(endpoint.into(),point)]),retain_for_ms:BTreeMap::from([(endpoint.into(),300_000)]),origin_leases:BTreeMap::from([(endpoint.into(),super::super::lease_authority::OriginLease{id:1})]),signed_claims:None,reserve_claims:None,reserve_ack:None,control_claims:None,selected_read,generation_epoch:None}
    }
    #[tokio::test(start_paused=true)]
    async fn cache_generation_post_seals_after_preparation_delay_without_renewing_origin() {
        // CPU-only signer/byte-path cause, not a native admission or etcd claim.
        let mut d=dispatch("sgl");
        d.worker.runtime_profile_digest="ab".repeat(32);
        d.worker.namespace_digest="cd".repeat(32);
        let key=[0x5a;32];
        let c=SigningContext{model:"CPU-post-model".into(),source_digest:"ef".repeat(32),full_render_digest:"01".repeat(32),token_ids_digest:"02".repeat(32),operation_id:"CPU-same-operation".into(),nonce:"CPU-same-nonce".into(),now_lower_ms:1_000_000,now_upper_ms:1_000_002,clock_observed_at:Instant::now()};
        let mut body=json!({"model":c.model,"rid":d.origin.native_request_id,"messages":[{"role":"user","content":"x".repeat(8*1024*1024)}],"cache_salt":"CPU-post"});
        cache_auth::attach_canonical_body(&mut body).unwrap();
        let payload=cache_auth::PrehashedPayload::new(&cache_auth::payload_for("generate",&body).unwrap()).unwrap();
        body[cache_auth::PROOF_FIELD]=generation_proof(&d,&key,&c,&payload).unwrap();
        let original=body[cache_auth::PROOF_FIELD]["claims"].clone();
        let post=GenerationPost::prepare(&body).unwrap();
        let allocation=post.prefix.as_ptr();
        // Represents durable/local preparation delay before the last sample.
        tokio::time::advance(Duration::from_secs(2)).await;
        assert!(generation_proof(&d,&key,&c,&post.payload).is_err(),"old before-preparation clock must reject");
        let clock=ClockBounds::cpu_fixture(Instant::now());
        let fresh=c.refreshed(&clock).unwrap();
        let proof=generation_proof(&d,&key,&fresh,&post.payload).unwrap();
        for field in ["operation_id","nonce","native_request_id","reservation_id","authority_revision","request_started_at_ms"] {assert_eq!(proof["claims"][field],original[field],"{field}");}
        assert_eq!(proof["claims"]["deadline_ms"].as_u64().unwrap()-proof["claims"]["issued_at_ms"].as_u64().unwrap(),998);
        let (bytes,_snapshot)=post.finish(&proof).unwrap();
        assert_eq!(bytes.as_ptr(),allocation,"late envelope append must not copy large body");
        let sealed:Value=serde_json::from_slice(&bytes).unwrap();
        assert_eq!(sealed["messages"],body["messages"]);
        let payload=cache_auth::payload_for("generate",&sealed).unwrap();
        assert_eq!(cache_auth::verify_signature(&key,&sealed[cache_auth::PROOF_FIELD],&payload).unwrap(),proof["claims"]);
        let mut tampered=payload.clone();tampered["cache_salt"]=json!("different");
        assert!(cache_auth::verify_signature(&key,&proof,&tampered).is_err());
        tokio::time::advance(Duration::from_millis(101)).await;
        assert!(generation_proof(&d,&key,&fresh,&cache_auth::PrehashedPayload::new(&payload).unwrap()).is_err());
        tokio::time::advance(Duration::from_secs(300)).await;
        let fresh=c.refreshed(&ClockBounds::cpu_fixture(Instant::now())).unwrap();
        assert!(matches!(generation_proof(&d,&key,&fresh,&cache_auth::PrehashedPayload::new(&payload).unwrap()),Err(Error::Evidence("write endpoint expired after native reserve"))));
    }
    #[test]
    fn multiple_write_points_follow_measured_selector_order_not_uuid_order() {
        use crate::native_cache_instruction_roles::Family;
        let mut ids=["f".repeat(32),format!("{:032x}",1)].into_iter();
        let raw=json!({"model":"CPU-order-only","messages":[{"role":"user","content":[
            {"type":"text","text":"first marked prefix","cache_control":{"type":"ephemeral","ttl":"5m"}},
            {"type":"text","text":"second marked prefix","cache_control":{"type":"ephemeral","ttl":"5m"}}]}],"max_tokens":2});
        let plan=crate::native_cache_plan::chat_cache_plan(&raw,Family::GlmFlash,||ids.next().unwrap()).unwrap();
        let expected=plan.writes.iter().map(|p|p["endpoint_id"].as_str().unwrap().to_owned()).collect::<Vec<_>>();
        assert_eq!(expected.len(),2);assert!(expected[0]>expected[1]);
        let mut d=dispatch("gf");let template=d.points.values().next().unwrap().clone();
        d.points.clear();d.retain_for_ms.clear();d.origin_leases.clear();
        for (index,id) in expected.iter().enumerate() {
            let mut point=template.clone();point.endpoint_id=id.clone();point.native_end_byte+=index as u64;
            d.points.insert(id.clone(),point);d.origin_leases.insert(id.clone(),super::super::lease_authority::OriginLease{id:1});
        }
        let mut rendered=crate::native_cache_plan::native_render_body(&plan);
        rendered["request_id"]=json!(d.origin.native_request_id);rendered["cache_salt"]=json!("CPU-order-only");
        let mut reserve=rendered.clone();d.attach_private_fields(&mut reserve).unwrap();
        let reserved=reserve["prompt_cache_commit_points"].clone();
        assert_eq!(reserved.as_array().unwrap().iter().map(|p|p["endpoint_id"].as_str().unwrap()).collect::<Vec<_>>(),
            expected.iter().map(String::as_str).collect::<Vec<_>>());
        // Exercise the already-ACKed projection branch with the existing typed
        // CPU ACK snapshot. This unit does not execute Reserve CAS or an allocator.
        let ack:Value=serde_json::from_str(include_str!("../vllm/bridge-wire-cold-reservation-ack-fixture.json")).unwrap();
        d.reserve_ack=Some(serde_json::from_value(ack["reservation_ack"].clone()).unwrap());
        d.origin.started-=Duration::from_millis(20);
        let mut generation=rendered.clone();d.attach_private_fields(&mut generation).unwrap();
        assert_eq!(generation["prompt_cache_commit_points"],reserved);
        assert!(reserved.as_array().unwrap().iter().all(|p|p["expires_at_ms"]==json!(d.origin.started_at_ms+300_000)));
        let key:Vec<u8>=(0..32).collect();let mut claims=ack[cache_auth::PROOF_FIELD]["claims"].clone();
        claims["operation"]=json!("generate");
        let proof=cache_auth::sign(&key,&claims,&generation).unwrap();
        cache_auth::verify_signature(&key,&proof,&generation).unwrap();
        let mut reordered=generation.clone();reordered["prompt_cache_commit_points"].as_array_mut().unwrap().reverse();
        assert!(cache_auth::verify_signature(&key,&proof,&reordered).is_err());
        for bad in ["missing","duplicate","foreign"] {
            let mut candidate=rendered.clone();
            match bad {
                "missing"=>{candidate["prompt_cache_blocks"].as_array_mut().unwrap().pop();},
                "duplicate"=>candidate["prompt_cache_blocks"][1]=candidate["prompt_cache_blocks"][0].clone(),
                "foreign"=>candidate["prompt_cache_blocks"][0]["endpoint_id"]=json!("write:foreign"),
                _=>unreachable!(),
            }
            assert!(d.attach_private_fields(&mut candidate).is_err(),"{bad}");
        }
    }
    #[test]
    fn terminal_no_write_requires_complete_terminal_ranks_and_preserves_verified_read() {
        let d=dispatch("sgl");let p=&d.points["endpoint"];
        let packet=json!({"event_kind":"terminal_no_write","receipt":{"reservation_id":d.reservation_id,"authority_revision":d.authority_revision,"native_origin_uuid":d.worker.native_origin_uuid,"runtime_profile_digest":d.worker.runtime_profile_digest,"namespace_digest":d.worker.namespace_digest,"request_started_at_ms":d.origin.started_at_ms,"endpoint_id":p.endpoint_id,"native_mark_tokens":p.native_mark_tokens,"native_page_tokens":p.native_page_tokens,"native_end_byte":p.native_end_byte,"native_page_prefix_digest":hex(&p.prefix.digest),"native_mark_prefix_digest":hex(&p.prefix.marked_endpoint_digest),"native_mark_render_prefix_digest":hex(&p.prefix.marked_render_digest),"rank_receipts":[{"rank_id":0,"generation_terminal":true,"actual_read_tokens":0,"fresh_write_ranges":[]}]}});
        let coverage=terminal_zero_coverage(&d,p,&packet).unwrap();
        use super::super::cache_accounting::{CacheAccounting,CoverageScope,VerifiedEventCoverage};
        let mut a=CacheAccounting::new(12,false,vec![CoverageScope{endpoint_id:p.endpoint_id.clone(),ttl_ms:p.ttl.ms(),retained_tokens:8,write:true,read:false},CoverageScope{endpoint_id:"read:actual".into(),ttl_ms:300_000,retained_tokens:4,write:false,read:true}]).unwrap();
        a.observe(coverage.clone()).unwrap();assert!(a.finish().is_none());
        a.observe(VerifiedEventCoverage{event_id:"actual-verified-read".into(),endpoint_id:"read:actual".into(),ttl_ms:300_000,read_tokens:4,fresh_write_ranges:vec![],terminal_no_write:false}).unwrap();
        let known=a.finish().unwrap();assert_eq!((known.input_tokens,known.cache_read_input_tokens,known.cache_creation_5m_input_tokens,known.cache_creation_1h_input_tokens),(8,4,0,0));
        for (key,value) in [("generation_terminal",json!(false)),("fresh_write_ranges",json!([{"start_tokens":0,"end_tokens":8}])),("actual_read_tokens",json!(1))] {
            let mut bad=packet.clone();bad["receipt"]["rank_receipts"][0][key]=value;assert!(terminal_zero_coverage(&d,p,&bad).is_err());
        }
        let mut bad=packet.clone();bad["receipt"]["rank_receipts"]=json!([]);assert!(terminal_zero_coverage(&d,p,&bad).is_err());
        let mut bad=packet.clone();bad["receipt"]["rank_receipts"][0].as_object_mut().unwrap().remove("fresh_write_ranges");assert!(terminal_zero_coverage(&d,p,&bad).is_err());
        let mut bad=packet.clone();bad["receipt"]["pin_receipt_id"]=json!("fake-pin");assert!(terminal_zero_coverage(&d,p,&bad).is_err());
        let mut contradictory=coverage;contradictory.event_id="actual-fresh-conflict".into();contradictory.terminal_no_write=false;contradictory.fresh_write_ranges=vec![(0,8)];assert!(a.observe(contradictory).is_err());
    }
    #[test] fn historical_component_bindings_with_synthetic_event_clock_parse() {
        for name in ["sgl","gf","qwen"] {
            let d=dispatch(name);let result=parse_receipt(&d,&d.worker.worker_url,&d.origin.native_request_id,&fixture(name),EventKind::WriteCommitted,Instant::now());
            assert!(result.is_ok(),"{name}: {:?}",result.err());
        }
    }
    #[test]
    fn late_query_uses_actual_qualified_event_time_and_rejects_expired_commit() {
        let mut d=dispatch("sgl");d.origin.started=Instant::now()-Duration::from_secs(301);
        let mut raw=fixture("sgl");
        for rank in raw["rank_receipts"].as_array_mut().unwrap() {for tier in rank["tier_receipts"].as_array_mut().unwrap() {
            tier["event_upper_ms"]=json!(d.origin.started_at_ms+10);tier["event_clock_error_ms"]=json!(1);
        }}
        let proof=parse_receipt(&d,&d.worker.worker_url,&d.origin.native_request_id,&raw,EventKind::WriteCommitted,Instant::now()).unwrap().0;
        assert_eq!(proof.evidence.observed_at_ms,d.origin.started_at_ms+10);
        assert!(proof.evidence.pinned_until_ms<d.origin.authority_now_ms(Instant::now()).unwrap());
        use super::super::cache_accounting::{CacheAccounting,CoverageScope,VerifiedEventCoverage};
        let mut a=CacheAccounting::new(10,false,vec![CoverageScope{endpoint_id:"endpoint".into(),ttl_ms:300_000,retained_tokens:8,write:true,read:false}]).unwrap();
        a.observe(VerifiedEventCoverage{event_id:"CPU-verified-early-event".into(),endpoint_id:"endpoint".into(),ttl_ms:300_000,read_tokens:0,fresh_write_ranges:vec![(0,8)],terminal_no_write:false}).unwrap();
        assert_eq!(a.finish().unwrap().cache_creation_5m_input_tokens,8);
        for (key,value) in [("event_upper_ms",json!(d.origin.started_at_ms+300_000)),("event_upper_ms",json!(d.origin.started_at_ms-1)),("event_clock_error_ms",json!(101))] {
            let mut bad=raw.clone();bad["rank_receipts"][0]["tier_receipts"][0][key]=value;
            assert!(parse_receipt(&d,&d.worker.worker_url,&d.origin.native_request_id,&bad,EventKind::WriteCommitted,Instant::now()).is_err());
        }
        for field in ["event_upper_ms","event_clock_error_ms"] {
            let mut bad=raw.clone();bad["rank_receipts"][0]["tier_receipts"][0].as_object_mut().unwrap().remove(field);
            assert!(parse_receipt(&d,&d.worker.worker_url,&d.origin.native_request_id,&bad,EventKind::WriteCommitted,Instant::now()).is_err());
        }
    }
    #[test] fn origin_profile_tenant_reservation_revision_endpoint_and_transport_bind() {
        let d=dispatch("sgl");let packet=fixture("sgl");
        for (key,value) in [("native_origin_uuid",json!("wrong")),("runtime_profile_digest",json!("wrong")),("namespace_digest",json!("wrong")),("reservation_id",json!("wrong")),("authority_revision",json!(2)),("endpoint_id",json!("unmarked")),("request_started_at_ms",json!(0))] {
            let mut p=packet.clone();p[key]=value;
            assert!(parse_receipt(&d,&d.worker.worker_url,&d.origin.native_request_id,&p,EventKind::WriteCommitted,Instant::now()).is_err(),"{key}");
        }
        assert!(parse_receipt(&d,"http://another-worker.invalid:1",&d.origin.native_request_id,&packet,EventKind::WriteCommitted,Instant::now()).is_err());
        assert!(parse_receipt(&d,&d.worker.worker_url,&d.origin.request_id,&packet,EventKind::WriteCommitted,Instant::now()).is_err(),"HTTP phase UUID must not substitute native body ID");
    }
    #[test] fn actual_pin_missing_short_ack_bad_domain_bad_event_and_duplicate_tier_reject() {
        let d=dispatch("sgl");let packet=fixture("sgl");
        for (key,value) in [("ack_retain_for_ms",json!(299999)),("native_clock_domain",json!("foreign-boot")),("monotonic_event_ns",json!(310_000_000_000u64)),("retain_until_monotonic_ns",json!(310_000_000_001u64)),("retained_tokens",json!(7)),("ack_retain_for_ms",json!(true)),("receive_monotonic_ns",json!(-1))] {
            let mut p=packet.clone();p["rank_receipts"][0]["tier_receipts"][0][key]=value;
            assert!(parse_receipt(&d,&d.worker.worker_url,&d.origin.native_request_id,&p,EventKind::WriteCommitted,Instant::now()).is_err(),"{key}");
        }
        let mut missing=packet.clone();missing["rank_receipts"][0]["tier_receipts"].as_array_mut().unwrap().pop();
        assert!(parse_receipt(&d,&d.worker.worker_url,&d.origin.native_request_id,&missing,EventKind::WriteCommitted,Instant::now()).is_err());
        let mut duplicate=packet.clone();let first=duplicate["rank_receipts"][0]["tier_receipts"][0].clone();duplicate["rank_receipts"][0]["tier_receipts"].as_array_mut().unwrap().push(first);
        assert!(parse_receipt(&d,&d.worker.worker_url,&d.origin.native_request_id,&duplicate,EventKind::WriteCommitted,Instant::now()).is_err());
        assert!(parse_receipt(&d,&d.worker.worker_url,&d.origin.native_request_id,&fixture("ds")["receipt"],EventKind::WriteCommitted,Instant::now()).is_err(),"L3 fragment must not become full endpoint receipt");
    }
    #[test] fn logical_origin_expiry_and_same_native_clock_duration() {
        let mut d=dispatch("sgl");let mut packet=fixture("sgl");
        // Arbitrary native wall epochs have no role in authority TTL or holds.
        packet["write_committed_at_ms"]=json!(9_999_999_999u64);packet["tier_receipts"][0]["committed_at_ms"]=json!(0);
        let (verified,_)=parse_receipt(&d,&d.worker.worker_url,&d.origin.native_request_id,&packet,EventKind::WriteCommitted,Instant::now()).unwrap();
        assert_eq!(verified.evidence.pinned_until_ms,1_300_000);
        d.origin.started=Instant::now()-Duration::from_secs(528);
        assert!(parse_receipt(&d,&d.worker.worker_url,&d.origin.native_request_id,&packet,EventKind::WriteCommitted,Instant::now()).is_err());
        assert_eq!(d.origin.elapsed_ms(d.origin.started+Duration::from_nanos(1)).unwrap(),1);
    }
    #[test] fn actual_source_read_clock_must_precede_old_logical_expiry() {
        let mut d=dispatch("sgl");let point=d.points["endpoint"].clone();
        d.selected_read=Some(SelectedRead{point:point.clone(),permit:ReadPermit{prefix:point.prefix.clone(),authorized_until_ms:1_300_000,request_started_at_ms:d.origin.started_at_ms,request_id:d.origin.request_id.clone()},shared_import:None});
        let mut packet=fixture("sgl");packet["actual_read_tokens"]=8.into();packet["rank_receipts"][0]["actual_read_tokens"]=8.into();
        for tier in packet["rank_receipts"][0]["tier_receipts"].as_array_mut().unwrap() {tier["source_read_upper_ms"]=1_299_999.into();tier["source_read_clock_error_ms"]=1.into();}
        let parse=|v:&Value|parse_receipt(&d,&d.worker.worker_url,&d.origin.native_request_id,v,EventKind::ReadHit,Instant::now());
        assert!(parse(&packet).is_ok());
        for (key,value) in [("source_read_upper_ms",json!(1_300_000)),("source_read_clock_error_ms",json!(101)),("source_read_upper_ms",json!(999999))] {
            let mut invalid=packet.clone();invalid["rank_receipts"][0]["tier_receipts"][0][key]=value;assert!(parse(&invalid).is_err(),"{key}");
        }
        let mut missing=packet;missing["rank_receipts"][0]["tier_receipts"][0].as_object_mut().unwrap().remove("source_read_upper_ms");assert!(parse(&missing).is_err());
    }
    #[test] fn same_physical_page_different_logical_mark_is_not_same_entry() {
        let p=prefix(2048);let mut q=p.clone();q.marked_tokens=2200;q.marked_endpoint_digest=[4;32];
        assert_ne!(p.key().unwrap(),q.key().unwrap());
        let mut inside_token=p.clone();inside_token.marked_render_digest=[5;32];assert_ne!(p.key().unwrap(),inside_token.key().unwrap());
        let mut unqualified=p.clone();unqualified.marked_endpoint_digest=[0;32];assert!(unqualified.key().is_err());
    }
    #[test] fn all_rank_native_holds_require_exact_set_and_count_minimum() {
        let mut d=dispatch("sgl");let mut packet=fixture("sgl");
        let clocks=BTreeMap::from([("L1".into(),"native-boot".into()),("L2".into(),"native-boot".into())]);
        d.worker.rank_tier_clocks=BTreeMap::from([(0,clocks.clone()),(1,clocks)]);
        let rank=|id|json!({"rank_id":id,"pin_receipt_id":packet["pin_receipt_id"],"published_tokens":8,"actual_read_tokens":0,"tier_receipts":packet["tier_receipts"]});
        packet["rank_receipts"]=json!([rank(0),rank(1)]);
        let parse=|p:&Value|parse_receipt(&d,&d.worker.worker_url,&d.origin.native_request_id,p,EventKind::WriteCommitted,Instant::now());
        assert!(parse(&packet).is_ok());
        let mut missing=packet.clone();missing["rank_receipts"].as_array_mut().unwrap().pop();assert!(parse(&missing).is_err());
        let mut duplicate=packet.clone();duplicate["rank_receipts"][1]["rank_id"]=0.into();assert!(parse(&duplicate).is_err());
        let mut count=packet.clone();count["rank_receipts"][1]["published_tokens"]=4.into();assert!(parse(&count).is_err());
        let mut foreign=packet.clone();foreign["rank_receipts"][1]["tier_receipts"][0]["native_clock_domain"]="foreign-rank-boot".into();assert!(parse(&foreign).is_err());
        let mut no_rank=packet.clone();no_rank.as_object_mut().unwrap().remove("rank_receipts");assert!(parse(&no_rank).is_err());
    }
    #[test] fn private_dispatch_body_recomputes_elapsed_and_rejects_user_fields() {
        let mut d=dispatch("sgl");let mut body=json!({"messages":[],"rid":"native-body-cpu-context","prompt_cache_blocks":[{"endpoint_id":"endpoint"}]});d.attach_private_fields(&mut body).unwrap();
        let p=&body["prompt_cache_commit_points"][0];assert!(p["retain_for_ms"].as_u64().unwrap()<=297000);
        assert_eq!(p["expires_at_ms"],json!(1300000));assert_eq!(body["prompt_cache_read_max_tokens"],json!(0));
        let mut injection=json!({"authority_revision":11});assert!(d.attach_private_fields(&mut injection).is_err());
        let mut expired=dispatch("sgl");expired.origin.started=Instant::now()-Duration::from_secs(528);
        let mut expired_body=json!({"rid":"native-body-cpu-context","prompt_cache_blocks":[{"endpoint_id":"endpoint"}]});expired.attach_private_fields(&mut expired_body).unwrap();
        assert_eq!(expired_body["prompt_cache_read_max_tokens"],0);assert_eq!(expired_body["prompt_cache_commit_points"][0]["retain_for_ms"],0);
        assert_eq!(expired_body["prompt_cache_commit_points"][0]["no_pin"],true);
    }
    #[tokio::test]
    #[ignore="requires owned disposable loopback etcd fixture"]
    async fn real_etcd_signed_zero_scope_and_cancel_revision_gate() {
        let store=EtcdStore::connect(vec![std::env::var("CACHE_AUTHORITY_TEST_ETCD").unwrap()]).await.unwrap();
        let mut example=dispatch("sgl");example.origin.started=Instant::now()-Duration::from_secs(528);
        example.worker.runtime_profile_digest="ab".repeat(32);example.worker.namespace_digest="cd".repeat(32);
        example.worker.rank_tier_clocks=BTreeMap::from([(0,BTreeMap::from([("L1".into(),"native-boot".into()),("L2".into(),"native-boot".into())]))]);
        let mut point=example.points.values().next().unwrap().clone();point.prefix.namespace=example.worker.namespace_digest.clone();point.prefix.runtime=example.worker.registry_runtime();
        let mut d=prepare_dispatch(&store,example.origin,example.worker,vec![point],None,None).await.unwrap();
        let now=d.origin.authority_now_ms(Instant::now()).unwrap();
        let mut c=SigningContext{model:"owned-cpu-model".into(),source_digest:"ef".repeat(32),full_render_digest:"01".repeat(32),token_ids_digest:"02".repeat(32),operation_id:"owned-cpu-zero-gen".into(),nonce:"owned-cpu-zero-gen-nonce".into(),now_lower_ms:now,now_upper_ms:now,clock_observed_at:Instant::now()};
        let mut body=json!({"model":c.model,"rid":d.origin.native_request_id,"messages":[{"role":"user","content":"ordinary generation continues"}],"prompt_cache_blocks":[{"endpoint_id":"endpoint"}]});
        signed_generation_body(&store,&mut d,&mut body,&[0x5a;32],&c,|_|async {Ok(ClockBounds::cpu_fixture(Instant::now()))}).await.unwrap();
        assert_eq!(body["prompt_cache_read_max_tokens"],0);assert_eq!(body["prompt_cache_commit_points"][0]["retain_for_ms"],0);assert_eq!(body["prompt_cache_commit_points"][0]["no_pin"],true);
        assert!(d.origin_leases.is_empty());
        assert!(cache_auth::verify_signature(&[0x5a;32],&body[cache_auth::PROOF_FIELD],&cache_auth::payload_for("generate",&body).unwrap()).is_ok());
        c.operation_id="owned-cpu-cancel".into();c.nonce="owned-cpu-cancel-nonce".into();c.clock_observed_at=Instant::now();
        let deadline=d.origin.operation_deadline().unwrap();
        signed_control_query(&store,&mut d,"cancel",&[0x5a;32],&c,deadline,|_|async {Ok(ClockBounds::cpu_fixture(Instant::now()))}).await.unwrap();
        // Typed native cancellation wire fixture tests parser/CAS, not a claim
        // that the production all-rank scheduler cancel was executed here.
        let sign_reply=|packet:Value| {
            let mut claims=d.control_claims.clone().unwrap();claims["operation"]="receipts".into();
            let digest=cache_auth::digest(&packet).unwrap();
            for k in ["operation_id","nonce"] {claims[k]=cache_auth::digest(&json!(["native-response",claims[k],digest])).unwrap().into();}
            let proof=cache_auth::sign(&[0x5a;32],&claims,&packet).unwrap();let mut packet=packet;packet[cache_auth::PROOF_FIELD]=proof;packet
        };
        let packet=json!({"native_request_id":d.origin.native_request_id,"reservation_id":d.reservation_id,"authority_revision":d.authority_revision,"cancelled":true,"rank_results":[{"rank_id":0,"cancelled":true}]});
        let mut missing=packet.clone();missing["rank_results"]=json!([]);
        assert!(reprepare_after_cancel(&store,&d,&d.worker.worker_url,&sign_reply(missing),&[0x5a;32]).await.is_err());store.confirm_dispatch(&d).await.unwrap();
        let mut partial=packet.clone();partial["rank_results"][0]["cancelled"]=false.into();
        assert!(reprepare_after_cancel(&store,&d,&d.worker.worker_url,&sign_reply(partial),&[0x5a;32]).await.is_err());
        let old_revision=d.authority_revision;
        let new=reprepare_after_cancel(&store,&d,&d.worker.worker_url,&sign_reply(packet),&[0x5a;32]).await.unwrap();assert!(new.authority_revision>old_revision);assert!(new.selected_read.is_none());
        assert!(new.origin_leases.is_empty());
        // A real raw mark before the first full page also remains ordinary gen.
        let mut p=new.points["endpoint"].clone();p.native_page_tokens=0;p.prefix.tokens=0;
        let empty=prepare_dispatch(&store,new.origin.clone(),new.worker.clone(),vec![p],None,None).await.unwrap();assert!(empty.origin_leases.is_empty());
    }

}
