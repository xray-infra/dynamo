//! Cross-origin discovery on the existing etcd control plane. A registry hit
//! authorizes an attempted native import; only a verified completed import and
//! destination hold can contribute cached input usage.
use super::*;
use super::frontend_clock::ClockBounds;
use super::frontend_qualification::{FrontendStartupMaterial, ParsedProfile};
use super::lease_authority::OriginLease;
use etcd_client::GetOptions;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use tokio::time::Instant;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Backend {
    backend_kind: String,
    backend_namespace_digest: String,
    tier: String,
}

/// Actual runtime backend namespace and byte ABI permit discovery. Only native
/// import completion and physical holds can establish a cached read.
#[derive(Clone, Debug)]
pub struct SharedBackendQualification {
    model: String,
    origin: String,
    runtime_profile_digest: String,
    compatibility_digest: String,
    ranks: BTreeSet<u32>,
    backend: Backend,
    // GF's resolved externalStore ABI includes the live filesystem identity.
    // Other native families provide their existing three-field backend ABI.
    backend_runtime_digest: Option<String>,
    promotion: Option<super::replicated_ds_promotion::QualifiedPromotion>,
}
impl SharedBackendQualification {
    pub(crate) fn from_startup(material:&FrontendStartupMaterial)->Result<Option<Self>> {
        let mut result=Self::from_profile(&material.deployment.model,&material.deployment.native_origin,&material.profile)?;
        if let Some(q)=result.as_mut() {
            if material.profile.profile.layout.get("replicated_promotion").is_some() {
                q.attach_replicated_ds_promotion(&material.profile,&material.profile.profile.native_origin_uuid)?;
            }
        }
        Ok(result)
    }
    fn from_profile(model:&str,origin:&str,profile:&ParsedProfile)->Result<Option<Self>> {
        let Some(raw)=profile.profile.layout.get("shared_cache") else{return Ok(None)};
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct RuntimeBackend {backend_kind:String,backend_namespace_digest:String,tier:String}
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct GfRuntimeBackend {backend_kind:String,backend_namespace_digest:String,backend_runtime_digest:String,tier:String}
        let (actual,backend_runtime_digest)=if profile.profile.family=="glm_flash" {
            let actual:GfRuntimeBackend=serde_json::from_value(raw.clone()).map_err(|_|Error::Evidence("GF shared backend runtime shape mismatch"))?;
            check_digest(&actual.backend_runtime_digest)?;
            (RuntimeBackend{backend_kind:actual.backend_kind,backend_namespace_digest:actual.backend_namespace_digest,tier:actual.tier},Some(actual.backend_runtime_digest))
        }else{
            (serde_json::from_value::<RuntimeBackend>(raw.clone()).map_err(|_|Error::Evidence("shared backend runtime shape mismatch"))?,None)
        };
        if model.is_empty() || origin.is_empty() || profile.rank_ids.is_empty() || profile.rank_ids.len()>256
            || actual.backend_kind.is_empty() || actual.backend_kind.len()>128 || actual.tier.is_empty() || actual.tier.len()>128 {
            return Err(Error::Evidence("shared backend missing actual geometry"));
        }
        for d in [&actual.backend_namespace_digest,&profile.kv_compat_digest,&profile.profile.runtime_profile_digest] {check_digest(d)?;}
        let backend=Backend{backend_kind:actual.backend_kind,backend_namespace_digest:actual.backend_namespace_digest,tier:actual.tier};
        Ok(Some(Self{model:model.into(),origin:origin.into(),runtime_profile_digest:profile.profile.runtime_profile_digest.clone(),
            compatibility_digest:profile.kv_compat_digest.clone(),ranks:profile.rank_ids.clone(),backend,backend_runtime_digest,promotion:None}))
    }
    fn attach_replicated_ds_promotion(&mut self,profile:&ParsedProfile,actual_native_identity:&str)->Result<()> {
        if profile.profile.family!="deepseek" || profile.profile.runtime_profile_digest!=self.runtime_profile_digest
            || profile.rank_ids!=self.ranks || profile.kv_compat_digest!=self.compatibility_digest {
            return Err(Error::Evidence("promotion requires actual loaded DeepSeek profile"));
        }
        let layout=&profile.profile.layout;
        let audit=&layout["execution_audit"];
        if audit["cohort"]["schema"]!="sglang.ds.cohort-kv-abi.v1" {return Err(Error::Evidence("promotion execution audit schema mismatch"));}
        let rows=audit["cohort"]["ranks"].as_array().ok_or(Error::Evidence("promotion actual resident cohort missing"))?;
        if rows.len()!=self.ranks.len() {return Err(Error::Evidence("promotion actual resident cohort incomplete"));}
        let mut resident=Vec::new();
        for (rank,row) in rows.iter().enumerate() {
            if row["rank_id"].as_u64()!=Some(rank as u64) || !row["descriptor"].is_object() {
                return Err(Error::Evidence("promotion actual resident rank/order mismatch"));
            }
            resident.push(row["descriptor"].clone());
        }
        let execution=audit["topologies"].as_array().ok_or(Error::Evidence("promotion actual target/draft topology missing"))?;
        let execution_digest=super::frontend_qualification::kv_digest(&json!({"resident":resident,"execution":execution}))
            .map_err(|_|Error::Evidence("promotion execution audit digest unavailable"))?;
        if layout["execution_cohort_digest"].as_str()!=Some(execution_digest.as_str()) {
            return Err(Error::Evidence("promotion execution audit digest differs from actual descriptors"));
        }
        let registered=layout["kv_compat_descriptor"]["registered_pools"].as_array()
            .ok_or(Error::Evidence("promotion actual registered component layout missing"))?;
        let mut components=BTreeSet::new();
        for row in registered {
            let name=row["name"].as_str().filter(|s|!s.is_empty()).ok_or(Error::Evidence("promotion component name missing"))?;
            if !components.insert(name.to_owned()) {return Err(Error::Evidence("promotion duplicated component"));}
        }
        self.promotion=Some(super::replicated_ds_promotion::QualifiedPromotion::from_runtime(&layout["replicated_promotion"],
            &self.runtime_profile_digest,&self.compatibility_digest,&self.backend.backend_namespace_digest,
            &self.ranks,&execution_digest,actual_native_identity,&components,&resident,execution)?);
        Ok(())
    }
    fn runtime(&self) -> [u8;32] {
        let mut h=Sha256::new(); h.update(b"dynamo-native-profile-binding-v1\0"); h.update(self.runtime_profile_digest.as_bytes()); h.finalize().into()
    }
    pub fn backend_namespace_digest(&self) -> &str { &self.backend.backend_namespace_digest }
    pub(crate) fn backend_runtime_digest(&self)->Option<&str> {self.backend_runtime_digest.as_deref()}
    pub fn kv_compat_digest(&self) -> &str { &self.compatibility_digest }
    pub fn tier(&self) -> &str { &self.backend.tier }
    pub fn model(&self) -> &str { &self.model }
    pub(crate) fn is_promotion_source(&self)->bool {self.promotion.as_ref().is_some_and(|p|p.source)}
}
fn check_digest(s: &str) -> Result<()> {
    if s.len()!=64 || s.bytes().all(|b|b==b'0') || !s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) { return Err(Error::Evidence("shared evidence requires actual nonzero lowercase SHA256")); }
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct SharedIdentity {
    model: String,
    compatibility_digest: String,
    backend_kind: String,
    backend_namespace_digest: String,
    tenant: String,
    namespace: String,
    digest: [u8;32],
    marked_endpoint_digest: [u8;32],
    marked_render_digest: [u8;32],
    tokens: u64,
    marked_tokens: u64,
}
impl SharedIdentity {
    fn new(q: &SharedBackendQualification, prefix: &Prefix) -> Result<Self> {
        prefix.key()?;
        if prefix.runtime != q.runtime() { return Err(Error::Evidence("shared prefix differs from actual signed runtime")); }
        Ok(Self { model:q.model.clone(), compatibility_digest:q.compatibility_digest.clone(), backend_kind:q.backend.backend_kind.clone(),
            backend_namespace_digest:q.backend.backend_namespace_digest.clone(), tenant:prefix.tenant.clone(), namespace:prefix.namespace.clone(),
            digest:prefix.digest, marked_endpoint_digest:prefix.marked_endpoint_digest, marked_render_digest:prefix.marked_render_digest,
            tokens:prefix.tokens, marked_tokens:prefix.marked_tokens })
    }
    fn root(&self) -> Result<String> {
        let mut h=Sha256::new(); h.update(b"dynamo-native-shared-logical-prefix-v1\0");
        h.update(cache_auth::canonical_bytes(&serde_json::to_value(self).map_err(|e|Error::Store(e.to_string()))?).map_err(Error::Store)?);
        Ok(format!("{ROOT}shared/{:x}/markers/",h.finalize()))
    }
}

/// Copied only from the actual signed common rank/tier receipt AFTER the
/// parent hook verifies complete native state, reserve ACK, clock, source,
/// full profile, incarnation, reservation and HTTP/native request identities.
#[derive(Clone, Debug)]
pub(crate) struct SharedRankRetain {
    pub rank_id:u32,
    pub backend_namespace_digest:String,
    pub shared_object_digest:String,
    pub tier:String,
    pub retained_tokens:u64,
    pub native_clock_domain:String,
    pub receive_monotonic_ns:u64,
    pub monotonic_event_ns:u64,
    pub retain_until_monotonic_ns:u64,
    pub ack_retain_for_ms:u64,
}

/// Private-construction proof, not a public cache configuration DTO. Different
/// ranks can hold different objects; the exact rank-to-object map is retained.
pub(crate) struct VerifiedSharedRetain {
    identity:SharedIdentity,
    evidence:NativeEvidence,
    rank_objects:BTreeMap<u32,String>,
    runtime_profile_digest:String,
    original_source_packet:Option<Value>,
}
impl VerifiedSharedRetain {
    pub(crate) fn from_verified_native(q:&SharedBackendQualification,e:&NativeEvidence,ranks:Vec<SharedRankRetain>) -> Result<Self> {
        let identity=SharedIdentity::new(q,&e.prefix)?;
        if e.worker!=q.origin || e.incarnation.is_empty() || e.receipt_id.is_empty() || e.native_request_id.is_empty()
            || e.committed_tokens<e.prefix.tokens || e.pinned_until_ms<=e.observed_at_ms {
            return Err(Error::Evidence("shared retain differs from verified actual native event"));
        }
        let ttl=e.pinned_until_ms.checked_sub(e.request_started_at_ms).ok_or(Error::Evidence("shared original expiry invalid"))?;
        if !matches!(ttl,300000|3600000) {return Err(Error::Evidence("shared original TTL invalid"));}
        let mut objects=BTreeMap::new();
        for rank in ranks {
            check_digest(&rank.shared_object_digest)?;
            if !q.ranks.contains(&rank.rank_id) || objects.insert(rank.rank_id,rank.shared_object_digest).is_some()
                || rank.backend_namespace_digest!=q.backend.backend_namespace_digest
                || rank.tier!=q.backend.tier
                || rank.retained_tokens<e.prefix.tokens || rank.native_clock_domain.is_empty()
                || rank.ack_retain_for_ms==0 || rank.ack_retain_for_ms>ttl || rank.ack_retain_for_ms<e.pinned_until_ms-e.observed_at_ms {
                return Err(Error::Evidence("shared retain lacks complete qualified rank backend"));
            }
            let until=rank.ack_retain_for_ms.checked_mul(1_000_000).and_then(|d|rank.receive_monotonic_ns.checked_add(d))
                .ok_or(Error::Evidence("shared rank duration overflow"))?;
            if until!=rank.retain_until_monotonic_ns || rank.monotonic_event_ns<rank.receive_monotonic_ns || rank.monotonic_event_ns>=until {
                return Err(Error::Evidence("shared rank did not commit within actual reserved hold"));
            }
        }
        if objects.keys().copied().collect::<BTreeSet<_>>()!=q.ranks {return Err(Error::Evidence("shared retain omitted native rank"));}
        Ok(Self {identity,evidence:e.clone(),rank_objects:objects,runtime_profile_digest:q.runtime_profile_digest.clone(),original_source_packet:None})
    }
    /// Called only by the complete native packet verifier, after original Gen
    /// HMAC, immutable ACK, all-rank events and hold checks succeeded.
    pub(crate) fn preserve_verified_source_packet(&mut self,packet:&Value) {self.original_source_packet=Some(packet.clone());}
}

#[derive(Clone,Debug,Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
struct SharedMarker {
    identity:SharedIdentity,
    source:Pin,
    source_runtime_profile_digest:String,
    source_native_request_id:String,
    request_started_at_ms:u64,
    ttl_ms:u64,
    rank_objects:BTreeMap<u32,String>,
    promotion:Option<super::replicated_ds_promotion::PromotionSourceAudit>,
    original_source_evidence:Option<NativeEvidence>,
    original_source_packet:Option<Value>,
}
impl SharedMarker {
    fn validate(&self,q:&SharedBackendQualification,identity:&SharedIdentity)->Result<()> {
        if self.identity!=*identity || self.source.worker.is_empty() || self.source.incarnation.is_empty() || self.source.receipt_id.is_empty()
            || self.source_native_request_id.is_empty() || !matches!(self.ttl_ms,300000|3600000)
            || self.request_started_at_ms.checked_add(self.ttl_ms)!=Some(self.source.pinned_until_ms) {
            return Err(Error::Evidence("shared marker identity/rank/expiry mismatch"));
        }
        check_digest(&self.source_runtime_profile_digest)?;
        for object in self.rank_objects.values() { check_digest(object)?; }
        if let Some(audit)=&self.promotion {
            let mapping=q.promotion.as_ref().ok_or(Error::Evidence("source promotion qualification absent"))?;
            mapping.verify_source(audit,&self.source_runtime_profile_digest,&self.source.incarnation,&self.rank_objects)?;
            let evidence=self.original_source_evidence.as_ref().ok_or(Error::Evidence("promotion original source evidence absent"))?;
            let packet=self.original_source_packet.as_ref().ok_or(Error::Evidence("promotion original signed source packet absent"))?;
            if pin_from(evidence)!=self.source || evidence.native_request_id!=self.source_native_request_id
                || evidence.request_started_at_ms!=self.request_started_at_ms || evidence.prefix.tokens!=self.identity.tokens
                || evidence.prefix.marked_tokens!=self.identity.marked_tokens || evidence.prefix.digest!=self.identity.digest
                || evidence.prefix.marked_endpoint_digest!=self.identity.marked_endpoint_digest
                || evidence.prefix.marked_render_digest!=self.identity.marked_render_digest
                || evidence.prefix.namespace!=self.identity.namespace || evidence.prefix.tenant!=self.identity.tenant
                || packet["receipt"]["rank_receipts"].as_array().map(|r|r.len())!=Some(4)
                || packet[cache_auth::PROOF_FIELD].as_object().is_none()
                || packet[cache_auth::PROOF_FIELD]["claims"]["runtime_profile_digest"].as_str()!=Some(self.source_runtime_profile_digest.as_str())
                || packet[cache_auth::PROOF_FIELD]["claims"]["native_incarnation"].as_str()!=Some(self.source.incarnation.as_str()) {
                return Err(Error::Evidence("promotion source marker differs from original four-rank proof"));
            }
        }else if self.rank_objects.keys().copied().collect::<BTreeSet<_>>()!=q.ranks {
            return Err(Error::Evidence("ordinary shared marker native rank set mismatch"));
        }
        Ok(())
    }
}

/// A lookup permit is NOT usage or proof of a native cache hit.
#[derive(Clone,Debug)]
pub struct SharedImportPermit { marker:SharedMarker,key:String,revision:i64,lease:i64 }
impl SharedImportPermit {
    pub fn rank_objects(&self)->&BTreeMap<u32,String> {&self.marker.rank_objects}
    pub fn source_worker(&self)->&str {&self.marker.source.worker}
    pub fn source_incarnation(&self)->&str {&self.marker.source.incarnation}
    pub fn source_expires_at_ms(&self)->u64 {self.marker.source.pinned_until_ms}
    pub(crate) fn original_source_proof(&self)->Option<(&NativeEvidence,&Value)> {
        self.marker.original_source_evidence.as_ref().zip(self.marker.original_source_packet.as_ref())
    }
    pub fn ttl(&self)->Ttl {if self.marker.ttl_ms==300000 {Ttl::FiveMinutes}else{Ttl::OneHour}}
    pub fn companion(&self,endpoint_id:&str)->Result<Value> {
        if endpoint_id.is_empty() || endpoint_id.len()>1024 {return Err(Error::Evidence("missing shared read endpoint"));}
        Ok(json!({"backend_namespace_digest":self.marker.identity.backend_namespace_digest,"kv_compat_digest":self.marker.identity.compatibility_digest,
            "endpoint_id":endpoint_id,"source_expires_at_ms":self.source_expires_at_ms(),
            "source_rank_objects":self.marker.rank_objects.iter().map(|(rank,object)|json!({"rank_id":rank,"shared_object_digest":object})).collect::<Vec<_>>() }))
    }
    /// Values must come from actual signed tier fields sampled AFTER completed
    /// import+hold by the profile-qualified native clock provider. This is the
    /// conservative path; no atomic source-read lease is inferred here.
    pub(crate) fn verify_source_read_clocks(&self,q:&SharedBackendQualification,request_started_at_ms:u64,rank_clocks:&BTreeMap<u32,(u64,u64)>)->Result<()> {
        let expected=if self.marker.promotion.is_some() {
            let mapping=q.promotion.as_ref().ok_or(Error::Evidence("source promotion qualification absent"))?;
            if mapping.source {return Err(Error::Evidence("source cohort cannot consume destination promotion"));}
            q.ranks.clone()
        }else{self.marker.rank_objects.keys().copied().collect()};
        if rank_clocks.keys().copied().collect::<BTreeSet<_>>()!=expected {
            return Err(Error::Evidence("source read clock proof omitted native rank"));
        }
        for (upper,error) in rank_clocks.values() {
            if *upper==0 || *error>100 || *upper<request_started_at_ms || *upper<=*error || *upper>=self.source_expires_at_ms() {
                return Err(Error::Evidence("actual import completed outside qualified source lifetime"));
            }
        }
        Ok(())
    }
    /// Invoke ONLY after the concrete destination parser verifies actual signed
    /// completed read/import and new non-evictable hold on every native rank.
    /// These are actual imported object IDs from that receipt, never the request
    /// companion echoed back. The parent checks each actual signed source-read
    /// upper/error bound against companion.source_expires_at_ms BEFORE calling
    /// this function (or a separately qualified atomic source read lease). A
    /// registry lookup cannot call this itself.
    pub(crate) fn verify_destination_import(&self,q:&SharedBackendQualification,e:&NativeEvidence,imported_objects:&BTreeMap<u32,String>,clock:&ClockBounds)->Result<u64> {
        let identity=SharedIdentity::new(q,&e.prefix)?;
        self.marker.validate(q,&identity)?;
        fresh_upper(clock)?;
        if e.kind!=EventKind::ReadHit || e.worker!=q.origin || e.incarnation.is_empty() || e.receipt_id.is_empty()
            || e.native_request_id.is_empty() || e.committed_tokens<identity.tokens || e.pinned_until_ms<=e.observed_at_ms {
            return Err(Error::Evidence("no actual all-rank destination import and hold"));
        }
        if self.marker.promotion.is_some() {
            let mapping=q.promotion.as_ref().ok_or(Error::Evidence("promotion destination qualification absent"))?;
            if mapping.source || e.incarnation!=mapping.mapping.destination_native_incarnation
                || imported_objects.keys().copied().collect::<BTreeSet<_>>()!=q.ranks
                || imported_objects.values().any(|object|self.marker.rank_objects.values().next()!=Some(object)) {
                return Err(Error::Evidence("no actual mapped eight-rank destination import"));
            }
        }else if imported_objects!=&self.marker.rank_objects {
            return Err(Error::Evidence("ordinary shared imported objects differ"));
        }
        Ok(identity.tokens)
    }
}
fn fresh_upper(clock:&ClockBounds)->Result<u64> {
    clock.ensure_fresh().map_err(|_|Error::Evidence("shared authority clock is stale"))?;
    let elapsed=Instant::now().checked_duration_since(clock.observed_at()).ok_or(Error::Evidence("shared clock reversed"))?;
    let ms=u64::try_from(elapsed.as_nanos().div_ceil(1_000_000)).map_err(|_|Error::Evidence("shared clock overflow"))?;
    clock.upper_ms().checked_add(ms).ok_or(Error::Evidence("shared clock overflow"))
}

impl EtcdStore {
    pub(crate) async fn publish_shared_marker(&self,q:&SharedBackendQualification,event:&VerifiedSharedRetain,lease:&OriginLease,reservation:&str,revision:i64,clock:&ClockBounds)->Result<bool> {
        if reservation.is_empty() || revision<=0 || lease.id==0 {return Err(Error::Evidence("missing immutable shared reservation"));}
        if SharedIdentity::new(q,&event.evidence.prefix)?!=event.identity || event.evidence.worker!=q.origin {return Err(Error::Evidence("shared publication qualification changed"));}
        if event.evidence.pinned_until_ms<=fresh_upper(clock)? {return Ok(false)};
        if bounded(self.client.clone().lease_time_to_live(lease.id,None)).await?.ttl()<=0 {return Ok(false)};
        let mut h=Sha256::new();h.update((event.evidence.worker.len()as u64).to_be_bytes());h.update(event.evidence.worker.as_bytes());h.update(event.evidence.incarnation.as_bytes());
        let key=format!("{}{:x}",event.identity.root()?,h.finalize());
        let marker=SharedMarker {identity:event.identity.clone(),source:pin_from(&event.evidence),source_runtime_profile_digest:event.runtime_profile_digest.clone(),
            source_native_request_id:event.evidence.native_request_id.clone(),
            request_started_at_ms:event.evidence.request_started_at_ms,ttl_ms:event.evidence.pinned_until_ms-event.evidence.request_started_at_ms,rank_objects:event.rank_objects.clone(),
            promotion:q.promotion.as_ref().filter(|p|p.source).map(|p|p.source_audit(&event.runtime_profile_digest)),
            original_source_evidence:if q.is_promotion_source(){Some(event.evidence.clone())}else{None},
            original_source_packet:if q.is_promotion_source(){event.original_source_packet.clone()}else{None}};
        marker.validate(q,&event.identity)?;
        let bytes=serde_json::to_vec(&marker).map_err(|e|Error::Store(e.to_string()))?;
        let reservation_key=format!("{ROOT}reservations/{reservation}");
        for _ in 0..8 {
            let old=bounded(self.client.clone().get(key.as_str(),None)).await?;
            let now=fresh_upper(clock)?;
            if event.evidence.pinned_until_ms<=now {return Ok(false)};
            let row=old.kvs().first();
            if let Some(row)=row {
                let previous:SharedMarker=serde_json::from_slice(row.value()).map_err(|e|Error::Store(e.to_string()))?;
                previous.validate(q,&event.identity)?;
                if previous.source.worker!=marker.source.worker || previous.source.incarnation!=marker.source.incarnation
                    || previous.source_runtime_profile_digest!=marker.source_runtime_profile_digest {return Err(Error::Evidence("shared marker source owner collision"));}
                if previous.source.pinned_until_ms>=marker.source.pinned_until_ms && previous.source.pinned_until_ms>now {return Ok(true)};
            }
            let compare=if let Some(row)=row {Compare::mod_revision(key.as_str(),CompareOp::Equal,row.mod_revision())}else{Compare::version(key.as_str(),CompareOp::Equal,0)};
            let result=bounded(self.client.clone().txn(Txn::new().when(vec![Compare::mod_revision(reservation_key.clone(),CompareOp::Equal,revision),compare])
                .and_then(vec![TxnOp::put(key.as_str(),bytes.clone(),Some(PutOptions::new().with_lease(lease.id)))]))).await?;
            if result.succeeded() {return Ok(event.evidence.pinned_until_ms>fresh_upper(clock)?)};
        }
        Err(Error::Contention)
    }
    pub(crate) async fn lookup_shared_marker(&self,q:&SharedBackendQualification,prefix:&Prefix,clock:&ClockBounds)->Result<Option<SharedImportPermit>> {
        let identity=SharedIdentity::new(q,prefix)?;
        fresh_upper(clock)?;
        let result=bounded(self.client.clone().get(identity.root()?,Some(GetOptions::new().with_prefix().with_limit((MAX_PINS+1)as i64)))).await?;
        if result.kvs().len()>MAX_PINS {return Err(Error::Evidence("too many live shared origin markers"));}
        let mut best:Option<SharedImportPermit>=None;
        for row in result.kvs() {
            let marker:SharedMarker=serde_json::from_slice(row.value()).map_err(|e|Error::Store(e.to_string()))?;
            // A differently qualified cohort is outside this lookup. It cannot
            // enable promotion or invalidate an ordinary same-cohort cache.
            if let Some(audit)=&marker.promotion {
                if !q.promotion.as_ref().is_some_and(|p|p.digest==audit.mapping_digest) {continue};
            }else if marker.rank_objects.keys().copied().collect::<BTreeSet<_>>()!=q.ranks {continue};
            marker.validate(q,&identity)?;
            if row.lease()==0 || marker.source.pinned_until_ms<=fresh_upper(clock)? {continue};
            let current=bounded(self.client.clone().get(row.key(),None)).await?;
            if marker.source.pinned_until_ms<=fresh_upper(clock)? {continue};
            if !current.kvs().first().is_some_and(|r|r.mod_revision()==row.mod_revision()&&r.lease()==row.lease()) {continue};
            if best.as_ref().is_none_or(|old|old.marker.source.pinned_until_ms<marker.source.pinned_until_ms) {
                best=Some(SharedImportPermit{marker,key:String::from_utf8(row.key().to_vec()).map_err(|e|Error::Store(e.to_string()))?,revision:row.mod_revision(),lease:row.lease()});
            }
        }
        Ok(best)
    }
    /// Immediately before dispatch, recheck exactly the selected shared marker;
    /// cancellation/republication/expiry never silently selects a new object.
    pub(crate) async fn confirm_shared_import(&self,permit:&SharedImportPermit,clock:&ClockBounds)->Result<bool> {
        let confirmed=self.confirm_shared_import_at_upper(permit,fresh_upper(clock)?).await?;
        clock.ensure_fresh().map_err(|_|Error::Evidence("shared authority clock expired during confirmation"))?;
        Ok(confirmed)
    }
    /// Caller supplies a fresh qualified SigningContext upper bound, already
    /// advanced to this call. Charge this read's own monotonic latency too;
    /// never treat an old source entry as live after a slow etcd operation.
    pub(crate) async fn confirm_shared_import_at_upper(&self,permit:&SharedImportPermit,now_upper_ms:u64)->Result<bool> {
        if now_upper_ms==0 {return Err(Error::Evidence("qualified shared confirmation clock missing"));}
        if permit.marker.source.pinned_until_ms<=now_upper_ms {return Ok(false)};
        let started=Instant::now();
        let current=bounded(self.client.clone().get(permit.key.as_str(),None)).await?;
        let elapsed=started.elapsed();
        if elapsed>Duration::from_millis(100) {return Err(Error::Evidence("qualified shared confirmation clock exceeded freshness"));}
        let now=now_upper_ms.checked_add(u64::try_from(elapsed.as_nanos().div_ceil(1_000_000)).map_err(|_|Error::Evidence("shared confirmation clock overflow"))?)
            .ok_or(Error::Evidence("shared confirmation clock overflow"))?;
        Ok(permit.marker.source.pinned_until_ms>now && current.kvs().first().is_some_and(|r|r.mod_revision()==permit.revision&&r.lease()==permit.lease))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::frontend_qualification::Profile;
    #[tokio::test]
    #[ignore="requires owned disposable loopback etcd fixture"]
    async fn real_etcd_cross_origin_import_permit_and_revoke() {
        let store=EtcdStore::connect(vec![std::env::var("CACHE_AUTHORITY_TEST_ETCD").unwrap()]).await.unwrap();
        let source=qualification("http://source","aa".repeat(32).as_str());
        let destination=qualification("http://destination","bb".repeat(32).as_str());
        let e=evidence(&source);let proof=VerifiedSharedRetain::from_verified_native(&source,&e,ranks(&source)).unwrap();
        let lease=store.grant_origin_lease(200_000).await.unwrap().unwrap();
        let reservation="owned-shared-import";
        let row=bounded(store.client.clone().put(format!("{ROOT}reservations/{reservation}"),"CPU-shared-proof",None)).await.unwrap();
        let clock=ClockBounds::cpu_fixture(Instant::now());
        assert!(store.publish_shared_marker(&source,&proof,&lease,reservation,row.header().unwrap().revision(),&clock).await.unwrap());
        let clock=ClockBounds::cpu_fixture(Instant::now());
        let permit=store.lookup_shared_marker(&destination,&prefix(&destination),&clock).await.unwrap().unwrap();
        assert_eq!(permit.source_worker(),"http://source");assert_ne!(permit.rank_objects()[&0],permit.rank_objects()[&1]);
        assert_eq!(permit.companion("read:e").unwrap()["source_expires_at_ms"],1_200_000);
        let mut read=evidence(&destination);read.kind=EventKind::ReadHit;
        let clock=ClockBounds::cpu_fixture(Instant::now());
        permit.verify_source_read_clocks(&destination,read.request_started_at_ms,&BTreeMap::from([(0,(1_000_002,1)),(1,(1_000_002,1))])).unwrap();
        assert_eq!(permit.verify_destination_import(&destination,&read,permit.rank_objects(),&clock).unwrap(),2048);
        assert!(store.confirm_shared_import(&permit,&clock).await.unwrap());
        bounded(store.client.clone().lease_revoke(lease.id)).await.unwrap();
        let clock=ClockBounds::cpu_fixture(Instant::now());
        assert!(!store.confirm_shared_import(&permit,&clock).await.unwrap());
        assert!(store.lookup_shared_marker(&destination,&prefix(&destination),&clock).await.unwrap().is_none());
    }
    fn qualification(origin:&str,runtime:&str)->SharedBackendQualification {
        let p=ParsedProfile {profile:Profile {family:"qwen".into(),native_origin_uuid:"CPU-incarnation".into(),
            layout:json!({"shared_cache":{"backend_kind":"shared_posix","backend_namespace_digest":"22".repeat(32),"tier":"l3"}}),runtime_profile_digest:runtime.into()},
            source_digest:"33".repeat(32),kv_compat_digest:"44".repeat(32),rank_ids:BTreeSet::from([0,1])};
        SharedBackendQualification::from_profile("model",origin,&p).unwrap().unwrap()
    }
    fn prefix(q:&SharedBackendQualification)->Prefix { Prefix {tenant:"tenant".into(),namespace:"actual-ABI-bound-namespace".into(),runtime:q.runtime(),digest:[1;32],marked_endpoint_digest:[2;32],marked_render_digest:[3;32],tokens:2048,marked_tokens:2100} }
    fn evidence(q:&SharedBackendQualification)->NativeEvidence { NativeEvidence {request_id:"http-origin".into(),native_request_id:"native-rid".into(),prefix:prefix(q),worker:q.origin.clone(),incarnation:"incarnation".into(),receipt_id:"native-shared-pin".into(),kind:EventKind::WriteCommitted,observed_at_ms:1_000_000,request_started_at_ms:900_000,pinned_until_ms:1_200_000,committed_tokens:2048} }
    fn ranks(q:&SharedBackendQualification)->Vec<SharedRankRetain> {(0..2).map(|rank|SharedRankRetain {rank_id:rank,backend_namespace_digest:q.backend.backend_namespace_digest.clone(),shared_object_digest:format!("{:064x}",rank+1),tier:"l3".into(),retained_tokens:2048,native_clock_domain:format!("actual-rank-{rank}"),receive_monotonic_ns:1_000_000,monotonic_event_ns:2_000_000,retain_until_monotonic_ns:200_001_000_000,ack_retain_for_ms:200_000}).collect()}
    #[test]
    fn stable_cross_origin_key_keeps_all_physical_boundaries() {
        let a=qualification("http://source","aa".repeat(32).as_str());let b=qualification("http://destination","bb".repeat(32).as_str());
        let key=SharedIdentity::new(&a,&prefix(&a)).unwrap().root().unwrap();
        assert_eq!(key,SharedIdentity::new(&b,&prefix(&b)).unwrap().root().unwrap());
        for field in 0..6 {let mut p=prefix(&b); match field {0=>p.tenant="other".into(),1=>p.namespace="other".into(),2=>p.digest=[4;32],3=>p.marked_endpoint_digest=[4;32],4=>p.marked_render_digest=[4;32],_=>p.marked_tokens+=1};assert_ne!(key,SharedIdentity::new(&b,&p).unwrap().root().unwrap());}
        let mut changed=b.clone();changed.backend.backend_namespace_digest="55".repeat(32);assert_ne!(key,SharedIdentity::new(&changed,&prefix(&changed)).unwrap().root().unwrap());
        assert!(SharedIdentity::new(&b,&prefix(&a)).is_err());
    }
    #[test]
    fn source_requires_actual_all_rank_backend_hold_and_import_is_not_a_lookup() {
        let source=qualification("http://source","aa".repeat(32).as_str());let e=evidence(&source);
        let valid=VerifiedSharedRetain::from_verified_native(&source,&e,ranks(&source)).unwrap();
        assert_ne!(valid.rank_objects[&0],valid.rank_objects[&1]);
        let mut missing=ranks(&source);missing.pop();assert!(VerifiedSharedRetain::from_verified_native(&source,&e,missing).is_err());
        let mut duplicate=ranks(&source);duplicate[1].rank_id=0;assert!(VerifiedSharedRetain::from_verified_native(&source,&e,duplicate).is_err());
        let mut local=ranks(&source);local[0].backend_namespace_digest="66".repeat(32);assert!(VerifiedSharedRetain::from_verified_native(&source,&e,local).is_err());
        let mut expired=ranks(&source);expired[0].monotonic_event_ns=expired[0].retain_until_monotonic_ns;assert!(VerifiedSharedRetain::from_verified_native(&source,&e,expired).is_err());
        let marker=SharedMarker{identity:valid.identity,source:pin_from(&e),source_runtime_profile_digest:source.runtime_profile_digest.clone(),source_native_request_id:e.native_request_id.clone(),request_started_at_ms:e.request_started_at_ms,ttl_ms:300000,rank_objects:valid.rank_objects,promotion:None,original_source_evidence:None,original_source_packet:None};
        let permit=SharedImportPermit {marker,key:"cpu-fixture".into(),revision:1,lease:1};
        assert_eq!(permit.companion("selected-read").unwrap()["source_rank_objects"][1]["rank_id"],1);
        assert_eq!(permit.companion("selected-read").unwrap()["source_expires_at_ms"],1_200_000);
        assert_eq!(permit.companion("selected-read").unwrap().as_object().unwrap().len(),5);
        let clocks=BTreeMap::from([(0,(1_199_999,100)),(1,(1_199_999,100))]);
        permit.verify_source_read_clocks(&source,900_000,&clocks).unwrap();
        for changed in [(1_200_000,100),(1_199_999,101),(899_999,100)] {
            let mut invalid=clocks.clone();invalid.insert(1,changed);assert!(permit.verify_source_read_clocks(&source,900_000,&invalid).is_err());
        }
        let mut missing_clocks=clocks;missing_clocks.remove(&1);assert!(permit.verify_source_read_clocks(&source,900_000,&missing_clocks).is_err());
        let destination=qualification("http://destination","bb".repeat(32).as_str());let mut read=evidence(&destination);let clock=ClockBounds::cpu_fixture(Instant::now());
        assert!(permit.verify_destination_import(&destination,&read,permit.rank_objects(),&clock).is_err());
        read.kind=EventKind::ReadHit;assert_eq!(permit.verify_destination_import(&destination,&read,permit.rank_objects(),&clock).unwrap(),2048);
        let mut wrong=permit.rank_objects().clone();wrong.insert(1,"77".repeat(32));assert!(permit.verify_destination_import(&destination,&read,&wrong,&clock).is_err());
        read.committed_tokens=2047;assert!(permit.verify_destination_import(&destination,&read,permit.rank_objects(),&clock).is_err());
    }
    #[test]
    fn native_ds_promotion_new_source_four_destination_eight_keeps_proof() {
        use super::super::replicated_ds_promotion::tests::{cpu_outer,cpu_parse};
        let outer=cpu_outer();
        let mut source=qualification("http://source",&"a".repeat(64));
        let mut destination=qualification("http://destination",&"b".repeat(64));
        source.ranks=(0..4).collect();destination.ranks=(0..8).collect();
        for q in [&mut source,&mut destination] {
            q.compatibility_digest="6".repeat(64);q.backend.backend_namespace_digest="9".repeat(64);
        }
        source.promotion=Some(cpu_parse(&outer,true).unwrap());
        destination.promotion=Some(cpu_parse(&outer,false).unwrap());
        let mut e=evidence(&source);e.incarnation="CPU_NOTQUALIFIED_source".into();
        let prototype=ranks(&source).remove(0);
        let holds=(0..4).map(|rank| {let mut hold=prototype.clone();hold.rank_id=rank;
            hold.native_clock_domain=format!("CPU_NOTQUALIFIED_rank_{rank}");hold.shared_object_digest="c".repeat(64);hold}).collect();
        let mut retained=VerifiedSharedRetain::from_verified_native(&source,&e,holds).unwrap();
        let mut packet=json!({"event_kind":"write_commit","receipt":{"rank_receipts":(0..4).map(|r|json!({"rank_id":r})).collect::<Vec<_>>()}});
        let claims=json!({"version":1,"operation":"receipts","operation_id":"CPU_original_Gen","nonce":"CPU_original_nonce",
            "model":"model","native_origin":"http://source","native_incarnation":e.incarnation,
            "source_digest":"d".repeat(64),"runtime_profile_digest":source.runtime_profile_digest,
            "reservation_id":"CPU_source_reservation","authority_revision":1,"native_request_id":e.native_request_id,
            "namespace_digest":"e".repeat(64),"full_render_digest":"f".repeat(64),"token_ids_digest":"1".repeat(64),
            "request_started_at_ms":900000,"issued_at_ms":1000000,"deadline_ms":1200000,"retain_for_ms":200000});
        let proof=cache_auth::sign(&[0x33;32],&claims,&packet).unwrap();packet[cache_auth::PROOF_FIELD]=proof;
        retained.preserve_verified_source_packet(&packet);
        let marker=SharedMarker{identity:retained.identity.clone(),source:pin_from(&e),
            source_runtime_profile_digest:source.runtime_profile_digest.clone(),
            source_native_request_id:e.native_request_id.clone(),request_started_at_ms:e.request_started_at_ms,ttl_ms:300000,
            rank_objects:retained.rank_objects.clone(),promotion:Some(source.promotion.as_ref().unwrap().source_audit(&source.runtime_profile_digest)),
            original_source_evidence:Some(e),original_source_packet:retained.original_source_packet.clone()};
        let permit=SharedImportPermit{marker,key:"CPU_NOTQUALIFIED_marker".into(),revision:1,lease:1};
        permit.marker.validate(&destination,&SharedIdentity::new(&destination,&prefix(&destination)).unwrap()).unwrap();
        assert_eq!(permit.rank_objects().len(),4);
        assert_eq!(permit.companion("read").unwrap().as_object().unwrap().len(),5);
        assert_eq!(permit.original_source_proof().unwrap().1,&packet);
        cache_auth::verify_signature(&[0x33;32],&packet[cache_auth::PROOF_FIELD],&cache_auth::payload_for("receipts",&packet).unwrap()).unwrap();
        let mut read=evidence(&destination);read.kind=EventKind::ReadHit;read.incarnation="CPU_NOTQUALIFIED_destination".into();
        let objects:BTreeMap<_,_>=(0..8).map(|r|(r,"c".repeat(64))).collect();
        let clocks:BTreeMap<_,_>=(0..8).map(|r|(r,(1_000_002,1))).collect();
        permit.verify_source_read_clocks(&destination,900000,&clocks).unwrap();
        assert_eq!(permit.verify_destination_import(&destination,&read,&objects,&ClockBounds::cpu_fixture(Instant::now())).unwrap(),2048);
        assert!(permit.verify_destination_import(&destination,&read,permit.rank_objects(),&ClockBounds::cpu_fixture(Instant::now())).is_err());
        let mut missing=clocks.clone();missing.remove(&7);assert!(permit.verify_source_read_clocks(&destination,900000,&missing).is_err());
        let mut expired=clocks;expired.insert(7,(1_200_000,1));assert!(permit.verify_source_read_clocks(&destination,900000,&expired).is_err());
        let mut wrong=objects;wrong.insert(7,"d".repeat(64));assert!(permit.verify_destination_import(&destination,&read,&wrong,&ClockBounds::cpu_fixture(Instant::now())).is_err());
        let mut ordinary=destination;ordinary.promotion=None;
        assert!(permit.marker.validate(&ordinary,&SharedIdentity::new(&ordinary,&prefix(&ordinary)).unwrap()).is_err());
    }
}
