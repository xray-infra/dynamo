//! Cache controls run through the existing native HTTP port and reqwest pool.
//! A typed service context is required; public JSON cannot opt into this path.
use super::{cache_auth, frontend_cache_hook::*, frontend_clock::ChronyClock, Error, EtcdStore, Prefix, Result, Ttl};
use serde_json::{json,Value};
use std::{collections::{BTreeMap,BTreeSet},sync::Arc,time::Duration};
use tokio::time::Instant;
use super::cache_accounting::{CacheAccounting,VerifiedCacheAccounting};
use super::shared_cache_authority::SharedBackendQualification;
use crate::native_cleanup::{NativeCleanupRegistry,CleanupReceipt,CleanupFailure};
use tokio::sync::{oneshot,mpsc};

fn observe_cache_http_error(path:&str,status:u16,value:&Value) {
    let safe=crate::native_pd::dispatch::native_error_fields_from_value(value);
    let message=safe.as_ref().and_then(|v|v.get("message")).and_then(Value::as_str);
    let detail=safe.as_ref().and_then(|v|v.get("detail")).and_then(Value::as_str);
    tracing::warn!(target:"dynamo_llm::native_cache", path, status,
        native_message=message.unwrap_or(""), native_detail=detail.unwrap_or(""), "native cache HTTP operation failed");
}

pub(crate) fn cache_http_error(status:u16,value:&Value)->Error {
    let summary=crate::native_pd::dispatch::native_error_fields_from_value(value);
    let cause=summary.as_ref().and_then(|v|v.get("message").or_else(||v.get("detail")))
        .and_then(Value::as_str).unwrap_or("Native cache operation rejected").to_owned();
    Error::NativeHttp { status, cause }
}

fn observe_receipt_error(stage:&'static str,native_request_id:&str,error:Error)->Error {
    let reason=match &error {
        Error::Evidence(reason)=>*reason,
        Error::Store(reason)=>match reason.as_str() {
            "payload mismatch"=>"payload mismatch",
            "signature mismatch"=>"signature mismatch",
            "claim fields mismatch"=>"claim fields mismatch",
            "integer claim mismatch"=>"integer claim mismatch",
            "string claim mismatch"=>"string claim mismatch",
            "time order mismatch"=>"time order mismatch",
            "digest mismatch"=>"digest mismatch",
            _=>"registry or codec operation failed",
        },
        Error::Contention=>"registry CAS contention",
        Error::NativeHttp{..}=>"native HTTP rejection",
    };
    // Evidence reasons and codec labels above are code-owned constants. Never
    // log the packet, proof, secret, or dynamic storage/transport error text.
    tracing::warn!(target:"dynamo_llm::native_cache",stage,native_request_id,reason,
        "native cache receipt rejected");
    error
}

pub struct CacheRequest {
    pub http_request_id:String,
    pub http_started_at:std::time::Instant,
    pub writes:BTreeMap<String,Ttl>,
    pub reads:BTreeSet<String>,
    start_lower_ms:std::sync::OnceLock<u64>,
}
impl CacheRequest {
    pub fn new(http_request_id:String,http_started_at:std::time::Instant,writes:BTreeMap<String,Ttl>,reads:BTreeSet<String>)->Self {
        Self{http_request_id,http_started_at,writes,reads,start_lower_ms:std::sync::OnceLock::new()}
    }
    /// Count, selected-worker recount and Reserve spend the same original
    /// control budget; another native operation cannot renew it.
    pub fn control_deadline(&self)->Result<Instant> {
        Instant::from_std(self.http_started_at).checked_add(Duration::from_secs(600))
            .ok_or(Error::Evidence("cache control deadline overflow"))
    }
}

fn control_remaining(deadline:Instant)->Result<Duration> {
    deadline.checked_duration_since(Instant::now()).filter(|left|!left.is_zero())
        .ok_or(Error::Evidence("native cache control deadline expired"))
}
fn cleanup_deadline(d:&Dispatch,realm_deadline:Instant)->Result<Instant> {
    let bounded=Instant::now().checked_add(Duration::from_secs(30)).ok_or(Error::Evidence("cache cleanup deadline overflow"))?;
    Ok(bounded.min(d.origin().operation_deadline()?).min(realm_deadline))
}

/// Startup-only admitted metadata. The profile/deployment loader and controller
/// qualification supply this material; no values are copied from request JSON.
pub struct FrontendWorkerBinding {
    pub worker:QualifiedWorker,
    pub model:String,
    pub source_digest:String,
    pub kv_compat_digest:String,
    pub count_path:String,
    pub tenant:String,
    pub default_cache_salt:String,
    pub secret:[u8;32],
    pub shared_backend:Option<SharedBackendQualification>,
}
impl FrontendWorkerBinding {
    /// The running native metadata provides identity and actual KV geometry.
    pub fn from_startup(material:super::frontend_qualification::FrontendStartupMaterial,native_incarnation:String,mut required_tiers:BTreeSet<String>,count_path:String,tenant:String,default_cache_salt:String)->Result<Self> {
        if native_incarnation!=material.profile.profile.native_origin_uuid {return Err(Error::Evidence("native cohort differs from runtime metadata"));}
        if native_incarnation.is_empty() || required_tiers.is_empty() {return Err(Error::Evidence("actual native cohort binding absent"));}
        let secret:[u8;32]=material.secret_bytes().try_into().map_err(|_|Error::Evidence("private model secret length"))?;
        let shared_backend=SharedBackendQualification::from_startup(&material)?;
        if let Some(qualification)=&shared_backend {required_tiers.insert(qualification.tier().into());}
        let worker=QualifiedWorker{worker_url:material.deployment.native_origin.clone(),native_origin_uuid:native_incarnation,
            runtime_profile_digest:material.profile.profile.runtime_profile_digest.clone(),namespace_digest:String::new(),native_clock_domain:"awaiting-signed-reserve-ACK".into(),
            required_tiers:required_tiers.clone(),native_id_field:material.native_id_field,
            rank_tier_clocks:material.profile.rank_ids.iter().map(|rank|(*rank,required_tiers.iter().map(|tier|(tier.clone(),String::new())).collect())).collect()};
        Ok(Self{worker,model:material.deployment.model,source_digest:material.profile.source_digest,kv_compat_digest:material.profile.kv_compat_digest,count_path,tenant,default_cache_salt,secret,shared_backend})
    }
}

pub struct NativeCacheFrontend {
    client:reqwest::Client,
    store:EtcdStore,
    workers:BTreeMap<String,FrontendWorkerBinding>,
    cleanup_registry:Option<Arc<NativeCleanupRegistry>>,
    #[cfg(test)] cpu_clock:bool,
}

pub struct RenderedCount {
    pub prompt_tokens:u64,
    pub worker_url:String,
    body:Value,
    tokens:Value,
    provenance:Value,
    origin:RequestOrigin,
    worker:QualifiedWorker,
    points:BTreeMap<String,RenderPoint>,
}
impl RenderedCount {
    /// Server render selectors are accepted only from this measured Count.
    pub(crate) fn validate_server_render_plan(&self,body:&Value)->Result<()> {
        if !body.is_object() {return Err(Error::Evidence("cached P/D body is not an object"));}
        for key in ["prompt_cache_blocks","prompt_cache_read_candidates","prompt_cache_message_order","rid","request_id","n","cache_salt"] {
            if body.get(key)!=self.body.get(key) {return Err(Error::Evidence("cached P/D render plan changed after Count"));}
        }
        self.cache_salt()?;
        native_id(body)?;
        Ok(())
    }
    /// The actual Count body's normalized salt; never a caller/default guess.
    pub fn cache_salt(&self)->Result<&str> {
        self.body.get("cache_salt").and_then(Value::as_str)
            .ok_or(Error::Evidence("rendered Count cache salt missing or invalid"))
    }
    /// Original trusted Count origin, retained across selected-worker recount.
    pub fn request_started_at_ms(&self)->u64 {self.origin.request_started_at_ms()}
}
pub struct PreparedCacheGeneration {pub body:Value,pub guard:CacheGenerationGuard}
/// Opaque request cleanup evidence. Native cancellation closes admission and
/// releases unconsumed reservations; committed TTL holds keep their expiry.
/// NoNativeReservation is a known no-native-reserve terminal, minted only by the owned task
/// when preparation failed before any Dispatch reached native HTTP.
pub struct VerifiedCacheCleanup {
    native_request_id:String,
    evidence:CacheCleanupEvidence,
}
enum CacheCleanupEvidence {
    NoNativeReservation,
    NativeCancellation {reservation_id:String,authority_revision:i64},
}
impl VerifiedCacheCleanup {
    pub(crate) fn native_request_id(&self)->&str {&self.native_request_id}
    pub(crate) fn cancellation(&self)->Option<(&str,i64)> {match &self.evidence {
        CacheCleanupEvidence::NoNativeReservation=>None,
        CacheCleanupEvidence::NativeCancellation{reservation_id,authority_revision}=>Some((reservation_id.as_str(),*authority_revision)),
    }}
}
enum CacheCleanupCommand {
    Begin(oneshot::Sender<Result<Vec<u8>>>),
    Finish(Option<oneshot::Sender<Result<Option<VerifiedCacheAccounting>>>>),
}
pub struct CacheGenerationGuard {
    commands:Option<mpsc::UnboundedSender<CacheCleanupCommand>>,
}

fn hex_digest(v:&Value)->Result<[u8;32]> {
    let s=v.as_str().ok_or(Error::Evidence("native digest type"))?;
    if s.len()!=64 || !s.bytes().all(|b|b.is_ascii_digit()||(b'a'..=b'f').contains(&b)) {return Err(Error::Evidence("native digest format"));}
    let mut bytes=[0;32];
    for (i,pair) in s.as_bytes().chunks_exact(2).enumerate() {
        bytes[i]=u8::from_str_radix(std::str::from_utf8(pair).unwrap(),16).map_err(|_|Error::Evidence("native digest format"))?;
    }
    Ok(bytes)
}
fn tokens_digest(tokens:&Value,n:u64)->Result<String> {
    let ids=tokens.as_array().ok_or(Error::Evidence("native token IDs missing"))?;
    let end=usize::try_from(n).map_err(|_|Error::Evidence("native boundary overflow"))?;
    if end>ids.len() {return Err(Error::Evidence("native boundary exceeds count"));}
    cache_auth::digest(&Value::Array(ids[..end].to_vec())).map_err(Error::Store)
}
fn native_id(body:&Value)->Result<String> {
    let ids=[body.get("rid"),body.get("request_id")].into_iter().flatten().collect::<Vec<_>>();
    if ids.len()!=1 {return Err(Error::Evidence("exactly one server native ID required"));}
    ids[0].as_str().filter(|s|!s.is_empty()).map(str::to_owned).ok_or(Error::Evidence("server native ID type"))
}
fn endpoint_points(reply:&Value,request:&CacheRequest,worker:&QualifiedWorker,tenant:&str)->Result<BTreeMap<String,RenderPoint>> {
    let provenance=&reply["prompt_cache_provenance"];
    let tokens=&reply["tokens"];
    let count=reply["count"].as_u64().ok_or(Error::Evidence("native count type"))?;
    let ids=tokens.as_array().ok_or(Error::Evidence("native token IDs missing"))?;
    if count!=ids.len() as u64 || count>4_194_304 || ids.iter().any(|i|i.as_u64().filter(|n|*n<=u32::MAX as u64).is_none())
        || reply["max_model_len"].as_u64().filter(|n|*n>=count).is_none()
        || provenance["num_prompt_tokens"]!=count || provenance["token_ids_digest"]!=tokens_digest(tokens,count)? {
        return Err(Error::Evidence("native exact token count mismatch"));
    }
    for (key,wanted) in [("native_origin_uuid",&worker.native_origin_uuid),("runtime_profile_digest",&worker.runtime_profile_digest),("namespace_digest",&worker.namespace_digest)] {
        if provenance[key]!=wanted.as_str() {return Err(Error::Evidence("native renderer startup binding mismatch"));}
    }
    hex_digest(&provenance["full_prompt_digest"])?;
    let expected:BTreeSet<_>=request.writes.keys().chain(request.reads.iter()).cloned().collect();
    let endpoints=provenance["endpoints"].as_array().ok_or(Error::Evidence("native renderer endpoints missing"))?;
    if request.writes.len()>4 || request.reads.len()>80 || expected.len()!=request.writes.len()+request.reads.len() {
        return Err(Error::Evidence("native write/read endpoint scope invalid"));
    }
    let mut points=BTreeMap::new();
    for endpoint in endpoints {
        let id=endpoint["endpoint_id"].as_str().ok_or(Error::Evidence("native endpoint ID type"))?;
        if !expected.contains(id) || points.contains_key(id) {return Err(Error::Evidence("unexpected/duplicate native endpoint"));}
        let mark=endpoint["native_mark_tokens"].as_u64().ok_or(Error::Evidence("native mark type"))?;
        let page=endpoint["native_page_tokens"].as_u64().ok_or(Error::Evidence("native page type"))?;
        let end_byte=endpoint["native_end_byte"].as_u64().ok_or(Error::Evidence("native raw byte boundary missing"))?;
        if page>mark || mark>count || endpoint["native_mark_prefix_digest"]!=tokens_digest(tokens,mark)?
            || endpoint["native_page_prefix_digest"]!=tokens_digest(tokens,page)? {return Err(Error::Evidence("native token-prefix provenance mismatch"));}
        let prefix=Prefix{tenant:tenant.into(),namespace:worker.namespace_digest.clone(),runtime:worker.registry_runtime(),
            digest:hex_digest(&endpoint["native_page_prefix_digest"])?,marked_endpoint_digest:hex_digest(&endpoint["native_mark_prefix_digest"])?,
            marked_render_digest:hex_digest(&endpoint["native_mark_render_prefix_digest"])?,marked_tokens:mark,tokens:page};
        points.insert(id.into(),RenderPoint{endpoint_id:id.into(),native_end_byte:end_byte,native_mark_tokens:mark,native_page_tokens:page,prefix,
            ttl:request.writes.get(id).copied().unwrap_or(Ttl::FiveMinutes)});
    }
    if points.keys().cloned().collect::<BTreeSet<_>>()!=expected {return Err(Error::Evidence("native renderer omitted marked/lookback endpoint"));}
    Ok(points)
}

impl NativeCacheFrontend {
    /// Existing reqwest pool and etcd client are reused; there is no listener.
    pub fn new(client:reqwest::Client,store:EtcdStore,bindings:Vec<FrontendWorkerBinding>)->Result<Self> {
        let mut workers=BTreeMap::new();
        for mut binding in bindings {
            if binding.worker.rank_tier_clocks.is_empty() || binding.worker.required_tiers.is_empty()
                || !matches!(binding.count_path.as_str(),"/v1/tokenize"|"/tokenize")
                || binding.tenant.is_empty() || binding.default_cache_salt.is_empty() {
                return Err(Error::Evidence("controller-qualified frontend binding missing"));
            }
            let url=reqwest::Url::parse(&binding.worker.worker_url).map_err(|_|Error::Evidence("native URL invalid"))?;
            if !matches!(url.scheme(),"http"|"https") || url.host_str().is_none() || !url.username().is_empty() || url.password().is_some()
                || url.path()!="/" || url.query().is_some() || url.fragment().is_some() {return Err(Error::Evidence("native origin must be fixed HTTP origin"));}
            binding.worker.worker_url=binding.worker.worker_url.trim_end_matches('/').to_owned();
            if workers.insert(binding.worker.worker_url.clone(),binding).is_some() {return Err(Error::Evidence("duplicate native qualification"));}
        }
        Ok(Self{client,store,workers,cleanup_registry:None,#[cfg(test)] cpu_clock:false})
    }
    pub fn with_cleanup_registry(mut self,registry:Arc<NativeCleanupRegistry>)->Self {
        self.cleanup_registry=Some(registry);self
    }
    pub fn cleanup_registry(&self)->Option<&Arc<NativeCleanupRegistry>> {self.cleanup_registry.as_ref()}
    #[cfg(test)]
    pub(crate) fn use_cpu_fixture_clock(mut self)->Self {self.cpu_clock=true;self}
    async fn clock(&self)->Result<super::frontend_clock::ClockBounds> {
        #[cfg(test)] if self.cpu_clock {return Ok(super::frontend_clock::ClockBounds::cpu_fixture(Instant::now()));}
        ChronyClock.qualify().await.map_err(|e|Error::Store(e.to_string()))
    }
    /// Product discovery reflects the actual loaded worker bindings.
    pub(crate) fn has_worker(&self,url:&str)->bool {
        self.workers.contains_key(url.trim_end_matches('/'))
    }
    fn binding(&self,url:&str)->Result<&FrontendWorkerBinding> {self.workers.get(url.trim_end_matches('/')).ok_or(Error::Evidence("native cache worker metadata unavailable"))}
    async fn context(&self,binding:&FrontendWorkerBinding,origin:&RequestOrigin,full:&str,tokens:&str)->Result<SigningContext> {
        let clock=self.clock().await?;
        let id=uuid::Uuid::new_v4().simple().to_string();
        let _=origin;
        Ok(SigningContext{model:binding.model.clone(),source_digest:binding.source_digest.clone(),full_render_digest:full.into(),token_ids_digest:tokens.into(),
            operation_id:id,nonce:uuid::Uuid::new_v4().simple().to_string(),now_lower_ms:clock.lower_ms(),now_upper_ms:clock.upper_ms(),clock_observed_at:clock.observed_at()})
    }
    async fn post(&self,url:&str,path:&str,body:&Value,deadline:Instant)->Result<Value> {
        let (status,reply)=self.post_response(url,path,body,deadline).await?;
        if !(200..300).contains(&status) {return Err(cache_http_error(status,&reply));}
        Ok(reply)
    }
    async fn post_response(&self,url:&str,path:&str,body:&Value,deadline:Instant)->Result<(u16,Value)> {
        let http=self.client.post(format!("{}{path}",url.trim_end_matches('/'))).json(body);
        let mut response=http.timeout(control_remaining(deadline)?).send().await.map_err(|e|Error::Store(e.to_string()))?;
        let status=response.status().as_u16();
        let mut bytes=Vec::new();
        while let Some(chunk)=response.chunk().await.map_err(|e|Error::Store(e.to_string()))? {
            if bytes.len().saturating_add(chunk.len())>64*1024*1024 {return Err(Error::Evidence("native cache reply exceeds bound"));}
            bytes.extend_from_slice(&chunk);
        }
        let reply=serde_json::from_slice(&bytes).map_err(|_|Error::Evidence("native cache reply is not JSON"))?;
        if !(200..300).contains(&status) {observe_cache_http_error(path,status,&reply);}
        Ok((status,reply))
    }
    pub async fn render_count(&self,url:&str,body:&Value,request:&CacheRequest,deadline:Instant)->Result<RenderedCount> {
        let deadline=deadline.min(request.control_deadline()?);
        control_remaining(deadline)?;
        let binding=self.binding(url)?;
        let clock=self.clock().await?;
        let started=Instant::from_std(request.http_started_at);
        let elapsed=clock.observed_at().checked_duration_since(started).ok_or(Error::Evidence("typed HTTP origin reversed"))?;
        let observed_lower=clock.lower_ms().checked_sub(u64::try_from(elapsed.as_nanos().div_ceil(1_000_000)).map_err(|_|Error::Evidence("origin elapsed overflow"))?).ok_or(Error::Evidence("origin clock underflow"))?;
        // Counter selection and selected-worker recount share exactly the
        // first qualified origin; a later clock sample cannot replace it.
        let start_lower=*request.start_lower_ms.get_or_init(||observed_lower);
        let origin=RequestOrigin::from_frontend_context(request.http_request_id.clone(),native_id(body)?,start_lower,started)?;
        let mut body=body.clone();
        if body["model"]!=binding.model {return Err(Error::Evidence("served model differs from qualified model"));}
        if body.get(cache_auth::PROOF_FIELD).is_some() || body.get("prompt_cache_provenance").is_some() {return Err(Error::Evidence("private cache proof/provenance prepopulated"));}
        let raw=body.get("cache_salt").and_then(Value::as_str).unwrap_or(&binding.default_cache_salt).to_owned();
        body["cache_salt"]=raw.clone().into();
        let mut worker=binding.worker.clone();worker.namespace_digest=cache_auth::namespace_for(&raw,None,&binding.kv_compat_digest).map_err(Error::Store)?;
        body["prompt_cache_namespace_digest"]=worker.namespace_digest.clone().into();
        let mut count_body=body.clone();
        // A count carries renderer selectors, not generation-only sampling knobs.
        for key in ["max_tokens","max_completion_tokens","max_output_tokens","stream","stream_options"] {count_body.as_object_mut().ok_or(Error::Evidence("native count body type"))?.remove(key);}
        cache_auth::attach_canonical_body(&mut count_body).map_err(Error::Store)?;
        let payload=cache_auth::payload_for("render",&count_body).map_err(Error::Store)?;
        let prepared=cache_auth::PrehashedPayload::new(&payload).map_err(Error::Store)?;
        let c=self.context(binding,&origin,"","").await?;
        let claims=json!({"version":1,"operation":"render","operation_id":c.operation_id,"nonce":c.nonce,"model":c.model,
            "native_origin":url,"native_incarnation":worker.native_origin_uuid,"source_digest":c.source_digest,"runtime_profile_digest":worker.runtime_profile_digest,
            "reservation_id":"","authority_revision":0,"native_request_id":native_id(&body)?,"namespace_digest":worker.namespace_digest,
            "full_render_digest":"","token_ids_digest":"","request_started_at_ms":start_lower,"issued_at_ms":c.now_upper_ms,
            "deadline_ms":(c.now_lower_ms+1000).min(start_lower.saturating_add(10_830_000)),"retain_for_ms":0,"payload_digest":prepared.digest()});
        c.ensure_fresh()?;
        let proof=cache_auth::sign_prehashed(&binding.secret,&claims,&prepared).map_err(Error::Store)?;
        let request_claims=proof["claims"].clone();count_body[cache_auth::PROOF_FIELD]=proof;
        // Count is read-only. Serialize before measuring the remaining shared
        // budget, then refuse a new native send at the exact monotonic deadline.
        let http=self.client.post(format!("{}{}",url.trim_end_matches('/'),binding.count_path)).json(&count_body);
        let remaining=deadline.checked_duration_since(Instant::now()).filter(|left|!left.is_zero())
            .ok_or_else(||Error::Store("native token count deadline expired before send".into()))?;
        let mut response=http.timeout(remaining).send().await.map_err(|e|Error::Store(e.to_string()))?;
        let status=response.status().as_u16();
        let mut bytes=Vec::new();
        while let Some(chunk)=response.chunk().await.map_err(|e|Error::Store(e.to_string()))? {
            if bytes.len().saturating_add(chunk.len())>64*1024*1024 {return Err(Error::Evidence("native cache reply exceeds bound"));}
            bytes.extend_from_slice(&chunk);
        }
        let reply:Value=serde_json::from_slice(&bytes).map_err(|_|Error::Evidence("native cache reply is not JSON"))?;
        if !(200..300).contains(&status) {
            observe_cache_http_error(&binding.count_path,status,&reply);
            return Err(cache_http_error(status,&reply));
        }
        verify_count_reply(&binding.secret,&request_claims,&reply)?;
        let points=endpoint_points(&reply,request,&worker,&binding.tenant)?;
        body["prompt_cache_provenance"]=reply["prompt_cache_provenance"].clone();
        Ok(RenderedCount{prompt_tokens:reply["count"].as_u64().unwrap(),worker_url:url.into(),body,tokens:reply["tokens"].clone(),provenance:reply["prompt_cache_provenance"].clone(),origin,worker,points})
    }
    pub async fn prepare_generation(self:&Arc<Self>,url:&str,body:Value,request:&CacheRequest,rendered:RenderedCount,epoch:Option<u64>,realm_deadline:Instant)->Result<PreparedCacheGeneration> {
        let realm_deadline=realm_deadline.min(rendered.origin.operation_deadline()?);
        let deadline=request.control_deadline()?.min(realm_deadline);
        control_remaining(deadline)?;
        let registry=self.cleanup_registry.as_ref().ok_or(Error::Evidence("native cache cleanup registry unavailable"))?;
        if url!=rendered.worker_url || native_id(&body)?!=native_id(&rendered.body)? {return Err(Error::Evidence("rendered count transport/body ID mismatch"));}
        let mut canonical=body;
        canonical["cache_salt"]=rendered.body["cache_salt"].clone();
        canonical["prompt_cache_namespace_digest"]=rendered.worker.namespace_digest.clone().into();
        if cache_auth::ordinary_body_for(&canonical).map_err(Error::Store)?!=cache_auth::ordinary_body_for(&rendered.body).map_err(Error::Store)? {return Err(Error::Evidence("generation ordinary body changed after count"));}
        canonical["prompt_cache_provenance"]=rendered.provenance.clone();
        let native_request_id=native_id(&canonical)?.to_owned();
        let write_ids=request.writes.keys().cloned().collect::<BTreeSet<_>>();let read_ids=request.reads.clone();
        let full=rendered.provenance["full_prompt_digest"].as_str().unwrap().to_owned();
        let tokens=rendered.provenance["token_ids_digest"].as_str().unwrap().to_owned();
        let prompt_tokens=rendered.prompt_tokens;
        let (commands,receiver)=mpsc::unbounded_channel();let (ready,prepared)=oneshot::channel();
        let frontend=self.clone();let url=url.to_owned();
        // The task owns preparation before its FIRST authority CAS. Neither a
        // dropped caller nor shutdown can lose a just-committed Dispatch.
        registry.register(format!("cache:{native_request_id}"),async move {
            frontend.cleanup_lifecycle(url,canonical,rendered,write_ids,read_ids,epoch,full,tokens,prompt_tokens,native_request_id,ready,receiver,deadline,realm_deadline).await
        }).map_err(|e|Error::Store(e.to_string()))?;
        let guard=CacheGenerationGuard{commands:Some(commands)};
        let body=prepared.await.map_err(|_|Error::Evidence("tracked cache preparation result unavailable"))??;
        Ok(PreparedCacheGeneration{body,guard})
    }
    async fn prepare_owned(&self,url:&str,mut canonical:Value,rendered:RenderedCount,write_ids:&BTreeSet<String>,read_ids:&BTreeSet<String>,epoch:Option<u64>,state:&mut Option<Dispatch>,deadline:Instant,realm_deadline:Instant)->Result<Value> {
        control_remaining(deadline)?;
        let binding=self.binding(url)?;
        let writes=write_ids.iter().map(|id|rendered.points.get(id).cloned().ok_or(Error::Evidence("write point absent"))).collect::<Result<Vec<_>>>()?;
        let reads=read_ids.iter().map(|id|rendered.points.get(id).cloned().ok_or(Error::Evidence("read point absent"))).collect::<Result<Vec<_>>>()?;
        let now=self.clock().await?;
        let mut selected=lookup_best_read(&self.store,&rendered.origin,&rendered.worker,&reads,now.upper_ms()).await?;
        if selected.is_none() {if let Some(qualification)=&binding.shared_backend {
            selected=lookup_shared_read(&self.store,&rendered.origin,&rendered.worker,&reads,qualification,&now).await?;
        }}
        control_remaining(deadline)?;
        *state=Some(prepare_dispatch(&self.store,rendered.origin,rendered.worker,writes,selected,epoch).await?);
        let full=rendered.provenance["full_prompt_digest"].as_str().unwrap().to_owned();
        let tokens=rendered.provenance["token_ids_digest"].as_str().unwrap().to_owned();
        let dispatch=state.as_mut().ok_or(Error::Evidence("cache dispatch absent"))?;
        if let Err(error)=self.reserve(url,dispatch,&canonical,&rendered.tokens,&full,&tokens,deadline).await {
            if matches!(error,Error::Evidence("authenticated native read source unavailable")) {
                let context=self.context(binding,&dispatch.origin(),&full,&tokens).await?;
                let cleanup=cleanup_deadline(dispatch,realm_deadline)?;
                let cancel=signed_control_query(&self.store,dispatch,"cancel",&binding.secret,&context,cleanup,|_|self.clock()).await?;
                let packet=self.post(url,"/v1/cache/reservations/cancel",&cancel,cleanup).await?;
                control_remaining(deadline)?;
                *dispatch=reprepare_after_cancel(&self.store,dispatch,url,&packet,&binding.secret).await?;
                if let Err(error)=self.reserve(url,dispatch,&canonical,&rendered.tokens,&full,&tokens,deadline).await {
                    return Err(error);
                }
            } else {
                return Err(error);
            }
        }
        let context=self.context(binding,&dispatch.origin(),&full,&tokens).await?;
        let signed=signed_generation_body(&self.store,dispatch,&mut canonical,&binding.secret,&context,|_|self.clock()).await;
        if matches!(&signed,Err(Error::Evidence("selected read expired after native reserve: authenticated native cancel required before reprepare"))) {
            let context=self.context(binding,&dispatch.origin(),&full,&tokens).await?;
            let cleanup=cleanup_deadline(dispatch,realm_deadline)?;
            let cancel=signed_control_query(&self.store,dispatch,"cancel",&binding.secret,&context,cleanup,|_|self.clock()).await?;
            let packet=self.post(url,"/v1/cache/reservations/cancel",&cancel,cleanup).await?;
            control_remaining(deadline)?;
            *dispatch=reprepare_after_cancel(&self.store,dispatch,url,&packet,&binding.secret).await?;
            if let Err(error)=self.reserve(url,dispatch,&canonical,&rendered.tokens,&full,&tokens,deadline).await {
                return Err(error);
            }
            let context=self.context(binding,&dispatch.origin(),&full,&tokens).await?;
            if let Err(error)=signed_generation_body(&self.store,dispatch,&mut canonical,&binding.secret,&context,|_|self.clock()).await {
                return Err(error);
            }
        } else if let Err(error)=signed {
            // No generation has been sent. Release native reserved capacity with
            // an authenticated all-rank cancellation; do not retry generation.
            return Err(error);
        }
        control_remaining(deadline)?;
        Ok(canonical)
    }
    async fn reserve(&self,url:&str,d:&mut Dispatch,body:&Value,tokens:&Value,full:&str,digest:&str,deadline:Instant)->Result<()> {
        control_remaining(deadline)?;
        let binding=self.binding(url)?;let c=self.context(binding,&d.origin(),full,digest).await?;
        let request=signed_reservation_body(&self.store,d,body,tokens,&binding.secret,&c,|_|self.clock()).await?;
        let (status,ack)=self.post_response(url,"/v1/cache/reservations",&request,deadline).await?;
        if status==409 && ack["error"]["type"]=="cache_read_source_unavailable" && ack.get(cache_auth::PROOF_FIELD).is_some() {
            verify_read_source_unavailable(d,url,&ack,&binding.secret)?;
            return Err(Error::Evidence("authenticated native read source unavailable"));
        }
        if !(200..300).contains(&status) {return Err(cache_http_error(status,&ack));}
        accept_reservation_ack(&self.store,d,url,&ack,&binding.secret).await
    }
    async fn cancel(&self,url:&str,d:&mut Dispatch,full:&str,tokens:&str,deadline:Instant)->Result<VerifiedCacheCleanup> {
        let binding=self.binding(url)?;let c=self.context(binding,&d.origin(),full,tokens).await?;
        let request=signed_control_query(&self.store,d,"cancel",&binding.secret,&c,deadline,|_|self.clock()).await?;
        let packet=self.post(url,"/v1/cache/reservations/cancel",&request,deadline).await?;
        cancel_dispatch(&self.store,d,url,&packet,&binding.secret).await?;
        Ok(VerifiedCacheCleanup{native_request_id:d.native_request_id().into(),evidence:CacheCleanupEvidence::NativeCancellation{reservation_id:packet["reservation_id"].as_str().ok_or(Error::Evidence("verified cancellation reservation ID absent"))?.into(),authority_revision:packet["authority_revision"].as_i64().ok_or(Error::Evidence("verified cancellation revision absent"))?}})
    }
    async fn query_accounting(&self,url:&str,d:&mut Dispatch,full:&str,tokens:&str,prompt_tokens:u64,deadline:Instant)->Result<Option<VerifiedCacheAccounting>> {
        let binding=self.binding(url)?;let c=self.context(binding,&d.origin(),full,tokens).await?;
        let request=signed_control_query(&self.store,d,"receipts",&binding.secret,&c,deadline,|_|self.clock()).await?;
        let packet=self.post(url,"/v1/cache/reservations/receipts",&request,deadline).await?;
        verify_receipt_query_response(d,&packet,&binding.secret).map_err(|error|observe_receipt_error("query_envelope",d.native_request_id(),error))?;
        if packet["native_request_id"]!=d.native_request_id() {return Err(Error::Evidence("receipt query native ID mismatch"));}
        let events=packet["receipts"].as_array().ok_or(Error::Evidence("native receipt query omitted events"))?;
        if events.len()>5 {return Err(Error::Evidence("native receipt event scope exceeded"));}
        let scopes=accounting_scopes(d);
        let mut accounting=CacheAccounting::new(prompt_tokens,scopes.is_empty(),scopes).map_err(Error::Evidence)?;
        for event in events {
            let now=self.clock().await?;
            if let Some(coverage)=observe_native_packet_accounting(&self.store,d,url,event,&binding.secret,&now,binding.shared_backend.as_ref()).await.map_err(|error|observe_receipt_error("native_event",d.native_request_id(),error))? {
                accounting.observe(coverage).map_err(|reason|observe_receipt_error("accounting",d.native_request_id(),Error::Evidence(reason)))?;
            }
        }
        let report=accounting.finish();
        if let Some(report)=&report {report.validate(prompt_tokens)?;}
        Ok(report)
    }
}

fn verify_count_reply(secret:&[u8],request:&Value,reply:&Value)->Result<()> {
    let mut payload=reply.clone();payload.as_object_mut().ok_or(Error::Evidence("flat count reply required"))?.remove(cache_auth::PROOF_FIELD);
    let response=cache_auth::verify_signature(secret,&reply[cache_auth::PROOF_FIELD],&payload).map_err(Error::Store)?;
    for key in ["version","operation","model","native_origin","native_incarnation","source_digest","runtime_profile_digest","reservation_id","authority_revision","native_request_id","namespace_digest","request_started_at_ms","issued_at_ms","deadline_ms","retain_for_ms"] {
        if response[key]!=request[key] {return Err(Error::Evidence("count response signed origin/request binding mismatch"));}
    }
    let digest=cache_auth::digest(&payload).map_err(Error::Store)?;
    for key in ["operation_id","nonce"] {if response[key]!=cache_auth::digest(&json!(["native-response",request[key],digest])).map_err(Error::Store)? {return Err(Error::Evidence("count response replay binding mismatch"));}}
    if response["full_render_digest"]!=reply["prompt_cache_provenance"]["full_prompt_digest"] || response["token_ids_digest"]!=reply["prompt_cache_provenance"]["token_ids_digest"] {return Err(Error::Evidence("count signed full-render mismatch"));}
    // The native accepted the request before its operation deadline. A slow
    // full-render response does not change the original logical cache origin.
    Ok(())
}

impl NativeCacheFrontend {
    #[allow(clippy::too_many_arguments)]
    async fn cleanup_lifecycle(self:Arc<Self>,url:String,body:Value,rendered:RenderedCount,write_ids:BTreeSet<String>,read_ids:BTreeSet<String>,epoch:Option<u64>,full:String,tokens:String,prompt_tokens:u64,native_request_id:String,ready:oneshot::Sender<Result<Value>>,mut commands:mpsc::UnboundedReceiver<CacheCleanupCommand>,deadline:Instant,realm_deadline:Instant)->std::result::Result<CleanupReceipt,CleanupFailure> {
        let mut dispatch=None;
        let preparation=self.prepare_owned(&url,body,rendered,&write_ids,&read_ids,epoch,&mut dispatch,deadline,realm_deadline).await;
        let generation_body=match preparation {
            Err(error)=>{
                let _=ready.send(Err(error));
                return if let Some(d)=dispatch.as_mut() {
                    let cleanup=cleanup_deadline(d,realm_deadline).map_err(|e|CleanupFailure::Verification(e.to_string()))?;
                    self.cancel(&url,d,&full,&tokens,cleanup).await.map_err(|e|CleanupFailure::Verification(e.to_string())).and_then(CleanupReceipt::from_cache)
                }else{
                    // Native Reserve/Gen require a Dispatch; neither can have
                    // been sent. Authority leases alone are not physical holds.
                    CleanupReceipt::from_cache(VerifiedCacheCleanup{native_request_id,evidence:CacheCleanupEvidence::NoNativeReservation})
                };
            }
            Ok(body)=>{let _=ready.send(Ok(body.clone()));body}
        };
        let d=dispatch.as_mut().ok_or_else(||CleanupFailure::Verification("prepared cache dispatch missing".into()))?;
        let mut generation_claimed=false;
        // Retain the large snapshot through the lifecycle; do not recursively
        // destroy it between the last HMAC and the caller's immediate POST.
        let mut generation_payload=None;
        let reply=loop {
            match commands.recv().await {
                Some(CacheCleanupCommand::Begin(reply))=>{
                    let result=match control_remaining(deadline) {
                        Ok(_)=>async {
                            let binding=self.binding(&url)?;
                            let context=self.context(binding,&d.origin(),&full,&tokens).await?;
                            let (bytes,payload)=signed_generation_post(&self.store,d,&generation_body,&binding.secret,&context,deadline,||self.clock()).await?;
                            generation_payload=Some(payload);
                            Ok(bytes)
                        }.await,
                        Err(error)=>Err(error),
                    };
                    if result.is_ok() {generation_claimed=true;}
                    let _=reply.send(result);
                }
                Some(CacheCleanupCommand::Finish(reply))=>break reply,
                None=>break None,
            }
        };
        let cleanup=cleanup_deadline(d,realm_deadline).map_err(|e|CleanupFailure::Verification(e.to_string()))?;
        let _generation_payload=generation_payload;
        let accounting=if generation_claimed {self.query_accounting(&url,d,&full,&tokens,prompt_tokens,cleanup).await}else{Ok(None)};
        // Accounting/query errors never bypass native cancellation.
        match self.cancel(&url,d,&full,&tokens,cleanup).await {
            Ok(proof)=>{
                let receipt=CleanupReceipt::from_cache(proof)?;
                if let Some(reply)=reply {let _=reply.send(accounting);}
                Ok(receipt)
            }
            Err(error)=>{
                let reason=error.to_string();
                if let Some(reply)=reply {let _=reply.send(Err(error));}
                Err(CleanupFailure::Verification(reason))
            }
        }
    }
}
impl CacheGenerationGuard {
    pub async fn begin_generation(&self)->Result<Vec<u8>> {
        let commands=self.commands.as_ref().ok_or(Error::Evidence("cache dispatch already finalized"))?;
        let (reply,result)=oneshot::channel();
        commands.send(CacheCleanupCommand::Begin(reply)).map_err(|_|Error::Evidence("tracked cache task unavailable"))?;
        result.await.map_err(|_|Error::Evidence("tracked cache generation claim unavailable"))?
    }
    pub async fn query_finish(&mut self,_cancelled:bool)->Result<Option<VerifiedCacheAccounting>> {
        let Some(commands)=self.commands.take() else {return Ok(None)};
        let (reply,result)=oneshot::channel();
        // This synchronous command survives dropping the caller's first await.
        commands.send(CacheCleanupCommand::Finish(Some(reply))).map_err(|_|Error::Evidence("tracked cache task unavailable"))?;
        result.await.map_err(|_|Error::Evidence("tracked cache cleanup result unavailable"))?
    }
}
impl Drop for CacheGenerationGuard {
    fn drop(&mut self) {
        // Closing the channel triggers the already registered task even without
        // a runtime on the dropping thread. A failed task remains Blocked.
        self.commands.take();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test(start_paused=true)]
    async fn cache_control_wait_spends_original_budget_and_shared_cleanup_deadline() {
        let start=Instant::now().into_std();
        let request=CacheRequest::new("CPU-original-budget".into(),start,BTreeMap::new(),BTreeSet::new());
        let deadline=request.control_deadline().unwrap();
        assert_eq!(deadline,Instant::from_std(start)+Duration::from_secs(600));
        // The actual existing deadline helper permits a normal 16s allocation
        // wait; a subsequent recount/reserve only sees what remains.
        tokio::time::sleep(Duration::from_secs(16)).await;
        assert_eq!(request.control_deadline().unwrap(),deadline);
        assert_eq!(control_remaining(deadline).unwrap(),Duration::from_secs(584));
        let realm=Instant::now()+Duration::from_secs(4);
        let bounded=deadline.min(realm);
        tokio::time::sleep(Duration::from_secs(3)).await;
        assert_eq!(control_remaining(bounded).unwrap(),Duration::from_secs(1));
        tokio::time::sleep(Duration::from_secs(1)).await;
        assert!(matches!(control_remaining(bounded),Err(Error::Evidence("native cache control deadline expired"))));
        // Query and cancel use this one deadline, never a new 30s per stage.
        let cleanup=Instant::now()+Duration::from_secs(30);
        tokio::time::sleep(Duration::from_secs(29)).await;
        assert_eq!(control_remaining(cleanup).unwrap(),Duration::from_secs(1));
        tokio::time::sleep(Duration::from_secs(1)).await;
        assert!(control_remaining(cleanup).is_err());
        tokio::time::advance(Duration::from_secs(600)).await;
        assert_eq!(request.control_deadline().unwrap(),deadline);
        assert!(control_remaining(deadline).is_err());
    }
    #[test]
    fn e55_cached_pd_requires_measured_server_render_plan() {
        let fixture:Value=serde_json::from_str(include_str!("../vllm/count-namespace-wire-fixture.json")).unwrap();
        let reply=&fixture["actual_count_wire"];let provenance=&reply["prompt_cache_provenance"];
        let key:Vec<u8>=(0..32).collect();
        verify_count_reply(&key,&fixture["canonical_native_request"][cache_auth::PROOF_FIELD]["claims"],reply).unwrap();
        let worker=QualifiedWorker{worker_url:"http://CPU-worker".into(),native_origin_uuid:provenance["native_origin_uuid"].as_str().unwrap().into(),runtime_profile_digest:provenance["runtime_profile_digest"].as_str().unwrap().into(),namespace_digest:provenance["namespace_digest"].as_str().unwrap().into(),native_clock_domain:"CPU".into(),required_tiers:BTreeSet::from(["gpu".into()]),native_id_field:NativeIdField::RequestId,rank_tier_clocks:BTreeMap::new()};
        let request=CacheRequest::new("CPU-NOT-QUALIFIED".into(),std::time::Instant::now(),BTreeMap::new(),provenance["endpoints"].as_array().unwrap().iter().map(|p|p["endpoint_id"].as_str().unwrap().to_string()).collect());
        let mut body=fixture["canonical_native_request"].clone();
        // The engine sends the measured selectors to its planner, not the
        // Count HTTP proof or the Count-only namespace/canonical projection.
        for field in [cache_auth::PROOF_FIELD,"prompt_cache_namespace_digest","prompt_cache_canonical_body"] {body.as_object_mut().unwrap().remove(field);}
        let rendered=RenderedCount{prompt_tokens:reply["count"].as_u64().unwrap(),worker_url:worker.worker_url.clone(),body:body.clone(),tokens:reply["tokens"].clone(),provenance:provenance.clone(),origin:RequestOrigin::from_frontend_context("CPU".into(),native_id(&body).unwrap(),1_000_000,Instant::now()).unwrap(),points:endpoint_points(reply,&request,&worker,"CPU").unwrap(),worker};
        rendered.validate_server_render_plan(&body).unwrap();
        crate::native_pd::dispatch::validate_pd_plan_body(&body,Some(&rendered)).unwrap();
        assert!(crate::native_pd::dispatch::validate_pd_plan_body(&body,None).is_err());
        for field in [cache_auth::PROOF_FIELD,"bootstrap_room","reservation_id"] {
            let mut injected=body.clone();injected[field]=json!("not-a-server-capability");
            assert!(crate::native_pd::dispatch::validate_pd_plan_body(&injected,Some(&rendered)).is_err(),"{field}");
        }
        for field in ["prompt_cache_blocks","prompt_cache_read_candidates","prompt_cache_message_order","request_id","cache_salt"] {
            let mut missing=body.clone();missing.as_object_mut().unwrap().remove(field);
            assert!(rendered.validate_server_render_plan(&missing).is_err(),"{field}");
            let mut changed=body.clone();changed[field]=json!("not-the-measured-value");
            assert!(rendered.validate_server_render_plan(&changed).is_err(),"{field}");
        }
        let mut n=body.clone();n["n"]=json!(2);assert!(rendered.validate_server_render_plan(&n).is_err());
        let mut duplicate=body;duplicate["rid"]=json!("foreign-root");assert!(rendered.validate_server_render_plan(&duplicate).is_err());
    }
    #[test]
    fn readonly_count_accepts_actual_native_endpoint_witnesses() {
        // Reuse an immutable actual renderer response; only the trusted caller's
        // role set is synthetic. No response fields or signature are rewritten.
        let fixture:Value=serde_json::from_str(include_str!("../vllm/count-namespace-wire-fixture.json")).unwrap();
        let reply=&fixture["actual_count_wire"];let p=&reply["prompt_cache_provenance"];
        let worker=QualifiedWorker{worker_url:"http://qualified-worker:8000".into(),native_origin_uuid:p["native_origin_uuid"].as_str().unwrap().into(),runtime_profile_digest:p["runtime_profile_digest"].as_str().unwrap().into(),namespace_digest:p["namespace_digest"].as_str().unwrap().into(),native_clock_domain:"CPU".into(),required_tiers:BTreeSet::from(["gpu".into()]),native_id_field:NativeIdField::RequestId,rank_tier_clocks:BTreeMap::new()};
        let reads=p["endpoints"].as_array().unwrap().iter().map(|p|p["endpoint_id"].as_str().unwrap().to_string()).collect();
        let request=CacheRequest::new("CPU-readonly-count".into(),std::time::Instant::now(),BTreeMap::new(),reads);
        assert_eq!(endpoint_points(reply,&request,&worker,"CPU").unwrap().len(),3);
        let missing=CacheRequest::new("CPU-omitted-reader".into(),std::time::Instant::now(),BTreeMap::new(),BTreeSet::from(["unrendered-read".into()]));
        assert!(endpoint_points(reply,&missing,&worker,"CPU").is_err());
    }
    #[test]
    fn final_native_count_and_qwen_control_wire_packets() {
        // These are existing producer CPU artifacts, not live qualification.
        let qwen:Value=serde_json::from_str(include_str!("../qwen/signed-count-wire-cpu-fixture.json")).unwrap();
        let mut q_request=qwen[cache_auth::PROOF_FIELD]["claims"].clone();
        q_request["operation_id"]="cpu-count-wire".into();q_request["nonce"]="cpu-count-wire-nonce".into();
        q_request["full_render_digest"]="".into();q_request["token_ids_digest"]="".into();
        verify_count_reply(&[b'C';32],&q_request,&qwen).unwrap();
        let gf:Value=serde_json::from_str(include_str!("../vllm/count-namespace-wire-fixture.json")).unwrap();
        let key:Vec<u8>=(0..32).collect();
        verify_count_reply(&key,&gf["canonical_native_request"][cache_auth::PROOF_FIELD]["claims"],&gf["actual_count_wire"]).unwrap();
        for (reply,writes,reads) in [(&qwen,vec!["CPU-logical-hel","CPU-logical-hell"],vec![]),
            (&gf["actual_count_wire"],vec!["write:authority-fixture-1"],vec!["read:authority-fixture-1","read:authority-fixture-2"])] {
            let p=&reply["prompt_cache_provenance"];
            let worker=QualifiedWorker{worker_url:"http://cpu".into(),native_origin_uuid:p["native_origin_uuid"].as_str().unwrap().into(),
                runtime_profile_digest:p["runtime_profile_digest"].as_str().unwrap().into(),namespace_digest:p["namespace_digest"].as_str().unwrap().into(),
                native_clock_domain:"CPU".into(),required_tiers:BTreeSet::from(["CPU".into()]),native_id_field:NativeIdField::Rid,rank_tier_clocks:BTreeMap::new()};
            let request=CacheRequest::new("CPU".into(),std::time::Instant::now(),writes.into_iter().map(|id|(id.into(),Ttl::FiveMinutes)).collect(),reads.into_iter().map(str::to_owned).collect());
            endpoint_points(reply,&request,&worker,"CPU").unwrap();
            let mut wrong=reply.clone();wrong["prompt_cache_provenance"]["endpoints"][0]["native_mark_prefix_digest"]="0".repeat(64).into();
            assert!(endpoint_points(&wrong,&request,&worker,"CPU").is_err());
        }
        let control:Value=serde_json::from_str(include_str!("../qwen/control/final-wire-cpu-result.json")).unwrap();
        for rank in control["ranks"].as_array().unwrap() {
            let cancel=&rank["cancel_packet"];
            cache_auth::verify_signature(&[b'x';32],&cancel[cache_auth::PROOF_FIELD],&cache_auth::payload_for("receipts",cancel).unwrap()).unwrap();
            assert_eq!(cancel["rank_results"].as_array().unwrap().iter().map(|r|r["rank_id"].as_u64().unwrap()).collect::<BTreeSet<_>>(),BTreeSet::from([0,1]));
            for event in rank["query_packet"]["receipts"].as_array().unwrap() {
                cache_auth::verify_signature(&[b'x';32],&event[cache_auth::PROOF_FIELD],&cache_auth::payload_for("receipts",event).unwrap()).unwrap();
                assert!(matches!(event["event_kind"].as_str(),Some("write_commit"|"read_refresh")));
                assert_eq!(event["receipt"]["rank_receipts"].as_array().unwrap().len(),2);
            }
        }
    }
}

#[cfg(test)]
#[path="native_cache_e19_tests.rs"]
mod e19_tests;
