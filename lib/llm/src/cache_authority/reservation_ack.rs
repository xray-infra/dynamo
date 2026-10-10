use super::*;

#[derive(Clone,Debug,Serialize,Deserialize)]
pub(crate) struct ReservationAck {
    reservation_id:String,authority_revision:i64,native_origin_uuid:String,
    source_digest:String,runtime_profile_digest:String,namespace_digest:String,
    pub rank_acks:Vec<RankAck>,
}
#[derive(Clone,Debug,Serialize,Deserialize)]
pub(crate) struct RankAck {pub rank_id:u32,pub point_acks:Vec<PointAck>}
#[derive(Clone,Debug,Serialize,Deserialize)]
pub(crate) struct PointAck {pub endpoint_id:String,pub tier_acks:Vec<TierAck>}
#[derive(Clone,Debug,Serialize,Deserialize)]
pub(crate) struct TierAck {
    pub tier:String,pub reserved_tokens:u64,pub native_clock_domain:String,
    pub receive_monotonic_ns:u64,pub retain_until_monotonic_ns:u64,pub ack_retain_for_ms:u64,
}

fn claims(d:&Dispatch,c:&SigningContext,operation:&str)->Result<Value> {
    let age=Instant::now().checked_duration_since(c.clock_observed_at).ok_or(Error::Evidence("qualified clock reversed"))?;
    if age>Duration::from_millis(100) || c.now_upper_ms<c.now_lower_ms || c.now_upper_ms-c.now_lower_ms>200 || c.now_lower_ms<d.origin.started_at_ms {return Err(Error::Evidence("fresh bounded clock qualification required"));}
    let elapsed=u64::try_from(age.as_nanos().div_ceil(1_000_000)).map_err(|_|Error::Evidence("clock elapsed overflow"))?;
    let issued=c.now_upper_ms.checked_add(elapsed).ok_or(Error::Evidence("clock overflow"))?;
    if d.origin.elapsed_ms(Instant::now())?>=10_830_000 {return Err(Error::Evidence("private cache operation window expired"));}
    let deadline=c.now_lower_ms.checked_add(elapsed).and_then(|n|n.checked_add(1000)).ok_or(Error::Evidence("clock overflow"))?.min(d.origin.started_at_ms.saturating_add(10_830_000));
    Ok(json!({"version":1,"operation":operation,"operation_id":c.operation_id,"nonce":c.nonce,"model":c.model,
        "native_origin":d.worker.worker_url,"native_incarnation":d.worker.native_origin_uuid,"source_digest":c.source_digest,
        "runtime_profile_digest":d.worker.runtime_profile_digest,"reservation_id":d.reservation_id,"authority_revision":d.authority_revision,
        "native_request_id":d.origin.native_request_id,"namespace_digest":d.worker.namespace_digest,
        "full_render_digest":c.full_render_digest,"token_ids_digest":c.token_ids_digest,"request_started_at_ms":d.origin.started_at_ms,
        "issued_at_ms":issued,"deadline_ms":deadline,"retain_for_ms":d.retain_for_ms.values().copied().max().unwrap_or(0),"payload_digest":""}))
}

/// Uses the existing same-worker cancel or receipt-query endpoint. Querying
/// cannot create entries; each returned event still requires its own proof.
pub async fn signed_control_query<F,Fut>(store:&EtcdStore,d:&mut Dispatch,operation:&str,secret:&[u8],c:&SigningContext,deadline:Instant,fresh_clock:F)->Result<Value>
where F:FnOnce(&cache_auth::PrehashedPayload)->Fut,Fut:std::future::Future<Output=Result<ClockBounds>> {
    if !matches!(operation,"cancel"|"receipts") {return Err(Error::Evidence("unsupported cache control query"));}
    store.confirm_dispatch(d).await?;
    let mut body=json!({"native_request_id":d.origin.native_request_id,"reservation_id":d.reservation_id,"authority_revision":d.authority_revision});
    let prepared=cache_auth::PrehashedPayload::new(&body).map_err(Error::Store)?;
    let clock=fresh_clock(&prepared).await?;
    let c=c.refreshed(&clock)?;
    let proof=control_proof(d,operation,secret,&c,deadline,&prepared)?;
    d.control_claims=Some(proof["claims"].clone());body.as_object_mut().unwrap().insert(cache_auth::PROOF_FIELD.into(),proof);
    Ok(body)
}

fn control_proof(d:&Dispatch,operation:&str,secret:&[u8],c:&SigningContext,deadline:Instant,prepared:&cache_auth::PrehashedPayload)->Result<Value> {
    if !matches!(operation,"cancel"|"receipts") {return Err(Error::Evidence("unsupported cache control query"));}
    // These operations cannot admit Gen or renew a retained cache lease. They
    // use the caller's one cleanup deadline, capped by the original realm.
    let deadline=deadline.min(d.origin.operation_deadline()?).min(c.clock_observed_at.checked_add(Duration::from_secs(30)).ok_or(Error::Evidence("control deadline overflow"))?);
    if Instant::now()>=deadline {return Err(Error::Evidence("native cache control deadline expired"));}
    let span=deadline.checked_duration_since(c.clock_observed_at).ok_or(Error::Evidence("native cache control deadline expired"))?;
    let span=u64::try_from(span.as_millis()).map_err(|_|Error::Evidence("clock overflow"))?;
    let upper=c.now_lower_ms.checked_add(span).ok_or(Error::Evidence("clock overflow"))?.min(d.origin.started_at_ms.saturating_add(10_830_000));
    let mut bound=claims(d,c,operation)?;
    if bound["issued_at_ms"].as_u64().ok_or(Error::Evidence("control issued time absent"))?>=upper {return Err(Error::Evidence("native cache control deadline expired"));}
    bound["deadline_ms"]=upper.into();
    bound["payload_digest"]=prepared.digest().into();
    c.ensure_fresh()?;
    cache_auth::sign_prehashed(secret,&bound,prepared).map_err(Error::Store)
}

pub(crate) fn verify_receipt_query_response(d:&Dispatch,packet:&Value,secret:&[u8])->Result<()> {
    let request=d.control_claims.as_ref().filter(|c|c["operation"]=="receipts").ok_or(Error::Evidence("no signed native receipt query"))?;
    let payload=cache_auth::payload_for("receipts",packet).map_err(Error::Store)?;
    let response=cache_auth::verify_signature(secret,&packet[cache_auth::PROOF_FIELD],&payload).map_err(Error::Store)?;
    if packet["native_request_id"]!=d.origin.native_request_id || response["operation"]!="receipts" {return Err(Error::Evidence("receipt query transport/native identity mismatch"));}
    for key in ["model","native_origin","native_incarnation","source_digest","runtime_profile_digest","reservation_id","authority_revision","native_request_id","namespace_digest","full_render_digest","token_ids_digest","request_started_at_ms","issued_at_ms","deadline_ms","retain_for_ms"] {
        if response[key]!=request[key] {return Err(Error::Evidence("receipt query signed context mismatch"));}
    }
    let digest=cache_auth::digest(&payload).map_err(Error::Store)?;
    for key in ["operation_id","nonce"] {if response[key]!=cache_auth::digest(&json!(["native-response",request[key],digest])).map_err(Error::Store)? {return Err(Error::Evidence("receipt query response replay binding mismatch"));}}
    Ok(())
}

/// Only this authenticated reserve outcome permits a pre-generation read-zero
/// reprepare. Native emits it after coherent rollback of the failed read borrow.
pub(crate) fn verify_read_source_unavailable(d:&Dispatch,actual_worker_url:&str,packet:&Value,secret:&[u8])->Result<()> {
    if d.selected_read.is_none() || actual_worker_url!=d.worker.worker_url {
        return Err(Error::Evidence("read-source rejection without selected native read"));
    }
    let request=d.reserve_claims.as_ref().ok_or(Error::Evidence("no signed reserve operation"))?;
    verify_read_rejection_packet(request,&d.origin.native_request_id,&d.reservation_id,d.authority_revision,packet,secret)
}

fn verify_read_rejection_packet(request:&Value,rid:&str,reservation:&str,revision:i64,packet:&Value,secret:&[u8])->Result<()> {
    let keys=packet.as_object().ok_or(Error::Evidence("malformed read-source rejection"))?.keys().map(String::as_str).collect::<BTreeSet<_>>();
    if keys!=BTreeSet::from(["native_request_id","reservation_id","authority_revision","error",cache_auth::PROOF_FIELD])
        || packet["native_request_id"]!=rid || packet["reservation_id"]!=reservation || packet["authority_revision"]!=revision
        || packet["error"]!=json!({"type":"cache_read_source_unavailable"}) || request["operation"]!="reserve" {
        return Err(Error::Evidence("native read-source rejection status/identity mismatch"));
    }
    let payload=cache_auth::payload_for("receipts",packet).map_err(Error::Store)?;
    let response=cache_auth::verify_signature(secret,&packet[cache_auth::PROOF_FIELD],&payload).map_err(Error::Store)?;
    if response["operation"]!="receipts" {return Err(Error::Evidence("read-source rejection response operation mismatch"));}
    for key in ["model","native_origin","native_incarnation","source_digest","runtime_profile_digest","reservation_id","authority_revision","native_request_id","namespace_digest","full_render_digest","token_ids_digest","request_started_at_ms","issued_at_ms","deadline_ms","retain_for_ms"] {
        if response[key]!=request[key] {return Err(Error::Evidence("read-source rejection signed context mismatch"));}
    }
    let digest=cache_auth::digest(&payload).map_err(Error::Store)?;
    for key in ["operation_id","nonce"] {
        if response[key]!=cache_auth::digest(&json!(["native-response",request[key],digest])).map_err(Error::Store)? {
            return Err(Error::Evidence("read-source rejection replay binding mismatch"));
        }
    }
    Ok(())
}

/// Called only after the actual all-rank native cancel response. Consumed cache
/// holds and published origin markers keep their original expiry. The renewed
/// request gets read zero and must obtain a new native reservation ACK.
pub async fn cancel_dispatch(store:&EtcdStore,d:&Dispatch,actual_worker_url:&str,packet:&Value,secret:&[u8])->Result<()> {
    let request=d.control_claims.as_ref().filter(|c|c["operation"]=="cancel").ok_or(Error::Evidence("no signed native cancel operation"))?;
    let payload=cache_auth::payload_for("receipts",packet).map_err(Error::Store)?;
    let response=cache_auth::verify_signature(secret,&packet[cache_auth::PROOF_FIELD],&payload).map_err(Error::Store)?;
    if actual_worker_url!=d.worker.worker_url || packet["native_request_id"]!=d.origin.native_request_id
        || packet["reservation_id"]!=d.reservation_id || packet["authority_revision"]!=d.authority_revision
        || packet["cancelled"]!=true || response["operation"]!="receipts" {return Err(Error::Evidence("native cancellation transport/status mismatch"));}
    for key in ["model","native_origin","native_incarnation","source_digest","runtime_profile_digest","reservation_id","authority_revision","native_request_id","namespace_digest","full_render_digest","token_ids_digest","request_started_at_ms","issued_at_ms","deadline_ms","retain_for_ms"] {
        if response[key]!=request[key] {return Err(Error::Evidence("native cancel signed context mismatch"));}
    }
    let event=cache_auth::digest(&payload).map_err(Error::Store)?;
    for key in ["operation_id","nonce"] {if response[key]!=cache_auth::digest(&json!(["native-response",request[key],event])).map_err(Error::Store)? {return Err(Error::Evidence("native cancel replay binding mismatch"));}}
    let mut ranks=BTreeSet::new();
    for r in packet["rank_results"].as_array().ok_or(Error::Evidence("native cancel lacks rank results"))? {
        let id=u32::try_from(r["rank_id"].as_u64().ok_or(Error::Evidence("native cancel rank type invalid"))?).map_err(|_|Error::Evidence("native cancel rank overflow"))?;
        if r["cancelled"]!=true || !ranks.insert(id) {return Err(Error::Evidence("native cancel rank failed/duplicate"));}
    }
    if ranks!=d.worker.rank_tier_clocks.keys().copied().collect() {return Err(Error::Evidence("native cancel incomplete topology"));}
    let key=format!("{ROOT}reservations/{}",d.reservation_id);
    let deleted=bounded(store.client.clone().txn(Txn::new().when(vec![Compare::mod_revision(key.as_str(),CompareOp::Equal,d.authority_revision)])
        .and_then(vec![TxnOp::delete(key.as_str(),None),TxnOp::delete(format!("{key}/ack"),None)]))).await?;
    if !deleted.succeeded() {return Err(Error::Evidence("cancelled reservation revision changed"));}
    Ok(())
}

pub async fn reprepare_after_cancel(store:&EtcdStore,d:&Dispatch,actual_worker_url:&str,packet:&Value,secret:&[u8])->Result<Dispatch> {
    cancel_dispatch(store,d,actual_worker_url,packet,secret).await?;
    prepare_dispatch(store,d.origin.clone(),d.worker.clone(),d.points.values().cloned().collect(),None,d.generation_epoch).await
}

/// Actual existing same-worker reserve control body. Token IDs and provenance
/// come from the authenticated native full renderer; no frontend re-encoding.
pub(crate) async fn signed_reservation_body<F,Fut>(store:&EtcdStore,d:&mut Dispatch,native_generation_body:&Value,token_ids:&Value,secret:&[u8],c:&SigningContext,fresh_clock:F)->Result<Value>
where F:FnOnce(&cache_auth::PrehashedPayload)->Fut,Fut:std::future::Future<Output=Result<ClockBounds>> {
    if d.worker.rank_tier_clocks.is_empty() || cache_auth::digest(token_ids).map_err(Error::Store)?!=c.token_ids_digest
        || token_ids.as_array().is_none() {return Err(Error::Evidence("qualified topology/actual count IDs missing"));}
    store.confirm_dispatch(d).await?;
    let mut native=native_generation_body.clone();d.attach_private_fields(&mut native)?;
    let provenance=native.get("prompt_cache_provenance").ok_or(Error::Evidence("actual native renderer provenance missing"))?;
    let mut body=json!({"model":c.model,"native_request_id":d.origin.native_request_id,"reservation_id":d.reservation_id,
        "authority_revision":d.authority_revision,"cache_request_started_at_ms":d.origin.started_at_ms,
        "cache_salt":native.get("cache_salt").ok_or(Error::Evidence("trusted native cache namespace salt missing"))?,
        "token_ids":token_ids,"prompt_cache_provenance":provenance,
        "prompt_cache_commit_points":native["prompt_cache_commit_points"],"prompt_cache_read_max_tokens":native["prompt_cache_read_max_tokens"],
        "prompt_cache_max_tokens":native["prompt_cache_max_tokens"]});
    if let Some(point)=native.get("prompt_cache_read_commit_point") {body["prompt_cache_read_commit_point"]=point.clone();}
    if let Some(shared)=native.get("prompt_cache_shared_read") {body["prompt_cache_shared_read"]=shared.clone();}
    if let Some(expiry)=native.get("prompt_cache_read_source_expires_at_ms") {body["prompt_cache_read_source_expires_at_ms"]=expiry.clone();}
    let prepared=cache_auth::PrehashedPayload::new(&body).map_err(Error::Store)?;
    let clock=fresh_clock(&prepared).await?;
    let c=c.refreshed(&clock)?;
    let mut bound=claims(d,&c,"reserve")?;
    bound["payload_digest"]=prepared.digest().into();
    c.ensure_fresh()?;
    let proof=cache_auth::sign_prehashed(secret,&bound,&prepared).map_err(Error::Store)?;
    d.reserve_claims=Some(proof["claims"].clone());
    body.as_object_mut().unwrap().insert(cache_auth::PROOF_FIELD.into(),proof);
    Ok(body)
}

/// Exact authenticated all-rank allocator acknowledgements, immutably persisted
/// on the existing reservation lease before native generation is admitted.
pub async fn accept_reservation_ack(store:&EtcdStore,d:&mut Dispatch,actual_worker_url:&str,packet:&Value,secret:&[u8])->Result<()> {
    if actual_worker_url!=d.worker.worker_url || packet["native_request_id"]!=d.origin.native_request_id {return Err(Error::Evidence("reserve transport/native ID mismatch"));}
    let request=d.reserve_claims.as_ref().ok_or(Error::Evidence("no signed reserve operation"))?;
    let payload=cache_auth::payload_for("receipts",packet).map_err(Error::Store)?;
    let response=cache_auth::verify_signature(secret,&packet[cache_auth::PROOF_FIELD],&payload).map_err(Error::Store)?;
    if response["operation"]!="receipts" {return Err(Error::Evidence("reserve ACK response operation mismatch"));}
    for key in ["model","native_origin","native_incarnation","source_digest","runtime_profile_digest","reservation_id","authority_revision","native_request_id","namespace_digest","full_render_digest","token_ids_digest","request_started_at_ms","issued_at_ms","deadline_ms","retain_for_ms"] {
        if response[key]!=request[key] {return Err(Error::Evidence("reserve ACK signed context mismatch"));}
    }
    let event=cache_auth::digest(&payload).map_err(Error::Store)?;
    for key in ["operation_id","nonce"] {
        if response[key]!=cache_auth::digest(&json!(["native-response",request[key],event])).map_err(Error::Store)? {return Err(Error::Evidence("reserve ACK response replay mismatch"));}
    }
    let ack:ReservationAck=serde_json::from_value(packet["reservation_ack"].clone()).map_err(|_|Error::Evidence("malformed actual reserve ACK"))?;
    if ack.reservation_id!=d.reservation_id || ack.authority_revision!=d.authority_revision || ack.native_origin_uuid!=d.worker.native_origin_uuid
        || ack.source_digest!=request["source_digest"] || ack.runtime_profile_digest!=d.worker.runtime_profile_digest || ack.namespace_digest!=d.worker.namespace_digest {return Err(Error::Evidence("reserve ACK native binding mismatch"));}
    let expected:BTreeSet<_>=d.points.keys().chain(d.selected_read.iter().map(|r|&r.point.endpoint_id)).cloned().collect();
    let mut ranks=BTreeSet::new();
    for rank in &ack.rank_acks {
        if !ranks.insert(rank.rank_id) || !d.worker.rank_tier_clocks.contains_key(&rank.rank_id) {return Err(Error::Evidence("reserve ACK rank mismatch"));}
        let mut points=BTreeSet::new();
        for point in &rank.point_acks {
            if !points.insert(point.endpoint_id.clone()) || !expected.contains(&point.endpoint_id) {return Err(Error::Evidence("reserve ACK point mismatch"));}
            let wanted=d.retain_for_ms[&point.endpoint_id];
            let p=d.points.get(&point.endpoint_id).or_else(||d.selected_read.as_ref().map(|r|&r.point)).unwrap();
            let mut tiers=BTreeSet::new();
            for tier in &point.tier_acks {
                if wanted==0 || !tiers.insert(tier.tier.clone()) || !d.worker.required_tiers.contains(&tier.tier)
                    || tier.native_clock_domain.is_empty() || tier.reserved_tokens<p.native_page_tokens || tier.ack_retain_for_ms!=wanted {
                    return Err(Error::Evidence("reserve ACK tier capacity/hold mismatch"));
                }
                if d.worker.rank_tier_clocks[&rank.rank_id].get(&tier.tier).is_some_and(|old|!old.is_empty() && old!=&tier.native_clock_domain) {return Err(Error::Evidence("qualified reserve clock domain changed"));}
                let deadline=tier.receive_monotonic_ns.checked_add(wanted.checked_mul(1_000_000).ok_or(Error::Evidence("ACK duration overflow"))?).ok_or(Error::Evidence("ACK deadline overflow"))?;
                if deadline!=tier.retain_until_monotonic_ns {return Err(Error::Evidence("reserve ACK duration is not origin remainder"));}
            }
            if (wanted>0 && tiers!=d.worker.required_tiers) || (wanted==0 && !tiers.is_empty()) {return Err(Error::Evidence("reserve ACK incomplete tier set"));}
        }
        if points!=expected {return Err(Error::Evidence("reserve ACK missing logical endpoint"));}
    }
    if ranks!=d.worker.rank_tier_clocks.keys().copied().collect() {return Err(Error::Evidence("reserve ACK missing rank"));}
    store.confirm_dispatch(d).await?;
    let reservation_key=format!("{ROOT}reservations/{}",d.reservation_id);
    let current=bounded(store.client.clone().get(reservation_key.as_str(),None)).await?;
    let row=current.kvs().first().ok_or(Error::Evidence("reserve ACK reservation expired"))?;
    let ack_key=format!("{reservation_key}/ack");
    let bytes=serde_json::to_vec(packet).map_err(|e|Error::Store(e.to_string()))?;
    let saved=bounded(store.client.clone().txn(Txn::new().when(vec![Compare::mod_revision(reservation_key,CompareOp::Equal,d.authority_revision),Compare::version(ack_key.as_str(),CompareOp::Equal,0)])
        .and_then(vec![TxnOp::put(ack_key,bytes,Some(PutOptions::new().with_lease(row.lease())))]))).await?;
    if !saved.succeeded() {return Err(Error::Evidence("reserve ACK already saved or revision changed"));}
    // Domains are authoritative only now, after actual signed ACK verification.
    // Point-specific receive/deadline/duration stay in the immutable ACK below.
    for rank in &ack.rank_acks {
        let clocks=d.worker.rank_tier_clocks.get_mut(&rank.rank_id).unwrap();
        for point in &rank.point_acks {for tier in &point.tier_acks {
            clocks.insert(tier.tier.clone(),tier.native_clock_domain.clone());
        }}
    }
    d.reserve_ack=Some(ack);
    Ok(())
}

#[cfg(test)]
mod wire_tests {
    use super::*;
    #[tokio::test(start_paused=true)]
    async fn cache_cleanup_control_uses_fresh_sample_and_original_shared_deadline() {
        // Production control signer/types; synthetic CPU clock, no native IO
        // or claim that an allocator cancellation has completed.
        let start=Instant::now();
        let origin=RequestOrigin::from_frontend_context("CPU-control".into(),"CPU-native-control".into(),1_000_000,start).unwrap();
        let worker=QualifiedWorker{worker_url:"http://CPU-worker.invalid:1".into(),native_origin_uuid:"CPU-incarnation".into(),runtime_profile_digest:"ab".repeat(32),namespace_digest:"cd".repeat(32),native_clock_domain:"CPU".into(),required_tiers:BTreeSet::from(["gpu".into()]),native_id_field:NativeIdField::Rid,rank_tier_clocks:BTreeMap::from([(0,BTreeMap::from([("gpu".into(),"CPU".into())]))])};
        let d=Dispatch{origin,worker,reservation_id:"CPU-reservation".into(),authority_revision:1,points:BTreeMap::new(),retain_for_ms:BTreeMap::new(),origin_leases:BTreeMap::new(),signed_claims:None,reserve_claims:None,reserve_ack:None,control_claims:None,selected_read:None,generation_epoch:None};
        let key=[0x5a;32];
        let mut c=SigningContext{model:"CPU-control-model".into(),source_digest:"ef".repeat(32),full_render_digest:"01".repeat(32),token_ids_digest:"02".repeat(32),operation_id:"CPU-control-operation".into(),nonce:"CPU-control-nonce".into(),now_lower_ms:1_000_000,now_upper_ms:1_000_002,clock_observed_at:start};
        let body=json!({"native_request_id":d.origin.native_request_id,"reservation_id":d.reservation_id,"authority_revision":d.authority_revision});
        let payload=cache_auth::PrehashedPayload::new(&body).unwrap();
        let cleanup=start+Duration::from_secs(30);
        tokio::time::advance(Duration::from_secs(2)).await;
        assert!(control_proof(&d,"cancel",&key,&c,cleanup,&payload).is_err(),"pre-etcd clock must not survive delayed preparation");
        c.now_lower_ms+=2_000;c.now_upper_ms+=2_000;c.clock_observed_at=Instant::now();
        let cancelled=control_proof(&d,"cancel",&key,&c,cleanup,&payload).unwrap();
        assert_eq!(cancelled["claims"]["deadline_ms"],1_030_000);
        assert_eq!(cancelled["claims"]["operation"],"cancel");
        assert_eq!(cache_auth::verify_signature(&key,&cancelled,&body).unwrap(),cancelled["claims"]);
        // Reserve/Gen admission remains one second; only cleanup spends the
        // already fixed cleanup budget rather than a new admission window.
        assert_eq!(claims(&d,&c,"reserve").unwrap()["deadline_ms"],1_003_000);
        tokio::time::advance(Duration::from_secs(10)).await;
        c.now_lower_ms+=10_000;c.now_upper_ms+=10_000;c.clock_observed_at=Instant::now();
        let queried=control_proof(&d,"receipts",&key,&c,cleanup,&payload).unwrap();
        assert_eq!(queried["claims"]["deadline_ms"],cancelled["claims"]["deadline_ms"]);
        for field in ["reservation_id","authority_revision","native_request_id","namespace_digest","request_started_at_ms","nonce"] {assert_eq!(queried["claims"][field],cancelled["claims"][field]);}
        let realm=start+Duration::from_secs(13);
        assert_eq!(control_proof(&d,"cancel",&key,&c,realm,&payload).unwrap()["claims"]["deadline_ms"],1_013_000);
        assert_eq!(control_proof(&d,"cancel",&key,&c,Instant::now()+Duration::from_secs(600),&payload).unwrap()["claims"]["deadline_ms"],1_042_000);
        let mut near_end=d.clone();near_end.origin.started=Instant::now()-Duration::from_secs(10_829);
        let mut near_clock=c.clone();near_clock.now_lower_ms=11_829_000;near_clock.now_upper_ms=11_829_002;
        assert_eq!(control_proof(&near_end,"cancel",&key,&near_clock,cleanup,&payload).unwrap()["claims"]["deadline_ms"],11_830_000);
        let mut tampered=body.clone();tampered["authority_revision"]=2.into();assert!(cache_auth::verify_signature(&key,&queried,&tampered).is_err());
        assert!(control_proof(&d,"generate",&key,&c,cleanup,&payload).is_err());
        tokio::time::advance(Duration::from_secs(18)).await;
        c.now_lower_ms+=18_000;c.now_upper_ms+=18_000;c.clock_observed_at=Instant::now();
        assert!(control_proof(&d,"cancel",&key,&c,cleanup,&payload).is_err());
    }
    #[test]
    fn read_source_unavailable_requires_exact_signed_reserve_context() {
        let key:Vec<u8>=(0..32).collect();
        let ack:Value=serde_json::from_str(include_str!("../vllm/bridge-wire-cold-reservation-ack-fixture.json")).unwrap();
        let mut request=ack[cache_auth::PROOF_FIELD]["claims"].clone();
        request["operation"]=json!("reserve");request["operation_id"]=json!("test-reserve-op");request["nonce"]=json!("test-reserve-nonce");
        let packet_for=|body:Value,context:&Value| {
            let mut response=context.clone();response["operation"]=json!("receipts");
            let digest=cache_auth::digest(&body).unwrap();
            for key in ["operation_id","nonce"] {response[key]=json!(cache_auth::digest(&json!(["native-response",context[key],digest])).unwrap());}
            let mut packet=body.clone();packet[cache_auth::PROOF_FIELD]=cache_auth::sign(&key,&response,&body).unwrap();packet
        };
        let body=json!({"native_request_id":request["native_request_id"],"reservation_id":request["reservation_id"],"authority_revision":request["authority_revision"],"error":{"type":"cache_read_source_unavailable"}});
        let verify=|packet:&Value|verify_read_rejection_packet(&request,request["native_request_id"].as_str().unwrap(),request["reservation_id"].as_str().unwrap(),request["authority_revision"].as_i64().unwrap(),packet,&key);
        assert!(verify(&packet_for(body.clone(),&request)).is_ok());
        let mut forged=packet_for(body.clone(),&request);forged["authority_revision"]=json!(0);assert!(verify(&forged).is_err());
        let mut other=body.clone();other["error"]=json!({"type":"capacity_exhausted"});assert!(verify(&packet_for(other,&request)).is_err());
        let mut other=request.clone();other["nonce"]=json!("new-reserve-nonce");assert!(verify(&packet_for(body.clone(),&other)).is_err());
        let mut other=body;other["reservation_ack"]=json!({});assert!(verify(&packet_for(other,&request)).is_err());
    }
    #[test]
    fn actual_gf_all_rank_signed_ack_and_event_packets_decode() {
        let key:Vec<u8>=(0..32).collect();
        let ack:Value=serde_json::from_str(include_str!("../vllm/bridge-wire-cold-reservation-ack-fixture.json")).unwrap();
        cache_auth::verify_signature(&key,&ack[cache_auth::PROOF_FIELD],&cache_auth::payload_for("receipts",&ack).unwrap()).unwrap();
        let decoded:ReservationAck=serde_json::from_value(ack["reservation_ack"].clone()).unwrap();
        assert_eq!(decoded.rank_acks.iter().map(|r|r.rank_id).collect::<BTreeSet<_>>(),BTreeSet::from([0,1,2,3]));
        for rank in &decoded.rank_acks {for point in &rank.point_acks {for tier in &point.tier_acks {
            assert_eq!(tier.tier,"gpu");
            assert_eq!(tier.receive_monotonic_ns+tier.ack_retain_for_ms*1_000_000,tier.retain_until_monotonic_ns);
        }}}
        for (raw,kind) in [(include_str!("../vllm/bridge-wire-write-commit-fixture.json"),"write_commit"),(include_str!("../vllm/bridge-wire-read-refresh-fixture.json"),"read_refresh")] {
            let event:Value=serde_json::from_str(raw).unwrap();
            cache_auth::verify_signature(&key,&event[cache_auth::PROOF_FIELD],&cache_auth::payload_for("receipts",&event).unwrap()).unwrap();
            assert_eq!(event["event_kind"],kind);
            let receipt:Receipt=serde_json::from_value(event["receipt"].clone()).unwrap();
            assert_eq!(receipt.rank_receipts.iter().map(|r|r.rank_id).collect::<BTreeSet<_>>(),BTreeSet::from([0,1,2,3]));
            assert_eq!(receipt.rank_receipts.iter().map(|r|r.published_tokens).min(),Some(receipt.published_tokens));
            assert_eq!(receipt.rank_receipts.iter().map(|r|r.actual_read_tokens).min(),Some(receipt.actual_read_tokens));
        }
        let cancel:Value=serde_json::from_str(include_str!("../vllm/bridge-wire-cancel-fixture.json")).unwrap();
        cache_auth::verify_signature(&key,&cancel[cache_auth::PROOF_FIELD],&cache_auth::payload_for("receipts",&cancel).unwrap()).unwrap();
        assert_eq!(cancel["cancelled"],true);
        assert_eq!(cancel["rank_results"].as_array().unwrap().len(),4);
    }
}
