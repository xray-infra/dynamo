//! New E19 source-cause gate only. Loopback signed CPU wire plus disposable real
//! etcd exercises actual ownership, verifier, CAS and shutdown registry. The
//! two-rank native lifecycle below is a declared CPU control fixture, not GPU
//! scheduler/weights/backend qualification or an old protocol matrix rerun.
use super::*;
use axum::{extract::State,routing::post,Json,Router};
use std::sync::{atomic::{AtomicBool,AtomicUsize,Ordering},Mutex};
use tokio::sync::Notify;
use crate::native_cleanup::CleanupState;

#[derive(Default)]
struct Wire {
    reserve_seen:Notify,
    reserve_release:Notify,
    cancel_seen:Notify,
    cancel_release:Notify,
    delay_reserve:AtomicBool,
    bad_cancel:AtomicBool,
    unconsumed_rank_refs:AtomicUsize,
    committed_ttl_hold:AtomicBool,
    gen_posts:AtomicUsize,
    last_reservation:Mutex<String>,
}
const KEY:[u8;32]=[0x71;32];
fn response(request:&Value,mut body:Value)->Value {
    let mut claims=request[cache_auth::PROOF_FIELD]["claims"].clone();
    claims["operation"]="receipts".into();
    let digest=cache_auth::digest(&body).unwrap();
    for key in ["operation_id","nonce"] {claims[key]=cache_auth::digest(&json!(["native-response",claims[key],digest])).unwrap().into();}
    body[cache_auth::PROOF_FIELD]=cache_auth::sign(&KEY,&claims,&body).unwrap();
    body
}
async fn reserve(State(w):State<Arc<Wire>>,Json(request):Json<Value>)->Json<Value> {
    let claims=&request[cache_auth::PROOF_FIELD]["claims"];
    cache_auth::verify_signature(&KEY,&request[cache_auth::PROOF_FIELD],&cache_auth::payload_for("reserve",&request).unwrap()).unwrap();
    *w.last_reservation.lock().unwrap()=claims["reservation_id"].as_str().unwrap().into();
    w.unconsumed_rank_refs.store(2,Ordering::SeqCst);
    w.reserve_seen.notify_one();
    if w.delay_reserve.load(Ordering::SeqCst) {w.reserve_release.notified().await;}
    let points=request["prompt_cache_commit_points"].as_array().unwrap();
    let rank=|id|json!({"rank_id":id,"point_acks":points.iter().map(|p|{
        let ms=p["retain_for_ms"].as_u64().unwrap();
        json!({"endpoint_id":p["endpoint_id"],"tier_acks":if ms==0 {json!([])}else{json!([{
            "tier":"gpu","reserved_tokens":p["native_page_tokens"],"native_clock_domain":"CPU-rank-clock",
            "receive_monotonic_ns":1000,"retain_until_monotonic_ns":1000+ms*1_000_000,"ack_retain_for_ms":ms}])}})
    }).collect::<Vec<_>>()});
    Json(response(&request,json!({"native_request_id":claims["native_request_id"],"reservation_ack":{
        "reservation_id":claims["reservation_id"],"authority_revision":claims["authority_revision"],
        "native_origin_uuid":claims["native_incarnation"],"source_digest":claims["source_digest"],
        "runtime_profile_digest":claims["runtime_profile_digest"],"namespace_digest":claims["namespace_digest"],
        "rank_acks":[rank(0),rank(1)]}})))
}
async fn receipts(Json(request):Json<Value>)->Json<Value> {
    cache_auth::verify_signature(&KEY,&request[cache_auth::PROOF_FIELD],&cache_auth::payload_for("receipts",&request).unwrap()).unwrap();
    Json(response(&request,json!({"native_request_id":request["native_request_id"],"receipts":[]})))
}
async fn cancel(State(w):State<Arc<Wire>>,Json(request):Json<Value>)->Json<Value> {
    cache_auth::verify_signature(&KEY,&request[cache_auth::PROOF_FIELD],&cache_auth::payload_for("cancel",&request).unwrap()).unwrap();
    w.cancel_seen.notify_one();
    w.cancel_release.notified().await;
    let good=!w.bad_cancel.load(Ordering::SeqCst);
    if good {w.unconsumed_rank_refs.store(0,Ordering::SeqCst);}
    Json(response(&request,json!({"native_request_id":request["native_request_id"],
        "reservation_id":request["reservation_id"],"authority_revision":request["authority_revision"],
        "cancelled":true,"rank_results":[{"rank_id":0,"cancelled":true},{"rank_id":1,"cancelled":good}]})))
}
async fn forbidden_gen(State(w):State<Arc<Wire>>)->Json<Value> {
    w.gen_posts.fetch_add(1,Ordering::SeqCst);Json(json!({"error":"E19 gate must not generate"}))
}

async fn fixture(store:EtcdStore,wire:Arc<Wire>,registry:Arc<NativeCleanupRegistry>,rid:&str)->(Arc<NativeCacheFrontend>,Value,CacheRequest,RenderedCount,tokio::task::JoinHandle<()>) {
    let listener=tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url=format!("http://{}",listener.local_addr().unwrap());
    let app=Router::new().route("/v1/cache/reservations",post(reserve))
        .route("/v1/cache/reservations/cancel",post(cancel)).route("/v1/cache/reservations/receipts",post(receipts))
        .route("/v1/chat/completions",post(forbidden_gen)).with_state(wire);
    let server=tokio::spawn(async move {axum::serve(listener,app).await.unwrap();});
    let profile="ab".repeat(32);let namespace="cd".repeat(32);
    let worker=QualifiedWorker{worker_url:url,native_origin_uuid:"CPU-incarnation".into(),runtime_profile_digest:profile,
        namespace_digest:namespace.clone(),native_clock_domain:"CPU-driver-clock".into(),required_tiers:BTreeSet::from(["gpu".into()]),native_id_field:NativeIdField::Rid,
        rank_tier_clocks:[0,1].into_iter().map(|id|(id,BTreeMap::from([("gpu".into(),"CPU-rank-clock".into())]))).collect()};
    let binding=FrontendWorkerBinding{worker:worker.clone(),model:"CPU-NOTQUALIFIED".into(),source_digest:"ef".repeat(32),kv_compat_digest:"12".repeat(32),count_path:"/tokenize".into(),tenant:"CPU".into(),default_cache_salt:"CPU".into(),secret:KEY,shared_backend:None};
    let frontend=Arc::new(NativeCacheFrontend::new(reqwest::Client::new(),store,vec![binding]).unwrap().use_cpu_fixture_clock().with_cleanup_registry(registry));
    let body=json!({"model":"CPU-NOTQUALIFIED","messages":[{"role":"user","content":"CPU E19"}],"rid":rid,"cache_salt":"CPU","prompt_cache_blocks":[{"endpoint_id":"CPU-write"}]});
    let token_ids=json!([0,1,2,3,4,5,6,7,8,9]);
    let token_ids_digest=cache_auth::digest(&token_ids).unwrap();
    let prefix_digest=hex_digest(&json!(cache_auth::digest(&json!([0,1,2,3,4,5,6,7])).unwrap())).unwrap();
    let point=RenderPoint{endpoint_id:"CPU-write".into(),native_end_byte:10,native_mark_tokens:8,native_page_tokens:8,
        prefix:Prefix{tenant:"CPU".into(),namespace,runtime:worker.registry_runtime(),digest:prefix_digest,marked_endpoint_digest:prefix_digest,marked_render_digest:[3;32],marked_tokens:8,tokens:8},ttl:Ttl::FiveMinutes};
    let request=CacheRequest::new(format!("http-{rid}"),std::time::Instant::now(),BTreeMap::from([("CPU-write".into(),Ttl::FiveMinutes)]),BTreeSet::new());
    let origin=RequestOrigin::from_frontend_context(format!("http-{rid}"),rid.into(),1_000_000,Instant::now()).unwrap();
    let rendered=RenderedCount{prompt_tokens:10,worker_url:worker.worker_url.clone(),body:body.clone(),tokens:token_ids,
        provenance:json!({"full_prompt_digest":"01".repeat(32),"token_ids_digest":token_ids_digest}),origin,worker,points:BTreeMap::from([("CPU-write".into(),point)])};
    (frontend,body,request,rendered,server)
}

#[tokio::test]
#[ignore="requires owned disposable loopback etcd; no GPU/native Gen"]
async fn e19_cache_owned_prepare_drop_and_finish_wait_for_verified_cleanup() {
    let store=EtcdStore::connect(vec![std::env::var("CACHE_AUTHORITY_TEST_ETCD").unwrap()]).await.unwrap();
    for scenario in ["prepare-cancel-after-CAS","body-drop","finish-caller-cancel","bad-allrank-ACK"] {
        let registry=Arc::new(NativeCleanupRegistry::new());let wire=Arc::new(Wire::default());
        wire.committed_ttl_hold.store(true,Ordering::SeqCst);
        wire.delay_reserve.store(scenario=="prepare-cancel-after-CAS",Ordering::SeqCst);
        wire.bad_cancel.store(scenario=="bad-allrank-ACK",Ordering::SeqCst);
        let rid=format!("e19-{}-{}",scenario,uuid::Uuid::new_v4());let id=format!("cache:{rid}");
        let (frontend,body,request,rendered,server)=fixture(store.clone(),wire.clone(),registry.clone(),&rid).await;
        let url=rendered.worker_url.clone();
        let mut prepare=tokio::spawn(async move {frontend.prepare_generation(&url,body,&request,rendered,None,request.control_deadline().unwrap()).await});
        tokio::time::timeout(Duration::from_secs(2),async {
            tokio::select! {
                _=wire.reserve_seen.notified()=>{},
                outcome=&mut prepare=>match outcome {
                    Ok(Err(error))=>panic!("{scenario}: preparation failed before Reserve: {error}"),
                    Ok(Ok(_))=>panic!("{scenario}: preparation returned without Reserve notification"),
                    Err(error)=>panic!("{scenario}: preparation task failed before Reserve: {error}"),
                }
            }
        }).await.unwrap();
        assert_eq!(registry.states().get(&id),Some(&CleanupState::Running));
        // Reserve reaching HTTP proves the real etcd CAS already committed.
        let reservation=wire.last_reservation.lock().unwrap().clone();
        assert!(!store.client.clone().get(format!("/dsh/anthropic-cache/v1/reservations/{reservation}"),None).await.unwrap().kvs().is_empty());
        if scenario=="prepare-cancel-after-CAS" {
            prepare.abort();let _=prepare.await;wire.reserve_release.notify_one();
        }else{
            let mut prepared=prepare.await.unwrap().unwrap();
            if scenario=="finish-caller-cancel" {
                prepared.guard.begin_generation().await.unwrap();
                let finish=tokio::spawn(async move {prepared.guard.query_finish(false).await});
                tokio::time::timeout(Duration::from_secs(2),wire.cancel_seen.notified()).await.unwrap();
                finish.abort();let _=finish.await;
                wire.cancel_seen.notify_one();
            }else{drop(prepared);}
        }
        tokio::time::timeout(Duration::from_secs(2),wire.cancel_seen.notified()).await.unwrap();
        assert_eq!(wire.unconsumed_rank_refs.load(Ordering::SeqCst),2);
        assert_eq!(registry.states().get(&id),Some(&CleanupState::Running));
        assert!(tokio::time::timeout(Duration::from_millis(20),registry.wait_for_physical_zero()).await.is_err());
        wire.cancel_release.notify_one();
        if scenario=="bad-allrank-ACK" {
            let blocked=tokio::time::timeout(Duration::from_secs(2),registry.await_quiescence()).await.unwrap().unwrap_err();
            assert!(matches!(blocked.entries.get(&id),Some(CleanupState::Blocked(_))));
            assert_eq!(wire.unconsumed_rank_refs.load(Ordering::SeqCst),2);
            assert!(tokio::time::timeout(Duration::from_millis(20),registry.wait_for_physical_zero()).await.is_err());
            assert!(!store.client.clone().get(format!("/dsh/anthropic-cache/v1/reservations/{reservation}"),None).await.unwrap().kvs().is_empty());
        }else{
            tokio::time::timeout(Duration::from_secs(2),registry.wait_for_physical_zero()).await.unwrap();
            assert!(registry.states().is_empty());assert_eq!(wire.unconsumed_rank_refs.load(Ordering::SeqCst),0);
            assert!(store.client.clone().get(format!("/dsh/anthropic-cache/v1/reservations/{reservation}"),None).await.unwrap().kvs().is_empty());
        }
        assert!(wire.committed_ttl_hold.load(Ordering::SeqCst));assert_eq!(wire.gen_posts.load(Ordering::SeqCst),0);
        server.abort();let _=server.await;
    }
}

#[tokio::test]
#[ignore="requires owned disposable loopback etcd; new canonical composition seam only"]
async fn e19_cached_signer_preserves_canonical_control_and_count_metadata() {
    use crate::native_request_cleanup::{ExpectedRequest,RequestControl};
    let store=EtcdStore::connect(vec![std::env::var("CACHE_AUTHORITY_TEST_ETCD").unwrap()]).await.unwrap();
    for wrong_canonical in [false,true] {
        let registry=Arc::new(NativeCleanupRegistry::new());let wire=Arc::new(Wire::default());
        let rid=uuid::Uuid::new_v4().simple().to_string();
        let (frontend,mut body,request,mut rendered,server)=fixture(store.clone(),wire.clone(),registry.clone(),&rid).await;
        assert_eq!(rendered.cache_salt().unwrap(),"CPU");
        assert_eq!(rendered.request_started_at_ms(),1_000_000);
        let salt=rendered.body.as_object_mut().unwrap().remove("cache_salt").unwrap();
        assert!(rendered.cache_salt().is_err());
        rendered.body["cache_salt"]=json!(19);assert!(rendered.cache_salt().is_err());
        rendered.body["cache_salt"]=salt;
        body["cache_salt"]=rendered.cache_salt().unwrap().into();
        // Actual typed ordinary-control validation/codec, using declared CPU
        // participants only. This is not an installed native cohort receipt.
        let participants=BTreeSet::from(["CPU-admission".to_string(),"CPU-scheduler".to_string()]);
        let canonical=serde_json::to_string(&cache_auth::ordinary_body_for(&body).unwrap()).unwrap();
        let expected=ExpectedRequest{control:RequestControl{schema_version:1,
            transaction_id:uuid::Uuid::new_v4().simple().to_string(),request_root:rid,
            expected_native_incarnation:"CPU-incarnation".into(),
            expected_cleanup_cohort_digest:cache_auth::digest(&serde_json::to_value(&participants).unwrap()).unwrap(),
            ordinary_body_digest:cache_auth::digest(&json!(canonical)).unwrap(),expected_choices:1},participants};
        expected.validate().unwrap();
        cache_auth::attach_canonical_body(&mut body).unwrap();
        body["prompt_cache_request_control"]=serde_json::to_value(&expected.control).unwrap();
        let original_control=body["prompt_cache_request_control"].clone();
        let original_canonical=body["prompt_cache_canonical_body"].clone();
        if wrong_canonical {body["prompt_cache_canonical_body"]=json!("{}");}
        // Cleanup remains actual signed two-rank fixture ACK + exact real CAS.
        wire.cancel_release.notify_one();
        let url=rendered.worker_url.clone();
        let result=frontend.prepare_generation(&url,body,&request,rendered,None,request.control_deadline().unwrap()).await;
        if wrong_canonical {
            assert!(matches!(result,Err(Error::Evidence("existing canonical native body differs from generation body"))));
        }else{
            let prepared=result.unwrap();
            assert_eq!(prepared.body["prompt_cache_request_control"],original_control);
            assert_eq!(prepared.body["prompt_cache_canonical_body"],original_canonical);
            assert_eq!(prepared.body["cache_salt"],json!("CPU"));
            let claims=cache_auth::verify_signature(&KEY,&prepared.body[cache_auth::PROOF_FIELD],
                &cache_auth::payload_for("generate",&prepared.body).unwrap()).unwrap();
            assert_eq!(claims["request_started_at_ms"],json!(1_000_000));
            let mut changed=prepared.body.clone();changed["prompt_cache_request_control"]["expected_choices"]=json!(2);
            assert!(cache_auth::verify_signature(&KEY,&changed[cache_auth::PROOF_FIELD],
                &cache_auth::payload_for("generate",&changed).unwrap()).is_err());
            drop(prepared);
        }
        tokio::time::timeout(Duration::from_secs(2),registry.wait_for_physical_zero()).await.unwrap();
        assert_eq!(wire.gen_posts.load(Ordering::SeqCst),0);
        server.abort();let _=server.await;
    }
}

#[tokio::test]
#[ignore="owned loopback etcd; E45 root/choice-map signing composition, no native child IO"]
async fn e45_cached_choices_keep_one_root_reservation_and_signed_room_map() {
    use crate::native_pd::{control::{ControlRequest,PdRole},dispatch::{GenerationBinding,PdGenerationTransport,RoomRange},qualification::PdRoleArtifact};
    use crate::native_request_cleanup::{ExpectedRequest,RequestControl};
    let store=EtcdStore::connect(vec![std::env::var("CACHE_AUTHORITY_TEST_ETCD").unwrap()]).await.unwrap();
    let registry=Arc::new(NativeCleanupRegistry::new());let wire=Arc::new(Wire::default());
    let root=uuid::Uuid::new_v4().simple().to_string();
    let (frontend,mut body,request,mut rendered,server)=fixture(store,wire.clone(),registry.clone(),&root).await;
    body["n"]=json!(2);
    rendered.body["n"]=json!(2);
    // Existing declared CPU role codec fixture supplies the complete peer
    // descriptor. This gate tests signature composition, not role startup or
    // native all-child IO/physical ownership qualification.
    let roles:Value=serde_json::from_str(include_str!("../native_pd/role-qualification-v1.synthetic.json")).unwrap();
    let prefill: PdRoleArtifact=serde_json::from_value(roles["artifact"].clone()).unwrap();
    let mut decode: PdRoleArtifact=serde_json::from_value(roles["peer_artifact"].clone()).unwrap();
    let prefill_incarnation="1".repeat(32);decode.native_incarnation="2".repeat(32);
    decode.registrations[0].native_rank_incarnation=decode.native_incarnation.clone();
    decode.native_role_registration_digest=cache_auth::digest(&json!({"ranks":decode.ranks,"registrations":decode.registrations})).unwrap();
    prefill.validate().unwrap();decode.validate().unwrap();
    let transaction_id=uuid::Uuid::new_v4().simple().to_string();
    let rooms=RoomRange::new(41,2).unwrap().bindings(&root).unwrap();
    assert_eq!(rooms.iter().map(|room|room.sample_rid.clone()).collect::<Vec<_>>(),
        vec![format!("{root}_0:sample:0"),format!("{root}_0:sample:1"),format!("{root}_0:prefill")]);
    let mut epoch_input=json!({"prefill_incarnation":prefill_incarnation,"decode_incarnation":decode.native_incarnation,
        "kv_compat_digest":prefill.kv_compat_digest,"qualified_rank_set_digest":prefill.qualified_rank_set_digest});
    let session_epoch=cache_auth::digest(&epoch_input).unwrap();
    epoch_input["transaction_id"]=json!(transaction_id);
    let binding=GenerationBinding{control:ControlRequest{schema_version:1,transaction_id:transaction_id.clone(),
        request_root:root.clone(),expected_native_incarnation:prefill_incarnation.clone(),
        expected_qualified_rank_set_digest:prefill.qualified_rank_set_digest.clone(),role:PdRole::Prefill,rooms},
        transport:PdGenerationTransport{prefill_incarnation:prefill_incarnation.clone(),decode_incarnation:decode.native_incarnation.clone(),
            prefill_native_origin:rendered.worker_url.clone(),decode_native_origin:decode.native_origin.clone(),
            kv_compat_digest:prefill.kv_compat_digest.clone(),qualified_rank_set_digest:prefill.qualified_rank_set_digest.clone(),
            transport_epoch:cache_auth::digest(&epoch_input).unwrap(),session_epoch},peer_artifact:decode};
    binding.validate_ordinary(&root,2,&transaction_id,&prefill_incarnation,&rendered.worker_url,&binding.control.expected_qualified_rank_set_digest).unwrap();
    let mut missing_warmup=binding.clone();missing_warmup.control.rooms.pop();
    assert!(missing_warmup.validate_ordinary(&root,2,&transaction_id,&prefill_incarnation,&rendered.worker_url,&binding.control.expected_qualified_rank_set_digest).is_err());
    let mut foreign_child=binding.clone();foreign_child.control.rooms[0].sample_rid=format!("{}_0:sample:0",uuid::Uuid::new_v4().simple());
    assert!(foreign_child.validate_ordinary(&root,2,&transaction_id,&prefill_incarnation,&rendered.worker_url,&binding.control.expected_qualified_rank_set_digest).is_err());
    body["prompt_cache_pd_binding"]=serde_json::to_value(&binding).unwrap();
    body["bootstrap_host"]=json!("127.0.0.1");body["bootstrap_port"]=json!(8998);body["bootstrap_room"]=json!(41);
    let participants=BTreeSet::from(["CPU-admission".to_string(),"CPU-scheduler".to_string()]);
    let canonical=serde_json::to_string(&cache_auth::ordinary_body_for(&body).unwrap()).unwrap();
    let expected=ExpectedRequest{control:RequestControl{schema_version:1,transaction_id,
        request_root:root.clone(),expected_native_incarnation:rendered.worker.native_origin_uuid.clone(),
        expected_cleanup_cohort_digest:cache_auth::digest(&serde_json::to_value(&participants).unwrap()).unwrap(),
        ordinary_body_digest:cache_auth::digest(&json!(canonical)).unwrap(),expected_choices:2},participants};
    expected.validate().unwrap();cache_auth::attach_canonical_body(&mut body).unwrap();
    body["prompt_cache_request_control"]=serde_json::to_value(&expected.control).unwrap();
    let url=rendered.worker_url.clone();
    let prepared=frontend.prepare_generation(&url,body,&request,rendered,None,request.control_deadline().unwrap()).await.unwrap();
    let proof=&prepared.body[cache_auth::PROOF_FIELD];
    let claims=cache_auth::verify_signature(&KEY,proof,&cache_auth::payload_for("generate",&prepared.body).unwrap()).unwrap();
    assert_eq!(claims["native_request_id"],json!(root));assert_eq!(claims["request_started_at_ms"],json!(1_000_000));
    assert_eq!(prepared.body["n"],json!(2));assert_eq!(prepared.body["prompt_cache_request_control"]["expected_choices"],json!(2));
    assert_eq!(prepared.body["prompt_cache_pd_binding"],serde_json::to_value(binding).unwrap());
    assert_eq!(wire.last_reservation.lock().unwrap().as_str(),claims["reservation_id"].as_str().unwrap());
    assert_eq!(registry.states().keys().cloned().collect::<Vec<_>>(),vec![format!("cache:{root}")]);
    for change in ["n","warmup","child","root"] {
        let mut altered=prepared.body.clone();
        match change {
            "n"=>{
                altered["n"]=json!(1);
                // Ordinary generation fields are protected by their exact
                // canonical string. Changing n alone fails native projection;
                // repairing that string must still fail the original MAC.
                altered["prompt_cache_canonical_body"]=json!(serde_json::to_string(&cache_auth::ordinary_body_for(&altered).unwrap()).unwrap());
            },
            "warmup"=>{altered["prompt_cache_pd_binding"]["control"]["rooms"].as_array_mut().unwrap().pop();},
            "child"=>altered["prompt_cache_pd_binding"]["control"]["rooms"][0]["sample_rid"]=json!("foreign_0:sample:0"),
            "root"=>altered["rid"]=json!("foreign"),
            _=>unreachable!(),
        }
        assert!(cache_auth::verify_signature(&KEY,proof,&cache_auth::payload_for("generate",&altered).unwrap()).is_err(),"{change}");
    }
    wire.cancel_release.notify_one();drop(prepared);
    tokio::time::timeout(Duration::from_secs(2),registry.wait_for_physical_zero()).await.unwrap();
    assert_eq!(wire.gen_posts.load(Ordering::SeqCst),0);
    assert_eq!(wire.unconsumed_rank_refs.load(Ordering::SeqCst),0);
    server.abort();let _=server.await;
}

#[tokio::test]
#[ignore="owned loopback etcd; E28 new large-payload signing cause only"]
async fn e28_cache_large_payload_precedes_fresh_clock_and_stale_sample_rejects() {
    use super::super::frontend_clock::ClockBounds;
    let store=EtcdStore::connect(vec![std::env::var("CACHE_AUTHORITY_TEST_ETCD").unwrap()]).await.unwrap();
    let registry=Arc::new(NativeCleanupRegistry::new());let wire=Arc::new(Wire::default());
    let rid=uuid::Uuid::new_v4().simple().to_string();
    let (frontend,mut body,request,rendered,server)=fixture(store.clone(),wire.clone(),registry,&rid).await;
    // Bounded declared CPU encoder payload. The first eight IDs match the
    // endpoint; the long tail is valid u32 data, not model-vocabulary evidence.
    let target=8*1024*1024;
    let n=8+(target-17+10)/11;
    let mut ids=(0..8).collect::<Vec<u32>>();ids.resize(n,u32::MAX);
    let token_ids=serde_json::to_value(ids).unwrap();
    assert!(serde_json::to_vec(&token_ids).unwrap().len()>=target);
    let token_digest=cache_auth::digest(&token_ids).unwrap();
    body["messages"][0]["content"]=json!("x".repeat(target));
    body["prompt_cache_provenance"]=json!({"full_prompt_digest":"01".repeat(32),"token_ids_digest":token_digest});
    let mut dispatch=prepare_dispatch(&store,rendered.origin.clone(),rendered.worker.clone(),
        rendered.points.values().cloned().collect(),None,None).await.unwrap();
    let binding=frontend.binding(&rendered.worker_url).unwrap();
    let context=frontend.context(binding,&rendered.origin,"01".repeat(32).as_str(),&token_digest).await.unwrap();
    let reserve_witness=Arc::new(Mutex::new(None::<String>));
    let capture=reserve_witness.clone();
    let reserve_body=signed_reservation_body(&store,&mut dispatch,&body,&token_ids,&KEY,&context,move |prepared| {
        // This is the actual opaque builder passed by the production function,
        // not a callback count/shape or an independently invented digest.
        *capture.lock().unwrap()=Some(prepared.digest().to_owned());
        async {Ok(ClockBounds::cpu_fixture(Instant::now()))}
    }).await.unwrap();
    let reserve_payload=cache_auth::payload_for("reserve",&reserve_body).unwrap();
    let reserve_claims=cache_auth::verify_signature(&KEY,&reserve_body[cache_auth::PROOF_FIELD],&reserve_payload).unwrap();
    assert_eq!(reserve_witness.lock().unwrap().as_deref(),reserve_claims["payload_digest"].as_str());
    assert_eq!(reserve_claims["payload_digest"],json!(cache_auth::digest(&reserve_payload).unwrap()));
    let mut altered=reserve_body.clone();altered["token_ids"][0]=json!(99);
    assert!(cache_auth::verify_signature(&KEY,&altered[cache_auth::PROOF_FIELD],&cache_auth::payload_for("reserve",&altered).unwrap()).is_err());
    // Execute the existing declared two-rank CPU ACK producer directly. Avoid
    // treating large JSON parser/network latency as this HMAC clock-cause gate.
    let Json(ack)=reserve(State(wire.clone()),Json(reserve_body)).await;
    accept_reservation_ack(&store,&mut dispatch,&rendered.worker_url,&ack,&KEY).await.unwrap();
    let context=frontend.context(binding,&rendered.origin,"01".repeat(32).as_str(),&token_digest).await.unwrap();
    let unsigned=body.clone();
    let generation_witness=Arc::new(Mutex::new(None::<String>));let capture=generation_witness.clone();
    signed_generation_body(&store,&mut dispatch,&mut body,&KEY,&context,move |prepared| {
        *capture.lock().unwrap()=Some(prepared.digest().to_owned());
        async {Ok(ClockBounds::cpu_fixture(Instant::now()))}
    }).await.unwrap();
    let generation_payload=cache_auth::payload_for("generate",&body).unwrap();
    let claims=cache_auth::verify_signature(&KEY,&body[cache_auth::PROOF_FIELD],&generation_payload).unwrap();
    assert_eq!(generation_witness.lock().unwrap().as_deref(),claims["payload_digest"].as_str());
    assert_eq!(claims["request_started_at_ms"],json!(rendered.request_started_at_ms()));
    let mut altered=body.clone();altered["prompt_cache_canonical_body"]=json!("tampered");
    assert!(cache_auth::verify_signature(&KEY,&altered[cache_auth::PROOF_FIELD],&cache_auth::payload_for("generate",&altered).unwrap()).is_err());
    // Real elapsed 101ms after the explicit final sample must fail the existing
    // 100ms gate. No threshold reduction, fallback clock or timeout widening.
    let context=frontend.context(binding,&rendered.origin,"01".repeat(32).as_str(),&token_digest).await.unwrap();
    let stale=signed_reservation_body(&store,&mut dispatch,&unsigned,&token_ids,&KEY,&context,|_|async {
        let clock=ClockBounds::cpu_fixture(Instant::now());
        tokio::time::sleep(Duration::from_millis(101)).await;Ok(clock)
    }).await;
    assert!(matches!(stale,Err(Error::Evidence("clock qualification must be refreshed before signing"))));
    let context=frontend.context(binding,&rendered.origin,"01".repeat(32).as_str(),&token_digest).await.unwrap();
    let mut stale_body=unsigned;
    let stale=signed_generation_body(&store,&mut dispatch,&mut stale_body,&KEY,&context,|_|async {
        let clock=ClockBounds::cpu_fixture(Instant::now());
        tokio::time::sleep(Duration::from_millis(101)).await;Ok(clock)
    }).await;
    assert!(matches!(stale,Err(Error::Evidence("clock qualification must be refreshed before signing"))));
    wire.cancel_release.notify_one();
    frontend.cancel(&rendered.worker_url,&mut dispatch,"01".repeat(32).as_str(),&token_digest,Instant::now()+Duration::from_secs(30)).await.unwrap();
    assert_eq!(wire.gen_posts.load(Ordering::SeqCst),0);assert!(request.reads.is_empty());
    server.abort();let _=server.await;
}

#[tokio::test]
#[ignore="owned loopback etcd; immutable Reserve ACK to Gen plan cause only"]
async fn cache_ack_keeps_exact_endpoint_plan_after_delay_without_renewing_expiry() {
    use super::super::frontend_clock::{ClockBounds,qualified_cpu_tracking_sample};
    let store=EtcdStore::connect(vec![std::env::var("CACHE_AUTHORITY_TEST_ETCD").unwrap()]).await.unwrap();
    let registry=Arc::new(NativeCleanupRegistry::new());let wire=Arc::new(Wire::default());
    let rid=uuid::Uuid::new_v4().simple().to_string();
    let (frontend,mut body,_,rendered,server)=fixture(store.clone(),wire.clone(),registry,&rid).await;
    body["prompt_cache_provenance"]=rendered.provenance.clone();
    let full="01".repeat(32);let digest=cache_auth::digest(&rendered.tokens).unwrap();
    let mut dispatch=prepare_dispatch(&store,rendered.origin.clone(),rendered.worker.clone(),
        rendered.points.values().cloned().collect(),None,None).await.unwrap();
    let binding=frontend.binding(&rendered.worker_url).unwrap();
    let context=frontend.context(binding,&rendered.origin,&full,&digest).await.unwrap();
    let reservation=signed_reservation_body(&store,&mut dispatch,&body,&rendered.tokens,&KEY,&context,
        |_|async {Ok(ClockBounds::cpu_fixture(Instant::now()))}).await.unwrap();
    let planned=reservation["prompt_cache_commit_points"].clone();
    let reserve_claims=cache_auth::verify_signature(&KEY,&reservation[cache_auth::PROOF_FIELD],
        &cache_auth::payload_for("reserve",&reservation).unwrap()).unwrap();
    assert!(planned[0]["retain_for_ms"].as_u64().unwrap()>0);
    let Json(ack)=reserve(State(wire.clone()),Json(reservation)).await;
    accept_reservation_ack(&store,&mut dispatch,&rendered.worker_url,&ack,&KEY).await.unwrap();
    // Real elapsed time used to make Gen change the exact reserved plan.
    tokio::time::sleep(Duration::from_millis(20)).await;
    let context=frontend.context(binding,&rendered.origin,&full,&digest).await.unwrap();
    let mut generated=body.clone();
    signed_generation_body(&store,&mut dispatch,&mut generated,&KEY,&context,
        |_|async {Ok(ClockBounds::cpu_fixture(Instant::now()))}).await.unwrap();
    let claims=cache_auth::verify_signature(&KEY,&generated[cache_auth::PROOF_FIELD],
        &cache_auth::payload_for("generate",&generated).unwrap()).unwrap();
    assert_eq!(generated["prompt_cache_commit_points"],planned);
    assert_eq!(claims["retain_for_ms"],reserve_claims["retain_for_ms"]);
    assert_eq!(planned[0]["expires_at_ms"],json!(1_300_000));
    assert_eq!(ack["reservation_ack"]["rank_acks"][0]["point_acks"][0]["tier_acks"][0]["ack_retain_for_ms"],planned[0]["retain_for_ms"]);
    let mut changed=generated.clone();changed["prompt_cache_commit_points"][0]["retain_for_ms"]=json!(1);
    assert!(cache_auth::verify_signature(&KEY,&changed[cache_auth::PROOF_FIELD],
        &cache_auth::payload_for("generate",&changed).unwrap()).is_err());
    // A fresh sample at the original logical expiry cannot turn the fixed
    // positive plan into a new TTL. Reject and use the actual signed cancel.
    let context=frontend.context(binding,&rendered.origin,&full,&digest).await.unwrap();
    let mut expired=body;
    let result=signed_generation_body(&store,&mut dispatch,&mut expired,&KEY,&context,|_|async {
        let at=Instant::now();let wall=1_300_000u128*1_000_000;
        Ok(qualified_cpu_tracking_sample("Reference ID : C0000201\nSystem time : 0 seconds fast of NTP time\nRoot dispersion : 0 seconds\nRoot delay : 0 seconds\nLeap status : Normal\n",wall,wall,at,at).unwrap())
    }).await;
    assert!(matches!(result,Err(Error::Evidence("write endpoint expired after native reserve"))));
    wire.cancel_release.notify_one();
    frontend.cancel(&rendered.worker_url,&mut dispatch,&full,&digest,Instant::now()+Duration::from_secs(30)).await.unwrap();
    assert_eq!(wire.unconsumed_rank_refs.load(Ordering::SeqCst),0);
    assert_eq!(wire.gen_posts.load(Ordering::SeqCst),0);
    server.abort();let _=server.await;
}
