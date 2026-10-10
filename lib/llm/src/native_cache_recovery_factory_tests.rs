//! Synthetic localhost current-startup producers through the normal maintained
//! material/binding factory. No cache reservation, native generation or GPU.
use super::*;
use axum::{Json, Router, extract::State, routing::get};
use dynamo_llm::cache_authority::frontend_qualification::build_profile;
use serde_json::{Value, json};
use std::{os::unix::fs::PermissionsExt, sync::{Mutex, atomic::{AtomicUsize, Ordering}}};

const MODEL: &str = "qwen3.8-flash-next";
struct Producer {
    profile: Mutex<Value>,
    unavailable: AtomicUsize,
    reads: AtomicUsize,
    barrier: Option<Arc<tokio::sync::Barrier>>,
}
struct Worker {
    url: String,
    producer: Arc<Producer>,
    task: tokio::task::JoinHandle<()>,
    files: tempfile::TempDir,
}
impl Drop for Worker { fn drop(&mut self) { self.task.abort(); } }

fn profile(incarnation: &str) -> Value {
    let layout = json!({"rank_ids":[0,1],"kv_compat_descriptor":{"schema":"synthetic-CPU-not-qualified","dtype":"fp8"}});
    let parsed = build_profile("qwen", MODEL, incarnation, layout.clone()).unwrap();
    json!({"family":"qwen","native_origin_uuid":incarnation,"layout":layout,
        "source_digest":parsed.source_digest,"runtime_profile_digest":parsed.profile.runtime_profile_digest,
        "kv_compat_digest":parsed.kv_compat_digest})
}

async fn worker(barrier: Option<Arc<tokio::sync::Barrier>>) -> Worker {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let producer = Arc::new(Producer { profile:Mutex::new(profile("synthetic-first-boot")),
        unavailable:AtomicUsize::new(0),reads:AtomicUsize::new(0),barrier });
    let router = Router::new().route("/server_info", get({
        let url=url.clone();
        move |State(p):State<Arc<Producer>>| {
            let url=url.clone();
            async move {
                p.reads.fetch_add(1,Ordering::SeqCst);
                if p.unavailable.load(Ordering::SeqCst)!=0 {
                    return (axum::http::StatusCode::SERVICE_UNAVAILABLE,Json(json!({"error":"synthetic cache producer offline"})));
                }
                if let Some(barrier)=&p.barrier { barrier.wait().await; }
                let profile=p.profile.lock().unwrap().clone();
                let identity=json!({"ready":true,"errors":[],"result":{
                    "native_origin_uuid":profile["native_origin_uuid"],"native_origin":url,
                    "layout":profile["layout"],"profile":profile}});
                (axum::http::StatusCode::OK,Json(json!({"internal_states":[
                    {"prompt_cache_startup_identity":identity},{"prompt_cache_startup_identity":identity}]})))
            }
        }
    })).with_state(producer.clone());
    let task=tokio::spawn(async move { axum::serve(listener,router).await.unwrap(); });
    let files=tempfile::tempdir().unwrap();
    let key=files.path().join("key");
    std::fs::write(&key,[0x63;32]).unwrap();
    std::fs::set_permissions(&key,std::fs::Permissions::from_mode(0o400)).unwrap();
    std::fs::write(files.path().join("deployment.json"),serde_json::to_vec(&json!({
        "model":MODEL,"native_origin":url,"authority_secret_path":key})).unwrap()).unwrap();
    Worker {url,producer,task,files}
}
fn configuration(workers:&[Worker])->CacheAuthorityConfig {
    CacheAuthorityConfig {etcd_endpoints:vec![workers[0].url.clone()],workers:workers.iter().map(|w|CacheWorkerConfig {
        worker_url:w.url.clone(),deployment_path:w.files.path().join("deployment.json"),
        required_tiers:BTreeSet::from(["gpu".into()]),tenant:"CPU-isolated-tenant".into(),default_cache_salt:"CPU-isolated-salt".into(),
    }).collect()}
}
fn origins(workers:&[Worker])->BTreeSet<String>{workers.iter().map(|w|w.url.clone()).collect()}
fn client()->reqwest::Client { reqwest::Client::builder().no_proxy().redirect(reqwest::redirect::Policy::none()).retry(reqwest::retry::never()).build().unwrap() }

#[tokio::test]
async fn cache_recovery_factory_offline_then_current_incarnation_without_gateway_restart() {
    let mut workers=Vec::new(); for _ in 0..4 {workers.push(worker(None).await);}
    let config=configuration(&workers); let urls=origins(&workers); let client=client();
    workers[0].producer.unavailable.store(1,Ordering::SeqCst);
    assert!(load_cache_bindings(&client,&config,MODEL,&urls,"/v1/tokenize").await.is_err());
    workers[0].producer.unavailable.store(0,Ordering::SeqCst);
    *workers[0].producer.profile.lock().unwrap()=profile("synthetic-recovered-current-boot");
    let installed=load_cache_bindings(&client,&config,MODEL,&urls,"/v1/tokenize").await.unwrap();
    assert_eq!(installed.len(),4);
    let current=installed.iter().find(|b|b.worker.worker_url==workers[0].url).unwrap();
    assert_eq!(current.worker.native_origin_uuid,"synthetic-recovered-current-boot");
    assert_eq!(current.worker.rank_tier_clocks.keys().copied().collect::<BTreeSet<_>>(),BTreeSet::from([0,1]));
    assert_eq!((current.tenant.as_str(),current.default_cache_salt.as_str()),("CPU-isolated-tenant","CPU-isolated-salt"));
    // The retained request binding is immutable even when a later request sees
    // another native boot. No prior hold/TTL is renewed by discovery.
    *workers[0].producer.profile.lock().unwrap()=profile("synthetic-second-current-boot");
    let next=load_cache_bindings(&client,&config,MODEL,&urls,"/v1/tokenize").await.unwrap();
    assert_eq!(current.worker.native_origin_uuid,"synthetic-recovered-current-boot");
    assert_eq!(next.iter().find(|b|b.worker.worker_url==workers[0].url).unwrap().worker.native_origin_uuid,"synthetic-second-current-boot");
}

#[tokio::test]
async fn cache_recovery_factory_reads_all_four_current_origins_in_parallel() {
    let barrier=Arc::new(tokio::sync::Barrier::new(4));
    let mut workers=Vec::new(); for _ in 0..4 {workers.push(worker(Some(barrier.clone())).await);}
    let config=configuration(&workers); let urls=origins(&workers); let client=client();
    let installed=tokio::time::timeout(Duration::from_secs(3),load_cache_bindings(&client,&config,MODEL,&urls,"/v1/tokenize")).await.unwrap().unwrap();
    assert_eq!(installed.len(),4);
    assert!(workers.iter().all(|w|w.producer.reads.load(Ordering::SeqCst)==1));
}

#[tokio::test]
async fn cache_recovery_factory_rejects_bad_rank_profile_and_private_material() {
    let workers=vec![worker(None).await,worker(None).await];
    let config=configuration(&workers); let urls=origins(&workers); let client=client();
    workers[0].producer.profile.lock().unwrap()["runtime_profile_digest"]=json!("0".repeat(64));
    assert!(load_cache_bindings(&client,&config,MODEL,&urls,"/v1/tokenize").await.is_err());
    *workers[0].producer.profile.lock().unwrap()=profile("synthetic-fixed-profile");
    workers[0].producer.profile.lock().unwrap()["layout"]["rank_ids"]=json!([0,0]);
    assert!(load_cache_bindings(&client,&config,MODEL,&urls,"/v1/tokenize").await.is_err());
    *workers[0].producer.profile.lock().unwrap()=profile("synthetic-current-profile");
    std::fs::set_permissions(workers[0].files.path().join("key"),std::fs::Permissions::from_mode(0o644)).unwrap();
    assert!(load_cache_bindings(&client,&config,MODEL,&urls,"/v1/tokenize").await.is_err());
}

#[tokio::test]
async fn cache_recovery_factory_never_installs_partial_or_duplicate_origin_set() {
    let workers=vec![worker(None).await,worker(None).await];
    let mut config=configuration(&workers); let urls=origins(&workers); let client=client();
    config.workers.pop();
    assert!(load_cache_bindings(&client,&config,MODEL,&urls,"/v1/tokenize").await.is_err());
    let mut config=configuration(&workers); config.workers[1].worker_url=workers[0].url.clone();
    assert!(load_cache_bindings(&client,&config,MODEL,&urls,"/v1/tokenize").await.is_err());
    assert!(workers.iter().all(|w|w.producer.reads.load(Ordering::SeqCst)==0),"reject wrong configured coverage before any discovery");
}
