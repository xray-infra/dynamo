// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo HTTP frontend with one in-process native worker transport.
//! Usage: native-http-frontend <private-config.json>

use std::{collections::{BTreeMap,BTreeSet}, net::SocketAddr, path::PathBuf, sync::Arc, time::Duration};

use anyhow::{Context, Result, bail};
use dynamo_llm::{
    cache_authority::{
        EtcdStore,
        frontend_qualification::load_startup_material,
        native_cache_frontend::{FrontendWorkerBinding, NativeCacheFrontend},
    },
    engines::StreamingEngineAdapter,
    http::service::service_v2::{HttpService, State},
    model_card::ModelDeploymentCard,
    native_cleanup_cohort::{OrdinaryCleanupBindingFiles, load_ordinary_cleanup_bindings},
    native_contract::NativeTier,
    native_fencing::{EtcdFenceStore, EtcdPrefixRecords, FenceConfig, NativeFence},
    native_http_engine::{NativeHttpEngine, NativeHttpEngineConfig, NativeCacheFrontendFactory},
    native_management::NativeManagement,
    native_pd::{
        control::PdRole,
        dispatch::{PdDispatcher, native_error_fields_from_value},
        qualification::{PdRoleMaterialFiles, VerifiedPdPair, load_pd_role},
    },
};
use dynamo_runtime::{
    CancellationToken, engine::AsyncEngineContextProvider, pipeline::Context as EngineContext,
};
use serde::Deserialize;

#[cfg(test)]
#[path = "../native_cache_recovery_factory_tests.rs"]
mod native_cache_recovery_factory_tests;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    listen: SocketAddr,
    engine: NativeHttpEngineConfig,
    bearer_token_file: Option<PathBuf>,
    fencing: Option<FencingConfig>,
    cache_authority: Option<CacheAuthorityConfig>,
    #[serde(default)]
    management_socket: Option<PathBuf>,
    ordinary_cleanup: Vec<OrdinaryCleanupBindingFiles>,
    #[serde(default)]
    pd_pairs: Vec<PdPairConfig>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PdPairConfig {
    prefill_origin: String,
    decode_origin: String,
    prefill: PdRoleMaterialFiles,
    decode: PdRoleMaterialFiles,
}

/// Current native cache settings and the model authority key.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CacheAuthorityConfig {
    etcd_endpoints: Vec<String>,
    workers: Vec<CacheWorkerConfig>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CacheWorkerConfig {
    worker_url: String,
    deployment_path: PathBuf,
    required_tiers: BTreeSet<String>,
    tenant: String,
    default_cache_salt: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CacheStartupDiscovery {
    profile: Option<CacheRuntimeProfile>,
    qualified: bool,
}
#[derive(Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct CacheRuntimeProfile {
    family: String,
    native_origin_uuid: String,
    layout: serde_json::Value,
    source_digest: String,
    runtime_profile_digest: String,
    kv_compat_digest: String,
}

#[derive(Deserialize)]
struct QwenServerInfo {
    internal_states: Vec<QwenCacheState>,
}
#[derive(Deserialize)]
struct QwenCacheState {
    prompt_cache_startup_identity: QwenCacheIdentity,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct QwenCacheIdentity {
    ready: bool,
    errors: Vec<QwenCacheStartupError>,
    result: Option<QwenCacheStartup>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct QwenCacheStartupError {
    rank_id: u32,
    #[serde(rename = "type")]
    error_type: String,
    message: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct QwenCacheStartup {
    native_origin_uuid: String,
    native_origin: String,
    layout: serde_json::Value,
    profile: CacheRuntimeProfile,
}

fn cache_startup_failure(ready: bool, material_present: bool, errors: Option<&[QwenCacheStartupError]>) -> String {
    let reasons = errors.map(|errors| {
        errors.iter().take(4).filter_map(|error| {
            let mut fields = native_error_fields_from_value(&serde_json::json!({
                "type": error.error_type,
                "message": error.message,
            }))?;
            fields.as_object_mut()?.insert("rank_id".into(), error.rank_id.into());
            Some(fields)
        }).collect::<Vec<_>>()
    });
    format!("native cache producer is not ready: {}", serde_json::json!({
        "ready": ready,
        "material_present": material_present,
        "error_count": errors.map(|errors| errors.len()),
        "reasons": reasons,
    }))
}

async fn load_cache_bindings(
    client: &reqwest::Client,
    cache: &CacheAuthorityConfig,
    model: &str,
    worker_urls: &BTreeSet<String>,
    count_path: &str,
) -> Result<Vec<FrontendWorkerBinding>> {
    if cache.etcd_endpoints.is_empty() || cache.etcd_endpoints.len() > 32 {
        bail!("cache authority requires bounded explicit etcd endpoints");
    }
    for endpoint in &cache.etcd_endpoints {
        let endpoint = reqwest::Url::parse(endpoint).context("invalid cache etcd endpoint")?;
        if !matches!(endpoint.scheme(), "http" | "https")
            || endpoint.host_str().is_none()
            || !endpoint.username().is_empty()
            || endpoint.password().is_some()
            || endpoint.path() != "/"
            || endpoint.query().is_some()
            || endpoint.fragment().is_some()
        {
            bail!("cache etcd endpoint must be an HTTP origin without credentials");
        }
    }
    let mut remaining = worker_urls.clone();
    for worker in &cache.workers {
        if !remaining.remove(&worker.worker_url) {
            bail!("cache qualification has a duplicate or foreign native worker");
        }
    }
    if !remaining.is_empty() || cache.workers.is_empty() {
        bail!("cache qualification must cover every configured native worker");
    }
    // Independent read-only current identities qualify in parallel. No partial
    // set is installed; offline origins remain a concrete cache 503.
    let futures = cache.workers.iter().map(|worker| async move {
        if worker.tenant.is_empty() || worker.default_cache_salt.is_empty() {
            bail!("cache qualification requires explicit tenant and default cache salt");
        }
        let endpoint = format!(
            "{}{}",
            worker.worker_url.trim_end_matches('/'),
            if model == "qwen3.8-flash-next" {
                "/server_info"
            } else {
                "/v1/requests/cache-startup"
            }
        );
        let mut response = client.get(&endpoint).timeout(Duration::from_secs(10)).send().await?;
        if response.status() != reqwest::StatusCode::OK || response.url().as_str() != endpoint {
            bail!("native cache startup discovery HTTP {}", response.status());
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response.chunk().await? {
            if bytes.len() + chunk.len() > 16 * 1024 * 1024 {
                bail!("native cache startup discovery exceeds limit");
            }
            bytes.extend_from_slice(&chunk);
        }
        let profile = if model == "qwen3.8-flash-next" {
            let info: QwenServerInfo = serde_json::from_slice(&bytes)?;
            if info.internal_states.is_empty() || info.internal_states.len() > 32 {
                bail!("native cache runtime states missing or exceed limit");
            }
            let mut actual_profile = None;
            for state in info.internal_states {
                let identity = state.prompt_cache_startup_identity;
                if !identity.ready || !identity.errors.is_empty() {
                    bail!("{}", cache_startup_failure(identity.ready, identity.result.is_some(), Some(&identity.errors)));
                }
                let result = identity
                    .result
                    .context("native cache startup identity missing")?;
                if result.native_origin.trim_end_matches('/')
                    != worker.worker_url.trim_end_matches('/')
                    || result.native_origin_uuid != result.profile.native_origin_uuid
                    || result.layout != result.profile.layout
                    || actual_profile
                        .as_ref()
                        .is_some_and(|profile| profile != &result.profile)
                {
                    bail!("native cache runtime state identity or layout differs");
                }
                actual_profile = Some(result.profile);
            }
            actual_profile.context("native cache runtime profile missing")?
        } else {
            let discovery: CacheStartupDiscovery = serde_json::from_slice(&bytes)?;
            if !discovery.qualified {
                bail!("{}", cache_startup_failure(discovery.qualified, discovery.profile.is_some(), None));
            }
            discovery
                .profile
                .context("native cache runtime profile missing")?
        };
        let material = load_startup_material(
            &worker.deployment_path,
            model,
            &worker.worker_url,
            &profile.family,
            &profile.native_origin_uuid,
            profile.layout,
        )
        .map_err(|reason| anyhow::anyhow!("cache runtime material invalid: {reason}"))?;
        if material.profile.source_digest != profile.source_digest
            || material.profile.profile.runtime_profile_digest != profile.runtime_profile_digest
            || material.profile.kv_compat_digest != profile.kv_compat_digest
        {
            bail!("native cache runtime profile digests mismatch");
        }
        let binding = FrontendWorkerBinding::from_startup(
            material,
            profile.native_origin_uuid,
            worker.required_tiers.clone(),
            count_path.into(),
            worker.tenant.clone(),
            worker.default_cache_salt.clone(),
        )?;
        Ok::<_,anyhow::Error>(binding)
    });
    futures::future::try_join_all(futures).await
}

struct CurrentCacheFactory {
    config: CacheAuthorityConfig,
    model: String,
    worker_urls: BTreeSet<String>,
    count_path: String,
}

#[async_trait::async_trait]
impl NativeCacheFrontendFactory for CurrentCacheFactory {
    async fn create(
        &self,
        client: reqwest::Client,
        registry: Arc<dynamo_llm::native_cleanup::NativeCleanupRegistry>,
    ) -> Result<Arc<NativeCacheFrontend>, dynamo_runtime::pipeline::Error> {
        let qualification: Result<Arc<NativeCacheFrontend>> = async {
            let bindings = load_cache_bindings(&client, &self.config, &self.model,
                &self.worker_urls, &self.count_path).await?;
            dynamo_llm::cache_authority::frontend_clock::ChronyClock.qualify().await
                .context("cache authority requires a qualified synchronized frontend clock")?;
            let store = EtcdStore::connect(self.config.etcd_endpoints.clone()).await?;
            Ok(Arc::new(NativeCacheFrontend::new(client, store, bindings)?
                .with_cleanup_registry(registry)))
        }.await;
        qualification.map_err(|error| {
            let fields = native_error_fields_from_value(&serde_json::json!({"message":error.to_string()}))
                .unwrap_or_else(|| serde_json::json!({"message":"current native cache qualification unavailable"}));
            tracing::warn!(cause = %fields, "current native cache factory unavailable");
            dynamo_runtime::error::DynamoError::builder()
                .error_type(dynamo_runtime::error::ErrorType::Unavailable)
                .message(format!("current native cache qualification unavailable: {fields}"))
                .build().into()
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FencingConfig {
    etcd_endpoints: Vec<String>,
    election_key: String,
    holder: String,
    max_effective_load: usize,
}

async fn load_prefix_renderer_metadata(worker_urls:&BTreeSet<String>)->Result<BTreeMap<String,serde_json::Value>> {
    let client=reqwest::Client::builder().no_proxy().redirect(reqwest::redirect::Policy::none())
        .retry(reqwest::retry::never()).timeout(Duration::from_secs(5)).build()?;
    let mut result=BTreeMap::new();
    for origin in worker_urls {
        let endpoint=format!("{}/v1/requests/cleanup-profile",origin.trim_end_matches('/'));
        let mut response=client.get(&endpoint).send().await?;
        if response.status()!=reqwest::StatusCode::OK || response.url().as_str()!=endpoint {
            bail!("native loaded renderer metadata origin/status mismatch");
        }
        let mut bytes=Vec::new();
        while let Some(chunk)=response.chunk().await? {
            if bytes.len().saturating_add(chunk.len())>16*1024*1024 {bail!("native loaded renderer metadata exceeds bound");}
            bytes.extend_from_slice(&chunk);
        }
        let key=origin.trim_end_matches('/').to_owned();
        if result.insert(key,serde_json::from_slice(&bytes)?).is_some() {bail!("duplicate native renderer origin");}
    }
    Ok(result)
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {},
            _ = term.recv() => {},
        }
    }
    #[cfg(not(unix))]
    let _ = tokio::signal::ctrl_c().await;
}

async fn drain_native_service(
    state: &State,
    management: &NativeManagement,
    cleanup: &dynamo_llm::native_cleanup::NativeCleanupRegistry,
    cancel: &CancellationToken,
) {
    state.start_draining();
    management.seal();
    management.close_idle();
    state.wait_inflight_zero().await;
    management.wait_http_bodies_zero().await;
    // Body EOF/disconnect is not a native release ACK. Keep the process and
    // fencing authority alive while physical cleanup is running or unknown.
    cleanup.wait_for_physical_zero().await;
    cancel.cancel();
}

#[tokio::main]
async fn main() -> Result<()> {
    dynamo_runtime::logging::init();
    let mut args = std::env::args_os().skip(1);
    let path = args
        .next()
        .context("Usage: native-http-frontend <private-config.json>")?;
    if args.next().is_some() {
        bail!("Usage: native-http-frontend <private-config.json>");
    }
    let config: Config = serde_json::from_slice(&std::fs::read(path)?)
        .context("invalid native HTTP frontend config")?;
    if config.engine.qualification_tier.is_some() && !config.listen.ip().is_loopback() {
        bail!("single-tier qualification is restricted to a loopback candidate listener");
    }
    if !config.listen.ip().is_loopback() && config.management_socket.is_none() {
        bail!("production native frontend requires a private management socket");
    }
    let bearer_token = match config.bearer_token_file {
        Some(path) => {
            let token = std::fs::read_to_string(path).context("read private bearer token file")?;
            let token = token.trim();
            if token.is_empty() || token.contains(['\r', '\n']) {
                bail!("bearer token must be a nonempty single line");
            }
            Some(Arc::<str>::from(token))
        }
        // The four existing public endpoints do not require inbound credentials.
        // A backend admission token is never promoted to public authentication.
        None => None,
    };

    let model = config.engine.model.clone();
    let ds = model == "deepseek-ai/DeepSeek-V4.1-Flash";
    if ds != config.fencing.is_some() {
        bail!("DeepSeek requires its shared authority configuration; other models do not use it");
    }
    if !config.listen.ip().is_loopback()
        && config
            .engine
            .workers
            .iter()
            .any(|worker| worker.max_input_tokens.is_none() || worker.capability_source.is_none())
    {
        bail!("every production worker requires a verified native input boundary and provenance");
    }
    let short_workers = config
        .engine
        .workers
        .iter()
        .filter(|worker| worker.tier == NativeTier::Short)
        .map(|worker| worker.url.clone())
        .collect();
    let worker_urls: BTreeSet<String> = config
        .engine
        .workers
        .iter()
        .map(|worker| worker.url.clone())
        .collect();
    let mut engine = NativeHttpEngine::new(config.engine)
        .map_err(|error| anyhow::anyhow!("invalid native worker configuration: {error}"))?;
    // Current native role and cleanup discovery precedes generation admission.
    let mut pd_roles = Vec::new();
    let mut allowed_origins = worker_urls
        .iter()
        .map(|url| url.trim_end_matches('/').to_owned())
        .collect::<BTreeSet<_>>();
    for pair in &config.pd_pairs {
        let prefill = load_pd_role(&pair.prefill, &model, &pair.prefill_origin, PdRole::Prefill)
            .await
            .map_err(|error| anyhow::anyhow!("invalid prefill role material: {error}"))?;
        let decode = load_pd_role(&pair.decode, &model, &pair.decode_origin, PdRole::Decode)
            .await
            .map_err(|error| anyhow::anyhow!("invalid decode role material: {error}"))?;
        let roles = VerifiedPdPair::from_roles(prefill, decode)
            .map_err(|error| anyhow::anyhow!("incompatible qualified P/D pair: {error}"))?;
        if !allowed_origins.contains(pair.decode_origin.trim_end_matches('/')) {
            bail!("P/D decode role must be an existing configured worker");
        }
        allowed_origins.insert(pair.prefill_origin.trim_end_matches('/').to_owned());
        pd_roles.push(Arc::new(roles));
    }
    let bindings = load_ordinary_cleanup_bindings(&config.ordinary_cleanup)
    .map_err(|error| anyhow::anyhow!("ordinary native cleanup qualification invalid: {error}"))?;
    if bindings.keys().cloned().collect::<BTreeSet<_>>() != allowed_origins {
        bail!(
            "ordinary private material must cover exactly configured workers and declared prefill roles"
        );
    }
    engine = engine.with_request_lifecycle_materials(
        worker_urls
            .iter()
            .map(|url| bindings[url.trim_end_matches('/')].clone())
            .collect(),
    )?;
    if ds {
        engine=engine.with_prefix_renderer_metadata(load_prefix_renderer_metadata(&worker_urls).await?)?;
    }
    // An explicitly configured P/D pair still requires its actual qualified
    // roles/cohorts. Ordinary origins qualify only when selected, so one offline
    // sibling cannot prevent a Qwen frontend from serving existing healthy ones.
    let mut pd_bindings = BTreeMap::new();
    for roles in &pd_roles {
        for role in [roles.prefill(), roles.decode()] {
            let origin = role.artifact().native_origin.trim_end_matches('/');
            if !pd_bindings.contains_key(origin) {
                let current = bindings[origin].qualify_current(
                    &engine.native_http_client(),
                    Duration::from_secs(60),
                    (*engine.cleanup_registry()).clone(),
                ).await.map_err(|error| anyhow::anyhow!("P/D ordinary cleanup invalid: {error}"))?;
                pd_bindings.insert(origin.to_owned(), current);
            }
        }
    }
    let dispatchers = pd_roles
        .into_iter()
        .map(|roles| {
            let prefill = pd_bindings[roles
                .prefill()
                .artifact()
                .native_origin
                .trim_end_matches('/')]
            .clone();
            let decode = pd_bindings[roles
                .decode()
                .artifact()
                .native_origin
                .trim_end_matches('/')]
            .clone();
            PdDispatcher::new(roles, prefill, decode)
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    engine = engine.with_pd_dispatchers(dispatchers)?;
    // P owns the cache lease; D owns only its ordinary and transfer lifecycle.
    let cache_worker_urls = worker_urls
        .iter()
        .map(|url| {
            config
                .pd_pairs
                .iter()
                .find(|pair| pair.decode_origin.trim_end_matches('/') == url.trim_end_matches('/'))
                .map(|pair| pair.prefill_origin.clone())
                .unwrap_or_else(|| url.clone())
        })
        .collect::<BTreeSet<_>>();
    if let Some(cache) = config.cache_authority {
        // The configured normal factory is request work. Ordinary startup and
        // generation never wait for an offline cache origin. Explicit requests
        // bind a complete actual current set, so native recovery needs no G restart.
        let count_path = engine.contract().count_path().to_owned();
        engine = engine.with_cache_frontend_factory(Arc::new(CurrentCacheFactory {
            config: cache, model: model.clone(), worker_urls: cache_worker_urls, count_path,
        }));
    }
    let fence = match config.fencing {
        Some(config) => {
            if config.etcd_endpoints.is_empty() {
                bail!("shared authority requires etcd endpoints");
            }
            let prefix_records = EtcdPrefixRecords::connect(config.etcd_endpoints.clone()).await?;
            let store = EtcdFenceStore::connect(config.etcd_endpoints).await?;
            let fence = NativeFence::new(
                FenceConfig {
                    election_key: config.election_key,
                    holder: config.holder,
                    short_workers,
                    max_effective_load: config.max_effective_load,
                },
                store,
                prefix_records,
            )?;
            // A standby remains live but generation fails closed until the lease
            // is acquired. The existing leader is never deleted for takeover.
            fence.start();
            engine = engine.with_fence(fence.clone());
            Some(fence)
        }
        None => None,
    };
    let engine = Arc::new(engine);
    let cleanup = engine.cleanup_registry();
    let cancel = CancellationToken::new();
    let load_observer = if fence.is_some() {
        let engine = engine.clone();
        let cancel = cancel.clone();
        Some(tokio::spawn(async move {
            let context = EngineContext::new(()).context();
            loop {
                tokio::select! {
                    _ = cancel.cancelled() => break,
                    _ = engine.refresh_short_loads(&context) => {},
                }
                tokio::select! {
                    _ = cancel.cancelled() => break,
                    _ = tokio::time::sleep(Duration::from_millis(500)) => {},
                }
            }
        }))
    } else {
        None
    };
    let service = HttpService::builder()
        .host(config.listen.ip().to_string())
        .port(config.listen.port())
        .enable_chat_endpoints(true)
        .enable_cmpl_endpoints(false)
        .enable_embeddings_endpoints(false)
        .enable_responses_endpoints(true)
        .enable_anthropic_endpoints(true)
        .enable_admin_api(false)
        .native_http_engine(Some(engine.clone()))
        .native_http_bearer_token(bearer_token)
        .cancel_token(Some(cancel.clone()))
        .build()?;
    let card = ModelDeploymentCard::with_name_only(&model);
    service.model_manager().add_chat_completions_model(
        &model,
        card.mdcsum(),
        Arc::new(StreamingEngineAdapter::new(engine)),
    )?;
    let state = service.state_clone();
    let management = Arc::new(NativeManagement::new(state.clone(), cleanup.clone()));
    let control_cancel = CancellationToken::new();
    let control = config
        .management_socket
        .as_ref()
        .map(|path| management.spawn_unix_control(path, control_cancel.clone()))
        .transpose()?;
    let listener = tokio::net::TcpListener::bind(config.listen).await?;
    let mut join = service
        .spawn_native_with_listener(cancel.clone(), listener, management.clone())
        .await?;
    eprintln!(
        "Dynamo native HTTP frontend listening on {} for {}",
        config.listen, model
    );
    tokio::select! {
        outcome = &mut join => {
            drain_native_service(&state, &management, &cleanup, &cancel).await;
            outcome??;
        },
        _ = shutdown_signal() => {
            drain_native_service(&state, &management, &cleanup, &cancel).await;
            join.await??;
        },
    }
    if let Some(fence) = fence {
        fence.stop();
    }
    control_cancel.cancel();
    if let Some(control) = control {
        control.await??;
    }
    cancel.cancel();
    if let Some(observer) = load_observer {
        observer.await?;
    }
    Ok(())
}

#[cfg(test)]
mod qwen_cache_startup_tests {
    use super::*;

    #[test]
    fn qwen_cache_startup_current_rank_errors_preserve_filtered_cause() {
        // Current /get_server_info identity, saved raw SHA256:
        // 4ecf22f8588aa16f39a765f17fddbc3498dc092453c7f74e2527326ba8f16506.
        let errors = (0..4).map(|rank_id| serde_json::json!({
            "rank_id": rank_id,
            "type": "OSError",
            "message": "[Errno 70] [Errno 70] Communication error on send: '/mnt/prompt-cache-client/beegfs/sglang-qwen-production-cache-ab2aaa59de004ac5ab92e7d94c83ac7b': '/mnt/prompt-cache-client/beegfs/sglang-qwen-production-cache-ab2aaa59de004ac5ab92e7d94c83ac7b'",
        })).collect::<Vec<_>>();
        let identity: QwenCacheIdentity = serde_json::from_value(serde_json::json!({
            "ready": false, "result": null, "errors": errors,
        })).unwrap();
        assert!(!identity.ready);
        assert!(identity.result.is_none());
        let reason = cache_startup_failure(identity.ready, identity.result.is_some(), Some(&identity.errors));
        let details: serde_json::Value = serde_json::from_str(
            reason.strip_prefix("native cache producer is not ready: ").unwrap()
        ).unwrap();
        assert_eq!(details["error_count"], 4);
        assert_eq!(details["reasons"].as_array().unwrap().len(), 4);
        for rank in 0..4 {
            assert_eq!(details["reasons"][rank]["rank_id"], rank);
            assert_eq!(details["reasons"][rank]["type"], "OSError");
            let message = details["reasons"][rank]["message"].as_str().unwrap();
            assert!(message.contains("Errno 70"));
            assert!(!message.contains("ab2aaa59de004ac5ab92e7d94c83ac7b"));
        }
    }

    #[test]
    fn qwen_cache_startup_rejects_obsolete_string_and_unknown_error_fields() {
        for errors in [
            serde_json::json!(["obsolete error text"]),
            serde_json::json!([{"rank_id": 0, "type": "OSError", "message": "Errno 70", "secret": "x"}]),
        ] {
            assert!(serde_json::from_value::<QwenCacheIdentity>(serde_json::json!({
                "ready": false, "result": null, "errors": errors,
            })).is_err());
        }
    }
}

#[cfg(test)]
mod native_drain_tests {
    use super::*;

    #[tokio::test(start_paused = true)]
    async fn native_sigterm_drain_keeps_admitted_body_alive_beyond_sixty_seconds() {
        let service = HttpService::builder().build().unwrap();
        let state = service.state_clone();
        let first_body = state.acquire_inflight();
        let last_body = state.acquire_inflight();
        let cancel = CancellationToken::new();
        let draining = {
            let state = state.clone();
            let cancel = cancel.clone();
            tokio::spawn(async move {
                let cleanup = dynamo_llm::native_cleanup::NativeCleanupRegistry::new();
                let management = NativeManagement::new(state.clone(), Arc::new(cleanup.clone()));
                drain_native_service(&state, &management, &cleanup, &cancel).await
            })
        };
        tokio::task::yield_now().await;
        assert!(!state.is_ready());
        tokio::time::advance(Duration::from_secs(61)).await;
        assert!(!cancel.is_cancelled());
        assert!(!draining.is_finished());
        assert_eq!(state.inflight_count(), 2);
        // Both normal EOF and client disconnect release this actual body guard.
        drop(first_body);
        tokio::task::yield_now().await;
        assert!(!cancel.is_cancelled());
        drop(last_body);
        draining.await.unwrap();
        assert!(cancel.is_cancelled());
        assert_eq!(state.inflight_count(), 0);
    }
}
