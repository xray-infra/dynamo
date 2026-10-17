// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Run the upstream Dynamo HTTP frontend against existing OpenAI HTTP workers.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, ensure};
use dynamo_llm::engines::StreamingEngineAdapter;
use dynamo_llm::frontend_config::FrontendApiConfig;
use dynamo_llm::http::service::service_v2::HttpService;
use dynamo_llm::openai_http_engine::{OpenAiHttpEngine, PdPrefillOrigin};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    listen: SocketAddr,
    model: String,
    /// Independent workers. Exactly one of `workers` and `pd` is configured.
    #[serde(default)]
    workers: Vec<String>,
    /// SGLang PD disaggregation: each request runs on one prefill and one decode.
    #[serde(default)]
    pd: Option<PdConfig>,
    /// Anthropic block-level `cache_control`: `reject` (default) because HTTP
    /// workers cannot place exact boundaries or TTLs, or `automatic` to accept
    /// valid directives as hints for ordinary automatic prefix caching.
    #[serde(default)]
    anthropic_cache_control: CacheControlMode,
}

#[derive(Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum CacheControlMode {
    #[default]
    Reject,
    Automatic,
}

fn frontend_api(mode: CacheControlMode) -> FrontendApiConfig {
    let mut api = FrontendApiConfig::default();
    api.anthropic_mut().set_enabled(true);
    api.anthropic_mut()
        .set_cache_control_automatic(matches!(mode, CacheControlMode::Automatic));
    api
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PdConfig {
    prefills: Vec<PrefillConfig>,
    decodes: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PrefillConfig {
    origin: String,
    bootstrap_port: u16,
}

fn build_engine(config: Config) -> anyhow::Result<OpenAiHttpEngine> {
    match config.pd {
        Some(pd) => {
            ensure!(config.workers.is_empty(), "configure either workers or pd, not both");
            OpenAiHttpEngine::new_pd(
                pd.prefills
                    .into_iter()
                    .map(|p| PdPrefillOrigin {
                        origin: p.origin,
                        bootstrap_port: p.bootstrap_port,
                    })
                    .collect(),
                pd.decodes,
            )
        }
        None => OpenAiHttpEngine::new(config.workers),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dynamo_runtime::logging::init();
    let mut arguments = std::env::args_os().skip(1);
    let path = arguments
        .next()
        .context("usage: openai-http-frontend <config.json>")?;
    ensure!(arguments.next().is_none(), "unexpected command arguments");
    let bytes = std::fs::read(path)?;
    let config: Config = serde_json::from_slice(&bytes)?;
    ensure!(!config.model.is_empty(), "model cannot be empty");
    let (model, listen) = (config.model.clone(), config.listen);
    let api = frontend_api(config.anthropic_cache_control);
    let engine = Arc::new(StreamingEngineAdapter::new(Arc::new(build_engine(config)?)));
    let service = HttpService::builder()
        .frontend_api_config(api)
        .enable_chat_endpoints(true)
        .enable_admin_api(false)
        .host(listen.ip().to_string())
        .port(listen.port())
        .build()?;
    // This is the upstream local text-engine registration, not a discovered
    // worker card or a fabricated worker capability/TTL authority.
    service.model_manager().add_chat_completions_model(
        &model,
        &format!("{:x}", Sha256::digest(&bytes)),
        engine,
    )?;
    let cancel = CancellationToken::new();
    let join = service.spawn(cancel.clone()).await;
    #[cfg(unix)]
    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    #[cfg(unix)]
    tokio::select! {
        _ = term.recv() => {},
        result = tokio::signal::ctrl_c() => { result?; },
    }
    #[cfg(not(unix))]
    tokio::signal::ctrl_c().await?;
    service.state().start_draining();
    while !service
        .state()
        .wait_inflight_zero_or_timeout(Duration::from_secs(15))
        .await
    {}
    cancel.cancel();
    join.await??;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_llm::http::service::axum::{Router, routing::post};
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const WORKER_RESPONSE: &str = concat!(
        "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"ok\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test-model\",\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n",
        "data: [DONE]\n\n",
    );

    #[tokio::test]
    async fn anthropic_cache_control_is_rejected_before_http_worker_dispatch() {
        let dispatches = Arc::new(AtomicUsize::new(0));
        let worker_dispatches = dispatches.clone();
        let worker = Router::new().route(
            "/v1/chat/completions",
            post(move || {
                let worker_dispatches = worker_dispatches.clone();
                async move {
                    worker_dispatches.fetch_add(1, Ordering::SeqCst);
                    ([("content-type", "text/event-stream")], WORKER_RESPONSE)
                }
            }),
        );
        let worker_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let worker_address = worker_listener.local_addr().unwrap();
        let worker_task = tokio::spawn(async move {
            dynamo_llm::http::service::axum::serve(worker_listener, worker)
                .await
                .unwrap();
        });

        let engine = Arc::new(StreamingEngineAdapter::new(Arc::new(
            OpenAiHttpEngine::new(vec![format!("http://{worker_address}")]).unwrap(),
        )));
        let mut api = FrontendApiConfig::default();
        api.anthropic_mut().set_enabled(true);
        api.anthropic_mut().set_strip_preamble(false);
        let service = HttpService::builder()
            .frontend_api_config(api)
            .enable_chat_endpoints(true)
            .enable_admin_api(false)
            .host("127.0.0.1".to_string())
            .port(0)
            .build()
            .unwrap();
        service
            .model_manager()
            .add_chat_completions_model("test-model", "test-checksum", engine)
            .unwrap();
        let frontend_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let frontend_address = frontend_listener.local_addr().unwrap();
        let cancel = CancellationToken::new();
        let frontend_task = service
            .spawn_with_listener(cancel.clone(), frontend_listener)
            .await;

        let client = reqwest::Client::new();
        let messages_url = format!("http://{frontend_address}/v1/messages");
        let ordinary = client
            .post(&messages_url)
            .json(&json!({
                "model": "test-model",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(ordinary.status(), reqwest::StatusCode::OK);
        assert_eq!(dispatches.load(Ordering::SeqCst), 1);

        let unsupported = client
            .post(messages_url)
            .json(&json!({
                "model": "test-model",
                "max_tokens": 8,
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": "hi",
                        "cache_control": {"type": "ephemeral"}
                    }]
                }]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(unsupported.status(), reqwest::StatusCode::BAD_REQUEST);
        let body: serde_json::Value = unsupported.json().await.unwrap();
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(
            body["error"]["message"],
            "Anthropic cache_control is not supported by the OpenAI HTTP worker transport"
        );
        assert_eq!(dispatches.load(Ordering::SeqCst), 1);

        cancel.cancel();
        frontend_task.await.unwrap().unwrap();
        worker_task.abort();
    }

    type Seen = Arc<std::sync::Mutex<Vec<serde_json::Value>>>;

    async fn serve(router: Router) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            dynamo_llm::http::service::axum::serve(listener, router)
                .await
                .unwrap();
        });
        (format!("http://{address}"), task)
    }

    async fn pd_frontend(
        prefill_status: u16,
        decode_delay: Duration,
    ) -> (String, Seen, Seen, CancellationToken) {
        use dynamo_llm::http::service::axum::Json;
        let (prefill_seen, decode_seen): (Seen, Seen) = Default::default();
        let seen = prefill_seen.clone();
        let prefill = Router::new().route(
            "/v1/chat/completions",
            post(move |Json(body): Json<serde_json::Value>| {
                let seen = seen.clone();
                async move {
                    seen.lock().unwrap().push(body);
                    (
                        reqwest::StatusCode::from_u16(prefill_status).unwrap(),
                        "{}",
                    )
                }
            }),
        );
        let seen = decode_seen.clone();
        let decode = Router::new().route(
            "/v1/chat/completions",
            post(move |Json(body): Json<serde_json::Value>| {
                let seen = seen.clone();
                async move {
                    seen.lock().unwrap().push(body);
                    tokio::time::sleep(decode_delay).await;
                    ([("content-type", "text/event-stream")], WORKER_RESPONSE)
                }
            }),
        );
        let (prefill_origin, _) = serve(prefill).await;
        let (decode_origin, _) = serve(decode).await;
        let config: Config = serde_json::from_value(json!({
            "listen": "127.0.0.1:0", "model": "test-model",
            "pd": {"prefills": [{"origin": format!("{prefill_origin}/"), "bootstrap_port": 8998}],
                   "decodes": [format!("{decode_origin}/")]}
        }))
        .unwrap();
        let engine = Arc::new(StreamingEngineAdapter::new(Arc::new(
            build_engine(config).unwrap(),
        )));
        let service = HttpService::builder()
            .enable_chat_endpoints(true)
            .enable_admin_api(false)
            .host("127.0.0.1".to_string())
            .port(0)
            .build()
            .unwrap();
        service
            .model_manager()
            .add_chat_completions_model("test-model", "test-checksum", engine)
            .unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let cancel = CancellationToken::new();
        service.spawn_with_listener(cancel.clone(), listener).await;
        (format!("http://{address}"), prefill_seen, decode_seen, cancel)
    }

    fn chat() -> serde_json::Value {
        json!({"model": "test-model", "max_tokens": 8,
               "messages": [{"role": "user", "content": "hi"}]})
    }

    #[tokio::test]
    async fn pd_request_shares_one_bootstrap_room_between_prefill_and_decode() {
        let (frontend, prefill, decode, cancel) = pd_frontend(200, Duration::ZERO).await;
        let response = reqwest::Client::new()
            .post(format!("{frontend}/v1/chat/completions"))
            .json(&chat())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let body: serde_json::Value = response.json().await.unwrap();
        assert_eq!(body["choices"][0]["message"]["content"], "ok");

        let (p, d) = (prefill.lock().unwrap()[0].clone(), decode.lock().unwrap()[0].clone());
        assert_eq!(p["bootstrap_host"], "127.0.0.1");
        assert_eq!(p["bootstrap_port"], 8998);
        assert!(p["bootstrap_room"].as_u64().is_some());
        for key in ["bootstrap_host", "bootstrap_port", "bootstrap_room", "messages"] {
            assert_eq!(p[key], d[key], "{key}");
        }
        assert_eq!(p["stream"], false);
        assert_eq!(d["stream"], true);
        cancel.cancel();
    }

    #[tokio::test]
    async fn pd_prefill_failure_fails_the_request_without_waiting_for_decode() {
        let (frontend, prefill, _, cancel) = pd_frontend(503, Duration::from_secs(60)).await;
        let response = tokio::time::timeout(
            Duration::from_secs(10),
            reqwest::Client::new()
                .post(format!("{frontend}/v1/chat/completions"))
                .json(&chat())
                .send(),
        )
        .await
        .expect("prefill failure must not wait for decode")
        .unwrap();
        assert!(!response.status().is_success());
        assert_eq!(prefill.lock().unwrap().len(), 1);
        cancel.cancel();
    }

    #[tokio::test]
    async fn automatic_cache_control_reaches_the_worker_and_malformed_is_rejected() {
        let dispatches = Arc::new(AtomicUsize::new(0));
        let worker_dispatches = dispatches.clone();
        let worker = Router::new().route(
            "/v1/chat/completions",
            post(move || {
                let worker_dispatches = worker_dispatches.clone();
                async move {
                    worker_dispatches.fetch_add(1, Ordering::SeqCst);
                    ([("content-type", "text/event-stream")], WORKER_RESPONSE)
                }
            }),
        );
        let (worker_origin, _) = serve(worker).await;
        let config: Config = serde_json::from_value(json!({
            "listen": "127.0.0.1:0", "model": "test-model",
            "workers": [format!("{worker_origin}/")],
            "anthropic_cache_control": "automatic"
        }))
        .unwrap();
        // Default preamble stripping stays on: automatic mode must not trip it.
        let api = frontend_api(config.anthropic_cache_control);
        let engine = Arc::new(StreamingEngineAdapter::new(Arc::new(
            build_engine(config).unwrap(),
        )));
        let service = HttpService::builder()
            .frontend_api_config(api)
            .enable_chat_endpoints(true)
            .enable_admin_api(false)
            .host("127.0.0.1".to_string())
            .port(0)
            .build()
            .unwrap();
        service
            .model_manager()
            .add_chat_completions_model("test-model", "test-checksum", engine)
            .unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let cancel = CancellationToken::new();
        service.spawn_with_listener(cancel.clone(), listener).await;
        let client = reqwest::Client::new();
        let url = format!("http://{address}/v1/messages");
        let block = |cache_control: serde_json::Value| json!({
            "model": "test-model", "max_tokens": 8,
            "system": [{"type": "text", "text": "sys", "cache_control": cache_control.clone()}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi",
                "cache_control": cache_control}]}]
        });
        let ok = client.post(&url).json(&block(json!({"type": "ephemeral"})))
            .send().await.unwrap();
        assert_eq!(ok.status(), reqwest::StatusCode::OK);
        assert_eq!(dispatches.load(Ordering::SeqCst), 1);
        let ttl = client.post(&url).json(&block(json!({"type": "ephemeral", "ttl": "1h"})))
            .send().await.unwrap();
        assert_eq!(ttl.status(), reqwest::StatusCode::OK);
        assert_eq!(dispatches.load(Ordering::SeqCst), 2);
        let malformed = client.post(&url).json(&block(json!({"type": "persistent"})))
            .send().await.unwrap();
        assert_eq!(malformed.status(), reqwest::StatusCode::BAD_REQUEST);
        assert_eq!(dispatches.load(Ordering::SeqCst), 2);
        cancel.cancel();
    }

    #[tokio::test]
    async fn pd_multi_turn_session_stays_on_one_prefill_through_http() {
        use dynamo_llm::http::service::axum::Json;
        let hits: Arc<std::sync::Mutex<Vec<usize>>> = Default::default();
        let mut prefills = Vec::new();
        for i in 0..8 {
            let hits = hits.clone();
            let router = Router::new().route(
                "/v1/chat/completions",
                post(move |Json(_): Json<serde_json::Value>| {
                    let hits = hits.clone();
                    async move {
                        hits.lock().unwrap().push(i);
                        "{}"
                    }
                }),
            );
            let (origin, _) = serve(router).await;
            prefills.push(json!({"origin": format!("{origin}/"), "bootstrap_port": 8998}));
        }
        let decode = Router::new().route(
            "/v1/chat/completions",
            post(|| async { ([("content-type", "text/event-stream")], WORKER_RESPONSE) }),
        );
        let (decode_origin, _) = serve(decode).await;
        let config: Config = serde_json::from_value(json!({
            "listen": "127.0.0.1:0", "model": "test-model",
            "pd": {"prefills": prefills, "decodes": [format!("{decode_origin}/")]}
        }))
        .unwrap();
        let engine = Arc::new(StreamingEngineAdapter::new(Arc::new(
            build_engine(config).unwrap(),
        )));
        let service = HttpService::builder()
            .enable_chat_endpoints(true)
            .enable_admin_api(false)
            .host("127.0.0.1".to_string())
            .port(0)
            .build()
            .unwrap();
        service
            .model_manager()
            .add_chat_completions_model("test-model", "test-checksum", engine)
            .unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let cancel = CancellationToken::new();
        service.spawn_with_listener(cancel.clone(), listener).await;
        let client = reqwest::Client::new();
        let mut messages = vec![
            json!({"role": "system", "content": "a long shared system prompt"}),
            json!({"role": "user", "content": "first question"}),
        ];
        for turn in 0..6 {
            let response = client
                .post(format!("http://{address}/v1/chat/completions"))
                .json(&json!({"model": "test-model", "max_tokens": 8, "temperature": 0,
                              "reasoning_effort": "low", "messages": messages}))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::OK);
            response.bytes().await.unwrap();
            messages.push(json!({"role": "assistant", "content": format!("answer {turn}")}));
            messages.push(json!({"role": "user", "content": format!("follow-up {turn}")}));
        }
        let hits = hits.lock().unwrap().clone();
        assert_eq!(hits.len(), 6);
        assert!(hits.iter().all(|&p| p == hits[0]), "turns spread over prefills: {hits:?}");
        cancel.cancel();
    }

    #[test]
    fn config_requires_exactly_one_worker_mode() {
        let both: Config = serde_json::from_value(json!({
            "listen": "127.0.0.1:0", "model": "m", "workers": ["http://a:1/"],
            "pd": {"prefills": [{"origin": "http://p:1/", "bootstrap_port": 8998}],
                   "decodes": ["http://d:1/"]}
        }))
        .unwrap();
        assert!(build_engine(both).is_err());
        let duplicate: Config = serde_json::from_value(json!({
            "listen": "127.0.0.1:0", "model": "m",
            "pd": {"prefills": [{"origin": "http://w:1/", "bootstrap_port": 8998}],
                   "decodes": ["http://w:1/"]}
        }))
        .unwrap();
        assert!(build_engine(duplicate).is_err());
        assert!(serde_json::from_value::<Config>(json!({
            "listen": "127.0.0.1:0", "model": "m",
            "pd": {"prefills": [{"origin": "http://p:1/"}], "decodes": ["http://d:1/"]}
        }))
        .is_err());
    }
}
