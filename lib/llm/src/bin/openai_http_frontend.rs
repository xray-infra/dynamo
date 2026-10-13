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
use dynamo_llm::openai_http_engine::OpenAiHttpEngine;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    listen: SocketAddr,
    model: String,
    workers: Vec<String>,
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
    let engine = Arc::new(StreamingEngineAdapter::new(Arc::new(
        OpenAiHttpEngine::new(config.workers)?,
    )));
    let mut api = FrontendApiConfig::default();
    api.anthropic_mut().set_enabled(true);
    let service = HttpService::builder()
        .frontend_api_config(api)
        .enable_chat_endpoints(true)
        .enable_admin_api(false)
        .host(config.listen.ip().to_string())
        .port(config.listen.port())
        .build()?;
    // This is the upstream local text-engine registration, not a discovered
    // worker card or a fabricated worker capability/TTL authority.
    service.model_manager().add_chat_completions_model(
        &config.model,
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
}
