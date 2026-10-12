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
