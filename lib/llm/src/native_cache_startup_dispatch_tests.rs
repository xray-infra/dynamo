//! New startup boundary only; this does not replay the cache HTTP matrices.
use super::*;
use serde_json::json;

#[tokio::test]
async fn native_cache_new_boundary_unqualified_marked_request_has_no_native_side_effect() {
    let calls = Arc::new(AtomicUsize::new(0));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let spy = calls.clone();
    let router = axum::Router::new().fallback(move || {
        let spy = spy.clone();
        async move {
            spy.fetch_add(1, Ordering::SeqCst);
            axum::http::StatusCode::INTERNAL_SERVER_ERROR
        }
    });
    let server = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    let engine = NativeHttpEngine::new(NativeHttpEngineConfig {
        model: "glm-5.3-flash".into(),
        qualification_tier: None,
        workers: vec![NativeWorkerConfig {
            url,
            tier: NativeTier::Shared,
            max_inflight: None,
            load_path: Some("/v1/loads?include=core".into()),
            max_input_tokens: None,
            capability_source: None,
            active_request_ids_source: Some("CPU rejection spy only".into()),
            native_dp_size: Some(1),
            native_id_format: NativeIdFormat::VllmChatId,
            identity: None,
            thinking_budget_qualification: None,
            forced_zero_qualification: None,
        }],
    })
    .unwrap();
    let request = json!({"model":"glm-5.3-flash","max_tokens":1,
        "messages":[{"role":"user","content":"eligible"}],
        "cache_control":{"type":"ephemeral","ttl":"1h"}});
    let plan = crate::native_cache_plan::messages_cache_plan(
        &request,
        crate::native_cache_instruction_roles::Family::GlmFlash,
        || "cpu-rejection-spy-only".into(),
    )
    .unwrap();
    let body = plan.body.clone();
    let context = dynamo_runtime::pipeline::Context::with_id_and_metadata(
        body.clone(),
        "owned-startup-rejection".into(),
        BTreeMap::new(),
    )
    .context();
    let result = engine
        .forward_messages_with_cache_count(
            body,
            context,
            MessagesCacheContext {
                origin: HttpIngressOrigin {
                    request_id: "owned-startup-rejection".into(),
                    started_at: Instant::now(),
                },
                plan,
            },
        )
        .await;
    let error = match result {
        Ok(_) => panic!("unqualified cache unexpectedly dispatched"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("qualified native cache frontend")
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "Count/reserve/Gen must all remain absent"
    );
    server.abort();
    let _ = server.await;
}
