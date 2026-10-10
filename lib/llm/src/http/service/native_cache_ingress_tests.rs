//! The exact public ingress guard shared by Messages generation and Count.
use super::*;
use crate::native_contract::NativeTier;
use crate::native_http_engine::{
    NativeHttpEngine, NativeHttpEngineConfig, NativeIdFormat, NativeWorkerConfig,
};
use serde_json::json;

#[tokio::test]
async fn native_cache_new_boundary_public_marker_rejects_before_dispatch() {
    let engine = NativeHttpEngine::new(NativeHttpEngineConfig {
        model: "glm-5.3-flash".into(),
        qualification_tier: None,
        workers: vec![NativeWorkerConfig {
            url: "http://127.0.0.1:8123".into(),
            tier: NativeTier::Shared,
            max_inflight: None,
            load_path: Some("/v1/loads?include=core".into()),
            max_input_tokens: None,
            capability_source: None,
            active_request_ids_source: Some("CPU public rejection fixture only".into()),
            native_dp_size: Some(1),
            native_id_format: NativeIdFormat::VllmChatId,
            identity: None,
            thinking_budget_qualification: None,
            forced_zero_qualification: None,
        }],
    })
    .unwrap();
    for ttl in ["5m", "1h"] {
        for content in ["", "eligible"] {
            let request = json!({"model":"glm-5.3-flash","max_tokens":1,
                "messages":[{"role":"user","content":content}],
                "cache_control":{"type":"ephemeral","ttl":ttl}});
            let response = match native_messages_plan(&request, engine.model()) {
                Err(response) => {
                    assert_eq!(content, "");
                    response
                }
                Ok(plan) => {
                    assert_eq!(content, "eligible");
                    assert_eq!(plan.writes.len(), 1);
                    require_native_cache_frontend(&engine, &plan).unwrap_err()
                }
            };
            assert_eq!(response.status(), if content.is_empty() {
                StatusCode::BAD_REQUEST
            } else {
                StatusCode::SERVICE_UNAVAILABLE
            });
            let bytes = axum::body::to_bytes(response.into_body(), 8192)
                .await
                .unwrap();
            let error: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(error["type"], "error");
            assert_eq!(error["error"]["type"], if content.is_empty() {
                "invalid_request_error"
            } else {
                "overloaded_error"
            });
        }
    }
    let ordinary = json!({"model":"glm-5.3-flash","max_tokens":1,
        "messages":[{"role":"user","content":""}]});
    let plan = native_messages_plan(&ordinary, engine.model()).unwrap();
    assert!(plan.writes.is_empty());
    assert!(require_native_cache_frontend(&engine, &plan).is_ok());
}
