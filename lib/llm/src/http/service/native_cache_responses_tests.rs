//! Narrow actual Responses codec gates for the verified cache accounting bridge.
//! Included as a child test module of the existing HTTP service openai module.
use super::{NativeHttpProtocol, NativeHttpStreamCodec, ResponseParams, native_chat_codec_value};
use crate::cache_authority::cache_accounting::VerifiedCacheAccounting;
use crate::protocols::openai::responses::{
    chat_completion_to_response, omit_unknown_native_reasoning_usage,
};
use serde_json::{Value, json};

fn fixture(stream: bool, known: bool) -> Value {
    let report = if known {
        serde_json::to_value(VerifiedCacheAccounting {
            input_tokens: 10,
            cache_read_input_tokens: 40,
            cache_creation_5m_input_tokens: 20,
            cache_creation_1h_input_tokens: 30,
        })
        .unwrap()
    } else {
        Value::Null
    };
    let mut raw = json!({"id":"verified-cache-fixture","created":1,
        "model":"qwen3.8-flash-next",
        "object":if stream {"chat.completion.chunk"} else {"chat.completion"},
        "choices":[{"index":0,"finish_reason":"stop","logprobs":null}],
        "usage":{"prompt_tokens":100,"completion_tokens":41,"total_tokens":141,
            "reasoning_tokens":16,"cached_tokens":99,
            "prompt_tokens_details":{"cached_tokens":99},
            "_dynamo_verified_cache_accounting":report}});
    raw["choices"][0][if stream { "delta" } else { "message" }] =
        json!({"role":"assistant","content":"done"});
    raw
}

#[test]
fn native_verified_cache_responses_json_uses_actual_read_or_unknown_usage() {
    for known in [true, false] {
        let canonical = native_chat_codec_value(fixture(false, known)).unwrap();
        assert!(
            !canonical
                .to_string()
                .contains("_dynamo_verified_cache_accounting")
        );
        let typed = serde_json::from_value(omit_unknown_native_reasoning_usage(canonical)).unwrap();
        let response =
            chat_completion_to_response(typed, &ResponseParams::default(), None).unwrap();
        let response = serde_json::to_value(response).unwrap();
        assert_eq!(response["status"], "completed");
        assert_eq!(response["output"][0]["content"][0]["text"], "done");
        if known {
            assert_eq!(response["usage"]["input_tokens"], 100);
            assert_eq!(
                response["usage"]["input_tokens_details"]["cached_tokens"],
                40
            );
            assert_eq!(response["usage"]["output_tokens"], 41);
            assert_eq!(
                response["usage"]["output_tokens_details"]["reasoning_tokens"],
                16
            );
        } else {
            assert!(response["usage"].is_null());
        }
        assert!(
            !response
                .to_string()
                .contains("_dynamo_verified_cache_accounting")
        );
    }
}

#[tokio::test]
async fn native_verified_cache_responses_sse_uses_actual_read_and_clears_unknown_usage() {
    use axum::response::IntoResponse;
    use http_body_util::BodyExt;
    for known in [true, false] {
        let mut codec = NativeHttpStreamCodec::new(
            NativeHttpProtocol::Responses(ResponseParams::default()),
            "qwen3.8-flash-next".into(),
            100,
        );
        let mut events = codec.start();
        if !known {
            // A later unknown authoritative result must clear earlier known usage.
            let mut earlier = fixture(true, true);
            earlier["choices"] = json!([]);
            events.extend(codec.chunk(earlier).unwrap());
        }
        events.extend(codec.chunk(fixture(true, known)).unwrap());
        events.extend(codec.end());
        let body = axum::response::sse::Sse::new(futures::stream::iter(
            events
                .into_iter()
                .map(|event| event.map_err(axum::Error::new)),
        ))
        .into_response()
        .into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes();
        let wire = std::str::from_utf8(&body).unwrap();
        assert!(!wire.contains("_dynamo_verified_cache_accounting"));
        let values: Vec<Value> = wire
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect();
        assert_eq!(
            values
                .iter()
                .filter(|v| v["type"] == "response.completed")
                .count(),
            1
        );
        assert!(values.iter().all(|v| v["type"] != "response.failed"));
        let response = &values
            .iter()
            .find(|v| v["type"] == "response.completed")
            .unwrap()["response"];
        assert_eq!(response["output"][0]["content"][0]["text"], "done");
        if known {
            assert_eq!(
                response["usage"]["input_tokens_details"]["cached_tokens"],
                40
            );
            assert_eq!(
                response["usage"]["output_tokens_details"]["reasoning_tokens"],
                16
            );
            assert_eq!(response["usage"]["input_tokens"], 100);
            assert_eq!(response["usage"]["output_tokens"], 41);
        } else {
            assert!(response["usage"].is_null());
        }
    }
}
