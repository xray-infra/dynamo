//! Same-worker ordinary request control. Generation is never replayed here.
use std::sync::Arc;
use std::time::Duration;

use crate::cache_authority::frontend_clock::ClockBounds;
use async_trait::async_trait;
use tokio::time::Instant;

pub use crate::native_pd::transport::QualifiedClaims;
use crate::native_pd::transport::{
    NativeControlHttp, PdTransportError, PreparedControlResponse, encode_bounded, transient,
};
use crate::native_request_cleanup::{
    self as control, ExpectedRequest, RequestStatus, VerifiedNativeTerminal,
};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RequestTransportError {
    #[error("ordinary native control transport: {0}")]
    Control(#[from] PdTransportError),
    #[error("ordinary native control verification: {0}")]
    Verification(String),
    #[error("ordinary native physical cleanup deadline elapsed")]
    Deadline,
}

#[async_trait]
pub trait RequestClaimProvider: Send + Sync {
    /// Fresh operation/nonce, actual qualified identity, and the original
    /// admission window. A control poll must not extend the generation fence.
    async fn claims(&self, operation: &str) -> Result<QualifiedClaims, String>;
    async fn clock_bounds(&self) -> Result<ClockBounds, String>;
}

pub struct NativeRequestTransport {
    http: NativeControlHttp,
}
impl NativeRequestTransport {
    pub fn new(
        origin: &str,
        secret: [u8; 32],
        timeout: Duration,
    ) -> Result<Self, RequestTransportError> {
        Ok(Self {
            http: NativeControlHttp::new(origin, secret, timeout)?,
        })
    }

    pub async fn status(
        &self,
        expected: &ExpectedRequest,
        provider: &dyn RequestClaimProvider,
    ) -> Result<RequestStatus, RequestTransportError> {
        self.call("request-status", "/v1/requests/status", expected, provider)
            .await
            .map(|(_, status, _, _)| status)
    }
    pub async fn abort(
        &self,
        expected: &ExpectedRequest,
        provider: &dyn RequestClaimProvider,
    ) -> Result<RequestStatus, RequestTransportError> {
        self.call("request-abort", "/v1/requests/abort", expected, provider)
            .await
            .map(|(_, status, _, _)| status)
    }

    async fn call(
        &self,
        operation: &str,
        path: &str,
        expected: &ExpectedRequest,
        provider: &dyn RequestClaimProvider,
    ) -> Result<
        (
            PreparedControlResponse,
            RequestStatus,
            QualifiedClaims,
            ClockBounds,
        ),
        RequestTransportError,
    > {
        // Bound input before canonical signing clones it as well as the actual
        // signed HTTP request. The response reader shares the same 8 MiB bound.
        encode_bounded(&expected.control)?;
        let prepared = control::PreparedRequestControl::new(expected)
            .map_err(RequestTransportError::Verification)?;
        let qualified = provider
            .claims(operation)
            .await
            .map_err(RequestTransportError::Verification)?;
        self.http.check_origin(operation, &qualified.claims)?;
        encode_bounded(&qualified.claims)?;
        let packet = prepared
            .sign(&self.http.secret, &qualified)
            .map_err(RequestTransportError::Verification)?;
        let packet = self
            .http
            .post(path, &qualified, encode_bounded(&packet)?)
            .await?;
        let packet =
            PreparedControlResponse::new(packet).map_err(RequestTransportError::Verification)?;
        let clock = provider
            .clock_bounds()
            .await
            .map_err(RequestTransportError::Verification)?;
        let status = control::verify_status(
            &self.http.secret,
            &qualified.claims,
            expected,
            &packet,
            &clock,
        )
        .map_err(|error| {
            tracing::warn!(target:"dynamo_llm::native_cleanup",operation,
                request_root=%expected.control.request_root,error=%error,
                "ordinary native control receipt verification failed");
            RequestTransportError::Verification(error)
        })?;
        Ok((packet, status, qualified, clock))
    }

    async fn terminal(
        &self,
        operation: &str,
        expected: &ExpectedRequest,
        provider: &dyn RequestClaimProvider,
    ) -> Result<Option<VerifiedNativeTerminal>, RequestTransportError> {
        let path = match operation {
            "request-status" => "/v1/requests/status",
            "request-abort" => "/v1/requests/abort",
            _ => {
                return Err(RequestTransportError::Verification(
                    "ordinary control operation mismatch".into(),
                ));
            }
        };
        let (packet, status, qualified, clock) =
            self.call(operation, path, expected, provider).await?;
        if !status.terminal {
            return Ok(None);
        }
        // The parent verifier alone constructs opaque evidence; a parsed bool
        // cannot be converted into a registry acknowledgement.
        control::verify_terminal(
            &self.http.secret,
            &qualified.claims,
            expected,
            &packet,
            &clock,
        )
        .map(Some)
        .map_err(RequestTransportError::Verification)
    }

    pub async fn cleanup(
        &self,
        expected: Arc<ExpectedRequest>,
        claims: Arc<dyn RequestClaimProvider>,
        deadline: Instant,
        poll: Duration,
    ) -> Result<VerifiedNativeTerminal, RequestTransportError> {
        if poll.is_zero() {
            return Err(RequestTransportError::Verification(
                "cleanup poll interval must be positive".into(),
            ));
        }
        let cleanup = async {
            loop {
                // Pending physical capture is transient. Repeat the idempotent
                // exact-request abort with fresh claims and the original window,
                // so every actual participant processes cancellation. A status
                // snapshot cannot complete an abort that has not reached it.
                match self
                    .terminal("request-abort", &expected, claims.as_ref())
                    .await
                {
                    Ok(Some(receipt)) => return Ok(receipt),
                    Ok(None) => {}
                    Err(error) if retry_control(&error) => {}
                    Err(error) => return Err(error),
                }
                tokio::time::sleep(poll).await;
            }
        };
        tokio::time::timeout_at(deadline, cleanup)
            .await
            .map_err(|_| RequestTransportError::Deadline)?
    }
}

fn retry_control(error: &RequestTransportError) -> bool {
    matches!(error, RequestTransportError::Control(error) if transient(error))
}

#[cfg(test)]
mod abort_retry_tests {
    use super::*;
    use crate::cache_authority::{cache_auth, frontend_clock::qualified_cpu_tracking_sample};
    use crate::native_cleanup_cohort::{CleanupCohortArtifact, install_cleanup_cohort};
    use crate::native_request_cleanup::RequestControl;
    use serde_json::{Value, json};
    use std::collections::BTreeSet;
    use std::sync::Mutex;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    const NOW: u64 = 1_000_000;
    const KEY: [u8; 32] = [42; 32];

    fn clock() -> ClockBounds {
        let at = Instant::now();
        qualified_cpu_tracking_sample(
            "Reference ID : C0000201\nSystem time : 0 seconds fast of NTP time\nRoot dispersion : 0 seconds\nRoot delay : 0 seconds\nLeap status : Normal\n",
            u128::from(NOW) * 1_000_000, u128::from(NOW) * 1_000_000, at, at,
        ).unwrap()
    }

    // CPU identity/time fixture; the normal PreparedRequestControl, signing,
    // real HTTP transport and opaque receipt verification remain production code.
    struct Claims {
        original: Value,
    }
    #[async_trait]
    impl RequestClaimProvider for Claims {
        async fn claims(&self, operation: &str) -> Result<QualifiedClaims, String> {
            let clock = clock();
            let mut claims = self.original.clone();
            claims["operation"] = operation.into();
            claims["operation_id"] = uuid::Uuid::new_v4().simple().to_string().into();
            claims["nonce"] = uuid::Uuid::new_v4().simple().to_string().into();
            claims["issued_at_ms"] = clock.lower_ms().into();
            Ok(QualifiedClaims { claims, clock })
        }
        async fn clock_bounds(&self) -> Result<ClockBounds, String> {
            Ok(clock())
        }
    }

    async fn run_case(mode: u16) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let participants: BTreeSet<String> = [
            "cpu-tp0", "cpu-tp1", "cpu-tp2", "cpu-tp3", "cpu-tokenizer", "cpu-ingress",
        ].into_iter().map(str::to_owned).collect();
        let artifact = CleanupCohortArtifact::parse(&serde_json::to_vec(&json!({
            "schema_version":1, "model":"CPU fixture only", "native_origin":origin,
            "native_incarnation":"cpu-native-birth", "source_digest":"a".repeat(64),
            "runtime_profile_digest":"b".repeat(64),
            "cleanup_cohort_digest":cache_auth::digest(&json!(participants)).unwrap(),
            "participants":participants.iter().map(|id| json!({
                "participant_id":id, "role":"CPU registered physical owner",
                "process_birth":format!("CPU process birth {id}"), "source_digest":"a".repeat(64),
                "runtime_registration_digest":cache_auth::digest(&json!(["CPU registration", id])).unwrap(),
            })).collect::<Vec<_>>(),
        })).unwrap()).unwrap();
        let installed = install_cleanup_cohort(artifact).unwrap();
        let expected = Arc::new(ExpectedRequest {
            control: RequestControl {
                schema_version: 1, transaction_id: "2".repeat(32), request_root: "3".repeat(32),
                expected_native_incarnation: installed.artifact().native_incarnation.clone(),
                expected_cleanup_cohort_digest: installed.cleanup_cohort_digest().into(),
                ordinary_body_digest: cache_auth::digest(&json!("CPU ordinary body")).unwrap(),
                expected_choices: 2,
            },
            participants: installed.participant_ids(),
        });
        let control = serde_json::to_value(&expected.control).unwrap();
        let provider = Arc::new(Claims { original: json!({
            "version":1, "operation":"request-abort", "operation_id":"cpu-initial-operation",
            "nonce":"cpu-initial-nonce", "model":"CPU fixture only", "native_origin":origin,
            "native_incarnation":"cpu-native-birth", "source_digest":"a".repeat(64),
            "runtime_profile_digest":"b".repeat(64), "reservation_id":"", "authority_revision":0,
            "native_request_id":"3".repeat(32), "namespace_digest":"", "full_render_digest":"",
            "token_ids_digest":"", "request_started_at_ms":NOW-100, "issued_at_ms":NOW-1,
            "deadline_ms":NOW+10_000, "retain_for_ms":0,
            "payload_digest":cache_auth::digest(&control).unwrap(),
        }) });
        let observed = Arc::new(Mutex::new(Vec::<Value>::new()));
        let records = observed.clone();
        let server_expected = expected.clone();
        let server_control = control.clone();
        let server = tokio::spawn(async move {
            let mut attempt = 0;
            loop {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut bytes = Vec::new();
                let (headers_end, content_length) = loop {
                    let mut chunk = [0; 4096];
                    let read = socket.read(&mut chunk).await.unwrap();
                    assert!(read > 0 && bytes.len() + read < 64 * 1024);
                    bytes.extend_from_slice(&chunk[..read]);
                    if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
                        let headers = std::str::from_utf8(&bytes[..end]).unwrap();
                        assert!(headers.starts_with("POST /v1/requests/abort HTTP/1.1"));
                        let length = headers.lines().find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().unwrap())
                        }).unwrap();
                        break (end + 4, length);
                    }
                };
                while bytes.len() < headers_end + content_length {
                    let mut chunk = [0; 4096];
                    let read = socket.read(&mut chunk).await.unwrap();
                    assert!(read > 0 && bytes.len() + read < 64 * 1024);
                    bytes.extend_from_slice(&chunk[..read]);
                }
                let mut packet: Value = serde_json::from_slice(
                    &bytes[headers_end..headers_end + content_length]
                ).unwrap();
                let proof = packet.as_object_mut().unwrap().remove(cache_auth::PROOF_FIELD).unwrap();
                cache_auth::verify_signature(&KEY, &proof, &packet).unwrap();
                assert_eq!(packet, server_control);
                assert_eq!(proof["claims"]["operation"], "request-abort");
                records.lock().unwrap().push(proof["claims"].clone());
                attempt += 1;
                let status = if mode == 200 && attempt > 1 { 200 } else if mode == 200 { 503 } else { mode };
                let reply = if status == 200 {
                    let c = &server_expected.control;
                    let rows = server_expected.participants.iter().map(|id| json!({
                        "participant_id":id, "ingress_closed":true, "tokenizer_drained":true,
                        "scheduler_drained":true, "transport_drained":true, "abort_ack":true,
                        "live_request_holds":0, "live_kv_holds":0, "live_meta_holds":0, "live_io_holds":0,
                    })).collect::<Vec<_>>();
                    let mut reply = json!({"schema_version":1, "transaction_id":c.transaction_id,
                        "request_root":c.request_root, "native_incarnation":c.expected_native_incarnation,
                        "cleanup_cohort_digest":c.expected_cleanup_cohort_digest,
                        "ordinary_body_digest":c.ordinary_body_digest, "expected_choices":2,
                        "captured_at_ms":NOW-2, "participants":rows, "terminal":true, "aborted":true});
                    let event = cache_auth::digest(&reply).unwrap();
                    let mut response_claims = proof["claims"].clone();
                    for field in ["operation_id", "nonce"] {
                        response_claims[field] = cache_auth::digest(
                            &json!(["native-response", proof["claims"][field], event])
                        ).unwrap().into();
                    }
                    reply[cache_auth::PROOF_FIELD] = cache_auth::sign(&KEY, &response_claims, &reply).unwrap();
                    reply
                } else { json!({"error":"ordinary capture is not ready"}) };
                let body = reply.to_string();
                socket.write_all(format!("HTTP/1.1 {status} Control\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
                if status == 200 || status == 400 { break; }
            }
        });
        let transport = NativeRequestTransport::new(&origin, KEY, Duration::from_millis(200)).unwrap();
        let result = transport.cleanup(expected.clone(), provider,
            Instant::now() + Duration::from_millis(if mode == 503 { 250 } else { 1000 }),
            Duration::from_millis(2)).await;
        match mode {
            200 => {
                let receipt = result.unwrap();
                assert_eq!(receipt.cleanup_id(), expected.cleanup_id());
                assert!(receipt.status().terminal && receipt.status().aborted);
                assert_eq!(receipt.status().participants.len(), 6);
                assert!(receipt.status().participants.iter().all(|row| row.abort_ack));
                server.await.unwrap();
            }
            400 => {
                assert!(matches!(result, Err(RequestTransportError::Control(PdTransportError::HttpStatus(400)))));
                server.await.unwrap();
            }
            503 => {
                assert!(matches!(result, Err(RequestTransportError::Deadline)));
                server.abort();
                let _ = server.await;
            }
            _ => unreachable!(),
        }
        let records = observed.lock().unwrap();
        match mode {
            200 => assert_eq!(records.len(), 2),
            400 => assert_eq!(records.len(), 1),
            503 => assert!(records.len() >= 2),
            _ => unreachable!(),
        }
        for claims in records.iter() {
            assert_eq!(claims["request_started_at_ms"], NOW-100);
            assert_eq!(claims["deadline_ms"], NOW+10_000);
            assert_eq!(claims["native_request_id"], expected.control.request_root);
            assert_eq!(claims["payload_digest"], cache_auth::digest(&control).unwrap());
        }
        if records.len() > 1 {
            assert_ne!(records[0]["operation_id"], records[1]["operation_id"]);
            assert_ne!(records[0]["nonce"], records[1]["nonce"]);
        }
    }

    #[tokio::test]
    async fn ordinary_cleanup_retries_exact_abort_after_pending_http_capture() { run_case(200).await; }

    #[tokio::test]
    async fn ordinary_cleanup_retries_exact_abort_preserves_original_deadline() { run_case(503).await; }

    #[tokio::test]
    async fn ordinary_cleanup_retries_exact_abort_rejects_permanent_refusal() { run_case(400).await; }
}
