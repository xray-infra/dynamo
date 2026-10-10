//! Same-worker signed P/D control RPCs and shutdown-owned physical cleanup.
//! Neither HTTP success nor task completion is a physical resource receipt.
use crate::cache_authority::cache_auth::PrehashedPayload;
use crate::cache_authority::{cache_auth, frontend_clock::ClockBounds};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Value, json};
use tokio::time::Instant;

use super::control::{self, ControlResponse, ExpectedTransaction, PdRole};

const MAX_RESPONSE_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum PdTransportError {
    #[error("invalid native P/D control request: {0}")]
    InvalidRequest(String),
    #[error("native P/D control result unknown (timeout={timed_out})")]
    Unknown { timed_out: bool },
    #[error("native P/D control HTTP status {0}")]
    HttpStatus(u16),
    #[error("native P/D control response exceeds 8 MiB")]
    ResponseTooLarge,
    #[error("native P/D control request exceeds 8 MiB")]
    RequestTooLarge,
    #[error("invalid native P/D physical receipt: {0}")]
    InvalidReceipt(String),
    #[error("native P/D physical cleanup deadline elapsed")]
    Deadline,
    #[error("native P/D cleanup finished without a physical terminal receipt")]
    Nonterminal,
}

/// Only `verify_response` may construct this receipt. Callers cannot register
/// an arbitrary deserialized `ControlResponse` as cleanup success.
#[derive(Debug)]
pub struct VerifiedReceipt(ControlResponse);
impl VerifiedReceipt {
    pub fn response(&self) -> &ControlResponse {
        &self.0
    }
    pub fn physical_terminal(&self) -> bool {
        self.0.terminal
    }
    pub fn cleanup_id(&self) -> String {
        cleanup_id(
            &self.0.transaction_id,
            self.0.role,
            &self.0.native_incarnation,
        )
    }
}
impl ExpectedTransaction {
    pub fn cleanup_id(&self) -> String {
        cleanup_id(
            &self.request.transaction_id,
            self.request.role,
            &self.request.expected_native_incarnation,
        )
    }
}
fn cleanup_id(transaction: &str, role: PdRole, incarnation: &str) -> String {
    let role = match role {
        PdRole::Prefill => "prefill",
        PdRole::Decode => "decode",
    };
    format!("pd:{transaction}:{role}:{incarnation}")
}

#[async_trait]
pub trait PdClaimProvider: Send + Sync {
    /// Produce complete claims from the caller's actual authority and pinned
    /// native identity. Every control RPC needs a fresh operation ID and nonce.
    async fn claims(&self, operation: &str) -> Result<QualifiedClaims, String>;
    async fn clock_bounds(&self) -> Result<ClockBounds, String>;
}

pub struct QualifiedClaims {
    pub claims: Value,
    pub clock: ClockBounds,
}
impl QualifiedClaims {
    pub fn ensure_fresh(&self) -> Result<(), String> {
        self.clock.ensure_fresh().map_err(|e| e.to_string())?;
        if self.claims["issued_at_ms"].as_u64() != Some(self.clock.lower_ms()) {
            return Err("signed issue time differs from qualified lower bound".into());
        }
        let (_, upper) = current_bounds(&self.clock)?;
        let start = self.claims["request_started_at_ms"]
            .as_u64()
            .ok_or("qualified request start missing")?;
        let deadline = self.claims["deadline_ms"]
            .as_u64()
            .ok_or("qualified request deadline missing")?;
        if self.clock.lower_ms() < start || upper >= deadline {
            return Err("qualified clock crosses original request admission window".into());
        }
        Ok(())
    }
}
pub fn current_bounds(clock: &ClockBounds) -> Result<(u64, u64), String> {
    clock.ensure_fresh().map_err(|e| e.to_string())?;
    let age = Instant::now()
        .checked_duration_since(clock.observed_at())
        .ok_or("qualified clock monotonic reversal")?;
    let lower_age =
        u64::try_from(age.as_millis()).map_err(|_| "qualified clock elapsed overflow")?;
    let upper_age = u64::try_from(age.as_nanos().div_ceil(1_000_000))
        .map_err(|_| "qualified clock elapsed overflow")?;
    Ok((
        clock
            .lower_ms()
            .checked_add(lower_age)
            .ok_or("qualified lower clock overflow")?,
        clock
            .upper_ms()
            .checked_add(upper_age)
            .ok_or("qualified upper clock overflow")?,
    ))
}
/// Parse and prehash before awaiting the receive clock. Private fields keep the
/// exact signed payload immutable between qualification and HMAC verification.
pub struct PreparedControlResponse {
    payload: Value,
    proof: Value,
    digest: PrehashedPayload,
}
impl PreparedControlResponse {
    pub fn new(mut packet: Value) -> Result<Self, String> {
        let proof = packet
            .as_object_mut()
            .ok_or("native control receipt must be an object")?
            .remove(cache_auth::PROOF_FIELD)
            .ok_or("native control receipt unsigned")?;
        Ok(Self {
            digest: PrehashedPayload::new(&packet)?,
            payload: packet,
            proof,
        })
    }
    pub(crate) fn payload(&self) -> &Value {
        &self.payload
    }
    pub(crate) fn verify(
        &self,
        secret: &[u8],
        request_claims: &Value,
        clock: &ClockBounds,
    ) -> Result<(), String> {
        clock.ensure_fresh().map_err(|e| e.to_string())?;
        let claims = cache_auth::verify_prehashed(secret, &self.proof, &self.digest)?;
        for (key, value) in request_claims
            .as_object()
            .ok_or("control claims object required")?
        {
            let expected = match key.as_str() {
                "payload_digest" => continue,
                "operation_id" | "nonce" => Value::String(cache_auth::digest(&json!([
                    "native-response",
                    value,
                    self.digest.digest()
                ]))?),
                _ => value.clone(),
            };
            if claims[key] != expected {
                return Err("native control response claim binding mismatch".into());
            }
        }
        clock.ensure_fresh().map_err(|e| e.to_string())
    }
}

pub struct NativePdTransport {
    http: NativeControlHttp,
}
/// Shared signed-control HTTP mechanics, not an additional gateway. Typed
/// P/D and ordinary verifiers own the payload and physical receipt semantics.
pub(crate) struct NativeControlHttp {
    origin: String,
    pub(crate) secret: [u8; 32],
    request_timeout: Duration,
    client: reqwest::Client,
}
impl NativeControlHttp {
    pub(crate) fn new(
        origin: &str,
        secret: [u8; 32],
        request_timeout: Duration,
    ) -> Result<Self, PdTransportError> {
        let url = reqwest::Url::parse(origin).map_err(|_| {
            PdTransportError::InvalidRequest("worker must be an HTTP origin".into())
        })?;
        if !matches!(url.scheme(), "http" | "https")
            || url.host_str().is_none()
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
            || url.path() != "/"
            || request_timeout.is_zero()
        {
            return Err(PdTransportError::InvalidRequest(
                "worker must be an HTTP origin and timeout positive".into(),
            ));
        }
        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .connect_timeout(request_timeout)
            .build()
            .map_err(|_| PdTransportError::InvalidRequest("HTTP client unavailable".into()))?;
        Ok(Self {
            origin: origin.trim_end_matches('/').into(),
            secret,
            request_timeout,
            client,
        })
    }

    pub(crate) fn check_origin(
        &self,
        operation: &str,
        claims: &Value,
    ) -> Result<(), PdTransportError> {
        if claims["operation"].as_str() != Some(operation)
            || claims["native_origin"].as_str() != Some(self.origin.as_str())
        {
            return Err(PdTransportError::InvalidRequest(
                "operation or same-worker origin mismatch".into(),
            ));
        }
        Ok(())
    }
    pub(crate) async fn post(
        &self,
        path: &str,
        qualified: &QualifiedClaims,
        body: Vec<u8>,
    ) -> Result<Value, PdTransportError> {
        qualified
            .ensure_fresh()
            .map_err(PdTransportError::InvalidRequest)?;
        let (_, upper) =
            current_bounds(&qualified.clock).map_err(PdTransportError::InvalidRequest)?;
        let claims = &qualified.claims;
        let deadline = claims["deadline_ms"]
            .as_u64()
            .ok_or_else(|| PdTransportError::InvalidRequest("claim deadline missing".into()))?;
        if deadline <= upper {
            return Err(PdTransportError::Deadline);
        }
        let timeout = self
            .request_timeout
            .min(Duration::from_millis(deadline - upper));
        let sent_at = Instant::now();
        let response = self
            .client
            .post(format!("{}{path}", self.origin))
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(body)
            .timeout(timeout)
            .send()
            .await
            .map_err(unknown)?;
        if response.status().as_u16() != 200 {
            let status = response.status().as_u16();
            let remaining =
                Duration::from_millis(deadline - upper).saturating_sub(sent_at.elapsed());
            let body_timeout = remaining.min(Duration::from_millis(250));
            let diagnosis = error_diagnosis(response, body_timeout).await;
            tracing::warn!(target:"dynamo_llm::native_cleanup",
                operation=?claims["operation"].as_str(), path, status,
                request_root=?claims["native_request_id"].as_str(),
                body_state=diagnosis.state, native_error=%diagnosis.detail,
                "native control HTTP request rejected");
            return Err(PdTransportError::HttpStatus(status));
        }
        if response
            .content_length()
            .is_some_and(|length| length > MAX_RESPONSE_BYTES as u64)
        {
            return Err(PdTransportError::ResponseTooLarge);
        }
        let mut bytes = Vec::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(unknown)?;
            if chunk.len() > MAX_RESPONSE_BYTES.saturating_sub(bytes.len()) {
                return Err(PdTransportError::ResponseTooLarge);
            }
            bytes.extend_from_slice(&chunk);
        }
        serde_json::from_slice(&bytes)
            .map_err(|_| PdTransportError::InvalidReceipt("malformed JSON".into()))
    }
}
impl NativePdTransport {
    pub fn new(
        origin: &str,
        secret: [u8; 32],
        request_timeout: Duration,
    ) -> Result<Self, PdTransportError> {
        Ok(Self {
            http: NativeControlHttp::new(origin, secret, request_timeout)?,
        })
    }
    pub async fn status(
        &self,
        expected: &ExpectedTransaction,
        provider: &dyn PdClaimProvider,
    ) -> Result<VerifiedReceipt, PdTransportError> {
        self.call(
            "pd-status",
            "/v1/pd/transactions/status",
            expected,
            provider,
        )
        .await
    }
    pub async fn abort(
        &self,
        expected: &ExpectedTransaction,
        provider: &dyn PdClaimProvider,
    ) -> Result<VerifiedReceipt, PdTransportError> {
        self.call("pd-abort", "/v1/pd/transactions/abort", expected, provider)
            .await
    }
    async fn call(
        &self,
        operation: &str,
        path: &str,
        expected: &ExpectedTransaction,
        provider: &dyn PdClaimProvider,
    ) -> Result<VerifiedReceipt, PdTransportError> {
        // Exact bounded control preparation precedes the clock await.
        encode_bounded(&expected.request)?;
        let prepared =
            control::PreparedPdControl::new(expected).map_err(PdTransportError::InvalidRequest)?;
        let qualified = provider
            .claims(operation)
            .await
            .map_err(PdTransportError::InvalidRequest)?;
        self.http.check_origin(operation, &qualified.claims)?;
        encode_bounded(&qualified.claims)?;
        let body = prepared
            .sign(&self.http.secret, &qualified)
            .map_err(PdTransportError::InvalidRequest)?;
        let body = encode_bounded(&body)?;
        let packet = self.http.post(path, &qualified, body).await?;
        let prepared =
            PreparedControlResponse::new(packet).map_err(PdTransportError::InvalidReceipt)?;
        let clock = provider
            .clock_bounds()
            .await
            .map_err(PdTransportError::InvalidReceipt)?;
        let response = control::verify_response(
            &self.http.secret,
            &qualified.claims,
            expected,
            &prepared,
            &clock,
        )
        .map_err(|error| {
            tracing::warn!(target:"dynamo_llm::native_cleanup", operation,
                role=?expected.request.role, request_root=%expected.request.request_root,
                error=%error,"native PD control receipt verification failed");
            PdTransportError::InvalidReceipt(error)
        })?;
        Ok(VerifiedReceipt(response))
    }

    /// Exact-transaction abort is an idempotent control action, never a
    /// generation replay. An unknown abort is followed by fresh status; only a
    /// verified still-live transaction permits another exact abort operation.
    pub async fn cleanup(
        &self,
        expected: Arc<ExpectedTransaction>,
        claims: Arc<dyn PdClaimProvider>,
        deadline: Instant,
        poll_interval: Duration,
    ) -> Result<VerifiedReceipt, PdTransportError> {
        if poll_interval.is_zero() {
            return Err(PdTransportError::InvalidRequest(
                "cleanup poll interval must be positive".into(),
            ));
        }
        let cleanup = async {
            match self.abort(&expected, claims.as_ref()).await {
                Ok(receipt) if receipt.physical_terminal() => return Ok(receipt),
                Ok(_) => {}
                Err(error) if transient(&error) => {}
                Err(error) => return Err(error),
            }
            loop {
                match self.status(&expected, claims.as_ref()).await {
                    Ok(receipt) if receipt.physical_terminal() => return Ok(receipt),
                    Ok(_) => match self.abort(&expected, claims.as_ref()).await {
                        Ok(receipt) if receipt.physical_terminal() => return Ok(receipt),
                        Ok(_) => {}
                        Err(error) if transient(&error) => {}
                        Err(error) => return Err(error),
                    },
                    Err(error) if transient(&error) => {}
                    Err(error) => return Err(error),
                }
                tokio::time::sleep(poll_interval).await;
            }
        };
        tokio::time::timeout_at(deadline, cleanup)
            .await
            .map_err(|_| PdTransportError::Deadline)?
    }
}
struct ErrorDiagnosis {
    state: &'static str,
    detail: String,
}
async fn error_diagnosis(response: reqwest::Response, timeout: Duration) -> ErrorDiagnosis {
    const MAX_ERROR_BYTES: usize = 16 * 1024;
    let read = async {
        let mut bytes = Vec::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(_) => {
                    return ErrorDiagnosis {
                        state: "read_error",
                        detail: String::new(),
                    };
                }
            };
            if chunk.len() > MAX_ERROR_BYTES.saturating_sub(bytes.len()) {
                return ErrorDiagnosis {
                    state: "truncated",
                    detail: String::new(),
                };
            }
            bytes.extend_from_slice(&chunk);
        }
        if bytes.is_empty() {
            return ErrorDiagnosis {
                state: "empty",
                detail: String::new(),
            };
        }
        match serde_json::from_slice::<Value>(&bytes) {
            Ok(value) => ErrorDiagnosis {
                state: "complete_json",
                detail: error_fields(&value),
            },
            Err(_) => ErrorDiagnosis {
                state: "non_json",
                detail: String::new(),
            },
        }
    };
    tokio::time::timeout(timeout, read)
        .await
        .unwrap_or_else(|_| ErrorDiagnosis {
            state: "timeout",
            detail: String::new(),
        })
}
fn error_fields(value: &Value) -> String {
    // Use the maintained bounded/redacted native diagnostic filter, including
    // the current control handler's scalar {"error": "reason"} response.
    super::dispatch::native_error_fields_from_value(value)
        .map(|fields| fields.to_string())
        .unwrap_or_default()
}

#[cfg(test)]
mod cancel_diagnostics_tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[tokio::test]
    async fn cancel_control_error_body_actual_tcp_is_bounded_and_never_logs_proof() {
        for (body,hold_open,state) in [
            (br#"{"detail":"native owner binding conflicts","proof":{"mac":"secret"},"prompt_cache_canonical_body":"private prompt"}"#.to_vec(),false,"complete_json"),
            (br#"{"success":false,"error":"native owner binding conflicts","proof":{"mac":"secret"},"prompt_cache_canonical_body":"private prompt"}"#.to_vec(),false,"complete_json"),
            (b"opaque server text".to_vec(),false,"non_json"),
            (vec![b'x';16*1024+1],false,"truncated"),
            (Vec::new(),true,"timeout"),
        ] {
            let listener=tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address=listener.local_addr().unwrap();
            let size=body.len();
            let server=tokio::spawn(async move {
                let (mut socket,_)=listener.accept().await.unwrap();
                let mut request=[0;4096];
                socket.read(&mut request).await.unwrap();
                let content_length=if hold_open {1}else{size};
                socket.write_all(format!("HTTP/1.1 409 Conflict\r\nContent-Length: {content_length}\r\nConnection: close\r\n\r\n").as_bytes()).await.unwrap();
                socket.write_all(&body).await.unwrap();
                if hold_open {std::future::pending::<()>().await;}
            });
            let response=reqwest::Client::builder().no_proxy().build().unwrap().get(format!("http://{address}/v1/pd/transactions/abort")).send().await.unwrap();
            assert_eq!(response.status().as_u16(),409);
            let diagnosis=error_diagnosis(response,Duration::from_millis(20)).await;
            assert_eq!(diagnosis.state,state);
            if state=="complete_json" {
                assert!(diagnosis.detail.contains("native owner binding conflicts"));
            } else { assert!(diagnosis.detail.is_empty()); }
            assert!(!diagnosis.detail.contains("secret"));
            assert!(!diagnosis.detail.contains("private prompt"));
            server.abort();
            let _=server.await;
        }
    }
}
pub(crate) fn encode_bounded(value: &impl serde::Serialize) -> Result<Vec<u8>, PdTransportError> {
    struct Bounded(Vec<u8>);
    impl std::io::Write for Bounded {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if bytes.len() > MAX_RESPONSE_BYTES.saturating_sub(self.0.len()) {
                return Err(std::io::Error::other("control size limit"));
            }
            self.0.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut writer = Bounded(Vec::new());
    serde_json::to_writer(&mut writer, value).map_err(|_| PdTransportError::RequestTooLarge)?;
    Ok(writer.0)
}
fn unknown(error: reqwest::Error) -> PdTransportError {
    PdTransportError::Unknown {
        timed_out: error.is_timeout(),
    }
}
pub(crate) fn transient(error: &PdTransportError) -> bool {
    matches!(
        error,
        PdTransportError::Unknown { .. } | PdTransportError::HttpStatus(500..=599)
    )
}
