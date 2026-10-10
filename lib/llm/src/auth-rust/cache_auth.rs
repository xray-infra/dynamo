use hmac::{Hmac, Mac};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

const DOMAIN: &[u8] = b"dynamo-native-prompt-cache-v1\0";
pub const PROOF_FIELD: &str = "prompt_cache_authority_proof";

pub fn ordinary_body_for(body: &Value) -> Result<Value, String> {
    let object = body.as_object().ok_or("body must be an object")?;
    let mut result = Map::new();
    for (key, value) in object {
        if !key.starts_with("prompt_cache_") && !matches!(key.as_str(),
            "reservation_id" | "authority_revision" | "cache_request_started_at_ms" |
            "retain_for_ms" | "_prompt_cache_verified_claims" |
            "bootstrap_host" | "bootstrap_port" | "bootstrap_room" | "disagg_prefill_dp_rank") {
            result.insert(key.clone(), value.clone());
        }
    }
    Ok(Value::Object(result))
}

pub fn attach_canonical_body(body: &mut Value) -> Result<(), String> {
    if body.get("prompt_cache_canonical_body").is_some() { return Err("canonical body already supplied".into()); }
    // Exact ordinary HTTP JSON bytes as serialized by the same serde_json
    // implementation used for the native POST. Floats are inside this protected
    // string and never reserialized by the typed cross-language HMAC codec.
    let source = serde_json::to_string(&ordinary_body_for(body)?).map_err(|e| e.to_string())?;
    body.as_object_mut().ok_or("body must be an object")?
        .insert("prompt_cache_canonical_body".into(), Value::String(source));
    Ok(())
}

fn ordered(value: &Value) -> Result<Value, String> {
    match value {
        Value::Number(n) if n.as_i64().is_none() => Err("exact bounded integer required".into()),
        Value::Array(items) => Ok(Value::Array(items.iter().map(ordered).collect::<Result<_, _>>()?)),
        Value::Object(items) => {
            let mut keys: Vec<_> = items.keys().collect();
            keys.sort();
            let mut result = Map::new();
            for key in keys { result.insert(key.clone(), ordered(&items[key])?); }
            Ok(Value::Object(result))
        }
        _ => Ok(value.clone()),
    }
}

pub fn canonical_bytes(value: &Value) -> Result<Vec<u8>, String> {
    serde_json::to_vec(&ordered(value)?).map_err(|e| e.to_string())
}

pub fn digest(value: &Value) -> Result<String, String> {
    Ok(hex(&Sha256::digest(canonical_bytes(value)?)))
}

fn canonical_ascii_bytes(value: &Value) -> Result<Vec<u8>, String> {
    // Match the existing native Python json.dumps ensure_ascii=True codec,
    // including UTF-16 surrogate pairs for supplementary characters.
    let json = String::from_utf8(canonical_bytes(value)?).map_err(|e| e.to_string())?;
    let mut ascii = String::new();
    for c in json.chars() {
        if c.is_ascii() { ascii.push(c); }
        else {
            let mut units = [0u16; 2];
            for unit in c.encode_utf16(&mut units) { ascii.push_str(&format!("\\u{unit:04x}")); }
        }
    }
    Ok(ascii.into_bytes())
}

pub fn kv_compat_digest(descriptor: &Value) -> Result<String, String> {
    if descriptor.as_object().filter(|v| !v.is_empty()).is_none() {
        return Err("missing actual validated native KV compatibility descriptor".into());
    }
    Ok(hex(&Sha256::digest(canonical_ascii_bytes(descriptor)?)))
}

pub fn namespace_for(raw_salt: &str, extra_key: Option<&str>, compat_digest: &str) -> Result<String, String> {
    if raw_salt.is_empty() || raw_salt.len() > 1024 || compat_digest.len() != 64 ||
        !compat_digest.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err("invalid actual native tenant salt/compatibility digest".into());
    }
    let value = serde_json::json!({"cache_salt": raw_salt, "extra_key": extra_key,
        "adapter": "base-no-lora", "kv_compat_digest": compat_digest});
    Ok(hex(&Sha256::digest(canonical_ascii_bytes(&value)?)))
}

fn hex(bytes: &[u8]) -> String { bytes.iter().map(|v| format!("{v:02x}")).collect() }

pub fn payload_for(operation: &str, body: &Value) -> Result<Value, String> {
    let object = body.as_object().ok_or("body must be an object")?;
    let control = matches!(operation, "reserve" | "cancel" | "receipts" | "pd-status" | "pd-abort" | "request-status" | "request-abort");
    if !control && !matches!(operation, "generate" | "render" | "request-generate") { return Err("unknown operation".into()); }
    let mut result = Map::new();
    for (key, value) in object {
        if key == PROOF_FIELD { continue; }
        if control || key.starts_with("prompt_cache_") || matches!(key.as_str(), "model" | "rid" | "request_id" | "cache_salt" | "cache_request_started_at_ms" | "authority_revision" | "reservation_id" |
            "bootstrap_host" | "bootstrap_port" | "bootstrap_room" | "disagg_prefill_dp_rank") {
            result.insert(key.clone(), value.clone());
        }
    }
    Ok(Value::Object(result))
}

pub fn validate_claims(claims: &Value) -> Result<(), String> {
    let c = claims.as_object().ok_or("claims must be an object")?;
    let fields = ["version", "operation", "operation_id", "nonce", "model", "native_origin", "native_incarnation", "source_digest", "runtime_profile_digest", "reservation_id", "authority_revision", "native_request_id", "namespace_digest", "full_render_digest", "token_ids_digest", "request_started_at_ms", "issued_at_ms", "deadline_ms", "retain_for_ms", "payload_digest"];
    if c.len() != fields.len() || fields.iter().any(|k| !c.contains_key(*k)) { return Err("claim fields mismatch".into()); }
    let int_keys = ["version", "authority_revision", "request_started_at_ms", "issued_at_ms", "deadline_ms", "retain_for_ms"];
    for key in fields {
        if int_keys.contains(&key) {
            if c[key].as_i64().filter(|n| *n >= 0).is_none() { return Err("integer claim mismatch".into()); }
        } else if c[key].as_str().filter(|s| s.len() <= 1024).is_none() { return Err("string claim mismatch".into()); }
    }
    if c["version"] != 1 || c["retain_for_ms"].as_u64().unwrap() > 3_600_000 { return Err("version or retention mismatch".into()); }
    let op = c["operation"].as_str().unwrap();
    if !matches!(op, "reserve" | "cancel" | "receipts" | "generate" | "render" | "pd-status" | "pd-abort" | "request-generate" | "request-status" | "request-abort") { return Err("unknown operation".into()); }
    let request_control = matches!(op, "request-generate" | "request-status" | "request-abort" | "pd-status" | "pd-abort");
    if request_control && (c["authority_revision"] != 0 || c["retain_for_ms"] != 0 || c["reservation_id"] != "" ||
        ["namespace_digest", "full_render_digest", "token_ids_digest"].iter().any(|key| c[*key] != "")) {
        return Err("ordinary lifecycle cannot claim cache reservation or rendered-token evidence".into());
    }
    for key in ["source_digest", "runtime_profile_digest", "namespace_digest", "payload_digest", "full_render_digest", "token_ids_digest"] {
        let s = c[key].as_str().unwrap();
        if request_control && matches!(key, "namespace_digest" | "full_render_digest" | "token_ids_digest") { continue; }
        if op == "render" && matches!(key, "full_render_digest" | "token_ids_digest") && s.is_empty() { continue; }
        if s.len() != 64 || !s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) { return Err("digest mismatch".into()); }
    }
    for key in ["operation_id", "nonce", "model", "native_origin", "native_incarnation", "native_request_id"] {
        if c[key].as_str().unwrap().is_empty() { return Err("empty identity".into()); }
    }
    let start = c["request_started_at_ms"].as_u64().unwrap();
    let issued = c["issued_at_ms"].as_u64().unwrap();
    let deadline = c["deadline_ms"].as_u64().unwrap();
    if !(start <= issued && issued <= deadline) { return Err("time order mismatch".into()); }
    if op != "render" && !request_control && (c["reservation_id"].as_str().unwrap().is_empty() || c["authority_revision"].as_u64().unwrap() == 0) { return Err("empty reservation".into()); }
    Ok(())
}

/// Internal immutable payload snapshot. Construction performs the existing
/// canonical digest traversal before a fresh signing clock is obtained. There
/// is no arbitrary-digest constructor or mutable access to its owned payload.
pub(crate) struct PrehashedPayload {
    _owned_payload: Value,
    digest: String,
}
impl PrehashedPayload {
    pub(crate) fn new(payload: &Value) -> Result<Self, String> {
        let owned_payload = payload.clone();
        let digest = digest(&owned_payload)?;
        Ok(Self { _owned_payload: owned_payload, digest })
    }
    pub(crate) fn digest(&self) -> &str { &self.digest }
}

/// The caller qualifies its clock immediately before this small-claims path.
/// Full payload traversal has already completed in the trusted typed builder.
pub(crate) fn sign_prehashed(secret: &[u8], claims: &Value, payload: &PrehashedPayload) -> Result<Value, String> {
    if secret.len() != 32 { return Err("secret length mismatch".into()); }
    claims.as_object().ok_or("claims must be object")?;
    if claims.get("payload_digest").and_then(Value::as_str) != Some(payload.digest()) {
        return Err("prehashed payload binding mismatch".into());
    }
    validate_claims(claims)?;
    let mut mac = Hmac::<Sha256>::new_from_slice(secret).map_err(|e| e.to_string())?;
    mac.update(DOMAIN);
    mac.update(&canonical_bytes(claims)?);
    Ok(serde_json::json!({"claims": claims, "mac": hex(&mac.finalize().into_bytes())}))
}

pub fn sign(secret: &[u8], claims: &Value, payload: &Value) -> Result<Value, String> {
    if secret.len() != 32 { return Err("secret length mismatch".into()); }
    let mut bound = claims.as_object().ok_or("claims must be object")?.clone();
    let payload = PrehashedPayload::new(payload)?;
    bound.insert("payload_digest".into(), Value::String(payload.digest().into()));
    sign_prehashed(secret, &Value::Object(bound), &payload)
}

/// Verify only the bounded claims/MAC after a fresh receive-clock sample. The
/// bounded parsed payload was already hashed by the trusted typed builder.
pub(crate) fn verify_prehashed(secret: &[u8], proof: &Value, payload: &PrehashedPayload) -> Result<Value, String> {
    let p = proof.as_object().ok_or("proof must be object")?;
    if p.len() != 2 || !p.contains_key("claims") || !p.contains_key("mac") { return Err("proof fields mismatch".into()); }
    if secret.len() != 32 { return Err("secret length mismatch".into()); }
    p["claims"].as_object().ok_or("claims must be object")?;
    if p["claims"].get("payload_digest").and_then(Value::as_str) != Some(payload.digest()) {
        return Err("payload mismatch".into());
    }
    validate_claims(&p["claims"])?;
    let hex_mac = p["mac"].as_str().ok_or("signature must be string")?;
    if hex_mac.len() != 64 { return Err("signature length mismatch".into()); }
    let mut bytes = Vec::with_capacity(32);
    for pair in hex_mac.as_bytes().chunks_exact(2) {
        let s = std::str::from_utf8(pair).map_err(|e| e.to_string())?;
        bytes.push(u8::from_str_radix(s, 16).map_err(|e| e.to_string())?);
    }
    let mut mac = Hmac::<Sha256>::new_from_slice(secret).map_err(|e| e.to_string())?;
    mac.update(DOMAIN);
    mac.update(&canonical_bytes(&p["claims"])?);
    mac.verify_slice(&bytes).map_err(|_| "signature mismatch")?;
    Ok(p["claims"].clone())
}

pub fn verify_signature(secret: &[u8], proof: &Value, payload: &Value) -> Result<Value, String> {
    let payload = PrehashedPayload::new(payload)?;
    verify_prehashed(secret, proof, &payload)
}
