//! Worker startup identity and actual KV layout. No source closure or manual
//! weight approval is needed to enable the signed native cache protocol.
use super::cache_auth;
use super::frontend_cache_hook::NativeIdField;
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::io::Read;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, String>;
const MAX_DOCUMENT: u64 = 16 * 1024 * 1024;

/// Debug deliberately reveals no key bytes. This value is not serializable.
pub struct AuthoritySecret([u8; 32]);
impl std::fmt::Debug for AuthoritySecret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AuthoritySecret([REDACTED])")
    }
}
impl AuthoritySecret {
    pub(crate) fn bytes(&self) -> &[u8] {
        &self.0
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Deployment {
    pub model: String,
    pub native_origin: String,
    pub authority_secret_path: PathBuf,
}
#[derive(Clone, Debug)]
pub struct Profile {
    pub family: String,
    pub native_origin_uuid: String,
    pub layout: Value,
    pub runtime_profile_digest: String,
}
#[derive(Clone, Debug)]
pub struct ParsedProfile {
    pub profile: Profile,
    pub source_digest: String,
    pub kv_compat_digest: String,
    pub rank_ids: BTreeSet<u32>,
}
/// Actual runtime identity plus its private signing key. Native reserve and
/// completion still prove the physical tier holds and all-rank results.
#[derive(Debug)]
pub struct FrontendStartupMaterial {
    pub deployment: Deployment,
    pub profile: ParsedProfile,
    pub native_id_field: NativeIdField,
    secret: AuthoritySecret,
}
impl FrontendStartupMaterial {
    pub(crate) fn secret_bytes(&self) -> &[u8] {
        self.secret.bytes()
    }
}
pub fn load_startup_material(
    deployment_path: &Path, model: &str, native_origin: &str,
    family: &str, native_origin_uuid: &str, resolved_layout: Value,
) -> Result<FrontendStartupMaterial> {
    let deployment = load_deployment(deployment_path, model, native_origin)?;
    let profile = build_profile(family, model, native_origin_uuid, resolved_layout)?;
    let secret = load_secret(&deployment.authority_secret_path)?;
    let native_id_field = if family == "glm_flash" { NativeIdField::RequestId } else { NativeIdField::Rid };
    Ok(FrontendStartupMaterial { deployment, profile, native_id_field, secret })
}

fn bounded_text(s: &str) -> Result<()> {
    if s.is_empty() || s.len() > 1024 {
        return Err("invalid private deployment string".into());
    }
    Ok(())
}
pub(crate) fn open_private(path: &Path, secret: bool) -> Result<File> {
    if !path.is_absolute() {
        return Err("private startup path must be absolute".into());
    }
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(path)
        .map_err(|_| "private startup file unavailable".to_string())?;
    let info = file
        .metadata()
        .map_err(|_| "private startup metadata unavailable")?;
    if !info.is_file() {
        return Err("startup configuration must be a regular file".into());
    }
    let mode = info.mode() & 0o7777;
    if secret {
        if info.uid()!=unsafe{libc::geteuid()} || !matches!(mode, 0o400 | 0o600) || info.len() != 32 {
            return Err("authority secret requires mode 0400/0600 and exactly 32 bytes".into());
        }
    } else if info.len() > MAX_DOCUMENT {
        return Err("startup configuration exceeds bound".into());
    }
    Ok(file)
}
pub(crate) fn document_bytes(path: &Path) -> Result<Vec<u8>> {
    let file = open_private(path, false)?;
    let mut bytes = Vec::new();
    file.take(MAX_DOCUMENT + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| "startup file read failed")?;
    if bytes.len() as u64 > MAX_DOCUMENT {
        return Err("startup file too large".into());
    }
    Ok(bytes)
}
fn json_file(path: &Path) -> Result<Value> {
    serde_json::from_slice(&document_bytes(path)?).map_err(|_| "invalid startup JSON".into())
}
pub fn load_secret(path: &Path) -> Result<AuthoritySecret> {
    let mut file = open_private(path, true)?;
    let mut bytes = [0u8; 32];
    file.read_exact(&mut bytes)
        .map_err(|_| "authority secret read failed")?;
    let mut extra = [0u8; 1];
    if file
        .read(&mut extra)
        .map_err(|_| "authority secret read failed")?
        != 0
    {
        return Err("authority secret changed while reading".into());
    }
    Ok(AuthoritySecret(bytes))
}
pub fn load_deployment(path: &Path, model: &str, native_origin: &str) -> Result<Deployment> {
    let d: Deployment =
        serde_json::from_value(json_file(path)?).map_err(|_| "deployment shape mismatch")?;
    if d.model != model || d.native_origin != native_origin {
        return Err("deployment differs from configured model/native origin".into());
    }
    for s in [&d.model, &d.native_origin] {
        bounded_text(s)?;
    }
    if !d.authority_secret_path.is_absolute() {
        return Err("secret path must be absolute".into());
    }
    let url = reqwest::Url::parse(native_origin).map_err(|_| "invalid native origin")?;
    if !matches!(url.scheme(), "http" | "https")
        || native_origin.bytes().any(|b| b <= 32 || b == 127)
        || url.port_or_known_default().is_none_or(|port| port == 0)
        || url.host_str().is_none()
        || matches!(url.host_str(), Some("0.0.0.0" | "::" | "[::]"))
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || !matches!(url.path(), "" | "/")
    {
        return Err("native origin must be a fixed qualified HTTP worker URL".into());
    }
    Ok(d)
}
/// Matches Python ensure_ascii=True for the stable byte-ABI descriptor.
pub(crate) fn kv_digest(value: &Value) -> Result<String> {
    let bytes = cache_auth::canonical_bytes(value)?;
    let text = std::str::from_utf8(&bytes).map_err(|_| "invalid descriptor UTF-8")?;
    let mut ascii = String::new();
    for ch in text.chars() {
        if ch.is_ascii() {
            ascii.push(ch);
        } else {
            for unit in ch.encode_utf16(&mut [0u16; 2]).iter() {
                ascii.push_str(&format!("\\u{unit:04x}"));
            }
        }
    }
    Ok(format!("{:x}", Sha256::digest(ascii.as_bytes())))
}
/// The native runtime supplies its resolved layout. Bind identity to actual KV
/// geometry instead of a file snapshot, source list or weight approval receipt.
pub fn build_profile(family: &str, model: &str, native_origin_uuid: &str, resolved_layout: Value) -> Result<ParsedProfile> {
    if !matches!(family, "glm" | "glm_flash" | "qwen" | "deepseek") {
        return Err("profile family mismatch".into());
    }
    bounded_text(model)?;
    bounded_text(native_origin_uuid)?;
    let source_digest = cache_auth::digest(&serde_json::json!({"family":family,"model":model}))?;
    let runtime_profile_digest = cache_auth::digest(&serde_json::json!({"family":family,"model":model,"layout":resolved_layout}))?;
    let p = Profile { family:family.into(), native_origin_uuid:native_origin_uuid.into(), layout:resolved_layout, runtime_profile_digest };
    let layout = p
        .layout
        .as_object()
        .ok_or("missing resolved native layout")?;
    let descriptor = layout
        .get("kv_compat_descriptor")
        .filter(|v| v.as_object().is_some_and(|o| !o.is_empty()))
        .ok_or("missing actual qualified KV compatibility descriptor")?;
    let key = if matches!(p.family.as_str(), "glm" | "glm_flash") {
        "expected_rank_ids"
    } else {
        "rank_ids"
    };
    let ids = layout
        .get(key)
        .and_then(Value::as_array)
        .filter(|a| !a.is_empty())
        .ok_or("missing exact actual native rank set")?;
    let mut rank_ids = BTreeSet::new();
    for id in ids {
        let rank = id
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .ok_or("rank IDs must be strict nonnegative u32")?;
        if !rank_ids.insert(rank) {
            return Err("duplicate qualified rank ID".into());
        }
    }
    Ok(ParsedProfile {
        kv_compat_digest: kv_digest(descriptor)?,
        source_digest,
        rank_ids,
        profile: p,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::{PermissionsExt, symlink};
    #[test]
    fn native_cache_runtime_identity_without_manual_receipt() {
        let root = std::env::temp_dir().join(format!("cache-config-{}",uuid::Uuid::new_v4()));
        std::fs::create_dir(&root).unwrap();
        let secret=root.join("key"); std::fs::write(&secret,[17;32]).unwrap();
        std::fs::set_permissions(&secret,std::fs::Permissions::from_mode(0o600)).unwrap();
        let deployment=root.join("deployment.json");
        let config=serde_json::json!({"model":"actual-model","native_origin":"http://127.0.0.1:8123","authority_secret_path":secret});
        std::fs::write(&deployment,serde_json::to_vec(&config).unwrap()).unwrap();
        std::fs::set_permissions(&deployment,std::fs::Permissions::from_mode(0o400)).unwrap();
        let layout=serde_json::json!({"rank_ids":[0,1],"kv_compat_descriptor":{"schema":"CPU-NOTQUALIFIED","dtype":"fp8"}});
        let material=load_startup_material(&deployment,"actual-model","http://127.0.0.1:8123","qwen","actual-boot",layout.clone()).unwrap();
        assert_eq!(material.secret_bytes(),&[17;32]);
        assert_eq!(material.profile.rank_ids,BTreeSet::from([0,1]));
        assert_eq!(material.profile.source_digest,cache_auth::digest(&serde_json::json!({"family":"qwen","model":"actual-model"})).unwrap());
        let other=build_profile("qwen","actual-model","next-boot",layout).unwrap();
        assert_eq!(material.profile.kv_compat_digest,other.kv_compat_digest);
        assert_ne!(material.profile.profile.native_origin_uuid,other.profile.native_origin_uuid);
        assert!(load_deployment(&deployment,"actual-model","http://127.0.0.1:8124").is_err());
        assert!(build_profile("qwen","actual-model","boot",serde_json::json!({"rank_ids":[0,0],"kv_compat_descriptor":{"x":1}})).is_err());
        let link=root.join("link");symlink(&secret,&link).unwrap();assert!(load_secret(&link).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
