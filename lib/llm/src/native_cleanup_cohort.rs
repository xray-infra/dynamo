//! Current native startup cohort discovery. Participants come from complete
//! runtime registration; no participant is inferred from TP/DP/core counts.
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::Metadata,
    io::Read,
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use serde::{Deserialize, Serialize};

use crate::cache_authority::cache_auth;
use crate::cache_authority::frontend_qualification;
use crate::native_cleanup::NativeCleanupRegistry;
use crate::native_request_lifecycle::QualifiedRequestLifecycle;
use futures::StreamExt;

pub const CLEANUP_COHORT_METADATA_PATH: &str = "/v1/requests/cleanup-cohort";
const DEEPSEEK_CLEANUP_PROFILE_PATH: &str = "/v1/requests/cleanup-profile";
const DEEPSEEK_MODEL: &str = "deepseek-ai/DeepSeek-V4.1-Flash";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DeepseekCleanupProfile {
    // This is the cache profile's qualification flag, not ordinary admission.
    qualified: bool,
    profile: DeepseekProfileIdentity,
    ordinary: DeepseekOrdinaryEvidence,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DeepseekProfileIdentity {
    native_origin_uuid: String,
    source_digest: String,
    runtime_profile_digest: String,
    layout: serde_json::Value,
    family: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DeepseekOrdinaryEvidence {
    discovery: DeepseekOrdinaryDiscovery,
    registration_report: serde_json::Value,
    installation: DeepseekOrdinaryInstallation,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DeepseekOrdinaryDiscovery {
    artifact: CleanupCohortArtifact,
    layout: serde_json::Value,
    qualified: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DeepseekOrdinaryInstallation {
    artifact: CleanupCohortArtifact,
    artifact_digest: String,
    participant_ids: Vec<String>,
    cleanup_cohort_digest: String,
}

pub fn parse_current_artifact(model: &str, bytes: &[u8]) -> Result<CleanupCohortArtifact, String> {
    if model != DEEPSEEK_MODEL {
        return CleanupCohortArtifact::parse(bytes);
    }
    let profile: DeepseekCleanupProfile = serde_json::from_slice(bytes)
        .map_err(|_| "ordinary DeepSeek cleanup-profile fields mismatch")?;
    let DeepseekOrdinaryEvidence {
        discovery,
        installation,
        registration_report: _report,
    } = profile.ordinary;
    let artifact = installation.artifact;
    artifact.validate()?;
    if !discovery.qualified
        || discovery.artifact != artifact
        || installation.artifact_digest != artifact.artifact_digest()?
        || installation.participant_ids
            != artifact.participant_ids().into_iter().collect::<Vec<_>>()
        || installation.cleanup_cohort_digest != artifact.cleanup_cohort_digest
        || profile.profile.family != "deepseek"
        || profile.profile.native_origin_uuid != artifact.native_incarnation
        || profile.profile.source_digest != artifact.source_digest
        || profile.profile.runtime_profile_digest != artifact.runtime_profile_digest
        || profile.profile.layout != discovery.layout
    {
        return Err("ordinary DeepSeek installed cohort evidence mismatch".into());
    }
    let _cache_profile_qualified = profile.qualified;
    Ok(artifact)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CleanupCohortParticipant {
    pub participant_id: String,
    pub role: String,
    pub process_birth: String,
    pub source_digest: String,
    pub runtime_registration_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CleanupCohortArtifact {
    pub schema_version: u32,
    pub native_origin: String,
    pub native_incarnation: String,
    pub model: String,
    pub source_digest: String,
    pub runtime_profile_digest: String,
    pub cleanup_cohort_digest: String,
    pub participants: Vec<CleanupCohortParticipant>,
}

impl CleanupCohortArtifact {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let artifact: Self = serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1
            || !identity(&self.native_origin)
            || !identity(&self.native_incarnation)
            || !identity(&self.model)
            || !digest(&self.source_digest)
            || !digest(&self.runtime_profile_digest)
            || !digest(&self.cleanup_cohort_digest)
            || self.participants.is_empty()
        {
            return Err("cleanup cohort artifact identity/schema incomplete".into());
        }
        for participant in &self.participants {
            if !identity(&participant.participant_id)
                || !identity(&participant.role)
                || !identity(&participant.process_birth)
                || !digest(&participant.source_digest)
                || !digest(&participant.runtime_registration_digest)
            {
                return Err("cleanup cohort participant registration incomplete".into());
            }
        }
        if self
            .participants
            .windows(2)
            .any(|pair| pair[0].participant_id >= pair[1].participant_id)
        {
            return Err(
                "cleanup cohort participants must be unique and sorted by opaque ID".into(),
            );
        }
        let participants = self.participant_ids();
        let actual = cache_auth::digest(
            &serde_json::to_value(participants).map_err(|error| error.to_string())?,
        )?;
        if actual != self.cleanup_cohort_digest {
            return Err("cleanup cohort opaque participant digest mismatch".into());
        }
        Ok(())
    }

    pub fn participant_ids(&self) -> BTreeSet<String> {
        self.participants
            .iter()
            .map(|participant| participant.participant_id.clone())
            .collect()
    }

    /// Full canonical payload digest additionally binds role, process birth and
    /// registration evidence. It is distinct from the existing ID-set digest.
    pub fn artifact_digest(&self) -> Result<String, String> {
        self.validate()?;
        cache_auth::digest(&serde_json::to_value(self).map_err(|error| error.to_string())?)
    }
}

pub struct InstalledCleanupCohort {
    artifact: CleanupCohortArtifact,
    artifact_digest: String,
}

impl InstalledCleanupCohort {
    pub fn artifact(&self) -> &CleanupCohortArtifact {
        &self.artifact
    }
    pub fn artifact_digest(&self) -> &str {
        &self.artifact_digest
    }
    pub fn participant_ids(&self) -> BTreeSet<String> {
        self.artifact.participant_ids()
    }
    pub fn cleanup_cohort_digest(&self) -> &str {
        &self.artifact.cleanup_cohort_digest
    }
}

/// Install the current complete runtime registration. Actual request cleanup
/// still requires the signed, per-participant physical completion receipt.
pub fn install_cleanup_cohort(
    actual_startup_registration: CleanupCohortArtifact,
) -> Result<InstalledCleanupCohort, String> {
    let artifact_digest = actual_startup_registration.artifact_digest()?;
    Ok(InstalledCleanupCohort {
        artifact: actual_startup_registration,
        artifact_digest,
    })
}

/// Current worker address, native model identity and private signing key.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OrdinaryCleanupBindingFiles {
    pub existing_authority_secret_path: PathBuf,
    pub model: String,
    pub native_origin: String,
}

/// Secret bytes are neither Debug-printed nor serialized.
pub struct OrdinaryCleanupStartupMaterial {
    binding: OrdinaryCleanupBindingFiles,
    secret: [u8; 32],
}

pub fn load_ordinary_cleanup_material(
    binding: &OrdinaryCleanupBindingFiles,
) -> Result<OrdinaryCleanupStartupMaterial, String> {
    for value in [&binding.model, &binding.native_origin] {
        if !identity(value) {
            return Err("ordinary startup configured identity missing".into());
        }
    }
    validate_origin(&binding.native_origin)?;
    let secret_bytes = stable_bytes(&binding.existing_authority_secret_path)?;
    let secret: [u8; 32] = secret_bytes
        .try_into()
        .map_err(|_| "ordinary existing model key length mismatch")?;
    Ok(OrdinaryCleanupStartupMaterial {
        binding: binding.clone(),
        secret,
    })
}

impl OrdinaryCleanupStartupMaterial {
    pub fn model(&self) -> &str {
        &self.binding.model
    }

    pub fn native_origin(&self) -> &str {
        &self.binding.native_origin
    }

    /// Only the selected origin must be online. Each admission installs its
    /// actual current cohort; no prior runtime artifact stands in for discovery.
    pub async fn qualify_current(
        &self,
        client: &reqwest::Client,
        timeout: Duration,
        registry: NativeCleanupRegistry,
    ) -> Result<Arc<QualifiedRequestLifecycle>, String> {
        let bytes = load_current_discovery(
            client, self.model(), self.native_origin(), timeout,
        ).await?;
        self.install_from_discovery(&bytes, registry)
    }

    pub fn install_from_discovery(
        &self,
        discovery_bytes: &[u8],
        registry: NativeCleanupRegistry,
    ) -> Result<Arc<QualifiedRequestLifecycle>, String> {
        if discovery_bytes.len() > MAX_STARTUP_DOCUMENT as usize {
            return Err("ordinary installed discovery too large".into());
        }
        let artifact = parse_current_artifact(&self.binding.model, discovery_bytes)?;
        if artifact.model != self.binding.model
            || artifact.native_origin.trim_end_matches('/')
                != self.binding.native_origin.trim_end_matches('/')
        {
            return Err(
                "ordinary current runtime model/origin differs from configured worker".into(),
            );
        }
        let installed = install_cleanup_cohort(artifact)?;
        Ok(Arc::new(QualifiedRequestLifecycle::new(
            installed,
            self.secret,
            &self.binding.native_origin,
            registry,
        )?))
    }
}

/// Validate all configured origins and private keys without requiring every
/// worker to be online at frontend startup. Runtime qualification is selected
/// admission work, not a global availability condition.
pub fn load_ordinary_cleanup_bindings(
    bindings: &[OrdinaryCleanupBindingFiles],
) -> Result<BTreeMap<String, Arc<OrdinaryCleanupStartupMaterial>>, String> {
    if bindings.is_empty() {
        return Err("ordinary native startup requires explicit worker/key bindings".into());
    }
    let mut origins = BTreeSet::new();
    let mut materials = BTreeMap::new();
    for binding in bindings {
        let normalized = reqwest::Url::parse(&binding.native_origin)
            .map_err(|_| "ordinary configured native origin invalid")?
            .origin()
            .ascii_serialization();
        if !origins.insert(normalized.clone()) {
            return Err("duplicate ordinary configured native origin".into());
        }
        materials.insert(normalized, Arc::new(load_ordinary_cleanup_material(binding)?));
    }
    Ok(materials)
}

/// Exact-origin, bounded discovery before a selected worker signs a new
/// request (and for configured P/D roles). Never reuse a prior cohort.
pub(crate) async fn load_current_discovery(
    client: &reqwest::Client,
    model: &str,
    origin: &str,
    timeout: Duration,
) -> Result<Vec<u8>, String> {
    let endpoint = format!(
        "{}{}", origin.trim_end_matches('/'),
        if model == DEEPSEEK_MODEL { DEEPSEEK_CLEANUP_PROFILE_PATH }
        else { CLEANUP_COHORT_METADATA_PATH },
    );
    let response = client.get(&endpoint).timeout(timeout).send().await
        .map_err(|_| "ordinary installed discovery unavailable")?;
    if response.status() != reqwest::StatusCode::OK
        || response.url().as_str() != endpoint
        || response.content_length().is_some_and(|length| length > MAX_STARTUP_DOCUMENT)
    {
        return Err("ordinary installed discovery origin/status/size mismatch".into());
    }
    let mut bytes = Vec::new();
    let mut chunks = response.bytes_stream();
    while let Some(chunk) = chunks.next().await {
        let chunk = chunk.map_err(|_| "ordinary installed discovery body failed")?;
        if bytes.len().saturating_add(chunk.len()) > MAX_STARTUP_DOCUMENT as usize {
            return Err("ordinary installed discovery too large".into());
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

const MAX_STARTUP_DOCUMENT: u64 = 16 * 1024 * 1024;

fn stable_bytes(path: &Path) -> Result<Vec<u8>, String> {
    // Reuse existing flags/regular-file/eUID/readonly/size policy and keep that
    // exact validated FD for the full read. This is not a path reopen check.
    let file = frontend_qualification::open_private(path, true)?;
    let before = file
        .metadata()
        .map_err(|_| "ordinary startup FD metadata unavailable")?;
    if before.mode() & 0o7777 != 0o400 {
        return Err("ordinary existing key must be mode 0400".into());
    }
    let limit = 32;
    let mut bytes = Vec::new();
    (&file)
        .take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| "ordinary startup FD read failed")?;
    let after = file
        .metadata()
        .map_err(|_| "ordinary startup FD metadata unavailable")?;
    if !same_file_state(&before, &after)
        || bytes.len() as u64 != before.len()
        || bytes.len() as u64 > limit
    {
        return Err("ordinary startup readonly file changed during held-FD read".into());
    }
    Ok(bytes)
}

fn same_file_state(before: &Metadata, after: &Metadata) -> bool {
    before.dev() == after.dev()
        && before.ino() == after.ino()
        && before.len() == after.len()
        && before.blocks() == after.blocks()
        && before.mtime() == after.mtime()
        && before.mtime_nsec() == after.mtime_nsec()
        && before.ctime() == after.ctime()
        && before.ctime_nsec() == after.ctime_nsec()
        && before.mode() == after.mode()
        && before.nlink() == after.nlink()
        && before.uid() == after.uid()
}

fn validate_origin(value: &str) -> Result<(), String> {
    let url =
        reqwest::Url::parse(value).map_err(|_| "ordinary configured native origin invalid")?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || matches!(url.host_str(), Some("0.0.0.0" | "::" | "[::]"))
        || url.port_or_known_default().is_none_or(|port| port == 0)
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || !matches!(url.path(), "" | "/")
    {
        return Err("ordinary configured native origin must be one fixed worker".into());
    }
    Ok(())
}

fn identity(value: &str) -> bool {
    !value.trim().is_empty() && value.len() <= 1024
}
fn digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Synthetic CPU discovery tests still use the normal material factory and
    // actual HTTP artifact validation; production startup does no network I/O.
    async fn current_test_bindings(
        bindings: &[OrdinaryCleanupBindingFiles],
        registry: NativeCleanupRegistry,
    ) -> Result<BTreeMap<String, Arc<QualifiedRequestLifecycle>>, String> {
        let materials = load_ordinary_cleanup_bindings(bindings)?;
        let client = reqwest::Client::new();
        let mut installed = BTreeMap::new();
        for (origin, material) in materials {
            installed.insert(origin, material.qualify_current(&client, Duration::from_secs(3), registry.clone()).await?);
        }
        Ok(installed)
    }

    fn artifact() -> CleanupCohortArtifact {
        let participants = vec![
            CleanupCohortParticipant {
                participant_id: "opaque-a".into(),
                role: "tokenizer-owner".into(),
                process_birth: "actual-process-birth-a".into(),
                source_digest: "a".repeat(64),
                runtime_registration_digest: "b".repeat(64),
            },
            CleanupCohortParticipant {
                participant_id: "opaque-b".into(),
                role: "io-owner".into(),
                process_birth: "actual-process-birth-b".into(),
                source_digest: "c".repeat(64),
                runtime_registration_digest: "d".repeat(64),
            },
        ];
        let ids: BTreeSet<_> = participants
            .iter()
            .map(|participant| participant.participant_id.clone())
            .collect();
        CleanupCohortArtifact {
            schema_version: 1,
            native_origin: "http://native-worker:30000".into(),
            native_incarnation: "actual-startup-incarnation".into(),
            model: "qwen3.8-flash-next".into(),
            source_digest: "e".repeat(64),
            runtime_profile_digest: "f".repeat(64),
            cleanup_cohort_digest: cache_auth::digest(&serde_json::to_value(ids).unwrap()).unwrap(),
            participants,
        }
    }

    #[test]
    fn cleanup_cohort_install_uses_current_complete_runtime() {
        let actual = artifact();
        let installed = install_cleanup_cohort(actual.clone()).unwrap();
        assert_eq!(
            installed.cleanup_cohort_digest(),
            actual.cleanup_cohort_digest
        );
        assert_eq!(installed.participant_ids().len(), 2);
        let mut current = actual.clone();
        current.native_incarnation.push_str("-current");
        assert!(install_cleanup_cohort(current).is_ok());
        let mut missing = actual;
        missing.participants.pop();
        assert!(install_cleanup_cohort(missing).is_err());
    }

    #[test]
    fn cleanup_cohort_schema_rejects_unsorted_duplicate_empty_or_self_asserted_completion() {
        let actual = artifact();
        let mut unsorted = actual.clone();
        unsorted.participants.reverse();
        assert!(unsorted.validate().is_err());
        let mut duplicate = actual.clone();
        duplicate.participants[1] = duplicate.participants[0].clone();
        assert!(duplicate.validate().is_err());
        let mut empty = actual.clone();
        empty.participants.clear();
        assert!(empty.validate().is_err());
        let mut blank = actual.clone();
        blank.participants[1].role = " ".into();
        assert!(blank.validate().is_err());
        let mut bad_digest = actual.clone();
        bad_digest.participants[1].runtime_registration_digest = "A".repeat(64);
        assert!(bad_digest.validate().is_err());
        let mut self_asserted = serde_json::to_value(actual).unwrap();
        self_asserted["complete"] = serde_json::json!(true);
        assert!(
            CleanupCohortArtifact::parse(&serde_json::to_vec(&self_asserted).unwrap()).is_err()
        );
    }

    #[tokio::test]
    async fn deepseek_cleanup_profile_actual_http_installs_only_processed_cohort() {
        use axum::{Router, routing::get};
        use std::os::unix::fs::PermissionsExt;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let mut actual = artifact();
        actual.model = DEEPSEEK_MODEL.into();
        actual.native_origin = origin.clone();
        let layout = serde_json::json!({"tp_world_size":4});
        let packet = serde_json::json!({
            "qualified":false,
            "profile":{
                "native_origin_uuid":actual.native_incarnation,
                "source_digest":actual.source_digest,
                "runtime_profile_digest":actual.runtime_profile_digest,
                "layout":layout,"family":"deepseek"
            },
            "ordinary":{
                "discovery":{"artifact":actual,"layout":layout,"qualified":true},
                "registration_report":{"schema_version":1,"participants":[],"registrations":[]},
                "installation":{
                    "artifact":actual,"artifact_digest":actual.artifact_digest().unwrap(),
                    "participant_ids":actual.participant_ids(),
                    "cleanup_cohort_digest":actual.cleanup_cohort_digest
                }
            }
        });
        let router = Router::new().route(
            DEEPSEEK_CLEANUP_PROFILE_PATH,
            get({
                let packet = packet.clone();
                move || {
                    let packet = packet.clone();
                    async move { axum::Json(packet) }
                }
            }),
        );
        let cancel = tokio_util::sync::CancellationToken::new();
        let server = tokio::spawn({
            let cancel = cancel.clone();
            async move {
                axum::serve(listener, router)
                    .with_graceful_shutdown(cancel.cancelled_owned())
                    .await
                    .unwrap();
            }
        });
        let directory = tempfile::tempdir().unwrap();
        let secret_path = directory.path().join("model.key");
        std::fs::write(&secret_path, [29u8; 32]).unwrap();
        std::fs::set_permissions(&secret_path, std::fs::Permissions::from_mode(0o400)).unwrap();
        let binding = OrdinaryCleanupBindingFiles {
            existing_authority_secret_path: secret_path,
            model: DEEPSEEK_MODEL.into(),
            native_origin: origin.clone(),
        };
        let registry = NativeCleanupRegistry::new();
        let installed = current_test_bindings(&[binding], registry.clone())
            .await
            .unwrap();
        assert!(installed[&origin].registry().shares_registry(&registry));
        for (pointer, value) in [
            ("/ordinary/installation", serde_json::Value::Null),
            ("/ordinary/discovery/qualified", serde_json::json!(false)),
            (
                "/ordinary/installation/artifact_digest",
                serde_json::json!("0".repeat(64)),
            ),
            (
                "/ordinary/installation/participant_ids",
                serde_json::json!(["opaque-a", "opaque-a"]),
            ),
            (
                "/ordinary/installation/cleanup_cohort_digest",
                serde_json::json!("0".repeat(64)),
            ),
            (
                "/ordinary/discovery/artifact/native_incarnation",
                serde_json::json!("different-birth"),
            ),
        ] {
            let mut invalid = packet.clone();
            *invalid.pointer_mut(pointer).unwrap() = value;
            assert!(
                parse_current_artifact(DEEPSEEK_MODEL, &serde_json::to_vec(&invalid).unwrap())
                    .is_err(),
                "{pointer}"
            );
        }
        // DS has one current endpoint/shape, not a fallback to another family's artifact.
        assert!(
            parse_current_artifact(DEEPSEEK_MODEL, &serde_json::to_vec(&actual).unwrap()).is_err()
        );
        cancel.cancel();
        server.await.unwrap();
    }

    #[tokio::test]
    async fn ordinary_cleanup_startup_uses_current_runtime_and_private_key() {
        use axum::{Router, routing::get};
        use std::{
            fs,
            os::unix::fs::{PermissionsExt, symlink},
            sync::Mutex,
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let mut actual = artifact();
        actual.native_origin = origin.clone();
        let response = Arc::new(Mutex::new(serde_json::to_value(&actual).unwrap()));
        let router = Router::new().route(
            CLEANUP_COHORT_METADATA_PATH,
            get({
                let response = response.clone();
                move || {
                    let response = response.clone();
                    async move { axum::Json(response.lock().unwrap().clone()) }
                }
            }),
        );
        let cancel = tokio_util::sync::CancellationToken::new();
        let server = tokio::spawn({
            let cancel = cancel.clone();
            async move {
                axum::serve(listener, router)
                    .with_graceful_shutdown(cancel.cancelled_owned())
                    .await
                    .unwrap();
            }
        });
        let directory = tempfile::tempdir().unwrap();
        let secret_path = directory.path().join("model.key");
        fs::write(&secret_path, [29u8; 32]).unwrap();
        fs::set_permissions(&secret_path, fs::Permissions::from_mode(0o400)).unwrap();
        let binding = OrdinaryCleanupBindingFiles {
            existing_authority_secret_path: secret_path.clone(),
            model: actual.model.clone(),
            native_origin: origin.clone(),
        };
        let registry = NativeCleanupRegistry::new();
        let result =
            current_test_bindings(std::slice::from_ref(&binding), registry.clone())
                .await
                .unwrap();
        assert!(result[&origin].registry().shares_registry(&registry));
        response.lock().unwrap()["native_incarnation"] =
            serde_json::json!("current-restarted-runtime");
        assert!(
            current_test_bindings(std::slice::from_ref(&binding), registry.clone())
                .await
                .is_ok()
        );
        let complete = response.lock().unwrap().clone();
        response.lock().unwrap()["participants"] = serde_json::json!([]);
        assert!(
            current_test_bindings(std::slice::from_ref(&binding), registry.clone())
                .await
                .is_err()
        );
        *response.lock().unwrap() = complete;
        let mut alias = binding.clone();
        alias.model.push_str("-wrong");
        assert!(
            current_test_bindings(&[alias], registry.clone())
                .await
                .is_err()
        );
        fs::set_permissions(&secret_path, fs::Permissions::from_mode(0o600)).unwrap();
        assert!(load_ordinary_cleanup_material(&binding).is_err());
        fs::set_permissions(&secret_path, fs::Permissions::from_mode(0o400)).unwrap();
        let link = directory.path().join("key-link");
        symlink(&secret_path, &link).unwrap();
        let mut linked = binding;
        linked.existing_authority_secret_path = link;
        assert!(load_ordinary_cleanup_material(&linked).is_err());
        assert!(
            current_test_bindings(&[], registry)
                .await
                .is_err()
        );
        cancel.cancel();
        server.await.unwrap();
    }
}
