//! Current P/D role discovery and byte compatibility.
//! Generation and control bind the discovered runtime identities using the model key.
use std::{collections::BTreeSet, path::PathBuf};

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Value, json};

use super::control::{PdRole, PhysicalRank};
use crate::cache_authority::{
    cache_auth,
    frontend_qualification::{self, AuthoritySecret},
};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PdRoleMaterialFiles {
    pub authority_secret_path: PathBuf,
}

fn required_option<'de, D: Deserializer<'de>, T: Deserialize<'de>>(
    deserializer: D,
) -> Result<Option<T>, D::Error> {
    Option::<T>::deserialize(deserializer)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeRoleRegistration {
    pub tp_rank: u32,
    pub dcp_rank: u32,
    pub replica_rank: u32,
    pub native_rank_incarnation: String,
    pub pid: u64,
    pub process_start_ticks: u64,
    pub boot_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PdRoleArtifact {
    pub schema_version: u32,
    pub role: PdRole,
    pub model: String,
    pub native_origin: String,
    pub native_incarnation: String,
    pub source_digest: String,
    pub runtime_profile_digest: String,
    pub weights_attestation_digest: String,
    pub kv_compat_digest: String,
    pub qualified_rank_set_digest: String,
    pub ranks: Vec<PhysicalRank>,
    #[serde(deserialize_with = "required_option")]
    pub prefill_bootstrap_host: Option<String>,
    #[serde(deserialize_with = "required_option")]
    pub prefill_bootstrap_port: Option<u16>,
    #[serde(deserialize_with = "required_option")]
    pub prefill_bootstrap_bind_host: Option<String>,
    pub native_role_registration_digest: String,
    pub registrations: Vec<NativeRoleRegistration>,
    pub kv_compat_descriptor: Value,
}

impl PdRoleArtifact {
    pub fn artifact_digest(&self) -> Result<String, String> {
        cache_auth::digest(&serde_json::to_value(self).map_err(|error| error.to_string())?)
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1 {
            return Err("PD role artifact version mismatch".into());
        }
        for value in [&self.model, &self.native_origin, &self.native_incarnation] {
            text(value)?;
        }
        origin(&self.native_origin)?;
        for value in [
            &self.source_digest,
            &self.runtime_profile_digest,
            &self.weights_attestation_digest,
            &self.kv_compat_digest,
            &self.qualified_rank_set_digest,
            &self.native_role_registration_digest,
        ] {
            hash(value)?;
        }
        if self.ranks.is_empty() || self.ranks.windows(2).any(|ranks| ranks[0] >= ranks[1]) {
            return Err("PD actual ranks must be complete, unique and sorted".into());
        }
        let coordinates: Vec<_> = self
            .ranks
            .iter()
            .map(|rank| [rank.tp_rank, rank.dcp_rank, rank.replica_rank])
            .collect();
        if cache_auth::digest(&json!(coordinates))? != self.qualified_rank_set_digest {
            return Err("PD rank set digest mismatch".into());
        }
        if self
            .kv_compat_descriptor
            .as_object()
            .is_none_or(|descriptor| descriptor.is_empty())
            || frontend_qualification::kv_digest(&self.kv_compat_descriptor)?
                != self.kv_compat_digest
        {
            return Err("PD complete native byte descriptor digest mismatch".into());
        }
        let geometry = |key| self.kv_compat_descriptor.get(key)
            .and_then(Value::as_u64).filter(|size| *size > 0)
            .ok_or("PD actual attention ownership missing or invalid");
        if geometry("attn_tp_size")?.checked_mul(geometry("attn_cp_size")?)
            != Some(geometry("tp_size")?)
        {
            return Err("PD actual attention ownership differs from TP size".into());
        }
        // Geometry and dtype/shape/stride rows are validated against actual
        // buffers by the native startup codec, not a frontend TP/device count.
        if self.registrations.len() != self.ranks.len() {
            return Err("PD complete startup registration coverage missing".into());
        }
        let (mut births, mut incarnations) = (BTreeSet::new(), BTreeSet::new());
        for (registration, rank) in self
            .registrations
            .iter()
            .zip(&self.ranks)
        {
            if registration.tp_rank != rank.tp_rank
                || registration.dcp_rank != rank.dcp_rank
                || registration.replica_rank != rank.replica_rank
            {
                return Err("PD ordered registration differs from actual ranks".into());
            }
            for value in [
                &registration.native_rank_incarnation,
                &registration.boot_id,
            ] {
                text(value)?;
            }
            if registration.pid == 0
                || registration.process_start_ticks == 0
                || !births.insert((
                    &registration.boot_id,
                    registration.pid,
                    registration.process_start_ticks,
                ))
                || !incarnations.insert(&registration.native_rank_incarnation)
            {
                return Err(
                    "PD actual process registration invalid or repeated".into(),
                );
            }
        }
        if cache_auth::digest(&json!({"ranks":self.ranks,"registrations":self.registrations}))?
            != self.native_role_registration_digest
            || self.native_incarnation != self.registrations[0].native_rank_incarnation
        {
            return Err("PD complete registration digest mismatch".into());
        }
        match self.role {
            PdRole::Prefill => {
                let advertised = self
                    .prefill_bootstrap_host
                    .as_deref()
                    .ok_or("PD advertised bootstrap host missing")?;
                text(advertised)?;
                text(
                    self.prefill_bootstrap_bind_host
                        .as_deref()
                        .ok_or("PD actual bootstrap bind missing")?,
                )?;
                if matches!(advertised, "0.0.0.0" | "::")
                    || self.prefill_bootstrap_port.is_none_or(|port| port == 0)
                    || host(&origin(&self.native_origin)?).as_deref() != Some(advertised)
                {
                    return Err("PD advertised bootstrap differs from approved own origin".into());
                }
            }
            PdRole::Decode
                if self.prefill_bootstrap_host.is_none()
                    && self.prefill_bootstrap_port.is_none()
                    && self.prefill_bootstrap_bind_host.is_none() => {}
            PdRole::Decode => return Err("PD decode cannot declare prefill listener".into()),
        }
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RoleDiscovery {
    artifact: PdRoleArtifact,
    qualified: bool,
}

pub struct VerifiedPdRole {
    artifact: PdRoleArtifact,
    artifact_digest: String,
    secret: AuthoritySecret,
}
impl VerifiedPdRole {
    pub fn artifact(&self) -> &PdRoleArtifact {
        &self.artifact
    }
    pub fn artifact_digest(&self) -> &str {
        &self.artifact_digest
    }
    pub(crate) fn secret_bytes(&self) -> &[u8] {
        self.secret.bytes()
    }
}

fn role_from_discovery(
    bytes: &[u8],
    secret: AuthoritySecret,
    model: &str,
    native_origin: &str,
    expected_role: PdRole,
) -> Result<VerifiedPdRole, String> {
    let discovery: RoleDiscovery =
        serde_json::from_slice(bytes).map_err(|_| "PD current role discovery fields mismatch")?;
    discovery.artifact.validate()?;
    if !discovery.qualified
        || discovery.artifact.role != expected_role
        || discovery.artifact.model != model
        || discovery.artifact.native_origin != native_origin
    {
        return Err(
            "PD current runtime role/model/origin mismatch or incomplete registration".into(),
        );
    }
    let artifact_digest = discovery.artifact.artifact_digest()?;
    Ok(VerifiedPdRole {
        artifact: discovery.artifact,
        artifact_digest,
        secret,
    })
}

pub async fn load_pd_role(
    files: &PdRoleMaterialFiles,
    model: &str,
    native_origin: &str,
    expected_role: PdRole,
) -> Result<VerifiedPdRole, String> {
    let mut url = origin(native_origin)?;
    url.set_path("/v1/pd/profile");
    let secret = frontend_qualification::load_secret(&files.authority_secret_path)?;
    let client = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .retry(reqwest::retry::never())
        .connect_timeout(std::time::Duration::from_secs(5))
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .map_err(|error| error.to_string())?;
    let mut response = client
        .get(url)
        .send()
        .await
        .map_err(|error| error.to_string())?;
    if !response.status().is_success() {
        return Err(format!(
            "PD current role discovery HTTP {}",
            response.status()
        ));
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|error| error.to_string())? {
        if bytes.len() + chunk.len() > 16 * 1024 * 1024 {
            return Err("PD current role discovery exceeds limit".into());
        }
        bytes.extend_from_slice(&chunk);
    }
    role_from_discovery(&bytes, secret, model, native_origin, expected_role)
}

pub struct VerifiedPdPair {
    prefill: VerifiedPdRole,
    decode: VerifiedPdRole,
}
impl VerifiedPdPair {
    pub fn from_roles(prefill: VerifiedPdRole, decode: VerifiedPdRole) -> Result<Self, String> {
        if prefill.artifact.role != PdRole::Prefill
            || decode.artifact.role != PdRole::Decode
            || prefill.secret_bytes() != decode.secret_bytes()
        {
            return Err("PD pair requires both current roles and the same model HMAC key".into());
        }
        pair_artifacts(&prefill.artifact, &decode.artifact)?;
        Ok(Self { prefill, decode })
    }
    pub fn prefill(&self) -> &VerifiedPdRole {
        &self.prefill
    }
    pub fn decode(&self) -> &VerifiedPdRole {
        &self.decode
    }
}

fn pair_artifacts(own: &PdRoleArtifact, peer: &PdRoleArtifact) -> Result<(), String> {
    own.validate()?;
    peer.validate()?;
    if own.role == peer.role
        || own.model != peer.model
        || own.native_origin == peer.native_origin
        || own.native_incarnation == peer.native_incarnation
        || transfer_descriptor(own) != transfer_descriptor(peer)
    {
        return Err("PD current own/peer role, model, ranks or byte compatibility differ".into());
    }
    Ok(())
}

fn transfer_descriptor(artifact: &PdRoleArtifact) -> serde_json::Map<String, Value> {
    artifact.kv_compat_descriptor.as_object().expect("validated descriptor").iter()
        .filter(|(key, _)| !matches!(key.as_str(), "tp_size" | "pp_size" | "dcp_size" | "logical_page" | "attn_tp_size" | "attn_cp_size"))
        .map(|(key, value)| (key.clone(), value.clone())).collect()
}

fn text(value: &str) -> Result<(), String> {
    if value.is_empty() || value.len() > 4096 {
        return Err("PD startup text missing or unbounded".into());
    }
    Ok(())
}
fn hash(value: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("PD startup digest invalid".into());
    }
    Ok(())
}
fn host(url: &reqwest::Url) -> Option<String> {
    url.host_str().map(|host| {
        host.trim_start_matches('[')
            .trim_end_matches(']')
            .to_owned()
    })
}
fn origin(value: &str) -> Result<reqwest::Url, String> {
    let url = reqwest::Url::parse(value).map_err(|_| "PD fixed native origin invalid")?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || matches!(host(&url).as_deref(), Some("0.0.0.0" | "::"))
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || !matches!(url.path(), "" | "/")
        || url.port_or_known_default().is_none_or(|port| port == 0)
    {
        return Err("PD native origin must be a fixed worker URL".into());
    }
    Ok(url)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, os::unix::fs::PermissionsExt, path::Path};

    #[derive(Deserialize)]
    struct RoleReceipt {
        artifact: PdRoleArtifact,
        peer_artifact: PdRoleArtifact,
        artifact_digest: String,
        peer_artifact_digest: String,
    }
    fn native_fixture() -> RoleReceipt {
        // Exact Python native-codec fixture; CPU synthetic and not production.
        serde_json::from_str(include_str!("role-qualification-v1.synthetic.json")).unwrap()
    }

    #[test]
    fn current_role_registrations_use_process_birth_without_transport_snapshot() {
        use crate::native_pd::control::ControlRequest;
        use crate::native_pd::dispatch::{GenerationBinding, PdGenerationTransport, RoomRange};
        // CPU schema projection of saved native rank/byte/birth metadata:
        // Obsolete transport snapshots/maps are removed, actual attention
        // ownership is projected, and digests are recomputed.
        // This is not a claim that the new native profile has already run.
        // The loader/binding are production types; the HMAC key is CPU fixture.
        let profiles: Value = serde_json::from_str(include_str!("role-qualification-current.json")).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let secret = directory.path().join("secret");
        write(&secret, &[13; 32], 0o400);
        let load = |name: &str, role| {
            let profile = &profiles[name];
            role_from_discovery(&serde_json::to_vec(profile).unwrap(),
                frontend_qualification::load_secret(&secret).unwrap(),
                profile["artifact"]["model"].as_str().unwrap(),
                profile["artifact"]["native_origin"].as_str().unwrap(), role).unwrap()
        };
        let pair = VerifiedPdPair::from_roles(load("prefill", PdRole::Prefill), load("decode", PdRole::Decode)).unwrap();
        let p = pair.prefill().artifact();
        let d = pair.decode().artifact();
        assert_ne!(p.qualified_rank_set_digest, d.qualified_rank_set_digest);
        assert_ne!(p.kv_compat_digest, d.kv_compat_digest);
        assert_eq!(p.kv_compat_descriptor["dcp_size"], json!(1));
        assert_eq!(d.kv_compat_descriptor["dcp_size"], json!(4));
        assert_eq!(p.kv_compat_descriptor["attn_tp_size"], json!(2));
        assert_eq!(p.kv_compat_descriptor["attn_cp_size"], json!(4));
        assert_eq!(d.kv_compat_descriptor["attn_tp_size"], json!(8));
        assert_eq!(d.kv_compat_descriptor["attn_cp_size"], json!(1));
        // CPU projection of the same decode processes with DCP1. Routing is
        // derived by the native authenticated peer installation, not startup.
        let mut dcp1 = d.clone();
        dcp1.kv_compat_descriptor["dcp_size"] = json!(1);
        dcp1.kv_compat_descriptor["logical_page"] = json!(64);
        for (rank, registration) in dcp1.ranks.iter_mut().zip(&mut dcp1.registrations) {
            rank.dcp_rank = 0;
            rank.replica_rank = rank.tp_rank;
            registration.dcp_rank = rank.dcp_rank;
            registration.replica_rank = rank.replica_rank;
        }
        dcp1.qualified_rank_set_digest = cache_auth::digest(&json!(dcp1.ranks.iter()
            .map(|r| [r.tp_rank, r.dcp_rank, r.replica_rank]).collect::<Vec<_>>())).unwrap();
        dcp1.native_role_registration_digest = cache_auth::digest(&json!({"ranks":dcp1.ranks,"registrations":dcp1.registrations})).unwrap();
        dcp1.kv_compat_digest = frontend_qualification::kv_digest(&dcp1.kv_compat_descriptor).unwrap();
        let dcp1_role = role_from_discovery(&serde_json::to_vec(&json!({"artifact":dcp1,"qualified":true})).unwrap(),
            frontend_qualification::load_secret(&secret).unwrap(), &d.model, &d.native_origin, PdRole::Decode).unwrap();
        let dcp1_pair = VerifiedPdPair::from_roles(load("prefill", PdRole::Prefill), dcp1_role).unwrap();
        let txn = "a".repeat(32);
        let root = "b".repeat(32);
        let mut transport = PdGenerationTransport {
            prefill_incarnation:p.native_incarnation.clone(), decode_incarnation:d.native_incarnation.clone(),
            prefill_native_origin:p.native_origin.clone(), decode_native_origin:d.native_origin.clone(),
            kv_compat_digest:p.kv_compat_digest.clone(), qualified_rank_set_digest:p.qualified_rank_set_digest.clone(),
            transport_epoch:String::new(), session_epoch:String::new(),
        };
        transport.transport_epoch = transport.epoch(Some(&txn)).unwrap();
        transport.session_epoch = transport.epoch(None).unwrap();
        for (own, peer) in [(p, d), (d, p),
            (dcp1_pair.prefill().artifact(), dcp1_pair.decode().artifact()),
            (dcp1_pair.decode().artifact(), dcp1_pair.prefill().artifact())] {
            let binding = GenerationBinding {
                control:ControlRequest {schema_version:1,transaction_id:txn.clone(),request_root:root.clone(),
                    expected_native_incarnation:own.native_incarnation.clone(),
                    expected_qualified_rank_set_digest:own.qualified_rank_set_digest.clone(),role:own.role,
                    rooms:RoomRange::new(37, 2).unwrap().bindings(&root).unwrap()},
                transport:transport.clone(), peer_artifact:peer.clone(),
            };
            binding.validate_ordinary(&root, 2, &txn, &own.native_incarnation, &own.native_origin, &own.qualified_rank_set_digest).unwrap();
            let mut wrong = binding.clone();
            wrong.transport.qualified_rank_set_digest = d.qualified_rank_set_digest.clone();
            assert!(wrong.validate_ordinary(&root, 2, &txn, &own.native_incarnation, &own.native_origin, &own.qualified_rank_set_digest).is_err());
            let mut wrong_peer = binding.clone();
            wrong_peer.peer_artifact.native_incarnation = "0".repeat(32);
            assert!(wrong_peer.validate_ordinary(&root, 2, &txn, &own.native_incarnation, &own.native_origin, &own.qualified_rank_set_digest).is_err());
        }
        let mut wrong_bootstrap = p.clone();
        wrong_bootstrap.prefill_bootstrap_port = Some(0);
        assert!(wrong_bootstrap.validate().is_err());
        let mut old_snapshot = serde_json::to_value(p).unwrap();
        old_snapshot["registrations"][0]["transport"] = json!({"role":"prefill"});
        assert!(serde_json::from_value::<PdRoleArtifact>(old_snapshot).is_err());
        let mut wrong_birth = p.clone();
        wrong_birth.registrations[0].process_start_ticks = 0;
        assert!(wrong_birth.validate().is_err());
        let mut duplicate_birth = p.clone();
        duplicate_birth.registrations[1].boot_id = duplicate_birth.registrations[0].boot_id.clone();
        duplicate_birth.registrations[1].pid = duplicate_birth.registrations[0].pid;
        duplicate_birth.registrations[1].process_start_ticks = duplicate_birth.registrations[0].process_start_ticks;
        assert!(duplicate_birth.validate().is_err());
        let mut wrong_digest = p.clone();
        wrong_digest.native_role_registration_digest = "0".repeat(64);
        assert!(wrong_digest.validate().is_err());
        let mut wrong_bytes = d.clone();
        wrong_bytes.kv_compat_descriptor["physical_page"] = json!(128);
        wrong_bytes.kv_compat_digest = frontend_qualification::kv_digest(&wrong_bytes.kv_compat_descriptor).unwrap();
        assert!(pair_artifacts(p, &wrong_bytes).is_err());
        let mut obsolete_map = serde_json::to_value(d).unwrap();
        obsolete_map["peer_rank_map"] = json!([]);
        assert!(serde_json::from_value::<PdRoleArtifact>(obsolete_map).is_err());
        for value in [Value::Null, json!(true), json!(0), json!(3)] {
            let mut wrong_ownership = p.clone();
            wrong_ownership.kv_compat_descriptor["attn_tp_size"] = value;
            wrong_ownership.kv_compat_digest = frontend_qualification::kv_digest(&wrong_ownership.kv_compat_descriptor).unwrap();
            assert!(wrong_ownership.validate().is_err());
        }
    }

    #[test]
    fn pd_role_native_codec_full_artifact_digest_and_required_nulls() {
        let receipt = native_fixture();
        receipt.artifact.validate().unwrap();
        receipt.peer_artifact.validate().unwrap();
        assert_eq!(
            receipt.artifact.artifact_digest().unwrap(),
            receipt.artifact_digest
        );
        assert_eq!(
            receipt.peer_artifact.artifact_digest().unwrap(),
            receipt.peer_artifact_digest
        );
        pair_artifacts(&receipt.artifact, &receipt.peer_artifact).unwrap();
        let mut wrong_birth = receipt.artifact.clone();
        wrong_birth.registrations[0].process_start_ticks += 1;
        assert!(wrong_birth.validate().is_err());
        let mut missing = receipt.artifact.clone();
        missing.registrations.pop();
        assert!(missing.validate().is_err());
        let mut duplicate = receipt.artifact.clone();
        duplicate.ranks[1] = duplicate.ranks[0].clone();
        assert!(duplicate.validate().is_err());
        let mut missing_null = serde_json::to_value(&receipt.peer_artifact).unwrap();
        missing_null
            .as_object_mut()
            .unwrap()
            .remove("prefill_bootstrap_bind_host");
        assert!(serde_json::from_value::<PdRoleArtifact>(missing_null).is_err());
        let mut self_approved = serde_json::to_value(&receipt.artifact).unwrap();
        self_approved["qualified"] = json!(true);
        assert!(serde_json::from_value::<PdRoleArtifact>(self_approved).is_err());
        let mut wrong_peer = receipt.peer_artifact.clone();
        wrong_peer.kv_compat_digest = "0".repeat(64);
        assert!(pair_artifacts(&receipt.artifact, &wrong_peer).is_err());
    }

    fn write(path: &Path, bytes: &[u8], mode: u32) {
        // The CPU fixture owns its temporary directory. Replace its readonly
        // receipt inode rather than weakening the production loader's policy.
        if path.exists() {
            fs::remove_file(path).unwrap();
        }
        fs::write(path, bytes).unwrap();
        fs::set_permissions(path, fs::Permissions::from_mode(mode)).unwrap();
    }
    fn fixture_roles(directory: &Path) -> (VerifiedPdRole, VerifiedPdRole) {
        let native = native_fixture();
        let secret = directory.join("secret");
        write(&secret, &[13; 32], 0o400);
        let load = |artifact: PdRoleArtifact| {
            role_from_discovery(
                &serde_json::to_vec(&json!({"artifact":artifact,"qualified":true})).unwrap(),
                frontend_qualification::load_secret(&secret).unwrap(),
                &artifact.model,
                &artifact.native_origin,
                artifact.role,
            )
            .unwrap()
        };
        (load(native.artifact), load(native.peer_artifact))
    }

    #[tokio::test]
    async fn pd_dispatcher_composes_same_boot_distinct_approved_runtime_profiles_without_io() {
        use crate::native_cleanup::NativeCleanupRegistry;
        use crate::native_cleanup_cohort::{
            CleanupCohortArtifact, CleanupCohortParticipant, install_cleanup_cohort,
        };
        use crate::native_pd::dispatch::PdDispatcher;
        use crate::native_request_lifecycle::QualifiedRequestLifecycle;
        use std::sync::Arc;

        // Explicit CPU actual-registration fixture; no native I/O.
        let directory = tempfile::tempdir().unwrap();
        let (prefill, decode) = fixture_roles(directory.path());
        let pair = Arc::new(VerifiedPdPair::from_roles(prefill, decode).unwrap());
        let ordinary_runtime =
            cache_auth::digest(&json!({"layout":"ordinary CPU fixture"})).unwrap();
        assert_ne!(
            ordinary_runtime,
            pair.prefill().artifact().runtime_profile_digest
        );
        let registry = NativeCleanupRegistry::new();
        let artifact = |role: &PdRoleArtifact| {
            let participants = vec![CleanupCohortParticipant {
                participant_id: "cpu-source-bound-complete-cohort".into(),
                role: "cpu-declared-resource-owner".into(),
                process_birth: "cpu-approved-birth".into(),
                source_digest: role.source_digest.clone(),
                runtime_registration_digest: "e".repeat(64),
            }];
            CleanupCohortArtifact {
                schema_version: 1,
                native_origin: role.native_origin.clone(),
                native_incarnation: role.native_incarnation.clone(),
                model: role.model.clone(),
                source_digest: role.source_digest.clone(),
                runtime_profile_digest: ordinary_runtime.clone(),
                cleanup_cohort_digest: cache_auth::digest(&json!([participants[0].participant_id]))
                    .unwrap(),
                participants,
            }
        };
        let lifecycle = |a: CleanupCohortArtifact, key: [u8; 32]| {
            let installed = install_cleanup_cohort(a.clone()).unwrap();
            Arc::new(
                QualifiedRequestLifecycle::new(installed, key, &a.native_origin, registry.clone())
                    .unwrap(),
            )
        };
        let p = artifact(pair.prefill().artifact());
        let d = artifact(pair.decode().artifact());
        let p_lifecycle = lifecycle(p.clone(), [13; 32]);
        let d_lifecycle = lifecycle(d.clone(), [13; 32]);
        assert!(PdDispatcher::new(pair.clone(), p_lifecycle.clone(), d_lifecycle.clone()).is_ok());
        for field in [
            "model",
            "native_origin",
            "native_incarnation",
            "source_digest",
        ] {
            let mut wrong = p.clone();
            match field {
                "model" => wrong.model.push_str("-different"),
                "native_origin" => wrong.native_origin = "http://127.0.0.1:45551".into(),
                "native_incarnation" => wrong.native_incarnation = "1".repeat(32),
                "source_digest" => wrong.source_digest = "1".repeat(64),
                _ => unreachable!(),
            }
            assert!(
                PdDispatcher::new(
                    pair.clone(),
                    lifecycle(wrong, [13; 32]),
                    d_lifecycle.clone()
                )
                .is_err(),
                "{field}"
            );
        }
        assert!(PdDispatcher::new(pair, lifecycle(p, [14; 32]), d_lifecycle).is_err());
        // The constructor has no Count, clock sampling, role-control or Gen POST.
    }

    #[test]
    fn pd_role_discovery_rejects_incomplete_or_wrong_role() {
        let directory = tempfile::tempdir().unwrap();
        let (prefill, decode) = fixture_roles(directory.path());
        let secret = directory.path().join("secret");
        for (qualified, role) in [(false, PdRole::Prefill), (true, PdRole::Decode)] {
            let bytes =
                serde_json::to_vec(&json!({"artifact":prefill.artifact(),"qualified":qualified}))
                    .unwrap();
            assert!(
                role_from_discovery(
                    &bytes,
                    frontend_qualification::load_secret(&secret).unwrap(),
                    &prefill.artifact().model,
                    &prefill.artifact().native_origin,
                    role
                )
                .is_err()
            );
        }
        VerifiedPdPair::from_roles(prefill, decode).unwrap();
    }
}
