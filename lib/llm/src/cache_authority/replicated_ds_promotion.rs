//! Actual runtime source/destination byte mapping; never supplied by a request.
use super::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ReplicatedDsPromotion {
    pub schema: String,
    pub mapping: String,
    pub source_tp: u32,
    pub destination_tp: u32,
    pub payload_compat_digest: String,
    pub source_execution_digest: String,
    pub destination_execution_digest: String,
    pub backend_namespace_digest: String,
    pub source_native_origin_uuid: String,
    pub destination_native_origin_uuid: String,
    pub source_native_incarnation: String,
    pub destination_native_incarnation: String,
    pub native_components: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeMapping {
    mapping: ReplicatedDsPromotion,
    source_rank_descriptors: Vec<serde_json::Value>,
    destination_rank_descriptors: Vec<serde_json::Value>,
    source_execution_topologies: Vec<serde_json::Value>,
    destination_execution_topologies: Vec<serde_json::Value>,
}

#[derive(Clone, Debug)]
pub(crate) struct QualifiedPromotion {
    pub mapping: ReplicatedDsPromotion,
    pub digest: String,
    pub source: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PromotionSourceAudit {
    pub mapping_digest: String,
    pub source_execution_digest: String,
    pub source_runtime_profile_digest: String,
    pub source_native_origin_uuid: String,
    pub source_native_incarnation: String,
}

fn digest(value: &str) -> Result<()> {
    if value.len() != 64 || value.bytes().all(|c| c == b'0')
        || !value.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)) {
        return Err(Error::Evidence("promotion requires qualified SHA256"));
    }
    Ok(())
}

impl QualifiedPromotion {
    pub(crate) fn from_runtime(value: &serde_json::Value, runtime: &str,
        compatibility: &str, backend: &str, ranks: &BTreeSet<u32>,
        actual_execution: &str, actual_native_identity: &str,
        actual_components: &BTreeSet<String>, actual_descriptors: &[serde_json::Value],
        actual_topologies: &[serde_json::Value]) -> Result<Self> {
        let outer: RuntimeMapping = serde_json::from_value(value.clone())
            .map_err(|_| Error::Evidence("promotion runtime mapping shape mismatch"))?;
        if outer.source_rank_descriptors.len()!=4 || outer.source_execution_topologies.len()!=4
            || outer.destination_rank_descriptors.len()!=8 || outer.destination_execution_topologies.len()!=8 {
            return Err(Error::Evidence("promotion runtime mapping lacks complete cohorts"));
        }
        let q=outer.mapping;
        for d in [&q.payload_compat_digest,&q.source_execution_digest,&q.destination_execution_digest,
            &q.backend_namespace_digest] {digest(d)?;}
        let components: BTreeSet<_> = q.native_components.iter().cloned().collect();
        if q.schema != "sglang.ds.replicated-promotion.v1"
            || q.mapping != "source4-primary-to-destination8-replicated"
            || q.source_tp != 4 || q.destination_tp != 8
            || q.payload_compat_digest != compatibility || q.backend_namespace_digest != backend
            || runtime.is_empty() || actual_native_identity.is_empty() || actual_execution.is_empty()
            || components.is_empty() || components.len() != q.native_components.len()
            || components.iter().any(|s| s.is_empty()) || components != *actual_components {
            return Err(Error::Evidence("promotion differs from actual qualified payload/backend/components"));
        }
        let source = ranks == &(0..4).collect::<BTreeSet<_>>()
            && actual_execution == q.source_execution_digest
            && actual_native_identity == q.source_native_origin_uuid
            && actual_native_identity == q.source_native_incarnation;
        let destination = ranks == &(0..8).collect::<BTreeSet<_>>()
            && actual_execution == q.destination_execution_digest
            && actual_native_identity == q.destination_native_origin_uuid
            && actual_native_identity == q.destination_native_incarnation;
        if source == destination { return Err(Error::Evidence("promotion does not bind this actual native cohort")); }
        let (resident, execution) = if source { (&outer.source_rank_descriptors, &outer.source_execution_topologies) }
            else { (&outer.destination_rank_descriptors, &outer.destination_execution_topologies) };
        if actual_descriptors != resident.as_slice() || actual_topologies != execution.as_slice() {
            return Err(Error::Evidence("promotion mapping differs from actual loaded execution audit"));
        }
        let mapping_digest = super::frontend_qualification::kv_digest(&serde_json::to_value(&q)
            .map_err(|_| Error::Evidence("promotion canonical digest unavailable"))?)
            .map_err(|_| Error::Evidence("promotion canonical digest unavailable"))?;
        Ok(Self { mapping: q, digest: mapping_digest, source })
    }

    pub(crate) fn source_audit(&self, runtime:&str) -> PromotionSourceAudit {
        PromotionSourceAudit { mapping_digest: self.digest.clone(),
            source_execution_digest: self.mapping.source_execution_digest.clone(),
            source_runtime_profile_digest: runtime.into(),
            source_native_origin_uuid: self.mapping.source_native_origin_uuid.clone(),
            source_native_incarnation: self.mapping.source_native_incarnation.clone() }
    }

    pub(crate) fn verify_source(&self, audit: &PromotionSourceAudit, runtime: &str,
        identity: &str, objects: &BTreeMap<u32,String>) -> Result<()> {
        let expected = self.source_audit(runtime);
        if audit.mapping_digest != expected.mapping_digest
            || audit.source_execution_digest != expected.source_execution_digest
            || audit.source_runtime_profile_digest != expected.source_runtime_profile_digest
            || audit.source_native_origin_uuid != expected.source_native_origin_uuid
            || audit.source_native_incarnation != expected.source_native_incarnation
            || identity != expected.source_native_incarnation
            || objects.keys().copied().collect::<BTreeSet<_>>() != (0..4).collect()
            || objects.values().collect::<BTreeSet<_>>().len() != 1 {
            return Err(Error::Evidence("promotion source lacks original qualified four-rank publication"));
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::{json,Value};
    pub(crate) fn cpu_outer()->Value {
        let h=|c:&str|c.repeat(64);
        json!({"mapping":{"schema":"sglang.ds.replicated-promotion.v1",
            "mapping":"source4-primary-to-destination8-replicated","source_tp":4,"destination_tp":8,
            "payload_compat_digest":h("6"),"source_execution_digest":h("7"),"destination_execution_digest":h("8"),
            "backend_namespace_digest":h("9"),
            "source_native_origin_uuid":"CPU_NOTQUALIFIED_source","source_native_incarnation":"CPU_NOTQUALIFIED_source",
            "destination_native_origin_uuid":"CPU_NOTQUALIFIED_destination","destination_native_incarnation":"CPU_NOTQUALIFIED_destination",
            "native_components":["target","draft"]},
            "source_rank_descriptors":(0..4).map(|r|json!({"CPU_NOTQUALIFIED_rank":r})).collect::<Vec<_>>(),
            "destination_rank_descriptors":(0..8).map(|r|json!({"CPU_NOTQUALIFIED_rank":r})).collect::<Vec<_>>(),
            "source_execution_topologies":(0..4).map(|r|json!({"CPU_NOTQUALIFIED_execution":r})).collect::<Vec<_>>(),
            "destination_execution_topologies":(0..8).map(|r|json!({"CPU_NOTQUALIFIED_execution":r})).collect::<Vec<_>>()})
    }
    pub(crate) fn cpu_parse(v:&Value,source:bool)->Result<QualifiedPromotion> {
        let(n,rt,exec,id,descriptors,topologies)=if source {
            (4,"a".repeat(64),"7".repeat(64),"CPU_NOTQUALIFIED_source",v["source_rank_descriptors"].as_array().unwrap(),v["source_execution_topologies"].as_array().unwrap())
        }else{(8,"b".repeat(64),"8".repeat(64),"CPU_NOTQUALIFIED_destination",v["destination_rank_descriptors"].as_array().unwrap(),v["destination_execution_topologies"].as_array().unwrap())};
        QualifiedPromotion::from_runtime(v,&rt,&"6".repeat(64),&"9".repeat(64),&(0..n).collect(),&exec,id,&BTreeSet::from(["target".into(),"draft".into()]),descriptors,topologies)
    }
    #[test]
    fn native_ds_promotion_runtime_mapping_preserves_exact_cohort() {
        let v=cpu_outer();let a=cpu_parse(&v,true).unwrap();let b=cpu_parse(&v,false).unwrap();
        assert!(a.source);assert!(!b.source);assert_eq!(a.digest,b.digest);
        let mut invalid=v.clone();invalid["mapping"]["source_native_incarnation"]=json!("wrong-boot");assert!(cpu_parse(&invalid,true).is_err());
        let mut incomplete=v;incomplete["source_rank_descriptors"].as_array_mut().unwrap().pop();assert!(cpu_parse(&incomplete,true).is_err());
    }
}
