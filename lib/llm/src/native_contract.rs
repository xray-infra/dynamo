//! Public four-model Chat contracts for the direct native HTTP transport.
//!
//! The returned JSON is the authoritative count/generation body. Routing views
//! must not reconstruct it from a typed DTO and lose native fields.
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::HashSet;

pub const CONTEXT_LIMIT: u64 = 1_048_576;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NativeTier {
    Short,
    Long,
    Shared,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct NativeContractError(pub String);
fn invalid(message: &str) -> NativeContractError {
    NativeContractError(message.into())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Family {
    Glm,
    GlmFlash,
    DeepSeek,
    Qwen,
}

/// Identity obtained from the pinned native instance by deployment/readiness
/// qualification. The private configuration is trusted; a digest is a binding
/// to recorded evidence, not a substitute for a native budget test.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeWorkerIdentity {
    pub model: String,
    pub model_identity: String,
    pub source_sha256: String,
    pub template_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingBudgetField {
    ThinkingTokenBudget,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThinkingBudgetQualification {
    pub identity: NativeWorkerIdentity,
    pub evidence_sha256: String,
    pub native_field: ThinkingBudgetField,
}

#[derive(Debug, Clone)]
pub struct VerifiedThinkingBudgetQualification {
    identity: NativeWorkerIdentity,
    native_field: ThinkingBudgetField,
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

impl NativeWorkerIdentity {
    pub fn validate(&self) -> Result<(), NativeContractError> {
        ModelContract::for_model(&self.model)?;
        if self.model_identity.trim().is_empty()
            || self.model_identity.len() > 4096
            || !valid_digest(&self.source_sha256)
            || !valid_digest(&self.template_sha256)
        {
            return Err(invalid("native worker qualification identity is invalid"));
        }
        Ok(())
    }
}

impl ThinkingBudgetQualification {
    pub fn verify(
        &self,
        current: &NativeWorkerIdentity,
    ) -> Result<VerifiedThinkingBudgetQualification, NativeContractError> {
        self.identity.validate()?;
        current.validate()?;
        if !valid_digest(&self.evidence_sha256) {
            return Err(invalid(
                "thinking budget qualification evidence digest is invalid",
            ));
        }
        if &self.identity != current {
            return Err(invalid(
                "thinking budget qualification does not match the native worker identity",
            ));
        }
        Ok(VerifiedThinkingBudgetQualification {
            identity: current.clone(),
            native_field: self.native_field,
        })
    }
}

impl VerifiedThinkingBudgetQualification {
    pub fn matches_current(&self, current: &NativeWorkerIdentity) -> bool {
        &self.identity == current
    }
}

/// Enables the public capability only after the transport has proved coverage
/// of every model tier. The transport must still filter each dispatch by that
/// worker's verified receipt, including connection-failure candidates.
#[derive(Debug, Clone)]
pub struct VerifiedModelThinkingBudget {
    model: String,
    native_field: ThinkingBudgetField,
}

impl VerifiedModelThinkingBudget {
    /// Private loopback canary only. The caller must enforce the same tier on
    /// all configured count/generation workers and suppress full-model cards.
    pub fn verify_single_tier(
        model: &str,
        tier: NativeTier,
        workers: &[(NativeTier, VerifiedThinkingBudgetQualification)],
    ) -> Result<Self, NativeContractError> {
        validate_qualification_scope(model, tier)?;
        let native_field = workers
            .first()
            .ok_or_else(|| invalid("scoped thinking budget qualification is empty"))?
            .1
            .native_field;
        if workers.iter().any(|(actual_tier, receipt)| {
            *actual_tier != tier
                || receipt.identity.model != model
                || receipt.native_field != native_field
        }) {
            return Err(invalid(
                "scoped thinking budget receipt model, tier or native field does not match",
            ));
        }
        Ok(Self {
            model: model.into(),
            native_field,
        })
    }
    pub fn verify(
        model: &str,
        workers: &[(NativeTier, VerifiedThinkingBudgetQualification)],
    ) -> Result<Self, NativeContractError> {
        let contract = ModelContract::for_model(model)?;
        let tiers: &[NativeTier] = if contract.short_limit.is_some() {
            &[NativeTier::Short, NativeTier::Long]
        } else {
            &[NativeTier::Shared]
        };
        if workers
            .iter()
            .any(|(tier, receipt)| receipt.identity.model != model || !tiers.contains(tier))
        {
            return Err(invalid(
                "thinking budget receipt model or tier does not match the public contract",
            ));
        }
        if tiers
            .iter()
            .any(|tier| !workers.iter().any(|(worker_tier, _)| worker_tier == tier))
        {
            return Err(invalid(
                "thinking budget qualification does not cover every public model tier",
            ));
        }
        let native_field = workers
            .first()
            .ok_or_else(|| invalid("thinking budget qualification is empty"))?
            .1
            .native_field;
        if workers
            .iter()
            .any(|(_, receipt)| receipt.native_field != native_field)
        {
            return Err(invalid("thinking budget native fields disagree"));
        }
        Ok(Self {
            model: model.into(),
            native_field,
        })
    }
}

/// Native forced-final-phase mode, accepted only through private qualification.
/// Public callers continue to use standard none / disabled protocol fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForcedZeroThinkingField {
    ForcedZeroThinking,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForcedZeroThinkingQualification {
    pub identity: NativeWorkerIdentity,
    pub evidence_sha256: String,
    pub native_field: ForcedZeroThinkingField,
}

#[derive(Debug, Clone)]
pub struct VerifiedForcedZeroThinkingQualification {
    identity: NativeWorkerIdentity,
    native_field: ForcedZeroThinkingField,
}

impl ForcedZeroThinkingQualification {
    pub fn verify(
        &self,
        current: &NativeWorkerIdentity,
    ) -> Result<VerifiedForcedZeroThinkingQualification, NativeContractError> {
        self.identity.validate()?;
        current.validate()?;
        if !valid_digest(&self.evidence_sha256) {
            return Err(invalid(
                "forced zero thinking qualification evidence digest is invalid",
            ));
        }
        if &self.identity != current {
            return Err(invalid(
                "forced zero thinking qualification does not match the native worker identity",
            ));
        }
        Ok(VerifiedForcedZeroThinkingQualification {
            identity: current.clone(),
            native_field: self.native_field,
        })
    }
}

impl VerifiedForcedZeroThinkingQualification {
    pub fn matches_current(&self, current: &NativeWorkerIdentity) -> bool {
        &self.identity == current
    }
}

#[derive(Debug, Clone)]
pub struct VerifiedModelForcedZeroThinking {
    model: String,
    native_field: ForcedZeroThinkingField,
}

impl VerifiedModelForcedZeroThinking {
    /// Private loopback canary only; never full-model capability evidence.
    pub fn verify_single_tier(
        model: &str,
        tier: NativeTier,
        workers: &[(NativeTier, VerifiedForcedZeroThinkingQualification)],
    ) -> Result<Self, NativeContractError> {
        validate_qualification_scope(model, tier)?;
        let native_field = workers
            .first()
            .ok_or_else(|| invalid("scoped forced zero qualification is empty"))?
            .1
            .native_field;
        if workers.iter().any(|(actual_tier, receipt)| {
            *actual_tier != tier
                || receipt.identity.model != model
                || receipt.native_field != native_field
        }) {
            return Err(invalid(
                "scoped forced zero receipt model, tier or native field does not match",
            ));
        }
        Ok(Self {
            model: model.into(),
            native_field,
        })
    }
    pub fn verify(
        model: &str,
        workers: &[(NativeTier, VerifiedForcedZeroThinkingQualification)],
    ) -> Result<Self, NativeContractError> {
        let contract = ModelContract::for_model(model)?;
        let tiers: &[NativeTier] = if contract.short_limit.is_some() {
            &[NativeTier::Short, NativeTier::Long]
        } else {
            &[NativeTier::Shared]
        };
        if workers
            .iter()
            .any(|(tier, receipt)| receipt.identity.model != model || !tiers.contains(tier))
        {
            return Err(invalid(
                "forced zero thinking receipt model or tier does not match the public contract",
            ));
        }
        if tiers
            .iter()
            .any(|tier| !workers.iter().any(|(worker_tier, _)| worker_tier == tier))
        {
            return Err(invalid(
                "forced zero thinking qualification does not cover every public model tier",
            ));
        }
        let native_field = workers
            .first()
            .ok_or_else(|| invalid("forced zero thinking qualification is empty"))?
            .1
            .native_field;
        if workers
            .iter()
            .any(|(_, receipt)| receipt.native_field != native_field)
        {
            return Err(invalid("forced zero thinking native fields disagree"));
        }
        Ok(Self {
            model: model.into(),
            native_field,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ModelContract {
    pub model: &'static str,
    pub owner: &'static str,
    pub context_limit: u64,
    pub output_limit: u64,
    pub short_limit: Option<u64>,
    pub short_reserve: u64,
    family: Family,
    thinking_budget_field: Option<ThinkingBudgetField>,
    forced_zero_thinking_field: Option<ForcedZeroThinkingField>,
}

fn validate_qualification_scope(model: &str, tier: NativeTier) -> Result<(), NativeContractError> {
    let contract = ModelContract::for_model(model)?;
    let valid = if contract.short_limit.is_some() {
        matches!(tier, NativeTier::Short | NativeTier::Long)
    } else {
        tier == NativeTier::Shared
    };
    if !valid {
        return Err(invalid(
            "qualification tier is not a native tier for the public model",
        ));
    }
    Ok(())
}

impl ModelContract {
    pub fn for_model(model: &str) -> Result<Self, NativeContractError> {
        let (model, owner, output_limit, short_limit, short_reserve, family) = match model {
            "zai-org/GLM-5.3" => (
                "zai-org/GLM-5.3",
                "zai-org",
                131_072,
                Some(65_536),
                0,
                Family::Glm,
            ),
            "glm-5.3-flash" => (
                "glm-5.3-flash",
                "zai-org",
                131_072,
                None,
                0,
                Family::GlmFlash,
            ),
            "deepseek-ai/DeepSeek-V4.1-Flash" => (
                "deepseek-ai/DeepSeek-V4.1-Flash",
                "deepseek-ai",
                393_216,
                Some(65_536),
                4_096,
                Family::DeepSeek,
            ),
            "qwen3.8-flash-next" => (
                "qwen3.8-flash-next",
                "qwen",
                262_144,
                Some(16_384),
                0,
                Family::Qwen,
            ),
            _ => return Err(invalid("unknown configured public model")),
        };
        Ok(Self {
            model,
            owner,
            context_limit: CONTEXT_LIMIT,
            output_limit,
            short_limit,
            short_reserve,
            family,
            thinking_budget_field: (family == Family::Qwen)
                .then_some(ThinkingBudgetField::ThinkingTokenBudget),
            // Both current GLM renderers implement `none` by entering the
            // native final phase, with the template unchanged.
            forced_zero_thinking_field: matches!(family, Family::Glm | Family::GlmFlash)
                .then_some(ForcedZeroThinkingField::ForcedZeroThinking),
        })
    }

    pub fn count_path(&self) -> &'static str {
        if self.family == Family::GlmFlash {
            "/tokenize"
        } else {
            "/v1/tokenize"
        }
    }

    pub fn supports_disabled_thinking(&self) -> bool {
        matches!(self.family, Family::DeepSeek | Family::Qwen)
            || self.forced_zero_thinking_field.is_some()
    }

    pub fn supports_forced_zero_thinking(&self) -> bool {
        self.forced_zero_thinking_field.is_some()
    }

    pub fn with_forced_zero_thinking_qualification(
        mut self,
        qualification: &VerifiedModelForcedZeroThinking,
    ) -> Result<Self, NativeContractError> {
        if qualification.model != self.model {
            return Err(invalid(
                "forced zero thinking qualification model does not match contract",
            ));
        }
        self.forced_zero_thinking_field = Some(qualification.native_field);
        Ok(self)
    }

    pub fn requires_forced_zero_thinking(body: &Value) -> bool {
        body.get("forced_zero_thinking").and_then(Value::as_bool) == Some(true)
    }

    pub fn supports_fixed_thinking_budget(&self) -> bool {
        self.thinking_budget_field.is_some()
    }

    pub fn with_thinking_budget_qualification(
        mut self,
        qualification: &VerifiedModelThinkingBudget,
    ) -> Result<Self, NativeContractError> {
        if qualification.model != self.model {
            return Err(invalid(
                "thinking budget qualification model does not match contract",
            ));
        }
        self.thinking_budget_field = Some(qualification.native_field);
        Ok(self)
    }

    pub fn requires_thinking_budget(body: &Value) -> bool {
        body.get("thinking_token_budget").is_some()
    }

    pub fn model_card(&self) -> Value {
        let mut efforts = vec!["minimal", "low", "medium", "high", "xhigh", "max"];
        if self.supports_disabled_thinking() {
            efforts.insert(0, "none");
        }
        json!({"id": self.model, "object": "model", "created": 0, "owned_by": self.owner,
            "context_window": self.context_limit, "max_input_tokens": self.context_limit,
            "max_output_tokens": self.output_limit,
            "gateway_capabilities": {"reasoning": {"efforts": efforts,
                "disabled_supported": self.supports_disabled_thinking(),
                "fixed_budget_supported": self.supports_fixed_thinking_budget()},
                "prompt_cache": {"mode": "automatic_prefix", "cache_key_semantics": "namespace",
                    "cache_read_tokens_source": "native_usage", "cache_write_billing": false,
                    "block_breakpoints_supported": false, "ttl_guaranteed": false,
                    "worker_affinity_guaranteed": false}}})
    }

    /// Read aliases without modifying the caller's body. Reasoning aliases are
    /// parsed first so DS default output is based on canonical none/max.
    pub fn output_budget(&self, body: &Value) -> Result<u64, NativeContractError> {
        let object = body
            .as_object()
            .ok_or_else(|| invalid("expected a request object"))?;
        let state = self.reasoning(object)?;
        self.output_with_state(object, &state)
    }
    pub fn requested_output(&self, body: &Value) -> Result<u64, NativeContractError> {
        self.output_budget(body)
    }

    fn output_with_state(
        &self,
        body: &Map<String, Value>,
        state: &Reasoning,
    ) -> Result<u64, NativeContractError> {
        let mut supplied = None;
        for key in ["max_tokens", "max_completion_tokens", "max_output_tokens"] {
            if let Some(value) = body.get(key) {
                let value = value.as_u64().filter(|v| (1..=self.output_limit).contains(v))
                    .ok_or_else(|| invalid("output budget must be a positive integer within the public output limit"))?;
                if supplied.is_some_and(|old| old != value) {
                    return Err(invalid("output budget aliases disagree"));
                }
                supplied = Some(value);
            }
        }
        Ok(supplied.unwrap_or_else(|| {
            if self.family != Family::DeepSeek || state.enabled == Some(false) {
                8_192
            } else if state.effort == Some(NativeEffort::Percent(100)) {
                131_072
            } else {
                65_536
            }
        }))
    }

    pub fn select_tier(&self, prompt: u64, output: u64) -> Result<NativeTier, NativeContractError> {
        if output == 0 || output > self.output_limit {
            return Err(invalid("output exceeds public output budget"));
        }
        let total = prompt
            .checked_add(output)
            .filter(|v| *v <= self.context_limit)
            .ok_or_else(|| invalid("prompt and requested output exceed public context"))?;
        Ok(match self.short_limit {
            None => NativeTier::Shared,
            Some(limit)
                if total
                    .checked_add(self.short_reserve)
                    .is_some_and(|v| v <= limit) =>
            {
                NativeTier::Short
            }
            Some(_) => NativeTier::Long,
        })
    }
    pub fn admit(&self, prompt: u64, output: u64) -> Result<NativeTier, NativeContractError> {
        self.select_tier(prompt, output)
    }

    pub fn normalize(&self, body: Value) -> Result<Value, NativeContractError> {
        let mut body = body
            .as_object()
            .cloned()
            .ok_or_else(|| invalid("expected a request object"))?;
        if body.get("model").and_then(Value::as_str) != Some(self.model) {
            return Err(invalid("expected the configured public model"));
        }
        validate_chat(&body)?;
        normalize_native_instruction_roles(&mut body, self.family)?;
        let state = self.reasoning(&body)?;
        let output = self.output_with_state(&body, &state)?;
        for key in ["max_tokens", "max_completion_tokens", "max_output_tokens"] {
            body.remove(key);
        }
        body.insert("max_tokens".into(), output.into());
        self.apply_reasoning(&mut body, state, output)?;
        normalize_cache(&mut body)?;
        Ok(Value::Object(body))
    }

    fn map_effort(
        &self,
        value: &Value,
        template: bool,
    ) -> Result<Option<NativeEffort>, NativeContractError> {
        if value.is_null() {
            return Ok(None);
        }
        if let Some(value) = value.as_str() {
            let mapped = match (self.family, value) {
                (_, "minimal" | "low") => "low",
                (Family::Qwen, "medium") => "medium",
                (Family::Qwen, "high" | "xhigh" | "max") => "xhigh",
                (Family::DeepSeek, "medium" | "high") => "high",
                (Family::DeepSeek, "xhigh") => "xhigh",
                (Family::DeepSeek, "max") => "max",
                (Family::Glm | Family::GlmFlash, "medium" | "high") => "high",
                (Family::Glm | Family::GlmFlash, "xhigh" | "max") => "max",
                _ => return Err(invalid("unsupported reasoning effort")),
            };
            return Ok(Some(if self.family == Family::DeepSeek {
                NativeEffort::Percent(match mapped {
                    "low" => 25,
                    "high" => 50,
                    "xhigh" => 75,
                    _ => 100,
                })
            } else {
                NativeEffort::Name(mapped)
            }));
        }
        if self.family == Family::DeepSeek {
            if template {
                if let Some(value) = value.as_u64().filter(|v| (1..=100).contains(v)) {
                    return Ok(Some(NativeEffort::Percent(value)));
                }
            }
            // Public legacy floating effort is a fraction. Integer JSON is not
            // a fraction, and bool is never a numeric effort.
            if value.as_number().is_some_and(|n| n.is_f64()) {
                if let Some(value) = value
                    .as_f64()
                    .filter(|v| v.is_finite() && (0.0..=0.99).contains(v))
                {
                    // Python round uses ties-to-even; preserve the frozen mapping.
                    return Ok(Some(NativeEffort::Percent(
                        (value * 100.0).round_ties_even().clamp(1.0, 100.0) as u64,
                    )));
                }
            }
        }
        Err(invalid("unsupported reasoning effort type or range"))
    }

    fn reasoning(&self, body: &Map<String, Value>) -> Result<Reasoning, NativeContractError> {
        let empty = Map::new();
        let kwargs = match body.get("chat_template_kwargs") {
            None => &empty,
            Some(value) => value
                .as_object()
                .ok_or_else(|| invalid("chat_template_kwargs must be an object"))?,
        };
        let mut result = Reasoning::default();
        if body.contains_key("_forced_zero_thinking_closed")
            || kwargs.contains_key("_forced_zero_thinking_closed")
        {
            return Err(invalid(
                "native forced zero thinking closed state cannot be supplied by a request",
            ));
        }
        if kwargs.contains_key("forced_zero_thinking") {
            return Err(invalid(
                "forced zero thinking mode is a top-level internal native field",
            ));
        }
        // Internal canonical bodies are normalized again by the transport.
        // Public HTTP guards must reject this private knob before normalization.
        if let Some(value) = body.get("forced_zero_thinking") {
            if !self.supports_forced_zero_thinking() || value.as_bool() != Some(true) {
                return Err(invalid(
                    "forced_zero_thinking requires a qualified native mode and boolean true",
                ));
            }
            result.set_enabled(false)?;
        }
        for container in [body, kwargs] {
            if ["max_thinking_tokens", "thinking_budget", "budget_tokens"]
                .iter()
                .any(|k| container.contains_key(*k))
            {
                return Err(invalid("unsupported fixed thinking budget field"));
            }
        }
        if kwargs.contains_key("thinking_token_budget") {
            return Err(invalid("thinking_token_budget must be top level"));
        }
        if let Some(value) = body.get("thinking_token_budget") {
            self.add_budget(&mut result, value)?;
        }
        for (container, template) in [(body, false), (kwargs, true)] {
            if let Some(value) = container.get("reasoning_effort") {
                self.add_effort(&mut result, value, template)?;
            }
            for (key, inverted) in [("enable_thinking", false), ("no_think", true)] {
                if let Some(value) = container.get(key) {
                    let value = value
                        .as_bool()
                        .ok_or_else(|| invalid("thinking toggle must be boolean"))?;
                    result.set_enabled(if inverted { !value } else { value })?;
                }
            }
            if let Some(value) = container.get("thinking") {
                if !template && value.is_object() {
                    let value = value.as_object().unwrap();
                    if value
                        .keys()
                        .any(|k| !["type", "budget_tokens"].contains(&k.as_str()))
                    {
                        return Err(invalid("unsupported thinking options"));
                    }
                    let kind = value
                        .get("type")
                        .and_then(Value::as_str)
                        .ok_or_else(|| invalid("thinking.type is required"))?;
                    if !["enabled", "adaptive", "disabled"].contains(&kind) {
                        return Err(invalid("invalid thinking.type"));
                    }
                    if let Some(budget) = value.get("budget_tokens") {
                        if kind != "enabled" {
                            return Err(invalid("thinking budget requires enabled thinking"));
                        }
                        self.add_budget(&mut result, budget)?;
                    }
                    result.set_enabled(kind != "disabled")?;
                } else {
                    result.set_enabled(value.as_bool().ok_or_else(|| {
                        invalid("thinking must be a boolean or supported object")
                    })?)?;
                }
            }
        }
        if let Some(value) = body.get("reasoning") {
            let value = value
                .as_object()
                .ok_or_else(|| invalid("reasoning must be an object"))?;
            if value
                .keys()
                .any(|k| !["effort", "reasoning_effort", "enabled", "enable"].contains(&k.as_str()))
            {
                return Err(invalid("unsupported reasoning options"));
            }
            for key in ["effort", "reasoning_effort"] {
                if let Some(value) = value.get(key) {
                    self.add_effort(&mut result, value, false)?;
                }
            }
            for key in ["enabled", "enable"] {
                if let Some(value) = value.get(key) {
                    result.set_enabled(
                        value
                            .as_bool()
                            .ok_or_else(|| invalid("reasoning enable must be boolean"))?,
                    )?;
                }
            }
        }
        if result.enabled == Some(false) && !self.supports_disabled_thinking() {
            return Err(invalid("native model does not support disabling thinking"));
        }
        Ok(result)
    }

    fn add_effort(
        &self,
        state: &mut Reasoning,
        value: &Value,
        template: bool,
    ) -> Result<(), NativeContractError> {
        if matches!(value.as_str(), Some("none" | "no_think")) {
            return state.set_enabled(false);
        }
        if let Some(effort) = self.map_effort(value, template)? {
            if state.effort.is_some_and(|old| old != effort) {
                return Err(invalid("reasoning effort aliases disagree"));
            }
            state.effort = Some(effort);
            state.set_enabled(true)?;
        }
        Ok(())
    }

    fn add_budget(&self, state: &mut Reasoning, value: &Value) -> Result<(), NativeContractError> {
        if !self.supports_fixed_thinking_budget() {
            return Err(invalid(
                "model has no qualified native fixed thinking token budget",
            ));
        }
        let budget = value
            .as_u64()
            .filter(|v| *v > 0 && *v <= u32::MAX as u64)
            .ok_or_else(|| invalid("thinking token budget must be a positive u32 integer"))?;
        if state.budget.is_some_and(|old| old != budget) {
            return Err(invalid("thinking token budget aliases disagree"));
        }
        state.budget = Some(budget);
        state.set_enabled(true)
    }

    fn apply_reasoning(
        &self,
        body: &mut Map<String, Value>,
        state: Reasoning,
        output: u64,
    ) -> Result<(), NativeContractError> {
        let had_kwargs = body.contains_key("chat_template_kwargs");
        let mut kwargs = body
            .remove("chat_template_kwargs")
            .and_then(|v| v.as_object().cloned())
            .unwrap_or_default();
        for key in [
            "reasoning_effort",
            "thinking",
            "enable_thinking",
            "no_think",
            "reasoning",
        ] {
            body.remove(key);
            kwargs.remove(key);
        }
        if let Some(budget) = state.budget {
            if budget >= output {
                return Err(invalid(
                    "thinking token budget must be less than max_tokens",
                ));
            }
            body.insert("thinking_token_budget".into(), budget.into());
        }
        if state.enabled == Some(false) && self.supports_forced_zero_thinking() {
            // This mode uses the native template and forces the engine into
            // final phase. False template toggles would alter the renderer and
            // are not the qualified implementation of forced zero thinking.
            body.insert("forced_zero_thinking".into(), true.into());
        } else if let Some(enabled) = state.enabled {
            kwargs.insert("enable_thinking".into(), enabled.into());
            if self.family != Family::Qwen {
                kwargs.insert("thinking".into(), enabled.into());
            }
        }
        if let Some(effort) = state.effort {
            match effort {
                NativeEffort::Name(name) => {
                    kwargs.insert("reasoning_effort".into(), name.into());
                }
                NativeEffort::Percent(percent) => {
                    let value = match percent {
                        25 => json!("low"),
                        50 => json!("high"),
                        75 => json!("xhigh"),
                        100 => json!("max"),
                        _ => json!(percent as f64 / 100.0),
                    };
                    body.insert("reasoning_effort".into(), value);
                }
            }
        }
        if had_kwargs || !kwargs.is_empty() {
            body.insert("chat_template_kwargs".into(), kwargs.into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeEffort {
    Name(&'static str),
    Percent(u64),
}
#[derive(Default)]
struct Reasoning {
    enabled: Option<bool>,
    effort: Option<NativeEffort>,
    budget: Option<u64>,
}
impl Reasoning {
    fn set_enabled(&mut self, value: bool) -> Result<(), NativeContractError> {
        if self.enabled.is_some_and(|old| old != value) {
            return Err(invalid("reasoning effort and thinking toggles disagree"));
        }
        self.enabled = Some(value);
        Ok(())
    }
}

fn normalize_cache(body: &mut Map<String, Value>) -> Result<(), NativeContractError> {
    if let Some(value) = body.get("prompt_cache_retention") {
        if value.as_str() != Some("in_memory") {
            return Err(invalid("explicit cache retention TTL is unsupported"));
        }
        body.remove("prompt_cache_retention");
    }
    let mut namespace: Option<String> = None;
    for key in ["prompt_cache_key", "cache_salt"] {
        if let Some(value) = body.get(key) {
            let value = value
                .as_str()
                .filter(|s| !s.is_empty() && s.len() <= 1024)
                .ok_or_else(|| {
                    invalid("cache namespace must be a nonempty string of at most 1024 UTF-8 bytes")
                })?;
            if namespace.as_ref().is_some_and(|old| old != value) {
                return Err(invalid("prompt_cache_key and cache_salt disagree"));
            }
            namespace = Some(value.into());
        }
    }
    body.remove("prompt_cache_key");
    if let Some(value) = namespace {
        body.insert("cache_salt".into(), value.into());
    }
    Ok(())
}

/// Installed DS/GF renderers accept developer themselves. Native template
/// source alone is insufficient evidence: GF's actual renderer handles the
/// role before Jinja. For rejecting renderers, adapt instructions to supported
/// system messages with explicit original role/priority. Never hoist history.
fn normalize_native_instruction_roles(
    body: &mut Map<String, Value>,
    family: Family,
) -> Result<(), NativeContractError> {
    if matches!(family, Family::DeepSeek | Family::GlmFlash) {
        return Ok(());
    }
    let messages = body
        .get_mut("messages")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| invalid("messages must be an array"))?;
    let leading = messages
        .iter()
        .take_while(|message| {
            matches!(
                message.get("role").and_then(Value::as_str),
                Some("system" | "developer")
            )
        })
        .count();
    for message in &mut messages[leading..] {
        if message.get("role").and_then(Value::as_str) == Some("developer") {
            let text = instruction_text(message.get("content"))?;
            message["role"] = json!("system");
            message["content"] = json!(format!(
                "[Instruction role: developer]\nSystem instructions take precedence over this developer instruction.\n\n{text}"
            ));
        }
    }
    if !messages[..leading]
        .iter()
        .any(|message| message.get("role").and_then(Value::as_str) == Some("developer"))
    {
        return Ok(());
    }
    if leading == 1 {
        messages[0]["role"] = json!("system");
        return Ok(());
    }
    let mut content = String::from(
        "The following instruction messages retain their original order and priority. System instructions take precedence over developer instructions.\n\n",
    );
    for (index, message) in messages[..leading].iter().enumerate() {
        let message = message
            .as_object()
            .ok_or_else(|| invalid("instruction message must be an object"))?;
        if message
            .keys()
            .any(|key| !["role", "content"].contains(&key.as_str()))
        {
            return Err(invalid(
                "leading instruction metadata cannot be merged without losing its semantics",
            ));
        }
        let role = message.get("role").and_then(Value::as_str).unwrap();
        let text = instruction_text(message.get("content"))?;
        content.push_str(&format!("[Instruction {}: {role}]\n", index + 1));
        content.push_str(&text);
        content.push_str("\n\n");
    }
    messages.splice(..leading, [json!({"role":"system","content":content})]);
    Ok(())
}

fn instruction_text(content: Option<&Value>) -> Result<String, NativeContractError> {
    match content {
        Some(Value::String(text)) => Ok(text.clone()),
        Some(Value::Array(blocks)) => {
            let mut text = String::new();
            for block in blocks {
                let object = block
                    .as_object()
                    .ok_or_else(|| invalid("instruction blocks must be text objects"))?;
                if object
                    .keys()
                    .any(|key| !["type", "text"].contains(&key.as_str()))
                    || object.get("type").and_then(Value::as_str) != Some("text")
                {
                    return Err(invalid(
                        "instruction block semantics cannot be merged without loss",
                    ));
                }
                text.push_str(
                    object
                        .get("text")
                        .and_then(Value::as_str)
                        .ok_or_else(|| invalid("instruction text must be a string"))?,
                );
            }
            Ok(text)
        }
        _ => Err(invalid("merged instruction messages require text content")),
    }
}

pub fn count_body(body: &Value) -> Value {
    let mut body = body.clone();
    if let Some(object) = body.as_object_mut() {
        for key in [
            "max_tokens",
            "max_completion_tokens",
            "max_output_tokens",
            "stream",
            "stream_options",
        ] {
            object.remove(key);
        }
    }
    body
}

pub fn canonicalize(body: Value, model: &str) -> Result<Value, NativeContractError> {
    ModelContract::for_model(model)?.normalize(body)
}
pub fn requested_output(body: &Value, model: &str) -> Result<u64, NativeContractError> {
    ModelContract::for_model(model)?.output_budget(body)
}
pub fn context_limit(model: &str) -> Result<u64, NativeContractError> {
    Ok(ModelContract::for_model(model)?.context_limit)
}

fn validate_chat(body: &Map<String, Value>) -> Result<(), NativeContractError> {
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .filter(|v| !v.is_empty())
        .ok_or_else(|| invalid("messages must be a nonempty array"))?;
    let mut pending = HashSet::new();
    let mut seen = HashSet::new();
    for message in messages {
        let message = message
            .as_object()
            .ok_or_else(|| invalid("message must be an object"))?;
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .filter(|v| ["system", "developer", "user", "assistant", "tool"].contains(v))
            .ok_or_else(|| invalid("unsupported message role"))?;
        if let Some(content) = message.get("content") {
            match content {
                Value::Null | Value::String(_) => (),
                Value::Array(blocks)
                    if blocks
                        .iter()
                        .all(|b| b.is_object() && b.get("type").is_some_and(Value::is_string)) =>
                {
                    ()
                }
                _ => {
                    return Err(invalid(
                        "message content must be text, typed blocks or null",
                    ));
                }
            }
        }
        if !pending.is_empty() && role != "tool" {
            return Err(invalid("tool results are required before the next turn"));
        }
        if role == "tool" {
            let id = message
                .get("tool_call_id")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("tool result id is required"))?;
            if !pending.remove(id) {
                return Err(invalid("tool result must match an unanswered call"));
            }
        }
        if let Some(calls) = message.get("tool_calls") {
            if role != "assistant" {
                return Err(invalid("tool_calls must belong to assistant"));
            }
            for call in calls
                .as_array()
                .ok_or_else(|| invalid("tool_calls must be an array"))?
            {
                if call.get("type").and_then(Value::as_str) != Some("function") {
                    return Err(invalid("historical tool call must be a function"));
                }
                let id = call
                    .get("id")
                    .and_then(Value::as_str)
                    .filter(|v| !v.is_empty())
                    .ok_or_else(|| invalid("tool call id is required"))?;
                if !seen.insert(id.to_owned()) {
                    return Err(invalid("tool call ids must be unique"));
                }
                let function = call
                    .get("function")
                    .and_then(Value::as_object)
                    .ok_or_else(|| invalid("tool call function is required"))?;
                if function
                    .get("name")
                    .and_then(Value::as_str)
                    .is_none_or(str::is_empty)
                {
                    return Err(invalid("tool call name is required"));
                }
                let arguments = function
                    .get("arguments")
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid("tool arguments must be a JSON string"))?;
                serde_json::from_str::<Value>(arguments)
                    .map_err(|_| invalid("historical tool arguments must be complete JSON"))?;
                pending.insert(id.to_owned());
            }
        }
    }
    if !pending.is_empty() {
        return Err(invalid("assistant tool calls are missing results"));
    }
    let mut names = HashSet::new();
    if let Some(tools) = body.get("tools") {
        for tool in tools
            .as_array()
            .ok_or_else(|| invalid("tools must be an array"))?
        {
            if tool.get("type").and_then(Value::as_str) != Some("function") {
                return Err(invalid("tools must be function definitions"));
            }
            let function = tool
                .get("function")
                .and_then(Value::as_object)
                .ok_or_else(|| invalid("tool function is required"))?;
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .filter(|v| !v.is_empty())
                .ok_or_else(|| invalid("tool function name is required"))?;
            if !names.insert(name) {
                return Err(invalid("tool function names must be unique"));
            }
            if function
                .get("strict")
                .is_some_and(|v| !v.is_null() && !v.is_boolean())
            {
                return Err(invalid("tool strict must be boolean or null"));
            }
            if let Some(schema) = function.get("parameters") {
                validate_schema(schema)?;
            }
        }
    }
    if let Some(choice) = body.get("tool_choice") {
        match choice.as_str() {
            Some("auto" | "none") => (),
            Some("required") if !names.is_empty() => (),
            None if choice.get("type").and_then(Value::as_str) == Some("function")
                && choice
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(Value::as_str)
                    .is_some_and(|n| names.contains(n)) =>
            {
                ()
            }
            _ => {
                return Err(invalid(
                    "tool_choice must select a declared function or supported mode",
                ));
            }
        }
    }
    if body
        .get("parallel_tool_calls")
        .is_some_and(|v| !v.is_boolean())
    {
        return Err(invalid("parallel_tool_calls must be boolean"));
    }
    if body.get("stream").is_some_and(|v| !v.is_boolean()) {
        return Err(invalid("stream must be boolean"));
    }
    if let Some(options) = body.get("stream_options") {
        let options = options
            .as_object()
            .ok_or_else(|| invalid("stream_options must be an object"))?;
        if options
            .get("include_usage")
            .is_some_and(|v| !v.is_boolean())
        {
            return Err(invalid("include_usage must be boolean"));
        }
    }
    Ok(())
}

/// Chat strict has stronger input-schema requirements than Messages strict.
/// Check the public Chat definition without rewriting optional fields, unions,
/// numeric constraints, or the existing native JSON Schema extensions.
pub fn validate_chat_strict_tools(body: &Value) -> Result<(), NativeContractError> {
    let Some(tools) = body.get("tools").and_then(Value::as_array) else {
        return Ok(());
    };
    for tool in tools {
        let function = &tool["function"];
        if function.get("strict") != Some(&Value::Bool(true)) {
            continue;
        }
        let root = function
            .get("parameters")
            .ok_or_else(|| invalid("strict tool requires parameters JSON Schema"))?;
        let mut schemas = vec![root];
        while let Some(schema) = schemas.pop() {
            let Some(object) = schema.as_object() else {
                continue;
            };
            let object_type = object.get("type").is_some_and(|value| {
                value.as_str() == Some("object")
                    || value
                        .as_array()
                        .is_some_and(|types| types.iter().any(|t| t.as_str() == Some("object")))
            });
            if object_type || object.contains_key("properties") {
                if object.get("additionalProperties") != Some(&Value::Bool(false)) {
                    return Err(invalid(
                        "Chat strict object schemas require additionalProperties: false",
                    ));
                }
                if let Some(properties) = object.get("properties").and_then(Value::as_object) {
                    let required = object.get("required").and_then(Value::as_array);
                    if properties.keys().any(|key| {
                        required.is_none_or(|required| {
                            !required
                                .iter()
                                .any(|field| field.as_str() == Some(key.as_str()))
                        })
                    }) {
                        return Err(invalid(
                            "Chat strict object schemas require every property in required; use a nullable type for optional values",
                        ));
                    }
                }
            }
            // Walk schema locations only. Values in enum/default/const/examples
            // are instance data and must never be interpreted as schemas.
            for key in [
                "properties",
                "patternProperties",
                "$defs",
                "definitions",
                "dependentSchemas",
            ] {
                if let Some(children) = object.get(key).and_then(Value::as_object) {
                    schemas.extend(children.values());
                }
            }
            for key in ["allOf", "anyOf", "oneOf", "prefixItems"] {
                if let Some(children) = object.get(key).and_then(Value::as_array) {
                    schemas.extend(children);
                }
            }
            for key in [
                "items",
                "contains",
                "additionalProperties",
                "unevaluatedProperties",
                "propertyNames",
                "not",
                "if",
                "then",
                "else",
            ] {
                if let Some(child) = object.get(key) {
                    if let Some(children) = child.as_array() {
                        schemas.extend(children);
                    } else {
                        schemas.push(child);
                    }
                }
            }
        }
    }
    Ok(())
}

/// Validate the supplied definition *as an instance of a trusted meta-schema*.
/// Never compile the user's schema: user regexes are not executed and external
/// $ref values are not fetched. The crate's bundled meta-schemas are reused.
fn validate_schema(schema: &Value) -> Result<(), NativeContractError> {
    use jsonschema::{Draft, Validator};
    use std::sync::OnceLock;
    static D4: OnceLock<Result<Validator, String>> = OnceLock::new();
    static D6: OnceLock<Result<Validator, String>> = OnceLock::new();
    static D7: OnceLock<Result<Validator, String>> = OnceLock::new();
    static D19: OnceLock<Result<Validator, String>> = OnceLock::new();
    static D20: OnceLock<Result<Validator, String>> = OnceLock::new();
    if !schema.is_object() {
        return Err(invalid("tool parameters must be a JSON Schema object"));
    }
    let (draft, url) = match schema.get("$schema") {
        None => (
            Draft::Draft202012,
            "https://json-schema.org/draft/2020-12/schema",
        ),
        Some(Value::String(value)) => match value.trim_end_matches('#') {
            "http://json-schema.org/draft-04/schema"
            | "https://json-schema.org/draft-04/schema" => {
                (Draft::Draft4, "http://json-schema.org/draft-04/schema")
            }
            "http://json-schema.org/draft-06/schema"
            | "https://json-schema.org/draft-06/schema" => {
                (Draft::Draft6, "http://json-schema.org/draft-06/schema")
            }
            "http://json-schema.org/draft-07/schema"
            | "https://json-schema.org/draft-07/schema" => {
                (Draft::Draft7, "http://json-schema.org/draft-07/schema")
            }
            "https://json-schema.org/draft/2019-09/schema" => (
                Draft::Draft201909,
                "https://json-schema.org/draft/2019-09/schema",
            ),
            "https://json-schema.org/draft/2020-12/schema" => (
                Draft::Draft202012,
                "https://json-schema.org/draft/2020-12/schema",
            ),
            _ => return Err(invalid("unsupported tool JSON Schema dialect")),
        },
        _ => return Err(invalid("tool JSON Schema dialect must be a string")),
    };
    let slot = match draft {
        Draft::Draft4 => &D4,
        Draft::Draft6 => &D6,
        Draft::Draft7 => &D7,
        Draft::Draft201909 => &D19,
        Draft::Draft202012 => &D20,
        _ => return Err(invalid("unsupported tool JSON Schema dialect")),
    };
    let validator = slot
        .get_or_init(|| {
            let trusted = json!({"$ref": url});
            jsonschema::options()
                .offline()
                .with_draft(draft)
                .should_validate_formats(false)
                .build(&trusted)
                .map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(|_| invalid("bundled tool meta-schema is unavailable"))?;
    if !validator.is_valid(schema) {
        return Err(invalid("invalid tool parameters JSON Schema"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    const DS: &str = "deepseek-ai/DeepSeek-V4.1-Flash";
    const QWEN: &str = "qwen3.8-flash-next";
    const GF: &str = "glm-5.3-flash";
    const GLM: &str = "zai-org/GLM-5.3";

    #[test]
    fn chat_strict_tools_validate_recursive_objects_without_rewriting_schema() {
        let nested = json!({"type":"object","properties":{"n":{"type":["number","null"],"multipleOf":0.5},
            "literal":{"enum":[{"type":"object","properties":{"x":{}}}]}},
            "required":["n","literal"],"additionalProperties":false});
        let mut body = request("qwen3.8-flash-next");
        body["tools"] = json!([{"type":"function","function":{"name":"lookup","strict":true,
            "parameters":{"type":"object","properties":{"rows":{"type":"array","items":{"$ref":"#/$defs/row"}}},
                "required":["rows"],"additionalProperties":false,"$defs":{"row":nested}}}}]);
        let original = body.clone();
        validate_chat_strict_tools(&body).unwrap();
        assert_eq!(body, original);
        for pointer in [
            "/tools/0/function/parameters",
            "/tools/0/function/parameters/$defs/row",
        ] {
            for field in ["required", "additionalProperties"] {
                let mut invalid = body.clone();
                invalid
                    .pointer_mut(pointer)
                    .unwrap()
                    .as_object_mut()
                    .unwrap()
                    .remove(field);
                assert!(
                    validate_chat_strict_tools(&invalid).is_err(),
                    "{pointer}/{field}"
                );
            }
        }
        body["tools"][0]["function"]["parameters"]["allOf"] =
            json!([{"properties":{"rows":{}},"required":["rows"],"additionalProperties":false}]);
        validate_chat_strict_tools(&body).unwrap();
    }

    #[test]
    fn chat_ordinary_tools_keep_best_effort_schema_and_parallel_requires_boolean() {
        for strict in [None, Some(json!(false)), Some(Value::Null)] {
            let mut body = request("qwen3.8-flash-next");
            body["tools"] = json!([{"type":"function","function":{"name":"lookup",
                "parameters":{"type":"object","properties":{"optional":{"type":"string"}}}}}]);
            if let Some(strict) = strict {
                body["tools"][0]["function"]["strict"] = strict;
            }
            validate_chat_strict_tools(&body).unwrap();
            let original = body["tools"].clone();
            let contract = ModelContract::for_model("qwen3.8-flash-next").unwrap();
            assert_eq!(contract.normalize(body.clone()).unwrap()["tools"], original);
            body["parallel_tool_calls"] = json!("false");
            assert!(contract.normalize(body.clone()).is_err());
            body["parallel_tool_calls"] = json!(false);
            assert!(contract.normalize(body).is_ok());
        }
    }
    fn request(model: &str) -> Value {
        json!({"model":model,"messages":[{"role":"user","content":"中文"}]})
    }
    fn identity(model: &str) -> NativeWorkerIdentity {
        NativeWorkerIdentity {
            model: model.into(),
            model_identity: "cpu-fake-only".into(),
            source_sha256: "a".repeat(64),
            template_sha256: "b".repeat(64),
        }
    }
    fn receipt(model: &str) -> VerifiedThinkingBudgetQualification {
        receipt_for_identity(identity(model))
    }
    fn receipt_for_identity(identity: NativeWorkerIdentity) -> VerifiedThinkingBudgetQualification {
        ThinkingBudgetQualification {
            identity: identity.clone(),
            evidence_sha256: "c".repeat(64),
            native_field: ThinkingBudgetField::ThinkingTokenBudget,
        }
        .verify(&identity)
        .unwrap()
    }
    fn qualified_contract(model: &str) -> ModelContract {
        let contract = ModelContract::for_model(model).unwrap();
        let workers = if contract.short_limit.is_some() {
            vec![
                (NativeTier::Short, receipt(model)),
                (NativeTier::Long, receipt(model)),
            ]
        } else {
            vec![(NativeTier::Shared, receipt(model))]
        };
        contract
            .with_thinking_budget_qualification(
                &VerifiedModelThinkingBudget::verify(model, &workers).unwrap(),
            )
            .unwrap()
    }
    fn normalized(model: &str, field: &str, value: Value) -> Result<Value, NativeContractError> {
        let mut body = request(model);
        body[field] = value;
        if model == GF {
            qualified_contract(model).normalize(body)
        } else {
            canonicalize(body, model)
        }
    }

    #[test]
    fn public_model_caps_and_defaults() {
        for (model, output, owner) in [
            (GLM, 131072, "zai-org"),
            (GF, 131072, "zai-org"),
            (DS, 393216, "deepseek-ai"),
            (QWEN, 262144, "qwen"),
        ] {
            let c = ModelContract::for_model(model).unwrap();
            assert_eq!(c.output_limit, output);
            assert_eq!(c.owner, owner);
            assert_eq!(c.model_card()["max_input_tokens"], CONTEXT_LIMIT);
            assert_eq!(
                c.requested_output(&request(model)).unwrap(),
                if model == DS { 65536 } else { 8192 }
            );
        }
        assert!(ModelContract::for_model("ds").is_err());
    }
    #[test]
    fn none_truthful_and_ds_defaults_canonical_aliases() {
        for model in [GLM, GF] {
            let contract = ModelContract::for_model(model).unwrap();
            assert_eq!(
                contract.model_card()["gateway_capabilities"]["reasoning"]["disabled_supported"],
                true
            );
            for selector in [
                json!({"reasoning_effort":"none"}),
                json!({"thinking":{"type":"disabled"}}),
                json!({"chat_template_kwargs":{"enable_thinking":false}}),
            ] {
                let mut body = request(model);
                body.as_object_mut()
                    .unwrap()
                    .extend(selector.as_object().unwrap().clone());
                let out = contract.normalize(body).unwrap();
                assert_eq!(out["forced_zero_thinking"], true);
                assert_eq!(count_body(&out)["forced_zero_thinking"], true);
                assert!(out.get("reasoning_effort").is_none());
                assert!(
                    out.get("chat_template_kwargs")
                        .is_none_or(|kwargs| kwargs.as_object().unwrap().is_empty())
                );
            }
        }
        for model in [DS, QWEN] {
            let out = normalized(model, "reasoning_effort", json!("none")).unwrap();
            assert_eq!(out["chat_template_kwargs"]["enable_thinking"], false);
            assert_eq!(out["max_tokens"], 8192);
        }
        assert_eq!(
            normalized(DS, "reasoning", json!({"effort":"max"})).unwrap()["max_tokens"],
            131072
        );
        assert_eq!(
            normalized(DS, "enable_thinking", json!(false)).unwrap()["max_tokens"],
            8192
        );
    }
    #[test]
    fn efforts_map_before_alias_comparison() {
        let mut body = request(QWEN);
        body["reasoning_effort"] = json!("high");
        body["chat_template_kwargs"] = json!({"reasoning_effort":"max","unrelated":null});
        let out = canonicalize(body, QWEN).unwrap();
        assert_eq!(out["chat_template_kwargs"]["reasoning_effort"], "xhigh");
        assert!(out["chat_template_kwargs"]["unrelated"].is_null());
        for (effort, native) in [
            ("minimal", "low"),
            ("medium", "high"),
            ("xhigh", "xhigh"),
            ("max", "max"),
        ] {
            assert_eq!(
                normalized(DS, "reasoning_effort", json!(effort)).unwrap()["reasoning_effort"],
                native
            );
        }
        assert_eq!(
            normalized(GLM, "reasoning_effort", json!("xhigh")).unwrap()["chat_template_kwargs"]["reasoning_effort"],
            "max"
        );
    }
    #[test]
    fn reasoning_conflicts_types_and_obsolete_aliases_fail() {
        for value in [json!(true), json!(1), json!(1.0), json!("ultra"), json!([])] {
            assert!(normalized(DS, "reasoning_effort", value).is_err());
        }
        let mut body = request(QWEN);
        body["reasoning_effort"] = json!("high");
        body["enable_thinking"] = json!(false);
        assert!(canonicalize(body, QWEN).is_err());
        assert!(normalized(DS, "chat_template_kwargs", json!({"reasoning_effort":101})).is_err());
        assert_eq!(
            normalized(DS, "chat_template_kwargs", json!({"reasoning_effort":99})).unwrap()["reasoning_effort"],
            0.99
        );
        assert_eq!(
            normalized(DS, "reasoning_effort", json!(0.245)).unwrap()["reasoning_effort"],
            0.24
        );
    }
    #[test]
    fn explicit_outputs_are_never_inflated_or_lost() {
        let mut body = request(GF);
        body["max_completion_tokens"] = json!(800);
        body["max_output_tokens"] = json!(800);
        let out = canonicalize(body, GF).unwrap();
        assert_eq!(out["max_tokens"], 800);
        assert!(out.get("max_completion_tokens").is_none());
        for value in [json!(false), json!(0), json!(-1), json!(1.0), json!(131073)] {
            assert!(normalized(GF, "max_tokens", value).is_err());
        }
        let mut body = request(GF);
        body["max_tokens"] = json!(800);
        body["max_output_tokens"] = json!(801);
        assert!(canonicalize(body, GF).is_err());
    }
    #[test]
    fn unqualified_models_reject_cap_and_qualified_cap_stays_inside_output() {
        for model in [GLM, GF, DS] {
            let contract = ModelContract::for_model(model).unwrap();
            let mut body = request(model);
            body["thinking_token_budget"] = json!(32);
            assert!(!contract.supports_fixed_thinking_budget());
            assert!(contract.normalize(body).is_err());
        }
        for model in [GLM, DS] {
            assert!(normalized(model, "thinking_token_budget", json!(32)).is_err());
        }
        let qwen = normalized(QWEN, "thinking_token_budget", json!(16)).unwrap();
        assert_eq!(qwen["thinking_token_budget"], 16);
        assert_eq!(qwen["chat_template_kwargs"]["enable_thinking"], true);
        let mut body = request(GF);
        body["max_tokens"] = json!(800);
        body["thinking"] = json!({"type":"enabled","budget_tokens":32});
        let out = qualified_contract(GF).normalize(body).unwrap();
        assert_eq!(out["thinking_token_budget"], 32);
        assert_eq!(out["max_tokens"], 800);
        for value in [json!(true), json!(0), json!(8192), json!(u64::MAX)] {
            assert!(normalized(GF, "thinking_token_budget", value).is_err());
        }
        assert!(
            normalized(
                GF,
                "chat_template_kwargs",
                json!({"thinking_token_budget":32})
            )
            .is_err()
        );
    }
    #[test]
    fn tier_boundaries_include_output_and_ds_reserve_only_short() {
        for (model, limit, reserve) in [(GLM, 65536, 0), (QWEN, 16384, 0), (DS, 65536, 4096)] {
            let c = ModelContract::for_model(model).unwrap();
            assert_eq!(
                c.select_tier(limit - reserve - 801, 800).unwrap(),
                NativeTier::Short
            );
            assert_eq!(
                c.select_tier(limit - reserve - 800, 800).unwrap(),
                NativeTier::Short
            );
            assert_eq!(
                c.select_tier(limit - reserve - 799, 800).unwrap(),
                NativeTier::Long
            );
            assert_eq!(
                c.select_tier(CONTEXT_LIMIT - 800, 800).unwrap(),
                NativeTier::Long
            );
            assert!(c.select_tier(CONTEXT_LIMIT - 799, 800).is_err());
            assert!(c.select_tier(u64::MAX, 1).is_err());
        }
        assert_eq!(
            ModelContract::for_model(GF)
                .unwrap()
                .select_tier(CONTEXT_LIMIT - 1, 1)
                .unwrap(),
            NativeTier::Shared
        );
    }
    #[test]
    fn namespace_equal_merge_utf8_bounds_and_ttl_rejection() {
        let mut body = request(QWEN);
        body["prompt_cache_key"] = json!("盐");
        body["cache_salt"] = json!("盐");
        let out = canonicalize(body, QWEN).unwrap();
        assert_eq!(out["cache_salt"], "盐");
        assert!(out.get("prompt_cache_key").is_none());
        for value in [
            json!(null),
            json!(false),
            json!(""),
            json!("盐".repeat(342)),
        ] {
            assert!(normalized(QWEN, "cache_salt", value).is_err());
        }
        let mut body = request(QWEN);
        body["prompt_cache_key"] = json!("S1");
        body["cache_salt"] = json!("S2");
        assert!(canonicalize(body, QWEN).is_err());
        assert!(normalized(QWEN, "prompt_cache_retention", json!("24h")).is_err());
        assert!(
            normalized(QWEN, "prompt_cache_retention", json!("in_memory"))
                .unwrap()
                .get("prompt_cache_retention")
                .is_none()
        );
    }
    #[test]
    fn raw_native_values_survive_and_count_uses_same_body() {
        let mut body = request(GF);
        body["max_tokens"] = json!(800);
        body["thinking_token_budget"] = json!(32);
        body["reasoning_budget"] = json!(0);
        body["cache_salt"] = json!("S1");
        body["extension"] = json!({"false":false,"null":null,"中文":[1.23456789]});
        body["stream"] = json!(false);
        let canonical = qualified_contract(GF).normalize(body.clone()).unwrap();
        assert_eq!(canonical["extension"], body["extension"]);
        assert_eq!(canonical["reasoning_budget"], 0);
        let count = count_body(&canonical);
        assert!(count.get("max_tokens").is_none());
        assert!(count.get("stream").is_none());
        assert_eq!(count["thinking_token_budget"], 32);
        assert_eq!(count["cache_salt"], "S1");
        assert_eq!(canonical["max_tokens"], 800);
    }
    #[test]
    fn qwen_patched_native_template_system_sequence_is_unchanged() {
        // The qualified native Qwen template supports both leading and middle
        // system messages. Direct HTTP must keep that sequence for native
        // count and generation, rather than re-render or merge it here.
        let mut body = request(QWEN);
        body["messages"] = json!([
            {"role":"system","content":"first instructions"},
            {"role":"system","content":"second instructions"},
            {"role":"user","content":"question"},
            {"role":"assistant","content":"answer"},
            {"role":"system","content":"middle instructions"},
            {"role":"user","content":"follow up"}
        ]);
        let canonical = canonicalize(body.clone(), QWEN).unwrap();
        assert_eq!(canonical["messages"], body["messages"]);
        assert_eq!(count_body(&canonical)["messages"], body["messages"]);
    }
    #[test]
    fn schema_meta_validation_no_user_regex_or_external_ref_execution() {
        for schema in [
            json!({"type":"invalid"}),
            json!({"required":[1]}),
            json!({"properties":{"x":{"type":5}}}),
            json!({"$schema":"https://example.org/dialect"}),
        ] {
            assert!(
                validate_schema(&schema).is_err(),
                "invalid schema was accepted: {schema}"
            );
        }
        for dialect in [
            "http://json-schema.org/draft-04/schema#",
            "http://json-schema.org/draft-06/schema#",
            "http://json-schema.org/draft-07/schema#",
            "https://json-schema.org/draft/2019-09/schema",
            "https://json-schema.org/draft/2020-12/schema",
        ] {
            assert!(
                validate_schema(
                    &json!({"$schema":dialect,"type":"object","properties":{"x":{"type":"string"}}})
                )
                .is_ok()
            );
        }
        assert!(validate_schema(&json!({"type":"string","pattern":"(?=uncompiled)","$ref":"http://127.0.0.1:1/never-fetch"})).is_ok());
    }
    #[test]
    fn tool_definition_and_history_reject_before_dispatch() {
        let mut body = request(QWEN);
        body["tool_choice"] = json!("required");
        assert!(canonicalize(body.clone(), QWEN).is_err());
        body["tools"] = json!([{"type":"function","function":{"name":"f","strict":true,"parameters":{"type":"object"}}}]);
        assert!(canonicalize(body, QWEN).is_ok());
        let mut body = request(QWEN);
        body["messages"] = json!([{"role":"assistant","content":null,"tool_calls":[{"id":"id","type":"function","function":{"name":"f","arguments":"{}"}}]},{"role":"tool","tool_call_id":"id","content":"ok"},{"role":"user","content":"next"}]);
        assert!(canonicalize(body.clone(), QWEN).is_ok());
        body["messages"][1]["tool_call_id"] = json!("bad");
        assert!(canonicalize(body, QWEN).is_err());
    }
    #[test]
    fn qualified_positive_caps_cover_all_models_without_changing_none() {
        for model in [GLM, GF, DS, QWEN] {
            let contract = qualified_contract(model);
            assert_eq!(
                contract.model_card()["gateway_capabilities"]["reasoning"]["fixed_budget_supported"],
                true
            );
            let mut body = request(model);
            body["max_tokens"] = json!(800);
            body["thinking_token_budget"] = json!(32);
            let canonical = contract.normalize(body.clone()).unwrap();
            assert_eq!(canonical["max_tokens"], 800);
            assert_eq!(canonical["thinking_token_budget"], 32);
            assert_eq!(count_body(&canonical)["thinking_token_budget"], 32);
            body["reasoning_effort"] = json!("none");
            assert!(contract.normalize(body).is_err());
            if model == GLM {
                let mut body = request(model);
                body["reasoning_effort"] = json!("none");
                assert!(contract.normalize(body).is_err());
            }
        }
    }
    #[test]
    fn qualification_binds_source_template_model_and_evidence() {
        let identity = identity(QWEN);
        let original = ThinkingBudgetQualification {
            identity: identity.clone(),
            evidence_sha256: "c".repeat(64),
            native_field: ThinkingBudgetField::ThinkingTokenBudget,
        };
        assert!(
            original
                .verify(&identity)
                .unwrap()
                .matches_current(&identity)
        );
        let mut invalid = original.clone();
        invalid.evidence_sha256 = "true".into();
        assert!(invalid.verify(&identity).is_err());
        let mut current = identity.clone();
        current.source_sha256 = "d".repeat(64);
        assert!(original.verify(&current).is_err());
        assert!(
            !original
                .verify(&identity)
                .unwrap()
                .matches_current(&current)
        );
        current = identity.clone();
        current.template_sha256 = "d".repeat(64);
        assert!(original.verify(&current).is_err());
        current = identity.clone();
        current.model_identity = "changed-weights".into();
        assert!(original.verify(&current).is_err());
        current = identity.clone();
        current.model = DS.into();
        assert!(original.verify(&current).is_err());
        current = identity.clone();
        current.source_sha256 = "A".repeat(64);
        assert!(original.verify(&current).is_err());
    }
    #[test]
    fn public_capability_requires_every_tier_and_model_matches() {
        let mut long_profile = identity(QWEN);
        long_profile.model_identity = "individually-qualified-long-profile".into();
        long_profile.source_sha256 = "d".repeat(64);
        long_profile.template_sha256 = "e".repeat(64);
        assert!(
            VerifiedModelThinkingBudget::verify(
                QWEN,
                &[
                    (NativeTier::Short, receipt(QWEN)),
                    (NativeTier::Long, receipt_for_identity(long_profile))
                ]
            )
            .is_ok()
        );
        for model in [GLM, DS, QWEN] {
            assert!(
                VerifiedModelThinkingBudget::verify(model, &[(NativeTier::Short, receipt(model))])
                    .is_err()
            );
            assert!(
                VerifiedModelThinkingBudget::verify(model, &[(NativeTier::Long, receipt(model))])
                    .is_err()
            );
            assert!(
                VerifiedModelThinkingBudget::verify(model, &[(NativeTier::Shared, receipt(model))])
                    .is_err()
            );
        }
        assert!(VerifiedModelThinkingBudget::verify(GF, &[]).is_err());
        assert!(
            VerifiedModelThinkingBudget::verify(GF, &[(NativeTier::Short, receipt(GF))]).is_err()
        );
        assert!(
            VerifiedModelThinkingBudget::verify(
                QWEN,
                &[
                    (NativeTier::Short, receipt(DS)),
                    (NativeTier::Long, receipt(DS))
                ]
            )
            .is_err()
        );
        let q =
            VerifiedModelThinkingBudget::verify(GF, &[(NativeTier::Shared, receipt(GF))]).unwrap();
        assert!(
            ModelContract::for_model(QWEN)
                .unwrap()
                .with_thinking_budget_qualification(&q)
                .is_err()
        );
    }
    fn forced_receipt(current: NativeWorkerIdentity) -> VerifiedForcedZeroThinkingQualification {
        ForcedZeroThinkingQualification {
            identity: current.clone(),
            evidence_sha256: "f".repeat(64),
            native_field: ForcedZeroThinkingField::ForcedZeroThinking,
        }
        .verify(&current)
        .unwrap()
    }
    fn forced_contract(model: &str) -> ModelContract {
        let contract = ModelContract::for_model(model).unwrap();
        let workers = if contract.short_limit.is_some() {
            vec![
                (NativeTier::Short, forced_receipt(identity(model))),
                (NativeTier::Long, forced_receipt(identity(model))),
            ]
        } else {
            vec![(NativeTier::Shared, forced_receipt(identity(model)))]
        };
        contract
            .with_forced_zero_thinking_qualification(
                &VerifiedModelForcedZeroThinking::verify(model, &workers).unwrap(),
            )
            .unwrap()
    }
    #[test]
    fn unqualified_forced_zero_does_not_promote_existing_none_capability() {
        for model in [DS, QWEN] {
            let c = ModelContract::for_model(model).unwrap();
            assert!(!c.supports_forced_zero_thinking());
            assert_eq!(c.supports_disabled_thinking(), [DS, QWEN].contains(&model));
            let mut raw = request(model);
            raw["forced_zero_thinking"] = json!(true);
            assert!(c.normalize(raw).is_err());
        }
    }
    #[test]
    fn qualified_none_and_disabled_use_only_internal_forced_mode() {
        for model in [GLM, GF, DS, QWEN] {
            let c = forced_contract(model);
            assert!(c.supports_disabled_thinking());
            assert_eq!(
                c.model_card()["gateway_capabilities"]["reasoning"]["disabled_supported"],
                true
            );
            for selector in [
                json!({"reasoning_effort":"none"}),
                json!({"thinking":{"type":"disabled"}}),
            ] {
                let mut body = request(model);
                for (key, value) in selector.as_object().unwrap() {
                    body[key] = value.clone();
                }
                body["chat_template_kwargs"] =
                    json!({"unrelated":false,"no_think":true,"reasoning_effort":"none"});
                body["messages"] = json!([
                    {"role":"assistant","content":null,"tool_calls":[{"id":"literal-id","type":"function","function":{"name":"f","arguments":"{\"literal\":\"<think>tool argument</think>\"}"}}]},
                    {"role":"tool","tool_call_id":"literal-id","content":"<think>tool result</think>"},
                    {"role":"user","content":"literal <think>text</think> and ```<think>code</think>``` must remain"}
                ]);
                let canonical = c.normalize(body.clone()).unwrap();
                assert_eq!(canonical["forced_zero_thinking"], true);
                assert_eq!(canonical["messages"], body["messages"]);
                assert_eq!(
                    canonical["chat_template_kwargs"],
                    json!({"unrelated":false})
                );
                assert!(canonical.get("reasoning_effort").is_none());
                assert_eq!(count_body(&canonical)["forced_zero_thinking"], true);
                assert_eq!(c.normalize(canonical.clone()).unwrap(), canonical);
                assert!(ModelContract::requires_forced_zero_thinking(&canonical));
                if model == DS {
                    assert_eq!(canonical["max_tokens"], 8192);
                }
            }
        }
    }
    #[test]
    fn forced_zero_conflicts_with_positive_cap_and_enabled_aliases() {
        for model in [GLM, GF, DS, QWEN] {
            let c = qualified_contract(model)
                .with_forced_zero_thinking_qualification(&{
                    let workers = if model == GF {
                        vec![(NativeTier::Shared, forced_receipt(identity(model)))]
                    } else {
                        vec![
                            (NativeTier::Short, forced_receipt(identity(model))),
                            (NativeTier::Long, forced_receipt(identity(model))),
                        ]
                    };
                    VerifiedModelForcedZeroThinking::verify(model, &workers).unwrap()
                })
                .unwrap();
            let mut body = request(model);
            body["reasoning_effort"] = json!("none");
            body["thinking_token_budget"] = json!(32);
            assert!(c.normalize(body).is_err());
            for value in [json!(false), json!("true"), json!(1), json!(null)] {
                let mut body = request(model);
                body["forced_zero_thinking"] = value;
                assert!(c.normalize(body).is_err());
            }
            let mut body = request(model);
            body["forced_zero_thinking"] = json!(true);
            body["reasoning_effort"] = json!("high");
            assert!(c.normalize(body).is_err());
            let mut body = request(model);
            body["chat_template_kwargs"] = json!({"forced_zero_thinking":true});
            assert!(c.normalize(body).is_err());
        }
    }
    #[test]
    fn forced_zero_receipt_identity_evidence_model_and_tier_fail_closed() {
        let current = identity(GLM);
        let mut q = ForcedZeroThinkingQualification {
            identity: current.clone(),
            evidence_sha256: "f".repeat(64),
            native_field: ForcedZeroThinkingField::ForcedZeroThinking,
        };
        let verified = q.verify(&current).unwrap();
        let mut changed = current.clone();
        changed.source_sha256 = "e".repeat(64);
        assert!(!verified.matches_current(&changed));
        assert!(q.verify(&changed).is_err());
        q.evidence_sha256 = "true".into();
        assert!(q.verify(&current).is_err());
        assert!(
            VerifiedModelForcedZeroThinking::verify(GLM, &[(NativeTier::Short, verified.clone())])
                .is_err()
        );
        assert!(
            VerifiedModelForcedZeroThinking::verify(
                DS,
                &[
                    (NativeTier::Short, verified.clone()),
                    (NativeTier::Long, verified.clone())
                ]
            )
            .is_err()
        );
        let mut long = identity(GLM);
        long.model_identity = "long-other-profile".into();
        long.source_sha256 = "e".repeat(64);
        let model = VerifiedModelForcedZeroThinking::verify(
            GLM,
            &[
                (NativeTier::Short, verified),
                (NativeTier::Long, forced_receipt(long)),
            ],
        )
        .unwrap();
        assert!(
            ModelContract::for_model(GF)
                .unwrap()
                .with_forced_zero_thinking_qualification(&model)
                .is_err()
        );
        assert!(
            ModelContract::for_model(GLM)
                .unwrap()
                .with_forced_zero_thinking_qualification(&model)
                .unwrap()
                .supports_disabled_thinking()
        );
    }
    #[test]
    fn native_closed_private_state_is_rejected_at_root_and_template_kwargs() {
        for model in [GLM, GF, DS, QWEN] {
            for c in [
                ModelContract::for_model(model).unwrap(),
                forced_contract(model),
            ] {
                let mut root = request(model);
                root["_forced_zero_thinking_closed"] = json!(true);
                assert!(c.normalize(root).is_err());
                let mut nested = request(model);
                nested["chat_template_kwargs"] = json!({"_forced_zero_thinking_closed":true});
                assert!(c.normalize(nested).is_err());
                let mut literal = request(model);
                literal["messages"][0]["content"] =
                    json!("literal _forced_zero_thinking_closed and <think>text</think>");
                assert_eq!(
                    c.normalize(literal.clone()).unwrap()["messages"],
                    literal["messages"]
                );
            }
        }
    }
    #[test]
    fn private_budget_scope_requires_nonempty_matching_actual_tier() {
        for (model, tier) in [
            (GLM, NativeTier::Short),
            (DS, NativeTier::Long),
            (QWEN, NativeTier::Short),
            (GF, NativeTier::Shared),
        ] {
            let one = vec![(tier, receipt(model))];
            let scoped =
                VerifiedModelThinkingBudget::verify_single_tier(model, tier, &one).unwrap();
            assert!(
                ModelContract::for_model(model)
                    .unwrap()
                    .with_thinking_budget_qualification(&scoped)
                    .unwrap()
                    .supports_fixed_thinking_budget()
            );
            assert!(VerifiedModelThinkingBudget::verify_single_tier(model, tier, &[]).is_err());
            if model != GF {
                assert!(VerifiedModelThinkingBudget::verify(model, &one).is_err());
            }
        }
        assert!(
            VerifiedModelThinkingBudget::verify_single_tier(
                QWEN,
                NativeTier::Shared,
                &[(NativeTier::Shared, receipt(QWEN))]
            )
            .is_err()
        );
        assert!(
            VerifiedModelThinkingBudget::verify_single_tier(
                GF,
                NativeTier::Short,
                &[(NativeTier::Short, receipt(GF))]
            )
            .is_err()
        );
        assert!(
            VerifiedModelThinkingBudget::verify_single_tier(
                QWEN,
                NativeTier::Short,
                &[(NativeTier::Long, receipt(QWEN))]
            )
            .is_err()
        );
        assert!(
            VerifiedModelThinkingBudget::verify_single_tier(
                QWEN,
                NativeTier::Short,
                &[(NativeTier::Short, receipt(DS))]
            )
            .is_err()
        );
    }
    #[test]
    fn private_forced_zero_scope_does_not_satisfy_full_model_qualification() {
        for (model, tier) in [
            (GLM, NativeTier::Long),
            (DS, NativeTier::Short),
            (QWEN, NativeTier::Short),
            (GF, NativeTier::Shared),
        ] {
            let one = vec![(tier, forced_receipt(identity(model)))];
            let scoped =
                VerifiedModelForcedZeroThinking::verify_single_tier(model, tier, &one).unwrap();
            assert!(
                ModelContract::for_model(model)
                    .unwrap()
                    .with_forced_zero_thinking_qualification(&scoped)
                    .unwrap()
                    .supports_forced_zero_thinking()
            );
            assert!(VerifiedModelForcedZeroThinking::verify_single_tier(model, tier, &[]).is_err());
            if model != GF {
                assert!(VerifiedModelForcedZeroThinking::verify(model, &one).is_err());
            }
        }
        assert!(
            VerifiedModelForcedZeroThinking::verify_single_tier(
                QWEN,
                NativeTier::Shared,
                &[(NativeTier::Shared, forced_receipt(identity(QWEN)))]
            )
            .is_err()
        );
        assert!(
            VerifiedModelForcedZeroThinking::verify_single_tier(
                GF,
                NativeTier::Long,
                &[(NativeTier::Long, forced_receipt(identity(GF)))]
            )
            .is_err()
        );
        assert!(
            VerifiedModelForcedZeroThinking::verify_single_tier(
                QWEN,
                NativeTier::Short,
                &[
                    (NativeTier::Short, forced_receipt(identity(QWEN))),
                    (NativeTier::Long, forced_receipt(identity(QWEN)))
                ]
            )
            .is_err()
        );
        assert!(
            VerifiedModelForcedZeroThinking::verify_single_tier(
                QWEN,
                NativeTier::Short,
                &[(NativeTier::Short, forced_receipt(identity(DS)))]
            )
            .is_err()
        );
    }
    #[test]
    fn developer_instructions_are_mapped_before_count_with_order_and_priority() {
        for model in [GLM, QWEN] {
            let c = ModelContract::for_model(model).unwrap();
            let mut single = request(model);
            single["messages"] = json!([{"role":"developer","content":"Developer instruction  中文\n"},{"role":"user","content":"question"}]);
            let out = c.normalize(single.clone()).unwrap();
            assert_eq!(out["messages"][0]["role"], "system");
            assert_eq!(
                out["messages"][0]["content"],
                single["messages"][0]["content"]
            );
            assert_eq!(out["messages"][1], single["messages"][1]);
            assert_eq!(count_body(&out)["messages"], out["messages"]);
            let mut mixed = request(model);
            mixed["messages"] = json!([
                {"role":"system","content":"System instructions  中文\n"},
                {"role":"developer","content":"Developer instructions <think>literal</think>\n"},
                {"role":"user","content":"question"}]);
            let out = c.normalize(mixed.clone()).unwrap();
            assert_eq!(out["messages"].as_array().unwrap().len(), 2);
            let text = out["messages"][0]["content"].as_str().unwrap();
            assert!(
                text.contains("System instructions take precedence over developer instructions.")
            );
            assert!(text.contains("[Instruction 1: system]\nSystem instructions  中文\n"));
            assert!(text.contains(
                "[Instruction 2: developer]\nDeveloper instructions <think>literal</think>\n"
            ));
            assert!(
                text.find("[Instruction 1: system]").unwrap()
                    < text.find("[Instruction 2: developer]").unwrap()
            );
            assert_eq!(out["messages"][1], mixed["messages"][2]);
            assert_eq!(count_body(&out)["messages"], out["messages"]);
            assert_eq!(c.normalize(out.clone()).unwrap(), out);
            mixed["messages"][1]["name"] = json!("cannot silently merge this attribute");
            assert!(c.normalize(mixed).is_err());
            let mut middle = request(model);
            middle["messages"] = json!([{"role":"user","content":"before"},{"role":"developer","content":"middle instruction"},{"role":"user","content":"after"}]);
            let out = c.normalize(middle.clone()).unwrap();
            assert_eq!(out["messages"][0], middle["messages"][0]);
            assert_eq!(out["messages"][2], middle["messages"][2]);
            assert_eq!(out["messages"][1]["role"], "system");
            assert_eq!(
                out["messages"][1]["content"],
                "[Instruction role: developer]\nSystem instructions take precedence over this developer instruction.\n\nmiddle instruction"
            );
            assert_eq!(count_body(&out)["messages"], out["messages"]);
            assert_eq!(c.normalize(out.clone()).unwrap(), out);
        }
    }
    #[test]
    fn native_role_support_is_preserved_for_ds_and_glm_flash() {
        for model in [DS, GF] {
            let c = ModelContract::for_model(model).unwrap();
            for messages in [
                json!([{"role":"developer","content":"leading"},{"role":"user","content":"user"}]),
                json!([{"role":"system","content":"system"},{"role":"developer","content":"developer"},{"role":"user","content":"user"}]),
                json!([{"role":"user","content":"before"},{"role":"developer","content":"middle"},{"role":"user","content":"after"}]),
                json!([{"role":"user","content":"before"},{"role":"system","content":"middle"},{"role":"user","content":"after"}]),
            ] {
                let mut body = request(model);
                body["messages"] = messages;
                let out = c.normalize(body.clone()).unwrap();
                assert_eq!(out["messages"], body["messages"]);
                assert_eq!(count_body(&out)["messages"], body["messages"]);
            }
        }
    }
}
