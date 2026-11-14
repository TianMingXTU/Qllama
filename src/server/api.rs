//! HTTP 层：OpenAI 兼容路由、SSE 流式、错误体、指标端点。
//!
//! 本模块只做协议转换；推理调用全部经由 `Scheduler`（见 ADR 0002）。
//! 断连取消：`CancelGuard` 在 handler future 被 drop（包括客户端断开、
//! 超时、提前返回）时置位取消标志并丢弃 Receiver，调度器下一个
//! decode step 停止生成并释放 KV。

use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use axum::extract::rejection::JsonRejection;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::Future;
use futures::Stream;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::sampler::SamplerConfig;
use crate::server::auth::{self, AuthContext, Role};
use crate::server::capabilities;
use crate::server::metrics;
use crate::server::{audit, qos, schema, template};
use crate::server::{ApiError, Failure, GenerationRequest, ServerState, Session, SessionEvent};
use crate::tokenizer::EncodeOptions;

/// OpenAI 错误体；429 附带 Retry-After。
fn error_response(error: &ApiError) -> Response {
    let err = json!({
        "message": error.message,
        "type": error.kind,
        "code": error.status,
    });
    let mut body = json!({"error": err});
    if !error.unsupported.is_empty() {
        let names = error
            .unsupported
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let mut markers = vec!["qllama.unsupported".to_string()];
        markers.extend(error.unsupported.iter().cloned());
        body["error"]["qllama.unsupported"] = json!(markers);
        body["qllama"] = capabilities::unsupported_contract(&names);
        body["qllama.unsupported"] = json!(names);
    }
    let status = StatusCode::from_u16(error.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let mut response = (status, Json(body)).into_response();
    if error.status == 429 {
        response.headers_mut().insert(
            axum::http::header::RETRY_AFTER,
            axum::http::HeaderValue::from_static("1"),
        );
    }
    response
}

/// API key 校验（可选，`--api-key` 启用）。只保护 /v1 推理端点。
fn check_auth(
    state: &ServerState,
    headers: &HeaderMap,
    required: Role,
) -> Result<AuthContext, ApiError> {
    let config = state.auth_config();
    if config.keys.is_empty() {
        let pairs = header_pairs(headers);
        return Ok(AuthContext {
            key_id: "anonymous".into(),
            key_hash: auth::hash_key("anonymous"),
            tenant: headers
                .get("x-qllama-tenant")
                .and_then(|value| value.to_str().ok())
                .unwrap_or("default")
                .to_string(),
            role: Role::Admin,
            trace_id: auth::parse_traceparent(&pairs)
                .unwrap_or_else(|| format!("{:032x}", uuid_like_u128())),
            request_id: format!("req-{}", uuid_like()),
        });
    }
    auth::authenticate(&config, &header_pairs(headers), required, unix_now()).map_err(|reason| {
        // 角色不足是「已认证但越权」，用 403 与 401 区分；其余鉴权失败仍 401。
        let (status, kind) = if reason == "insufficient_role" {
            (403, "permission_error")
        } else {
            (401, "authentication_error")
        };
        ApiError {
            status,
            kind,
            message: format!("authentication failed: {reason}"),
            unsupported: Vec::new(),
        }
    })
}

pub fn router(state: Arc<ServerState>) -> Router<Arc<ServerState>> {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/metrics", get(metrics))
        .route("/v1/models", get(list_models))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(DefaultBodyLimit::max(state.limits.max_request_bytes))
}

async fn healthz() -> &'static str {
    "ok"
}

async fn metrics(State(state): State<Arc<ServerState>>) -> String {
    state.metrics.set_runtime(
        metrics::current_rss_bytes(),
        state.active_sessions(),
        state.loaded_models(),
        state.queue_depth.load(AtomicOrdering::Relaxed),
        state.started,
    );
    state.metrics.render()
}

async fn list_models(State(state): State<Arc<ServerState>>, headers: HeaderMap) -> Response {
    if let Err(error) = check_auth(&state, &headers, Role::Read) {
        return error_response(&error);
    }
    let data = state
        .registry_snapshot()
        .into_iter()
        .map(|entry| {
            json!({
                "id": entry.id,
                "object": "model",
                "owned_by": "qllama",
                "digest": entry.digest,
                "quantization": entry.quantization,
                "context_length": entry.context_length,
            })
        })
        .collect::<Vec<_>>();
    Json(json!({"object": "list", "data": data})).into_response()
}

/// 将 axum 的 422/400 JSON 解析拒绝统一为 OpenAI 形状的 400 错误体。
///
/// 请求正文不会被记录；错误体只包含解析器的分类信息。
fn parse_json<T: serde::de::DeserializeOwned>(
    body: Result<Json<T>, JsonRejection>,
) -> Result<T, ApiError> {
    body.map(|Json(value)| value).map_err(|rejection| {
        let status = rejection.status();
        ApiError {
            status: 400,
            kind: "invalid_request_error",
            message: format!("invalid JSON request ({status}): {}", rejection.body_text()),
            unsupported: Vec::new(),
        }
    })
}

/// OpenAI 请求统一契约：所有字段显式解析，未知字段由 serde 400 拒绝。
#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
struct CommonFields {
    model: Option<String>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    stop: Option<StopInput>,
    stream: Option<bool>,
    seed: Option<u64>,
    stream_options: Option<StreamOptions>,
    frequency_penalty: Option<Value>,
    presence_penalty: Option<Value>,
    logprobs: Option<Value>,
    logit_bias: Option<Value>,
    response_format: Option<Value>,
    tools: Option<Value>,
    tool_choice: Option<Value>,
    parallel_tool_calls: Option<Value>,
    echo: Option<Value>,
    best_of: Option<Value>,
    user: Option<String>,
    n: Option<usize>,
    reasoning: Option<Value>,
    reasoning_effort: Option<Value>,
    repeat_penalty: Option<f32>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
struct StreamOptions {
    include_usage: Option<bool>,
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct CompletionsBody {
    #[serde(flatten)]
    common: CommonFields,
    prompt: PromptInput,
}

#[derive(Deserialize, Debug)]
#[serde(untagged)]
enum PromptInput {
    Text(String),
    Array(Vec<String>),
}

#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
enum StopInput {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct ChatBody {
    #[serde(flatten)]
    common: CommonFields,
    messages: Vec<ChatMessage>,
}

/// 可复现合同与响应元数据；不包含 prompt/response 正文。
fn response_metadata(state: &ServerState, seed: Option<u64>, user: &Option<String>) -> Value {
    let snapshot = state.capability_snapshot();
    json!({
        "seed": seed,
        "model_digest": state.model_digest(),
        "quantization": snapshot["reproducibility"]["quantization"].clone(),
        "kv_dtype": snapshot["reproducibility"]["kv_dtype"].clone(),
        "chat_template": snapshot["reproducibility"]["chat_template"].clone(),
        "template_fallback": snapshot["template_fallback"].clone(),
        "prefix_cache": state.runtime_options().prefix_cache,
        "config_snapshot": snapshot["reproducibility"]["config_snapshot"].clone(),
        "user_accepted_without_effect": user.is_some(),
    })
}

fn header_pairs(headers: &HeaderMap) -> Vec<(String, String)> {
    headers
        .iter()
        .map(|(name, value)| {
            (
                name.as_str().to_ascii_lowercase(),
                value.to_str().unwrap_or_default().to_string(),
            )
        })
        .collect()
}

/// 请求三态中「明确降级」的字段：返回 400 并列出 qllama.unsupported。
fn reject_unsupported_fields(common: &CommonFields) -> Result<(), ApiError> {
    let mut unsupported = Vec::new();
    let fields = [
        ("frequency_penalty", common.frequency_penalty.as_ref()),
        ("presence_penalty", common.presence_penalty.as_ref()),
        ("logprobs", common.logprobs.as_ref()),
        ("logit_bias", common.logit_bias.as_ref()),
        ("tools", common.tools.as_ref()),
        ("tool_choice", common.tool_choice.as_ref()),
        ("parallel_tool_calls", common.parallel_tool_calls.as_ref()),
        ("echo", common.echo.as_ref()),
        ("best_of", common.best_of.as_ref()),
        ("reasoning", common.reasoning.as_ref()),
        ("reasoning_effort", common.reasoning_effort.as_ref()),
    ];
    for (name, value) in fields {
        if value.is_some_and(|value| !value.is_null()) {
            unsupported.push(name);
        }
    }
    if common.n.is_some_and(|n| n > 1) {
        unsupported.push("n_greater_than_1");
    }
    if unsupported.is_empty() {
        Ok(())
    } else {
        Err(ApiError {
            status: 400,
            kind: "invalid_request_error",
            message: format!("unsupported request fields: {}", unsupported.join(", ")),
            unsupported: Vec::new(),
        }
        .with_unsupported(unsupported))
    }
}

/// 解析 response_format：json_object 或本项目支持的 JSON Schema 子集。
/// tools 与 response_format 组合冲突时明确 400，避免静默忽略任一字段。
fn parse_response_format(common: &CommonFields) -> Result<Option<schema::JsonSchema>, ApiError> {
    let Some(format) = common.response_format.as_ref() else {
        return Ok(None);
    };
    if format.is_null() {
        return Ok(None);
    }
    let format_type = format
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::bad_request("response_format.type must be a string"))?;
    match format_type {
        "json_object" => Ok(Some(schema::json_object_schema())),
        "json_schema" => {
            let value = format
                .get("json_schema")
                .and_then(|schema| schema.get("schema"))
                .or_else(|| format.get("schema"))
                .ok_or_else(|| {
                    ApiError::bad_request("response_format.json_schema.schema is required")
                })?;
            schema::from_value(value)
                .map(Some)
                .map_err(ApiError::bad_request)
        }
        _ => Err(ApiError::bad_request(
            "response_format.type must be json_object or json_schema",
        )),
    }
}

fn extract_stop(stop: Option<StopInput>) -> Vec<String> {
    match stop {
        Some(StopInput::One(text)) => vec![text],
        Some(StopInput::Many(items)) => items,
        None => Vec::new(),
    }
}

/// 把 OpenAI 参数翻译成内部请求；所有校验失败返回 400。
fn build_request(
    state: &ServerState,
    prompt_text: &str,
    common: &CommonFields,
) -> Result<GenerationRequest, ApiError> {
    let tokenizer = state.engine.tokenizer.as_ref().ok_or_else(|| {
        ApiError::bad_request("model has no tokenizer metadata; text prompts are unavailable")
    })?;
    let prompt_ids = tokenizer
        .encode_with_options(
            prompt_text,
            &EncodeOptions {
                add_bos: true,
                add_eos: false,
            },
        )
        .map_err(ApiError::bad_request)?;
    if prompt_ids.is_empty() {
        return Err(ApiError::bad_request("prompt encodes to no tokens"));
    }
    if prompt_ids.len() >= state.engine.ctx_size {
        return Err(ApiError::bad_request(format!(
            "prompt is {} tokens, leaving no room to generate within context size {}",
            prompt_ids.len(),
            state.engine.ctx_size
        )));
    }
    let stop_texts = extract_stop(common.stop.clone());
    if stop_texts.len() > 4 {
        return Err(ApiError::bad_request(
            "at most 4 stop sequences are supported",
        ));
    }
    if stop_texts.iter().any(String::is_empty) {
        return Err(ApiError::bad_request("stop sequences must not be empty"));
    }
    let temperature = common.temperature.unwrap_or(1.0);
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(ApiError::bad_request(
            "temperature must be a finite non-negative number",
        ));
    }
    let top_p = common.top_p.unwrap_or(1.0);
    if !top_p.is_finite() || top_p <= 0.0 {
        return Err(ApiError::bad_request(
            "top_p must be a finite positive number",
        ));
    }
    let repeat_penalty = common.repeat_penalty.unwrap_or(1.0);
    if !repeat_penalty.is_finite() || repeat_penalty <= 0.0 {
        return Err(ApiError::bad_request(
            "repeat_penalty must be a finite positive number",
        ));
    }
    if common.max_tokens == Some(0) {
        return Err(ApiError::bad_request("max_tokens must be positive"));
    }
    let max_tokens = common
        .max_tokens
        .unwrap_or(16)
        .min(state.engine.ctx_size - prompt_ids.len());
    if max_tokens == 0 {
        return Err(ApiError::bad_request("prompt fills the context"));
    }
    let schema = parse_response_format(common)?;
    // 词表可达性预检：对象/数组 schema 依赖词表中的 '{' / '['。夹具小模型
    // 词表缺结构字符时，掩码会让每一步都无合法候选——在请求边界给出明确
    // 400，而不是生成到一半 500（诚实降级，不做静默放宽）。
    if let Some(ref schema) = schema {
        let tokenizer_ref = tokenizer;
        let text_of = |id: usize| -> Option<String> {
            let id = i32::try_from(id).ok()?;
            tokenizer_ref.token_text(id).map(str::to_string)
        };
        schema::check_vocab_support(schema.root(), &text_of, tokenizer.vocab_size())
            .map_err(ApiError::bad_request)?;
    }
    Ok(GenerationRequest {
        prompt_ids,
        max_tokens,
        sampler: SamplerConfig {
            temperature,
            top_k: common.top_k.unwrap_or(0),
            top_p,
            repeat_penalty,
        },
        seed: common.seed,
        stop_texts,
        schema,
    })
}

/// 按 `model` 字段精确路由；`id@digest` 版本钉死不匹配时返回 400。
fn check_model_name(state: &ServerState, model: Option<&str>) -> Result<(), ApiError> {
    let Some(requested) = model else {
        return Ok(());
    };
    let registry = state.registry_snapshot();
    if registry.iter().any(|entry| entry.id == requested) {
        return Ok(());
    }
    if let Some((id, digest)) = requested.split_once('@') {
        if let Some(entry) = registry.iter().find(|entry| entry.id == id) {
            return if entry.digest == digest {
                Ok(())
            } else {
                Err(ApiError::bad_request("model digest mismatch"))
            };
        }
    }
    Err(ApiError::not_found(format!(
        "model {requested:?} is not registered"
    )))
}

/// 显式回退标记：GGUF 没有 tokenizer.chat_template 时使用 ChatML。
fn render_chat_prompt(state: &ServerState, messages: &[ChatMessage]) -> Result<String, ApiError> {
    let tokenizer = state.engine.tokenizer.as_ref();
    let context = template::TemplateContext {
        messages: messages
            .iter()
            .map(|message| template::TemplateMessage {
                role: message.role.clone(),
                content: message.content.clone(),
            })
            .collect(),
        bos_token: tokenizer
            .and_then(|t| t.bos_id())
            .and_then(|id| tokenizer.and_then(|t| t.token_text(id)))
            .unwrap_or("")
            .to_string(),
        eos_token: tokenizer
            .and_then(|t| t.eos_id())
            .and_then(|id| tokenizer.and_then(|t| t.token_text(id)))
            .unwrap_or("")
            .to_string(),
        add_generation_prompt: true,
    };
    match tokenizer.and_then(|t| t.chat_template()) {
        Some(source) => template::render_template(source, &context).map_err(ApiError::bad_request),
        None => Ok(template::chatml_fallback(&context)),
    }
}

/// 每请求一条观测：绝不记录 prompt/response/logits 正文。
#[derive(Clone)]
struct RequestObservation {
    request_id: String,
    trace_id: String,
    key_hash: String,
    tenant: String,
    model: String,
    endpoint: &'static str,
    status: u16,
    error_type: Option<&'static str>,
    rejection_reason: Option<&'static str>,
    prompt_tokens: usize,
    completion_tokens: usize,
    duration: f64,
    ttft: f64,
    tpot: f64,
    prefill_tps: f64,
    decode_tps: f64,
    queue_wait: f64,
    batch_size: usize,
    cache_hit: bool,
    created: Instant,
}

impl RequestObservation {
    fn new(context: &AuthContext, state: &ServerState, endpoint: &'static str) -> Self {
        Self {
            request_id: context.request_id.clone(),
            trace_id: context.trace_id.clone(),
            key_hash: context.key_hash.clone(),
            tenant: context.tenant.clone(),
            model: state.engine.model_name.clone(),
            endpoint,
            status: 200,
            error_type: None,
            rejection_reason: None,
            prompt_tokens: 0,
            completion_tokens: 0,
            duration: 0.0,
            ttft: 0.0,
            tpot: 0.0,
            prefill_tps: 0.0,
            decode_tps: 0.0,
            queue_wait: 0.0,
            batch_size: 1,
            cache_hit: false,
            created: Instant::now(),
        }
    }

    fn with_error(mut self, error: &ApiError) -> Self {
        self.status = error.status;
        self.error_type = Some(error.kind);
        self.rejection_reason = match error.status {
            429 => Some("queue_or_quota"),
            503 => Some("kv_budget"),
            504 => Some("timeout"),
            _ => None,
        };
        self.duration = self.created.elapsed().as_secs_f64();
        self
    }

    fn with_generation(
        mut self,
        state: &ServerState,
        completion_tokens: usize,
        ttft: Duration,
        total: Duration,
        cache_hit: bool,
    ) -> Self {
        self.completion_tokens = completion_tokens;
        self.duration = self.created.elapsed().as_secs_f64().max(f64::EPSILON);
        self.ttft = ttft.as_secs_f64();
        self.tpot = if completion_tokens > 0 {
            total.as_secs_f64() / completion_tokens as f64
        } else {
            0.0
        };
        self.prefill_tps = if self.ttft > 0.0 {
            self.prompt_tokens as f64 / self.ttft
        } else {
            0.0
        };
        self.decode_tps = if total.as_secs_f64() > 0.0 {
            completion_tokens as f64 / total.as_secs_f64()
        } else {
            0.0
        };
        self.batch_size = state.kv_pool().stats().active_leases.max(1);
        self.cache_hit = cache_hit;
        self
    }

    fn record(&self, state: &ServerState) {
        state.metrics.record_request(
            self.status,
            &self.model,
            self.endpoint,
            self.duration,
            self.ttft,
            self.tpot,
            self.prefill_tps,
            self.decode_tps,
            self.queue_wait,
            self.batch_size,
            self.cache_hit,
            (self.prompt_tokens + self.completion_tokens) as u64,
            self.error_type,
            self.rejection_reason,
        );
        state
            .metrics
            .bind_request(&self.request_id, self.status, &self.model, self.endpoint);
        let line = json!({
            "ts": unix_now(),
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "key_hash": self.key_hash,
            "tenant": self.tenant,
            "model": self.model,
            "endpoint": self.endpoint,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "duration_ms": (self.duration * 1000.0).round() as u64,
            "status": self.status,
            "error_type": self.error_type,
            "batch_size": self.batch_size,
            "cache_hit": self.cache_hit,
        });
        eprintln!("{line}");
        let _ = state.audit_log().write(&audit::AuditEvent {
            request_id: &self.request_id,
            trace_id: &self.trace_id,
            key_hash: &self.key_hash,
            tenant: &self.tenant,
            model: &self.model,
            endpoint: self.endpoint,
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            duration_ms: (self.duration * 1000.0).round() as u64,
            status: self.status,
            error_type: self.error_type,
            quota_decision: if self.status == 429 {
                "rejected"
            } else {
                "allowed"
            },
            cache_hit: self.cache_hit,
        });
    }
}

async fn completions(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    body: Result<Json<CompletionsBody>, JsonRejection>,
) -> Response {
    let body = match parse_json(body) {
        Ok(body) => body,
        Err(error) => return error_response(&error),
    };
    dispatch(
        &state,
        &headers,
        body.common,
        &body.prompt,
        false,
        "/v1/completions",
    )
    .await
}

async fn chat_completions(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    body: Result<Json<ChatBody>, JsonRejection>,
) -> Response {
    let body = match parse_json(body) {
        Ok(body) => body,
        Err(error) => return error_response(&error),
    };
    let prompt_text = match render_chat_prompt(&state, &body.messages) {
        Ok(text) => text,
        Err(error) => {
            let observation = dummy_observation(&state, &headers, "/v1/chat/completions", &error);
            observation.record(&state);
            return error_response(&error);
        }
    };
    dispatch(
        &state,
        &headers,
        body.common,
        &PromptInput::Text(prompt_text),
        true,
        "/v1/chat/completions",
    )
    .await
}

fn dummy_observation(
    state: &ServerState,
    headers: &HeaderMap,
    endpoint: &'static str,
    error: &ApiError,
) -> RequestObservation {
    let context = check_auth(state, headers, Role::Inference).ok();
    let pairs = header_pairs(headers);
    let request_id = context
        .as_ref()
        .map(|c| c.request_id.clone())
        .unwrap_or_else(|| format!("req-{}", uuid_like()));
    let trace_id = context
        .as_ref()
        .map(|c| c.trace_id.clone())
        .or_else(|| auth::parse_traceparent(&pairs))
        .unwrap_or_else(|| format!("{:032x}", uuid_like_u128()));
    let observation = RequestObservation {
        request_id,
        trace_id,
        key_hash: context
            .as_ref()
            .map(|c| c.key_hash.clone())
            .unwrap_or_default(),
        tenant: context
            .as_ref()
            .map(|c| c.tenant.clone())
            .unwrap_or_default(),
        model: state.engine.model_name.clone(),
        endpoint,
        status: 200,
        error_type: None,
        rejection_reason: None,
        prompt_tokens: 0,
        completion_tokens: 0,
        duration: 0.0,
        ttft: 0.0,
        tpot: 0.0,
        prefill_tps: 0.0,
        decode_tps: 0.0,
        queue_wait: 0.0,
        batch_size: 1,
        cache_hit: false,
        created: Instant::now(),
    };
    observation.with_error(error)
}

/// 公共请求校验/入队路径：先鉴权，再按字段三态与 registry 路由处理。
async fn dispatch(
    state: &Arc<ServerState>,
    headers: &HeaderMap,
    common: CommonFields,
    prompt: &PromptInput,
    chat: bool,
    endpoint: &'static str,
) -> Response {
    let context = match check_auth(state, headers, Role::Inference) {
        Ok(context) => context,
        Err(error) => {
            let observation = dummy_observation(state, headers, endpoint, &error);
            observation.record(state);
            return error_response(&error);
        }
    };
    let mut observation = RequestObservation::new(&context, state, endpoint);
    let early_error = if chat
        && match prompt {
            PromptInput::Text(text) => text.is_empty(),
            PromptInput::Array(items) => items.is_empty(),
        } {
        Some(ApiError::bad_request("messages must not be empty"))
    } else {
        reject_unsupported_fields(&common)
            .err()
            .or_else(|| check_model_name(state, common.model.as_deref()).err())
    };
    if let Some(error) = early_error {
        observation = observation.with_error(&error);
        observation.record(state);
        return error_response(&error);
    }
    let prompt_text = match prompt {
        PromptInput::Text(text) => text.clone(),
        PromptInput::Array(items) => {
            if items.len() != 1 {
                let error = ApiError::bad_request("prompt array must contain exactly one string");
                observation = observation.with_error(&error);
                observation.record(state);
                return error_response(&error);
            }
            items[0].clone()
        }
    };
    let request = match build_request(state, &prompt_text, &common) {
        Ok(request) => request,
        Err(error) => {
            observation = observation.with_error(&error);
            observation.record(state);
            return error_response(&error);
        }
    };
    observation.prompt_tokens = request.prompt_ids.len();
    let include_usage = common
        .stream_options
        .as_ref()
        .and_then(|options| options.include_usage)
        .unwrap_or(false);
    run_session(
        state.clone(),
        request,
        common.stream.unwrap_or(false),
        chat,
        include_usage,
        common,
        observation,
    )
    .await
}

async fn run_session(
    state: Arc<ServerState>,
    request: GenerationRequest,
    stream: bool,
    chat: bool,
    include_usage: bool,
    common: CommonFields,
    mut observation: RequestObservation,
) -> Response {
    let key_id = observation.key_hash.clone();
    let estimated = request.prompt_ids.len() + request.max_tokens;
    let permit = match state
        .qos()
        .acquire(&key_id, estimated, unix_now().saturating_mul(1000))
    {
        Ok(permit) => permit,
        Err(reason) => {
            let error = ApiError::queue_full().with_rejection(&reason);
            observation = observation.with_error(&error);
            observation.record(&state);
            return error_response(&error);
        }
    };
    if state.queue_depth.load(AtomicOrdering::Relaxed) >= state.limits.queue_capacity {
        let error = ApiError::queue_full();
        observation = observation.with_error(&error);
        observation.record(&state);
        permit.release(&state.qos(), error.status);
        return error_response(&error);
    }
    let prompt_tokens = request.prompt_ids.len();
    let cancelled = Arc::new(AtomicBool::new(false));
    let (events_tx, events_rx) = tokio::sync::mpsc::unbounded_channel::<SessionEvent>();
    let session = Session {
        request,
        events: events_tx,
        cancelled: cancelled.clone(),
    };
    if state.submit.send(session).is_err() {
        let error = ApiError::internal("scheduler is not running");
        observation = observation.with_error(&error);
        observation.record(&state);
        permit.release(&state.qos(), error.status);
        return error_response(&error);
    }
    let guard = CancelGuard::new(cancelled, events_rx);
    let timeout = state.limits.request_timeout;
    let completion_id = format!("cmpl-{}", uuid_like());
    let created = unix_now();
    let model_name = state.engine.model_name.clone();
    let execution = ExecutionContext {
        state: state.clone(),
        completion_id,
        created,
        model_name,
        chat,
        include_usage,
        prompt_tokens,
        metadata: response_metadata(&state, common.seed, &common.user),
        observation,
        permit,
        timeout,
    };
    if stream {
        stream_response(guard, execution)
    } else {
        collect_response(guard, common, execution).await
    }
}

/// 一次请求的执行上下文；把流式/非流式共享参数收束为单一值对象。
struct ExecutionContext {
    state: Arc<ServerState>,
    completion_id: String,
    created: u64,
    model_name: String,
    chat: bool,
    include_usage: bool,
    prompt_tokens: usize,
    metadata: Value,
    observation: RequestObservation,
    permit: qos::QosPermit,
    timeout: Duration,
}

fn usage_body(prompt_tokens: usize, completion_tokens: usize) -> Value {
    json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    })
}

/// 非流式：聚合全部 delta，返回 usage 与可复现元数据，记录一次观测。
async fn collect_response(
    mut guard: CancelGuard,
    common: CommonFields,
    execution: ExecutionContext,
) -> Response {
    let ExecutionContext {
        state,
        completion_id,
        created,
        chat,
        prompt_tokens,
        mut observation,
        permit,
        timeout,
        ..
    } = execution;
    let mut text = String::new();
    loop {
        let event = match tokio::time::timeout(timeout, guard.recv()).await {
            Ok(Some(event)) => event,
            Ok(None) => {
                let error = ApiError::internal("generation channel closed before completion");
                observation = observation.with_error(&error);
                observation.record(&state);
                permit.release(&state.qos(), error.status);
                return error_response(&error);
            }
            Err(_) => {
                let error = ApiError {
                    status: 504,
                    kind: "timeout",
                    message: "request exceeded the server timeout".into(),
                    unsupported: Vec::new(),
                };
                observation = observation.with_error(&error);
                observation.record(&state);
                permit.release(&state.qos(), error.status);
                return error_response(&error);
            }
        };
        match event {
            SessionEvent::Delta(delta) => text.push_str(&delta),
            SessionEvent::Done {
                tokens,
                reason,
                ttft,
                total,
                cache_hit,
            } => {
                observation =
                    observation.with_generation(&state, tokens.len(), ttft, total, cache_hit);
                observation.record(&state);
                permit.release(&state.qos(), observation.status);
                let choices = if chat {
                    json!([{
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": reason,
                    }])
                } else {
                    json!([{
                        "text": text,
                        "index": 0,
                        "finish_reason": reason,
                    }])
                };
                let mut body = json!({
                    "id": completion_id,
                    "object": if chat { "chat.completion" } else { "text_completion" },
                    "created": created,
                    "model": state.engine.model_name,
                    "choices": choices,
                    "usage": usage_body(prompt_tokens, tokens.len()),
                });
                body["qllama"] = response_metadata(&state, common.seed, &common.user);
                return Json(body).into_response();
            }
            SessionEvent::Failed(failure) => {
                let error = failure_to_error(&failure);
                observation = observation.with_error(&error);
                observation.record(&state);
                permit.release(&state.qos(), error.status);
                return error_response(&error);
            }
        }
    }
}

fn failure_to_error(failure: &Failure) -> ApiError {
    match failure {
        Failure::QueueFull => ApiError::queue_full(),
        Failure::Overloaded(message) => ApiError::overloaded(message.clone()),
        Failure::Cancelled => ApiError {
            status: 499,
            kind: "client_closed_request",
            message: "request cancelled".into(),
            unsupported: Vec::new(),
        },
        Failure::Internal(message) => ApiError::internal(message.clone()),
    }
}

/// 事件流包装：持有 Receiver 与取消标志，drop 时置位。
struct CancelGuard {
    cancelled: Arc<AtomicBool>,
    receiver: tokio::sync::mpsc::UnboundedReceiver<SessionEvent>,
}

impl CancelGuard {
    fn new(
        cancelled: Arc<AtomicBool>,
        receiver: tokio::sync::mpsc::UnboundedReceiver<SessionEvent>,
    ) -> Self {
        Self {
            cancelled,
            receiver,
        }
    }
    async fn recv(&mut self) -> Option<SessionEvent> {
        self.receiver.recv().await
    }
}

impl Drop for CancelGuard {
    fn drop(&mut self) {
        self.cancelled.store(true, AtomicOrdering::Relaxed);
    }
}

/// SSE 事件流：每个 delta 一帧，结束帧 finish_reason，最后 [DONE]。
struct GenerationStream {
    guard: Option<CancelGuard>,
    state: Arc<ServerState>,
    completion_id: String,
    created: u64,
    model_name: String,
    chat: bool,
    include_usage: bool,
    prompt_tokens: usize,
    metadata: Value,
    observation: Option<RequestObservation>,
    permit: Option<qos::QosPermit>,
    pending: std::collections::VecDeque<Event>,
    done_sent: bool,
    timeout: Duration,
    sleep: Option<Pin<Box<tokio::time::Sleep>>>,
}

impl GenerationStream {
    fn chunk(&self, text: &str, finish_reason: Option<&str>) -> Event {
        let choice = if self.chat {
            json!({
                "index": 0,
                "delta": {"content": text},
                "finish_reason": finish_reason,
            })
        } else {
            json!({
                "text": text,
                "index": 0,
                "finish_reason": finish_reason,
            })
        };
        let mut body = json!({
            "id": self.completion_id,
            "object": if self.chat { "chat.completion.chunk" } else { "text_completion.chunk" },
            "created": self.created,
            "model": self.model_name,
            "choices": [choice],
        });
        if self.include_usage {
            body["usage"] = Value::Null;
        }
        Event::default().data(body.to_string())
    }

    fn usage_chunk(&self, completion_tokens: usize) -> Event {
        let mut body = json!({
            "id": self.completion_id,
            "object": if self.chat { "chat.completion.chunk" } else { "text_completion.chunk" },
            "created": self.created,
            "model": self.model_name,
            "choices": [],
            "usage": usage_body(self.prompt_tokens, completion_tokens),
        });
        body["qllama"] = self.metadata.clone();
        Event::default().data(body.to_string())
    }

    fn finish_observation(&mut self, error: Option<&ApiError>) {
        if let Some(mut observation) = self.observation.take() {
            if let Some(error) = error {
                observation = observation.with_error(error);
            }
            observation.record(&self.state);
            if let Some(permit) = self.permit.take() {
                permit.release(&self.state.qos(), observation.status);
            }
        }
    }
}

impl Stream for GenerationStream {
    type Item = Result<Event, std::convert::Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        if let Some(event) = this.pending.pop_front() {
            return Poll::Ready(Some(Ok(event)));
        }
        if this.done_sent {
            this.guard.take();
            return Poll::Ready(None);
        }
        let guard = this.guard.as_mut().expect("guard lives until stream ends");
        let sleep = this
            .sleep
            .get_or_insert_with(|| Box::pin(tokio::time::sleep(this.timeout)));
        match Pin::new(&mut guard.receiver).poll_recv(cx) {
            Poll::Ready(Some(event)) => {
                this.sleep = None;
                match event {
                    SessionEvent::Delta(delta) => Poll::Ready(Some(Ok(this.chunk(&delta, None)))),
                    SessionEvent::Done {
                        tokens,
                        reason,
                        ttft,
                        total,
                        cache_hit,
                    } => {
                        let completion_tokens = tokens.len();
                        if let Some(ref mut observation) = this.observation {
                            let finished = std::mem::replace(
                                observation,
                                RequestObservation {
                                    status: 0,
                                    ..observation.clone()
                                },
                            )
                            .with_generation(
                                &this.state,
                                completion_tokens,
                                ttft,
                                total,
                                cache_hit,
                            );
                            *observation = finished;
                        }
                        // OpenAI 语义：usage 必须出现在终止帧（空 choices）中，
                        // 因此顺序为 finish_reason → usage → [DONE]。
                        this.pending.push_back(this.chunk("", Some(reason)));
                        if this.include_usage {
                            this.pending.push_back(this.usage_chunk(completion_tokens));
                        }
                        this.pending.push_back(Event::default().data("[DONE]"));
                        this.done_sent = true;
                        this.finish_observation(None);
                        Poll::Ready(Some(Ok(this.pending.pop_front().unwrap())))
                    }
                    SessionEvent::Failed(failure) => {
                        let error = failure_to_error(&failure);
                        let message = error.message.clone();
                        let kind = error.kind;
                        this.pending.push_back(Event::default().data(
                            json!({"error": {"message": message, "type": kind}}).to_string(),
                        ));
                        this.pending.push_back(Event::default().data("[DONE]"));
                        this.done_sent = true;
                        this.finish_observation(Some(&error));
                        Poll::Ready(Some(Ok(this.pending.pop_front().unwrap())))
                    }
                }
            }
            Poll::Ready(None) => {
                this.pending.push_back(Event::default().data(
                    json!({"error": {"message": "generation channel closed", "type": "internal_error"}})
                        .to_string(),
                ));
                this.pending.push_back(Event::default().data("[DONE]"));
                this.done_sent = true;
                let error = ApiError::internal("generation channel closed");
                this.finish_observation(Some(&error));
                Poll::Ready(Some(Ok(this.pending.pop_front().unwrap())))
            }
            Poll::Pending => match sleep.as_mut().poll(cx) {
                Poll::Ready(()) => {
                    this.pending.push_back(Event::default().data(
                        json!({"error": {"message": "request exceeded the server timeout", "type": "timeout"}})
                            .to_string(),
                    ));
                    this.pending.push_back(Event::default().data("[DONE]"));
                    this.done_sent = true;
                    let error = ApiError {
                        status: 504,
                        kind: "timeout",
                        message: "request exceeded the server timeout".into(),
                        unsupported: Vec::new(),
                    };
                    this.finish_observation(Some(&error));
                    Poll::Ready(Some(Ok(this.pending.pop_front().unwrap())))
                }
                Poll::Pending => Poll::Pending,
            },
        }
    }
}

fn stream_response(guard: CancelGuard, execution: ExecutionContext) -> Response {
    let ExecutionContext {
        state,
        completion_id,
        created,
        model_name,
        chat,
        include_usage,
        prompt_tokens,
        metadata,
        observation,
        permit,
        timeout,
    } = execution;
    let stream = GenerationStream {
        guard: Some(guard),
        state,
        completion_id,
        created,
        model_name,
        chat,
        include_usage,
        prompt_tokens,
        metadata,
        observation: Some(observation),
        permit: Some(permit),
        pending: std::collections::VecDeque::new(),
        done_sent: false,
        timeout,
        sleep: None,
    };
    Sse::new(stream)
        .keep_alive(
            KeepAlive::new()
                .interval(Duration::from_secs(15))
                .text("keepalive"),
        )
        .into_response()
}

fn uuid_like() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(0);
    format!("{:x}", nanos ^ ((std::process::id() as u64) << 16))
}

fn uuid_like_u128() -> u128 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    now ^ ((std::process::id() as u128) << 64)
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
