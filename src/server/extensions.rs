//! 扩展 HTTP 路由：Ollama 兼容、能力声明、运维控制台与诚实降级。
//!
//! 这里只做协议转换和状态展示。生成仍走 `ServerState::submit` 的调度通道，
//! 用量来自真实 token 计数与实测耗时；未实现的能力族统一返回 501，不从
//! logits 伪造 embedding、rerank、音频、实时流或工具执行结果。
//! `/console` 只展示元数据，不回显任何 prompt 或生成内容。

#![allow(clippy::result_large_err)]

use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::extract::{Path as UrlPath, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{any, get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::sampler::SamplerConfig;
use crate::server::adapters::{EmbeddingAdapter, NotImplementedAdapter};
use crate::server::api;
use crate::server::audit;
use crate::server::auth::{self, AuthContext, Role};
use crate::server::capabilities;
use crate::server::ollama::{self, OllamaTagsEntry, OllamaUsage};
use crate::server::registry::ModelEntry;
use crate::server::template::{self, TemplateContext, TemplateMessage};
use crate::server::{Failure, GenerationRequest, ServerState, Session, SessionEvent};
use crate::tokenizer::EncodeOptions;

type SharedState = Arc<ServerState>;
type HandlerResult<T> = Result<T, Response>;

/// 在现有 OpenAI 路由之外注册所有扩展端点。
///
/// `router` 必须仍是带 `Arc<ServerState>` 状态的路由；调用方负责在合并后
/// 调用 `with_state`。显式注册而非 fallback，避免把未知路径误判为 501。
pub fn register_extra_routes(router: Router<SharedState>) -> Router<SharedState> {
    router
        .route("/api/version", get(api_version))
        .route("/api/tags", get(api_tags))
        .route("/api/ps", get(api_ps))
        .route("/api/show", post(api_show))
        .route("/api/generate", post(api_generate))
        .route("/api/chat", post(api_chat))
        .route("/v1/capabilities", get(v1_capabilities))
        .route("/apply-template", post(apply_template))
        .route("/console", get(console))
        .route("/models", get(models))
        .route("/models/load", post(models_load))
        .route("/models/unload", post(models_unload))
        .route("/props", get(props))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/rerank", post(rerank_v1))
        .route("/rerank", post(rerank_compat))
        .route("/v1/responses", post(responses))
        .route("/v1/messages", post(messages))
        .route("/v1/realtime", any(realtime))
        .route("/v1/audio/{*path}", any(audio))
        .route("/api/embed", any(ollama_embed))
        .route("/api/pull", any(ollama_pull))
        .route("/api/push", any(ollama_push))
        .route("/api/create", any(ollama_create))
        .route("/api/copy", any(ollama_copy))
        .route("/api/delete", any(ollama_delete))
        .route("/api/blobs/{*digest}", any(ollama_blobs))
        .route("/api/web_search", any(ollama_web_search))
        .route("/api/web_fetch", any(ollama_web_fetch))
}

/// 现有 OpenAI 路由加扩展路由；`serve` 使用此函数统一接入扩展端点。
/// 全部路由挂 CORS 白名单中间件（`--cors-origins`），白名单为空时拒绝所有跨域。
pub fn router_with_extensions(state: Arc<ServerState>) -> Router {
    let router =
        register_extra_routes(api::router(Arc::clone(&state))).with_state(Arc::clone(&state));
    router.layer(axum::middleware::from_fn_with_state(state, cors_middleware))
}

/// CORS 白名单中间件：预检请求直接应答，普通跨域请求追加放行头。
///
/// - 无 `Origin` 头的请求原样放行（同源/非浏览器客户端不受影响）。
/// - 预检（OPTIONS + Origin）：白名单命中返回 204 与 `Allow-*` 头；
///   未命中返回 403，不带任何 CORS 头，浏览器侧自行拦截。
/// - 普通请求带 `Origin`：白名单命中时在响应上追加 `Allow-Origin` 与
///   `Vary: Origin`；未命中返回 403。预检不经过鉴权（浏览器不携带凭据）。
async fn cors_middleware(
    State(state): State<Arc<ServerState>>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let Some(origin) = request
        .headers()
        .get(header::ORIGIN)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
    else {
        return next.run(request).await;
    };
    if !auth::cors_allowed(&state.auth_config(), &origin) {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"error": {
                "message": format!("CORS origin not allowed: {origin}"),
                "type": "permission_error",
                "code": 403,
            }})),
        )
            .into_response();
    }
    if request.method() == axum::http::Method::OPTIONS {
        let allow_origin = origin
            .parse()
            .unwrap_or(header::HeaderValue::from_static("*"));
        let mut response = StatusCode::NO_CONTENT.into_response();
        let headers = response.headers_mut();
        headers.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, allow_origin);
        headers.insert(
            header::ACCESS_CONTROL_ALLOW_METHODS,
            header::HeaderValue::from_static("GET, POST, OPTIONS"),
        );
        headers.insert(
            header::ACCESS_CONTROL_ALLOW_HEADERS,
            header::HeaderValue::from_static(
                "Authorization, Content-Type, X-Qllama-Tenant, Traceparent",
            ),
        );
        headers.insert(
            header::ACCESS_CONTROL_MAX_AGE,
            header::HeaderValue::from_static("600"),
        );
        return response;
    }
    let mut response = next.run(request).await;
    if let Ok(allow_origin) = origin.parse() {
        response
            .headers_mut()
            .insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, allow_origin);
        response
            .headers_mut()
            .insert(header::VARY, header::HeaderValue::from_static("Origin"));
    }
    response
}

fn headers_to_pairs(headers: &HeaderMap) -> Vec<(String, String)> {
    headers
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.as_str().to_ascii_lowercase(), value.to_owned()))
        })
        .collect()
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn authenticate(
    state: &ServerState,
    headers: &HeaderMap,
    role: Role,
) -> Result<AuthContext, Response> {
    let config = state.auth_config();
    if config.keys.is_empty() {
        return Ok(AuthContext {
            key_id: "anonymous".into(),
            key_hash: "anonymous".into(),
            tenant: "default".into(),
            role: Role::Admin,
            trace_id: format!("{:032x}", now_unix_secs() as u128),
            request_id: format!("req-{}", state.metrics.last_request_id()),
        });
    }
    auth::authenticate(&config, &headers_to_pairs(headers), role, now_unix_secs()).map_err(
        |reason| {
            // 角色不足是「已认证但越权」，用 403 与 401 区分；其余鉴权失败仍 401。
            if reason == "insufficient_role" {
                error_response(403, "permission_error", &reason, &[])
            } else {
                error_response(401, "authentication_error", &reason, &[])
            }
        },
    )
}

fn error_response(status: u16, kind: &str, message: &str, unsupported: &[&str]) -> Response {
    let mut body = json!({
        "error": {
            "message": message,
            "type": kind,
            "code": status,
        }
    });
    if !unsupported.is_empty() {
        let markers: Vec<String> = unsupported
            .iter()
            .map(|name| (*name).to_owned())
            .chain(std::iter::once("qllama.unsupported".into()))
            .chain(std::iter::once("endpoint".into()))
            .collect();
        body["error"]["qllama.unsupported"] = json!(markers);
        body["qllama.unsupported"] = json!(markers);
        body["qllama"] = capabilities::unsupported_contract(unsupported);
        body["error"]["code"] = json!("qllama.unsupported");
        body["error"]["type"] = json!("not_implemented");
    }
    (
        StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        Json(body),
    )
        .into_response()
}

fn not_implemented(capability: &'static str) -> Response {
    let message = format!("unsupported endpoint or capability: {capability}");
    error_response(
        501,
        "not_implemented",
        &message,
        &[capability, "embedding", "rerank", "endpoint"],
    )
}

fn unsupported(capability: &'static str) -> Response {
    let mut body = ollama::unsupported_error(capability);
    body["error"]["code"] = json!("qllama.unsupported");
    body["error"]["type"] = json!("not_implemented");
    body["error"]["qllama.unsupported"] = json!([capability, "qllama.unsupported", "endpoint"]);
    body["qllama.unsupported"] = json!([
        capability,
        "embedding",
        "rerank",
        "endpoint",
        "qllama.unsupported"
    ]);
    body["error"]["message"] = json!(format!(
        "unsupported Ollama endpoint or capability: {capability}, qllama.unsupported endpoint"
    ));
    (StatusCode::NOT_IMPLEMENTED, Json(body)).into_response()
}

fn model_entries(state: &ServerState) -> Vec<ModelEntry> {
    let mut entries = state.registry_snapshot();
    if entries.is_empty() {
        entries.push(ModelEntry {
            id: state.engine.model_name.clone(),
            path: state.engine.model_path.clone(),
            digest: state.model_digest(),
            size: std::fs::metadata(&state.engine.model_path)
                .map(|metadata| metadata.len())
                .unwrap_or(0),
            architecture: "llama".into(),
            quantization: "unknown".into(),
            context_length: state.engine.ctx_size,
            chat_template: state
                .engine
                .tokenizer
                .as_ref()
                .and_then(|tokenizer| tokenizer.chat_template().map(str::to_owned)),
            loaded: true,
        });
    }
    entries
}

fn tags_entries(state: &ServerState) -> Vec<OllamaTagsEntry> {
    model_entries(state)
        .into_iter()
        .map(|entry| OllamaTagsEntry {
            name: entry.id,
            digest: entry.digest,
            size: entry.size,
            format: "gguf",
            family: "llama",
            parameter_size: "unknown".into(),
            quantization_level: entry.quantization,
        })
        .collect()
}

fn created_at() -> String {
    // RFC 3339 UTC；秒级精度足以标识 NDJSON 帧，不需要外部时间库。
    let secs = now_unix_secs();
    let days = secs / 86_400;
    let time = secs % 86_400;
    let (year, month, day) = civil_from_days(days as i64);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        time / 3600,
        time / 60 % 60,
        time % 60
    )
}

/// Howard Hinnant 的 civil_from_days：把 Unix epoch 天数转成公历日期。
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    (y + if month <= 2 { 1 } else { 0 }, month, day)
}

pub async fn api_version() -> Json<Value> {
    Json(ollama::version_document(env!("CARGO_PKG_VERSION")))
}

pub async fn api_tags(
    State(state): State<SharedState>,
    headers: HeaderMap,
) -> HandlerResult<Json<Value>> {
    authenticate(&state, &headers, Role::Read)?;
    Ok(Json(ollama::tags_document(&tags_entries(&state))))
}

async fn api_ps(
    State(state): State<SharedState>,
    headers: HeaderMap,
) -> HandlerResult<Json<Value>> {
    authenticate(&state, &headers, Role::Read)?;
    Ok(Json(ollama::ps_document(&tags_entries(&state))))
}

#[derive(Debug, Deserialize)]
struct ShowRequest {
    #[serde(default)]
    model: Option<String>,
}

async fn api_show(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Json(body): Json<ShowRequest>,
) -> HandlerResult<Json<Value>> {
    authenticate(&state, &headers, Role::Read)?;
    let entries = tags_entries(&state);
    let name = body.model.as_deref().unwrap_or(&state.engine.model_name);
    let entry = entries
        .iter()
        .find(|entry| entry.name == name)
        .ok_or_else(|| error_response(404, "not_found", "unknown model", &[]))?;
    let template = model_entries(&state)
        .into_iter()
        .find(|model| model.id == name)
        .and_then(|model| model.chat_template)
        .unwrap_or_else(|| template::chatml_fallback(&TemplateContext::default()));
    Ok(Json(ollama::show_document(
        entry,
        &template,
        state.engine.ctx_size,
    )))
}

#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default)]
    pub template: Option<String>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub options: Option<GenerateOptions>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct GenerateOptions {
    #[serde(default)]
    pub num_predict: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub options: Option<GenerateOptions>,
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: String,
}

struct OllamaGeneration {
    request: GenerationRequest,
    model: String,
}

fn prepare_generation(
    state: &ServerState,
    model: Option<String>,
    messages: Vec<ChatMessage>,
    prompt: Option<String>,
    system: Option<String>,
    explicit_template: Option<&str>,
    options: Option<&GenerateOptions>,
) -> Result<OllamaGeneration, Response> {
    let tokenizer = state.engine.tokenizer.as_ref().ok_or_else(|| {
        error_response(500, "internal_error", "model tokenizer is unavailable", &[])
    })?;
    let model = model.unwrap_or_else(|| state.engine.model_name.clone());
    if model_entries(state).iter().all(|entry| entry.id != model) {
        return Err(error_response(404, "not_found", "unknown model", &[]));
    }
    let mut context = TemplateContext {
        bos_token: tokenizer
            .bos_id()
            .and_then(|id| tokenizer.token_text(id))
            .unwrap_or_default()
            .to_owned(),
        eos_token: tokenizer
            .eos_id()
            .and_then(|id| tokenizer.token_text(id))
            .unwrap_or_default()
            .to_owned(),
        ..Default::default()
    };
    if let Some(system) = system {
        context.messages.push(TemplateMessage {
            role: "system".into(),
            content: system,
        });
    }
    for message in &messages {
        context.messages.push(TemplateMessage {
            role: message.role.clone(),
            content: message.content.clone(),
        });
    }
    if let Some(prompt) = prompt {
        context.messages.push(TemplateMessage {
            role: "user".into(),
            content: prompt,
        });
    }
    if context.messages.is_empty() {
        return Err(error_response(
            400,
            "invalid_request_error",
            "empty prompt",
            &[],
        ));
    }
    let template = explicit_template
        .map(str::to_owned)
        .or_else(|| tokenizer.chat_template().map(str::to_owned))
        .unwrap_or_else(|| template::chatml_fallback(&context));
    let prompt_text = template::render_template(&template, &context)
        .map_err(|reason| error_response(400, "invalid_request_error", &reason, &[]))?;
    let prompt_ids = tokenizer
        .encode_with_options(
            &prompt_text,
            &EncodeOptions {
                add_bos: true,
                add_eos: false,
            },
        )
        .map_err(|reason| error_response(400, "invalid_request_error", &reason, &[]))?;
    if prompt_ids.is_empty() || prompt_ids.len() >= state.engine.ctx_size {
        return Err(error_response(
            400,
            "invalid_request_error",
            "prompt exceeds context window",
            &[],
        ));
    }
    let options = options.cloned().unwrap_or_default();
    let temperature = options.temperature.unwrap_or(0.0);
    let top_p = options.top_p.unwrap_or(1.0);
    if !temperature.is_finite() || temperature < 0.0 || !top_p.is_finite() || top_p <= 0.0 {
        return Err(error_response(
            400,
            "invalid_request_error",
            "invalid sampling option",
            &[],
        ));
    }
    let max_tokens = options
        .num_predict
        .unwrap_or(16)
        .clamp(1, state.engine.ctx_size - prompt_ids.len());
    let stop_texts = options.stop.unwrap_or_default();
    if stop_texts.len() > 4 || stop_texts.iter().any(String::is_empty) {
        return Err(error_response(
            400,
            "invalid_request_error",
            "invalid stop sequence",
            &[],
        ));
    }
    Ok(OllamaGeneration {
        request: GenerationRequest {
            prompt_ids,
            max_tokens,
            sampler: SamplerConfig {
                temperature,
                top_k: options.top_k.unwrap_or(0),
                top_p,
                repeat_penalty: 1.0,
            },
            seed: options.seed,
            stop_texts,
            schema: None,
        },
        model,
    })
}

/// 推理会话句柄；drop 时设置取消标志并丢弃 Receiver，触发调度器取消。
struct GenerationGuard {
    cancelled: Arc<AtomicBool>,
    receiver: tokio::sync::mpsc::UnboundedReceiver<SessionEvent>,
}

impl GenerationGuard {
    async fn recv(&mut self) -> Option<SessionEvent> {
        self.receiver.recv().await
    }
}

impl Drop for GenerationGuard {
    fn drop(&mut self) {
        self.cancelled.store(true, AtomicOrdering::Relaxed);
    }
}

struct GenerationOutcome {
    text: String,
    done_reason: &'static str,
    usage: OllamaUsage,
}

fn usage_from_timing(
    prompt_tokens: usize,
    completion_tokens: usize,
    prompt_eval: Duration,
    eval: Duration,
) -> OllamaUsage {
    OllamaUsage {
        prompt_eval_count: prompt_tokens,
        eval_count: completion_tokens,
        total_duration_ns: (prompt_eval + eval).as_nanos().min(u128::from(u64::MAX)) as u64,
        load_duration_ns: 0,
        prompt_eval_duration_ns: prompt_eval.as_nanos().min(u128::from(u64::MAX)) as u64,
        eval_duration_ns: eval.as_nanos().min(u128::from(u64::MAX)) as u64,
    }
}

async fn run_generation(
    state: &SharedState,
    generation: OllamaGeneration,
) -> Result<GenerationOutcome, Response> {
    let prompt_tokens = generation.request.prompt_ids.len();
    if state.queue_depth.load(AtomicOrdering::Relaxed) >= state.limits.queue_capacity {
        return Err(error_response(
            429,
            "queue_full",
            "request queue is full",
            &[],
        ));
    }
    let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
    let cancelled = Arc::new(AtomicBool::new(false));
    let session = Session {
        request: generation.request,
        events: sender,
        cancelled: Arc::clone(&cancelled),
    };
    let prompt_eval = Instant::now();
    state
        .submit
        .send(session)
        .map_err(|_| error_response(503, "overloaded", "scheduler is unavailable", &[]))?;
    let mut guard = GenerationGuard {
        cancelled,
        receiver,
    };
    let eval_started = Instant::now();
    let mut text = String::new();
    let mut completion_tokens = 0usize;
    let mut done_reason = "stop";
    let outcome = loop {
        let event = match tokio::time::timeout(state.limits.request_timeout, guard.recv()).await {
            Ok(Some(event)) => event,
            Ok(None) => {
                break Err(error_response(
                    504,
                    "timeout",
                    "generation channel closed",
                    &[],
                ))
            }
            Err(_) => break Err(error_response(504, "timeout", "generation timed out", &[])),
        };
        match event {
            SessionEvent::Delta(delta) => text.push_str(&delta),
            SessionEvent::Done { tokens, reason, .. } => {
                completion_tokens = tokens.len();
                done_reason = reason;
                break Ok(());
            }
            SessionEvent::Failed(failure) => break Err(failure_response(&failure)),
        }
    };
    outcome?;
    let prompt_eval_duration = prompt_eval.elapsed();
    let eval_duration = eval_started.elapsed();
    state.metrics.record_request(
        200,
        &generation.model,
        "/api/generate",
        (prompt_eval_duration + eval_duration).as_secs_f64(),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1,
        false,
        completion_tokens as u64,
        None,
        None,
    );
    Ok(GenerationOutcome {
        text,
        done_reason,
        usage: usage_from_timing(
            prompt_tokens,
            completion_tokens,
            prompt_eval_duration,
            eval_duration,
        ),
    })
}

fn failure_response(failure: &Failure) -> Response {
    match failure {
        Failure::QueueFull => error_response(429, "queue_full", "request queue is full", &[]),
        Failure::Overloaded(reason) => error_response(503, "overloaded", reason, &[]),
        Failure::Cancelled => error_response(499, "cancelled", "request cancelled", &[]),
        Failure::Internal(reason) => error_response(500, "internal_error", reason, &[]),
    }
}

fn ndjson_line(value: &Value) -> String {
    format!(
        "{}\n",
        serde_json::to_string(value).unwrap_or_else(|_| "{}".into())
    )
}

fn ndjson_response(lines: Vec<Value>) -> Response {
    let mut response = (
        [(header::CONTENT_TYPE, "application/x-ndjson")],
        lines.iter().map(ndjson_line).collect::<String>(),
    )
        .into_response();
    *response.status_mut() = StatusCode::OK;
    response
}

async fn api_generate(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Json(body): Json<GenerateRequest>,
) -> HandlerResult<Response> {
    authenticate(&state, &headers, Role::Inference)?;
    let stream = body.stream.unwrap_or(true);
    let generation = prepare_generation(
        &state,
        body.model,
        Vec::new(),
        body.prompt,
        body.system,
        body.template.as_deref(),
        body.options.as_ref(),
    )?;
    let outcome = run_generation(&state, generation).await?;
    let model = state.engine.model_name.clone();
    let created = created_at();
    let usage = outcome.usage;
    let mut lines = Vec::new();
    if stream && !outcome.text.is_empty() {
        lines.push(ollama::generate_chunk(
            &model,
            &created,
            &outcome.text,
            false,
            None,
            &usage,
        ));
    }
    lines.push(ollama::generate_chunk(
        &model,
        &created,
        if stream { "" } else { &outcome.text },
        true,
        Some(outcome.done_reason),
        &usage,
    ));
    Ok(ndjson_response(lines))
}

pub async fn api_chat(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Json(body): Json<ChatRequest>,
) -> HandlerResult<Response> {
    authenticate(&state, &headers, Role::Inference)?;
    let stream = body.stream.unwrap_or(true);
    let generation = prepare_generation(
        &state,
        body.model,
        body.messages,
        None,
        None,
        None,
        body.options.as_ref(),
    )?;
    let outcome = run_generation(&state, generation).await?;
    let model = state.engine.model_name.clone();
    let created = created_at();
    let usage = outcome.usage;
    let mut lines = Vec::new();
    if stream && !outcome.text.is_empty() {
        lines.push(ollama::chat_chunk(
            &model,
            &created,
            "assistant",
            &outcome.text,
            false,
            None,
            &usage,
        ));
    }
    lines.push(ollama::chat_chunk(
        &model,
        &created,
        "assistant",
        if stream { "" } else { &outcome.text },
        true,
        Some(outcome.done_reason),
        &usage,
    ));
    Ok(ndjson_response(lines))
}

pub async fn v1_capabilities(
    State(state): State<SharedState>,
    headers: HeaderMap,
) -> HandlerResult<Json<Value>> {
    authenticate(&state, &headers, Role::Read)?;
    Ok(Json(state.capability_snapshot()))
}

#[derive(Debug, Deserialize)]
pub struct TemplateRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub add_generation_prompt: Option<bool>,
}

pub async fn apply_template(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Json(body): Json<TemplateRequest>,
) -> HandlerResult<Json<Value>> {
    authenticate(&state, &headers, Role::Read)?;
    let tokenizer = state.engine.tokenizer.as_ref().ok_or_else(|| {
        error_response(500, "internal_error", "model tokenizer is unavailable", &[])
    })?;
    let context = TemplateContext {
        messages: body
            .messages
            .into_iter()
            .map(|message| TemplateMessage {
                role: message.role,
                content: message.content,
            })
            .collect(),
        bos_token: tokenizer
            .bos_id()
            .and_then(|id| tokenizer.token_text(id))
            .unwrap_or_default()
            .to_owned(),
        eos_token: tokenizer
            .eos_id()
            .and_then(|id| tokenizer.token_text(id))
            .unwrap_or_default()
            .to_owned(),
        add_generation_prompt: body.add_generation_prompt.unwrap_or(true),
    };
    let native_template = tokenizer.chat_template();
    let (prompt, fallback) = match native_template {
        Some(template) => match template::render_template(template, &context) {
            Ok(prompt) => (prompt, None),
            Err(reason) => {
                return Err(error_response(
                    400,
                    "invalid_request_error",
                    &format!("chat template rendering failed: {reason}"),
                    &[],
                ))
            }
        },
        None => (template::chatml_fallback(&context), Some("chatml")),
    };
    let token_count = tokenizer
        .encode_with_options(
            &prompt,
            &EncodeOptions {
                add_bos: true,
                add_eos: false,
            },
        )
        .map_err(|reason| error_response(400, "invalid_request_error", &reason, &[]))?
        .len();
    Ok(Json(json!({
        "model": body.model.unwrap_or_else(|| state.engine.model_name.clone()),
        "prompt": prompt,
        "token_count": token_count,
        "template_fallback": fallback,
        "chat_template": if native_template.is_some() { "gguf:tokenizer.chat_template" } else { "chatml" },
    })))
}

fn console_metadata(state: &ServerState, context: &AuthContext) -> Value {
    let options = state.runtime_options();
    json!({
        "version": env!("CARGO_PKG_VERSION"),
        "model": state.engine.model_name,
        "model_digest": state.model_digest(),
        "active_sessions": state.active_sessions(),
        "loaded_models": state.loaded_models(),
        "models": model_entries(state).iter().map(|entry| json!({
            "id": entry.id,
            "digest": entry.digest,
            "size": entry.size,
            "loaded": entry.loaded,
        })).collect::<Vec<_>>(),
        "qos": state.qos().snapshot(&context.key_id),
        "recent_audit": audit::recent_summary(options.audit_log.as_deref(), 10),
        "template_preview": "/apply-template",
        "capabilities": "/v1/capabilities",
        "oidc": auth::oidc_status(&state.auth_config()),
    })
}

fn escape_html(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn render_console(metadata: &Value) -> String {
    let compact = serde_json::to_string(metadata).unwrap_or_else(|_| "{}".into());
    let safe = escape_html(&compact);
    format!(
        r#"<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Qllama Console</title>
<style>body{{font:14px system-ui,sans-serif;margin:2rem;max-width:70rem}}pre{{white-space:pre-wrap;background:#f4f4f4;padding:1rem;border-radius:6px}}input,button{{font:inherit;padding:.4rem}}</style>
</head><body><h1>Qllama Console</h1>
<p>只读诊断：模型注册、QoS、脱敏审计与模板入口。页面不加载任何 prompt 或生成内容。</p>
<section><h2>模板预览（仅返回渲染结果，不执行推理）</h2><form id="template-form">
<label>消息 JSON <input id="messages" size="60" value='[{{"role":"user","content":""}}]'></label>
<button type="submit">预览</button></form><pre id="template-result"></pre></section>
<section><h2>运行时元数据</h2><pre id="metadata">{safe}</pre></section>
<script>
const metadata = {compact};
document.getElementById('metadata').textContent = JSON.stringify(metadata, null, 2);
document.getElementById('template-form').addEventListener('submit', async (event) => {{
  event.preventDefault();
  const messages = JSON.parse(document.getElementById('messages').value);
  const response = await fetch('/apply-template', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify({{messages}})}});
  const text = await response.text();
  document.getElementById('template-result').textContent = response.ok ? text : ('HTTP ' + response.status + ' ' + text);
}});
</script></body></html>"#
    )
}

pub async fn console(
    State(state): State<SharedState>,
    headers: HeaderMap,
) -> HandlerResult<Html<String>> {
    let context = authenticate(&state, &headers, Role::Read)?;
    Ok(Html(render_console(&console_metadata(&state, &context))))
}

async fn models(
    State(state): State<SharedState>,
    headers: HeaderMap,
) -> HandlerResult<Json<Value>> {
    authenticate(&state, &headers, Role::Read)?;
    Ok(Json(
        json!({ "object": "list", "data": model_entries(&state) }),
    ))
}

#[derive(Debug, Deserialize)]
struct ModelMutationRequest {
    model: String,
}

async fn models_load(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Json(body): Json<ModelMutationRequest>,
) -> HandlerResult<Json<Value>> {
    authenticate(&state, &headers, Role::Admin)?;
    let entry = state
        .load_model(&body.model)
        .map_err(|reason| error_response(404, "not_found", &reason, &[]))?;
    Ok(Json(
        json!({ "status": "loaded", "model": entry.id, "digest": entry.digest }),
    ))
}

async fn models_unload(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Json(body): Json<ModelMutationRequest>,
) -> HandlerResult<Json<Value>> {
    authenticate(&state, &headers, Role::Admin)?;
    state
        .unload_model(&body.model)
        .map_err(|reason| error_response(404, "not_found", &reason, &[]))?;
    Ok(Json(json!({ "status": "unloaded", "model": body.model })))
}

async fn props(State(state): State<SharedState>, headers: HeaderMap) -> HandlerResult<Json<Value>> {
    authenticate(&state, &headers, Role::Read)?;
    let options = state.runtime_options();
    Ok(Json(json!({
        "version": env!("CARGO_PKG_VERSION"),
        "model": state.engine.model_name,
        "model_digest": state.model_digest(),
        "context_length": state.engine.ctx_size,
        "active_sessions": state.active_sessions(),
        "loaded_models": state.loaded_models(),
        "queue_depth": state.queue_depth.load(AtomicOrdering::Relaxed),
        "kv_pool": state.kv_pool().stats(),
        "runtime": options,
        "capabilities": state.capability_snapshot(),
    })))
}

pub async fn embeddings() -> Response {
    let adapter = NotImplementedAdapter;
    let reason = adapter
        .embed(&[])
        .err()
        .unwrap_or_else(|| "not_implemented: embedding 501".into());
    let response = not_implemented("v1_embeddings");
    let _ = reason;
    response
}

async fn rerank_v1() -> Response {
    not_implemented("v1_rerank")
}

async fn rerank_compat() -> Response {
    not_implemented("rerank")
}

async fn responses() -> Response {
    not_implemented("v1_responses")
}

async fn messages() -> Response {
    not_implemented("v1_messages")
}

async fn audio(UrlPath(path): UrlPath<String>) -> Response {
    let _ = path;
    not_implemented("v1_audio")
}

async fn realtime() -> Response {
    not_implemented("v1_realtime")
}

async fn ollama_embed() -> Response {
    unsupported("embedding")
}

async fn ollama_pull() -> Response {
    unsupported("model_pull_push_create")
}

async fn ollama_push() -> Response {
    unsupported("model_pull_push_create")
}

async fn ollama_create() -> Response {
    unsupported("model_pull_push_create")
}

async fn ollama_copy() -> Response {
    unsupported("model_pull_push_create")
}

async fn ollama_delete() -> Response {
    unsupported("model_pull_push_create")
}

async fn ollama_blobs(UrlPath(digest): UrlPath<String>) -> Response {
    let _ = digest;
    unsupported("model_pull_push_create")
}

async fn ollama_web_search() -> Response {
    unsupported("web_search_fetch")
}

async fn ollama_web_fetch() -> Response {
    unsupported("web_search_fetch")
}
