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
use std::time::Duration;

use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::Future;
use futures::Stream;
use serde::Deserialize;
use serde_json::json;

use crate::sampler::SamplerConfig;
use crate::server::{ApiError, Failure, GenerationRequest, ServerState, Session, SessionEvent};
use crate::tokenizer::EncodeOptions;

/// OpenAI 错误体；429 附带 Retry-After。
fn error_response(error: &ApiError) -> Response {
    let body = json!({
        "error": {
            "message": error.message,
            "type": error.kind,
            "code": error.status,
        }
    });
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
fn check_auth(state: &ServerState, headers: &HeaderMap) -> Result<(), ApiError> {
    let Some(expected) = &state.api_key else {
        return Ok(());
    };
    let actual = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    match actual {
        Some(key) if constant_time_eq(key.as_bytes(), expected.as_bytes()) => Ok(()),
        _ => Err(ApiError {
            status: 401,
            kind: "authentication_error",
            message: "missing or invalid API key".into(),
        }),
    }
}

/// 常量时间比较，避免 API key 上的时序侧信道。
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

pub fn router(state: Arc<ServerState>) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/metrics", get(metrics))
        .route("/v1/models", get(list_models))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(DefaultBodyLimit::max(state.limits.max_request_bytes))
        .with_state(state)
}

async fn healthz() -> &'static str {
    "ok"
}

async fn metrics(State(state): State<Arc<ServerState>>) -> String {
    let uptime = state.started.elapsed().as_secs_f64();
    let requests = state.metrics.requests_total.load(AtomicOrdering::Relaxed);
    let tokens = state.metrics.tokens_generated.load(AtomicOrdering::Relaxed);
    let queued = state.queue_depth.load(AtomicOrdering::Relaxed);
    format!(
        "# HELP qllama_requests_total Total generation requests\n\
         # TYPE qllama_requests_total counter\n\
         qllama_requests_total {requests}\n\
         # HELP qllama_tokens_generated_total Total tokens generated\n\
         # TYPE qllama_tokens_generated_total counter\n\
         qllama_tokens_generated_total {tokens}\n\
         # HELP qllama_queue_depth Requests waiting in queue\n\
         # TYPE qllama_queue_depth gauge\n\
         qllama_queue_depth {queued}\n\
         # HELP qllama_uptime_seconds Server uptime\n\
         # TYPE qllama_uptime_seconds gauge\n\
         qllama_uptime_seconds {uptime:.1}\n"
    )
}

async fn list_models(State(state): State<Arc<ServerState>>) -> Response {
    let body = json!({
        "object": "list",
        "data": [{
            "id": state.engine.model_name,
            "object": "model",
            "owned_by": "qllama",
        }]
    });
    Json(body).into_response()
}

/// OpenAI completions 请求体。未列出的字段（tools、logprobs 等，见 ADR 0004）
/// 由 serde 默认忽略——OpenAI 客户端会发送它们，忽略即预期行为。
#[derive(Deserialize, Debug)]
struct CompletionsBody {
    model: Option<String>,
    prompt: PromptInput,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    stop: Option<StopInput>,
    stream: Option<bool>,
    seed: Option<u64>,
    n: Option<usize>,
}

#[derive(Deserialize, Debug)]
#[serde(untagged)]
enum PromptInput {
    Text(String),
    Array(Vec<String>),
}

#[derive(Deserialize, Debug)]
#[serde(untagged)]
enum StopInput {
    One(String),
    Many(Vec<String>),
}

/// chat messages 数组的最小模型：role + content。
#[derive(Deserialize, Debug)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize, Debug)]
struct ChatBody {
    model: Option<String>,
    messages: Vec<ChatMessage>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    stop: Option<StopInput>,
    stream: Option<bool>,
    seed: Option<u64>,
    n: Option<usize>,
}

/// 把 OpenAI 参数翻译成内部请求；校验失败返回 400。
/// OpenAI 请求体里与生成参数相关的字段，统一打包后传入 `build_request`。
struct RequestParams<'a> {
    state: &'a ServerState,
    prompt_text: &'a str,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    stop: Vec<String>,
    seed: Option<u64>,
    n: Option<usize>,
}

fn build_request(params: RequestParams<'_>) -> Result<GenerationRequest, ApiError> {
    let RequestParams {
        state,
        prompt_text,
        max_tokens,
        temperature,
        top_p,
        top_k,
        stop,
        seed,
        n,
    } = params;
    let Some(tokenizer) = state.engine.tokenizer.as_ref() else {
        return Err(ApiError::bad_request(
            "model has no tokenizer metadata; text prompts are not supported by this model",
        ));
    };
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

    if let Some(n) = n.filter(|&n| n != 1) {
        return Err(ApiError::bad_request(format!(
            "n={n} is not supported by this serial engine; submit separate requests"
        )));
    }
    if stop.len() > 4 {
        return Err(ApiError::bad_request(
            "at most 4 stop sequences are supported",
        ));
    }
    if stop.iter().any(String::is_empty) {
        return Err(ApiError::bad_request("stop sequences must not be empty"));
    }

    let temperature = temperature.unwrap_or(1.0);
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(ApiError::bad_request(
            "temperature must be a finite non-negative number",
        ));
    }
    let top_p = top_p.unwrap_or(1.0);
    if !top_p.is_finite() || top_p <= 0.0 {
        return Err(ApiError::bad_request(
            "top_p must be a finite positive number",
        ));
    }

    let max_tokens = max_tokens
        .unwrap_or(16)
        .min(state.engine.ctx_size - prompt_ids.len());
    if max_tokens == 0 {
        return Err(ApiError::bad_request(
            "prompt fills the context; no room to generate",
        ));
    }

    Ok(GenerationRequest {
        prompt_ids,
        max_tokens,
        sampler: SamplerConfig {
            temperature,
            top_k: top_k.unwrap_or(0),
            top_p,
            // OpenAI API 没有 repeat penalty 字段；默认不惩罚。
            repeat_penalty: 1.0,
        },
        seed,
        stop_texts: stop,
    })
}

fn extract_stop(stop: Option<StopInput>) -> Vec<String> {
    match stop {
        Some(StopInput::One(s)) => vec![s],
        Some(StopInput::Many(v)) => v,
        None => Vec::new(),
    }
}

/// 模型名一致性检查：客户端带 `model` 时必须与加载的模型一致（前缀匹配
/// 兼容客户端把文件名写成无扩展名形式的情况）。
fn check_model_name(state: &ServerState, model: Option<&str>) -> Result<(), ApiError> {
    if let Some(requested) = model {
        if requested != state.engine.model_name && !state.engine.model_name.starts_with(requested) {
            return Err(ApiError::not_found(format!(
                "model {requested:?} is not loaded; this server serves {:?}",
                state.engine.model_name
            )));
        }
    }
    Ok(())
}

async fn completions(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    Json(body): Json<CompletionsBody>,
) -> Response {
    if let Err(error) = check_auth(&state, &headers) {
        return error_response(&error);
    }
    if let Err(error) = check_model_name(&state, body.model.as_deref()) {
        return error_response(&error);
    }
    let prompt_text = match &body.prompt {
        PromptInput::Text(text) => text.clone(),
        PromptInput::Array(items) => {
            if items.len() != 1 {
                return error_response(&ApiError::bad_request(
                    "prompt array with multiple entries is not supported; pass a single string",
                ));
            }
            items[0].clone()
        }
    };
    let request = match build_request(RequestParams {
        state: &state,
        prompt_text: &prompt_text,
        max_tokens: body.max_tokens,
        temperature: body.temperature,
        top_p: body.top_p,
        top_k: body.top_k,
        stop: extract_stop(body.stop),
        seed: body.seed,
        n: body.n,
    }) {
        Ok(request) => request,
        Err(error) => return error_response(&error),
    };
    run_session(state, request, body.stream.unwrap_or(false)).await
}

/// 降级 chat 模板（ADR 0004）：通用 ChatML 拼法。没有 jinja 解析器，
/// 模型官方模板差异在 docs/api.md 如实标注。
fn render_chat_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for message in messages {
        prompt.push_str("<|im_start|>");
        prompt.push_str(&message.role);
        prompt.push('\n');
        prompt.push_str(&message.content);
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

async fn chat_completions(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    Json(body): Json<ChatBody>,
) -> Response {
    if let Err(error) = check_auth(&state, &headers) {
        return error_response(&error);
    }
    if let Err(error) = check_model_name(&state, body.model.as_deref()) {
        return error_response(&error);
    }
    if body.messages.is_empty() {
        return error_response(&ApiError::bad_request("messages must not be empty"));
    }
    let prompt_text = render_chat_prompt(&body.messages);
    let request = match build_request(RequestParams {
        state: &state,
        prompt_text: &prompt_text,
        max_tokens: body.max_tokens,
        temperature: body.temperature,
        top_p: body.top_p,
        top_k: body.top_k,
        stop: extract_stop(body.stop),
        seed: body.seed,
        n: body.n,
    }) {
        Ok(request) => request,
        Err(error) => return error_response(&error),
    };
    run_session(state, request, body.stream.unwrap_or(false)).await
}

/// 公共执行路径：背压检查 → 入队 → 等事件 → 非流式聚合或 SSE 流式。
///
/// 超时覆盖排队 + 推理全链（ADR 0004）。
async fn run_session(
    state: Arc<ServerState>,
    request: GenerationRequest,
    stream: bool,
) -> Response {
    // 背压：队列满直接 429，请求根本不入队。
    if state.queue_depth.load(AtomicOrdering::Relaxed) >= state.limits.queue_capacity {
        return error_response(&ApiError::queue_full());
    }
    state
        .metrics
        .requests_total
        .fetch_add(1, AtomicOrdering::Relaxed);
    let prompt_tokens = request.prompt_ids.len();

    let cancelled = Arc::new(AtomicBool::new(false));
    let (events_tx, events_rx) = tokio::sync::mpsc::unbounded_channel::<SessionEvent>();
    let session = Session {
        request,
        events: events_tx,
        cancelled: cancelled.clone(),
    };
    if state.submit.send(session).is_err() {
        return error_response(&ApiError::internal("scheduler is not running"));
    }

    // CancelGuard 在 handler/stream 被 drop 时置位取消，调度器下一个
    // decode step 停止；实际 <=2s 取决于单步耗时（见 Phase 3 验证）。
    let guard = CancelGuard::new(cancelled, events_rx);
    let timeout = state.limits.request_timeout;
    if stream {
        let completion_id = format!("cmpl-{}", uuid_like());
        let created = unix_now();
        let model_name = state.engine.model_name.clone();
        stream_response(
            state,
            guard,
            timeout,
            completion_id,
            created,
            model_name,
            false,
        )
    } else {
        collect_response(state, guard, timeout, prompt_tokens).await
    }
}

/// 非流式：聚合全部 delta，返回单个 JSON。
async fn collect_response(
    state: Arc<ServerState>,
    mut guard: CancelGuard,
    timeout: Duration,
    prompt_tokens: usize,
) -> Response {
    let mut text = String::new();
    let (completion_tokens, reason) = loop {
        let event = match tokio::time::timeout(timeout, guard.recv()).await {
            Ok(Some(event)) => event,
            Ok(None) => {
                return error_response(&ApiError::internal(
                    "generation channel closed before completion",
                ));
            }
            Err(_) => {
                return error_response(&ApiError {
                    status: 504,
                    kind: "timeout",
                    message: "request exceeded the server timeout".into(),
                });
            }
        };
        match event {
            SessionEvent::Delta(delta) => text.push_str(&delta),
            SessionEvent::Done {
                tokens, reason: r, ..
            } => {
                break (tokens.len(), r.to_string());
            }
            SessionEvent::Failed(failure) => return failure_response(failure),
        }
    };
    state
        .metrics
        .tokens_generated
        .fetch_add(completion_tokens as u64, AtomicOrdering::Relaxed);
    let body = json!({
        "id": format!("cmpl-{}", uuid_like()),
        "object": "text_completion",
        "created": unix_now(),
        "model": state.engine.model_name,
        "choices": [{
            "text": text,
            "index": 0,
            "finish_reason": reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    });
    Json(body).into_response()
}

fn failure_response(failure: Failure) -> Response {
    match failure {
        Failure::QueueFull => error_response(&ApiError::queue_full()),
        Failure::Overloaded(message) => error_response(&ApiError::overloaded(message)),
        Failure::Cancelled => error_response(&ApiError {
            status: 499,
            kind: "client_closed_request",
            message: "request cancelled".into(),
        }),
        Failure::Internal(message) => error_response(&ApiError::internal(message)),
    }
}

/// 事件流包装：持有 Receiver 与取消标志，drop 时置位。
///
/// 同时实现 `Stream` 后，SSE 响应体被客户端断开（hyper 会 drop body）
/// 时 Guard 随之 drop，取消立即生效，无需轮询。
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

/// SSE 事件流：把 SessionEvent 转成 OpenAI chunk，末尾追加 [DONE]。
struct GenerationStream {
    guard: Option<CancelGuard>,
    state: Arc<ServerState>,
    completion_id: String,
    created: u64,
    model_name: String,
    /// 终止帧发送后置 true，下一轮只发 [DONE]
    terminal_sent: bool,
    done_sent: bool,
    timeout: Duration,
    /// 每次 poll 挂起的超时 future；用 Box::pin 复用，不重建。
    sleep: Option<Pin<Box<tokio::time::Sleep>>>,
}

impl GenerationStream {
    fn chunk(&self, text: &str, finish_reason: Option<&str>) -> Event {
        Event::default().data(
            json!({
                "id": self.completion_id,
                "object": "text_completion.chunk",
                "created": self.created,
                "model": self.model_name,
                "choices": [{
                    "text": text,
                    "index": 0,
                    "finish_reason": finish_reason,
                }]
            })
            .to_string(),
        )
    }
}

impl Stream for GenerationStream {
    type Item = Result<Event, std::convert::Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        if this.done_sent {
            return Poll::Ready(None);
        }
        if this.terminal_sent {
            this.done_sent = true;
            return Poll::Ready(Some(Ok(Event::default().data("[DONE]"))));
        }
        // 每轮 poll 收一个事件；超时用 tokio Sleep 保活。
        let guard = this.guard.as_mut().expect("guard lives until stream ends");
        let sleep = this
            .sleep
            .get_or_insert_with(|| Box::pin(tokio::time::sleep(this.timeout)));
        match Pin::new(&mut guard.receiver).poll_recv(cx) {
            Poll::Ready(Some(event)) => {
                this.sleep = None;
                let sse = match event {
                    SessionEvent::Delta(delta) => this.chunk(&delta, None),
                    SessionEvent::Done { tokens, reason, .. } => {
                        this.state
                            .metrics
                            .tokens_generated
                            .fetch_add(tokens.len() as u64, AtomicOrdering::Relaxed);
                        this.terminal_sent = true;
                        this.chunk("", Some(reason))
                    }
                    SessionEvent::Failed(failure) => {
                        let message = match failure {
                            Failure::QueueFull => "queue is full".to_string(),
                            Failure::Overloaded(message) | Failure::Internal(message) => message,
                            Failure::Cancelled => "request cancelled".to_string(),
                        };
                        this.terminal_sent = true;
                        Event::default().data(
                            json!({"error": {"message": message, "type": "server_error"}})
                                .to_string(),
                        )
                    }
                };
                Poll::Ready(Some(Ok(sse)))
            }
            Poll::Ready(None) => {
                this.terminal_sent = true;
                Poll::Ready(Some(Ok(Event::default().data(
                    json!({"error": {"message": "generation channel closed", "type": "internal_error"}})
                        .to_string(),
                ))))
            }
            Poll::Pending => match sleep.as_mut().poll(cx) {
                Poll::Ready(()) => {
                    this.terminal_sent = true;
                    Poll::Ready(Some(Ok(Event::default().data(
                        json!({"error": {"message": "request exceeded the server timeout", "type": "timeout"}})
                            .to_string(),
                    ))))
                }
                Poll::Pending => Poll::Pending,
            },
        }
    }
}

/// SSE 流式：每个 delta 一个 data 事件，结束帧带 finish_reason，最后 [DONE]。
/// 15s keepalive 心跳（Phase 2.3）。chat 端点 object 差异见 `chat` 参数。
fn stream_response(
    state: Arc<ServerState>,
    guard: CancelGuard,
    timeout: Duration,
    completion_id: String,
    created: u64,
    model_name: String,
    _chat: bool,
) -> Response {
    let stream = GenerationStream {
        guard: Some(guard),
        state,
        completion_id,
        created,
        model_name,
        terminal_sent: false,
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

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
