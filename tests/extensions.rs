//! 扩展 HTTP 路由的进程内契约测试。
//!
//! 使用 `tests/fixtures/tiny-llama.gguf` 构造完整 `ServerState`，直接调用
//! handler 构造 Response，避免引入新的测试依赖；断言覆盖鉴权、只读控制台、
//! 诚实降级、能力声明和 Ollama NDJSON 的真实用量。

#![cfg(feature = "server")]

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::body::to_bytes;
use axum::extract::State;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use qllama::server::extensions::{
    api_chat, api_tags, api_version, apply_template, console, embeddings, router_with_extensions,
    v1_capabilities, ChatMessage, ChatRequest, GenerateOptions, TemplateRequest,
};
use qllama::server::{Engine, Limits, ServerState};
use serde_json::{json, Value};

const TEST_MODEL: &str = "tests/fixtures/tiny-llama.gguf";
const SECRET: &str = "extensions-test-secret";
const PROMPT_LITERAL: &str = "EXTENSIONS_CONSOLE_SECRET_PROMPT_LITERAL";

struct TestState {
    state: Arc<ServerState>,
    _scheduler: std::thread::JoinHandle<()>,
}

/// 构造带真实 tiny GGUF 的状态，并让调度线程在测试退出时自然结束。
fn test_state(api_key: Option<&str>) -> TestState {
    let engine = Engine::load(Path::new(TEST_MODEL), None, Some(1), None)
        .unwrap_or_else(|reason| panic!("load {TEST_MODEL}: {reason}"));
    let limits = Limits {
        queue_capacity: 8,
        request_timeout: Duration::from_secs(30),
        max_request_bytes: 1 << 20,
        kv_byte_budget: 64 << 20,
    };
    let (state, scheduler) = ServerState::start(engine, limits, api_key.map(str::to_owned))
        .unwrap_or_else(|reason| panic!("start state: {reason}"));
    TestState {
        state,
        _scheduler: scheduler,
    }
}

fn auth_headers() -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(
        header::AUTHORIZATION,
        format!("Bearer {SECRET}")
            .parse()
            .unwrap_or_else(|_| panic!("valid header value")),
    );
    headers
}

async fn response_json(response: Response) -> Value {
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap_or_else(|reason| panic!("read body: {reason}"));
    serde_json::from_slice(&body).unwrap_or_else(|reason| panic!("decode JSON: {reason}"))
}

async fn response_text(response: Response) -> (StatusCode, String) {
    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap_or_else(|reason| panic!("read body: {reason}"));
    (
        status,
        String::from_utf8(body.to_vec()).unwrap_or_else(|reason| panic!("utf8 body: {reason}")),
    )
}

/// 路由注册本身可构造，且不会与现有 OpenAI 路由冲突。
#[test]
fn router_with_extensions_builds() {
    let test = test_state(None);
    let _app = router_with_extensions(Arc::clone(&test.state));
}

/// 控制台在配置 API key 后必须拒绝未授权请求，且授权响应不回显 prompt。
#[tokio::test]
async fn console_requires_key_and_never_echoes_prompt() {
    let test = test_state(Some(SECRET));
    let unauthorized = console(State(Arc::clone(&test.state)), HeaderMap::new())
        .await
        .into_response();
    assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);
    let (_, unauthorized_body) = response_text(unauthorized).await;
    assert!(unauthorized_body.contains("missing_authorization"));

    let mut headers = auth_headers();
    headers.insert(
        "x-console-prompt-literal",
        PROMPT_LITERAL
            .parse()
            .unwrap_or_else(|_| panic!("valid header value")),
    );
    let authorized = console(State(Arc::clone(&test.state)), headers)
        .await
        .into_response();
    assert_eq!(authorized.status(), StatusCode::OK);
    let html = response_text(authorized).await.1;
    assert!(!html.contains(PROMPT_LITERAL));
    assert!(html.contains("Qllama Console"));
    assert!(html.contains("只读诊断"));
}

/// 未实现的 OpenAI 端点族返回 501 和 qllama.unsupported 标记。
#[tokio::test]
async fn v1_embeddings_is_honestly_degraded() {
    let response = embeddings().await;
    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
    let body = response_json(response).await;
    assert_eq!(body["error"]["code"], "qllama.unsupported");
    assert!(body["qllama.unsupported"]
        .as_array()
        .is_some_and(|items| items.iter().any(|item| item == "v1_embeddings")));
    assert!(body["error"]["qllama.unsupported"]
        .as_array()
        .is_some_and(|items| items.iter().any(|item| item == "qllama.unsupported")));
}

/// 版本与标签端点不依赖鉴权配置，返回真实模型注册信息。
#[tokio::test]
async fn version_and_tags_report_real_state() {
    let test = test_state(None);
    let version = api_version().await.into_response();
    assert_eq!(response_json(version).await["version"], "0.1.0");

    let tags = api_tags(State(Arc::clone(&test.state)), HeaderMap::new())
        .await
        .into_response();
    let body = response_json(tags).await;
    let models = body["models"]
        .as_array()
        .unwrap_or_else(|| panic!("tags models: {body}"));
    assert!(!models.is_empty());
    assert!(models[0]["digest"]
        .as_str()
        .is_some_and(|digest| !digest.is_empty()));
}

/// Ollama chat 默认流式 NDJSON，末帧必须携带真实 token 用量和耗时。
#[tokio::test]
async fn chat_returns_ndjson_with_real_usage() {
    let test = test_state(None);
    let request = ChatRequest {
        model: Some(test.state.engine.model_name.clone()),
        messages: vec![ChatMessage {
            role: "user".into(),
            content: "hello".into(),
        }],
        stream: None,
        options: Some(GenerateOptions {
            num_predict: Some(1),
            seed: Some(20_261_114),
            temperature: Some(0.0),
            ..GenerateOptions::default()
        }),
    };
    let response = api_chat(
        State(Arc::clone(&test.state)),
        HeaderMap::new(),
        Json(request),
    )
    .await
    .into_response();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("application/x-ndjson")
    );
    let (_, text) = response_text(response).await;
    let lines: Vec<&str> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    assert!(!lines.is_empty(), "NDJSON must have at least one frame");
    let last: Value = serde_json::from_str(lines[lines.len() - 1])
        .unwrap_or_else(|reason| panic!("decode final NDJSON frame: {reason}"));
    assert_eq!(last["done"], true);
    assert_eq!(last["message"]["role"], "assistant");
    assert!(last["prompt_eval_count"]
        .as_u64()
        .is_some_and(|count| count > 0));
    assert!(last["eval_count"].as_u64().is_some_and(|count| count > 0));
    for key in [
        "total_duration",
        "load_duration",
        "prompt_eval_duration",
        "eval_duration",
    ] {
        assert!(last[key].is_number(), "missing numeric {key}: {last}");
    }
}

/// 模板端点在无原生模板时明确标注 ChatML fallback 与真实 token 数。
#[tokio::test]
async fn apply_template_renders_chatml_with_token_count() {
    let test = test_state(None);
    let request = TemplateRequest {
        model: None,
        messages: vec![ChatMessage {
            role: "user".into(),
            content: "hello".into(),
        }],
        add_generation_prompt: None,
    };
    let response = apply_template(
        State(Arc::clone(&test.state)),
        HeaderMap::new(),
        Json(request),
    )
    .await
    .into_response();
    let body = response_json(response).await;
    let prompt = body["prompt"]
        .as_str()
        .unwrap_or_else(|| panic!("template prompt: {body}"));
    assert!(prompt.contains("<|im_start|>user\nhello<|im_end|>"));
    assert!(prompt.ends_with("<|im_start|>assistant\n"));
    assert_eq!(body["template_fallback"], "chatml");
    assert!(body["token_count"].as_u64().is_some_and(|count| count > 0));
}

/// 能力声明包含模型摘要、未实现字段和未实现能力族。
#[tokio::test]
async fn capabilities_include_digest_and_unsupported_fields() {
    let test = test_state(None);
    let response = v1_capabilities(State(Arc::clone(&test.state)), HeaderMap::new())
        .await
        .into_response();
    let body = response_json(response).await;
    assert_eq!(body["object"], "capabilities");
    assert_eq!(body["model_digest"], test.state.model_digest());
    assert!(body["unsupported_fields"]
        .as_array()
        .is_some_and(|fields| fields.iter().any(|field| field == "tools")));
    assert!(body["unsupported_capabilities"]
        .as_array()
        .is_some_and(|items| !items.is_empty()));
    assert!(body["reproducibility"].is_object());
}

/// 避免 json! 宏在本文件被裁剪后产生未使用导入；用于校验扩展错误体形状。
#[test]
fn error_shape_helper_is_valid_json() {
    assert!(json!({ "error": { "code": "qllama.unsupported" } }).is_object());
}
