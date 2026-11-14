//! 能力发现响应：把实际实现、显式降级与可复现配置冻结为单一合同。
//!
//! 本模块只输出启动期静态快照与预定义能力清单，不读取请求正文。

use serde_json::{json, Value};

/// 请求解析层显式覆盖的全部字段；未列出的字段一律以 400 拒绝。
pub const SUPPORTED_FIELDS: &[&str] = &[
    "model",
    "prompt",
    "messages",
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "stop",
    "stream",
    "seed",
    "stream_options",
    "frequency_penalty",
    "presence_penalty",
    "logprobs",
    "logit_bias",
    "response_format",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "echo",
    "best_of",
    "user",
    "n",
    "reasoning",
    "reasoning_effort",
    "repeat_penalty",
];

/// 可安全接受但不影响输出；响应元数据中可见，不是静默忽略。
pub const ACCEPTED_WITHOUT_EFFECT: &[&str] = &["user"];

/// 已解析但暂不实现的字段；请求中出现时返回 400 + `qllama.unsupported`。
pub const UNSUPPORTED_FIELDS: &[&str] = &[
    "frequency_penalty",
    "presence_penalty",
    "logprobs",
    "logit_bias",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "echo",
    "best_of",
    "reasoning",
    "reasoning_effort",
    "n_greater_than_1",
];

/// 未实现的端点族/能力；在支持矩阵和错误体里统一声明为下一批。
pub const UNSUPPORTED_CAPABILITIES: &[&str] = &[
    "v1_embeddings",
    "v1_rerank",
    "rerank",
    "v1_responses",
    "v1_messages",
    "v1_audio",
    "v1_realtime",
    "image_audio_video_content",
    "gpu_backend",
    "speculative_decoding",
    "lora",
    "additional_quantization_formats",
    "non_llama_architectures",
    "web_search_fetch",
    "model_pull_push_create",
    "oidc_sso",
    "builtin_tls",
    "automatic_tool_execution",
    "mcp_server_tools",
];

/// 已实现的 HTTP 端点；与 `api::router` 的路由表保持同步。
pub const IMPLEMENTED_ENDPOINTS: &[&str] = &[
    "/healthz",
    "/metrics",
    "/console",
    "/apply-template",
    "/v1/capabilities",
    "/v1/models",
    "/v1/completions",
    "/v1/chat/completions",
    "/models",
    "/props",
    "/api/version",
    "/api/tags",
    "/api/ps",
    "/api/show",
    "/api/generate",
    "/api/chat",
];

/// 返回明确 501 的第一方降级端点族。
pub const DEGRADED_ENDPOINTS: &[&str] = &[
    "/v1/embeddings",
    "/v1/rerank",
    "/rerank",
    "/v1/responses",
    "/v1/messages",
    "/v1/audio/*",
    "/v1/realtime",
    "/api/embed",
    "/api/pull",
    "/api/push",
    "/api/create",
    "/api/copy",
    "/api/delete",
    "/api/blobs/*",
    "/api/web_search",
    "/api/web_fetch",
];

/// HTTP 状态码全集；保持既有语义，501 仅用于明确的端点族降级。
pub const ERROR_CODES: &[u16] = &[400, 401, 404, 413, 429, 499, 500, 501, 503, 504];

/// 启动期冻结的能力与可复现快照。
#[derive(Clone, Debug)]
pub struct CapabilitySnapshot {
    pub version: &'static str,
    pub model_digest: String,
    pub quantization: String,
    pub kv_dtype: &'static str,
    pub chat_template: String,
    pub template_fallback: Option<&'static str>,
    pub concurrency_mode: &'static str,
    pub continuous_batching: bool,
    pub single_request_parallel: bool,
    pub prefix_cache: bool,
    pub config_snapshot: Value,
}

/// 返回能力发现文档；所有字段来自启动期快照，不读取请求正文。
pub fn capability_document(snapshot: &CapabilitySnapshot) -> Value {
    json!({
        "object": "capabilities",
        "version": snapshot.version,
        "positioning": "Rust-native CPU-only text generation inference engine",
        "model_digest": snapshot.model_digest,
        "quantization": snapshot.quantization,
        "kv_dtype": snapshot.kv_dtype,
        "chat_template": snapshot.chat_template,
        "template_fallback": snapshot.template_fallback,
        "concurrency_mode": snapshot.concurrency_mode,
        "continuous_batching": snapshot.continuous_batching,
        "single_request_parallel": snapshot.single_request_parallel,
        "prefix_cache": snapshot.prefix_cache,
        "supported_fields": SUPPORTED_FIELDS,
        "accepted_without_effect": ACCEPTED_WITHOUT_EFFECT,
        "unsupported_fields": UNSUPPORTED_FIELDS,
        "unsupported_capabilities": UNSUPPORTED_CAPABILITIES,
        "endpoints": {
            "implemented": IMPLEMENTED_ENDPOINTS,
            "degraded": DEGRADED_ENDPOINTS,
        },
        "error_codes": ERROR_CODES,
        "reproducibility": {
            "contract": "same seed, model digest, quantization, KV dtype, chat template and config snapshot produce identical token output",
            "seed": Value::Null,
            "config_snapshot": snapshot.config_snapshot,
        },
        "support_matrix": "docs/support-matrix.md",
    })
}

/// `qllama.unsupported` 的标准载荷；错误响应与能力发现共用同一列表。
///
/// 返回对象适合合并进 OpenAI `error` 对象：`{"qllama":{"unsupported":[...]}}`。
pub fn unsupported_contract(items: &[&str]) -> Value {
    json!({ "qllama": { "unsupported": items } })
}

#[cfg(all(test, feature = "server"))]
mod tests {
    use super::*;

    #[test]
    fn capability_document_contains_reproducibility_and_degradation() {
        let snapshot = CapabilitySnapshot {
            version: "0.1.0",
            model_digest: "sha256:abc".into(),
            quantization: "F32".into(),
            kv_dtype: "f32",
            chat_template: "chatml".into(),
            template_fallback: Some("chatml"),
            concurrency_mode: "continuous_step_parallel",
            continuous_batching: true,
            single_request_parallel: true,
            prefix_cache: false,
            config_snapshot: json!({"seed": 1, "threads": 2}),
        };
        let doc = capability_document(&snapshot);
        assert_eq!(doc["model_digest"], "sha256:abc");
        assert_eq!(doc["template_fallback"], "chatml");
        assert_eq!(doc["reproducibility"]["config_snapshot"]["threads"], 2);
        assert!(doc["unsupported_capabilities"]
            .as_array()
            .is_some_and(|v| v.iter().any(|item| item == "v1_embeddings")));
        assert_eq!(doc["accepted_without_effect"][0], "user");
    }

    #[test]
    fn unsupported_contract_has_openai_error_fragment_shape() {
        let contract = unsupported_contract(&["best_of"]);
        assert_eq!(contract["qllama"]["unsupported"][0], "best_of");
    }
}
