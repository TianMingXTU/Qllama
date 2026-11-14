//! Ollama 原生 API 的响应形状构造。
//!
//! 本模块只负责序列化协议字段：用量、计时和 token 数全部由调用方传入实测值，
//! 不生成时间、摘要或模型内容，也不记录提示词与响应正文。

use serde_json::{json, Value};

/// 不受支持的 Ollama 原生端点。
///
/// 路径中的 `:digest` 保留 Ollama 文档使用的占位符写法。
pub const UNSUPPORTED_ENDPOINTS: &[&str] = &[
    "/api/embed",
    "/api/pull",
    "/api/push",
    "/api/create",
    "/api/copy",
    "/api/delete",
    "/api/blobs/:digest",
    "/api/web_search",
    "/api/web_fetch",
];

/// 一次生成请求的真实用量与计时，单位为次数和纳秒。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OllamaUsage {
    pub prompt_eval_count: usize,
    pub eval_count: usize,
    pub total_duration_ns: u64,
    pub load_duration_ns: u64,
    pub prompt_eval_duration_ns: u64,
    pub eval_duration_ns: u64,
}

/// 一个本地模型的静态描述，供 tags、ps 和 show 复用。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OllamaTagsEntry {
    pub name: String,
    pub digest: String,
    pub size: u64,
    pub format: &'static str,
    pub family: &'static str,
    pub parameter_size: String,
    pub quantization_level: String,
}

/// 构造 `/api/version` 响应。
pub fn version_document(version: &str) -> Value {
    json!({ "version": version })
}

/// 将条目的公共描述字段构造成 Ollama `details` 对象。
fn details(entry: &OllamaTagsEntry) -> Value {
    json!({
        "format": entry.format,
        "family": entry.family,
        "parameter_size": &entry.parameter_size,
        "quantization_level": &entry.quantization_level,
    })
}

/// 将条目构造成列表中的模型对象；调用方没有提供真实时间，故不填时间字段。
fn model_document(entry: &OllamaTagsEntry) -> Value {
    json!({
        "name": &entry.name,
        "model": &entry.name,
        "digest": &entry.digest,
        "size": entry.size,
        "details": details(entry),
    })
}

/// 构造 `/api/tags` 响应。
pub fn tags_document(models: &[OllamaTagsEntry]) -> Value {
    json!({ "models": models.iter().map(model_document).collect::<Vec<_>>() })
}

/// 构造 `/api/ps` 响应。
///
/// 缺少真实的过期时间和显存位置，不伪造 `expires_at` 或 `size_vram`。
pub fn ps_document(models: &[OllamaTagsEntry]) -> Value {
    tags_document(models)
}

/// 构造 `/api/show` 响应，仅包含静态模型描述和调用方提供的模板、上下文长度。
pub fn show_document(entry: &OllamaTagsEntry, template: &str, context_length: usize) -> Value {
    json!({
        "modelfile": "",
        "parameters": "",
        "template": template,
        "details": details(entry),
        "model_info": {
            "general.architecture": "llama",
            "llama.context_length": context_length,
        },
    })
}

/// 构造流式响应中两类 chunk 共用的字段。
fn chunk_document(
    model: &str,
    created_at: &str,
    done: bool,
    done_reason: Option<&str>,
    usage: &OllamaUsage,
) -> Value {
    json!({
        "model": model,
        "created_at": created_at,
        "done": done,
        "done_reason": done_reason,
        "prompt_eval_count": usage.prompt_eval_count,
        "eval_count": usage.eval_count,
        "total_duration": usage.total_duration_ns,
        "load_duration": usage.load_duration_ns,
        "prompt_eval_duration": usage.prompt_eval_duration_ns,
        "eval_duration": usage.eval_duration_ns,
    })
}

/// 构造 `/api/generate` 的流式 chunk；`response` 是本次增量文本。
pub fn generate_chunk(
    model: &str,
    created_at: &str,
    response: &str,
    done: bool,
    done_reason: Option<&str>,
    usage: &OllamaUsage,
) -> Value {
    let mut document = chunk_document(model, created_at, done, done_reason, usage);
    document["response"] = json!(response);
    document
}

/// 构造 `/api/chat` 的流式 chunk；`content` 是本次增量文本。
pub fn chat_chunk(
    model: &str,
    created_at: &str,
    role: &str,
    content: &str,
    done: bool,
    done_reason: Option<&str>,
    usage: &OllamaUsage,
) -> Value {
    let mut document = chunk_document(model, created_at, done, done_reason, usage);
    document["message"] = json!({ "role": role, "content": content });
    document
}

/// 构造不支持能力的 501 错误，只携带能力名，不携带请求内容。
pub fn unsupported_error(capability: &str) -> Value {
    json!({
        "error": {
            "message": "unsupported Ollama endpoint or capability",
            "type": "not_implemented_error",
            "code": 501,
        },
        "qllama": { "unsupported": [capability] },
    })
}
