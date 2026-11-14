//! 审计日志：结构化事件的脱敏序列化与 JSONL 追加写入。
//!
//! 审计事件只允许携带元数据（请求 id、租户、token 数、状态码等），
//! 绝不允许携带 prompt/response/logits 等业务体。[`redact_event`] 作为
//! 最后防线，会把字符串字段中的控制字符替换为 U+FFFD 并截断到 128 字符，
//! 但调用方仍 MUST NOT 把 prompt 之类的内容塞进元数据字段。
//!
//! 与其他模块的关系：事件中的 `key_hash` 来自 `auth::hash_key` 的假名化
//! 哈希，`request_id`/`trace_id` 来自 `auth::AuthContext`。

use std::io::Write;
use std::path::Path;
use std::sync::Mutex;

/// 字符串字段的最大保留长度（按字符计），超出部分丢弃。
const MAX_FIELD_CHARS: usize = 128;

/// 审计事件允许落盘的字段白名单；读取侧也用它过滤，防止旧文件中的
/// 未知字段回流到控制台输出。
const AUDIT_KEYS: [&str; 13] = [
    "request_id",
    "trace_id",
    "key_hash",
    "tenant",
    "model",
    "endpoint",
    "prompt_tokens",
    "completion_tokens",
    "duration_ms",
    "status",
    "error_type",
    "quota_decision",
    "cache_hit",
];

/// 一条审计事件的元数据（全部为借用字段，由调用方在请求收尾时构造）。
///
/// 注意：各 `&str` 字段只允许放元数据；传入 prompt/response 等业务体
/// 属于调用方违约——脱敏只保证控制字符与超长内容不会原样落盘，不构成
/// 对违约的豁免。
#[derive(Clone, Copy, Debug)]
pub struct AuditEvent<'a> {
    /// 请求 id（来自鉴权上下文）。
    pub request_id: &'a str,
    /// 链路追踪 id（来自鉴权上下文）。
    pub trace_id: &'a str,
    /// 密钥假名化哈希（`auth::hash_key`），非明文。
    pub key_hash: &'a str,
    /// 租户标识。
    pub tenant: &'a str,
    /// 模型名。
    pub model: &'a str,
    /// 端点路径（如 `/v1/chat/completions`）。
    pub endpoint: &'a str,
    /// 输入 token 数。
    pub prompt_tokens: usize,
    /// 输出 token 数。
    pub completion_tokens: usize,
    /// 请求总耗时（毫秒）。
    pub duration_ms: u64,
    /// HTTP 状态码。
    pub status: u16,
    /// 错误类型标识（如 `invalid_credentials`），成功时为 `None`。
    pub error_type: Option<&'a str>,
    /// 限流判定结果（`auth`/`qos` 的错误标识或 `allowed`）。
    pub quota_decision: &'a str,
    /// KV cache 是否命中。
    pub cache_hit: bool,
}

/// 把字符串字段净化为可安全落盘的形态：控制字符替换为 U+FFFD，
/// 截断到 [`MAX_FIELD_CHARS`] 个字符。截断内容不追加省略标记，避免
/// 生成新的可变内容。
fn sanitize_field(value: &str) -> String {
    value
        .chars()
        .map(|c| if c.is_control() { '\u{FFFD}' } else { c })
        .take(MAX_FIELD_CHARS)
        .collect()
}

/// 把事件脱敏为仅含 [`AUDIT_KEYS`] 白名单字段的 JSON 对象。
///
/// 输出恰好包含 13 个白名单键，不存在 prompt/response/secret 等键；
/// 字符串字段经 [`sanitize_field`] 净化，保证单行 JSON 不含裸换行。
pub fn redact_event(event: &AuditEvent<'_>) -> serde_json::Value {
    serde_json::json!({
        "request_id": sanitize_field(event.request_id),
        "trace_id": sanitize_field(event.trace_id),
        "key_hash": sanitize_field(event.key_hash),
        "tenant": sanitize_field(event.tenant),
        "model": sanitize_field(event.model),
        "endpoint": sanitize_field(event.endpoint),
        "prompt_tokens": event.prompt_tokens,
        "completion_tokens": event.completion_tokens,
        "duration_ms": event.duration_ms,
        "status": event.status,
        "error_type": event.error_type.map(sanitize_field),
        "quota_decision": sanitize_field(event.quota_decision),
        "cache_hit": event.cache_hit,
    })
}

/// JSONL 审计日志写入器。
///
/// `path` 为 `None` 时不落盘（写入为空操作）；为 `Some` 时按追加方式
/// 打开文件，不创建父目录（目录由部署方负责，缺失时报错而非静默补建）。
///
/// 线程安全说明：内部 `Mutex<()>` 只串行化经同一个 `AuditLog` 实例的
/// 写入；它不是进程级锁——多个 `AuditLog` 实例指向同一文件、或外部
/// 进程并发写同一文件时，仍可能出现行交错，部署方应保证单一写入者。
#[derive(Debug)]
pub struct AuditLog {
    /// 审计文件路径；`None` 表示不落盘。
    path: Option<std::path::PathBuf>,
    /// 进程内写锁，见结构体文档的线程安全说明。
    write_lock: Mutex<()>,
}

impl AuditLog {
    /// 创建写入器；`path` 传 `None` 即关闭落盘。
    pub fn new(path: Option<&Path>) -> Self {
        Self {
            path: path.map(Path::to_path_buf),
            write_lock: Mutex::new(()),
        }
    }

    /// 追加一条事件（一行 JSON + 换行）。
    ///
    /// 落盘路径为 `None` 时为空操作并返回 `Ok`；打开/写入失败时把
    /// `std::io::Error` 原样返回给调用方，由 HTTP 层决定如何上报。
    pub fn write(&self, event: &AuditEvent<'_>) -> std::io::Result<()> {
        let Some(path) = &self.path else {
            return Ok(());
        };
        // 锁内仅做序列化与追加写，无 panic 源；锁毒化时直接复用内部数据。
        let _guard = self
            .write_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(path)?;
        let line = redact_event(event).to_string();
        std::writeln!(file, "{line}")
    }
}

/// 读取审计文件最后 `limit` 条事件（保持文件顺序，旧在前）。
///
/// 每行按 JSON 解析后仅保留 [`AUDIT_KEYS`] 白名单内的键，未知键一律
/// 丢弃；无法解析的行跳过。读取/解析失败（文件不存在、非 UTF-8 等）
/// 返回空列表——本函数是只读诊断入口，失败可见性由 metrics/日志另行
/// 暴露，这里不伪造数据。
pub fn recent_summary(path: Option<&Path>, limit: usize) -> Vec<serde_json::Value> {
    let Some(path) = path else {
        return Vec::new();
    };
    let Ok(content) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let lines: Vec<&str> = content.lines().collect();
    let start = lines.len().saturating_sub(limit);
    let mut out = Vec::new();
    for line in &lines[start..] {
        let Ok(serde_json::Value::Object(map)) = serde_json::from_str(line) else {
            continue;
        };
        let mut filtered = serde_json::Map::new();
        for key in AUDIT_KEYS {
            if let Some(value) = map.get(key) {
                filtered.insert(key.to_string(), value.clone());
            }
        }
        out.push(serde_json::Value::Object(filtered));
    }
    out
}
