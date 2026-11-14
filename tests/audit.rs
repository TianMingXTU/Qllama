#![cfg(feature = "server")]
//! `audit` 模块的行为测试。
//!
//! 用独立临时目录（std 文件 API，不依赖新 dev 依赖）验证脱敏、JSONL
//! 追加写入与读取过滤；用例结束后清理自己创建的文件/目录。

use qllama::server::audit::{recent_summary, redact_event, AuditEvent, AuditLog};
use std::path::PathBuf;

/// 创建唯一的临时目录，返回其路径；由各用例自行清理。
fn temp_dir(label: &str) -> PathBuf {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("qllama-audit-{label}-{pid}-{nanos}"));
    std::fs::create_dir(&path).unwrap();
    path
}

/// 构造一条基线事件（字符串字段用借用字面量）。
fn baseline_event() -> AuditEvent<'static> {
    AuditEvent {
        request_id: "req-1",
        trace_id: "4bf92f3577b34da6a3ce929d0e0e4736",
        key_hash: "e71fa2190541574b",
        tenant: "default",
        model: "tiny-llama",
        endpoint: "/v1/chat/completions",
        prompt_tokens: 12,
        completion_tokens: 34,
        duration_ms: 56,
        status: 200,
        error_type: None,
        quota_decision: "allowed",
        cache_hit: true,
    }
}

#[test]
fn redact_event_has_exact_allowed_keys() {
    let event = baseline_event();
    let json = redact_event(&event);
    let obj = json.as_object().unwrap();
    assert_eq!(obj.len(), 13);
    for key in [
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
    ] {
        assert!(obj.contains_key(key), "缺少白名单键 {key}");
    }
    for forbidden in ["prompt", "response", "messages", "logits", "body", "secret"] {
        assert!(!obj.contains_key(forbidden), "含敏感键 {forbidden}");
    }
    let serialized = json.to_string();
    assert!(!serialized.contains("\"prompt\""));
    assert!(!serialized.contains("\"secret\""));
    assert_eq!(json["prompt_tokens"], 12);
    assert_eq!(json["completion_tokens"], 34);
    assert_eq!(json["status"], 200);
    assert_eq!(json["cache_hit"], true);
}

#[test]
fn redact_event_sanitizes_control_characters() {
    let mut event = baseline_event();
    event.tenant = "team\n\"x\"\\y\r\u{0007}";
    event.error_type = Some("bad\ninput");
    let json = redact_event(&event);
    assert_eq!(json["tenant"], "team\u{FFFD}\"x\"\\y\u{FFFD}\u{FFFD}");
    assert_eq!(json["error_type"], "bad\u{FFFD}input");
    // U+FFFD 而非裸换行/制表符，保证单行 JSONL 不被注入额外行。
    let serialized = json.to_string();
    assert!(!serialized.contains('\n'));
    assert!(!serialized.contains('\r'));
}

#[test]
fn redact_event_truncates_long_metadata() {
    let long = "x".repeat(300);
    let mut event = baseline_event();
    event.model = &long;
    event.endpoint = &long;
    let json = redact_event(&event);
    assert_eq!(json["model"].as_str().unwrap().chars().count(), 128);
    assert_eq!(json["endpoint"].as_str().unwrap().chars().count(), 128);
    // 即便误把业务体内容塞进元数据字段，也不会原样落盘。
    assert!(!json.to_string().contains(&long));
}

#[test]
fn write_and_recent_summary_roundtrip() {
    let dir = temp_dir("jsonl");
    let path = dir.join("audit.jsonl");
    let log = AuditLog::new(Some(&path));
    let first = baseline_event();
    log.write(&first).unwrap();
    let mut second = baseline_event();
    second.request_id = "req-2";
    second.status = 429;
    second.quota_decision = "request_rate";
    log.write(&second).unwrap();

    // 文件恰好两行，且每行均为合法 JSON。
    let content = std::fs::read_to_string(&path).unwrap();
    assert_eq!(content.lines().count(), 2);
    for line in content.lines() {
        let _: serde_json::Value = serde_json::from_str(line).unwrap();
    }

    let all = recent_summary(Some(&path), 10);
    assert_eq!(all.len(), 2);
    assert_eq!(all[0]["request_id"], "req-1");
    assert_eq!(all[1]["request_id"], "req-2");
    assert_eq!(all[1]["status"], 429);

    let last = recent_summary(Some(&path), 1);
    assert_eq!(last.len(), 1);
    assert_eq!(last[0]["request_id"], "req-2");

    std::fs::remove_file(&path).unwrap();
    std::fs::remove_dir(&dir).unwrap();
}

#[test]
fn recent_summary_filters_unknown_keys_and_tolerates_bad_lines() {
    let dir = temp_dir("filter");
    let path = dir.join("audit.jsonl");
    std::fs::write(
        &path,
        concat!(
            "{\"request_id\":\"req-1\",\"status\":200,\"prompt\":\"SECRET-PROMPT\",",
            "\"secret\":\"SECRET-VALUE\",\"body\":\"SECRET-BODY\"}\n",
            "not-json\n",
            "{\"request_id\":\"req-2\",\"endpoint\":\"/health\",\"prompt\":\"x\"}\n"
        ),
    )
    .unwrap();
    let events = recent_summary(Some(&path), 10);
    assert_eq!(events.len(), 2);
    assert_eq!(events[0]["request_id"], "req-1");
    assert!(!events[0].as_object().unwrap().contains_key("prompt"));
    assert!(!events[0].as_object().unwrap().contains_key("secret"));
    assert!(!events[0].as_object().unwrap().contains_key("body"));
    let serialized = serde_json::to_string(&events).unwrap();
    assert!(!serialized.contains("SECRET-PROMPT"));
    assert!(!serialized.contains("SECRET-VALUE"));

    // 不存在的文件与 limit=0 均得到空列表（不伪造数据）。
    let missing = dir.join("missing.jsonl");
    assert!(recent_summary(Some(&missing), 10).is_empty());
    assert!(recent_summary(Some(&path), 0).is_empty());
    assert!(recent_summary(None, 10).is_empty());

    std::fs::remove_file(&path).unwrap();
    std::fs::remove_dir(&dir).unwrap();
}

#[test]
fn write_without_path_is_noop_and_missing_parent_is_error() {
    let log = AuditLog::new(None);
    log.write(&baseline_event()).unwrap();
    // 无路径时不产生任何文件，直接返回 Ok。

    let dir = temp_dir("missing-parent");
    let path = dir.join("nested").join("audit.jsonl");
    let log = AuditLog::new(Some(&path));
    let err = log.write(&baseline_event()).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    assert!(!path.exists());
    std::fs::remove_dir(&dir).unwrap();
}
