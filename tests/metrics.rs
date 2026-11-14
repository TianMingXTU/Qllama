#![cfg(feature = "server")]
//! `metrics` 模块的行为测试。
//!
//! 重点验证 Prometheus 文本的系列完整性、累计桶语义、标签转义、
//! request_id 对账序列以及"渲染内容不含业务体"的约束。

use qllama::server::metrics::{current_rss_bytes, Metrics};
use std::time::Instant;

/// 记录一次合法请求，参数取自固定字面量。
#[allow(clippy::too_many_arguments)]
fn record(m: &Metrics, status: u16, duration: f64, cache_hit: bool, tokens: u64) {
    m.record_request(
        status,
        "tiny-llama",
        "/v1/chat/completions",
        duration,
        0.02,
        0.01,
        100.0,
        50.0,
        0.005,
        2,
        cache_hit,
        tokens,
        None,
        None,
    );
}

/// 从渲染文本中取出某序列行的值部分（按前缀匹配首行）。
fn line(text: &str, prefix: &str) -> String {
    text.lines()
        .find(|l| l.starts_with(prefix))
        .unwrap_or_else(|| panic!("渲染文本缺少以 {prefix} 开头的行"))
        .to_string()
}

#[test]
fn render_contains_all_required_metric_names() {
    let m = Metrics::new();
    record(&m, 200, 0.03, true, 10);
    m.record_request(
        401,
        "tiny-llama",
        "/v1/models",
        0.02,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1,
        false,
        0,
        Some("invalid_credentials"),
        None,
    );
    m.bind_request("req-42", 200, "tiny-llama", "/v1/chat/completions");
    m.set_runtime(4096, 2, 1, 3, Instant::now());
    let text = m.render();
    for name in [
        "qllama_requests_total",
        "qllama_tokens_generated_total",
        "qllama_request_duration_seconds",
        "qllama_time_to_first_token_seconds",
        "qllama_time_per_output_token_seconds",
        "qllama_prefill_tokens_per_second",
        "qllama_decode_tokens_per_second",
        "qllama_queue_wait_seconds",
        "qllama_batch_size",
        "qllama_kv_cache_hit_total",
        "qllama_kv_cache_miss_total",
        "qllama_rss_bytes",
        "qllama_errors_total",
        "qllama_requests_rejected_total",
        "qllama_active_sessions",
        "qllama_loaded_models",
        "qllama_queue_depth",
        "qllama_uptime_seconds",
        "qllama_last_request",
    ] {
        assert!(text.contains(name), "渲染文本缺少 {name}");
        assert!(
            text.contains(&format!("# TYPE {name}")),
            "{name} 缺少 TYPE 行"
        );
        assert!(
            text.contains(&format!("# HELP {name}")),
            "{name} 缺少 HELP 行"
        );
    }
    // 无标签与带标签的 qllama_requests_total 同时存在。
    assert!(text.contains("qllama_requests_total 2\n"));
    assert!(text.contains("qllama_requests_total{status=\"200\""));
    assert_eq!(
        line(&text, "qllama_tokens_generated_total "),
        "qllama_tokens_generated_total 10"
    );
    assert_eq!(
        line(&text, "qllama_kv_cache_hit_total "),
        "qllama_kv_cache_hit_total 1"
    );
    assert_eq!(
        line(&text, "qllama_kv_cache_miss_total "),
        "qllama_kv_cache_miss_total 1"
    );
    assert!(text.contains("qllama_errors_total{type=\"invalid_credentials\"}"));
}

#[test]
fn labelled_counters_and_rejections_accumulate() {
    let m = Metrics::new();
    record(&m, 200, 0.01, false, 1);
    record(&m, 200, 0.5, false, 2);
    m.record_request(
        401,
        "tiny-llama",
        "/v1/models",
        0.02,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1,
        false,
        0,
        Some("invalid_credentials"),
        Some("invalid_credentials"),
    );
    let text = m.render();
    assert!(text.contains("qllama_requests_total 3\n"));
    assert!(text.contains("qllama_requests_total{status=\"200\",model=\"tiny-llama\",endpoint=\"/v1/chat/completions\"} 2"));
    assert!(text.contains(
        "qllama_requests_total{status=\"401\",model=\"tiny-llama\",endpoint=\"/v1/models\"} 1"
    ));
    assert!(text.contains("qllama_errors_total{type=\"invalid_credentials\"} 1"));
    assert!(text.contains("qllama_requests_rejected_total{reason=\"invalid_credentials\"} 1"));
    assert_eq!(
        line(&text, "qllama_tokens_generated_total "),
        "qllama_tokens_generated_total 3"
    );
    assert_eq!(
        line(&text, "qllama_kv_cache_miss_total "),
        "qllama_kv_cache_miss_total 3"
    );
}

#[test]
fn histogram_buckets_are_cumulative_and_inf_equals_count() {
    let m = Metrics::new();
    // 0.005 落入 0.01 桶；0.06 落入 0.1 桶；100.0 超 30 秒上界。
    record(&m, 200, 0.005, true, 0);
    record(&m, 200, 0.06, true, 0);
    record(&m, 200, 100.0, true, 0);
    let text = m.render();
    assert!(text.contains("qllama_request_duration_seconds_bucket{le=\"0.01\"} 1"));
    assert!(text.contains("qllama_request_duration_seconds_bucket{le=\"0.1\"} 2"));
    assert!(text.contains("qllama_request_duration_seconds_bucket{le=\"30\"} 2"));
    assert!(text.contains("qllama_request_duration_seconds_bucket{le=\"+Inf\"} 3"));
    assert!(text.contains("qllama_request_duration_seconds_count 3"));
    // 总和为三次观测之和（浮点容差）。
    let sum_line = line(&text, "qllama_request_duration_seconds_sum ");
    let sum: f64 = sum_line.rsplit(' ').next().unwrap().parse().unwrap();
    assert!((sum - 100.065).abs() < 1e-9, "sum={sum}");
    // 桶计数单调不减。
    let values: Vec<u64> = text
        .lines()
        .filter(|l| l.starts_with("qllama_request_duration_seconds_bucket"))
        .map(|l| l.rsplit(' ').next().unwrap().parse().unwrap())
        .collect();
    assert_eq!(values.len(), 11);
    assert!(values.windows(2).all(|w| w[0] <= w[1]));
    // summary 分位数与 count/sum 齐备。
    for label in ["0.5", "0.95", "0.99"] {
        assert!(text.contains(&format!(
            "qllama_time_to_first_token_seconds{{quantile=\"{label}\"}}"
        )));
    }
    assert!(text.contains("qllama_time_to_first_token_seconds_count 3"));
    assert!(text.contains("qllama_time_to_first_token_seconds_sum "));
}

#[test]
fn label_values_are_escaped() {
    let m = Metrics::new();
    m.bind_request("req\"1\\\nsecret-prompt", 200, "model\"x", "/v1/\nchat");
    m.record_request(
        500,
        "model\"x",
        "/v1/chat",
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        1,
        false,
        0,
        Some("err\n\"q\""),
        Some("reason\\x"),
    );
    let text = m.render();
    assert!(text.contains("qllama_last_request{request_id=\"req\\\"1\\\\\\nsecret-prompt\""));
    assert!(text.contains("model=\"model\\\"x\""));
    assert!(text.contains("qllama_errors_total{type=\"err\\n\\\"q\\\"\"} 1"));
    assert!(text.contains("qllama_requests_rejected_total{reason=\"reason\\\\x\"} 1"));
    // 转义后不会出现裸换行把一条序列拆成两行。
    assert!(!text.contains("err\n\"q"));
}

#[test]
fn bind_request_emits_request_id_and_accessors_track_it() {
    let m = Metrics::new();
    assert_eq!(m.last_request_id(), "");
    m.bind_request("req-007", 429, "tiny", "/v1/completions");
    assert_eq!(m.last_request_id(), "req-007");
    let text = m.render();
    assert!(text.contains("qllama_last_request{request_id=\"req-007\",status=\"429\",model=\"tiny\",endpoint=\"/v1/completions\"} 1"));
    m.set_last_request_id("req-008");
    assert_eq!(m.last_request_id(), "req-008");
    assert!(m.render().contains("request_id=\"req-008\""));
    // 未绑定时仅设置 id 也能建档。
    let fresh = Metrics::new();
    fresh.set_last_request_id("req-009");
    assert_eq!(fresh.last_request_id(), "req-009");
    assert!(fresh.render().contains("request_id=\"req-009\""));
}

#[test]
fn render_never_contains_business_payload() {
    let m = Metrics::new();
    record(&m, 200, 0.01, true, 5);
    m.bind_request("req-safe", 200, "tiny", "/v1/chat");
    let text = m.render();
    for forbidden in [
        "\"messages\"",
        "logits",
        "completion_text",
        "response_body",
        "SECRET-PROMPT",
    ] {
        assert!(!text.contains(forbidden), "渲染文本含 {forbidden}");
    }
}

#[test]
fn rss_bytes_is_non_negative() {
    let rss = current_rss_bytes();
    // u64 天然 >= 0；在 Linux 上读取成功时应大于 0，其余平台为 0。
    if cfg!(target_os = "linux") {
        assert!(rss > 0, "Linux /proc/self/status 应给出非零 RSS");
    } else {
        assert_eq!(rss, 0);
    }
}
