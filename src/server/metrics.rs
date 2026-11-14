//! Prometheus 指标注册与文本渲染。
//!
//! `Metrics` 是进程内唯一的指标聚合点：HTTP 层在请求收尾时调用
//! [`Metrics::record_request`]，运行时巡检调用 [`Metrics::set_runtime`]；
//! [`Metrics::render`] 输出 Prometheus 文本格式（含 HELP/TYPE、标签转义、
//! 累计直方图桶与样本分位数）。只聚合元数据与计数，绝无 prompt/response
//! 等业务体进入任何序列。
//!
//! 与其他模块的关系：`error_type`/`rejection_reason` 取自 `auth`/`qos`
//! 的稳定错误标识；`bind_request` 记录的 request_id 来自
//! `auth::AuthContext`，用于把日志行与指标序列对账。
//!
//! 命名兼容说明：`qllama_requests_total` 保留无标签序列（向后兼容），
//! 同时输出同名带 `{status,model,endpoint}` 标签的序列——Prometheus 文本
//! 协议允许同名指标的不同标签集并存，抓取端按完整标签集区分时间序列。

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

/// 请求耗时直方图的有限桶上界（秒）；`+Inf` 桶由总计数隐含。
const HIST_BUCKETS: [f64; 10] = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0];

/// 摘要序列保留的最近样本数上限，用于分位数计算；超出丢弃最旧样本，
/// 总数与总和仍覆盖全部观测。
const SAMPLE_CAP: usize = 4096;

/// 监控计数使用的内存序：仅用于监控聚合，无同步语义，Relaxed 足够。
const RELAXED: Ordering = Ordering::Relaxed;

/// 原子累加一个 f64（按位模式做 CAS 循环；失败重试直到成功）。
fn add_f64(counter: &AtomicU64, value: f64) {
    let _ = counter.try_update(RELAXED, RELAXED, |bits| {
        Some((f64::from_bits(bits) + value).to_bits())
    });
}

/// 读取原子存放的 f64。
fn load_f64(counter: &AtomicU64) -> f64 {
    f64::from_bits(counter.load(RELAXED))
}

/// Prometheus 标签值转义：反斜杠、双引号、换行。
fn escape_label(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

/// 锁毒化恢复：指标状态只由本模块更新、临界区内无 panic 源，
/// 直接复用内部数据比让后续抓取 panic 更安全。
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// 直方图：有限桶计数 + 总数 + 总和；渲染时做累计化。
///
/// `buckets[0..10]` 对应 [`HIST_BUCKETS`] 的有限上界；`buckets[10]`
/// 承接超过 30 秒的观测（渲染时由 `_count` 作为 `+Inf` 桶输出）。
#[derive(Debug, Default)]
struct Histogram {
    /// 各桶落入计数；末位为超界桶，见结构体文档。
    buckets: [AtomicU64; 11],
    /// 全部观测数（即 `+Inf` 桶）。
    count: AtomicU64,
    /// 全部观测总和，按位模式存放。
    sum_bits: AtomicU64,
}

impl Histogram {
    /// 记录一次观测；非有限值忽略（避免 NaN 污染累计和与输出）。
    fn observe(&self, value: f64) {
        if !value.is_finite() {
            return;
        }
        self.count.fetch_add(1, RELAXED);
        add_f64(&self.sum_bits, value);
        let idx = HIST_BUCKETS.partition_point(|&edge| edge < value);
        self.buckets[idx].fetch_add(1, RELAXED);
    }

    /// 渲染为 `_bucket`（累计）、`_sum`、`_count` 序列。
    fn render(&self, name: &str, help: &str, out: &mut String) {
        out.push_str("# HELP ");
        out.push_str(name);
        out.push(' ');
        out.push_str(help);
        out.push('\n');
        out.push_str("# TYPE ");
        out.push_str(name);
        out.push_str(" histogram\n");
        let count = self.count.load(RELAXED);
        let mut cumulative = 0u64;
        for (idx, &edge) in HIST_BUCKETS.iter().enumerate() {
            cumulative += self.buckets[idx].load(RELAXED);
            out.push_str(&format!("{name}_bucket{{le=\"{edge}\"}} {cumulative}\n"));
        }
        out.push_str(&format!("{name}_bucket{{le=\"+Inf\"}} {count}\n"));
        out.push_str(&format!("{name}_sum {}\n", load_f64(&self.sum_bits)));
        out.push_str(&format!("{name}_count {count}\n"));
    }
}

/// 摘要序列：全量 count/sum + 最近样本的 p50/p95/p99 分位数。
#[derive(Debug, Default)]
struct SummarySeries {
    /// 全部观测数。
    count: AtomicU64,
    /// 全部观测总和，按位模式存放。
    sum_bits: AtomicU64,
    /// 最近 [`SAMPLE_CAP`] 个样本，用于分位数估计。
    samples: Mutex<Vec<f64>>,
}

impl SummarySeries {
    /// 记录一次观测；非有限值忽略。
    fn observe(&self, value: f64) {
        if !value.is_finite() {
            return;
        }
        self.count.fetch_add(1, RELAXED);
        add_f64(&self.sum_bits, value);
        let mut samples = lock(&self.samples);
        if samples.len() >= SAMPLE_CAP {
            samples.remove(0);
        }
        samples.push(value);
    }

    /// 按最近秩法（nearest-rank）从升序样本取分位数；空样本返回 0.0。
    fn quantile(sorted: &[f64], q: f64) -> f64 {
        if sorted.is_empty() {
            return 0.0;
        }
        let idx = ((sorted.len() as f64 * q).ceil() as usize).clamp(1, sorted.len()) - 1;
        sorted[idx]
    }

    /// 渲染为 `_count`、`_sum` 与 `{quantile=...}` 序列。
    fn render(&self, name: &str, help: &str, out: &mut String) {
        out.push_str("# HELP ");
        out.push_str(name);
        out.push(' ');
        out.push_str(help);
        out.push('\n');
        out.push_str("# TYPE ");
        out.push_str(name);
        out.push_str(" summary\n");
        let count = self.count.load(RELAXED);
        let quantiles: [(f64, &str); 3] = [(0.5, "0.5"), (0.95, "0.95"), (0.99, "0.99")];
        let samples = lock(&self.samples);
        let mut sorted = samples.clone();
        drop(samples);
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for (q, label) in quantiles {
            let v = Self::quantile(&sorted, q);
            out.push_str(&format!("{name}{{quantile=\"{label}\"}} {v}\n"));
        }
        out.push_str(&format!("{name}_sum {}\n", load_f64(&self.sum_bits)));
        out.push_str(&format!("{name}_count {count}\n"));
    }
}

/// 进程内指标聚合器；所有字段均为原子量或互斥锁保护的映射，可跨线程共享。
#[derive(Debug, Default)]
pub struct Metrics {
    /// 无标签请求总数（向后兼容保留）。
    requests_total: AtomicU64,
    /// 生成 token 总数。
    tokens_generated_total: AtomicU64,
    /// 带 `{status,model,endpoint}` 标签的请求计数。
    labelled_requests: Mutex<HashMap<(u16, String, String), u64>>,
    /// 请求耗时直方图。
    duration: Histogram,
    /// 首字延迟摘要。
    ttft: SummarySeries,
    /// 每输出 token 耗时摘要。
    tpot: SummarySeries,
    /// prefill 吞吐摘要。
    prefill_tps: SummarySeries,
    /// decode 吞吐摘要。
    decode_tps: SummarySeries,
    /// 排队等待摘要。
    queue_wait: SummarySeries,
    /// 批大小摘要。
    batch_size: SummarySeries,
    /// KV cache 命中计数。
    kv_hit: AtomicU64,
    /// KV cache 未命中计数。
    kv_miss: AtomicU64,
    /// 进程常驻内存（字节）。
    rss_bytes: AtomicU64,
    /// 按 `type` 标签分类的错误计数。
    errors: Mutex<HashMap<String, u64>>,
    /// 按 `reason` 标签分类的拒绝计数。
    rejected: Mutex<HashMap<String, u64>>,
    /// 活跃会话数。
    active_sessions: AtomicU64,
    /// 已加载模型数。
    loaded_models: AtomicU64,
    /// 当前排队深度。
    queue_depth: AtomicU64,
    /// 进程起始时刻，用于计算 uptime。
    started: Mutex<Option<Instant>>,
    /// 最近一次请求的对账信息 `(request_id, status, model, endpoint)`。
    last_request: Mutex<Option<(String, u16, String, String)>>,
}

impl Metrics {
    /// 创建全零指标集。
    pub fn new() -> Self {
        Self::default()
    }

    /// 记录一次已完成/被拒绝的请求。
    ///
    /// 同时递增无标签与带标签的 `qllama_requests_total`；`tokens` 累入
    /// `qllama_tokens_generated_total`；`duration` 入直方图；其余浮点观
    /// 测入各自摘要序列；`cache_hit` 决定命中/未命中计数；
    /// `error_type`/`rejection_reason` 非空时分别累入错误与拒绝计数
    /// （取值应为 `auth`/`qos` 的稳定错误标识）。非有限浮点观测被忽略。
    /// 本方法不更新 `qllama_last_request`，对账信息由 [`Metrics::bind_request`] 写入。
    #[allow(clippy::too_many_arguments)]
    pub fn record_request(
        &self,
        status: u16,
        model: &str,
        endpoint: &str,
        duration: f64,
        ttft: f64,
        tpot: f64,
        prefill_tps: f64,
        decode_tps: f64,
        queue_wait: f64,
        batch_size: usize,
        cache_hit: bool,
        tokens: u64,
        error_type: Option<&str>,
        rejection_reason: Option<&str>,
    ) {
        self.requests_total.fetch_add(1, RELAXED);
        self.tokens_generated_total.fetch_add(tokens, RELAXED);
        *lock(&self.labelled_requests)
            .entry((status, model.to_string(), endpoint.to_string()))
            .or_insert(0) += 1;
        self.duration.observe(duration);
        self.ttft.observe(ttft);
        self.tpot.observe(tpot);
        self.prefill_tps.observe(prefill_tps);
        self.decode_tps.observe(decode_tps);
        self.queue_wait.observe(queue_wait);
        self.batch_size.observe(batch_size as f64);
        if cache_hit {
            self.kv_hit.fetch_add(1, RELAXED);
        } else {
            self.kv_miss.fetch_add(1, RELAXED);
        }
        if let Some(err) = error_type {
            *lock(&self.errors).entry(err.to_string()).or_insert(0) += 1;
        }
        if let Some(reason) = rejection_reason {
            *lock(&self.rejected).entry(reason.to_string()).or_insert(0) += 1;
        }
    }

    /// 更新运行时巡检量：RSS、活跃会话、已加载模型、排队深度，并记录
    /// 进程起始时刻用于 [`Metrics::render`] 计算 `qllama_uptime_seconds`。
    pub fn set_runtime(
        &self,
        rss_bytes: u64,
        active_sessions: usize,
        loaded_models: usize,
        queue_depth: usize,
        started: Instant,
    ) {
        self.rss_bytes.store(rss_bytes, RELAXED);
        self.active_sessions.store(active_sessions as u64, RELAXED);
        self.loaded_models.store(loaded_models as u64, RELAXED);
        self.queue_depth.store(queue_depth as u64, RELAXED);
        *lock(&self.started) = Some(started);
    }

    /// 绑定最近一次请求的对账信息；`render` 会输出
    /// `qllama_last_request{request_id,status,model,endpoint} 1`，
    /// 供契约测试把日志行与指标序列对上。
    pub fn bind_request(&self, request_id: &str, status: u16, model: &str, endpoint: &str) {
        *lock(&self.last_request) = Some((
            request_id.to_string(),
            status,
            model.to_string(),
            endpoint.to_string(),
        ));
    }

    /// 最近绑定的请求 id；未绑定时返回空串。
    pub fn last_request_id(&self) -> String {
        lock(&self.last_request)
            .as_ref()
            .map(|(id, ..)| id.clone())
            .unwrap_or_default()
    }

    /// 仅更新最近请求 id（不改动其余标签）；从未绑定时以零值标签建档。
    pub fn set_last_request_id(&self, id: &str) {
        let mut guard = lock(&self.last_request);
        match guard.as_mut() {
            Some(entry) => entry.0 = id.to_string(),
            None => *guard = Some((id.to_string(), 0, String::new(), String::new())),
        }
    }

    /// 渲染 Prometheus 文本格式。带标签序列按键排序保证输出稳定；
    /// 无样本的带标签序列仅输出 HELP/TYPE 元数据行。
    pub fn render(&self) -> String {
        let mut out = String::new();

        out.push_str(
            "# HELP qllama_requests_total 请求总数（无标签序列向后兼容；带标签序列同名并存）。\n",
        );
        out.push_str("# TYPE qllama_requests_total counter\n");
        out.push_str(&format!(
            "qllama_requests_total {}\n",
            self.requests_total.load(RELAXED)
        ));
        let mut labelled: Vec<((u16, String, String), u64)> = lock(&self.labelled_requests)
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        labelled.sort();
        for ((status, model, endpoint), count) in labelled {
            out.push_str(&format!(
                "qllama_requests_total{{status=\"{status}\",model=\"{}\",endpoint=\"{}\"}} {count}\n",
                escape_label(&model),
                escape_label(&endpoint),
            ));
        }

        out.push_str("# HELP qllama_tokens_generated_total 生成 token 总数。\n");
        out.push_str("# TYPE qllama_tokens_generated_total counter\n");
        out.push_str(&format!(
            "qllama_tokens_generated_total {}\n",
            self.tokens_generated_total.load(RELAXED)
        ));

        self.duration.render(
            "qllama_request_duration_seconds",
            "请求总耗时（秒）分布。",
            &mut out,
        );
        self.ttft.render(
            "qllama_time_to_first_token_seconds",
            "首字延迟（秒）。",
            &mut out,
        );
        self.tpot.render(
            "qllama_time_per_output_token_seconds",
            "每输出 token 耗时（秒）。",
            &mut out,
        );
        self.prefill_tps.render(
            "qllama_prefill_tokens_per_second",
            "prefill 吞吐（token/秒）。",
            &mut out,
        );
        self.decode_tps.render(
            "qllama_decode_tokens_per_second",
            "decode 吞吐（token/秒）。",
            &mut out,
        );
        self.queue_wait.render(
            "qllama_queue_wait_seconds",
            "排队等待时长（秒）。",
            &mut out,
        );
        self.batch_size
            .render("qllama_batch_size", "批大小分布。", &mut out);

        out.push_str("# HELP qllama_kv_cache_hit_total KV cache 命中次数。\n");
        out.push_str("# TYPE qllama_kv_cache_hit_total counter\n");
        out.push_str(&format!(
            "qllama_kv_cache_hit_total {}\n",
            self.kv_hit.load(RELAXED)
        ));
        out.push_str("# HELP qllama_kv_cache_miss_total KV cache 未命中次数。\n");
        out.push_str("# TYPE qllama_kv_cache_miss_total counter\n");
        out.push_str(&format!(
            "qllama_kv_cache_miss_total {}\n",
            self.kv_miss.load(RELAXED)
        ));

        out.push_str("# HELP qllama_rss_bytes 进程常驻内存（字节）。\n");
        out.push_str("# TYPE qllama_rss_bytes gauge\n");
        out.push_str(&format!(
            "qllama_rss_bytes {}\n",
            self.rss_bytes.load(RELAXED)
        ));

        out.push_str("# HELP qllama_errors_total 按 type 分类的错误计数。\n");
        out.push_str("# TYPE qllama_errors_total counter\n");
        let mut errors: Vec<(String, u64)> = lock(&self.errors)
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        errors.sort();
        for (err, count) in errors {
            out.push_str(&format!(
                "qllama_errors_total{{type=\"{}\"}} {count}\n",
                escape_label(&err),
            ));
        }

        out.push_str("# HELP qllama_requests_rejected_total 按 reason 分类的拒绝计数。\n");
        out.push_str("# TYPE qllama_requests_rejected_total counter\n");
        let mut rejected: Vec<(String, u64)> = lock(&self.rejected)
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        rejected.sort();
        for (reason, count) in rejected {
            out.push_str(&format!(
                "qllama_requests_rejected_total{{reason=\"{}\"}} {count}\n",
                escape_label(&reason),
            ));
        }

        out.push_str("# HELP qllama_active_sessions 当前活跃会话数。\n");
        out.push_str("# TYPE qllama_active_sessions gauge\n");
        out.push_str(&format!(
            "qllama_active_sessions {}\n",
            self.active_sessions.load(RELAXED)
        ));
        out.push_str("# HELP qllama_loaded_models 已加载模型数。\n");
        out.push_str("# TYPE qllama_loaded_models gauge\n");
        out.push_str(&format!(
            "qllama_loaded_models {}\n",
            self.loaded_models.load(RELAXED)
        ));
        out.push_str("# HELP qllama_queue_depth 当前排队深度。\n");
        out.push_str("# TYPE qllama_queue_depth gauge\n");
        out.push_str(&format!(
            "qllama_queue_depth {}\n",
            self.queue_depth.load(RELAXED)
        ));

        out.push_str("# HELP qllama_uptime_seconds 进程运行时长（秒）。\n");
        out.push_str("# TYPE qllama_uptime_seconds gauge\n");
        let uptime = lock(&self.started)
            .as_ref()
            .map(|start| start.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        out.push_str(&format!("qllama_uptime_seconds {uptime}\n"));

        out.push_str("# HELP qllama_last_request 最近一次请求的对账序列。\n");
        out.push_str("# TYPE qllama_last_request gauge\n");
        if let Some((id, status, model, endpoint)) = lock(&self.last_request).as_ref() {
            out.push_str(&format!(
                "qllama_last_request{{request_id=\"{}\",status=\"{status}\",model=\"{}\",endpoint=\"{}\"}} 1\n",
                escape_label(id),
                escape_label(model),
                escape_label(endpoint),
            ));
        }

        out
    }
}

/// 读取当前进程 RSS（字节）。Linux 下解析 `/proc/self/status` 的 `VmRSS`
/// （单位 kB，与页大小无关，避免假设 4KiB 页）；读取失败返回 0——
/// RSS 本身是尽力而为的巡检量。其余平台返回 0（不引入 FFI/依赖）。
pub fn current_rss_bytes() -> u64 {
    if !cfg!(target_os = "linux") {
        return 0;
    }
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb: u64 = rest
                .trim()
                .trim_end_matches("kB")
                .trim()
                .parse()
                .unwrap_or(0);
            return kb.saturating_mul(1024);
        }
    }
    0
}
