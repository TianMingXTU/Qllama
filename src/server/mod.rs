//! OpenAI 兼容 HTTP 推理服务层。
//!
//! 结构对应 ADR 0001：Engine（模型，只读共享）/ Session（每请求 KV + 采样状态）
//! / Scheduler（请求队列与 decode step 调度）。线程模型：tokio 负责 HTTP 与
//! 超时，模型前同步执行，结果经 channel 送回。
//!
//! 调度器支持两种模式：`--no-cont-batching` 保留逐请求串行基线；默认模式在
//! 每个 decode step 用 rayon 并行推进已就绪的独立前向。每个请求拥有独立
//! KV lease，不把不同 prompt 拼成一个融合张量。
//!
//! 取消语义：HTTP 侧断连/超时放弃时事件 Receiver 被 drop 且取消标志置位，
//! 调度器在下一个 decode step 停止生成，KV 随 session 释放。

pub mod adapters;
pub mod api;
pub mod audit;
pub mod auth;
pub mod capabilities;
pub mod extensions;
pub mod metrics;
pub mod ollama;
pub mod qos;
pub mod registry;
pub mod schema;
pub mod template;
pub mod tools;

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{mpsc, Arc, RwLock};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::kv_cache::pool::KvPool;
use crate::kv_cache::KvCache;
use crate::sampler::{Sampler, SamplerConfig};
use crate::tokenizer::Tokenizer;
use crate::transformer::Transformer;

/// 服务层错误：带 HTTP 状态码语义，`api.rs` 负责转成 OpenAI 错误体。
#[derive(Debug)]
pub struct ApiError {
    pub status: u16,
    pub kind: &'static str,
    pub message: String,
    /// 请求中显式拒绝的字段/能力；错误体写入 `error.qllama.unsupported`。
    pub unsupported: Vec<String>,
}

impl ApiError {
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: 400,
            kind: "invalid_request_error",
            message: message.into(),
            unsupported: Vec::new(),
        }
    }
    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: 404,
            kind: "invalid_request_error",
            message: message.into(),
            unsupported: Vec::new(),
        }
    }
    pub fn queue_full() -> Self {
        Self {
            status: 429,
            kind: "rate_limit_error",
            message: "queue is full; retry later".into(),
            unsupported: Vec::new(),
        }
    }
    pub fn overloaded(message: impl Into<String>) -> Self {
        Self {
            status: 503,
            kind: "service_unavailable",
            message: message.into(),
            unsupported: Vec::new(),
        }
    }
    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            status: 500,
            kind: "internal_error",
            message: message.into(),
            unsupported: Vec::new(),
        }
    }

    /// 附加不支持字段清单；错误响应会写入 `error.qllama.unsupported`。
    pub fn with_unsupported(mut self, items: Vec<&'static str>) -> Self {
        self.unsupported = items.into_iter().map(str::to_string).collect();
        self
    }

    /// 附加配额/背压拒绝原因，供指标标签使用。
    pub fn with_rejection(mut self, reason: &str) -> Self {
        self.message = format!("{} ({reason})", self.message);
        self
    }
}

/// 只读共享的模型容器。权重在 mmap 上，`Arc` 跨请求共享无需锁。
pub struct Engine {
    pub model: Transformer,
    pub tokenizer: Option<Tokenizer>,
    /// 请求可用的最大上下文（min(模型上限, --ctx-size)）
    pub ctx_size: usize,
    /// `--model-dir` 白名单根目录，启动时路径校验用
    pub model_dir: PathBuf,
    /// 启动时加载的模型文件路径；registry digest 与能力发现使用。
    pub model_path: PathBuf,
    pub model_name: String,
    /// 推理用 rayon 池；decode 的 matmul 在这里并行
    pub pool: rayon::ThreadPool,
}

impl Engine {
    /// 加载模型。`model_dir` 是路径白名单：`model` 必须位于其中，防止
    /// 任意文件读取（安全默认值）。
    pub fn load(
        model_path: &Path,
        model_dir: Option<&Path>,
        threads: Option<usize>,
        ctx_size: Option<usize>,
    ) -> Result<Arc<Self>, String> {
        let model_dir = model_dir.map(Path::to_path_buf).unwrap_or_else(|| {
            model_path
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_default()
        });
        let canonical_model = model_path
            .canonicalize()
            .map_err(|e| format!("cannot open model {}: {e}", model_path.display()))?;
        let canonical_dir = model_dir
            .canonicalize()
            .unwrap_or_else(|_| model_dir.clone());
        if !canonical_model.starts_with(&canonical_dir) {
            return Err(format!(
                "model {} is outside the allowed model directory {}",
                canonical_model.display(),
                canonical_dir.display()
            ));
        }

        let gguf = crate::gguf::GgufFile::from_path(&canonical_model)
            .map_err(|e| format!("cannot read model {}: {e}", canonical_model.display()))?;
        let model = Transformer::from_gguf(&gguf)?;
        let tokenizer = match Tokenizer::from_gguf(&gguf) {
            Ok(tokenizer) => Some(tokenizer),
            // 缺分词器的模型没有文本入口：prompt 必须分词，记录 None 并在
            // 请求时明确报 400（不静默降级）。
            Err(error) => {
                eprintln!(
                    "qllama-server: model has no usable tokenizer ({error}); \
                     /v1/completions and /v1/chat/completions will return 400"
                );
                None
            }
        };

        let threads = threads
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            })
            .max(1);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| format!("cannot build rayon pool: {e}"))?;

        let config = model.config();
        let ctx_size = ctx_size
            .unwrap_or(config.max_position_embeddings)
            .min(config.max_position_embeddings);
        let model_name = canonical_model
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("qllama")
            .to_string();

        Ok(Arc::new(Self {
            model,
            tokenizer,
            ctx_size,
            model_dir: canonical_dir,
            model_path: canonical_model,
            model_name,
            pool,
        }))
    }

    /// 每 token 的 KV cache 字节数：K/V 两份 × f32 × num_layers × num_kv_heads
    /// × head_dim。
    pub fn kv_bytes_per_token(&self) -> usize {
        let config = self.model.config();
        config.num_layers * config.num_kv_heads * config.head_dim() * 2 * std::mem::size_of::<f32>()
    }

    /// KV cache 每请求一个（ADR 0003）。容量按请求实际需要分配，不取上限。
    ///
    /// 返回 Err 而不是让分配失败杀死进程：Rust 的 Vec 分配失败走
    /// `handle_alloc_error` 直接 abort，`catch_unwind` 拦不住，所以
    /// 唯一诚实的保护是调用方（`Scheduler::execute`）在分配前做字节预算
    /// 检查；这里再用 `catch_unwind` 兜一层（如 panic 而非 alloc abort）。
    pub fn new_kv_cache(&self, capacity: usize) -> Result<KvCache, String> {
        let config = self.model.config();
        let capacity = capacity.min(self.ctx_size);
        std::panic::catch_unwind(|| {
            KvCache::new(
                config.num_layers,
                capacity,
                config.num_kv_heads,
                config.head_dim(),
            )
        })
        .map_err(|_| format!("KV allocation of {capacity} tokens failed"))
    }
}

/// 单请求生成参数；`api.rs` 从 OpenAI 请求体解析并校验后构造。
#[derive(Debug, Clone)]
pub struct GenerationRequest {
    pub prompt_ids: Vec<i32>,
    pub max_tokens: usize,
    pub sampler: SamplerConfig,
    pub seed: Option<u64>,
    /// 最多 4 条 stop 序列（在解码后的累计文本上做前缀截断）
    pub stop_texts: Vec<String>,
    /// 可选结构化输出约束；`Debug` 供调度线程持有，生成循环据此 mask logits。
    pub schema: Option<crate::server::schema::JsonSchema>,
}

/// 会话级失败原因；`api.rs` 按变体映射 HTTP 状态码。
#[derive(Debug)]
pub enum Failure {
    /// 队列满（背压）→ 429
    QueueFull,
    /// 资源保护触发（KV 预算/分配失败）→ 503
    Overloaded(String),
    /// 客户端断连或服务端超时取消 → 499
    Cancelled,
    /// 生成循环内部错误 → 500
    Internal(String),
}

/// 单请求的流式输出通道：调度器把文本片段送进通道，HTTP 层逐片段 flush。
///
/// 用 tokio unbounded 通道：调度器侧 `send` 是同步调用且永不阻塞，
/// Receiver 被 drop 后 `send` 返回 Err——这就是断连检测。
pub enum SessionEvent {
    /// 增量文本片段
    Delta(String),
    /// 生成结束；携带完整 token 序列与停止原因（"stop" | "length"）
    Done {
        tokens: Vec<i32>,
        reason: &'static str,
        ttft: Duration,
        total: Duration,
        cache_hit: bool,
    },
    /// 致命错误：session 终止
    Failed(Failure),
}

/// 一个在途请求：调度器持有直到完成或取消。
pub struct Session {
    pub request: GenerationRequest,
    pub events: tokio::sync::mpsc::UnboundedSender<SessionEvent>,
    /// 取消标志：HTTP 侧断连/超时置位（api.rs 的 CancelGuard），
    /// 调度器在出队时与每个 decode step 检查
    pub cancelled: Arc<AtomicBool>,
}

/// 服务端限制项（ADR 0004）。
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub queue_capacity: usize,
    /// 单请求总预算：排队 + 推理 + 流式传输
    pub request_timeout: Duration,
    pub max_request_bytes: usize,
    /// KV cache 字节预算：单请求 KV 估存超过它即拒绝（503）。
    /// 串行引擎同一时刻只有一个请求持有 KV，全局预算与单请求上限等价；
    /// 以字节计而非 token 计，因为不同模型的每 token KV 开销差异巨大
    /// （层数 × KV 头数 × head_dim），token 预算无法换算成真实内存压力。
    pub kv_byte_budget: usize,
}

/// 启动时冻结的运行时选项；`/v1/capabilities` 与可复现合同直接暴露。
#[derive(Clone, Debug, serde::Serialize)]
pub struct RuntimeOptions {
    pub public_health: bool,
    pub public_metrics: bool,
    pub cors_origins: Vec<String>,
    pub oidc_enabled: bool,
    pub continuous_batching: bool,
    pub single_request_parallel: bool,
    pub prefix_cache: bool,
    pub kv_block_size: usize,
    pub audit_log: Option<PathBuf>,
    pub api_key_file: Option<PathBuf>,
    pub tls_cert: Option<PathBuf>,
    pub tls_key: Option<PathBuf>,
}

impl Default for RuntimeOptions {
    fn default() -> Self {
        Self {
            public_health: true,
            public_metrics: true,
            cors_origins: Vec::new(),
            oidc_enabled: false,
            continuous_batching: true,
            single_request_parallel: true,
            prefix_cache: false,
            kv_block_size: 32,
            audit_log: None,
            api_key_file: None,
            tls_cert: None,
            tls_key: None,
        }
    }
}

/// 鉴权、QoS 与可复现配置的可变覆盖层；三参数 `start` 保留兼容默认值。
#[derive(Clone, Debug, Default)]
pub struct RuntimeConfig {
    pub auth: auth::AuthConfig,
    pub options: RuntimeOptions,
    pub default_quota: qos::Quota,
    pub per_key_quota: std::collections::HashMap<String, qos::Quota>,
    pub key_file_reload: bool,
}

/// 可在同一 decode step 并行调用的生成 worker；只持有共享模型与 KV 池，
/// 不持有 `mpsc::Receiver`，因此满足 `Send + Sync`。
#[derive(Clone)]
struct GenerationWorker {
    engine: Arc<Engine>,
    kv_pool: Arc<KvPool>,
}

/// Scheduler：请求队列与 decode step 调度。
///
/// 默认模式把已就绪 session 组批，每个 step 并行执行各 session 的独立前向；
/// `--no-cont-batching` 保持逐请求串行，用于 A/B 与回归。
pub struct Scheduler {
    engine: Arc<Engine>,
    receiver: mpsc::Receiver<Session>,
    queue: VecDeque<Session>,
    limits: Limits,
    /// 当前排队请求数（不含正在执行的），api 层背压检查与 /metrics 共用
    queue_depth: Arc<AtomicUsize>,
    kv_pool: Arc<KvPool>,
    cont_batching_enabled: Arc<AtomicBool>,
}

impl Scheduler {
    fn new(
        engine: Arc<Engine>,
        receiver: mpsc::Receiver<Session>,
        limits: Limits,
        queue_depth: Arc<AtomicUsize>,
        kv_pool: Arc<KvPool>,
        cont_batching_enabled: Arc<AtomicBool>,
    ) -> Self {
        Self {
            engine,
            receiver,
            queue: VecDeque::new(),
            limits,
            queue_depth,
            kv_pool,
            cont_batching_enabled,
        }
    }

    /// 调度器主循环。阻塞收新请求 → 入队 → 逐个执行（FIFO，无抢占）。
    ///
    /// 所有提交端 drop 后（服务关闭）：channel 里已缓冲的请求照常收完，
    /// 再把队列里的请求跑完退出——优雅排空，不丢弃在途请求。
    pub fn run(mut self) {
        loop {
            match self.receiver.recv_timeout(Duration::from_millis(100)) {
                Ok(session) => {
                    if session.cancelled.load(AtomicOrdering::Relaxed) {
                        continue; // 排队期间就被取消，直接丢弃
                    }
                    if self.queue.len() >= self.limits.queue_capacity {
                        let _ = session
                            .events
                            .send(SessionEvent::Failed(Failure::QueueFull));
                        continue;
                    }
                    self.queue_depth.fetch_add(1, AtomicOrdering::Relaxed);
                    self.queue.push_back(session);
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                // Disconnected 只在 channel 已空且全部 sender drop 时出现：
                // 缓冲的请求已全部入队。
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    self.run_queued();
                    break;
                }
            }
            self.run_queued();
        }
    }

    /// 执行队列中所有请求。默认按 decode step 组批；`--no-cont-batching`
    /// 走逐个执行的串行基线。
    fn run_queued(&mut self) {
        if !self.cont_batching_enabled.load(AtomicOrdering::Relaxed) || self.queue.len() < 2 {
            while let Some(session) = self.queue.pop_front() {
                self.queue_depth.fetch_sub(1, AtomicOrdering::Relaxed);
                if !session.cancelled.load(AtomicOrdering::Relaxed) {
                    self.execute(session);
                }
            }
            return;
        }
        let drained_len = self.queue.len();
        let batch: Vec<Session> = std::mem::take(&mut self.queue)
            .into_iter()
            .filter(|session| !session.cancelled.load(AtomicOrdering::Relaxed))
            .collect();
        self.queue_depth
            .fetch_sub(drained_len, AtomicOrdering::Relaxed);
        if batch.is_empty() {
            return;
        }
        let worker = GenerationWorker {
            engine: Arc::clone(&self.engine),
            kv_pool: Arc::clone(&self.kv_pool),
        };
        self.engine.pool.scope(|scope| {
            for session in batch {
                let worker = worker.clone();
                scope.spawn(move |_| worker.execute(session));
            }
        });
    }

    fn execute(&mut self, session: Session) {
        let worker = GenerationWorker {
            engine: Arc::clone(&self.engine),
            kv_pool: Arc::clone(&self.kv_pool),
        };
        worker.execute(session);
    }
}

impl GenerationWorker {
    /// 分配 KV lease，然后执行单个请求的完整生成循环。
    ///
    /// 资源保护在 `KvPool::acquire` 内完成：按固定 block 对齐容量并做全局
    /// 字节预算检查，失败直接 503；lease 在取消、完成和错误路径均自动归还。
    fn execute(&self, session: Session) {
        let capacity = session
            .request
            .prompt_ids
            .len()
            .saturating_add(session.request.max_tokens)
            .min(self.engine.ctx_size);
        let mut kv = match self.kv_pool.acquire(capacity) {
            Ok(lease) => lease,
            Err(message) => {
                let _ = session
                    .events
                    .send(SessionEvent::Failed(Failure::Overloaded(format!(
                        "KV pool allocation failed: {message}"
                    ))));
                return;
            }
        };
        let started = Instant::now();
        if let Err(message) = self.run_generation(&session, &mut kv, capacity, started) {
            let _ = session
                .events
                .send(SessionEvent::Failed(Failure::Internal(message)));
        }
    }

    /// 生成循环本体：与 CLI `generate()` 同构，但输出走 channel。
    ///
    /// TTFT 从出队开始计（含 KV 分配与 prefill，不含排队时间）。
    /// stop 序列匹配在累计解码文本上做：为跨 token 的 stop 保留
    /// `holdback` 字节不发送，直到能确认匹配或不匹配（见循环内注释）。
    /// 客户端断连：`events.send` 返回 Err，立即停止，KV 随 session 释放。
    fn run_generation(
        &self,
        session: &Session,
        kv: &mut KvCache,
        capacity: usize,
        started: Instant,
    ) -> Result<(), String> {
        let mut sampler = match session.request.seed {
            Some(seed) => Sampler::with_seed(session.request.sampler.clone(), seed),
            None => Sampler::new(session.request.sampler.clone()),
        };
        let eos_id = self.engine.tokenizer.as_ref().and_then(|t| t.eos_id());
        // 取出模型引用，避免 install 闭包捕获含 mpsc::Receiver 的 `self`
        //（Receiver 非 Sync → &Scheduler 非 Send → 闭包非 Send）。
        let model = &self.engine.model;
        let pool = &self.engine.pool;
        let mut ids = session.request.prompt_ids.clone();
        let mut batch = session.request.prompt_ids.clone();
        let mut position = 0usize;
        let mut generated = Vec::with_capacity(session.request.max_tokens);
        let mut ttft: Option<Duration> = None;
        // StreamingDecoder 必须跨 step 存活：一个 UTF-8 字符可能拆在多个 token 里。
        let mut decoder = self.engine.tokenizer.as_ref().map(|t| t.stream());
        // 累计解码文本与已发送字节偏移。
        let mut decoded = String::new();
        let mut emitted = 0usize;
        // 为可能的 stop 序列跨 token 匹配保留的尾部字节数：
        // 任何 stop 都可能差 1 字节才完整，所以最后 (最长stop-1) 字节
        // 在确认前不发。数学保证 stop 匹配位置不会落在已发送区域内。
        let mut json_state = session
            .request
            .schema
            .as_ref()
            .map(|_| crate::server::schema::JsonState::new());
        let holdback = session
            .request
            .stop_texts
            .iter()
            .map(String::len)
            .max()
            .unwrap_or(0)
            .saturating_sub(1);

        for _step in 0..session.request.max_tokens {
            if session.cancelled.load(AtomicOrdering::Relaxed) {
                let _ = session
                    .events
                    .send(SessionEvent::Failed(Failure::Cancelled));
                return Ok(());
            }
            if position + batch.len() > capacity {
                break; // 上下文用尽，按 "length" 结束
            }

            let logits = pool.install(|| model.forward_with_cache(&batch, position, kv))?;
            let vocab = logits.dims()[1];
            let last_start = (logits.dims()[0] - 1) * vocab;
            let row = logits
                .data()
                .get(last_start..last_start + vocab)
                .ok_or_else(|| "logits buffer shorter than shape".to_string())?;
            let mut masked;
            let row = if let Some(state) = json_state.as_ref() {
                masked = row.to_vec();
                let tokenizer = self.engine.tokenizer.as_ref();
                crate::server::schema::apply_mask(
                    &mut masked,
                    state,
                    session.request.schema.as_ref(),
                    |id| {
                        let token_id = i32::try_from(id).ok()?;
                        tokenizer
                            .and_then(|tokenizer| tokenizer.token_text(token_id))
                            .map(str::to_string)
                    },
                );
                masked.as_slice()
            } else {
                row
            };
            let next = sampler.sample_slice(row, &ids)?;
            if let Some(state) = json_state.as_mut() {
                let next_text = self
                    .engine
                    .tokenizer
                    .as_ref()
                    .and_then(|tokenizer| tokenizer.token_text(next))
                    .unwrap_or("");
                if let Err(message) = state.push_token(next_text) {
                    let _ = session
                        .events
                        .send(SessionEvent::Failed(Failure::Internal(message)));
                    return Ok(());
                }
            }
            position += batch.len();
            batch.clear();

            // EOS 不计入输出——与 llama.cpp/CLI 语义一致。
            if Some(next) == eos_id {
                self.finish(
                    session,
                    &decoded,
                    emitted,
                    generated,
                    "stop",
                    GenerationTiming { ttft, started },
                );
                return Ok(());
            }

            generated.push(next);
            ids.push(next);
            batch.push(next);

            let piece = match &mut decoder {
                Some(decoder) => decoder.push(next)?,
                // 无分词器：送 token id 十进制文本，让调用方仍能观察到流式
                //（诚实降级，启动时已警告）。
                None => format!(" {next}"),
            };
            decoded.push_str(&piece);
            if ttft.is_none() {
                ttft = Some(started.elapsed());
            }

            // stop 序列：在累计文本上找最早出现位置；命中即截断其后内容并结束。
            if let Some(stop_at) = session
                .request
                .stop_texts
                .iter()
                .filter_map(|stop| decoded.find(stop.as_str()))
                .min()
            {
                self.finish(
                    session,
                    &decoded[..stop_at],
                    emitted,
                    generated,
                    "stop",
                    GenerationTiming { ttft, started },
                );
                return Ok(());
            }

            // 发送扣除 holdback 后的安全前缀（对齐到字符边界）。
            let mut safe_len = decoded.len().saturating_sub(holdback);
            while !decoded.is_char_boundary(safe_len) {
                safe_len -= 1;
            }
            if safe_len > emitted
                && session
                    .events
                    .send(SessionEvent::Delta(decoded[emitted..safe_len].to_string()))
                    .is_err()
            {
                return Ok(()); // 客户端已断连：停止生成，释放 KV
            }
            if safe_len > emitted {
                emitted = safe_len;
            }
            if json_state.as_ref().is_some_and(|state| state.is_complete()) {
                let piece_sent = decoded[..emitted].to_string();
                let value = json_state.as_mut().and_then(|state| state.finish().ok());
                if let Some(ref schema) = session.request.schema {
                    if value
                        .as_ref()
                        .and_then(|value| {
                            crate::server::schema::validate_json_object(schema, value).err()
                        })
                        .is_some()
                    {
                        let _ = session.events.send(SessionEvent::Failed(Failure::Internal(
                            "constrained JSON output violated schema".into(),
                        )));
                        return Ok(());
                    }
                }
                if piece_sent != decoded {
                    let _ = session.events.send(SessionEvent::Delta(piece_sent));
                }
                self.finish(
                    session,
                    &decoded,
                    emitted,
                    generated,
                    "stop",
                    GenerationTiming { ttft, started },
                );
                return Ok(());
            }
        }
        // 上下文/token 预算耗尽时结构化输出仍未完成：宁可显式失败，
        // 也不返回不可解析的垃圾文本（诚实降级）。
        if json_state
            .as_ref()
            .is_some_and(|state| !state.is_complete())
        {
            let _ = session.events.send(SessionEvent::Failed(Failure::Internal(
                "constrained JSON output did not complete within max_tokens".into(),
            )));
            return Ok(());
        }
        self.finish(
            session,
            &decoded,
            emitted,
            generated,
            "length",
            GenerationTiming { ttft, started },
        );
        Ok(())
    }

    /// 结束会话：把剩余未发送文本冲出去，再发 Done。
    ///
    /// `emitted` 是已发送字节偏移；本函数总是生成循环的收尾步（调用后即
    /// return），故按值传入，不回写。`timing` 携带 TTFT 与起始时刻。
    fn finish(
        &self,
        session: &Session,
        decoded: &str,
        emitted: usize,
        generated: Vec<i32>,
        reason: &'static str,
        timing: GenerationTiming,
    ) {
        if decoded.len() > emitted {
            let _ = session
                .events
                .send(SessionEvent::Delta(decoded[emitted..].to_string()));
        }
        let _ = session.events.send(SessionEvent::Done {
            tokens: generated,
            reason,
            ttft: timing.ttft.unwrap_or_else(|| timing.started.elapsed()),
            total: timing.started.elapsed(),
            cache_hit: false,
        });
    }
}

/// 单次生成的时间统计：TTFT（首 token 延迟）与起始时刻（用于计算总耗时）。
struct GenerationTiming {
    ttft: Option<Duration>,
    started: Instant,
}

/// 服务端共享状态：axum State。
pub struct ServerState {
    pub engine: Arc<Engine>,
    pub submit: mpsc::Sender<Session>,
    pub limits: Limits,
    /// 兼容旧调用方：配置单个 secret 时使用；多 key 路径用 `auth_config`。
    pub api_key: Option<String>,
    pub metrics: Arc<metrics::Metrics>,
    pub queue_depth: Arc<AtomicUsize>,
    pub started: Instant,
    auth_config: Arc<RwLock<auth::AuthConfig>>,
    options: RuntimeOptions,
    audit_log: Arc<audit::AuditLog>,
    qos: Arc<qos::QosManager>,
    registry: Arc<RwLock<registry::ModelRegistry>>,
    kv_pool: Arc<KvPool>,
    model_digest: String,
    config_snapshot: Value,
    cont_batching_enabled: Arc<AtomicBool>,
}

impl ServerState {
    /// 装配服务状态并启动调度线程。保留三参数签名供既有集成测试使用。
    pub fn start(
        engine: Arc<Engine>,
        limits: Limits,
        api_key: Option<String>,
    ) -> Result<(Arc<Self>, std::thread::JoinHandle<()>), String> {
        let mut runtime = RuntimeConfig::default();
        if let Some(secret) = &api_key {
            runtime.auth.keys.push(auth::ApiKey {
                id: "primary".into(),
                secret: secret.clone(),
                role: auth::Role::Admin,
                tenant: "default".into(),
                revoked: false,
                expires_at: None,
            });
        }
        Self::start_with_runtime(engine, limits, api_key, runtime)
    }

    /// 装配带多 key、QoS、审计、registry 与可复现快照的服务状态。
    pub fn start_with_runtime(
        engine: Arc<Engine>,
        limits: Limits,
        api_key: Option<String>,
        runtime: RuntimeConfig,
    ) -> Result<(Arc<Self>, std::thread::JoinHandle<()>), String> {
        let (submit, receiver) = mpsc::channel::<Session>();
        let queue_depth = Arc::new(AtomicUsize::new(0));
        let cont_batching_enabled = Arc::new(AtomicBool::new(runtime.options.continuous_batching));
        let mut kv_pool = KvPool::new(
            engine.model.config(),
            runtime.options.kv_block_size.max(1),
            limits.kv_byte_budget.max(1),
        );
        kv_pool.enable_prefix_cache(runtime.options.prefix_cache);
        let kv_pool = Arc::new(kv_pool);
        let scheduler = Scheduler::new(
            engine.clone(),
            receiver,
            limits,
            queue_depth.clone(),
            kv_pool.clone(),
            cont_batching_enabled.clone(),
        );
        let handle = std::thread::Builder::new()
            .name("qllama-scheduler".into())
            .spawn(move || scheduler.run())
            .map_err(|e| format!("cannot spawn scheduler thread: {e}"))?;
        let registry = registry::ModelRegistry::new(&engine.model_dir)
            .map_err(|e| format!("cannot initialize model registry: {e}"))?;
        let model_digest = registry
            .list()
            .iter()
            .find(|entry| entry.path == engine.model_path)
            .map(|entry| entry.digest.clone())
            .or_else(|| registry::compute_file_digest(&engine.model_path).ok())
            .unwrap_or_else(|| "unavailable".into());
        let config_snapshot = json!({
            "version": env!("CARGO_PKG_VERSION"),
            "listen_default": "127.0.0.1:8080",
            "threads": engine.pool.current_num_threads(),
            "ctx_size": engine.ctx_size,
            "limits": {
                "queue_capacity": limits.queue_capacity,
                "request_timeout_secs": limits.request_timeout.as_secs(),
                "max_request_bytes": limits.max_request_bytes,
                "kv_budget_mib": limits.kv_byte_budget / (1024 * 1024),
            },
            "runtime": runtime.options,
        });
        let state = Self {
            engine,
            submit,
            limits,
            api_key,
            metrics: Arc::new(metrics::Metrics::new()),
            queue_depth,
            started: Instant::now(),
            auth_config: Arc::new(RwLock::new(runtime.auth)),
            options: runtime.options.clone(),
            audit_log: Arc::new(audit::AuditLog::new(runtime.options.audit_log.as_deref())),
            qos: Arc::new(qos::QosManager::new(
                runtime.default_quota,
                runtime.per_key_quota,
            )),
            registry: Arc::new(RwLock::new(registry)),
            kv_pool,
            model_digest,
            config_snapshot,
            cont_batching_enabled,
        };
        state.metrics.set_last_request_id("startup");
        Ok((Arc::new(state), handle))
    }

    /// 返回当前鉴权配置快照（明确重载语义：仅启动时加载或显式调用 `reload_keys`）。
    pub fn auth_config(&self) -> auth::AuthConfig {
        self.auth_config
            .read()
            .unwrap_or_else(|poison| poison.into_inner())
            .clone()
    }

    /// 启动时加载/替换密钥；轮换语义是进程内显式调用，不监听文件变化。
    pub fn reload_keys(&self, keys: Vec<auth::ApiKey>) {
        if let Ok(mut guard) = self.auth_config.write() {
            guard.keys = keys;
        }
    }

    pub fn runtime_options(&self) -> RuntimeOptions {
        self.options.clone()
    }

    pub fn audit_log(&self) -> Arc<audit::AuditLog> {
        Arc::clone(&self.audit_log)
    }

    pub fn qos(&self) -> Arc<qos::QosManager> {
        Arc::clone(&self.qos)
    }

    pub fn registry_snapshot(&self) -> Vec<registry::ModelEntry> {
        self.registry
            .read()
            .unwrap_or_else(|poison| poison.into_inner())
            .list()
    }

    pub fn load_model(&self, id: &str) -> Result<registry::ModelEntry, String> {
        let mut registry = self
            .registry
            .write()
            .map_err(|_| "model_registry_lock_poisoned".to_string())?;
        let entry = registry.resolve(id)?;
        registry.mark_loaded(&entry.id)?;
        Ok(entry)
    }

    pub fn unload_model(&self, id: &str) -> Result<(), String> {
        let mut registry = self
            .registry
            .write()
            .map_err(|_| "model_registry_lock_poisoned".to_string())?;
        registry.unload(id)
    }

    pub fn kv_pool(&self) -> Arc<KvPool> {
        Arc::clone(&self.kv_pool)
    }

    pub fn model_digest(&self) -> String {
        self.model_digest.clone()
    }

    pub fn capability_snapshot(&self) -> Value {
        let tokenizer = self.engine.tokenizer.as_ref();
        let custom_template = tokenizer.and_then(|t| t.chat_template()).is_some();
        capabilities::capability_document(&capabilities::CapabilitySnapshot {
            version: env!("CARGO_PKG_VERSION"),
            model_digest: self.model_digest(),
            quantization: "F32/F16/Q4_0/Q8_0".into(),
            kv_dtype: "f32",
            chat_template: if custom_template {
                "gguf:tokenizer.chat_template".into()
            } else {
                "chatml".into()
            },
            template_fallback: (!custom_template).then_some("chatml"),
            concurrency_mode: "continuous_batching:independent_forward",
            continuous_batching: self.cont_batching_enabled.load(AtomicOrdering::Relaxed),
            single_request_parallel: self.options.single_request_parallel,
            prefix_cache: self.options.prefix_cache,
            config_snapshot: self.config_snapshot.clone(),
        })
    }

    pub fn active_sessions(&self) -> usize {
        self.kv_pool.stats().active_leases
    }

    pub fn loaded_models(&self) -> usize {
        self.registry
            .read()
            .map(|registry| registry.loaded_count())
            .unwrap_or(1)
    }
}

/// 启动 axum 服务并阻塞直到收到 SIGINT/SIGTERM。返回即表示已停止接收
/// 新连接且在途请求处理完毕；此后 `ServerState` 的全部引用随调用方释放，
/// submit 通道关闭，调度线程排空后退出（由调用方 join）。
pub async fn serve(state: Arc<ServerState>, listen: std::net::SocketAddr) -> anyhow::Result<()> {
    let app = extensions::router_with_extensions(state);
    let listener = tokio::net::TcpListener::bind(listen).await?;
    eprintln!("qllama-server listening on {listen}");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c().await.ok();
    };
    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{signal, SignalKind};
        match signal(SignalKind::terminate()) {
            Ok(mut term) => {
                term.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
