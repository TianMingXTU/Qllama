//! OpenAI 兼容 HTTP 推理服务层。
//!
//! 结构对应 ADR 0001：Engine（模型，只读共享）/ Session（每请求 KV + 采样状态）
//! / Scheduler（单线程队列 + 串行推理）。线程模型对应 ADR 0002：tokio 只做
//! HTTP 与超时，推理在调度器自己的 OS 线程里同步执行，结果经 channel 送回。
//!
//! 诚实性约定（红线）：本引擎是**串行逐请求**执行——调度器一次只跑一个
//! 请求的 decode 循环，其余请求在 FIFO 队列里等待。这是排队，不是 batching；
//! 文档与指标里不得声称 batch/parallel inference。
//!
//! 取消语义：HTTP 侧断连/超时放弃时事件 Receiver 被 drop 且取消标志置位，
//! 调度器在下一个 decode step 停止生成，KV 随 session 释放。

pub mod api;

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

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
}

impl ApiError {
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: 400,
            kind: "invalid_request_error",
            message: message.into(),
        }
    }
    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: 404,
            kind: "invalid_request_error",
            message: message.into(),
        }
    }
    pub fn queue_full() -> Self {
        Self {
            status: 429,
            kind: "rate_limit_error",
            message: "queue is full; retry later".into(),
        }
    }
    pub fn overloaded(message: impl Into<String>) -> Self {
        Self {
            status: 503,
            kind: "service_unavailable",
            message: message.into(),
        }
    }
    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            status: 500,
            kind: "internal_error",
            message: message.into(),
        }
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
            model_name,
            pool,
        }))
    }

    /// 每 token 的 KV cache 字节数：K/V 两份 × f32 × num_layers × num_kv_heads
    /// × head_dim。
    fn kv_bytes_per_token(&self) -> usize {
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

/// Scheduler：单线程 FIFO 队列 + 串行执行。
///
/// 组批窗口（攒批 decode）在本引擎上没有意义：一次只有一个请求在 decode，
/// 其余在排队，"批"里永远只有一个。这是排队不是 batching，如实记录。
pub struct Scheduler {
    engine: Arc<Engine>,
    receiver: mpsc::Receiver<Session>,
    queue: VecDeque<Session>,
    limits: Limits,
    /// 当前排队请求数（不含正在执行的），api 层背压检查与 /metrics 共用
    queue_depth: Arc<AtomicUsize>,
}

impl Scheduler {
    fn new(
        engine: Arc<Engine>,
        receiver: mpsc::Receiver<Session>,
        limits: Limits,
        queue_depth: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            engine,
            receiver,
            queue: VecDeque::new(),
            limits,
            queue_depth,
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

    /// 执行队列中所有请求（FIFO）。
    fn run_queued(&mut self) {
        while let Some(session) = self.queue.pop_front() {
            self.queue_depth.fetch_sub(1, AtomicOrdering::Relaxed);
            if session.cancelled.load(AtomicOrdering::Relaxed) {
                continue;
            }
            self.execute(session);
        }
    }

    /// 执行单个请求的完整生成循环。
    ///
    /// 资源保护：先按模型形状估算 KV 字节数，超预算直接 503，不做分配
    /// （分配失败在 Rust 里是 abort，拦不住，必须在分配前拒绝）。
    fn execute(&mut self, session: Session) {
        let capacity = session
            .request
            .prompt_ids
            .len()
            .saturating_add(session.request.max_tokens)
            .min(self.engine.ctx_size);
        let estimated = capacity.saturating_mul(self.engine.kv_bytes_per_token());
        if estimated > self.limits.kv_byte_budget {
            let _ = session
                .events
                .send(SessionEvent::Failed(Failure::Overloaded(format!(
                    "request needs ~{} MiB of KV cache, exceeding server budget {} MiB",
                    estimated / (1024 * 1024),
                    self.limits.kv_byte_budget / (1024 * 1024)
                ))));
            return;
        }
        let kv = match self.engine.new_kv_cache(capacity) {
            Ok(kv) => kv,
            Err(message) => {
                let _ = session
                    .events
                    .send(SessionEvent::Failed(Failure::Overloaded(message)));
                return;
            }
        };
        let started = Instant::now();
        if let Err(message) = self.run_generation(&session, kv, capacity, started) {
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
        mut kv: KvCache,
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

            let logits = pool.install(|| model.forward_with_cache(&batch, position, &mut kv))?;
            let vocab = logits.dims()[1];
            let last_start = (logits.dims()[0] - 1) * vocab;
            let row = logits
                .data()
                .get(last_start..last_start + vocab)
                .ok_or_else(|| "logits buffer shorter than shape".to_string())?;
            let next = sampler.sample_slice(row, &ids)?;
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
        });
    }
}

/// 单次生成的时间统计：TTFT（首 token 延迟）与起始时刻（用于计算总耗时）。
struct GenerationTiming {
    ttft: Option<Duration>,
    started: Instant,
}

/// 进程级指标计数器（/metrics 渲染用）。queue_depth 由 Scheduler 与
/// ServerState 共享，见 `ServerState::start`。
pub struct Metrics {
    pub requests_total: AtomicU64,
    pub tokens_generated: AtomicU64,
}

impl Metrics {
    fn new() -> Self {
        Self {
            requests_total: AtomicU64::new(0),
            tokens_generated: AtomicU64::new(0),
        }
    }
}

/// 服务端共享状态：axum State。
pub struct ServerState {
    pub engine: Arc<Engine>,
    pub submit: mpsc::Sender<Session>,
    pub limits: Limits,
    pub api_key: Option<String>,
    pub metrics: Arc<Metrics>,
    /// 当前排队请求数，与 Scheduler 共享（背压检查 + /metrics）
    pub queue_depth: Arc<AtomicUsize>,
    /// 启动时刻，用于 uptime 指标
    pub started: Instant,
}

impl ServerState {
    /// 装配服务状态并启动调度线程。返回的 JoinHandle 供进程退出前
    /// 等待排空完成。
    pub fn start(
        engine: Arc<Engine>,
        limits: Limits,
        api_key: Option<String>,
    ) -> Result<(Arc<Self>, std::thread::JoinHandle<()>), String> {
        let (submit, receiver) = mpsc::channel::<Session>();
        let queue_depth = Arc::new(AtomicUsize::new(0));
        let scheduler = Scheduler::new(engine.clone(), receiver, limits, queue_depth.clone());
        let handle = std::thread::Builder::new()
            .name("qllama-scheduler".into())
            .spawn(move || scheduler.run())
            .map_err(|e| format!("cannot spawn scheduler thread: {e}"))?;
        Ok((
            Arc::new(Self {
                engine,
                submit,
                limits,
                api_key,
                metrics: Arc::new(Metrics::new()),
                queue_depth,
                started: Instant::now(),
            }),
            handle,
        ))
    }
}

/// 启动 axum 服务并阻塞直到收到 SIGINT/SIGTERM。返回即表示已停止接收
/// 新连接且在途请求处理完毕；此后 `ServerState` 的全部引用随调用方释放，
/// submit 通道关闭，调度线程排空后退出（由调用方 join）。
pub async fn serve(state: Arc<ServerState>, listen: std::net::SocketAddr) -> anyhow::Result<()> {
    let app = api::router(state);
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
