//! qllama-server：OpenAI 兼容 HTTP 推理服务入口。
//!
//! 职责仅限参数解析与进程级装配；具体逻辑在 `qllama::server` 模块。
//! 退出码约定：正常退出 0，启动失败 1，信号触发优雅退出 0。

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "qllama-server",
    version,
    about = "OpenAI-compatible HTTP inference server"
)]
struct ServerArgs {
    /// GGUF 模型文件路径；只允许 `--model-dir` 白名单目录下的文件
    #[arg(long)]
    model: PathBuf,

    /// 模型文件白名单根目录；`--model` 必须位于其中，缺省时用 `--model` 的父目录
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// 监听地址；默认仅本机回环，`0.0.0.0:8080` 才对外
    #[arg(long, default_value = "127.0.0.1:8080")]
    listen: String,

    /// 计算线程数（rayon 池）；默认取逻辑核数
    #[arg(long)]
    threads: Option<usize>,

    /// KV cache 容量上限，同时是单请求 prompt+生成的 token 上限
    #[arg(long)]
    ctx_size: Option<usize>,

    /// 队列容量；排队超过该数的请求收到 429
    #[arg(long, default_value_t = 32)]
    queue_capacity: usize,

    /// 单请求总超时（秒）：排队 + 推理 + 流式传输的总预算
    #[arg(long, default_value_t = 300)]
    request_timeout_secs: u64,

    /// 请求体大小上限（字节）
    #[arg(long, default_value_t = 1 << 20)]
    max_request_bytes: usize,

    /// 可选 API key；设置后所有 /v1 推理请求须带 `Authorization: Bearer <key>`
    #[arg(long)]
    api_key: Option<String>,

    /// KV cache 字节预算（MiB）：单请求估算超过它返回 503，防止大上下文
    /// 分配拖垮进程；默认 2048 MiB
    #[arg(long, default_value_t = 2048)]
    kv_budget_mib: usize,
}

fn main() {
    let args = ServerArgs::parse();
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        // runtime 起不来时没有 logger 可用，只能直接 stderr。
        Err(error) => {
            eprintln!("qllama-server: cannot start tokio runtime: {error}");
            std::process::exit(1);
        }
    };
    let code = runtime.block_on(run(args));
    std::process::exit(code);
}

async fn run(args: ServerArgs) -> i32 {
    // 模型加载是同步 mmap + 权重解析，用 spawn_blocking 从 runtime 里隔离。
    let load = tokio::task::spawn_blocking({
        let model = args.model.clone();
        let model_dir = args.model_dir.clone();
        move || {
            qllama::server::Engine::load(&model, model_dir.as_deref(), args.threads, args.ctx_size)
        }
    });
    let engine = match load.await {
        Ok(Ok(engine)) => engine,
        Ok(Err(error)) => {
            eprintln!("qllama-server: {error}");
            return 1;
        }
        Err(join_error) => {
            eprintln!("qllama-server: loader task failed: {join_error}");
            return 1;
        }
    };

    let listen: SocketAddr = match args.listen.parse() {
        Ok(addr) => addr,
        Err(error) => {
            eprintln!("qllama-server: invalid --listen {}: {error}", args.listen);
            return 1;
        }
    };

    let limits = qllama::server::Limits {
        queue_capacity: args.queue_capacity,
        request_timeout: Duration::from_secs(args.request_timeout_secs),
        max_request_bytes: args.max_request_bytes,
        kv_byte_budget: args.kv_budget_mib.saturating_mul(1024 * 1024),
    };

    let (state, scheduler) =
        match qllama::server::ServerState::start(engine, limits, args.api_key.clone()) {
            Ok(pair) => pair,
            Err(error) => {
                eprintln!("qllama-server: {error}");
                return 1;
            }
        };

    let result = qllama::server::serve(state.clone(), listen).await;
    if let Err(error) = &result {
        eprintln!("qllama-server: fatal: {error}");
    }
    // 优雅退出：释放最后一个 submit sender，等调度线程把在途请求跑完。
    // join 放在 spawn_blocking 里，避免阻塞 tokio worker 造成连接无法排空。
    drop(state);
    let drain = tokio::task::spawn_blocking(move || scheduler.join());
    match tokio::time::timeout(Duration::from_secs(args.request_timeout_secs.max(1)), drain).await {
        Ok(Ok(Ok(()))) => eprintln!("qllama-server: in-flight requests drained"),
        Ok(Ok(Err(_))) => eprintln!("qllama-server: scheduler thread panicked"),
        Ok(Err(join_error)) => eprintln!("qllama-server: drain task failed: {join_error}"),
        Err(_) => eprintln!("qllama-server: drain deadline exceeded; exiting anyway"),
    }
    match result {
        Ok(()) => {
            eprintln!("qllama-server: shut down cleanly");
            0
        }
        Err(_) => 1,
    }
}
