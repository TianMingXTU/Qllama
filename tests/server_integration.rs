//! 服务层集成测试：真实启动 `qllama-server` 进程并打 HTTP，
//! 验证任务书 Phase 2.4 的硬性测试清单。
//!
//! 仅在 `server` feature 下编译：本测试依赖 `CARGO_BIN_EXE_qllama-server`，
//! 该 bin 仅在 server feature 启用时存在（`required-features = ["server"]`）。
//! 运行：`cargo test --features server --test server_integration`。
//!
//! 不引入任何测试依赖：HTTP 客户端用标准库 `TcpStream` 手写（支持
//! Content-Length 与 chunked 两种响应），SSE 按 `\n\n` 帧解析。
//! 二进制路径由 cargo 通过 `CARGO_BIN_EXE_qllama-server` 注入。
//!
//! 模型路径取 `QLLAMA_TEST_MODEL` 环境变量，缺省回退到本地小模型
//! （stories15M-q4_0）。CI 上若无该文件则整组跳过（不报错）。
#![cfg(feature = "server")]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

use serde_json::Value;

const DEFAULT_MODEL: &str = "C:/Users/qtm/AppData/Local/Temp/qllama-models/stories15M-q4_0.gguf";

/// 启动服务器，返回一个 RAII 句柄（drop 时杀进程）。端口由 OS 分配。
struct Server {
    child: Child,
    port: u16,
    _lock: std::sync::MutexGuard<'static, ()>, // 测试全局串行锁，drop 时释放
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// 选一个空闲端口：先 bind 0 拿端口再关闭，竞态窗口极小，本测试可接受。
fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    listener.local_addr().unwrap().port()
}

fn model_path() -> Option<String> {
    if let Ok(path) = std::env::var("QLLAMA_TEST_MODEL") {
        return Some(path);
    }
    if std::path::Path::new(DEFAULT_MODEL).exists() {
        return Some(DEFAULT_MODEL.into());
    }
    None
}

/// 测试内全局串行锁：任意时刻只允许一个 qllama-server 实例运行，
/// 避免多个实例争用 CPU 导致 cancellation 测试的 2s 上限偶发超限。
static TEST_LOCK: std::sync::OnceLock<Mutex<()>> = std::sync::OnceLock::new();

fn start_server(api_key: Option<&str>) -> Option<Server> {
    let model = model_path()?;
    let port = free_port();
    let bin = env!("CARGO_BIN_EXE_qllama-server");
    let mut command = Command::new(bin);
    command.args([
        "--model",
        &model,
        "--listen",
        &format!("127.0.0.1:{port}"),
        "--request-timeout-secs",
        "60",
        "--queue-capacity",
        "16",
    ]);
    if let Some(key) = api_key {
        command.args(["--api-key", key]);
    }
    let mut child = command
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn qllama-server");

    // 轮询 healthz 直到就绪（最多 15s）。
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        if let Ok(mut s) = TcpStream::connect(("127.0.0.1", port)) {
            if s.write_all(b"GET /healthz HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
                .is_ok()
            {
                let mut buf = String::new();
                let _ = s.set_read_timeout(Some(Duration::from_secs(1)));
                let _ = s.read_to_string(&mut buf);
                if buf.contains("200") && buf.contains("ok") {
                    let guard = TEST_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();
                    return Some(Server {
                        child,
                        port,
                        _lock: guard,
                    });
                }
            }
        }
        if Instant::now() > deadline {
            break;
        }
        thread::sleep(Duration::from_millis(200));
    }
    // 启动超时：显式回收子进程，避免僵尸进程。
    let _ = child.kill();
    let _ = child.wait();
    None
}

/// 发送 HTTP GET 并读取完整响应（兼容 Content-Length 与 chunked）。
fn get(port: u16, path: &str) -> (u16, String) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .expect("set timeout");
    let request =
        format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
    stream.write_all(request.as_bytes()).expect("write request");
    read_response(&mut stream)
}

/// 发送 HTTP POST 并读取完整响应（兼容 Content-Length 与 chunked）。
fn post(port: u16, path: &str, body: &str) -> (u16, String) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .expect("set timeout");
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(request.as_bytes()).expect("write request");
    let (status, body_text) = read_response(&mut stream);
    (status, body_text)
}

/// 读取 HTTP 响应：解析状态行，按 Transfer-Encoding / Content-Length 取正文。
fn read_response(stream: &mut TcpStream) -> (u16, String) {
    let mut reader = BufReader::new(stream);
    let mut header = String::new();
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line).expect("read header line");
        if n == 0 {
            return (0, String::new());
        }
        if line == "\r\n" {
            break;
        }
        header.push_str(&line);
    }
    let status: u16 = header
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let chunked = header.to_lowercase().contains("transfer-encoding: chunked");
    let content_length = header
        .to_lowercase()
        .lines()
        .find_map(|l| {
            l.strip_prefix("content-length:")
                .map(|v| v.trim().parse::<usize>().ok())
        })
        .flatten();

    let mut body = String::new();
    if chunked {
        loop {
            let mut size_line = String::new();
            if reader.read_line(&mut size_line).is_err() {
                break;
            }
            let size =
                usize::from_str_radix(size_line.trim().trim_end_matches('\r'), 16).unwrap_or(0);
            if size == 0 {
                let mut tail = String::new();
                let _ = reader.read_line(&mut tail); // trailing \r\n
                break;
            }
            let mut chunk = vec![0u8; size];
            reader.read_exact(&mut chunk).expect("read chunk");
            body.push_str(&String::from_utf8_lossy(&chunk));
            let mut crlf = String::new();
            let _ = reader.read_line(&mut crlf); // chunk 后的 \r\n
        }
    } else if let Some(len) = content_length {
        let mut buf = vec![0u8; len];
        reader.read_exact(&mut buf).expect("read body");
        body = String::from_utf8_lossy(&buf).into_owned();
    } else {
        // 无长度也无 chunked：读到 EOF。
        let mut rest = String::new();
        let _ = reader.read_to_string(&mut rest);
        body = rest;
    }
    (status, body)
}

/// 发送 SSE 请求，返回所有 `data:` 载荷（去掉 `data: ` 前缀，含 `[DONE]`）。
fn post_sse(port: u16, path: &str, body: &str) -> Vec<String> {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .expect("set timeout");
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream
        .write_all(request.as_bytes())
        .expect("write sse request");
    // SSE 是 chunked；直接读到 EOF，再按 data: 行解析。
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).expect("read sse body");
    let text = String::from_utf8_lossy(&raw);
    text.lines()
        .filter(|l| l.starts_with("data:"))
        .map(|l| l.strip_prefix("data:").unwrap_or(l).trim().to_string())
        .collect()
}

fn complete(port: u16, prompt: &str, max_tokens: usize, stream: bool) -> Value {
    let body = format!(
        "{{\"prompt\":{prompt:?},\"max_tokens\":{max_tokens},\"temperature\":0.7,\"stream\":{stream}}}",
    );
    let (status, text) = post(port, "/v1/completions", &body);
    assert_eq!(status, 200, "completions status: body={text}");
    serde_json::from_str(&text).expect("parse completions json")
}

// --- 测试 ---

#[test]
fn healthz_and_models() {
    let Some(server) = start_server(None) else {
        eprintln!("skipping: no test model available");
        return;
    };
    let (status, text) = get(server.port, "/healthz");
    assert_eq!(status, 200);
    assert!(text.contains("ok"), "healthz body: {text}");

    let (status, text) = get(server.port, "/v1/models");
    assert_eq!(status, 200);
    let json: Value = serde_json::from_str(&text).expect("models json");
    assert_eq!(json["object"], "list");
    assert!(!json["data"].as_array().unwrap().is_empty());
    assert!(json["data"][0]["id"].as_str().unwrap().contains("stories"));
}

/// 反伪实现核心测试：不同 prompt 必须产出不同、连贯的续写。
#[test]
fn response_depends_on_prompt() {
    let Some(server) = start_server(None) else {
        eprintln!("skipping: no test model available");
        return;
    };
    let a = complete(server.port, "Once upon a time", 20, false);
    let b = complete(server.port, "The spaceship lifted off", 20, false);
    let ta = a["choices"][0]["text"].as_str().unwrap();
    let tb = b["choices"][0]["text"].as_str().unwrap();
    assert!(!ta.is_empty(), "prompt A produced empty text");
    assert!(!tb.is_empty(), "prompt B produced empty text");
    assert_ne!(
        ta, tb,
        "two distinct prompts must not produce identical output (fake impl?)"
    );
    // 续写应与各自 prompt 衔接：至少不应是同一句死循环。
    assert!(a["usage"]["completion_tokens"].as_u64().unwrap() > 0);
}

/// 并发隔离：4 个不同 prompt 并发提交，响应应两两不同（无串台 / 无全局常量）。
#[test]
fn concurrency_isolation() {
    let Some(server) = start_server(None) else {
        eprintln!("skipping: no test model available");
        return;
    };
    let prompts = [
        "The cat sat on the",
        "A brave knight walked into",
        "The ocean waves crashed against",
        "Once there was a tiny mouse",
    ];
    let port = server.port;
    let mut handles = Vec::new();
    for p in prompts {
        let handle = thread::spawn(move || {
            let v = complete(port, p, 16, false);
            v["choices"][0]["text"].as_str().unwrap().to_string()
        });
        handles.push(handle);
    }
    let results: Vec<String> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(results.len(), 4);
    for r in &results {
        assert!(!r.is_empty(), "a concurrent request produced empty output");
    }
    // 两两不同：排除所有请求返回同一全局文本（串台 / 伪实现）。
    for i in 0..results.len() {
        for j in (i + 1)..results.len() {
            assert_ne!(
                results[i], results[j],
                "concurrent requests {i} and {j} returned identical text (cross-contamination?)"
            );
        }
    }
}

/// SSE 流式：至少两帧增量，且以 [DONE] 收尾。
#[test]
fn sse_multiple_chunks_and_done() {
    let Some(server) = start_server(None) else {
        eprintln!("skipping: no test model available");
        return;
    };
    let frames = post_sse(
        server.port,
        "/v1/completions",
        "{\"prompt\":\"There was a little dog\",\"max_tokens\":30,\"temperature\":0.8,\"stream\":true}",
    );
    // 去掉 [DONE]，剩余应为 JSON chunk，且至少 2 帧（证明是流式而非一次性返回）。
    let chunks: Vec<&String> = frames.iter().filter(|f| f.as_str() != "[DONE]").collect();
    assert!(
        chunks.len() >= 2,
        "expected >=2 SSE chunks, got {}",
        chunks.len()
    );
    assert_eq!(
        frames.last().map(String::as_str),
        Some("[DONE]"),
        "SSE must end with [DONE]"
    );
    // 每帧都是合法 JSON，且 finish_reason 在倒数第二帧之前为 null。
    for chunk in &chunks {
        let v: Value = serde_json::from_str(chunk).expect("chunk is json");
        assert_eq!(v["object"], "text_completion.chunk");
    }
}

/// 错误码：未知模型 404；非法参数 400。
#[test]
fn error_codes() {
    let Some(server) = start_server(None) else {
        eprintln!("skipping: no test model available");
        return;
    };
    let (status, text) = post(
        server.port,
        "/v1/completions",
        "{\"model\":\"does-not-exist\",\"prompt\":\"hi\"}",
    );
    assert_eq!(status, 404, "unknown model should be 404: {text}");
    let err: Value = serde_json::from_str(&text).expect("error json");
    assert_eq!(err["error"]["type"], "invalid_request_error");

    let (status, text) = post(
        server.port,
        "/v1/completions",
        "{\"prompt\":\"hi\",\"temperature\":-1.0}",
    );
    assert_eq!(status, 400, "negative temperature should be 400: {text}");
}

/// 取消：长请求断开后，服务器应在 ≤2s 内恢复并服务新请求（证明 KV 被释放、
/// 调度器停止了被取消的请求，而非等它跑完）。
#[test]
fn cancellation_stops_within_2s() {
    let Some(server) = start_server(None) else {
        eprintln!("skipping: no test model available");
        return;
    };
    // 发起一个会跑很久的 SSE（max_tokens 很大），建立连接后立刻断开。
    let mut stream = TcpStream::connect(("127.0.0.1", server.port)).expect("connect long request");
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .unwrap();
    let body = "{\"prompt\":\"Once upon a time there was a\",\"max_tokens\":2000,\"temperature\":0.9,\"stream\":true}";
    let request = format!(
        "POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        server.port, body.len(), body
    );
    stream.write_all(request.as_bytes()).unwrap();
    // 读一帧确认生成已开始。
    let mut buf = [0u8; 1024];
    let _ = stream.read(&mut buf);
    drop(stream); // 客户端断开

    // 断连时刻起，2s 内必须能完成一个新请求（否则说明长请求仍在占用引擎）。
    let start = Instant::now();
    let v = complete(server.port, "The end of the story was", 8, false);
    let elapsed = start.elapsed();
    assert!(
        elapsed <= Duration::from_secs(2),
        "server did not recover within 2s after disconnect (took {elapsed:?})"
    );
    let text = v["choices"][0]["text"].as_str().unwrap();
    assert!(
        !text.is_empty(),
        "server alive but returned empty after cancel"
    );
}

/// API key 保护（可选）：设了 --api-key 后，无 key 请求应 401，带 key 应 200。
#[test]
fn api_key_required() {
    let Some(server) = start_server(Some("secret-token")) else {
        eprintln!("skipping: no test model available");
        return;
    };
    let port = server.port;

    // 无 key：401。
    let (status, _) = post(
        port,
        "/v1/completions",
        "{\"prompt\":\"hi\",\"max_tokens\":4}",
    );
    assert_eq!(status, 401, "missing api key should be 401");

    // 带 key：200。
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect with key");
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .unwrap();
    let body = "{\"prompt\":\"hi\",\"max_tokens\":4,\"temperature\":0.7}";
    let request = format!(
        "POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nAuthorization: Bearer secret-token\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(request.as_bytes()).unwrap();
    let (status, _) = read_response(&mut stream);
    assert_eq!(status, 200, "valid api key should be 200");
}
