# Qllama 运维手册

面向把 qllama-server 跑在生产单机上的运维人员。

## 启动

### 最小启动

```bash
qllama-server --model /var/lib/qllama/model.gguf
# 监听 127.0.0.1:8080；--model 必须位于白名单目录内（默认取模型父目录）
```

### 配置文件

```bash
qllama-server --config /etc/qllama/config.json
```

字段与 CLI 一一对应（见 `config.example.json`）；同一参数 CLI 覆盖配置。
未知字段直接报错（`deny_unknown_fields`），防止拼写错误静默失效。

### 全量参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model` | 必填 | GGUF 路径，须在 `--model-dir` 白名单内 |
| `--model-dir` | model 父目录 | 路径白名单根 |
| `--listen` | `127.0.0.1:8080` | 对外暴露必须改 `0.0.0.0:8080` 且设 api-key |
| `--threads` | 逻辑核数 | rayon 计算池大小 |
| `--ctx-size` | 模型上限 | 单请求 prompt+生成 token 上限 |
| `--queue-capacity` | 32 | 排队上限，超过返回 429 |
| `--request-timeout-secs` | 300 | 单请求全链预算（排队+推理+传输） |
| `--max-request-bytes` | 1 MiB | 请求体上限 |
| `--api-key` | 无 | Bearer 认证，常量时间比较 |
| `--kv-budget-mib` | 2048 | 单请求 KV 字节预算，超限 503 |
| `--config` | — | JSON 配置文件 |

`--version` 输出 `qllama-server 0.1.0`。

## 容器部署

```bash
mkdir models && cp model.gguf models/
docker compose up -d --build
curl http://127.0.0.1:8080/healthz
```

compose 默认把端口绑在宿主机回环上；对外暴露需改 ports 并设置 api-key。
镜像非 root（uid 1001）、带 HEALTHCHECK（`/healthz`，30s 间隔）。

## systemd 部署

```bash
sudo useradd --system qllama
sudo install -d -o qllama /var/lib/qllama
sudo cp model.gguf /var/lib/qllama/
sudo cp target/release/qllama-server /usr/local/bin/
sudo cp deploy/qllama.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now qllama
```

单元内置 `ProtectSystem=strict` 等加固项与 `MemoryMax`；
`TimeoutStopSec` 与服务端排空上限一致（300s）。

## 优雅退出语义

`SIGTERM`（或 Ctrl+C）后：

1. axum 停止接收新连接，等待在途请求完成；
2. 最后一个 submit 通道关闭，调度线程把队列里已接受的请求**跑完**
   （不丢弃），逐请求受 `max_tokens` 约束；
3. 调度线程退出，进程以 0 退出。排空等待上限为 `--request-timeout-secs`，
   超时强制退出（stderr 提示 `drain deadline exceeded`）。

库级排空由集成测试验证（`graceful_shutdown_drains_accepted_sessions`）；
进程级信号路径在 Windows 上无法自动化，Unix 上用
`kill -TERM <pid> && time ...` 人工确认退出码 0。

## 资源保护

- **KV 字节预算**：每请求 KV ≈ `capacity × layers × kv_heads × head_dim × 2 × 4B`。
  超过 `--kv-budget-mib` 的请求直接 503，不做分配——Rust 分配失败是
  abort，必须在分配前拒绝。例：TinyLlama（22 层/6 KV 头/64 维）
  每 token ≈ 66 KiB，2048 上下文 ≈ 132 MiB。
- **队列背压**：`qllama_queue_depth >= queue-capacity` 时新请求 429 +
  `Retry-After: 1`。
- **请求体上限**：413。
- **超时**：单请求 504，超时同时取消该请求的生成。

## 容量规划

串行引擎：吞吐 ≈ 单请求吞吐。经验值（8745H Zen4 8C/16T，release 构建）：
TinyLlama Q4_0 ≈ 4 tok/s（t=8）、stories15M F32 显著更快。排队请求的
TTFT = 队列等待 + prefill。压测方法见下。

## 压测与回归

```bash
# 1000 请求 RSS 增长 <5%（数分钟，标记 --ignored）
cargo test --features server --test server_integration soak -- --ignored --nocapture
```

记录基线与复测数据到 docs/progress.md。RSS 通过 `tasklist`（Windows）
或 `/proc/<pid>/status`（Linux）采样；先 10 个请求预热再取基线。

## 故障排查

| 症状 | 排查 |
|------|------|
| 启动即退出，`outside the allowed model directory` | `--model` 不在 `--model-dir` 内；用 `--model-dir` 指向真实目录 |
| 请求 400 `no tokenizer metadata` | GGUF 缺分词器元数据；换带 tokenizer 的模型文件 |
| 全部请求 503 KV budget | `--kv-budget-mib` 过小或模型 KV 开销大；按上文公式估算调大 |
| 大量 429 | 提高队列容量或横向扩容（串行引擎无批处理收益） |
| 内存持续增长 | 先跑 soak 测试定位；已知分配路径均有界（KV 按请求容量、事件通道随 session 释放） |
| 401 | 客户端缺 `Authorization: Bearer` 或 key 不符 |

## 安全默认

- 默认监听 `127.0.0.1`——暴露公网必须显式 `--listen 0.0.0.0:8080`。
- `--model-dir` 白名单防止任意路径读取。
- `--api-key` 常量时间比较；key 只在内存，日志不落盘。
- 日志只输出启动/退出行到 stderr，不记录 prompt 与生成内容。
