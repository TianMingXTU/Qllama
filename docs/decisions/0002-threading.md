# ADR 0002：线程模型——tokio 管 I/O，rayon 管计算

状态：已接受　日期：2026-11-11

## 背景

一次 decode step 是 154 次同步 matmul（MLP 占 63%），单步毫秒级到秒级。tokio
的 async worker 线程被设计为"短暂阻塞即可让出"，长计算会饿死同 worker 上的
其他连接，表现为：一个慢请求卡住整个服务的 accept/超时/心跳。

## 决策

- **tokio**（rt-multi-thread）：HTTP 解析、SSE 写出、连接管理、超时计时。
  handler 里只做参数校验、channel 收发。
- **rayon 专用池**：`ThreadPoolBuilder::new().num_threads(n).build()`，容量
  `--threads` 控制。所有 `Transformer::forward_with_cache` / sampler 调用包在
  `pool.spawn_blocking`？——不。`spawn_blocking` 走 tokio 自己的阻塞池，无法
  控制 CPU 核数与优先级；改用 **`rayon::spawn` + oneshot 回传**，或同步计算
  包在 `tokio::task::block_in_place` 之外：最终选择 **调度器跑在自己的
  std::thread**，推理同步执行，结果经 mpsc channel 送回 tokio 侧。

```
tokio (axum handler) ──mpsc──▶ scheduler thread ──同步──▶ rayon pool
       ▲                            │
       └────────oneshot/mpsc────────┘  (流片段 / 完成 / 错误)
```

scheduler 线程是普通 OS 线程，阻塞毫秒级完全无害；tokio 侧的 handler 只在
channel 上 await。

## 红线落点

- 禁止在 async fn 里直接调 `forward_with_cache`（clippy 层面无法检测，靠
  review 与 ADR 约束；`docs/ops-runbook.md` 写入排查方法）。
- rayon 池线程数 = `--threads`，默认物理核数；不与 tokio worker 竞争配置。

## 备选与否决

- `spawn_async` 式 async matmul：不存在，权重计算是纯 CPU 同步代码。
- 每请求独立 rayon 池：池创建开销与线程爆炸，否决。
- `block_in_place`：要求 multi-thread runtime 且当前线程可迁移，与 rayon
  混用时 panic 风险高，否决。
