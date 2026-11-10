# Changelog

## [0.1.0] — 2026-11-11

### 新增

- OpenAI 兼容 HTTP 服务层（`server` feature，`qllama-server` 二进制）：
  - 端点：`/healthz`、`/metrics`、`/v1/models`、`/v1/completions`、
    `/v1/chat/completions`
  - SSE 流式（15s keepalive、`[DONE]` 终止帧）与非流式 JSON
  - 串行调度器：FIFO 队列、429+Retry-After 背压、断连/超时取消
    （≤2s 停止，KV 随 session 释放）
  - 限制项：`--queue-capacity`、`--request-timeout-secs`、
    `--max-request-bytes`、`--kv-budget-mib`（KV 字节预算，超限 503）
  - 安全默认：仅监听回环、`--model-dir` 路径白名单、可选 `--api-key`
    （常量时间比较）、stderr 日志不含 prompt 内容
  - `--config` JSON 配置（CLI 覆盖配置；未知字段报错）
  - 优雅退出：SIGTERM/Ctrl+C → 停止接收 → 排空在途请求 → exit 0
- ADR 0001-0004（Engine/Session/Scheduler、线程模型、KV 策略、API 范围）
- 集成测试 8 项：反伪实现（响应依赖 prompt）、4 并发隔离、取消 ≤2s、
  SSE≥2 块+[DONE]、错误码、`--api-key`、KV 预算 503、优雅排空
- 模糊测试（proptest）：GGUF 解析器任意字节/截断/垃圾尾巴不 panic；
  分词器任意文本编码 id 越界检查、流式解码不 panic
- 部署交付：多阶段 Dockerfile（非 root + HEALTHCHECK + VOLUME）、
  docker-compose.yml、systemd 单元、config.example.json

### 说明

- 依赖新增（均在任务书授权或显式指定范围内）：运行时 tokio/axum/futures
  （`server` feature 门控）；测试 proptest（任务书 Phase 3 点名）。
- 引擎为串行逐请求执行（排队非 batching），文档如实声明。
- KV 预算以字节计而非 token 计：不同模型每 token KV 开销差异两个数量级，
  字节口径才能约束真实内存压力。
