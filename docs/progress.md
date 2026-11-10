# 进度日志

按 60 分钟节奏追加 5 行摘要；每阶段结束附证据与未覆盖项。

## 2026-11-11 Phase 2（服务核心）

- 服务层落地：`src/server/mod.rs`（Engine/Session/Scheduler）+ `src/server/api.rs`
  （HTTP/SSE）+ `src/bin/server.rs`（入口），全部在 `server` feature 后面。
- ADR 0001-0004 已先行提交（5098dca）；装配路径收敛为 Engine::load →
  ServerState::start 一条线，无重复调度器生成。
- 取消语义：tokio unbounded 事件通道，Receiver drop / 取消标志 → 调度器
  下一个 decode step 停止；KV 随 session 释放。
- 资源保护：KV 按模型形状估字节（layers×kv_heads×head_dim×2×f32），超
  `--kv-budget-mib` 直接 503 不分配（Rust 分配失败是 abort，拦不住）。
- 证据：默认 228 测试全过；集成测试 7 项全过（反伪实现 / 4 并发隔离 /
  取消 ≤2s / SSE≥2 块+[DONE] / 404+400 错误码 / --api-key 401+200）；
  冒烟：3 个不同 prompt 产出 3 段不同且连贯的续写。
- 提交：5437ebd feat(server)，37a796f test(server)。

## 待办（Phase 3-4）

- Phase 3: proptest 模糊（GGUF/tokenizer，新 dev 依赖需说明）、优雅退出
  实测（SIGTERM→drain→exit 0）、KV 预算 503 实测、1000 请求压测 RSS<5%。
- Phase 4: Dockerfile/compose/systemd/config.example.json、docs/api.md、
  docs/ops-runbook.md、docs/roadmap.md、CHANGELOG、README 快速开始、--version。
- 已知偏差：KV 预算从"token 数"改为"字节数"（token 数无法换算成真实内存
  压力，见 Limits 注释）；chat 模板为降级 ChatML（ADR 0004）。
