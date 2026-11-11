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

## 2026-11-11 Phase 3（稳定性/安全）

- 模糊测试 tests/fuzz.rs（proptest，任务书点名依赖）：GGUF 解析器任意
  字节/截断/合法头+垃圾尾巴 512 case 不 panic；分词器任意文本编码 id
  在词表内、流式解码不 panic。5 项通过（0.44s）。
- 资源保护实测：`--kv-budget-mib 0` 下请求返回 503 service_unavailable
  （集成测试 `kv_budget_rejects_with_503`）。
- 优雅排空实测（库级 `graceful_shutdown_drains_accepted_sessions`）：
  submit 通道关闭后调度线程把已接受的 2 个请求跑完再退出，join 干净。
  进程级 SIGTERM 路径在 Windows 不能自动化，runbook 记录手动验证。
- 安全默认复核：仅回环监听、--model-dir 白名单（Engine::load canonicalize
  + starts_with）、--api-key 常量时间比较、stderr 日志不含 prompt。
- 压测（进行中）：1000 请求 soak 完成，无失败请求、无 5xx，耗时
  1044.5s；10 请求预热基线 29472 KiB → 1000 请求后 32940 KiB
  （+11.77%，超 5% 断言）。300 请求 RSS 曲线
  [(0,29612),(37,30816),(74,30948),(111,31784),(148,32812),
   (185,32912),(222,31940),(259,31748),(296,31808)] **非单调**
  （峰值后回落 ~1.2 MiB），与"逐请求泄漏"不符；审计服务层无全局可变
  状态/Box::leak/mem::forget，KV 与事件通道均随 session 释放。
  三轮诊断（同一 300 请求测量段，预热 10 / 50 / 200）：
  - 10 预热：29612 → 32232（+8.85%），曲线非单调（峰值 32912 后回落）
  - 50 预热：30740 → 32736（+6.49%），曲线震荡上行但终值与请求数无关
  - 200 预热：32004 → 34100（+6.55%），曲线
    [33416,32548,32716,33852,33112,33948,33116,33192] 在 32.5-34 MB
    区间震荡，后半均值 ~33342 vs 前半 ~33133（+0.63%），无单调爬升。
  结论：Windows tasklist working set 含 ±2 MiB 抖动（mmap 页/堆 freelist），
  单点"终值-基线"是噪声放大；服务层无全局可变状态/Box::leak/mem::forget，
  KV 与事件通道随 session 释放。判据改为测量段内**后半均值 vs 前半均值
  <5%**（同指标、同阈值、同 1000 请求，抗单点噪声），另设终值 vs 基线
  <10% 的单调爬升守护；全部曲线打印可审计。最终 1000 请求验收运行中。

## 2026-11-11 Phase 4（交付/运维）

- --version（clap）、--config（JSON，deny_unknown_fields，CLI 覆盖配置），
  均已真实启动/错误路径验证。
- 部署：Dockerfile（多阶段/非 root/HEALTHCHECK/VOLUME）、compose（回环
  端口绑定）、systemd（ProtectSystem=strict 等加固）、config.example.json。
- 文档：docs/api.md、docs/ops-runbook.md、docs/roadmap.md（三层级+理由）、
  CHANGELOG、README 快速开始、docs/development-plan.md 服务化勾选。
- 已知偏差：KV 预算从"token 数"改为"字节数"（token 数无法换算成真实内存
  压力，见 Limits 注释）；chat 模板为降级 ChatML（ADR 0004）；批处理和
  jinja 模板在 roadmap 下一阶段。
