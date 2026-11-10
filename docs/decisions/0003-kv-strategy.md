# ADR 0003：KV 策略——每请求独立连续 KV，按需增长

状态：已接受　日期：2026-11-11

## 决策

- 每个请求创建独立的 `KvCache`，容量 = `min(请求 max_tokens + prompt, ctx 上限)`，
  即按需分配而不是一律取模型最大上下文。
- 分配失败（`Vec` OOM / 超过全局 token 预算）→ 返回 503，不 OOM kill。
- 布局沿用现有 `[max_seq_len, num_kv_heads, head_dim]` 连续存储，`get()` 复制
  前缀的方式不改（阶段 7 的性能分析表明 KV 读取不是 t=1 瓶颈，见
  `docs/mvp-perf.md`）。

## 与 vLLM PagedAttention 的差距（如实记录）

| 维度 | 本轮 | vLLM |
|---|---|---|
| KV 存储 | 每请求连续 `[cap, kv_heads, dim]` × 2 | 物理块表（block ≈ 16 token）+ 逻辑映射 |
| 内存碎片 | 容量按请求上限预留，存在预留浪费 | 按块分配，碎片极小 |
| 前缀共享 | 无（每请求独立复制 prompt 进 KV） | 相同前缀的请求共享物理块 |
| 复制/回收 | 取消时整块 drop | 按块回收、可 copy-on-write |

## 为什么本轮不实现

1. **收益错配**：本轮吞吐瓶颈是 MLP matmul（63%，`docs/mvp-profile.md`），不是
   KV 分配或读取。PagedAttention 解决的问题在当前负载下排不进前二。
2. **风险面**：块表把 `get()` 从"复制前缀"改成"跨块 gather"，attention 内核
   的每个索引路径都要改写；数值等价性验证成本高，而红线要求逐位回归。
3. **依赖顺序**：前缀共享需要先有块表；SIMD（吞吐 2-4×）比它收益大且互不
   依赖，优先级更高。

## 触发重估的条件

- 单卡内存成为并发上限的直接约束（RSS 曲线在压测里逼近主机内存）；
- 或出现长前缀重复 prompt 的真实负载（前缀缓存的命中率可测）。

届时按 roadmap 的顺序：块表 → 前缀共享 → swap/offload。
