# ADR 0001：Engine / Session / Scheduler 三层抽象

状态：已接受　日期：2026-11-11

## 背景

CLI 的 `generate()`（`src/main.rs:540`）把模型、KV cache、采样器、输出全部绑在
一个函数里，KV cache 是栈上局部变量，一次调用一条生成流水线。HTTP 服务要并发
处理多个请求，这个结构直接失效。

## 决策

```
Engine            持有 Transformer + Tokenizer + 模型配置（只读，Arc 共享）
  └── Session     每请求一个：私有 KvCache + Sampler + StreamingDecoder + 生成状态
Scheduler         任务队列 + 组批窗口 + 背压；从 tokio 侧收请求，把计算派给
                  rayon 池，把结果/流片段送回请求方
```

- **Engine 不可变**：`Transformer` 内部权重在 mmap 上只读，`Arc` 共享给所有
  session，无需锁。加载一次，进程生命周期内不换模型（多模型热切换进 roadmap）。
- **Session 拥有全部可变状态**：KV、RNG、重复惩罚的 context 都在 session 内。
  并发请求之间零共享可变状态 → 并发隔离不需要任何锁，这是测试 #3（并发不串味）
  成立的结构性保证。
- **Scheduler 不做推理**：只管排队、组批、取消、背压。推理调用发生在 rayon
  worker 上，tokio 线程绝不阻塞（见 ADR 0002）。

## 为什么不能沿用 CLI 的全局单 KV

1. KV cache 形状按 `[max_seq_len, num_kv_heads, head_dim]` 预分配（TinyLlama
   ctx=2048 约 72 MiB），单实例只能服务一个请求的序列；第二个请求写入会污染
   第一个请求的位置数据。
2. `KvCache` 的 `current_len` 是单游标，无法表达两个独立序列的进度。
3. 取消语义要求释放单个请求的 KV——session 独占所有权时 drop 即释放，全局
   单例做不到。

## 不做的

- Session 池化复用（KV 重新分配的成本可接受，先不做对象池）。
- Engine 内部的原子化 metrics（调度器层统计即可）。
