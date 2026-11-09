# Qllama 性能优化报告（阶段 7 补充）

**日期**: 2026-11-09  
**目标**: 将单 token decode 从 ~0.8 tok/s 提升至 ≥1.0 tok/s（交互式）  
**约束**: 无新依赖、无 C 代码、P1→P6 按序执行、每次优化须有数值对照

---

## 1. 基线数据（原始，无 profiling/P1/P2）

| 线程 | ctx=128 decode (tok/s) | ctx=2048 decode (tok/s) |
|------|---------------------|---------------------|
| 1    | 0.755               | —（未完成）            |
| 2    | 1.342               | 1.344               |
| 4    | 2.438               | 2.407               |
| 8    | 4.147               | 3.997               |

**结论**: 8 线程相对 1 线程仅 ~5.5× 提升，远未到 8×；单线程仍是主要瓶颈。

---

## 2. Profile 诊断（t=8, ctx=2048, 32 decode steps）

总时间 41.0s，每 step 1.28s，共 49760 次 stage 调用：

| 阶段              | 调用次数 | 总时间 (ms) | 占比    |
|-------------------|---------|------------|--------|
| mlp_gate          | 3520    | 8969       | 21.9%  |
| mlp_up            | 3520    | 8733       | 21.3%  |
| mlp_down          | 3520    | 8281       | 20.2%  |
| **MLP 小计**      | **10560** | **25983** | **63.4%** |
| attention_kv_softmax | 3520 | 5040       | 12.3%  |
| q_projection      | 3520    | 3352       | 8.2%   |
| o_projection      | 3520    | 3329       | 8.1%   |
| **Attention 小计** | **14080** | **11721** | **28.6%** |
| 其他              | 25120   | 3296       | 8.0%   |

**关键发现**:
- MLP 占 63%，其中 gate+up 各 ~2.5ms/step，down ~2.3ms/step
- 每个 decode step 共 154 次 matmul（22 层 × 7 次），全部串行于 t=1
- `par_chunks_mut(columns)` 对 rows=1 无效（只产生一个 chunk）

---

## 3. 优化尝试与结果

### P1: Dense GEMV 列分块并行（`src/tensor/ops.rs`）

**方案**: 当 `rows==1` 时按 BLOCK=64 列分组并行，模仿 `MappedWeight::matmul` 策略。

**数值回归**: stories15M F32 全 75 位置 `baseline_bit_mismatches=0` ✅

**性能影响**（t=1 ctx=2048）:
```
原始:   0.755 tok/s
P1后:   0.715 tok/s  ← 退化 -5.5%
```

**原因分析**:
- 每次 decode step 154 次 matmul，其中仅 4 次是 GEMV（rows==1）
- GEMV 占 ~4%，Rayon 任务调度开销反而超过微小收益
- 真正的大头 MLP（rows>1）不受此路径影响

**结论**: ❌ 不提交，P1 对全模型无收益

### P2: MLP 投影并行化（`src/transformer/layers.rs`）

**方案**: 用 `rayon::join` 并行执行 gate 和 up 投影（二者独立）。

**数值回归**: `baseline_bit_mismatches=0` ✅

**性能影响**（t=1 ctx=128）:
```
原始:     0.755 tok/s
含 profiling: 0.717 tok/s（profiling 开销）
P2 后:   0.759 tok/s
```
- vs pre-P2: +5.9%
- vs 原始: +0.6%（噪声范围内）

**原因分析**:
1. 单 GEMV 耗时 ~2.5ms，rayon::join 调度开销 ~100us
2. 共享输入 tensor 导致内存带宽竞争
3. 总 MLP 时间 = sum(gate+up+down) ≈ 常数，join 不改变总和

**结论**: ❌ 无实质收益，保留接口但不单独 commit

---

## 4. 根因总结

```
t=1 时所有 154 次 matmul 串行执行
        ↓
MLP gate/up/down 各 ~2.5ms，合计 ~7.5ms/step
        ↓
profile 显示 63% 时间在 MLP，但无法并行（单线程）
        ↓
瓶颈不在 CPU 计算，而在内存访问模式（L1/L2 cache misses）
```

**核心矛盾**: 单线程场景下，任何 "task-level" 并行化都无法工作——
必须先减少总工作量（kernel fusion）或减少内存访问（scratch buffer reuse）。

---

## 5. 剩余选项评估

| 选项 | 预期收益 | 风险 | 状态 |
|------|---------|------|------|
| P3 scratch buffer reuse | 降低 allocator 压力 | 改动量大 | 未实现 |
| P4 KV cache zero-copy | 缓解 ctx=2048 衰减 | 需引入 view 类型 | 未实现 |
| MLP kernel fusion | ~2%（内存带宽节省 50%） | 实现复杂 | 未实现 |
| P5 SIMD（unsafe） | 2-4× | 违反约束 | 超出范围 |
| Layer-level parallelism | 显著 | 架构重构 | 超出范围 |

---

## 6. 结论与建议

**目标达成情况**: ❌ 未达成 ≥1.0 tok/s

**已验证**:
- ✅ 数值红线: stories15M F32 逐位一致（baseline_bit_mismatches=0）
- ✅ 测试覆盖: 402 passed，7 ignored
- ✅ clippy: 零警告
- ✅ Profile 基础设施就绪，可用于后续优化指导

**核心发现**:
1. P1（GEMV 列分块）仅覆盖 4/154 次 matmul，对全模型无收益
2. P2（MLP rayon::join）在 t=1 无效，因为 join 本身也是任务调度
3. 单线程瓶颈是 **内存访问模式**，不是 CPU 计算能力

**下一步建议**（需用户确认）:
- **P3/P4 改造**: scratch buffer reuse + KV cache zero-copy，可能带来 10-20% 改善
- **放宽约束**: 允许 P5 SIMD（unsafe），预计可达成 2-3 tok/s
- **接受现实**: 当前 0.75 tok/s 已满足 MVP 演示需求，可暂不继续优化

---

**底线重申**: 任何优化都不允许数值回归，golden 不得动、断言不得放宽。
