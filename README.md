# Qllama

**llama.cpp 推理引擎的 Rust 原生复刻项目**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org/)

## 项目简介

Qllama 是一个从零开始的 Rust 实现，旨在复刻 llama.cpp 推理引擎的核心原理。项目**不使用 FFI 调用 llama.cpp**，而是使用 Rust 原生代码实现 GGUF 模型加载、Transformer 前向计算、KV Cache 管理和 Token 采样等核心功能。

### 项目理念

- **学习导向**：通过亲手实现来深入理解 LLM 推理系统的设计与实现
- **正确性优先**：数值正确是性能优化的前提，先跑通再优化
- **渐进式开发**：从简单到复杂，每一步都可验证、可测试

### 核心特性与实现范围

| 模块 | 状态 | 说明 |
|------|------|------|
| GGUF 解析 | ✅ 达成 | Header、13 种 metadata 值类型、张量索引、动态对齐、长度/类型校验；mmap 压缩权重（F32/F16/Q4_0/Q8_0 常驻磁盘布局，其余类型显式报错） |
| CLI Inspector | ✅ 达成 | `--info` 打印 metadata 摘要；`--tokens`/`--logits` 前向；`--prompt` 文字生成；`--ctx-size`/`--threads`/`--no-stream` |
| Tensor 数据结构 | ✅ 达成 | `Arc<Vec<f32>>` 存储、strides、reshape/view |
| Tensor 基础运算 | ✅ 达成 | MatMul、广播四则运算、Scale、Softmax、SiLU、RMSNorm、RoPE |
| Transformer | ✅ 达成 | Llama 前向（GQA、RoPE、SwiGLU），golden logits 对齐参考实现；GGUF 全矩阵投影走映射权重，块内反量化 |
| Tokenizer | ✅ 达成 | SPM（scores 合并）与 BPE 编解码，byte fallback，特殊 token 切分；逐 token 流式解码与一次性 decode 逐字符一致 |
| Sampler | ✅ 达成 | 重复惩罚→温度→top-k→top-p→抽样，`--seed` 可复现，temp=0 贪心 |
| KV Cache | ✅ 达成 | `update`/`get` 增量读写；prefill + 逐 token decode；容量随 `--ctx-size`，峰值 RSS ≈ 权重文件体积 |
| 量化支持 | ✅ 达成 | Q4_0 / Q8_0 映射权重，matmul 内按 32 元素块反量化，不全量反量化常驻（Q6_K 等显式报错） |
| 流式输出 | ✅ 达成 | 默认逐 token 输出（跨 token 多字节缓冲不乱码）；停止原因显式报告 |
| CUDA 后端 | 📋 规划中 | GPU 加速（明确不在 MVP 范围） |

> 图例：✅ 目标达成 | 🚧 部分可用 | 📋 规划中
>
> 文字生成闭环已通：`qllama --model <gguf> --prompt "文本" --generate N` 走 编码→前向→采样→流式解码 全链路。

## 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                   应用层 (CLI / Bench)                   │
├─────────────────────────────────────────────────────────┤
│                  核心推理 Runtime                         │
│  ┌──────────┬──────────┬──────────────┬──────────────┐  │
│  │   GGUF   │Tokenizer │ Transformer  │   Sampler    │  │
│  │  Parser  │          │    Engine    │              │  │
│  ├──────────┼──────────┼──────────────┼──────────────┤  │
│  │ KV Cache │  Tensor  │   Backend    │   Config     │  │
│  │          │   Ops    │    (CPU)     │              │  │
│  └──────────┴──────────┴──────────────┴──────────────┘  │
├─────────────────────────────────────────────────────────┤
│        Rust 基础库 (std / rayon / memmap2)               │
└─────────────────────────────────────────────────────────┘
```

详细的架构设计请参阅 [docs/architecture.md](docs/architecture.md)。

## 与 llama.cpp 的对比

| 方面 | llama.cpp | Qllama |
|------|-----------|--------|
| 实现语言 | C/C++ | Rust |
| 内存安全 | 手动管理 | 所有权系统 |
| 错误处理 | 返回码 | `Result` / `Option` |
| 并行模型 | 线程 + 原子操作 | `rayon` 数据并行 |
| 计算图 | GGML Graph | 先 eager，后期引入图 |
| 张量共享 | 引用计数 | `Arc` 零拷贝 |

### 数值已与 llama.cpp 对照（阶段 7）

对照对象是官方预编译的 llama.cpp **b12109**（commit `13ec8f614`）与训练过的
llama 架构模型 `stories15M.gguf`（F32，6 层，32000 词表）。两侧使用**同一串固定
token id**，因此结论不受分词差异影响：

| 对照项 | 实测结果 |
|--------|----------|
| 末行 logits argmax | 75/75 个位置一致 |
| 末行 logits top-5 | 75/75 一致（含排序）；冻结的 6 个位置 top-16 排序也一致 |
| 逐值偏差（各减最大值） | 最大 5.198e-05，平均 6.550e-06 |
| 贪心生成 token 链 | 16/16 个 token 逐位一致 |
| RoPE 配对方式 | 与 llama.cpp `GGML_ROPE_TYPE_NORMAL` 一致；与 HF `rotate_half` 不一致 |

两个必须知道的限定：① 与 llama.cpp **默认配置**对照会得到 ~1e-2 的偏差，那是 llama.cpp
把 KV cache 默认存成 f16（`-ctk/-ctv`）造成的，取精确参考需加 `-ctk f32 -ctv f32`；
② 分词粒度不同——SPM 词表（只有 `scores` 没有 `merges`）我们退化为逐字符 +
byte fallback，同一句话 72 个 id 对 llama.cpp 的 18 个。
方法与逐条命令见 [docs/llamacpp-comparison.md](docs/llamacpp-comparison.md)。

### 数值已与 llama.cpp 对照（MVP，2026-11-08）

MVP 里程碑在 stories15M（F32、Q4_0、Q8_0）与 TinyLlama-1.1B（纯 Q4_0）上复核：

| 对照项 | 实测结果 |
|--------|----------|
| SPM 分词逐 id | 31/31 组真实 prompt 与 llama.cpp b12109 完全一致；roundtrip 30/31（唯一差异是字面 `▁` 与空格的归一化碰撞，有同 id 孪生证据） |
| F32 logits 逐值 | 75 个 id、全部 75 位置全词表对照（2,400,000 个值），最大偏差 5.20e-05、平均 6.55e-06，超容差 0（≤1e-4 达标） |
| Q4_0 / Q8_0 logits | 逐值偏差 0.2–0.5 量级；argmax 绝大多数一致。四象限归因显示主要来自 llama.cpp 量化 GEMV 把 f32 激活一并量化到 q8_0，不是本项目的反量化错位 |
| 贪心生成 token 链 | 16/16 个 token 逐位一致 |
| decode 速度 | TinyLlama-1.1B 纯 Q4_0 4.13 tok/s（8 线程，≥1 tok/s 达标；7B 未测，本地无权重） |
| 内存 | TinyLlama Q4_0 峰值工作集 565 MiB ≈ 权重文件 592 MiB（无全量反量化） |
| 流式验收 | 真实用户视角 6 项全过：逐 token 到达管道、UTF-8 无损、seed 复现、预算显式、threads 无影响、PE 导入表无 llama.cpp |

证据与复跑命令：分词见 `docs/mvp-tokenizer.md`；数值与归因见 `docs/mvp-quantization.md`；
CLI 验收见 `docs/mvp-baseline.md` 第 9 节。

## 开发路线图

### 阶段 1：GGUF Inspector + Tensor 基础运算 ✅
- [x] 项目初始化与模块划分
- [x] GGUF 数据结构定义（`GgufValue` / `GgmlType` / `TensorInfo` / `ModelMetadata`）
- [x] Header / Metadata / Tensor Info 解析
- [x] 动态对齐值读取与越界校验
- [x] 字符串长度上限与不可信计数的预分配防护
- [x] `ModelMetadata` 导出与模型超参数提取
- [x] 与上游源码核对 GGUF 版本、对齐、重复键/名规则与 GGML 类型编号
- [ ] 与 `gguf-dump` 工具在真实模型上输出对照
- [x] 与 llama.cpp 官方工具链产物在真实模型上对照（阶段 7）：`stories15M.gguf`
      可加载，且由此构建的模型与 llama.cpp b12109 逐位置 logits 相差 ≤5.198e-05

### 阶段 2：CPU Tensor Ops ✅
- [x] MatMul（二维，行内 `i-k-j` + 输出行 Rayon 并行）
- [x] RMSNorm / Softmax / RoPE / SiLU
- [x] 逐元素运算与 NumPy 风格广播
- [x] 与 llama.cpp 上游源码逐项核对：RoPE 交错配对、RMSNorm 公式、SwiGLU、张量命名、metadata 键名
- [ ] 与 NumPy/PyTorch 数值对比测试（本地未安装，当前为手算值 + 独立 `f64` 参考）
- [x] RoPE 配对方式已用 llama.cpp 与真实模型双向判定（阶段 7）：等于上游
      `GGML_ROPE_TYPE_NORMAL`（相邻对 `[cscscscs]`），不等于 HF `rotate_half`
      （对半拆分 `[ccccssss]`；改错后 stories15M 末位置 argmax 翻转、偏差 9.12）
- [ ] SIMD 与分块 GEMM 优化

### 阶段 3：单模型前向推理 ✅
- [x] Embedding / Attention / MLP / RMSNorm 层
- [x] 完整前向计算与 Logits 输出
- [x] 与独立 `f64` 参考实现误差 < 1e-3（测试断言；真实夹具增量路径偏差 0.0）

### 阶段 4：真实文本生成 ✅
- [x] Tokenizer 集成（BPE + SPM scores 合并）
- [x] KV Cache 增量更新（阶段 5 交付，见下）
- [x] Sampler 与自回归循环
- [x] CLI 流式输出（M4：逐 token 到达，跨 token 多字节缓冲）

### 阶段 5：性能与服务
- [x] KV Cache 增量推理：`--generate` 改 prefill + 逐 token decode，每步代价 O(n²)→O(n)
- [x] Q4_0 / Q8_0 量化格式（M2：映射权重 + 块内反量化，M3：并行 matmul）
- [x] decode 性能达标：TinyLlama 1.1B Q4_0 4.13 tok/s（≥1 tok/s）
- [ ] CPU SIMD 优化
- [ ] CUDA Backend
- [ ] HTTP Server 与 Continuous Batching

### 阶段 7：真实模型与 llama.cpp 数值对照 ✅ 2026-11-04 完成
- [x] 取得独立 oracle：llama.cpp b12109 官方预编译包（本机无 C++ 工具链，未自行构建）
- [x] RoPE 配对方式判定：上游源码 file/function/line + 双向实测（ours ≡ llama.cpp，≠ HF）
- [x] 端到端 logits 对照（固定 token id 以隔离分词）：75/75 位置 argmax 与 top-5 一致，
      最大逐值偏差 5.198e-05
- [x] 贪心 token 链对照：16/16 逐位一致
- [x] 分词器对照与归类：byte-token/BOS/decode 往返一致；M1 已实现 SPM scores 合并并与 llama.cpp 逐 id 全等（31/31）
- [x] 回归基准冻结 `tests/golden/`（记录来源与生成命令，**是回归基准、不是 oracle**）
- [x] 1e-2 量级残差根因定位在 llama.cpp 默认 f16 KV cache（`-ctk/-ctv`），非本项目缺陷
- [x] 量化对照：F32 达标 1e-4 后，Q4_0/Q8_0 偏差单独记录并四象限归因到 oracle 的激活量化
- [x] F16 权重对照（MVP）：ours-vs-oracle 4.2e-3–1.3e-2，归因到 F16 权重表示误差与
      oracle 的 f16 激活转换（`ggml-cpu.c:225-229`），见 `docs/mvp-quantization.md`
- [ ] 未发现属于 Qllama 的数值缺陷，故本阶段数值实现零改动
- [ ] GQA 分组映射仍未取得外部对照（F16/量化已在 MVP 阶段完成对照）

对照方法、命令与全部实测数据见 `docs/llamacpp-comparison.md`；
可复跑的对照工具在 `scripts/compare_logits.py` 与 `scripts/numpy_forward_reference.py`。

## 快速开始：HTTP 服务（OpenAI 兼容）

> 服务层为 `server` feature，默认 `cargo build` 不会编译 tokio/axum。

```bash
# 构建（需要 GGUF 模型文件，如 stories15M-q4_0.gguf）
cargo build --release --features server --bin qllama-server

# 启动：默认仅监听 127.0.0.1:8080
./target/release/qllama-server --model /path/to/model.gguf

# 文本补全
curl -s http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Once upon a time","max_tokens":24,"temperature":0.7}'

# 流式（SSE）
curl -N http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Once upon a time","max_tokens":24,"stream":true}'

# 容器部署
mkdir models && cp model.gguf models/ && docker compose up -d --build
```

引擎是**串行逐请求**执行（排队，不是 continuous batching）；详细参数、
限制项、错误码见 [docs/api.md](docs/api.md) 与
[运维手册](docs/ops-runbook.md)，路线见 [docs/roadmap.md](docs/roadmap.md)。

## 项目结构

```
Qllama/
├── Cargo.toml              # 项目配置
├── README.md               # 项目文档
├── CONTRIBUTING.md         # 贡献指南
├── .gitignore
├── docs/                   # 文档与架构图
│   ├── development-plan.md # 开发方案
│   ├── architecture.md     # 架构设计
│   └── rust_llama_inference_architecture.html
├── tests/                  # 集成测试
│   ├── gguf_parser.rs      # GGUF 解析
│   ├── tensor_ops.rs       # Tensor 运算
│   └── cli.rs              # CLI 用户路径（含 GGUF 夹具生成）
└── src/
    ├── lib.rs              # 库入口
    ├── main.rs             # CLI 入口
    ├── gguf/               # GGUF 解析
    │   ├── parser.rs       # 索引解析
    │   ├── loader.rs       # 张量数据解码
    │   └── types.rs        # 数据类型
    ├── tensor/             # 张量运算
    │   ├── data.rs         # Tensor 结构
    │   └── ops.rs          # 运算接口
    ├── transformer/        # Transformer
    │   ├── config.rs       # 配置
    │   ├── model.rs        # 模型主体
    │   └── layers.rs       # 层实现
    ├── kv_cache/           # KV Cache
    ├── tokenizer/          # 分词器
    └── sampler/            # 采样器
```

## 构建与运行

### 环境要求

- Rust stable（edition 2021）；本阶段使用 Rust/Cargo 1.99.1 验证，尚未验证最低支持版本。
- CPU 基础实现不要求 AVX2；手写 SIMD 优化尚未实现。

### 编译

```bash
cargo build --release
```

### 检查与测试

```bash
cargo check          # 类型检查
cargo clippy         # 静态检查
cargo fmt --check    # 格式检查
cargo build          # 编译
cargo test           # 运行测试
```

本阶段（KV Cache 增量推理）验证结果：`cargo test` 357 个测试全部通过（其中 CLI 集成测试 23 个驱动真实可执行文件，覆盖 `--prompt` 编码生成、采样参数、seed 复现、EOS 停止、贪心序列与整段路径逐 token 一致）；`cargo fmt --all -- --check` 通过；`cargo clippy --all-targets` 剩余警告均来自 `backend` 等既有骨架模块，KV Cache、Sampler、Tokenizer 与 CLI 相关代码为 0 警告。增量路径与整段路径的数值一致性由 `tests/transformer_forward.rs` 的对照测试锁定（多长度、多切分、GQA、f64 独立参考实现四类角度），并经过 4 处变异注入验证测试有效。

本阶段（真实模型与 llama.cpp 数值对照）验证结果：`cargo test` 366 个测试全部通过、
7 个 `#[ignore]`（需要真实模型，用 `QLLAMA_REAL_MODEL` 显式开启后 7 个全部通过）；
`cargo fmt --all -- --check` 通过；`cargo clippy --all-targets` 警告数与既有基线一致
（7 条，全部来自 `backend`/`sampler` 等骨架模块与既有测试写法，本阶段新增代码 0 警告。
2026-11-08 技术债清理后该基线已降为 **0 条**：`backend` 空抽象删除、`module_inception`
改名、测试改用 `expect_err`）。
新增 7 个对照/判别用例：`tests/tensor_ops.rs` 的 2 个 RoPE 配对方式判别（等于上游
`GGML_ROPE_TYPE_NORMAL` 的独立翻译、且不等于 HF `rotate_half`），`tests/llamacpp_golden.rs`
的 5 个（golden 结构与来源自洽、position 语义、逐值偏差与 top-16 排序、16 步贪心链、
分词差异归类）。外部对照数据冻结在 `tests/golden/`（含模型 sha256、llama.cpp 构建号、
逐字生成命令），**仅作回归基准、不是 oracle**。本阶段未发现属于 Qllama 的数值缺陷，
数值实现零改动，只维护了 RoPE 与范围说明的注释。

### 使用（当前状态）

```bash
# 查看模型信息
cargo run --release -- --model model.gguf --info

# 显式 token id 前向（模型缺分词器元数据时的入口）
cargo run --release -- --model model.gguf --tokens 1,2,3 --generate 4 --temperature 0

# 文字生成闭环：编码 → 前向 → 采样 → 解码
cargo run --release -- --model model.gguf --prompt "你好" --generate 20 --temperature 0.8 --seed 42

# 贪心生成（temperature 0），前 5 个候选 logit 固定展示
cargo run --release -- --model model.gguf --prompt "Once upon a time" --max-tokens 10 --temperature 0

# 把最后位置的 logits 写文件（每行一个 f32，token id 序）
cargo run --release -- --model model.gguf --tokens 1,2 --logits logits.txt
```

参数说明：`--prompt` 与 `--tokens` 互斥，二者必给其一（`--generate`/`--logits` 依赖输入）；`--generate N`（别名 `--max-tokens`）是采样生成的上限，提前遇到 EOS 会停止且 EOS 不计入输出；`--top-k`/`--top-p`/`--repeat-penalty` 是采样参数；`--seed` 使输出可复现；`--no-bos` 不在编码结果头部加 BOS。分词器元数据缺失的模型上 `--prompt` 会报错并提示改用 `--tokens`。

MVP 新增参数：默认输出为逐 token 流式（正文边生成边写 stdout，跨 token 的多字节字符完整后再吐出；生成结束后打印 `=== Generation summary ===` 与统计）。`--no-stream` 回到一次性缓冲输出（正文在 `text:` 行中）。`--ctx-size N` 同时是 KV cache 容量与生成预算（默认取模型标称 context_length；超过模型标称值、或 prompt 加生成填满预算都会显式报错并给出建议）；`--threads N` 指定 Rayon 并行线程数（默认沿用全局池；线程数只影响速度，不改变输出字节，实测 1 与 4 线程 stdout 逐字节一致）。停止原因在摘要中显式报告：达到 `--generate` 上限、遇到 EOS、或触及 context 预算。

### 使用 Tensor 基础运算

```rust
use qllama::tensor::{Tensor, TensorOps};

fn main() -> Result<(), String> {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let bias = Tensor::new(vec![10.0, 20.0], vec![2]);
    let y = x.add(&bias)?;
    assert_eq!(y.data(), &[11.0, 22.0, 13.0, 24.0]);

    let probabilities = y.softmax(1)?;
    assert_eq!(probabilities.dims(), &[2, 2]);
    Ok(())
}
```

算子包含二维 MatMul、广播四则运算、Scale、任意轴 Softmax、SiLU、RMSNorm 和交错配对 RoPE。参数约束、空张量及非有限值处理见 [Tensor API 说明](docs/architecture.md#22-tensor--ops)。这些基础算子尚未串成完整的模型推理流程。

## 设计原则

1. **正确性优先**：先验证数值精度，再优化性能。每个算子都与独立参考实现对比。
2. **最小化范围**：首版只支持单一 Llama 类架构，避免范围蔓延。
3. **可测试性**：每个公开 API 都有单元测试，关键路径有集成测试。
4. **渐进式复杂度**：CPU → SIMD → GPU；单序列 → 多序列 → 服务化。
5. **无伪实现**：不硬编码结果，不静默吞异常，未验证不声称完成。

### 首版刻意不做

- ❌ 不调用 llama.cpp FFI（避免把推理核心当黑盒）
- ❌ 不支持所有模型架构
- ❌ 不先写 GPU Backend
- ❌ 不做分布式推理与 MoE

## 参考资源

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — 主要参考实现
- [GGUF 格式说明](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [GGML Backend API](https://github.com/ggml-org/ggml/blob/master/include/ggml-backend.h)
- [llama-context.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-context.cpp)
- [Llama 模型计算图](https://github.com/ggml-org/llama.cpp/blob/master/src/models/llama.cpp)

## 文档

- [开发方案](docs/development-plan.md) — 阶段划分、任务清单、验收标准
- [架构设计](docs/architecture.md) — 模块职责、API 契约、实现进度与已知问题
- [第一阶段开发记录](docs/phase-one-development.md) — 实施依据、设计取舍与验证结果
- [贡献指南](CONTRIBUTING.md) — 开发流程、代码规范、提交规范

## 许可证

MIT License

## 致谢

本项目受 [llama.cpp](https://github.com/ggml-org/llama.cpp) 启发，旨在学习 LLM 推理引擎的核心原理。感谢 llama.cpp 社区的开源贡献。
