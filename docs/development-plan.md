# Qllama 开发方案

## 1. 项目概述

### 1.1 项目定位

Qllama 是一个 Rust 原生实现的 LLM 推理引擎，目标是通过复刻 llama.cpp 的核心功能来深入理解 LLM 推理系统的设计与实现。

**核心原则**：
- **学习导向**：通过实现来理解，而非简单调用
- **正确性优先**：数值正确是性能优化的前提
- **渐进式开发**：从简单到复杂，每一步都可验证

### 1.2 技术栈

| 组件 | 技术选择 | 理由 |
|------|----------|------|
| 核心语言 | Rust 2024 | 内存安全、零成本抽象、现代工具链 |
| 并行计算 | rayon | 简单易用的数据并行框架 |
| 内存映射 | memmap2 | 高效的大文件访问 |
| 序列化 | 自定义实现 | GGUF 是二进制格式，需要手写解析 |
| 测试框架 | 内置 + proptest | 单元测试 + 属性测试 |

### 1.3 不做什么（首版）

- ❌ 不调用 llama.cpp FFI
- ❌ 不支持所有模型架构（只支持 Llama 类）
- ❌ 不先实现 GPU 后端
- ❌ 不做分布式推理
- ❌ 不支持 MoE 架构

## 2. 架构设计

### 2.1 系统架构

```
┌────────────────────────────────────────────────────────┐
│                    Application Layer                    │
│  ┌──────────────┬──────────────┬──────────────┐       │
│  │ rust-llm-cli │rust-llm-server│rust-llm-bench│       │
│  │  (CLI 交互)   │  (HTTP API)  │ (性能基准)    │       │
│  └──────────────┴──────────────┴──────────────┘       │
└────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────┐
│                  Core Inference Runtime                  │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Model Management                    │   │
│  │  ┌──────────┬──────────┬──────────┐            │   │
│  │  │ GGUF     │ Model    │ Weight   │            │   │
│  │  │ Parser   │ Registry │ Manager  │            │   │
│  │  └──────────┴──────────┴──────────┘            │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Inference Engine                    │   │
│  │  ┌──────────┬──────────┬──────────┐            │   │
│  │  │Tokenizer │Transformer│ Sampler │            │   │
│  │  │          │  Engine   │         │            │   │
│  │  ├──────────┼──────────┼──────────┤            │   │
│  │  │ KV Cache │ Context  │ Batch    │            │   │
│  │  │          │ Manager  │ Scheduler│            │   │
│  │  └──────────┴──────────┴──────────┘            │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Tensor & Compute                    │   │
│  │  ┌──────────┬──────────┬──────────┐            │   │
│  │  │ Tensor   │ Ops      │ Backend  │            │   │
│  │  │ Storage  │ (MatMul) │ (CPU)    │            │   │
│  │  └──────────┴──────────┴──────────┘            │   │
│  └─────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────┐
│                  Infrastructure Layer                    │
│  ┌──────────┬──────────┬──────────┬──────────┐        │
│  │ std/alloc│  rayon   │ memmap2  │  clap    │        │
│  │          │ (并行)   │ (mmap)   │ (CLI)    │        │
│  └──────────┴──────────┴──────────┴──────────┘        │
└────────────────────────────────────────────────────────┘
```

### 2.2 模块职责

#### 2.2.1 GGUF Parser

**职责**：解析 GGUF 二进制格式，提取元数据和权重索引。

**关键数据结构**：
```rust
pub struct GgufFile {
    header: GgufHeader,
    metadata: HashMap<String, GgufValue>,
    tensor_infos: Vec<TensorInfo>,
    tensor_data_offset: u64,
}

pub struct GgufHeader {
    magic: [u8; 4],        // "GGUF"
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
}

pub struct TensorInfo {
    name: String,
    n_dims: u32,
    dims: [u64; 4],
    ggml_type: GgmlType,
    offset: u64,           // 相对 tensor_data_start
}
```

**验收标准**：
- 能正确解析官方 llama.cpp 导出的 GGUF 文件
  —— 阶段 7 用 llama.cpp 官方工具链产物实测通过：`stories15M.gguf`（GGUF v3、
  57 个张量、18 条元数据）与 `tiny-f32.gguf` 均可加载，且由该文件构建出的模型与
  llama.cpp b12109 逐位置 logits 相差 ≤5.198e-05 —— 维度反转、张量名、超参缺省
  （如 `attention.head_count_kv` 缺失时取 `head_count`，与上游
  `llama-model.cpp:1556-1559` 一致）都得到外部验证。
- 元数据与 `gguf-dump` 工具输出一致
- Tensor 数量、名称、形状、类型正确
- 错误处理：损坏文件、无效 Magic、越界访问

#### 2.2.2 Tensor & Ops

**职责**：提供张量数据结构和基础运算。

**关键数据结构**：
```rust
pub struct Tensor {
    data: Arc<[u8]>,       // 支持内存映射和所有权两种模式
    dims: Vec<usize>,
    strides: Vec<usize>,
    dtype: DType,
}

pub enum DType {
    F32,
    F16,
    Q4_0,
    Q8_0,
    // 后期扩展
}

pub trait TensorOps {
    fn matmul(&self, other: &Tensor) -> Result<Tensor>;
    fn rms_norm(&self, weight: &Tensor, eps: f32) -> Result<Tensor>;
    fn softmax(&self, dim: usize) -> Result<Tensor>;
    fn rope(&self, positions: &[i32], freq_base: f32) -> Result<Tensor>;
    fn silu(&self) -> Result<Tensor>;
    fn mul(&self, other: &Tensor) -> Result<Tensor>;
    fn add(&self, other: &Tensor) -> Result<Tensor>;
}
```

**验收标准**：
- MatMul 与 NumPy/PyTorch 结果误差 < 1e-5
- 支持 FP32 基础运算
- 支持 2D 和 3D 张量
- 正确处理广播

#### 2.2.3 Transformer Engine

**职责**：实现 Transformer 层的前向计算。

**关键组件**：
```rust
pub struct TransformerConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_kv_heads: usize,
    num_layers: usize,
    vocab_size: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
}

pub struct Transformer {
    config: TransformerConfig,
    layers: Vec<TransformerBlock>,
    embed_tokens: Embedding,
    lm_head: Linear,
    norm: RmsNorm,
}

pub struct TransformerBlock {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

pub struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}
```

**计算流程**（单层）：
```
Input Hidden State
    ↓
RMSNorm
    ↓
Q/K/V Projection
    ↓
RoPE (Q, K)
    ↓
KV Cache Write
    ↓
Scaled Dot-Product Attention
    ↓
Output Projection
    ↓
Residual Add
    ↓
RMSNorm
    ↓
Gate/Up Projection
    ↓
SiLU(Gate) × Up
    ↓
Down Projection
    ↓
Residual Add
    ↓
Output Hidden State
```

**验收标准**：
- 单层输出与 PyTorch 参考实现误差 < 1e-4
- 支持多 Batch 输入
- 支持 GQA（Grouped Query Attention）
- 性能：单层前向 < 10ms（CPU，hidden_size=4096）

#### 2.2.4 KV Cache

**职责**：管理每个序列的历史 K/V 状态。

**关键数据结构**：
```rust
pub struct KvCache {
    layers: Vec<KvCacheLayer>,
    max_seq_len: usize,
    num_heads: usize,
    head_dim: usize,
}

pub struct KvCacheLayer {
    k_cache: Tensor,  // [max_seq_len, num_heads, head_dim]
    v_cache: Tensor,
    current_len: usize,
}

impl KvCache {
    pub fn update(&mut self, layer_idx: usize, k: &Tensor, v: &Tensor, positions: &[i32]);
    pub fn get(&self, layer_idx: usize, positions: &[i32]) -> (Tensor, Tensor);
    pub fn clear(&mut self);
}
```

**验收标准**：
- 正确处理增量更新
- 支持多序列（后期）
- 内存占用可预测
- 无内存泄漏

#### 2.2.5 Tokenizer

**职责**：文本与 Token ID 之间的转换。

**关键功能**：
- BPE/WordPiece 分词算法
- 特殊 Token 处理
- 词表加载

**验收标准**：
- 与 llama.cpp tokenization 结果一致
  —— 阶段 7 在真实模型上实测：**部分达成**。一致的部分：BOS 追加、特殊 token、
  byte/单字符 token 的词表索引（数字串 `1234567890` 的 10 个 id 与 llama.cpp 逐个相同）、
  decode 文本往返。**不一致的部分**：词片段切分——`tokenizer.ggml.model = "llama"`
  的 SPM 词表只有 `scores` 没有 `merges`，本实现按阶段 4 既定范围不做 scores 驱动的
  贪心最长匹配，退化为逐字符 + byte fallback，同一句话我们 72 个 id、llama.cpp 18 个。
  这是已文档化的范围限制（`src/tokenizer/vocab.rs` 头部已知限制第 1 条），不是缺陷；
  代价是序列长度约为 llama.cpp 的 3~4 倍。详见 `docs/llamacpp-comparison.md` §7。
- 处理 Unicode 字符
- 支持常见词表格式

#### 2.2.6 Sampler

**职责**：从 Logits 采样下一个 Token。

**关键功能**：
```rust
pub struct SamplerConfig {
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repeat_penalty: f32,
}

pub trait Sampler {
    fn sample(&mut self, logits: &Tensor, context: &[TokenId]) -> Result<TokenId>;
}
```

**验收标准**：
- 确定性采样（固定 seed）可复现
- 支持常见采样策略
- 性能：< 1ms

### 2.3 数据流

#### 2.3.1 模型加载流程

```
GGUF File
    ↓
File Header 解析
    ↓
Metadata 解析
    ↓
Tensor Info 索引构建
    ↓
权重数据映射（mmap）
    ↓
模型架构识别
    ↓
权重分配到设备（CPU）
    ↓
Transformer 实例化
```

#### 2.3.2 推理流程

```
用户输入 Prompt
    ↓
Chat Template 应用
    ↓
Tokenization
    ↓
Position 编码
    ↓
┌──────────────────────┐
│  Prefill Phase       │
│  (并行处理多个 Token) │
└──────────────────────┘
    ↓
KV Cache 填充
    ↓
获取首个 Token Logits
    ↓
┌──────────────────────┐
│  Decode Loop         │
│  ┌────────────────┐  │
│  │ 取最后一个 Token│  │
│  │     ↓          │  │
│  │ 前向计算        │  │
│  │     ↓          │  │
│  │ 更新 KV Cache  │  │
│  │     ↓          │  │
│  │ 采样下一个Token│  │
│  │     ↓          │  │
│  │ 判断是否停止   │  │
│  └────────────────┘  │
└──────────────────────┘
    ↓
Detokenization
    ↓
输出文本
```

## 3. 开发计划

### 3.1 阶段划分

#### 阶段 1：GGUF Inspector（预计 2 周）

**目标**：能够解析和检查 GGUF 文件。

**任务清单**：
1. 实现二进制读取基础设施
   - [ ] GGUF Header 解析
   - [ ] 变长字符串读取
   - [ ] 变长数组读取
   - [ ] Metadata KV 解析
   
2. Tensor Info 解析
   - [ ] Tensor 名称、形状、类型读取
   - [ ] Offset 计算
   - [ ] 对齐处理
   
3. 验证工具
   - [ ] CLI 输出元数据摘要
   - [ ] 与 gguf-dump 工具对比
   - [ ] 错误处理测试

**交付物**：
- `gguf` 模块（可独立使用）
- `gguf-inspector` CLI 工具
- 测试用例（至少 10 个不同模型）

#### 阶段 2：CPU Tensor Ops（预计 3 周）

**目标**：实现基础的张量运算。

**任务清单**：
1. Tensor 数据结构
   - [ ] 基础 Tensor 定义
   - [ ] 内存管理（Arc 共享）
   - [ ] 形状和步长计算
   
2. 基础运算
   - [ ] MatMul（naive 实现）
   - [ ] RMSNorm
   - [ ] Softmax
   - [ ] RoPE
   - [ ] SiLU
   - [ ] 逐元素运算（add、mul）
   
3. 性能优化
   - [ ] MatMul 并行化（rayon）
   - [ ] Cache-friendly 访问模式
   
4. 测试框架
   - [x] 与 NumPy/PyTorch 对比测试
     —— 落地形式是测试内独立写出的 f64 参考实现与手工常量（`tests/tensor_ops.rs`），
     阶段 7 另加两个算子级外部对照：RoPE 配对方式等于 llama.cpp
     `GGML_ROPE_TYPE_NORMAL` 的独立翻译、且不等于 HF `rotate_half`；
     端到端 logits 与 llama.cpp b12109 相差 ≤5.198e-05。
   - [x] 边界条件测试
   - [ ] 性能基准

**交付物**：
- `tensor` 模块
- `ops` 模块
- 测试套件（覆盖率 > 80%）
- 性能基准数据

#### 阶段 3：单模型前向推理（预计 4 周）

**目标**：能够对单个模型进行前向推理并验证输出。

**任务清单**：
1. 模型定义
   - [x] TransformerConfig
   - [x] Transformer 结构
   - [x] 权重加载（`gguf::loader`，仅 F32/F16；量化类型返回明确错误）
   
2. 单层实现
   - [x] Attention 模块（含 GQA、因果掩码、RoPE）
   - [x] MLP 模块（SwiGLU）
   - [x] RMSNorm 模块
   - [x] 残差连接
   
3. 完整前向
   - [x] Embedding 查找
   - [x] 层循环
   - [x] 最终 Norm 和 LM Head
   - [x] Logits 输出

4. 验证
   - [x] 单层输出对比（对独立 f64 参考实现，误差 ≤1e-4）
   - [x] 完整模型输出对比（对独立 f64 参考实现，误差 ≤1e-3）
   - [x] 不同输入长度测试（1..max_position_embeddings 逐一核对形状）
   - [x] 与 llama.cpp 逐层数值对比（需真实模型文件，留待阶段 4）
     —— 阶段 7 改做**端到端 logits 对照**（更严：一层错就全错，无需逐层夹具）。
     75 个位置 argmax 与 top-5 全部一致，最大逐值偏差 5.198e-05；
     逐层对照未做，见 `docs/llamacpp-comparison.md` §5。

**交付物**：
- `transformer` 模块
- `model` 模块
- 验证报告（误差分析）
- 示例代码

**本阶段实现要点**（阅读代码前先看这里）：

- **GGUF 维度顺序与 `Tensor` 相反**。ggml 的 `ne[0]` 是最快变化的连续维度，
  而 `Tensor` 的 `dims[0]` 是最慢外层，因此加载器必须反转 GGUF 的 dims。
  反转后权重在内存中的逻辑矩阵就是 `[out_features, in_features]`。
- **`Linear` 的权重约定**：构造时接收 `[out, in]`，内部一次性转置为 `[in, out]`，
  使前向只需一次 `matmul`。选择构造期转置而非每次前向转置，是因为后者会对每个
  token 都复制一遍权重矩阵。
- **RoPE 元素配对约定**：`ops::rope` 旋转相邻元素对 `(x[0],x[1]), (x[2],x[3]), ...`，
  即 llama.cpp 的 `GGML_ROPE_TYPE_NORM`。**阶段 7 已用上游源码与真实模型双向确认**：
  `src/llama-model.cpp:3601`（tag b12110）的 `llama_model_rope_type()` 里，
  `LLM_ARCH_LLAMA` case 起于 `:3631`、整组统一的 `return LLAMA_ROPE_TYPE_NORM;`
  在 `:3678`；`ggml/src/ggml-cpu/ops.cpp:6419-6420` 的 NORMAL 分支调用
  `rotate_pairs(n_dims, /*n_offset=*/1, ..., /*scale=*/1)`，其函数体（`:6276-6292`）
  因此旋转相邻两元素；`ggml/include/ggml.h:1910-1913` 的示例表写作
  `NORMAL → [cscscscs]`、`NEOX → [ccccssss]`。反向证据：把本实现改成 HuggingFace
  `rotate_half` 的对半拆分后，在训练过的 stories15M 上末位置 argmax 从 13 翻到 450、
  逐元素最大偏差 9.12（随机权重的微型夹具只有 5.8e-4，判别不出这类错误）。
  完整证据链见 `docs/llamacpp-comparison.md` §4 与 `src/transformer/layers.rs` 模块文档。
- **RoPE 只编码相对位置**：`R(θ_q)q · R(θ_k)k = q·R(θ_k-θ_q)k`，因此整段序列一起平移
  `start_pos` 时输出不变。接入 KV Cache 后 `start_pos` 才会真正参与计算。
- **GQA 头映射**：query 头 `h` 复用 KV 头 `h / (num_heads / num_kv_heads)`（整除分组），
  与 ggml 的 flash attention（`ik2 = iq2 / rk2`）及 HF 的 `repeat_kv` 一致。
- **`transpose`/`slice` 是复制式实现**，返回连续张量。这是正确性优先的选择，
  后续可优化为非连续 strides 视图以消除 Attention 多头路径上的拷贝。
- **本阶段不支持量化**：Q4_0/Q8_0 等类型由加载器返回明确错误。

#### 阶段 4：真实文本生成（预计 3 周）✅ 2026-11-03 完成

**目标**：能够生成连贯的文本。

**任务清单**：
1. Tokenizer ✅
   - [x] 词表加载（GGUF metadata：tokens/merges/token_type/bos/eos/unk/add_space_prefix）
   - [x] BPE 编码（min-rank 循环、byte fallback；SPM scores 驱动合并未实现——真实 Llama GGUF 无 merges，退化为逐字符+byte fallback，见 tokenizer 模块头限制说明）
   - [x] 特殊 Token 处理（Control/UserDefined/Unknown 最长匹配切分）
   
2. KV Cache ⏸ 本阶段明确不做（任务书裁剪；生成循环保持每步重算整段 O(n²)）
   - [ ] Cache 数据结构
   - [ ] 增量更新
   - [ ] 位置管理
   
3. Sampler ✅
   - [x] 温度采样（temp<=0 贪心）
   - [x] Top-k（k=0 不限制）
   - [x] Top-p（含越过阈值的候选；p>=1.0 不限制）
   - [x] 重复惩罚（llama.cpp 方向：logit<=0 乘、>0 除）
   - [x] EOS 检测（CLI 生成循环提前停止，EOS 不计入输出）
   
4. 生成循环 ✅（流式输出未做）
   - [x] Prefill 处理（--prompt 编码后一次前向）
   - [x] Decode 循环（每步重算整段，接 KV Cache 后改增量）
   - [ ] 流式输出
   
5. CLI 工具 ✅
   - [x] 参数解析（--prompt/--tokens 互斥、--generate(=--max-tokens)/--temperature/--top-k/--top-p/--repeat-penalty/--seed/--no-bos/--logits）
   - [ ] 进度显示
   - [ ] 资源监控

**交付物**：
- `tokenizer` 模块 ✅
- `sampler` 模块 ✅
- CLI 文字生成闭环 ✅（`--prompt tok5 --generate 3 --temperature 0` 端到端）
- `kv_cache` 模块 ⏸ 移到阶段 5
- `generate` 模块（独立生成器抽象）⏸ 生成逻辑暂居 CLI，接 KV Cache 时再抽出

**验收数据**（2026-11-03）：cargo test 330 通过（新增 61：sampler 31 + tokenizer 30…含 CLI 10）；cargo clippy 无新警告；cargo fmt --check 干净；真实模型 tiny-f32 `--prompt` 闭环跑通（随机权重玩具模型，输出文本无语义属预期）。

#### 阶段 5：性能与服务（后期，预计 8+ 周）

**目标**：优化性能并提供服务化能力。

**任务清单**：
1. 量化支持
   - [ ] Q4_0 解码
   - [ ] Q8_0 解码
   - [ ] 量化 MatMul
   
2. CPU 优化
   - [ ] SIMD（AVX2/AVX512）
   - [ ] 多线程优化
   - [ ] 内存布局优化
   
3. CUDA Backend
   - [ ] CUDA 绑定
   - [ ] GPU Kernel
   - [ ] 内存传输优化
   
4. 服务化
   - [x] HTTP Server（2026-11-11：OpenAI 兼容子集 + SSE + 串行 FIFO 调度 +
         取消/背压/资源保护；ADR 0001-0004，集成测试 8 项）
   - [ ] Batch 调度（当前为串行逐请求执行 = 排队，非 continuous batching；
         见 docs/roadmap.md 下一阶段 3）
   - [ ] 多序列管理（多请求 KV 隔离已完成；共享批处理待 Batch 调度）

**交付物**：
- `backend/cuda` 模块
- `server` 模块
- 性能基准报告

### 3.2 里程碑

| 里程碑 | 时间 | 交付标准 |
|--------|------|----------|
| M1: GGUF 可解析 | 第 2 周 | 能解析任意 GGUF 文件并输出元数据 |
| M2: 运算正确 | 第 5 周 | 张量运算与参考实现误差 < 1e-5 |
| M3: 单步推理正确 | 第 9 周 | 单次前向输出 Logits 误差 < 1e-4 |
| M4: 可生成文本 | 第 12 周 | 能生成连贯文本，质量接近 llama.cpp |
| M5: 性能达标 | 第 20+ 周 | 性能达到 llama.cpp 的 50%+ |

### 3.3 开发规范

#### 代码规范
- 遵循 Rust 2024 标准风格
- 使用 `cargo fmt` 格式化
- 使用 `cargo clippy` 静态检查
- 文档注释覆盖率 > 50%

#### 测试规范
- 每个公开 API 都有单元测试
- 关键路径有集成测试
- 性能关键模块有基准测试
- 使用 proptest 进行属性测试

#### 提交规范
- 遵循 Conventional Commits
- 每个 PR 只做一件事
- PR 必须通过所有测试
- PR 必须有描述性的标题和内容

## 4. 技术细节

### 4.1 GGUF 格式详解

#### 文件布局

```
+-------------------+
| Header (固定)      |  4 + 4 + 8 + 8 = 24 字节
+-------------------+
| Metadata KV       |  变长
+-------------------+
| Tensor Info List  |  变长
+-------------------+
| Alignment Padding |  对齐到 32 字节
+-------------------+
| Tensor Data Blob  |  连续权重数据
+-------------------+
```

#### 数据类型

| 类型 | 大小 | 说明 |
|------|------|------|
| GGUF_TYPE_UINT8 | 1 | 无符号 8 位整数 |
| GGUF_TYPE_INT8 | 1 | 有符号 8 位整数 |
| GGUF_TYPE_UINT16 | 2 | 无符号 16 位整数 |
| GGUF_TYPE_INT16 | 2 | 有符号 16 位整数 |
| GGUF_TYPE_UINT32 | 4 | 无符号 32 位整数 |
| GGUF_TYPE_INT32 | 4 | 有符号 32 位整数 |
| GGUF_TYPE_FLOAT32 | 4 | 32 位浮点数 |
| GGUF_TYPE_BOOL | 1 | 布尔值 |
| GGUF_TYPE_STRING | 变长 | 字符串 |
| GGUF_TYPE_ARRAY | 变长 | 数组 |
| GGUF_TYPE_UINT64 | 8 | 无符号 64 位整数 |
| GGUF_TYPE_INT64 | 8 | 有符号 64 位整数 |
| GGUF_TYPE_FLOAT64 | 8 | 64 位浮点数 |

#### 关键 Metadata

- `general.architecture`：模型架构（如 "llama"）
- `general.name`：模型名称
- `llama.context_length`：最大上下文长度
- `llama.embedding_length`：隐藏层大小
- `llama.block_count`：Transformer 层数
- `llama.attention.head_count`：注意力头数
- `llama.attention.head_count_kv`：KV 头数
- `tokenizer.ggml.model`：分词器类型
- `tokenizer.ggml.tokens`：词表

### 4.2 Transformer 计算细节

#### Attention 计算

```
给定：
  Q: [batch, num_heads, seq_len, head_dim]
  K: [batch, num_kv_heads, seq_len, head_dim]
  V: [batch, num_kv_heads, seq_len, head_dim]
  
步骤：
1. RoPE(Q, positions) → Q_rot
2. RoPE(K, positions) → K_rot
3. KV Cache 更新
4. 读取历史 K/V
5. Repeat K/V 以匹配 num_heads (GQA)
6. Scores = MatMul(Q_rot, K_rot^T) / sqrt(head_dim)
7. Mask = Causal Mask
8. Scores = Scores + Mask
9. Attn = Softmax(Scores)
10. Output = MatMul(Attn, V)
11. Output = Output Projection
```

#### MLP 计算（SwiGLU）

```
给定：
  x: [batch, seq_len, hidden_size]
  gate_proj: [intermediate_size, hidden_size]
  up_proj: [intermediate_size, hidden_size]
  down_proj: [hidden_size, intermediate_size]
  
步骤：
1. gate = MatMul(x, gate_proj^T)
2. up = MatMul(x, up_proj^T)
3. hidden = SiLU(gate) * up
4. output = MatMul(hidden, down_proj^T)
```

#### RMSNorm

```
给定：
  x: [batch, seq_len, hidden_size]
  weight: [hidden_size]
  eps: float
  
步骤：
1. rms = sqrt(mean(x^2) + eps)
2. x_norm = x / rms
3. output = x_norm * weight
```

### 4.3 性能优化策略

#### 内存访问优化
- 连续内存布局
- Cache-friendly 访问顺序
- 避免不必要的拷贝

#### 计算优化
- MatMul 并行化
- SIMD 向量化
- 循环展开
- 算子融合

#### I/O 优化
- 内存映射大文件
- 异步加载（后期）
- 缓存热数据

## 5. 质量保证

### 5.1 正确性验证

#### 数值精度测试
- 每个算子都与 NumPy/PyTorch 对比
- 误差阈值：FP32 < 1e-5，FP16 < 1e-3
- 测试用例覆盖：不同形状、边界值、极端值

#### 端到端测试
- 与 llama.cpp 输出对比
- 固定输入，确定性采样
- 文本生成质量评估

### 5.2 性能基准

#### 关键指标
- TTFT（Time to First Token）
- TPS（Tokens Per Second）
- 内存占用（RSS/VRAM）
- CPU/GPU 利用率

#### 基准场景
- 不同上下文长度（128, 512, 2048, 4096）
- 不同 Batch Size（1, 2, 4, 8）
- 不同模型大小（7B, 13B）

### 5.3 测试覆盖

| 模块 | 单元测试 | 集成测试 | 基准测试 |
|------|----------|----------|----------|
| gguf | ✓ | ✓ | - |
| tensor | ✓ | - | ✓ |
| ops | ✓ | - | ✓ |
| transformer | ✓ | ✓ | ✓ |
| tokenizer | ✓ | ✓ | - |
| kv_cache | ✓ | ✓ | - |
| sampler | ✓ | - | - |

## 6. 风险与应对

### 6.1 技术风险

| 风险 | 影响 | 应对策略 |
|------|------|----------|
| 数值精度问题 | 高 | 早期建立参考对比框架 |
| 性能差距大 | 中 | 分阶段优化，先正确后快速 |
| 量化实现复杂 | 中 | 先支持 FP32，后期加量化 |
| CUDA 绑定困难 | 中 | 充分学习现有 Rust CUDA 项目 |

### 6.2 项目风险

| 集成 | 影响 | 应对策略 |
|------|------|----------|
| 范围蔓延 | 高 | 严格按阶段推进，不做额外功能 |
| 进度延期 | 中 | 预留 buffer，及时调整计划 |
| 文档滞后 | 低 | 代码和文档同步更新 |

## 7. 参考资源

### 7.1 源码参考
- [llama.cpp](https://github.com/ggml-org/llama.cpp)：主要参考实现
- [GGUF 格式说明](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)：文件格式定义
- [GGML Backend API](https://github.com/ggml-org/ggml/blob/master/include/ggml-backend.h)：后端接口设计

### 7.2 技术文章
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [LLM 推理优化技术综述](https://zhuanlan.zhihu.com/p/651565999)
- [KV Cache 优化技术](https://arxiv.org/abs/2211.05102)

### 7.3 Rust 生态
- [candle](https://github.com/huggingface/candle)：HuggingFace 的 Rust ML 框架
- [burn](https://github.com/tracel-ai/burn)：Rust 深度学习框架
- [ndarray](https://github.com/rust-ndarray/ndarray)：Rust 多维数组库

---

**文档版本**：v1.0  
**最后更新**：2026-11-01  
**维护者**：Qllama 开发团队
