# 能力审计（2026-11-10，服务化前基线）

本文件记录服务化改造开始时的真实能力状态，证据全部来自代码与实测，不采信文档描述。

## 1. 静态检查

| 检查 | 命令 | 结果 |
|---|---|---|
| 测试 | `cargo test --all-features` | 403 passed, 7 ignored, 0 failed |
| clippy | `cargo clippy --all-targets` | 零警告 |
| fmt | `cargo fmt --all -- --check` | 审计开始时 3 文件有差异，已修复提交 `0f12502` |

ignored 的 7 条均为真实模型门控用例（需 `QLLAMA_REAL_MODEL` 等环境变量），未设置变量时跳过，符合「不为全绿放宽断言」的纪律。

## 2. 能力表

| 模块 | 状态 | 证据 |
|---|---|---|
| GGUF v3 解析 | 有 | `src/gguf/parser.rs`，测试 `tests/gguf_parser.rs` |
| F32 权重 | 有 | `src/gguf/loader.rs` `load_tensor` |
| F16 解码 | 有 | `src/gguf/weight.rs` 手写解码，65536 项 numpy 对照表测试 `tests/gguf_loader.rs::f16_decode_matches_numpy_table` |
| Q8_0 量化（34B/block） | 有 | `src/gguf/weight.rs`，mmap 内按 block 反量化，`mapped_q8_preserves_signed_bytes_and_half_scale` 测试 |
| Q4_0 量化（18B/block） | 有 | `src/gguf/weight.rs`，低半字节=偶数下标，`mapped_q4_rows_use_separate_low_and_high_nibble_planes` 测试 |
| mmap 零拷贝 | 有 | `src/gguf/parser.rs:33` `tensor_data: Option<Arc<Mmap>>`；matmul 内逐 block 反量化，无全量 f32 常驻 |
| 量化 matmul | 有 | `src/gguf/weight.rs` `MappedWeight::matmul`，rows==1 时列分块并行（BLOCK=64） |
| 稠密 matmul 单行并行 | 有 | `src/tensor/ops.rs` `matmul_row_parallel`（commit 7607b67），`single_row_matmul_preserves_deterministic_f32_results` 测试 |
| Tokenizer 分派 | 有 | `src/tokenizer/vocab.rs:92` 按 `tokenizer.ggml.model` 分派 SPM/BPE，未知值报错 |
| SPM 分词 | 有 | scores 合并、▁ 空格语义；与 llama.cpp b12109 对照（`docs/mvp-tokenizer.md`） |
| 流式 CLI 输出 | 有 | `src/main.rs:323` 流式（默认）/缓冲/纯 id 三种输出 |
| `--ctx-size` / `--threads` | 有 | `qllama --help` 可见；CLI 测试 `tests/cli.rs` |
| 数值对照 | 有 | stories15M F32 与 llama.cpp b12109 全 75 位置 argmax/top-5 一致，最大偏差 5.198e-05（`tests/golden/llamacpp_stories15M_b12109_f32cache.tsv` 头部记录） |
| 量化 logits 对照 | 有 | `scripts/attribute_quant_deviation.py`、`docs/llamacpp-comparison.md`；Q4_0 偏差归因于反量化误差 |
| 基准设施 | 有 | `src/bin/bench.rs`（--feature bench）+ `scripts/bench_cpu.py`（1/2/4/8 线程 × ctx 128/2048，median/min/max + spread） |
| profile 埋点 | 有 | `--feature profile`，每层每阶段纳秒/调用数/分配数（`src/profile.rs`） |
| **HTTP 服务层（axum/tokio）** | **无** | Cargo.toml 无 axum/tokio；无任何 server 代码 |
| **Session/Scheduler/多请求 KV** | **无** | KV cache 为单请求全局对象（`src/kv_cache/cache.rs`） |
| **chat template** | **无** | grep `chat_template` 无命中 |
| **SSE / OpenAI API** | **无** | 无 |
| fuzz（proptest） | **无** | dev-dependencies 仅 serde_json |
| Docker / systemd / config | **无** | 无 |

## 3. 资源

- 模型（`C:/Users/qtm/AppData/Local/Temp/qllama-models/`）：
  - `stories15M.gguf`（F32, 93.8M）、`stories15M-f16.gguf`（47.3M）、`stories15M-q8_0.gguf`（25.4M）、`stories15M-q4_0.gguf`（18.2M）
  - `tinyllama-1.1b-pure-q4_0.gguf`（592M，TinyLlama-1.1B 纯 Q4_0）
  - `tiny-f32/f16/q8_0.gguf`（2.7–4.6M 测试用小模型）
- llama.cpp：`C:/Users/qtm/AppData/Local/Temp/qllama-s7/llama-server.exe`，**version 0.8.0-dev (build 12109, commit 13ec8f614)**，win-cpu-x64。可用于同机对比与门控测试。
- `QLLAMA_REAL_MODEL` 未设置；测试门控按需显式开启。

## 4. 硬件

| 项 | 值 |
|---|---|
| CPU | AMD Ryzen 7 8745H（Zen 4, Hawk Point）8 核 16 线程，3.8GHz |
| SIMD | AVX2 / FMA / AVX-512（Zen 4 支持；当前 Rust 代码未用 SIMD，为标量实现） |
| 内存 | 31.3 GB |
| OS | Windows 11 Pro 10.0.26300 |
| rustc | 1.99.1 (58cf8f9c3 2026-10-13) |

## 5. 差距清单（本轮要做）

1. **服务层完全缺失**：axum/tokio 未引入，无 Engine/Session/Scheduler 抽象，KV 单请求假设 → 阶段 2 主体。
2. **多请求 KV 隔离**：现 `KvCache` 是全局单实例，需要每请求独立实例（阶段 2.1 ADR 0001）。
3. **chat template**：无解析与应用逻辑 → 显式降级模式或 `tokenizer.chat_template` 支持（阶段 2.3）。
4. **fuzz**：proptest 未引入（dev-dependency 允许添加，属测试依赖非运行时依赖）→ 阶段 3。
5. **容器化与运维物料**：全无 → 阶段 4。
6. 单线程 decode 吞吐 0.744 tok/s、8 线程 4.04 tok/s（TinyLlama Q4_0, 592M）——服务层以此为吞吐基线；`docs/mvp-perf.md` 已记录优化瓶颈（MLP 63%）。

## 6. 决策门判定

量化（Q4_0/Q8_0 + mmap + block 反量化）与分词（SPM/BPE 分派 + llama.cpp 对照）**均已存在且通过验收**，无需执行阶段 1 补齐。CLI 流式/`--ctx-size`/`--threads` 均在。**直接进入阶段 2 服务化**。

## 7. 本轮阶段计划

| 阶段 | 内容 | 预计 |
|---|---|---|
| 2.1 | 4 份 ADR（engine/session、threading、KV 策略、API scope） | 40 min |
| 2.2–2.3 | Engine/Session/Scheduler + OpenAI 兼容 API + SSE | 2 h |
| 2.4 | 服务层测试（含「响应依赖 prompt」防假实现测试） | 40 min |
| 3 | fuzz + 优雅退出 + 资源保护 + 安全默认值 | 1.5 h |
| 4 | Docker/systemd/config/api.md/runbook/roadmap/CHANGELOG | 1 h |

时间盒外（写进 roadmap 不做）：SIMD、前缀缓存、多模型热切换、PagedAttention 块共享、GPU。
