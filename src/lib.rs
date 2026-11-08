//! Qllama：Rust 原生的 Llama 推理引擎。
//!
//! 不链接 llama.cpp/ggml，GGUF 加载、Transformer 前向、KV Cache 与采样全部
//! 用 Rust 实现。数值行为以 llama.cpp 为对照基准（对照方法与数据见
//! docs/llamacpp-comparison.md，仓库内测试与 scripts/ 可复跑）。

pub mod gguf;
pub mod kv_cache;
pub mod sampler;
pub mod tensor;
pub mod tokenizer;
pub mod transformer;

pub use gguf::GgufFile;
pub use tensor::Tensor;
pub use transformer::Transformer;
