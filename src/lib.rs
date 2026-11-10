//! Qllama：Rust 原生的 Llama 推理引擎。
//!
//! 不链接 llama.cpp/ggml，GGUF 加载、Transformer 前向、KV Cache 与采样全部
//! 用 Rust 实现。数值行为以 llama.cpp 为对照基准（对照方法与数据见
//! docs/llamacpp-comparison.md，仓库内测试与 scripts/ 可复跑）。

// 关闭 profile 时连计时器与计数器调用也不生成，正常构建没有埋点开销。
macro_rules! profile_call {
    ($stage:ident, $body:expr) => {{
        #[cfg(feature = "profile")]
        let _span = crate::profile::Span::new(crate::profile::Stage::$stage);
        $body
    }};
}

#[cfg(feature = "profile")]
pub mod profile;

pub mod gguf;
pub mod kv_cache;
pub mod sampler;
#[cfg(feature = "server")]
pub mod server;
pub mod tensor;
pub mod tokenizer;
pub mod transformer;

pub use gguf::GgufFile;
pub use tensor::Tensor;
pub use transformer::Transformer;
