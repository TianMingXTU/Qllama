//! 采样器：把 logits 行转成下一个 token id。

mod engine;

pub use engine::Sampler;
pub use engine::SamplerConfig;
