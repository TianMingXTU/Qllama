//! Llama 架构前向计算：配置、层与模型装配。

mod config;
pub mod layers;
mod model;

pub use config::TransformerConfig;
pub use layers::{Attention, Embedding, Linear, Mlp, RmsNorm, TransformerBlock};
pub use model::Transformer;
