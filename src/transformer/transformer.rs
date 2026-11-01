//! Transformer model

use super::config::TransformerConfig;
use crate::tensor::Tensor;

/// Transformer model
pub struct Transformer {
    /// Model configuration
    config: TransformerConfig,
    // TODO(覃会建/qintianming511@gmail.com/666): 阶段 3 增加 embed_tokens / layers / norm / lm_head
}

impl Transformer {
    /// Create a new Transformer model
    pub fn new(config: TransformerConfig) -> Self {
        // TODO(覃会建/qintianming511@gmail.com/666): 阶段 3 初始化权重
        Self { config }
    }

    /// Get model configuration
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// Forward pass
    pub fn forward(&self, input_ids: &[i32]) -> Result<Tensor, String> {
        // TODO(覃会建/qintianming511@gmail.com/666): 阶段 3 实现
        // 1. Embedding lookup
        // 2. Layer loop
        // 3. Final norm
        // 4. LM head
        let _ = input_ids;
        Err("Forward pass not yet implemented".to_string())
    }
}
