//! Sampler implementation

use crate::tensor::Tensor;

/// Sampler configuration
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repeat_penalty: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.1,
        }
    }
}

/// Sampler
pub struct Sampler {
    config: SamplerConfig,
}

impl Sampler {
    /// Create a new sampler
    pub fn new(config: SamplerConfig) -> Self {
        Self { config }
    }

    /// Sample next token from logits
    pub fn sample(&mut self, logits: &Tensor, context: &[i32]) -> Result<i32, String> {
        // TODO: Implement sampling
        Err("Sampling not yet implemented".to_string())
    }
}
