//! Transformer configuration

/// Configuration for Transformer model
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    /// Hidden size (embedding dimension)
    pub hidden_size: usize,
    /// Intermediate size in FFN
    pub intermediate_size: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of KV heads (for GQA)
    pub num_kv_heads: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum position embeddings
    pub max_position_embeddings: usize,
    /// RMS normalization epsilon
    pub rms_norm_eps: f32,
    /// RoPE theta (base frequency)
    pub rope_theta: f32,
}

impl TransformerConfig {
    /// Create a new configuration
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        num_attention_heads: usize,
        num_kv_heads: usize,
        num_layers: usize,
        vocab_size: usize,
        max_position_embeddings: usize,
    ) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_kv_heads,
            num_layers,
            vocab_size,
            max_position_embeddings,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
        }
    }

    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Create Llama-7B like configuration
    pub fn llama_7b() -> Self {
        Self::new(
            4096,  // hidden_size
            11008, // intermediate_size
            32,    // num_attention_heads
            32,    // num_kv_heads
            32,    // num_layers
            32000, // vocab_size
            2048,  // max_position_embeddings
        )
    }

    /// Create Llama-13B like configuration
    pub fn llama_13b() -> Self {
        Self::new(
            5120,  // hidden_size
            13824, // intermediate_size
            40,    // num_attention_heads
            40,    // num_kv_heads
            40,    // num_layers
            32000, // vocab_size
            2048,  // max_position_embeddings
        )
    }
}
