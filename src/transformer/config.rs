//! Transformer configuration

use crate::gguf::{
    read_dimension, read_optional_dimension, read_positive_f32, GgufFile, GgufValue,
};

/// llama.cpp 在缺失 `attention.layer_norm_rms_epsilon` 时使用的默认值。
const DEFAULT_RMS_NORM_EPS: f32 = 1e-5;
/// llama.cpp 在缺失 `rope.freq_base` 时使用的默认值。
const DEFAULT_ROPE_THETA: f32 = 10_000.0;

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

    /// 从 GGUF 元数据提取配置。
    ///
    /// 键名按 `<general.architecture>.<...>` 拼装，与 llama.cpp 的 `LLM_KV` 表
    /// 一致。`vocab_size` 取自 `token_embd.weight` 的实际形状而非
    /// `tokenizer.ggml.tokens` 的长度：权重才是形状的唯一真相。
    ///
    /// 三个可选键缺失时取 llama.cpp 的默认值：`attention.head_count_kv`
    /// （缺失即多头注意力，等于 `attention.head_count`）、
    /// `attention.layer_norm_rms_epsilon`、`rope.freq_base`。
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let metadata = &gguf.metadata;
        let architecture = match metadata.get("general.architecture") {
            Some(GgufValue::String(value)) => value.as_str(),
            _ => return Err("Missing or invalid metadata key: general.architecture".to_string()),
        };
        let key = |name: &str| format!("{architecture}.{name}");

        let embedding = gguf
            .find_tensor("token_embd.weight")
            .ok_or_else(|| "Missing tensor: token_embd.weight".to_string())?;
        // GGUF 的 dims 按 ne[] 顺序存放，ne = {n_embd, n_vocab}。
        let embedding_dims = embedding
            .dims
            .iter()
            .rev()
            .map(|&dim| {
                usize::try_from(dim)
                    .map_err(|_| format!("token_embd.weight dimension {dim} exceeds usize"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let &[vocab_size, hidden_size] = embedding_dims.as_slice() else {
            return Err(format!(
                "token_embd.weight must be 2D [vocab_size, hidden_size], got dims {:?}",
                embedding.dims
            ));
        };

        // 词表数组与嵌入行数描述的是同一个词表，不一致时 argmax 出的 id 会映射到
        // 错误的 token 文本。llama.cpp 同样以 n_vocab 为准，不一致即拒绝加载。
        if let Some(GgufValue::Array(tokens)) = metadata.get("tokenizer.ggml.tokens") {
            if tokens.len() != vocab_size {
                return Err(format!(
                    "tokenizer.ggml.tokens has {} entries but token_embd.weight has {vocab_size} rows",
                    tokens.len()
                ));
            }
        }

        let num_attention_heads = read_dimension(metadata, &key("attention.head_count"))?;
        if !hidden_size.is_multiple_of(num_attention_heads) {
            return Err(format!(
                "embedding_length {hidden_size} is not divisible by attention.head_count {num_attention_heads}"
            ));
        }
        let num_kv_heads = match read_optional_dimension(metadata, &key("attention.head_count_kv"))?
        {
            Some(value) => value,
            None => num_attention_heads,
        };
        if num_kv_heads > num_attention_heads {
            return Err(format!(
                "attention.head_count_kv {num_kv_heads} must not exceed attention.head_count {num_attention_heads}"
            ));
        }
        if !num_attention_heads.is_multiple_of(num_kv_heads) {
            return Err(format!(
                "attention.head_count {num_attention_heads} must be divisible by attention.head_count_kv {num_kv_heads}"
            ));
        }

        // rope.dimension_count 是每个头的旋转维度。本阶段只实现整头旋转，
        // 因此它必须等于 hidden_size / head_count；不一致说明模型用了部分旋转
        // （partial RoPE），静默按整头处理会得到看不出错在哪的 logits。
        let head_dim = hidden_size / num_attention_heads;
        if let Some(rot) = read_optional_dimension(metadata, &key("rope.dimension_count"))? {
            if rot != head_dim {
                return Err(format!(
                    "rope.dimension_count {rot} does not match head_dim {head_dim}; partial RoPE is not supported yet"
                ));
            }
        }

        // 可选浮点参数：键缺失取 llama.cpp 默认值，存在则必须通过有限正数校验。
        let eps_key = key("attention.layer_norm_rms_epsilon");
        let rms_norm_eps = if metadata.contains_key(&eps_key) {
            read_positive_f32(metadata, &eps_key)?
        } else {
            DEFAULT_RMS_NORM_EPS
        };
        let theta_key = key("rope.freq_base");
        let rope_theta = if metadata.contains_key(&theta_key) {
            read_positive_f32(metadata, &theta_key)?
        } else {
            DEFAULT_ROPE_THETA
        };

        Ok(Self {
            hidden_size,
            intermediate_size: read_dimension(metadata, &key("feed_forward_length"))?,
            num_attention_heads,
            num_kv_heads,
            num_layers: read_dimension(metadata, &key("block_count"))?,
            vocab_size,
            max_position_embeddings: read_dimension(metadata, &key("context_length"))?,
            rms_norm_eps,
            rope_theta,
        })
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
