//! GGUF data types and structures

use std::collections::HashMap;

/// GGUF file header
#[derive(Debug, Clone)]
pub struct GgufHeader {
    /// Magic number, should be "GGUF"
    pub magic: [u8; 4],
    /// GGUF version
    pub version: u32,
    /// Number of tensors
    pub tensor_count: u64,
    /// Number of metadata key-value pairs
    pub metadata_kv_count: u64,
}

/// GGUF metadata value
#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

/// Tensor information
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name
    pub name: String,
    /// Number of dimensions
    pub n_dims: u32,
    /// Dimensions
    pub dims: Vec<u64>,
    /// GGML tensor type
    pub ggml_type: GgmlType,
    /// Offset relative to tensor data start
    pub offset: u64,
}

/// GGML tensor data type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    /// 仅为公开 API 兼容保留；上游已移除该类型，解析器拒绝其文件编号 4。
    Q4_2,
    /// 仅为公开 API 兼容保留；上游已移除该类型，解析器拒绝其文件编号 5。
    Q4_3,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    // K-quant 系列沿用 GGML 原始命名（Q*_K），避免与上游文档和 GGUF 类型编号对照时产生歧义。
    #[allow(non_camel_case_types)]
    Q2_K,
    #[allow(non_camel_case_types)]
    Q3_K,
    #[allow(non_camel_case_types)]
    Q4_K,
    #[allow(non_camel_case_types)]
    Q5_K,
    #[allow(non_camel_case_types)]
    Q6_K,
    #[allow(non_camel_case_types)]
    Q8_K,
    I8,
    I16,
    I32,
    I64,
    F64,
}

impl GgmlType {
    /// 返回标量类型的单元素字节数。
    ///
    /// 块量化类型和已移除的 Q4_2/Q4_3 返回 0 哨兵值，不能据此计算张量字节数；
    /// 量化存储大小还需要块内元素数量、缩放因子等布局信息。
    pub fn element_size(&self) -> usize {
        match self {
            GgmlType::F32 => 4,
            GgmlType::F16 => 2,
            GgmlType::Q4_0 => 0,
            GgmlType::Q4_1 => 0,
            GgmlType::Q5_0 => 0,
            GgmlType::Q5_1 => 0,
            GgmlType::Q8_0 => 0,
            GgmlType::Q8_1 => 0,
            GgmlType::Q2_K => 0,
            GgmlType::Q3_K => 0,
            GgmlType::Q4_K => 0,
            GgmlType::Q5_K => 0,
            GgmlType::Q6_K => 0,
            GgmlType::Q8_K => 0,
            GgmlType::I8 => 1,
            GgmlType::I16 => 2,
            GgmlType::I32 => 4,
            GgmlType::I64 => 8,
            GgmlType::F64 => 8,
            GgmlType::Q4_2 | GgmlType::Q4_3 => 0,
        }
    }

    /// 判断是否为当前支持的块量化类型；已移除的 Q4_2/Q4_3 返回 false。
    pub fn is_quantized(&self) -> bool {
        matches!(
            self,
            GgmlType::Q4_0
                | GgmlType::Q4_1
                | GgmlType::Q5_0
                | GgmlType::Q5_1
                | GgmlType::Q8_0
                | GgmlType::Q8_1
                | GgmlType::Q2_K
                | GgmlType::Q3_K
                | GgmlType::Q4_K
                | GgmlType::Q5_K
                | GgmlType::Q6_K
                | GgmlType::Q8_K
        )
    }
}

/// Model metadata extracted from GGUF
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Architecture (e.g., "llama")
    pub architecture: String,
    /// Model name
    pub name: String,
    /// Context length
    pub context_length: usize,
    /// Embedding size (hidden size)
    pub embedding_length: usize,
    /// Number of layers
    pub block_count: usize,
    /// Number of attention heads
    pub attention_head_count: usize,
    /// Number of KV heads
    pub attention_head_count_kv: usize,
    /// Layer normalization epsilon
    pub layer_norm_eps: f32,
    /// RoPE theta
    pub rope_freq_base: f32,
    /// Vocabulary size
    pub vocab_size: usize,
}

impl ModelMetadata {
    /// 提取 GGUF 元数据中的模型超参数。
    ///
    /// `general.name`、`attention.head_count_kv`、`attention.layer_norm_rms_epsilon` 与
    /// `rope.freq_base` 缺失时使用规范默认值，但这些可选字段存在却类型不符时仍报错；
    /// 模型维度必须为正且可装入 `usize`，浮点参数必须为有限正值。
    pub fn from_gguf_metadata(metadata: &HashMap<String, GgufValue>) -> Result<Self, String> {
        let get_string = |key: &str| -> Result<String, String> {
            metadata
                .get(key)
                .and_then(|v| match v {
                    GgufValue::String(s) => Some(s.clone()),
                    _ => None,
                })
                .ok_or_else(|| format!("Missing or invalid metadata key: {}", key))
        };

        // 读取模型维度：无符号或非负有符号整数均可，零与越界值直接报错。
        let get_dimension = |key: &str| -> Result<usize, String> {
            let value = metadata
                .get(key)
                .and_then(|v| match v {
                    GgufValue::Uint32(n) => Some(u64::from(*n)),
                    GgufValue::Uint64(n) => Some(*n),
                    GgufValue::Int32(n) => u64::try_from(*n).ok(),
                    GgufValue::Int64(n) => u64::try_from(*n).ok(),
                    _ => None,
                })
                .ok_or_else(|| format!("Missing or invalid metadata key: {}", key))?;
            if value == 0 {
                return Err(format!("Metadata key {} must be positive", key));
            }
            usize::try_from(value)
                .map_err(|_| format!("Metadata key {} exceeds usize range: {}", key, value))
        };

        // 读取浮点参数：必须为正有限值，防止 NaN/Inf 进入后续计算。
        let get_f32 = |key: &str| -> Result<f32, String> {
            let value = metadata
                .get(key)
                .and_then(|v| match v {
                    GgufValue::Float32(f) => Some(*f),
                    GgufValue::Float64(f) => Some(*f as f32),
                    _ => None,
                })
                .ok_or_else(|| format!("Missing or invalid metadata key: {}", key))?;
            if !value.is_finite() || value <= 0.0 {
                return Err(format!(
                    "Metadata key {} must be a positive finite number",
                    key
                ));
            }
            Ok(value)
        };

        // 可选字符串：缺失时用默认值，存在但类型不符时报错。
        let get_optional_string = |key: &str| -> Result<String, String> {
            match metadata.get(key) {
                None => Ok("unknown".to_string()),
                Some(GgufValue::String(s)) => Ok(s.clone()),
                Some(_) => Err(format!("Missing or invalid metadata key: {}", key)),
            }
        };

        // 可选浮点参数：缺失时取默认值，存在则必须为有限正数。
        let get_positive_f32_or_default = |key: &str, default: f32| -> Result<f32, String> {
            match metadata.get(key) {
                None => Ok(default),
                Some(_) => get_f32(key),
            }
        };

        let architecture = get_string("general.architecture")?;
        let name = get_optional_string("general.name")?;

        // Architecture-specific metadata
        let arch_prefix = format!("{}.", architecture);

        let context_length = get_dimension(&format!("{}context_length", arch_prefix))?;
        let embedding_length = get_dimension(&format!("{}embedding_length", arch_prefix))?;
        let block_count = get_dimension(&format!("{}block_count", arch_prefix))?;
        let attention_head_count = get_dimension(&format!("{}attention.head_count", arch_prefix))?;
        let head_count_kv_key = format!("{}attention.head_count_kv", arch_prefix);
        let attention_head_count_kv = match metadata.get(&head_count_kv_key) {
            None => attention_head_count,
            Some(_) => get_dimension(&head_count_kv_key)?,
        };

        // 官方键名位于 attention 子命名空间，写成 layer_norm_rms_epsilon 会读不到真实模型的配置。
        let layer_norm_eps = get_positive_f32_or_default(
            &format!("{}attention.layer_norm_rms_epsilon", arch_prefix),
            1e-5,
        )?;
        let rope_freq_base =
            get_positive_f32_or_default(&format!("{}rope.freq_base", arch_prefix), 10000.0)?;

        // 词表必须是字符串数组，否则分词器无法把 token ID 映射回文本。
        let tokens = metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| match v {
                GgufValue::Array(arr) if !arr.is_empty() => Some(arr),
                _ => None,
            })
            .ok_or_else(|| "Missing or invalid tokenizer.ggml.tokens".to_string())?;
        if !tokens
            .iter()
            .all(|token| matches!(token, GgufValue::String(_)))
        {
            return Err("tokenizer.ggml.tokens must contain only strings".to_string());
        }
        let vocab_size = tokens.len();

        Ok(Self {
            architecture,
            name,
            context_length,
            embedding_length,
            block_count,
            attention_head_count,
            attention_head_count_kv,
            layer_norm_eps,
            rope_freq_base,
            vocab_size,
        })
    }
}
