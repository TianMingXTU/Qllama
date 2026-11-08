//! `Transformer::from_gguf` 的真实模型路径测试。
//!
//! 数值回归用例读取 `tests/fixtures/tiny-llama.gguf`——一个由
//! `tests/fixtures/generate_tiny_llama.py` 生成的微型 Llama GGUF。期望值不是
//! 从被测代码反推的，而是由独立实现的 Python 参考前向算出后固化下来的常量，
//! 因此这一组用例真正约束的是"加载 + 前向"整条链路，而不是某个实现的现状。
//!
//! 错误路径用例用文件末尾的最小 GGUF 写入器现造畸形文件，避免为每个失败场景
//! 都提交一份夹具。

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use qllama::gguf::GgufFile;
use qllama::transformer::{Transformer, TransformerConfig};

/// 仓库内的微型 Llama 夹具，`CARGO_MANIFEST_DIR` 保证从任意工作目录都能定位。
fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("tiny-llama.gguf")
}

fn load_fixture() -> GgufFile {
    let path = fixture_path();
    GgufFile::from_path(&path).unwrap_or_else(|error| panic!("解析 {:?} 失败: {error}", path))
}

/// 夹具模型的 token id 序列，与生成期望值时使用的输入一致。
const TOKENS: [i32; 3] = [3, 17, 42];

#[test]
fn from_gguf_reads_the_llama_configuration_from_metadata() {
    let gguf = load_fixture();
    let model = Transformer::from_gguf(&gguf).expect("装配夹具模型失败");
    let config = model.config();

    assert_eq!(config.hidden_size, 16);
    assert_eq!(config.intermediate_size, 32);
    assert_eq!(config.num_attention_heads, 4);
    // 夹具刻意使用 GQA（kv 头数是 query 头数的一半），否则这条路径不会被覆盖。
    assert_eq!(config.num_kv_heads, 2);
    assert_eq!(config.num_layers, 2);
    assert_eq!(config.vocab_size, 64);
    assert_eq!(config.max_position_embeddings, 128);
    assert_eq!(config.rms_norm_eps, 1e-6);
    assert_eq!(config.rope_theta, 10000.0);
    assert_eq!(config.head_dim(), 4);
}

#[test]
fn config_from_gguf_requires_rms_norm_epsilon() {
    let model = model_with(|builder| {
        builder.without_metadata("llama.attention.layer_norm_rms_epsilon");
    });
    let error = TransformerConfig::from_gguf(&model).expect_err("缺少 RMSNorm epsilon 时应报错");
    assert!(
        error.contains("llama.attention.layer_norm_rms_epsilon"),
        "错误信息应指出缺失的 epsilon 键，实得: {error}"
    );
}

#[test]
fn from_gguf_requires_rms_norm_epsilon() {
    let model = model_with(|builder| {
        builder.without_metadata("llama.attention.layer_norm_rms_epsilon");
    });
    let error = Transformer::from_gguf(&model).expect_err("缺少 RMSNorm epsilon 时应拒绝加载");
    assert!(
        error.contains("llama.attention.layer_norm_rms_epsilon"),
        "错误信息应指出缺失的 epsilon 键，实得: {error}"
    );
}

#[test]
fn from_gguf_preserves_explicit_epsilon_with_optional_defaults() {
    let model = model_with(|builder| {
        builder
            .set("llama.attention.layer_norm_rms_epsilon", Value::F32(2e-5))
            .without_metadata("llama.attention.head_count_kv")
            .without_metadata("llama.rope.freq_base");
        // 缺省 KV 头数表示 MHA，K/V 投影也必须扩展到与 Q 相同的输出维度。
        for (name, dims) in &mut builder.tensors {
            if name.ends_with(".attn_k.weight") || name.ends_with(".attn_v.weight") {
                dims[1] = u64::from(N_EMBD);
            }
        }
    });
    let config = TransformerConfig::from_gguf(&model).expect("可选字段缺失时应采用默认值");
    assert_eq!(config.num_kv_heads, N_HEAD as usize);
    assert_eq!(config.rope_theta, 10_000.0);
    assert_eq!(config.rms_norm_eps, 2e-5);

    let transformer = Transformer::from_gguf(&model).expect("默认配置应能装配完整模型");
    assert_eq!(transformer.config().num_kv_heads, N_HEAD as usize);
    assert_eq!(transformer.config().rope_theta, 10_000.0);
    assert_eq!(transformer.config().rms_norm_eps, 2e-5);
    let logits = transformer.forward(&[1, 2], 0).expect("默认配置应能前向");
    assert_eq!(logits.dims(), [2, VOCAB as usize]);
    assert!(logits.data().iter().all(|value| value.is_finite()));
}

#[test]
fn from_gguf_rejects_invalid_rms_norm_epsilon() {
    for value in [
        Value::Str("0.00002".to_string()),
        Value::U32(1),
        Value::F32(0.0),
        Value::F32(-1.0),
        Value::F32(f32::NAN),
        Value::F32(f32::INFINITY),
        Value::F32(f32::NEG_INFINITY),
    ] {
        let model = model_with(|builder| {
            builder.set("llama.attention.layer_norm_rms_epsilon", value);
        });
        let config_error =
            TransformerConfig::from_gguf(&model).expect_err("非法 RMSNorm epsilon 时应报错");
        let model_error =
            Transformer::from_gguf(&model).expect_err("非法 RMSNorm epsilon 时应拒绝加载");
        for error in [config_error, model_error] {
            assert!(
                error.contains("llama.attention.layer_norm_rms_epsilon"),
                "错误信息应指出非法的 epsilon 键，实得: {error}"
            );
        }
    }
}

#[test]
fn from_gguf_builds_every_layer_with_the_expected_shapes() {
    let gguf = load_fixture();
    let model = Transformer::from_gguf(&gguf).expect("装配夹具模型失败");

    // 形状不匹配会在 from_parts 里被拒绝，能构造成功即说明各层维度自洽。
    // 这里再前向一次，确认装配结果真的可计算而不只是形状对得上。
    let logits = model.forward(&TOKENS, 0).expect("前向失败");
    assert_eq!(logits.dims(), [TOKENS.len(), 64]);
}

/// 期望 logits：由独立实现的 Python 参考前向对同一夹具算出。
///
/// 重跑 `python tests/fixtures/generate_tiny_llama.py` 可复现；改动夹具后必须
/// 同步更新这里的常量，否则本用例会失败——这正是它存在的意义。
const EXPECTED_LOGITS: [f32; 64] = [
    0.241964, 0.266576, -0.563725, 0.346897, -0.285221, 0.133072, -0.142828, 0.058700, -0.580136,
    0.062684, 0.438004, 0.260078, 0.538473, -0.303162, 0.575375, -0.120397, -0.661618, 0.249160,
    0.443428, 0.069275, 0.198794, -0.523222, -0.074055, 0.114713, -0.171454, 0.215881, -0.214310,
    0.118529, 0.530096, -0.239329, 0.258476, -0.219541, 0.116128, 0.426986, -0.209413, -0.563008,
    0.149911, -0.327313, -0.048737, 0.086085, -0.020468, -0.036880, -0.238687, -0.458884, 0.269062,
    0.307758, -0.046312, 0.175337, -0.173909, 0.362716, 0.252646, 0.182579, -0.177625, 0.512326,
    -0.203290, -0.264150, 0.072430, -0.278846, 0.072587, 0.248895, -0.300666, 0.220664, 0.100834,
    -0.736157,
];

#[test]
fn forward_matches_the_independently_computed_logits() {
    let gguf = load_fixture();
    let model = Transformer::from_gguf(&gguf).expect("装配夹具模型失败");

    let logits = model.forward(&TOKENS, 0).expect("前向失败");
    let vocab_size = logits.dims()[1];
    let last = logits.dims()[0] - 1;
    let actual = &logits.data()[last * vocab_size..(last + 1) * vocab_size];

    assert_eq!(actual.len(), EXPECTED_LOGITS.len());
    for (index, (&expected, &got)) in EXPECTED_LOGITS.iter().zip(actual).enumerate() {
        assert!(
            (expected - got).abs() <= 1e-4,
            "token {index} 的 logit 应为 {expected}，实得 {got}"
        );
    }
}

/// 因果掩码 + 相对位置 RoPE 的直接推论：长序列前缀的 logits 与短序列完全一致。
///
/// 这条性质与 RoPE 采用相邻对还是对半拆分无关，因此它独立地锁死了"因果掩码是否
/// 真的挡住了未来位置"和"多头合并是否把各头放回了正确的输出列"——这两处错了，
/// 前缀不变性会立刻被破坏。
#[test]
fn forward_is_invariant_under_sequence_prefixing() {
    let gguf = load_fixture();
    let model = Transformer::from_gguf(&gguf).expect("装配夹具模型失败");
    let vocab_size = 64;

    let full = model.forward(&TOKENS, 0).expect("前向失败");
    for prefix_len in 1..=TOKENS.len() {
        let short = model
            .forward(&TOKENS[..prefix_len], 0)
            .expect("前缀前向失败");
        let full_row = &full.data()[(prefix_len - 1) * vocab_size..prefix_len * vocab_size];
        let short_row = &short.data()[(prefix_len - 1) * vocab_size..prefix_len * vocab_size];
        for (index, (&expected, &got)) in full_row.iter().zip(short_row).enumerate() {
            assert!(
                (expected - got).abs() <= 1e-4,
                "前缀长度 {prefix_len} 的 token {index} logit 漂移: {expected} -> {got}"
            );
        }
    }
}

#[test]
fn from_gguf_reports_the_missing_tensor_by_name() {
    let model = model_with(|builder| {
        builder.without("blk.0.attn_k.weight");
    });
    let error = Transformer::from_gguf(&model).expect_err("缺少张量时应报错");
    assert!(
        error.contains("blk.0.attn_k.weight"),
        "错误信息应指出缺失的张量名，实得: {error}"
    );
}

#[test]
fn from_gguf_rejects_a_block_count_the_file_cannot_satisfy() {
    let model = model_with(|builder| {
        builder.set("llama.block_count", Value::U32(3));
    });
    let error = Transformer::from_gguf(&model).expect_err("层数超出实际张量时应报错");
    assert!(
        error.contains("blk.2."),
        "错误信息应指出越界的层，实得: {error}"
    );
}

#[test]
fn from_gguf_rejects_partial_rope() {
    // n_embd=8、head_count=2 时 head_dim 为 4，而 rope.dimension_count=2 表示
    // 只旋转一半维度。本阶段只实现整头旋转，必须显式拒绝而不是按整头算出一个
    // 看不出错的结果。
    let model = model_with(|builder| {
        builder
            .set("llama.attention.head_count", Value::U32(2))
            .set("llama.rope.dimension_count", Value::U32(2));
    });
    let error = Transformer::from_gguf(&model).expect_err("部分旋转应报错");
    assert!(
        error.contains("partial RoPE"),
        "错误信息应说明不支持部分旋转，实得: {error}"
    );
}

#[test]
fn from_gguf_rejects_an_indivisible_head_dimension() {
    let model = model_with(|builder| {
        builder.set("llama.attention.head_count", Value::U32(3));
    });
    let error = Transformer::from_gguf(&model).expect_err("hidden 不能被头数整除应报错");
    assert!(
        error.contains("divisible"),
        "错误信息应说明整除关系，实得: {error}"
    );
}

#[test]
fn from_gguf_rejects_a_vocabulary_that_disagrees_with_the_embedding() {
    // 词表数组与 token_embd 行数描述同一个词表；不一致时 argmax 出的 id 会落到
    // 错误的 token 文本上，必须拒绝加载而不是挑一个用。
    let model = model_with(|builder| {
        builder.set(
            "tokenizer.ggml.tokens",
            Value::Tokens((0..16).map(|i| format!("tok{i}")).collect()),
        );
    });
    let error = Transformer::from_gguf(&model).expect_err("词表与嵌入行数不一致应报错");
    assert!(
        error.contains("tokenizer.ggml.tokens"),
        "错误信息应指出词表不一致，实得: {error}"
    );
}

#[test]
fn from_gguf_falls_back_to_tied_embeddings() {
    // llama.cpp 对没有 output.weight 的模型复用 token_embd，这里确认同一行为。
    let model = model_with(|builder| {
        builder.without("output.weight");
    });
    let transformer =
        Transformer::from_gguf(&model).expect("缺少 output.weight 时应回退到共享权重");
    let logits = transformer.forward(&[1, 2], 0).expect("前向失败");
    assert_eq!(logits.dims(), [2, 32]);
}

#[test]
fn from_gguf_rejects_quantized_weights() {
    // 量化张量在索引里就存在；本用例的字节数不是合法 Q4_0 块布局，加载必须
    // 给出类型错误而不是解出垃圾数值。真实的 Q4_0/Q8_0 解码由 tests/gguf_loader.rs
    // 的 mapped_* 用例与真实模型验收覆盖。
    let model = model_with(|builder| {
        builder.quantize("blk.0.attn_q.weight");
    });
    let error = Transformer::from_gguf(&model).expect_err("量化权重应报错");
    assert!(
        error.contains("blk.0.attn_q.weight") && error.contains("Q4_0"),
        "错误信息应指出量化张量与类型，实得: {error}"
    );
}

// ---------------------------------------------------------------------------
// 最小 GGUF 写入器
// ---------------------------------------------------------------------------

/// metadata 值的几种写法，只覆盖本文件用到的类型。
enum Value {
    U32(u32),
    F32(f32),
    Str(String),
    Tokens(Vec<String>),
}

/// 一个可增删张量、可改 metadata 的微型 Llama GGUF 构造器。
///
/// 只为错误路径服务：所有张量都是 F32，取值只要求确定，不要求有数值意义。
struct MiniGguf {
    metadata: Vec<(String, Value)>,
    tensors: Vec<(String, Vec<u64>)>,
    quantized: Vec<String>,
}

const N_EMBD: u32 = 8;
const N_FF: u32 = 16;
const N_LAYER: u32 = 2;
const N_HEAD: u32 = 4;
const N_KV: u32 = 2;
const VOCAB: u32 = 32;
const ALIGNMENT: u64 = 32;

impl MiniGguf {
    fn new() -> Self {
        // GQA 下 K/V 的输出特征数是 n_kv * head_dim，与 Q 不同；写成与 Q 相同的
        // [n_embd, n_embd] 会让 Attention::new 的形状校验先于被测逻辑报错。
        let head_dim = N_EMBD / N_HEAD;
        let q_features = N_HEAD * head_dim;
        let kv_features = N_KV * head_dim;
        let mut builder = Self {
            metadata: vec![
                (
                    "general.architecture".to_string(),
                    Value::Str("llama".to_string()),
                ),
                ("general.name".to_string(), Value::Str("mini".to_string())),
                ("llama.context_length".to_string(), Value::U32(128)),
                ("llama.embedding_length".to_string(), Value::U32(N_EMBD)),
                ("llama.feed_forward_length".to_string(), Value::U32(N_FF)),
                ("llama.block_count".to_string(), Value::U32(N_LAYER)),
                ("llama.attention.head_count".to_string(), Value::U32(N_HEAD)),
                (
                    "llama.attention.head_count_kv".to_string(),
                    Value::U32(N_KV),
                ),
                (
                    "llama.attention.layer_norm_rms_epsilon".to_string(),
                    Value::F32(1e-6),
                ),
                ("llama.rope.freq_base".to_string(), Value::F32(10000.0)),
                (
                    "tokenizer.ggml.tokens".to_string(),
                    Value::Tokens((0..VOCAB).map(|i| format!("tok{i}")).collect()),
                ),
            ],
            tensors: Vec::new(),
            quantized: Vec::new(),
        };
        builder.tensors.push((
            "token_embd.weight".to_string(),
            vec![u64::from(N_EMBD), u64::from(VOCAB)],
        ));
        for il in 0..N_LAYER {
            for (suffix, dims) in [
                ("attn_norm.weight", vec![u64::from(N_EMBD)]),
                (
                    "attn_q.weight",
                    vec![u64::from(N_EMBD), u64::from(q_features)],
                ),
                (
                    "attn_k.weight",
                    vec![u64::from(N_EMBD), u64::from(kv_features)],
                ),
                (
                    "attn_v.weight",
                    vec![u64::from(N_EMBD), u64::from(kv_features)],
                ),
                (
                    "attn_output.weight",
                    vec![u64::from(q_features), u64::from(N_EMBD)],
                ),
                ("ffn_norm.weight", vec![u64::from(N_EMBD)]),
                ("ffn_gate.weight", vec![u64::from(N_EMBD), u64::from(N_FF)]),
                ("ffn_up.weight", vec![u64::from(N_EMBD), u64::from(N_FF)]),
                ("ffn_down.weight", vec![u64::from(N_FF), u64::from(N_EMBD)]),
            ] {
                builder.tensors.push((format!("blk.{il}.{suffix}"), dims));
            }
        }
        builder
            .tensors
            .push(("output_norm.weight".to_string(), vec![u64::from(N_EMBD)]));
        builder.tensors.push((
            "output.weight".to_string(),
            vec![u64::from(N_EMBD), u64::from(VOCAB)],
        ));
        builder
    }

    fn set(&mut self, key: &str, value: Value) -> &mut Self {
        match self.metadata.iter_mut().find(|(name, _)| name == key) {
            Some(entry) => entry.1 = value,
            None => self.metadata.push((key.to_string(), value)),
        }
        self
    }

    fn without_metadata(&mut self, key: &str) -> &mut Self {
        self.metadata.retain(|(name, _)| name != key);
        self
    }

    fn without(&mut self, name: &str) -> &mut Self {
        self.tensors.retain(|(tensor, _)| tensor != name);
        self
    }

    fn quantize(&mut self, name: &str) -> &mut Self {
        self.quantized.push(name.to_string());
        self
    }

    /// 写入文件并返回可直接解析的 `GgufFile`。
    fn build(&self) -> GgufFile {
        // 同一进程内的用例并行执行，路径必须每次不同，否则会互相覆盖文件。
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "qllama-from-gguf-{}-{unique}.gguf",
            std::process::id()
        ));
        let mut body = Vec::new();
        body.extend_from_slice(b"GGUF");
        put_u32(&mut body, 3);
        put_u64(&mut body, self.tensors.len() as u64);
        put_u64(&mut body, self.metadata.len() as u64);
        for (key, value) in &self.metadata {
            put_string(&mut body, key);
            put_value(&mut body, value);
        }

        let mut data = Vec::new();
        let mut offsets = Vec::with_capacity(self.tensors.len());
        for (_name, dims) in &self.tensors {
            let pad = (ALIGNMENT - (data.len() as u64 % ALIGNMENT)) % ALIGNMENT;
            data.extend(std::iter::repeat_n(0u8, pad as usize));
            offsets.push(data.len() as u64);
            let count: usize = dims.iter().map(|&dim| dim as usize).product();
            for index in 0..count {
                put_f32(&mut data, 0.001 * (index as f32 + 1.0) - 0.5);
            }
        }

        for ((name, dims), offset) in self.tensors.iter().zip(&offsets) {
            put_string(&mut body, name);
            put_u32(&mut body, dims.len() as u32);
            for dim in dims {
                put_u64(&mut body, *dim);
            }
            // 量化用例把类型改成 Q4_0（编号 2）；element_size 为 0，因此字节数按
            // 元素数的一半给足，足以让加载器走到"不支持"的分支。
            let ggml_type = if self.quantized.iter().any(|tensor| tensor == name) {
                2
            } else {
                0
            };
            put_u32(&mut body, ggml_type);
            put_u64(&mut body, *offset);
        }

        let pad = (ALIGNMENT - (body.len() as u64 % ALIGNMENT)) % ALIGNMENT;
        body.extend(std::iter::repeat_n(0u8, pad as usize));
        body.extend(data);
        std::fs::write(&path, body).expect("写入临时夹具失败");

        let file = GgufFile::from_path(&path).expect("解析临时夹具失败");
        std::fs::remove_file(&path).ok();
        file
    }
}

fn model_with(configure: impl FnOnce(&mut MiniGguf)) -> GgufFile {
    let mut builder = MiniGguf::new();
    configure(&mut builder);
    builder.build()
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_f32(out: &mut Vec<u8>, value: f32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_string(out: &mut Vec<u8>, value: &str) {
    put_u64(out, value.len() as u64);
    out.extend_from_slice(value.as_bytes());
}

fn put_value(out: &mut Vec<u8>, value: &Value) {
    match value {
        Value::U32(v) => {
            put_u32(out, 4);
            put_u32(out, *v);
        }
        Value::F32(v) => {
            put_u32(out, 6);
            put_f32(out, *v);
        }
        Value::Str(v) => {
            put_u32(out, 8);
            put_string(out, v);
        }
        Value::Tokens(items) => {
            put_u32(out, 9);
            put_u32(out, 8);
            put_u64(out, items.len() as u64);
            for item in items {
                put_string(out, item);
            }
        }
    }
}
