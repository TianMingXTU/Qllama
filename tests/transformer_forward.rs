//! Transformer 前向推理的集成测试。
//!
//! 端到端数值由 [`reference_transformer`] 独立算出：f64、朴素循环、按公式直书，
//! 不调用待测代码的任何路径。权重在测试内自行生成，与
//! `Transformer::tiny_for_testing` 的种子无关，避免两侧共享隐藏假设。

use qllama::kv_cache::KvCache;
use qllama::transformer::layers::{Attention, Embedding, Linear, Mlp, RmsNorm, TransformerBlock};
use qllama::transformer::{Transformer, TransformerConfig};

/// 参考实现与待测代码共用的微型配置。
struct TinyConfig {
    hidden: usize,
    intermediate: usize,
    heads: usize,
    kv_heads: usize,
    layers: usize,
    vocab: usize,
    max_pos: usize,
    seq: usize,
    eps: f64,
    theta: f64,
}

fn tiny_config() -> TinyConfig {
    TinyConfig {
        hidden: 4,
        intermediate: 6,
        heads: 2,
        kv_heads: 1,
        layers: 2,
        vocab: 5,
        max_pos: 8,
        seq: 3,
        eps: 1e-5,
        theta: 10000.0,
    }
}

impl TinyConfig {
    fn to_config(&self) -> TransformerConfig {
        let mut config = TransformerConfig::new(
            self.hidden,
            self.intermediate,
            self.heads,
            self.kv_heads,
            self.layers,
            self.vocab,
            self.max_pos,
        );
        config.rms_norm_eps = self.eps as f32;
        config.rope_theta = self.theta as f32;
        config
    }

    fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }
}

// ---------------------------------------------------------------------------
// 确定性权重生成（测试自有，与 src 内的实现无关）
// ---------------------------------------------------------------------------

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    fn unit(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.state >> 40) as f64) / ((1u32 << 24) as f64) - 0.5
    }

    fn matrix(&mut self, rows: usize, cols: usize) -> Vec<Vec<f64>> {
        let scale = 1.0 / (cols as f64).sqrt();
        (0..rows)
            .map(|_| (0..cols).map(|_| self.unit() * scale).collect())
            .collect()
    }

    /// 归一化权重：1 附近的小扰动。
    fn norm(&mut self, len: usize) -> Vec<f64> {
        (0..len).map(|_| 1.0 + self.unit() * 0.1).collect()
    }
}

struct ReferenceLayer {
    input_norm: Vec<f64>,
    wq: Vec<Vec<f64>>,
    wk: Vec<Vec<f64>>,
    wv: Vec<Vec<f64>>,
    wo: Vec<Vec<f64>>,
    post_norm: Vec<f64>,
    gate: Vec<Vec<f64>>,
    up: Vec<Vec<f64>>,
    down: Vec<Vec<f64>>,
}

struct ReferenceWeights {
    embedding: Vec<Vec<f64>>,
    layers: Vec<ReferenceLayer>,
    final_norm: Vec<f64>,
    lm_head: Vec<Vec<f64>>,
}

fn reference_weights(config: &TinyConfig) -> ReferenceWeights {
    let mut rng = Lcg::new(0x5EED_1234);
    let head_dim = config.head_dim();
    let hidden = config.hidden;

    ReferenceWeights {
        embedding: rng.matrix(config.vocab, hidden),
        layers: (0..config.layers)
            .map(|_| ReferenceLayer {
                input_norm: rng.norm(hidden),
                wq: rng.matrix(config.heads * head_dim, hidden),
                wk: rng.matrix(config.kv_heads * head_dim, hidden),
                wv: rng.matrix(config.kv_heads * head_dim, hidden),
                wo: rng.matrix(hidden, config.heads * head_dim),
                post_norm: rng.norm(hidden),
                gate: rng.matrix(config.intermediate, hidden),
                up: rng.matrix(config.intermediate, hidden),
                down: rng.matrix(hidden, config.intermediate),
            })
            .collect(),
        final_norm: rng.norm(hidden),
        lm_head: rng.matrix(config.vocab, hidden),
    }
}

// ---------------------------------------------------------------------------
// 独立参考实现
// ---------------------------------------------------------------------------

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(&a, &b)| a * b).sum()
}

fn reference_rms_norm(x: &[f64], weight: &[f64], eps: f64) -> Vec<f64> {
    let mean_square: f64 = x.iter().map(|&v| v * v).sum::<f64>() / x.len() as f64;
    let denominator = (mean_square + eps).sqrt();
    x.iter()
        .zip(weight)
        .map(|(&v, &g)| v / denominator * g)
        .collect()
}

fn reference_attention(
    x: &[Vec<f64>],
    layer: &ReferenceLayer,
    config: &TinyConfig,
    start_pos: usize,
) -> Vec<Vec<f64>> {
    let seq_len = x.len();
    let head_dim = config.head_dim();
    let group = config.heads / config.kv_heads;

    let mut q = vec![vec![0.0; config.heads * head_dim]; seq_len];
    let mut k = vec![vec![0.0; config.kv_heads * head_dim]; seq_len];
    let mut v = vec![vec![0.0; config.kv_heads * head_dim]; seq_len];
    for (position, token) in x.iter().enumerate() {
        for (out, row) in layer.wq.iter().enumerate() {
            q[position][out] = dot(token, row);
        }
        for (out, row) in layer.wk.iter().enumerate() {
            k[position][out] = dot(token, row);
        }
        for (out, row) in layer.wv.iter().enumerate() {
            v[position][out] = dot(token, row);
        }
    }

    let rotate = |tensor: &mut Vec<Vec<f64>>, head_count: usize| {
        for (position, token) in tensor.iter_mut().enumerate() {
            let base = (start_pos + position) as f64;
            for head in 0..head_count {
                for pair in 0..head_dim / 2 {
                    let frequency = config.theta.powf(-2.0 * pair as f64 / head_dim as f64);
                    let (sin, cos) = (base * frequency).sin_cos();
                    let first = head * head_dim + 2 * pair;
                    let (a, b) = (token[first], token[first + 1]);
                    token[first] = a * cos - b * sin;
                    token[first + 1] = a * sin + b * cos;
                }
            }
        }
    };
    rotate(&mut q, config.heads);
    rotate(&mut k, config.kv_heads);

    let scale = 1.0 / (head_dim as f64).sqrt();
    let hidden = config.heads * head_dim;
    let mut merged = vec![vec![0.0; hidden]; seq_len];
    for head in 0..config.heads {
        let kv_head = head / group;
        for query in 0..seq_len {
            let scores: Vec<f64> = (0..seq_len)
                .map(|key| {
                    if key > query {
                        return f64::NEG_INFINITY;
                    }
                    let mut sum = 0.0;
                    for dim in 0..head_dim {
                        sum += q[query][head * head_dim + dim] * k[key][kv_head * head_dim + dim];
                    }
                    sum * scale
                })
                .collect();
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exponentials: Vec<f64> = scores.iter().map(|&s| (s - maximum).exp()).collect();
            let total: f64 = exponentials.iter().sum();
            // 概率必须是 exponentials / total。scores 里仍留着未指数化的原始值
            // 和被掩掉的 -inf，直接拿它除会得到 -inf 与 NaN。
            let probabilities: Vec<f64> = exponentials.iter().map(|&e| e / total).collect();
            for (dim, output) in merged[query]
                .iter_mut()
                .skip(head * head_dim)
                .take(head_dim)
                .enumerate()
            {
                *output = (0..seq_len)
                    .map(|key| probabilities[key] * v[key][kv_head * head_dim + dim])
                    .sum();
            }
        }
    }

    let mut output = vec![vec![0.0; hidden]; seq_len];
    for (position, token) in merged.iter().enumerate() {
        for (out, row) in layer.wo.iter().enumerate() {
            output[position][out] = dot(token, row);
        }
    }
    output
}

fn reference_transformer(
    tokens: &[i32],
    weights: &ReferenceWeights,
    config: &TinyConfig,
    start_pos: usize,
) -> Vec<Vec<f64>> {
    let mut hidden_states: Vec<Vec<f64>> = tokens
        .iter()
        .map(|&token| weights.embedding[token as usize].clone())
        .collect();

    for layer in &weights.layers {
        let normed: Vec<Vec<f64>> = hidden_states
            .iter()
            .map(|token| reference_rms_norm(token, &layer.input_norm, config.eps))
            .collect();
        let attended = reference_attention(&normed, layer, config, start_pos);
        hidden_states = hidden_states
            .iter()
            .zip(&attended)
            .map(|(residual, delta)| residual.iter().zip(delta).map(|(&a, &b)| a + b).collect())
            .collect();

        let normed: Vec<Vec<f64>> = hidden_states
            .iter()
            .map(|token| reference_rms_norm(token, &layer.post_norm, config.eps))
            .collect();
        let fed: Vec<Vec<f64>> = normed
            .iter()
            .map(|token| {
                let gate: Vec<f64> = layer.gate.iter().map(|row| dot(token, row)).collect();
                let up: Vec<f64> = layer.up.iter().map(|row| dot(token, row)).collect();
                let inner: Vec<f64> = gate
                    .iter()
                    .zip(&up)
                    .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
                    .collect();
                layer.down.iter().map(|row| dot(&inner, row)).collect()
            })
            .collect();
        hidden_states = hidden_states
            .iter()
            .zip(&fed)
            .map(|(residual, delta)| residual.iter().zip(delta).map(|(&a, &b)| a + b).collect())
            .collect();
    }

    hidden_states
        .iter()
        .map(|token| reference_rms_norm(token, &weights.final_norm, config.eps))
        .map(|normalized| {
            weights
                .lm_head
                .iter()
                .map(|row| dot(&normalized, row))
                .collect()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// 由参考权重装配待测模型
// ---------------------------------------------------------------------------

fn to_tensor(rows: &[Vec<f64>], dims: Vec<usize>) -> qllama::tensor::Tensor {
    qllama::tensor::Tensor::new(
        rows.iter().flatten().map(|&value| value as f32).collect(),
        dims,
    )
}

fn linear_from(rows: &[Vec<f64>]) -> Linear {
    Linear::new(to_tensor(rows, vec![rows.len(), rows[0].len()]), None).expect("构造 Linear 失败")
}

fn rms_from(values: &[f64], eps: f32) -> RmsNorm {
    RmsNorm::new(
        qllama::tensor::Tensor::new(
            values.iter().map(|&v| v as f32).collect(),
            vec![values.len()],
        ),
        eps,
    )
    .expect("构造 RmsNorm 失败")
}

fn model_from_reference(config: &TinyConfig, weights: &ReferenceWeights) -> Transformer {
    let head_dim = config.head_dim();
    let layers: Vec<TransformerBlock> = weights
        .layers
        .iter()
        .map(|layer| TransformerBlock {
            input_layernorm: rms_from(&layer.input_norm, config.eps as f32),
            self_attn: Attention::new(
                linear_from(&layer.wq),
                linear_from(&layer.wk),
                linear_from(&layer.wv),
                linear_from(&layer.wo),
                config.heads,
                config.kv_heads,
                head_dim,
                config.theta as f32,
            )
            .expect("构造 Attention 失败"),
            post_attention_layernorm: rms_from(&layer.post_norm, config.eps as f32),
            mlp: Mlp::new(
                linear_from(&layer.gate),
                linear_from(&layer.up),
                linear_from(&layer.down),
            )
            .expect("构造 Mlp 失败"),
        })
        .collect();

    Transformer::from_parts(
        config.to_config(),
        Embedding::new(to_tensor(
            &weights.embedding,
            vec![config.vocab, config.hidden],
        ))
        .expect("构造 Embedding 失败"),
        layers,
        rms_from(&weights.final_norm, config.eps as f32),
        linear_from(&weights.lm_head),
    )
    .expect("装配模型失败")
}

fn tokens_for(config: &TinyConfig) -> Vec<i32> {
    (0..config.seq)
        .map(|i| (i as i32) % config.vocab as i32)
        .collect()
}

fn assert_close_rows(actual: &[f32], expected: &[f64], tolerance: f32, context: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{context}: 元素个数应为 {}，得到 {}",
        expected.len(),
        actual.len()
    );
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (got - want as f32).abs() <= tolerance,
            "{context}: 第 {index} 个元素应为 {want}，得到 {got}"
        );
    }
}

// ---------------------------------------------------------------------------
// 用例
// ---------------------------------------------------------------------------

#[test]
fn forward_returns_one_logit_row_per_token() {
    let config = tiny_config();
    let model = Transformer::tiny_for_testing(config.to_config()).expect("构造失败");
    let tokens = tokens_for(&config);

    let logits = model.forward(&tokens, 0).expect("前向失败");
    assert_eq!(logits.dims(), &[config.seq, config.vocab]);
    assert_eq!(logits.len(), config.seq * config.vocab);
}

#[test]
fn forward_matches_an_independent_reference_implementation() {
    // 端到端数值对照独立参考实现，覆盖嵌入、两层残差、GQA、RoPE 与语言头。
    let config = tiny_config();
    let weights = reference_weights(&config);
    let model = model_from_reference(&config, &weights);
    let tokens = tokens_for(&config);

    let logits = model.forward(&tokens, 0).expect("前向失败");
    let expected = reference_transformer(&tokens, &weights, &config, 0);

    assert_eq!(logits.dims(), &[config.seq, config.vocab]);
    for (position, expected_row) in expected.iter().enumerate() {
        let columns = logits.dims()[1];
        assert_close_rows(
            &logits.data()[position * columns..(position + 1) * columns],
            expected_row,
            1e-3,
            &format!("start_pos=0 第 {position} 行"),
        );
    }
}

#[test]
fn forward_matches_reference_at_a_nonzero_start_position() {
    // start_pos 只改变 RoPE 的角度基准，参考实现同样支持，用于交叉验证。
    let config = tiny_config();
    let weights = reference_weights(&config);
    let model = model_from_reference(&config, &weights);
    let tokens = tokens_for(&config);

    let logits = model.forward(&tokens, 2).expect("前向失败");
    let expected = reference_transformer(&tokens, &weights, &config, 2);

    for (position, expected_row) in expected.iter().enumerate() {
        let columns = logits.dims()[1];
        assert_close_rows(
            &logits.data()[position * columns..(position + 1) * columns],
            expected_row,
            1e-3,
            &format!("start_pos=2 第 {position} 行"),
        );
    }
}

#[test]
fn forward_is_invariant_to_a_uniform_position_shift() {
    // RoPE 是旋转，因此 q·k 只依赖相对位置：R(θ_q)q · R(θ_k)k = q·R(θ_k-θ_q)k。
    // 整段序列一起平移时相对位置不变，注意力输出（进而整个模型输出）应保持不变。
    // 这正是无 KV Cache 时 start_pos 不影响结果的原因；接入增量解码后，
    // 缓存里的 K/V 已按各自的历史位置旋转过，start_pos 才会真正参与计算。
    let config = tiny_config();
    let model = Transformer::tiny_for_testing(config.to_config()).expect("构造失败");
    let tokens = tokens_for(&config);

    let at_zero = model.forward(&tokens, 0).expect("前向失败");
    let shifted = model.forward(&tokens, 3).expect("前向失败");

    assert_eq!(at_zero.dims(), shifted.dims());
    for (index, (&left, &right)) in at_zero.data().iter().zip(shifted.data()).enumerate() {
        assert!(
            (left - right).abs() <= 1e-5,
            "第 {index} 个 logit 应基本不变：{left} vs {right}"
        );
    }
}

#[test]
fn forward_is_deterministic_for_the_same_input() {
    let config = tiny_config();
    let model = Transformer::tiny_for_testing(config.to_config()).expect("构造失败");
    let tokens = tokens_for(&config);

    let first = model.forward(&tokens, 0).expect("第一次前向失败");
    let second = model.forward(&tokens, 0).expect("第二次前向失败");
    assert_eq!(first.data(), second.data(), "同一输入必须得到同一结果");
}

#[test]
fn forward_produces_finite_logits() {
    let config = tiny_config();
    let model = Transformer::tiny_for_testing(config.to_config()).expect("构造失败");
    let tokens = tokens_for(&config);

    let logits = model.forward(&tokens, 0).expect("前向失败");
    assert!(
        logits.data().iter().all(|value| value.is_finite()),
        "logits 含非有限值: {:?}",
        logits.data()
    );
}

#[test]
fn forward_logits_depend_on_the_token_sequence() {
    let config = tiny_config();
    let model = Transformer::tiny_for_testing(config.to_config()).expect("构造失败");

    let first = model.forward(&[0, 1, 2], 0).expect("前向失败");
    let second = model.forward(&[2, 1, 0], 0).expect("前向失败");
    assert_ne!(
        first.data(),
        second.data(),
        "不同 token 序列必须得到不同 logits"
    );
}

#[test]
fn forward_handles_empty_input() {
    let config = tiny_config();
    let model = Transformer::tiny_for_testing(config.to_config()).expect("构造失败");

    let logits = model.forward(&[], 0).expect("空输入应成功");
    assert_eq!(logits.dims(), &[0, config.vocab]);
    assert!(logits.is_empty());
}

#[test]
fn forward_rejects_out_of_range_token_ids() {
    let config = tiny_config();
    let model = Transformer::tiny_for_testing(config.to_config()).expect("构造失败");

    let negative = model.forward(&[-1], 0).expect_err("负 id 应被拒绝");
    assert!(negative.contains("-1"), "错误应指出 id: {negative}");

    let too_large = model
        .forward(&[config.vocab as i32], 0)
        .expect_err("越界 id 应被拒绝");
    assert!(
        too_large.contains("vocabulary"),
        "错误应说明词表上限: {too_large}"
    );
}

#[test]
fn forward_rejects_positions_beyond_max_position_embeddings() {
    let config = tiny_config();
    let model = Transformer::tiny_for_testing(config.to_config()).expect("构造失败");
    let tokens = tokens_for(&config);

    // 末位置 = start_pos + seq - 1 = 5 + 2 = 7 < 8，应通过。
    model.forward(&tokens, 5).expect("末位置在界内应成功");

    // 末位置 = 6 + 2 = 8 >= 8，应被拒绝。
    let error = model.forward(&tokens, 6).expect_err("末位置越界应被拒绝");
    assert!(
        error.contains("max_position_embeddings"),
        "错误应说明位置上限: {error}"
    );
}

#[test]
fn from_parts_rejects_inconsistent_shapes() {
    let config = tiny_config();
    let weights = reference_weights(&config);
    let good = model_from_reference(&config, &weights);

    // 层数与 config 不符。
    let mut fewer_layers = good.config().clone();
    fewer_layers.num_layers += 1;
    let rebuilt = Transformer::from_parts(
        fewer_layers,
        Embedding::new(to_tensor(
            &weights.embedding,
            vec![config.vocab, config.hidden],
        ))
        .unwrap(),
        Vec::new(),
        rms_from(&weights.final_norm, config.eps as f32),
        linear_from(&weights.lm_head),
    )
    .expect_err("层数不符应被拒绝");
    assert!(rebuilt.contains("layers"), "错误应指出层数: {rebuilt}");

    // 词表大小与 config 不符。
    let wrong_vocab = Transformer::from_parts(
        config.to_config(),
        Embedding::new(to_tensor(
            &weights.embedding,
            vec![config.vocab + 1, config.hidden],
        ))
        .unwrap(),
        good_layers(&config, &weights),
        rms_from(&weights.final_norm, config.eps as f32),
        linear_from(&weights.lm_head),
    )
    .expect_err("词表大小不符应被拒绝");
    assert!(
        wrong_vocab.contains("vocab_size"),
        "错误应指出词表大小: {wrong_vocab}"
    );
}

/// 重新装配一组与 config 自洽的层，供形状不一致的用例复用。
fn good_layers(config: &TinyConfig, weights: &ReferenceWeights) -> Vec<TransformerBlock> {
    let head_dim = config.head_dim();
    weights
        .layers
        .iter()
        .map(|layer| TransformerBlock {
            input_layernorm: rms_from(&layer.input_norm, config.eps as f32),
            self_attn: Attention::new(
                linear_from(&layer.wq),
                linear_from(&layer.wk),
                linear_from(&layer.wv),
                linear_from(&layer.wo),
                config.heads,
                config.kv_heads,
                head_dim,
                config.theta as f32,
            )
            .unwrap(),
            post_attention_layernorm: rms_from(&layer.post_norm, config.eps as f32),
            mlp: Mlp::new(
                linear_from(&layer.gate),
                linear_from(&layer.up),
                linear_from(&layer.down),
            )
            .unwrap(),
        })
        .collect()
}

#[test]
fn forward_output_shape_scales_with_sequence_length() {
    let config = tiny_config();
    let model = Transformer::tiny_for_testing(config.to_config()).expect("构造失败");

    for seq_len in 1..=config.max_pos {
        let tokens: Vec<i32> = (0..seq_len as i32)
            .map(|i| i % config.vocab as i32)
            .collect();
        let logits = model.forward(&tokens, 0).expect("前向失败");
        assert_eq!(
            logits.dims(),
            &[seq_len, config.vocab],
            "seq_len={seq_len} 时形状不符"
        );
    }
}

// ---------------------------------------------------------------------------
// KV Cache 增量路径与整段路径的数值一致性（阶段 5 核心验收）
//
// 整段路径 `forward` 是对照基准：它的数值已由上面的 f64 参考实现锁定。
// 这里的每个用例都从不同切分角度重放同一序列，要求增量路径的末行（或每行）
// 与整段路径逐值一致。三条差异点各有用例针对性覆盖：
//   - RoPE 历史不重算：任意"先在 cache 里走一段、再显式 offset 喂新 token"
//     的切分，若读回时二次旋转立刻偏掉；
//   - 因果掩码带 offset：单 token decode 行只许看 <= start_pos 的列；
//   - GQA 头索引：kv_heads < heads 的配置单独覆盖。
// ---------------------------------------------------------------------------

/// 路径间容差：两条路径的算子序列一致，理论上只差 softmax 里
/// `-inf -> exp -> 0` 的加零项与求和次序，1e-5 留有充分余量。
const PATH_TOLERANCE: f32 = 1e-5;

fn cache_for(config: &TinyConfig) -> KvCache {
    KvCache::new(
        config.layers,
        config.max_pos,
        config.kv_heads,
        config.head_dim(),
    )
}

/// 取 logits 最后一行的拷贝。
fn last_row(logits: &qllama::tensor::Tensor) -> Vec<f32> {
    let [seq_len, vocab] = logits.dims() else {
        panic!("logits 应为 2D，得到 {:?}", logits.dims());
    };
    logits.data()[(seq_len - 1) * vocab..*seq_len * vocab].to_vec()
}

/// 按给定切分增量跑完整个序列，返回每一批输出的**全部行**（按位置拼接）。
///
/// `splits` 是各批次的长度，必须恰好覆盖全部 token：如 `[1, 1, 3]` 表示
/// 先喂 1 个、再喂 1 个、最后一批 3 个。每批都是独立的
/// `forward_with_cache` 调用，位置由 cache 的 `current_len` 自然推进，
/// 批与批之间的绝对位置衔接正是被测行为的一部分。
///
/// 返回全部行而不是只返回末行：多 token 批次的前几行与整段路径对应行
/// 同样必须一致——只比末行会放过"掩码对所有行统一用末行窗口"这类缺陷。
fn incremental_all_rows_by_splits(
    model: &Transformer,
    config: &TinyConfig,
    tokens: &[i32],
    splits: &[usize],
) -> Vec<Vec<f32>> {
    assert_eq!(
        splits.iter().sum::<usize>(),
        tokens.len(),
        "切分必须覆盖全部 token"
    );
    assert!(!splits.is_empty(), "至少需要一个批次");

    let mut cache = cache_for(config);
    let mut position = 0;
    let mut rows = Vec::with_capacity(tokens.len());
    for &batch_len in splits {
        let batch = &tokens[position..position + batch_len];
        let logits = model
            .forward_with_cache(batch, position, &mut cache)
            .unwrap_or_else(|error| {
                panic!("位置 {position} 批量 {batch_len} 增量前向失败: {error}")
            });
        assert_eq!(logits.dims(), [batch_len, config.vocab]);
        let vocab = config.vocab;
        for row in 0..batch_len {
            rows.push(logits.data()[row * vocab..(row + 1) * vocab].to_vec());
        }
        position += batch_len;
    }
    rows
}

/// 只取末行的便捷包装。
fn incremental_last_row_by_splits(
    model: &Transformer,
    config: &TinyConfig,
    tokens: &[i32],
    splits: &[usize],
) -> Vec<f32> {
    incremental_all_rows_by_splits(model, config, tokens, splits)
        .pop()
        .expect("至少应有一行输出")
}

/// 逐元素断言两行 logits 相等（容差内）。
fn assert_rows_close(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: 元素个数不符");
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() <= PATH_TOLERANCE,
            "{context}: 第 {index} 个元素应为 {want}，得到 {got}"
        );
    }
}

#[test]
fn prefill_via_cache_matches_full_forward_every_row() {
    // prefill（start_pos=0 的整批）与整段路径算子序列完全相同、形状相同，
    // K/V 只是多经过一次 cache 写入/读出的纯拷贝，因此两条路径应**逐位相同**
    // ——这里用精确相等而不是容差断言，任何偏差都说明算子序列真的分叉了。
    // 注意：decode 路径（矩形掩码）不适用精确相等，见下面的容差测试。
    let config = tiny_config();
    let model = Transformer::tiny_for_testing(config.to_config()).expect("构造失败");
    let tokens = tokens_for(&config);

    let full = model.forward(&tokens, 0).expect("整段前向失败");
    let mut cache = cache_for(&config);
    let cached = model
        .forward_with_cache(&tokens, 0, &mut cache)
        .expect("prefill 失败");

    assert_eq!(cached.dims(), full.dims());
    assert_eq!(cached.data(), full.data(), "prefill 与整段路径应逐位相同");
    assert_eq!(cache.seq_len(), tokens.len());
}

#[test]
fn incremental_all_rows_match_full_forward_at_nonzero_start() {
    // 多 token 批次在 start_pos>0 时，**每一行**都必须与整段路径对应位置一致，
    // 而不只是末行。只比末行会放过"掩码边界对所有行统一取末行窗口"这类缺陷：
    // 末行碰巧正确、批次内更早的行却看到了未来的 key。
    // 切分 [1, 2, 2] 让批次 1 覆盖位置 1..3、批次 2 覆盖位置 3..5，两个批次
    // 都只有末行会被"只比末行"的检查覆盖，其余行正是本用例的靶子。
    let config = tiny_config();
    let model = Transformer::tiny_for_testing(config.to_config()).expect("构造失败");
    let tokens: Vec<i32> = (0..5).map(|i| i % config.vocab as i32).collect();

    let full = model.forward(&tokens, 0).expect("整段前向失败");
    let columns = config.vocab;
    let expected: Vec<Vec<f32>> = (0..tokens.len())
        .map(|row| full.data()[row * columns..(row + 1) * columns].to_vec())
        .collect();

    let actual = incremental_all_rows_by_splits(&model, &config, &tokens, &[1, 2, 2]);
    assert_eq!(actual.len(), expected.len());
    for (row, (got, want)) in actual.iter().zip(&expected).enumerate() {
        assert_rows_close(got, want, &format!("切分 [1,2,2] 第 {row} 行"));
    }
}

#[test]
fn incremental_decode_matches_full_forward_at_every_length() {
    // 首 token prefill、其余逐 token decode（最贴近真实生成循环的切分），
    // 序列长度从 1 到 4 全覆盖。
    let config = tiny_config();
    let model = Transformer::tiny_for_testing(config.to_config()).expect("构造失败");

    for seq_len in 1..5 {
        let tokens: Vec<i32> = (0..seq_len as i32)
            .map(|i| i % config.vocab as i32)
            .collect();
        let expected = last_row(&model.forward(&tokens, 0).expect("整段前向失败"));
        let splits = vec![1; seq_len];
        let actual = incremental_last_row_by_splits(&model, &config, &tokens, &splits);
        assert_rows_close(&actual, &expected, &format!("长度 {seq_len} 逐 token 解码"));
    }
}

#[test]
fn incremental_splits_all_match_full_forward() {
    // 同一序列的不同批量切分必须收敛到同一末行：位置簿记若与批次形状
    // 耦合（比如把 batch 内偏移当成绝对位置），换一种切分就会偏。
    let config = tiny_config();
    let model = Transformer::tiny_for_testing(config.to_config()).expect("构造失败");
    let tokens: Vec<i32> = (0..5).map(|i| i % config.vocab as i32).collect();

    let expected = last_row(&model.forward(&tokens, 0).expect("整段前向失败"));
    for splits in [
        vec![1, 1, 1, 1, 1],
        vec![1, 2, 2],
        vec![2, 3],
        vec![3, 1, 1],
        vec![5],
        vec![4, 1],
    ] {
        let actual = incremental_last_row_by_splits(&model, &config, &tokens, &splits);
        assert_rows_close(&actual, &expected, &format!("切分 {splits:?}"));
    }
}

#[test]
fn incremental_decode_matches_full_forward_under_gqa() {
    // GQA（heads=4, kv_heads=2, 分组大小 2）：cache 里每个 KV 头被两个
    // query 头共享。若 kv_head 错取 head 而非 head/group，两组 query 头会
    // 读到同一份 K/V，末行立刻偏掉。
    let mut config = tiny_config();
    config.hidden = 8;
    config.heads = 4;
    config.kv_heads = 2;
    let model = Transformer::tiny_for_testing(config.to_config()).expect("构造失败");
    let tokens: Vec<i32> = (0..5).map(|i| i % config.vocab as i32).collect();

    let expected = last_row(&model.forward(&tokens, 0).expect("整段前向失败"));
    let actual = incremental_last_row_by_splits(&model, &config, &tokens, &[1, 1, 1, 1, 1]);
    assert_rows_close(&actual, &expected, "GQA 4 头 / 2 KV 头逐 token 解码");
}

#[test]
fn incremental_decode_matches_the_independent_reference() {
    // 增量路径不只对齐整段路径，还要对齐测试自有的 f64 参考实现：
    // 否则两条路径一起错也会显得"一致"。切分取 [1, 2]：一次单 token
    // decode 接一次两 token 批量，覆盖两种批次形状。
    let config = tiny_config();
    let weights = reference_weights(&config);
    let model = model_from_reference(&config, &weights);
    let tokens = tokens_for(&config);

    let expected = reference_transformer(&tokens, &weights, &config, 0);
    let expected_last = expected.last().expect("参考实现应有输出行");
    let actual = incremental_last_row_by_splits(&model, &config, &tokens, &[1, 2]);

    assert_eq!(actual.len(), expected_last.len());
    for (index, (&got, &want)) in actual.iter().zip(expected_last).enumerate() {
        assert!(
            (got - want as f32).abs() <= 1e-3,
            "第 {index} 个元素参考值为 {want}，增量路径得到 {got}"
        );
    }
}

#[test]
fn repeated_prefill_is_rejected_by_the_cache() {
    // 同一份 cache 连续 prefill 两遍，第二次必须被 KvCache 的连续性
    // 规则拒绝，而不是把序列写重或静默跳过。
    let config = tiny_config();
    let model = Transformer::tiny_for_testing(config.to_config()).expect("构造失败");
    let tokens = tokens_for(&config);
    let mut cache = cache_for(&config);

    model
        .forward_with_cache(&tokens, 0, &mut cache)
        .expect("首次 prefill 应成功");
    assert_eq!(cache.seq_len(), tokens.len());

    let error = model
        .forward_with_cache(&tokens, 0, &mut cache)
        .expect_err("重复 prefill 应被拒绝");
    assert!(
        error.contains("must continue"),
        "错误应指出位置必须接续 current_len，实际: {error}"
    );
    // 被拒绝的写入不得推进长度
    assert_eq!(cache.seq_len(), tokens.len());
}

#[test]
fn incremental_forward_rejects_positions_beyond_max_position_embeddings() {
    // 与整段路径同一道边界守卫：末位置越界必须报错，且 cache 不得被
    // 部分写入（守卫在任何投影之前）。
    let config = tiny_config();
    let model = Transformer::tiny_for_testing(config.to_config()).expect("构造失败");
    let mut cache = cache_for(&config);

    let error = model
        .forward_with_cache(&[0], config.max_pos, &mut cache)
        .expect_err("末位置越界应被拒绝");
    assert!(
        error.contains("exceeds max_position_embeddings"),
        "实际: {error}"
    );
    assert_eq!(cache.seq_len(), 0, "越界调用不得写进 cache");
}

#[test]
fn incremental_empty_batch_returns_no_rows_and_keeps_the_prefix() {
    // 空批（new_len=0）是合法空操作：返回 [0, vocab] 空 logits，cache 的
    // 已写前缀不变，后续新 token 仍能接续。整段路径有 forward_handles_empty_input
    // 锁定同类行为，增量路径在这里对称锁定，避免将来在 update/mask 的空路径
    // 上加校验时静默改变契约。
    let config = tiny_config();
    let model = Transformer::tiny_for_testing(config.to_config()).expect("构造失败");
    let tokens = tokens_for(&config);
    let mut cache = cache_for(&config);

    let prefilled = model
        .forward_with_cache(&tokens, 0, &mut cache)
        .expect("prefill 失败");
    assert_eq!(prefilled.dims(), [tokens.len(), config.vocab]);

    let empty = model
        .forward_with_cache(&[], tokens.len(), &mut cache)
        .expect("空批应成功");
    assert_eq!(empty.dims(), [0, config.vocab]);
    assert_eq!(cache.seq_len(), tokens.len(), "空批不得推进 current_len");

    // 空批之后继续 decode 仍能接上
    let next = model
        .forward_with_cache(&[1], tokens.len(), &mut cache)
        .expect("空批后的 decode 应成功");
    assert_eq!(next.dims(), [1, config.vocab]);
    assert_eq!(cache.seq_len(), tokens.len() + 1);
}
