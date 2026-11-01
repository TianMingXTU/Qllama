//! Transformer 基础层的集成测试。
//!
//! 期望值分两类：简单场景直接手工计算并写成字面量；多头注意力这类组合场景
//! 由 [`reference_attention`] 等独立参考实现算出——参考实现用 f64、朴素三重
//! 循环、按公式直书，不调用待测代码的任何路径。

use qllama::tensor::{Tensor, TensorOps};
use qllama::transformer::layers::{Attention, Embedding, Linear, Mlp, RmsNorm, TransformerBlock};

/// 由 `[out_features][in_features]` 的行构造权重张量。
///
/// 与 [`Linear`] 的约定一致：GGUF 内存布局就是"每行一个输出特征"。
fn weight(rows: &[&[f32]]) -> Tensor {
    let out_features = rows.len();
    let in_features = rows[0].len();
    let data: Vec<f32> = rows.iter().flat_map(|row| row.iter().copied()).collect();
    assert_eq!(data.len(), out_features * in_features);
    Tensor::new(data, vec![out_features, in_features])
}

fn linear(rows: &[&[f32]]) -> Linear {
    Linear::new(weight(rows), None).expect("构造 Linear 失败")
}

fn linear_with_bias(rows: &[&[f32]], bias: Vec<f32>) -> Linear {
    Linear::new(weight(rows), Some(Tensor::new(bias, vec![rows.len()]))).expect("构造失败")
}

/// eps 取 0 的归一化层，便于手工推算分母。
fn norm(values: &[f32]) -> RmsNorm {
    RmsNorm::new(Tensor::new(values.to_vec(), vec![values.len()]), 0.0).expect("构造 RmsNorm 失败")
}

fn identity_matrix(n: usize) -> Tensor {
    let mut data = vec![0.0f32; n * n];
    for (index, slot) in data.iter_mut().enumerate() {
        if index % (n + 1) == 0 {
            *slot = 1.0;
        }
    }
    Tensor::new(data, vec![n, n])
}

/// 单位阵按行拆出，供需要 `&[&[f32]]` 的构造辅助函数使用。
fn identity_rows(n: usize) -> Vec<Vec<f32>> {
    identity_matrix(n)
        .data()
        .chunks(n)
        .map(|chunk| chunk.to_vec())
        .collect()
}

fn zero_rows(rows: usize, cols: usize) -> Vec<Vec<f32>> {
    vec![vec![0.0; cols]; rows]
}

fn as_refs(rows: &[Vec<f32>]) -> Vec<&[f32]> {
    rows.iter().map(|row| row.as_slice()).collect()
}

#[allow(clippy::too_many_arguments)]
fn attention(
    wq: &[&[f32]],
    wk: &[&[f32]],
    wv: &[&[f32]],
    wo: &[&[f32]],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    theta: f32,
) -> Attention {
    Attention::new(
        linear(wq),
        linear(wk),
        linear(wv),
        linear(wo),
        num_heads,
        num_kv_heads,
        head_dim,
        theta,
    )
    .expect("构造 Attention 失败")
}

/// 取第 `index` 行。
fn row(tensor: &Tensor, index: usize) -> Vec<f32> {
    let columns = tensor.dims()[1];
    tensor.data()[index * columns..(index + 1) * columns].to_vec()
}

/// 逐元素比对 f32 实测与 f64 期望。
fn assert_close(actual: &[f32], expected: &[f64], tolerance: f32, context: &str) {
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
// Embedding
// ---------------------------------------------------------------------------

#[test]
fn embedding_looks_up_token_rows() {
    // 权重 [vocab=3, hidden=2]，第 i 行是 token i 的向量。
    let embedding = Embedding::new(Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]))
        .expect("构造失败");

    let output = embedding.forward(&[2, 0]).expect("前向失败");
    assert_eq!(output.dims(), &[2, 2]);
    assert_eq!(row(&output, 0), vec![5.0, 6.0]);
    assert_eq!(row(&output, 1), vec![1.0, 2.0]);
}

#[test]
fn embedding_rejects_out_of_range_and_negative_ids() {
    let embedding = Embedding::new(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])).unwrap();

    let negative = embedding.forward(&[-1]).expect_err("负 id 应被拒绝");
    assert!(negative.contains("-1"), "错误应指出 id: {negative}");

    let too_large = embedding.forward(&[2]).expect_err("越界 id 应被拒绝");
    assert!(too_large.contains('2'), "错误应指出 id: {too_large}");
}

#[test]
fn embedding_handles_empty_input() {
    let embedding = Embedding::new(Tensor::new(vec![1.0, 2.0], vec![1, 2])).unwrap();
    let output = embedding.forward(&[]).expect("空输入应成功");
    assert_eq!(output.dims(), &[0, 2]);
    assert!(output.is_empty());
}

#[test]
fn embedding_rejects_non_2d_weight() {
    let error = Embedding::new(Tensor::new(vec![1.0, 2.0], vec![2])).expect_err("一维权重应被拒绝");
    assert!(error.contains("2D"), "错误应说明需要二维: {error}");
}

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------

#[test]
fn linear_matches_hand_computed_matmul() {
    // W = [[1,2,3],[4,5,6]] 是 [out=2, in=3]；X = [[1,0,2],[0,1,3]]。
    // X @ W^T：
    //   行0 = [1*1+0*2+2*3, 1*4+0*5+2*6] = [7, 16]
    //   行1 = [0*1+1*2+3*3, 0*4+1*5+3*6] = [11, 23]
    let layer = linear(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
    let input = Tensor::new(vec![1.0, 0.0, 2.0, 0.0, 1.0, 3.0], vec![2, 3]);

    let output = layer.forward(&input).expect("前向失败");
    assert_eq!(output.dims(), &[2, 2]);
    assert_eq!(row(&output, 0), vec![7.0, 16.0]);
    assert_eq!(row(&output, 1), vec![11.0, 23.0]);
}

#[test]
fn linear_transposes_the_gguf_weight_layout() {
    // 这一用例锁定"权重按 [out, in] 存储、前向转置"的约定。
    // W = [[1,2],[3,4]]，X = I。正确约定下 out = X @ W^T = [[1,3],[2,4]]；
    // 若把 W 误当作 [in, out] 直接用，会得到 W 本身 [[1,2],[3,4]]。
    let layer = linear(&[&[1.0, 2.0], &[3.0, 4.0]]);
    let input = identity_matrix(2);

    let output = layer.forward(&input).expect("前向失败");
    assert_eq!(row(&output, 0), vec![1.0, 3.0]);
    assert_eq!(row(&output, 1), vec![2.0, 4.0]);
    // 内部权重确实是转置后的 [in, out]。
    assert_eq!(layer.weight().dims(), &[2, 2]);
    assert_eq!(layer.weight().data(), &[1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn linear_adds_bias() {
    // W = [[1,2],[3,4]]，bias = [10, 100]，X = [[1,1]]。
    // out = [1+2+10, 3+4+100] = [13, 107]
    let layer = linear_with_bias(&[&[1.0, 2.0], &[3.0, 4.0]], vec![10.0, 100.0]);
    let input = Tensor::new(vec![1.0, 1.0], vec![1, 2]);

    let output = layer.forward(&input).expect("前向失败");
    assert_eq!(row(&output, 0), vec![13.0, 107.0]);
}

#[test]
fn linear_rejects_mismatched_features_and_ranks() {
    let layer = linear(&[&[1.0, 2.0], &[3.0, 4.0]]);

    let wrong_features = layer
        .forward(&Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]))
        .expect_err("特征数不符应被拒绝");
    assert!(
        wrong_features.contains("features"),
        "错误应说明特征数: {wrong_features}"
    );

    let wrong_rank = layer
        .forward(&Tensor::new(vec![1.0, 2.0], vec![2]))
        .expect_err("一维输入应被拒绝");
    assert!(
        wrong_rank.contains("2D"),
        "错误应说明需要二维: {wrong_rank}"
    );
}

#[test]
fn linear_rejects_mismatched_bias_length() {
    let error = Linear::new(
        weight(&[&[1.0, 2.0], &[3.0, 4.0]]),
        Some(Tensor::new(vec![1.0], vec![1])),
    )
    .expect_err("偏置长度不符应被拒绝");
    assert!(
        error.contains("out_features"),
        "错误应指出 out_features: {error}"
    );
}

// ---------------------------------------------------------------------------
// RmsNorm
// ---------------------------------------------------------------------------

#[test]
fn rms_norm_matches_hand_computed_values() {
    // x = [3, 4]，rms = sqrt((9+16)/2) = sqrt(12.5) = 3.5355339。
    // 权重 [1,1]：out = [0.8485281, 1.1313709]
    // 权重 [2,1]：out = [1.6970563, 1.1313709]
    let input = Tensor::new(vec![3.0, 4.0], vec![1, 2]);

    let plain = norm(&[1.0, 1.0]).forward(&input).expect("前向失败");
    assert_close(
        &row(&plain, 0),
        &[3.0 / 12.5f64.sqrt(), 4.0 / 12.5f64.sqrt()],
        1e-6,
        "权重 [1,1]",
    );

    let scaled = norm(&[2.0, 1.0]).forward(&input).expect("前向失败");
    assert_close(
        &row(&scaled, 0),
        &[2.0 * 3.0 / 12.5f64.sqrt(), 4.0 / 12.5f64.sqrt()],
        1e-6,
        "权重 [2,1]",
    );
}

#[test]
fn rms_norm_normalizes_each_row_independently() {
    // 行0 = [3,4] → rms = sqrt(12.5)；行1 = [1,1] → rms = 1。
    let input = Tensor::new(vec![3.0, 4.0, 1.0, 1.0], vec![2, 2]);

    let output = norm(&[1.0, 1.0]).forward(&input).expect("前向失败");
    assert_eq!(output.dims(), &[2, 2]);
    assert_close(
        &row(&output, 0),
        &[3.0 / 12.5f64.sqrt(), 4.0 / 12.5f64.sqrt()],
        1e-6,
        "行0",
    );
    assert_eq!(row(&output, 1), vec![1.0, 1.0]);
}

#[test]
fn rms_norm_applies_eps() {
    // x = [2, 0]，eps = 4：rms = sqrt(4/2 + 4) = sqrt(6) = 2.4494897。
    let input = Tensor::new(vec![2.0, 0.0], vec![1, 2]);
    let layer = RmsNorm::new(Tensor::new(vec![1.0, 1.0], vec![2]), 4.0).unwrap();

    let output = layer.forward(&input).expect("前向失败");
    assert_close(&row(&output, 0), &[2.0 / 6.0f64.sqrt(), 0.0], 1e-6, "eps=4");
}

#[test]
fn rms_norm_rejects_non_1d_weight() {
    let error =
        RmsNorm::new(Tensor::new(vec![1.0, 1.0], vec![1, 2]), 0.0).expect_err("二维权重应被拒绝");
    assert!(error.contains("1D"), "错误应说明需要一维: {error}");
}

// ---------------------------------------------------------------------------
// Mlp
// ---------------------------------------------------------------------------

#[test]
fn mlp_computes_swiglu() {
    // gate = [[1,2],[3,4]]，up = [[0,1],[1,0]]，down = [[1,2],[3,4]]，X = [[1,0]]。
    // gate_out = X @ gate^T = [1*1+0*2, 1*3+0*4] = [1, 3]
    // up_out   = X @ up^T   = [1*0+0*1, 1*1+0*0] = [0, 1]
    // hidden   = silu([1,3]) * [0,1] = [0, silu(3)]
    // down_out = hidden @ down^T，down^T = [[1,3],[2,4]]：
    //   [0*1+silu(3)*2, 0*3+silu(3)*4] = [2*silu(3), 4*silu(3)]
    let silu_three = 3.0 / (1.0 + (-3.0f64).exp());
    let mlp = Mlp::new(
        linear(&[&[1.0, 2.0], &[3.0, 4.0]]),
        linear(&[&[0.0, 1.0], &[1.0, 0.0]]),
        linear(&[&[1.0, 2.0], &[3.0, 4.0]]),
    )
    .expect("构造失败");

    let output = mlp
        .forward(&Tensor::new(vec![1.0, 0.0], vec![1, 2]))
        .expect("前向失败");
    assert_eq!(output.dims(), &[1, 2]);
    assert_close(
        &row(&output, 0),
        &[2.0 * silu_three, 4.0 * silu_three],
        1e-6,
        "SwiGLU 输出",
    );
}

#[test]
fn mlp_rejects_incompatible_projections() {
    // gate 与 up 形状不同。
    let mismatched = Mlp::new(
        linear(&[&[1.0, 2.0], &[3.0, 4.0]]),
        linear(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]),
        linear(&[&[1.0, 2.0], &[3.0, 4.0]]),
    )
    .expect_err("形状不一致应被拒绝");
    assert!(
        mismatched.contains("same shape"),
        "错误应说明形状: {mismatched}"
    );

    // down_proj 的输入特征数与中间层大小不符：gate/up 的中间层是 2，
    // 而 down = [[1,0,0],[0,1,0]] 是 [out=2, in=3]，输入特征数为 3。
    let wrong_down = Mlp::new(
        linear(&[&[1.0, 2.0], &[3.0, 4.0]]),
        linear(&[&[1.0, 2.0], &[3.0, 4.0]]),
        linear(&[&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]]),
    )
    .expect_err("down_proj 输入维度不符应被拒绝");
    assert!(
        wrong_down.contains("intermediate"),
        "错误应指出中间层大小: {wrong_down}"
    );
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

#[test]
fn attention_single_token_round_trips_through_the_pipeline() {
    // seq_len=1 时 RoPE 旋转角为 0、因果掩码无项可遮，整条链路退化为恒等映射：
    // 每个头的 softmax 输入只有一个元素故必为 1，注意力输出就是 v 本身。
    // hidden=4, heads=2, head_dim=2，wq/wk/wv 取单位阵。
    // 头0 取 v[0..2] = (1,2)，头1 取 v[2..4] = (3,4)，合并为 [1,2,3,4]。
    // wo 把第 0、2 列分别搬到输出 0、1 位：out = [1, 3, 0, 0]。
    let identity_data = identity_rows(4);
    let identity = as_refs(&identity_data);
    let layer = attention(
        &identity,
        &identity,
        &identity,
        &[
            &[1.0, 0.0, 0.0, 0.0],
            &[0.0, 0.0, 1.0, 0.0],
            &[0.0, 0.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0, 0.0],
        ],
        2,
        2,
        2,
        10000.0,
    );

    let output = layer
        .forward(&Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]), 0)
        .expect("前向失败");
    assert_eq!(output.dims(), &[1, 4]);
    assert_eq!(row(&output, 0), vec![1.0, 3.0, 0.0, 0.0]);
}

#[test]
fn attention_keeps_each_head_in_its_own_output_columns() {
    // 与上一条相同，但 wo 取单位阵：合并顺序若把两个头调换，输出会变成 [3,4,1,2]。
    let identity_data = identity_rows(4);
    let identity = as_refs(&identity_data);
    let layer = Attention::new(
        linear(&identity),
        linear(&identity),
        linear(&identity),
        linear(&identity),
        2,
        2,
        2,
        10000.0,
    )
    .expect("构造失败");

    let output = layer
        .forward(&Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]), 0)
        .expect("前向失败");
    assert_eq!(row(&output, 0), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn attention_masks_future_positions() {
    // 位置 0 只能看到自己，因此其输出恒等于 v[0] = x[0]，与位置 1 的取值无关。
    // 反过来，改动 x[0] 必须改变位置 1 的输出。
    let identity_data = identity_rows(2);
    let identity = as_refs(&identity_data);
    let layer = Attention::new(
        linear(&identity),
        linear(&identity),
        linear(&identity),
        linear(&identity),
        1,
        1,
        2,
        10000.0,
    )
    .expect("构造失败");

    let first = layer
        .forward(&Tensor::new(vec![1.0, 0.0, 7.0, 9.0], vec![2, 2]), 0)
        .expect("前向失败");
    let second = layer
        .forward(&Tensor::new(vec![1.0, 0.0, -3.0, 5.0], vec![2, 2]), 0)
        .expect("前向失败");

    assert_eq!(row(&first, 0), vec![1.0, 0.0], "位置 0 只看自己");
    assert_eq!(
        row(&first, 0),
        row(&second, 0),
        "位置 1 的输入不得影响位置 0 的输出"
    );
    assert_ne!(
        row(&first, 1),
        row(&second, 1),
        "位置 1 必须能影响自己的输出"
    );

    // 改动 x[0] 应改变位置 1 的输出。这里刻意让两个位置的量级相当：自注意力
    // 分数是 |x[1]|²，若 x[1] 远大于 x[0]，softmax 会退化成只看自己，
    // 对位置 0 的依赖被淹没，断言就失去意义。
    let weak_first = layer
        .forward(&Tensor::new(vec![10.0, 0.0, 1.0, 1.0], vec![2, 2]), 0)
        .expect("前向失败");
    let strong_first = layer
        .forward(&Tensor::new(vec![20.0, 0.0, 1.0, 1.0], vec![2, 2]), 0)
        .expect("前向失败");
    assert_ne!(
        row(&weak_first, 1),
        row(&strong_first, 1),
        "位置 1 应关注位置 0"
    );
}

#[test]
fn attention_shares_kv_heads_across_query_heads() {
    // hidden=4, heads=2, head_dim=2, num_kv_heads=1 → group=2，
    // query 头 0 与头 1 都复用唯一的 KV 头 0。
    // wk = wv = [[1,0,0,0],[0,1,0,0]] 使 kv 头 0 恒为 (x0, x1) = (1,2)。
    // 两个 query 头的 softmax 都只有一个有效元素，输出均为 (1,2)，合并为 [1,2,1,2]。
    let identity_data = identity_rows(4);
    let identity = as_refs(&identity_data);
    let kv_pick_first_data = [vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
    let kv_pick_first = as_refs(&kv_pick_first_data);

    let layer = Attention::new(
        linear(&identity),
        linear(&kv_pick_first),
        linear(&kv_pick_first),
        linear(&identity),
        2,
        1,
        2,
        10000.0,
    )
    .expect("构造失败");

    let output = layer
        .forward(&Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]), 0)
        .expect("前向失败");
    assert_eq!(row(&output, 0), vec![1.0, 2.0, 1.0, 2.0]);
}

#[test]
fn attention_matches_an_independent_reference_implementation() {
    // 三个配置覆盖 MHA、GQA 与非零 start_pos；权重用确定性的小数值铺满，
    // 使每个头、每个位置都参与运算。
    let cases = [
        (2usize, 4usize, 2usize, 2usize, 0usize), // seq, hidden, heads, kv, start_pos
        (3, 4, 2, 1, 0),                          // GQA：group = 2
        (2, 6, 3, 3, 2),                          // MHA，start_pos 非零
    ];

    for (seq_len, hidden, heads, kv_heads, start_pos) in cases {
        let head_dim = hidden / heads;
        let wq = reference_matrix(hidden, hidden, 0x51AA_0001);
        let wk = reference_matrix(kv_heads * head_dim, hidden, 0x51AA_0002);
        let wv = reference_matrix(kv_heads * head_dim, hidden, 0x51AA_0003);
        let wo = reference_matrix(hidden, hidden, 0x51AA_0004);
        let input = reference_input(seq_len, hidden);

        let layer = Attention::new(
            Linear::new(tensor(&wq, vec![hidden, hidden]), None).unwrap(),
            Linear::new(tensor(&wk, vec![kv_heads * head_dim, hidden]), None).unwrap(),
            Linear::new(tensor(&wv, vec![kv_heads * head_dim, hidden]), None).unwrap(),
            Linear::new(tensor(&wo, vec![hidden, hidden]), None).unwrap(),
            heads,
            kv_heads,
            head_dim,
            10000.0,
        )
        .expect("构造失败");

        let input_tensor = tensor(&input, vec![seq_len, hidden]);
        let output = layer.forward(&input_tensor, start_pos).expect("前向失败");
        assert_eq!(
            output.dims(),
            &[seq_len, hidden],
            "配置 seq={seq_len} hidden={hidden} heads={heads} kv={kv_heads}"
        );

        let expected = reference_attention(
            &input, &wq, &wk, &wv, &wo, heads, kv_heads, head_dim, start_pos, 10000.0,
        );
        let context =
            format!("seq={seq_len} hidden={hidden} heads={heads} kv={kv_heads} start={start_pos}");
        for (position, expected_row) in expected.iter().enumerate() {
            assert_close(&row(&output, position), expected_row, 1e-4, &context);
        }
    }
}

#[test]
fn attention_uses_interleaved_rope_pairs() {
    // 直接锁定 RoPE 的元素配对约定，因为它是接真实 GGUF 模型时最容易出错的一环。
    // head_dim=4、start_pos=1、theta=10000：
    //   对 0 频率 = theta^0 = 1，角度 1 rad
    //   对 1 频率 = theta^(-0.5) = 0.01，角度 0.01 rad
    // 交错对（本实现）旋转 (x0,x1) 与 (x2,x3)：
    //   [1,2] → [1*cos1 - 2*sin1, 1*sin1 + 2*cos1] = [-1.1426397, 1.9220756]
    //   [3,4] → [3*cos.01 - 4*sin.01, 3*sin.01 + 4*cos.01] = [2.9598507, 4.0297995]
    // HuggingFace 的 rotate_half 会旋转 (x0,x2) 与 (x1,x3)，得到
    // [-1.9841106, 1.9599007, 2.4633779, 4.0197997]——与本实现不同。
    // 若这个用例失败，说明有人改成了对半拆分，接真实模型前必须先核对 llama.cpp 的
    // GGML_ROPE_TYPE_NORM。
    let cos1 = 1.0f64.cos();
    let sin1 = 1.0f64.sin();
    let cos001 = 0.01f64.cos();
    let sin001 = 0.01f64.sin();

    let q = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
    let rotated = q.rope(4, 1, 10000.0).expect("RoPE 失败");

    let expected = [
        1.0 * cos1 - 2.0 * sin1,
        1.0 * sin1 + 2.0 * cos1,
        3.0 * cos001 - 4.0 * sin001,
        3.0 * sin001 + 4.0 * cos001,
    ];
    assert_close(rotated.data(), &expected, 1e-6, "交错对 RoPE");

    // 明确排除对半拆分的结果。
    let split_half = [
        1.0 * cos1 - 3.0 * sin1,
        2.0 * cos001 - 4.0 * sin001,
        1.0 * sin1 + 3.0 * cos1,
        2.0 * sin001 + 4.0 * cos001,
    ];
    assert!(
        (rotated.data()[1] - split_half[1] as f32).abs() > 1e-3,
        "实现不应对半拆分：第 1 个元素为 {}，对半拆分会得到 {}",
        rotated.data()[1],
        split_half[1]
    );
}

#[test]
fn attention_rejects_mismatched_head_configuration() {
    // num_heads 不能被 num_kv_heads 整除。
    let error = Attention::new(
        linear(&[&[1.0, 0.0], &[0.0, 1.0]]),
        linear(&[&[1.0, 0.0], &[0.0, 1.0]]),
        linear(&[&[1.0, 0.0], &[0.0, 1.0]]),
        linear(&[&[1.0, 0.0], &[0.0, 1.0]]),
        3,
        2,
        2,
        10000.0,
    )
    .expect_err("不可整除的头数应被拒绝");
    assert!(error.contains("divisible"), "错误应说明整除关系: {error}");

    // head_dim 与投影输出不符。
    let error = Attention::new(
        linear(&[&[1.0, 0.0], &[0.0, 1.0]]),
        linear(&[&[1.0, 0.0], &[0.0, 1.0]]),
        linear(&[&[1.0, 0.0], &[0.0, 1.0]]),
        linear(&[&[1.0, 0.0], &[0.0, 1.0]]),
        1,
        1,
        3,
        10000.0,
    )
    .expect_err("head_dim 不符应被拒绝");
    assert!(
        error.contains("num_heads*head_dim"),
        "错误应指出头维度乘积: {error}"
    );

    // 输入不是二维。
    let layer = attention(
        &[&[1.0, 0.0], &[0.0, 1.0]],
        &[&[1.0, 0.0], &[0.0, 1.0]],
        &[&[1.0, 0.0], &[0.0, 1.0]],
        &[&[1.0, 0.0], &[0.0, 1.0]],
        1,
        1,
        2,
        10000.0,
    );
    let error = layer
        .forward(&Tensor::new(vec![1.0, 2.0], vec![2]), 0)
        .expect_err("一维输入应被拒绝");
    assert!(error.contains("2D"), "错误应说明需要二维: {error}");
}

// ---------------------------------------------------------------------------
// TransformerBlock
// ---------------------------------------------------------------------------

#[test]
fn block_adds_both_residuals() {
    // 注意力是全等的（seq_len=1 时 RoPE 与掩码都不起作用），MLP 权重全零。
    // 输入先经 input_layernorm：[3,4] → [3,4]/sqrt(12.5) = [0.8485281, 1.1313709]。
    // block 输出 = x + attn(norm(x)) + mlp(...) = [3,4] + [0.8485281, 1.1313709] + 0
    //            = [3.8485281, 5.1313709]
    // 两个残差都必须存在：少了第一个会得到 [0.8485281, 1.1313709]，
    // 少了第二个会同样丢掉 MLP 的增量（这里 MLP 为零，由下一条用例补足）。
    let rms = 12.5f64.sqrt();
    let identity_data = identity_rows(2);
    let identity = as_refs(&identity_data);
    let zeros_data = zero_rows(2, 2);
    let zeros = as_refs(&zeros_data);

    let block = TransformerBlock {
        input_layernorm: norm(&[1.0, 1.0]),
        self_attn: Attention::new(
            linear(&identity),
            linear(&identity),
            linear(&identity),
            linear(&identity),
            1,
            1,
            2,
            10000.0,
        )
        .unwrap(),
        post_attention_layernorm: norm(&[1.0, 1.0]),
        mlp: Mlp::new(linear(&zeros), linear(&zeros), linear(&zeros)).unwrap(),
    };

    let output = block
        .forward(&Tensor::new(vec![3.0, 4.0], vec![1, 2]), 0)
        .expect("前向失败");
    assert_close(
        &row(&output, 0),
        &[3.0 + 3.0 / rms, 4.0 + 4.0 / rms],
        1e-6,
        "第一个残差后的输出",
    );
}

#[test]
fn block_passes_the_mlp_output_through_the_second_residual() {
    // 注意力全零 → 第一个残差不产生增量；MLP 用"取首元素"的投影，
    // 使输出可手工推算。
    // 归一化权重取 rms 本身，把 post_attention_layernorm 变成恒等映射。
    // x = [3,4]，rms = sqrt(12.5) = 3.5355339，normed = x。
    // gate = up = down = [[1,0],[0,0]]：
    //   gate_out = up_out = [3, 0]
    //   hidden = [silu(3)*3, 0] = [2.8611257, 0]
    //   down_out = [2.8611257, 0]
    // out = x + down_out = [5.8611257, 4]
    let rms = 12.5f64.sqrt();
    let zeros_data = zero_rows(2, 2);
    let zeros = as_refs(&zeros_data);
    let pick_first_data = [vec![1.0, 0.0], vec![0.0, 0.0]];
    let pick_first = as_refs(&pick_first_data);

    let block = TransformerBlock {
        input_layernorm: norm(&[1.0, 1.0]),
        self_attn: Attention::new(
            linear(&zeros),
            linear(&zeros),
            linear(&zeros),
            linear(&zeros),
            1,
            1,
            2,
            10000.0,
        )
        .unwrap(),
        post_attention_layernorm: RmsNorm::new(
            Tensor::new(vec![rms as f32, rms as f32], vec![2]),
            0.0,
        )
        .unwrap(),
        mlp: Mlp::new(
            linear(&pick_first),
            linear(&pick_first),
            linear(&pick_first),
        )
        .unwrap(),
    };

    let output = block
        .forward(&Tensor::new(vec![3.0, 4.0], vec![1, 2]), 0)
        .expect("前向失败");
    let silu_three = 3.0 / (1.0 + (-3.0f64).exp());
    assert_close(
        &row(&output, 0),
        &[3.0 + silu_three * 3.0, 4.0],
        1e-5,
        "第二个残差后的输出",
    );
}

#[test]
fn block_rejects_non_2d_input() {
    let identity_data = identity_rows(2);
    let identity = as_refs(&identity_data);
    let block = TransformerBlock {
        input_layernorm: norm(&[1.0, 1.0]),
        self_attn: Attention::new(
            linear(&identity),
            linear(&identity),
            linear(&identity),
            linear(&identity),
            1,
            1,
            2,
            10000.0,
        )
        .unwrap(),
        post_attention_layernorm: norm(&[1.0, 1.0]),
        mlp: Mlp::new(linear(&identity), linear(&identity), linear(&identity)).unwrap(),
    };

    let error = block
        .forward(&Tensor::new(vec![1.0, 2.0], vec![2]), 0)
        .expect_err("一维输入应被拒绝");
    assert!(error.contains("2D"), "错误应说明需要二维: {error}");
}

// ---------------------------------------------------------------------------
// 独立参考实现
// ---------------------------------------------------------------------------

/// 把参考实现的 `[rows][cols]` 摊平成一维 f32 缓冲。
///
/// 参考实现用 f64 累加以压低舍入误差，喂给待测代码前再降到 f32——这与生产路径
/// 上"权重以 f32 存储"一致，因此两侧差异只应来自累加顺序。
fn flatten(rows: &[Vec<f64>]) -> Vec<f32> {
    rows.iter().flatten().map(|&value| value as f32).collect()
}

fn tensor(rows: &[Vec<f64>], dims: Vec<usize>) -> Tensor {
    Tensor::new(flatten(rows), dims)
}

/// 确定性小数值矩阵：`[rows][cols]`，值在 [-0.5, 0.5) 内且可复现。
fn reference_matrix(rows: usize, cols: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut state = seed;
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    ((state >> 40) as f64) / ((1u32 << 24) as f64) - 0.5
                })
                .collect()
        })
        .collect()
}

fn reference_input(seq_len: usize, hidden: usize) -> Vec<Vec<f64>> {
    reference_matrix(seq_len, hidden, 0x51AA_0005)
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(&a, &b)| a * b).sum()
}

/// 独立参考实现：按公式直书多头注意力，f64、朴素循环、不调用待测代码。
///
/// RoPE 采用与 `ops::rope` 相同的交错对约定——这是经上游源码确认的实现选择，
/// 参考实现复现该选择是为了校验索引与形状处理，而非重新决定约定本身。
#[allow(clippy::too_many_arguments)]
fn reference_attention(
    input: &[Vec<f64>],
    wq: &[Vec<f64>],
    wk: &[Vec<f64>],
    wv: &[Vec<f64>],
    wo: &[Vec<f64>],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    start_pos: usize,
    theta: f64,
) -> Vec<Vec<f64>> {
    let seq_len = input.len();
    let group = num_heads / num_kv_heads;
    let hidden = num_heads * head_dim;

    let mut q = vec![vec![0.0; hidden]; seq_len];
    let mut k = vec![vec![0.0; num_kv_heads * head_dim]; seq_len];
    let mut v = vec![vec![0.0; num_kv_heads * head_dim]; seq_len];
    for (position, token) in input.iter().enumerate() {
        for (out, row) in wq.iter().enumerate() {
            q[position][out] = dot(token, row);
        }
        for (out, row) in wk.iter().enumerate() {
            k[position][out] = dot(token, row);
        }
        for (out, row) in wv.iter().enumerate() {
            v[position][out] = dot(token, row);
        }
    }

    let rotate = |tensor: &mut Vec<Vec<f64>>, head_count: usize| {
        for (position, token) in tensor.iter_mut().enumerate() {
            let angle_base = (start_pos + position) as f64;
            for head in 0..head_count {
                for pair in 0..head_dim / 2 {
                    let frequency = theta.powf(-2.0 * pair as f64 / head_dim as f64);
                    let (sin, cos) = (angle_base * frequency).sin_cos();
                    let first = head * head_dim + 2 * pair;
                    let second = first + 1;
                    let (a, b) = (token[first], token[second]);
                    token[first] = a * cos - b * sin;
                    token[second] = a * sin + b * cos;
                }
            }
        }
    };
    rotate(&mut q, num_heads);
    rotate(&mut k, num_kv_heads);

    let scale = 1.0 / (head_dim as f64).sqrt();
    let mut merged = vec![vec![0.0; hidden]; seq_len];
    for head in 0..num_heads {
        let kv_head = head / group;
        for query in 0..seq_len {
            let mut scores = vec![0.0; seq_len];
            for (key, score) in scores.iter_mut().enumerate() {
                *score = if key > query {
                    f64::NEG_INFINITY
                } else {
                    let mut sum = 0.0;
                    for dim in 0..head_dim {
                        sum += q[query][head * head_dim + dim] * k[key][kv_head * head_dim + dim];
                    }
                    sum * scale
                };
            }
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exponentials: Vec<f64> = scores.iter().map(|&s| (s - maximum).exp()).collect();
            let total: f64 = exponentials.iter().sum();
            for (dim, output) in merged[query]
                .iter_mut()
                .skip(head * head_dim)
                .take(head_dim)
                .enumerate()
            {
                *output = (0..seq_len)
                    .map(|key| exponentials[key] / total * v[key][kv_head * head_dim + dim])
                    .sum();
            }
        }
    }

    let mut output = vec![vec![0.0; wo.len()]; seq_len];
    for (position, token) in merged.iter().enumerate() {
        for (out, row) in wo.iter().enumerate() {
            output[position][out] = dot(token, row);
        }
    }
    output
}
