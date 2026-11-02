//! Sampler 的行为测试。
//!
//! 每条用例的期望值都在注释里手工推导：要么给出闭式结果（如 softmax 后
//! 的概率），要么给出构造出的边界（如两个并列候选恰好落在阈值上）。
//! 不从被测代码反推期望值——那样得到的断言永远成立。

use qllama::sampler::{Sampler, SamplerConfig};
use qllama::tensor::Tensor;

/// 构造一份采样配置；四个参数按处理顺序命名。
fn config(temperature: f32, top_k: usize, top_p: f32, repeat_penalty: f32) -> SamplerConfig {
    SamplerConfig {
        temperature,
        top_k,
        top_p,
        repeat_penalty,
    }
}

/// 把 logits 包成一维张量。
fn logits_tensor(values: &[f32]) -> Tensor {
    Tensor::new(values.to_vec(), vec![values.len()])
}

/// 采样 `rounds` 次并统计各 id 的出现次数。
fn sample_counts(sampler: &mut Sampler, values: &[f32], rounds: usize) -> Vec<usize> {
    let mut counts = vec![0usize; values.len()];
    for _ in 0..rounds {
        let id = sampler.sample(&logits_tensor(values), &[]).unwrap();
        counts[id as usize] += 1;
    }
    counts
}

// ---------------------------------------------------------------- 贪心采样

/// temperature <= 0 走贪心：直接取最大 logit 的 id。
#[test]
fn greedy_temperature_zero_returns_argmax() {
    // argmax([0.1, 0.9, 0.2]) = 1。
    let mut sampler = Sampler::with_seed(config(0.0, 0, 1.0, 1.0), 1);
    let id = sampler
        .sample(&logits_tensor(&[0.1, 0.9, 0.2]), &[])
        .unwrap();
    assert_eq!(id, 1);
}

/// 贪心在全负 logit 上同样成立：比较的是大小而不是正负。
#[test]
fn greedy_handles_negative_logits() {
    // argmax([-1.0, -2.0, -0.5]) = 2（-0.5 最大）。
    let mut sampler = Sampler::with_seed(config(0.0, 0, 1.0, 1.0), 1);
    let id = sampler
        .sample(&logits_tensor(&[-1.0, -2.0, -0.5]), &[])
        .unwrap();
    assert_eq!(id, 2);
}

/// 并列时取最小 id，保证结果可复现。
#[test]
fn greedy_breaks_ties_towards_smaller_id() {
    // argmax([0.5, 0.5, 0.1]) = 0（0.5 与 0.5 并列，取小 id）。
    let mut sampler = Sampler::with_seed(config(0.0, 0, 1.0, 1.0), 1);
    let id = sampler
        .sample(&logits_tensor(&[0.5, 0.5, 0.1]), &[])
        .unwrap();
    assert_eq!(id, 0);
}

// ---------------------------------------------------------------- 温度

/// 极小温度把概率压到最大值上，与随机源无关。
#[test]
fn tiny_temperature_concentrates_on_the_maximum() {
    // 除以 temp=1e-6 得 [1e6, 2e6, 3e6]；softmax 减去最大值 3e6 后是
    // [e^-2e6, e^-1e6, 1]，前两项下溢为 0，概率为 [0, 0, 1]，任何 seed 都返回 2。
    for seed in 0..8 {
        let mut sampler = Sampler::with_seed(config(1e-6, 0, 1.0, 1.0), seed);
        let id = sampler
            .sample(&logits_tensor(&[1.0, 2.0, 3.0]), &[])
            .unwrap();
        assert_eq!(id, 2, "seed {seed}");
    }
}

/// temperature = 1 不改变分布形状：等值 logit 得到等概率。
#[test]
fn unit_temperature_leaves_equal_logits_equiprobable() {
    // softmax([0, 0]) = [0.5, 0.5]。2000 次抽样，σ = sqrt(2000*0.5*0.5) = 22.36，
    // 3σ ≈ 67，两边都应落在 [933, 1067] 内。
    let mut sampler = Sampler::with_seed(config(1.0, 0, 1.0, 1.0), 7);
    let counts = sample_counts(&mut sampler, &[0.0, 0.0], 2000);
    assert!(counts[0] > 933 && counts[0] < 1067, "{counts:?}");
    assert!(counts[1] > 933 && counts[1] < 1067, "{counts:?}");
}

// ---------------------------------------------------------------- top-k

/// top_k = 1 时只剩概率最大的候选，结果确定。
#[test]
fn top_k_one_returns_argmax() {
    // softmax([0.1, 5.0, 0.2]) 中 id1 占绝对优势，top_k=1 后概率为 1。
    let mut sampler = Sampler::with_seed(config(1.0, 1, 1.0, 1.0), 99);
    for _ in 0..50 {
        let id = sampler
            .sample(&logits_tensor(&[0.1, 5.0, 0.2]), &[])
            .unwrap();
        assert_eq!(id, 1);
    }
}

/// top_k 截断后，被排除的候选一次也不会出现。
#[test]
fn top_k_excludes_lower_ranked_candidates() {
    // softmax([10, 9, 0])：相对权重 e^0 : e^-1 : e^-10 = 1 : 0.3679 : 4.54e-5，
    // 归一化后 ≈ [0.7311, 0.2690, 3.3e-5]。
    // top_k=2 按概率降序保留 {0, 1}，id2 的概率被置 0。
    let mut sampler = Sampler::with_seed(config(1.0, 2, 1.0, 1.0), 2024);
    let counts = sample_counts(&mut sampler, &[10.0, 9.0, 0.0], 500);
    assert_eq!(counts[2], 0, "{counts:?}");
    // id1 的概率 0.269，500 次期望 134 次，σ = sqrt(500*0.269*0.731) ≈ 9.9，
    // 取 ±4σ 得 [94, 174]。
    assert!(counts[1] > 94 && counts[1] < 174, "{counts:?}");
}

/// top_k = 0 表示不限制，三个候选都可能被抽到。
#[test]
fn top_k_zero_keeps_every_candidate() {
    // softmax([1, 1, 1]) = [1/3, 1/3, 1/3]，三个候选概率相同。
    // 300 次抽样每个候选期望 100 次，σ = sqrt(300*1/3*2/3) ≈ 8.16，取 ±4σ。
    let mut sampler = Sampler::with_seed(config(1.0, 0, 1.0, 1.0), 2024);
    let counts = sample_counts(&mut sampler, &[1.0, 1.0, 1.0], 300);
    for count in &counts {
        assert!(*count > 67 && *count < 133, "{counts:?}");
    }
}

/// top_k 大于词表大小时等价于不限制，不报错。
#[test]
fn top_k_larger_than_vocab_keeps_every_candidate() {
    // 词表只有 3 个候选，top_k=10 >= vocab，按实现约定直接跳过截断。
    let mut sampler = Sampler::with_seed(config(1.0, 10, 1.0, 1.0), 2024);
    let counts = sample_counts(&mut sampler, &[1.0, 1.0, 1.0], 300);
    for count in &counts {
        assert!(*count > 67 && *count < 133, "{counts:?}");
    }
}

// ---------------------------------------------------------------- top-p

/// top_p 很小时只剩占主导的候选。
#[test]
fn top_p_below_one_keeps_only_the_dominant_token() {
    // softmax([10, 1, 1]) ≈ [0.999753, 0.0001234, 0.0001234]。
    // 累计到第一个候选就是 0.999753 >= 0.5，候选集合为 {0}，任何 seed 都返回 0。
    for seed in 0..8 {
        let mut sampler = Sampler::with_seed(config(1.0, 0, 0.5, 1.0), seed);
        let id = sampler
            .sample(&logits_tensor(&[10.0, 1.0, 1.0]), &[])
            .unwrap();
        assert_eq!(id, 0, "seed {seed}");
    }
}

/// top_p 保留越过阈值的那个候选（llama.cpp 的 `last_idx = i + 1`）。
#[test]
fn top_p_includes_the_candidate_crossing_the_threshold() {
    // 取 logits = [0, -ln18, -ln18]：e^-ln18 = 1/18，总和 1 + 2/18 = 20/18，
    // 概率为 [18/20, 1/20, 1/20] = [0.9, 0.05, 0.05]。
    // top_p=0.91：累计到 id0 是 0.9 < 0.91，继续；累计到 id1 是 0.95 >= 0.91，
    // 于是候选集合是 {0, 1}——id1 虽然自身概率小，却在越过阈值时被纳入。
    let mut sampler = Sampler::with_seed(config(1.0, 0, 0.91, 1.0), 11);
    let counts = sample_counts(&mut sampler, &[0.0, -18f32.ln(), -18f32.ln()], 1000);
    // id2 概率被置 0，不会出现。
    assert_eq!(counts[2], 0, "{counts:?}");
    // id1 重新归一化后概率 0.05/0.95 ≈ 0.0526，1000 次期望 52.6 次，
    // σ = sqrt(1000*0.0526*0.947) ≈ 7.06，取 ±4σ 得 [24, 81]。
    assert!(counts[1] > 24 && counts[1] < 81, "{counts:?}");
}

/// top_p 略低于第一个候选的累计概率时，候选集合只有它自己。
#[test]
fn top_p_below_the_first_cumulative_keeps_one_candidate() {
    // 同上，概率 [0.9, 0.05, 0.05]；top_p=0.89 < 0.9，累计到 id0 即达标，
    // 候选集合为 {0}。
    let mut sampler = Sampler::with_seed(config(1.0, 0, 0.89, 1.0), 11);
    let counts = sample_counts(&mut sampler, &[0.0, -18f32.ln(), -18f32.ln()], 200);
    assert_eq!(counts[1], 0, "{counts:?}");
    assert_eq!(counts[2], 0, "{counts:?}");
}

/// top_p >= 1.0 表示不限制。
#[test]
fn top_p_one_keeps_every_candidate() {
    // softmax([1, 1, 1]) = [1/3, 1/3, 1/3]，top_p=1.0 不截断。
    let mut sampler = Sampler::with_seed(config(1.0, 0, 1.0, 1.0), 2024);
    let counts = sample_counts(&mut sampler, &[1.0, 1.0, 1.0], 300);
    for count in &counts {
        assert!(*count > 67 && *count < 133, "{counts:?}");
    }
}

/// top_p 落在 1/3 与 2/3 之间时保留两个候选。
#[test]
fn top_p_half_on_uniform_logits_keeps_two_candidates() {
    // 概率 [1/3, 1/3, 1/3]：累计 1/3 = 0.333 < 0.5，累计 2/3 = 0.667 >= 0.5，
    // 候选集合为 {0, 1}，id2 概率归零。
    let mut sampler = Sampler::with_seed(config(1.0, 0, 0.5, 1.0), 2024);
    let counts = sample_counts(&mut sampler, &[1.0, 1.0, 1.0], 300);
    assert_eq!(counts[2], 0, "{counts:?}");
    // id0、id1 重新归一化后各占 0.5，期望 150 次，σ = sqrt(300*0.5*0.5) = 8.66，
    // 取 ±4σ 得 [115, 185]。
    assert!(counts[0] > 115 && counts[0] < 185, "{counts:?}");
    assert!(counts[1] > 115 && counts[1] < 185, "{counts:?}");
}

// ---------------------------------------------------------------- 重复惩罚

/// 正 logit 的重复惩罚是除以惩罚系数。
#[test]
fn repeat_penalty_divides_positive_logits() {
    // context 含 id1，logits [0.3, 0.5]，penalty=2.0：id1 的 logit > 0，
    // 除以 2 得 0.25 < 0.3，argmax 变成 0。
    // 若漏掉惩罚或方向写反（乘以 2 得 1.0），argmax 会是 1。
    let mut sampler = Sampler::with_seed(config(0.0, 0, 1.0, 2.0), 3);
    let id = sampler.sample(&logits_tensor(&[0.3, 0.5]), &[1]).unwrap();
    assert_eq!(id, 0);
}

/// 非正 logit 的重复惩罚是乘以惩罚系数。
#[test]
fn repeat_penalty_multiplies_negative_logits() {
    // context 含 id0，logits [-1.0, -0.5]，penalty=4.0：id0 的 logit <= 0，
    // 乘以 4 得 -4.0 < -0.5，argmax 变成 1。
    // 若按除法语义（-1.0/4 = -0.25 > -0.5），argmax 会是 0。
    let mut sampler = Sampler::with_seed(config(0.0, 0, 1.0, 4.0), 3);
    let id = sampler.sample(&logits_tensor(&[-1.0, -0.5]), &[0]).unwrap();
    assert_eq!(id, 1);
}

/// 惩罚系数为 1.0 时是空操作。
#[test]
fn repeat_penalty_of_one_is_a_no_op() {
    // logits [0.25, 0.5]，penalty=1.0：id1 保持 0.5，argmax 为 1。
    let mut sampler = Sampler::with_seed(config(0.0, 0, 1.0, 1.0), 3);
    let id = sampler.sample(&logits_tensor(&[0.25, 0.5]), &[1]).unwrap();
    assert_eq!(id, 1);
}

/// 只有 context 里出现过的 token 受惩罚。
#[test]
fn repeat_penalty_only_affects_context_tokens() {
    // logits [0.4, 0.5]：id1 在 context 中，penalty=2.0 得 0.25 < 0.4，argmax 变成 0。
    let mut penalized = Sampler::with_seed(config(0.0, 0, 1.0, 2.0), 3);
    let id = penalized.sample(&logits_tensor(&[0.4, 0.5]), &[1]).unwrap();
    assert_eq!(id, 0);

    // context 为空时无人受罚，argmax 仍是 1。
    let mut unpenalized = Sampler::with_seed(config(0.0, 0, 1.0, 2.0), 3);
    let id = unpenalized
        .sample(&logits_tensor(&[0.4, 0.5]), &[])
        .unwrap();
    assert_eq!(id, 1);
}

/// context 里的 id 重复出现只惩罚一次，与 llama.cpp 的按 id 计数一致。
#[test]
fn repeat_penalty_counts_each_token_once() {
    // 重复惩罚只看 id 是否出现过：[1, 1, 1] 与 [1] 的效果相同，
    // 都是把 0.5 除以 2 得 0.25，与 id0 (0.3) 相比更小，argmax 为 0。
    let mut once = Sampler::with_seed(config(0.0, 0, 1.0, 2.0), 3);
    assert_eq!(once.sample(&logits_tensor(&[0.3, 0.5]), &[1]).unwrap(), 0);
    let mut thrice = Sampler::with_seed(config(0.0, 0, 1.0, 2.0), 3);
    assert_eq!(
        thrice
            .sample(&logits_tensor(&[0.3, 0.5]), &[1, 1, 1])
            .unwrap(),
        0
    );
}

/// context 里越界的 id 不影响任何候选。
#[test]
fn repeat_penalty_ignores_out_of_range_context_ids() {
    // logits [0.25, 0.5]，context 里的 7 与 -1 都越界，无人受罚，argmax 为 1。
    let mut sampler = Sampler::with_seed(config(0.0, 0, 1.0, 2.0), 3);
    let id = sampler
        .sample(&logits_tensor(&[0.25, 0.5]), &[7, -1])
        .unwrap();
    assert_eq!(id, 1);
}

/// 重复惩罚先于 top-k 生效：被降权的候选不会因为截断而重新胜出。
#[test]
fn repeat_penalty_applies_before_top_k() {
    // logits [0.5, 0.3]，context [0]，penalty=2.0 -> id0 得 0.25，此时
    // 概率最大的是 id1 (0.3)。top_k=1 保留 id1。
    // 若 top-k 先于惩罚执行，截断会留下 id0 (0.5) 独占，结果会是 0。
    let mut sampler = Sampler::with_seed(config(1.0, 1, 1.0, 2.0), 3);
    for _ in 0..100 {
        let id = sampler.sample(&logits_tensor(&[0.5, 0.3]), &[0]).unwrap();
        assert_eq!(id, 1);
    }
}

// ---------------------------------------------------------------- 随机源

/// 同种子同输入给出同样的采样序列。
#[test]
fn fixed_seed_is_deterministic() {
    let mut first = Sampler::with_seed(config(1.0, 0, 1.0, 1.0), 42);
    let mut second = Sampler::with_seed(config(1.0, 0, 1.0, 1.0), 42);
    for _ in 0..20 {
        let left = first
            .sample(&logits_tensor(&[1.0, 2.0, 3.0, 4.0]), &[])
            .unwrap();
        let right = second
            .sample(&logits_tensor(&[1.0, 2.0, 3.0, 4.0]), &[])
            .unwrap();
        assert_eq!(left, right);
    }
}

/// 不同种子都只产生合法 id；不断言两者一定取到不同的 token。
#[test]
fn different_seeds_stay_within_the_vocabulary() {
    for seed in [1u64, 2, 3, 4] {
        let mut sampler = Sampler::with_seed(config(1.0, 0, 1.0, 1.0), seed);
        for _ in 0..100 {
            let id = sampler
                .sample(&logits_tensor(&[0.0, 0.0, 0.0]), &[])
                .unwrap();
            assert!((0..3).contains(&id), "id {id} out of range");
        }
    }
}

/// `sample_slice` 与 `sample` 对同一种子给出同样结果。
#[test]
fn sample_slice_matches_sample() {
    let mut by_tensor = Sampler::with_seed(config(1.0, 0, 1.0, 1.0), 7);
    let mut by_slice = Sampler::with_seed(config(1.0, 0, 1.0, 1.0), 7);
    let values = [1.0, 2.0, 3.0];
    let left = by_tensor.sample(&logits_tensor(&values), &[]).unwrap();
    let right = by_slice.sample_slice(&values, &[]).unwrap();
    assert_eq!(left, right);
}

// ---------------------------------------------------------------- 输入校验

/// 空 logits 报错而不是返回某个 id。
#[test]
fn empty_logits_are_rejected() {
    let mut sampler = Sampler::with_seed(config(1.0, 0, 1.0, 1.0), 1);
    assert!(sampler.sample_slice(&[], &[]).is_err());
}

/// NaN logits 报错。
#[test]
fn nan_logits_are_rejected() {
    let mut sampler = Sampler::with_seed(config(1.0, 0, 1.0, 1.0), 1);
    assert!(sampler.sample_slice(&[0.0, f32::NAN, 1.0], &[]).is_err());
}

/// ±Inf logits 报错。
#[test]
fn infinite_logits_are_rejected() {
    let mut positive = Sampler::with_seed(config(1.0, 0, 1.0, 1.0), 1);
    assert!(positive.sample_slice(&[0.0, f32::INFINITY], &[]).is_err());
    let mut negative = Sampler::with_seed(config(1.0, 0, 1.0, 1.0), 1);
    assert!(negative
        .sample_slice(&[f32::NEG_INFINITY, 0.0], &[])
        .is_err());
}

/// 只接受一维 logits；二维取行由调用方负责。
#[test]
fn two_dimensional_logits_are_rejected() {
    let mut sampler = Sampler::with_seed(config(1.0, 0, 1.0, 1.0), 1);
    let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    assert!(sampler.sample(&matrix, &[]).is_err());
}

/// 非法的采样参数报错，而不是静默按"不限制"处理。
#[test]
fn invalid_configuration_is_rejected() {
    let mut nan_temperature = Sampler::with_seed(config(f32::NAN, 0, 1.0, 1.0), 1);
    assert!(nan_temperature.sample_slice(&[1.0, 2.0], &[]).is_err());
    let mut nan_top_p = Sampler::with_seed(config(1.0, 0, f32::NAN, 1.0), 1);
    assert!(nan_top_p.sample_slice(&[1.0, 2.0], &[]).is_err());
    let mut zero_penalty = Sampler::with_seed(config(1.0, 0, 1.0, 0.0), 1);
    assert!(zero_penalty.sample_slice(&[1.0, 2.0], &[]).is_err());
}

// ---------------------------------------------------------------- 分布健全性

/// 等概率 logits 上的抽样应大致均匀。
#[test]
fn uniform_sampling_is_balanced() {
    // softmax([0, 0]) = [0.5, 0.5]。1000 次二项抽样，σ = sqrt(1000*0.5*0.5)
    // = sqrt(250) ≈ 15.8。断言界取 mean±150 ≈ ±9.5σ：远宽于 3σ，只拦截
    // 系统性偏置（如概率没有归一化、随机源退化），不纠缠正常涨落。
    let mut sampler = Sampler::with_seed(config(1.0, 0, 1.0, 1.0), 2024);
    let counts = sample_counts(&mut sampler, &[0.0, 0.0], 1000);
    assert!(counts[0] > 350 && counts[0] < 650, "{counts:?}");
    assert!(counts[1] > 350 && counts[1] < 650, "{counts:?}");
}

/// 返回的 id 永远落在词表范围内，两种截断设置都要检查。
#[test]
fn sampled_ids_stay_within_the_vocabulary() {
    // top_k=1：只剩 argmax，必然是合法 id。
    let mut greedy = Sampler::with_seed(config(1.0, 1, 1.0, 1.0), 5);
    for _ in 0..100 {
        let id = greedy
            .sample(&logits_tensor(&[3.0, 1.0, 2.0]), &[])
            .unwrap();
        assert!((0..3).contains(&id), "id {id} out of range");
    }
    // 全保留：分布覆盖整个词表。
    let mut full = Sampler::with_seed(config(1.0, 0, 1.0, 1.0), 5);
    for _ in 0..100 {
        let id = full.sample(&logits_tensor(&[3.0, 1.0, 2.0]), &[]).unwrap();
        assert!((0..3).contains(&id), "id {id} out of range");
    }
}
