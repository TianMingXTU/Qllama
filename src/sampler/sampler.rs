//! Sampler：把模型输出的 logits 转成下一个 token id。
//!
//! 处理链依次是重复惩罚 → 温度 → softmax → top-k → top-p → 采样，
//! 每一步的语义都对照 llama.cpp 的 `src/llama-sampler.cpp`。

use std::cmp::Ordering;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::tensor::Tensor;

/// 采样参数。
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// 温度缩放系数；`<= 0` 表示贪心采样（直接取 argmax）。
    pub temperature: f32,
    /// top-k 保留的候选个数；`0` 表示不限制。
    pub top_k: usize,
    /// top-p 核采样阈值；`>= 1.0` 表示不限制。
    pub top_p: f32,
    /// 重复惩罚系数；`1.0` 表示不惩罚。
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

impl SamplerConfig {
    /// 校验参数是否落在可用范围内。
    ///
    /// 温度与 top-p 必须是有限数：NaN 会让所有比较为假，从而静默走
    /// "不限制"分支；重复惩罚必须为正数，否则后面的除法会得到无穷或 NaN。
    fn validate(&self) -> Result<(), String> {
        if !self.temperature.is_finite() {
            return Err(format!(
                "Sampler temperature must be finite, got {}",
                self.temperature
            ));
        }
        if !self.top_p.is_finite() {
            return Err(format!("Sampler top_p must be finite, got {}", self.top_p));
        }
        if !self.repeat_penalty.is_finite() || self.repeat_penalty <= 0.0 {
            return Err(format!(
                "Sampler repeat_penalty must be finite and positive, got {}",
                self.repeat_penalty
            ));
        }
        Ok(())
    }
}

/// 采样器：按配置把一行 logits 转成一个 token id。
///
/// 随机源固定为 `rand` 0.8 的 `StdRng`（ChaCha12）：`seed_from_u64` 在所有
/// 平台上给出同一序列，因此测试与 `--seed` 都能复现同一条生成结果。
pub struct Sampler {
    config: SamplerConfig,
    rng: StdRng,
}

impl Sampler {
    /// 用系统熵创建采样器。
    ///
    /// 生成结果不可复现；需要复现时改用 [`Sampler::with_seed`]。
    pub fn new(config: SamplerConfig) -> Self {
        Self {
            config,
            rng: StdRng::from_entropy(),
        }
    }

    /// 用固定种子创建采样器，同样的配置与输入给出同样的采样序列。
    pub fn with_seed(config: SamplerConfig, seed: u64) -> Self {
        Self {
            config,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// 从一行 logits 采样下一个 token id。
    ///
    /// `logits` 必须是形状 `[vocab_size]` 的一维张量：二维 logits 的取行
    /// 交给调用方（CLI 已有 `last_position_row`），放进采样器只会让维度
    /// 约定含糊。`context` 是已经生成的 token id（含 prompt），重复惩罚按
    /// 其中出现过的 id 生效；越界或为负的 id 被忽略。
    ///
    /// 返回的 id 落在 `[0, vocab_size)`。logits 为空、含 NaN/±Inf，或参数
    /// 非法时返回错误，而不是给出一个无法解释的 id。
    pub fn sample(&mut self, logits: &Tensor, context: &[i32]) -> Result<i32, String> {
        if logits.ndim() != 1 {
            return Err(format!(
                "Sampling requires a 1D logits tensor of shape [vocab_size], got {:?}",
                logits.dims()
            ));
        }
        if logits.dims()[0] != logits.len() {
            return Err(format!(
                "Logits tensor shape {:?} does not match its buffer length {}",
                logits.dims(),
                logits.len()
            ));
        }
        self.sample_slice(logits.data(), context)
    }

    /// 从 f32 切片采样下一个 token id。
    ///
    /// 与 [`Sampler::sample`] 语义相同，供调用方手上只有 logits 行时使用。
    pub fn sample_slice(&mut self, logits: &[f32], context: &[i32]) -> Result<i32, String> {
        if logits.is_empty() {
            return Err("Sampling requires a non-empty logits slice".to_string());
        }
        if let Some((index, &value)) = logits.iter().enumerate().find(|(_, v)| !v.is_finite()) {
            return Err(format!(
                "Logits must be finite, but element {index} is {value}"
            ));
        }
        self.config.validate()?;

        let mut working = logits.to_vec();
        apply_repeat_penalty(&mut working, context, self.config.repeat_penalty);
        if let Some((index, &value)) = working.iter().enumerate().find(|(_, v)| !v.is_finite()) {
            return Err(format!(
                "Repeat penalty produced a non-finite logit at element {index}: {value}"
            ));
        }

        // llama.cpp 的 llama_sampler_temp_impl 在 temp<=0 时走贪心：把最大
        // logit 之外的候选全部置为 -INF，等价于只在唯一候选上采样。
        if self.config.temperature <= 0.0 {
            let index = argmax(&working);
            return to_token_id(index);
        }

        for value in &mut working {
            *value /= self.config.temperature;
        }
        if let Some((index, &value)) = working.iter().enumerate().find(|(_, v)| !v.is_finite()) {
            return Err(format!(
                "Temperature {} produced a non-finite logit at element {index}: {value}",
                self.config.temperature
            ));
        }

        let mut probabilities = softmax(&working);
        apply_top_k(&mut probabilities, self.config.top_k);
        apply_top_p(&mut probabilities, self.config.top_p);

        let index = sample_from_distribution(&probabilities, &mut self.rng)?;
        to_token_id(index)
    }
}

/// 把下标转成 token id；词表超过 i32 上限时报错而不是截断。
fn to_token_id(index: usize) -> Result<i32, String> {
    i32::try_from(index).map_err(|_| format!("Token id {index} does not fit in i32"))
}

/// 对 context 中出现过的 token id 施加重复惩罚。
///
/// 对应 llama.cpp 的 `llama_sampler_penalties_apply`（src/llama-sampler.cpp
/// L2977-2981）：`logit <= 0` 乘以惩罚系数，`logit > 0` 除以惩罚系数——
/// 正 logit 若做乘法会更大、负 logit 若做除法会更接近 0，惩罚都会反向。
/// 注意：任务书把这组方向写反了，以 llama.cpp 源码为准。同一个 id 在
/// context 里出现多次只惩罚一次。
///
/// 本实现不对 BOS/EOS/Control token 豁免，与 llama.cpp 的 penalties sampler
/// 一致：后者同样只按 token id 计数，不区分 token 类别。
fn apply_repeat_penalty(logits: &mut [f32], context: &[i32], penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    let mut penalized = vec![false; logits.len()];
    for &token in context {
        let Ok(index) = usize::try_from(token) else {
            continue;
        };
        if index >= logits.len() || penalized[index] {
            continue;
        }
        penalized[index] = true;
        let logit = &mut logits[index];
        if *logit <= 0.0 {
            *logit *= penalty;
        } else {
            *logit /= penalty;
        }
    }
}

/// 返回最大 logit 的下标；并列时取最小 id。
fn argmax(logits: &[f32]) -> usize {
    let mut best = 0;
    for (index, &value) in logits.iter().enumerate().skip(1) {
        if value.total_cmp(&logits[best]) == Ordering::Greater {
            best = index;
        }
    }
    best
}

/// 数值稳定的 softmax：减去最大值后以 f64 求指数并归一化。
///
/// 输入已由调用方保证有限且非空，因此最大值有限、指数和为正。
fn softmax(logits: &[f32]) -> Vec<f64> {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponentials: Vec<f64> = logits
        .iter()
        .map(|&value| (f64::from(value) - f64::from(maximum)).exp())
        .collect();
    let total: f64 = exponentials.iter().sum();
    exponentials.iter().map(|&value| value / total).collect()
}

/// top-k：按概率降序保留前 k 个候选，其余置 0 后重新归一化。
///
/// 并列时按 id 升序取，结果可复现。`k` 为 0 或不小于词表大小时不限制。
fn apply_top_k(probabilities: &mut [f64], k: usize) {
    if k == 0 || k >= probabilities.len() {
        return;
    }
    let order = sorted_indices(probabilities);
    let kept = &order[..k];
    let total: f64 = kept.iter().map(|&index| probabilities[index]).sum();
    let mut keep_flags = vec![false; probabilities.len()];
    for &index in kept {
        keep_flags[index] = true;
    }
    for (index, probability) in probabilities.iter_mut().enumerate() {
        *probability = if keep_flags[index] {
            *probability / total
        } else {
            0.0
        };
    }
}

/// top-p：按概率降序累加，保留累计值首次达到 `p` 的候选（含该候选），
/// 其余置 0 后重新归一化。
///
/// 对应 llama.cpp 的 `llama_sampler_top_p_apply`：源码里 `last_idx = i + 1`
/// 落在 `cum_sum >= ctx->p` 的分支内，因此越过阈值的那个候选也在集合内。
/// `p >= 1.0` 时不限制。
fn apply_top_p(probabilities: &mut [f64], p: f32) {
    if p >= 1.0 {
        return;
    }
    let order = sorted_indices(probabilities);
    let threshold = f64::from(p);
    let mut cumulative = 0.0f64;
    let mut keep_count = order.len();
    for (position, &index) in order.iter().enumerate() {
        cumulative += probabilities[index];
        if cumulative >= threshold {
            keep_count = position + 1;
            break;
        }
    }
    let kept = &order[..keep_count];
    let total: f64 = kept.iter().map(|&index| probabilities[index]).sum();
    let mut keep_flags = vec![false; probabilities.len()];
    for &index in kept {
        keep_flags[index] = true;
    }
    for (index, probability) in probabilities.iter_mut().enumerate() {
        *probability = if keep_flags[index] {
            *probability / total
        } else {
            0.0
        };
    }
}

/// 按概率降序返回下标；并列时按 id 升序。
fn sorted_indices(probabilities: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..probabilities.len()).collect();
    order.sort_by(|&left, &right| {
        probabilities[right]
            .total_cmp(&probabilities[left])
            .then_with(|| left.cmp(&right))
    });
    order
}

/// 从归一化分布中抽一个下标。
///
/// 对应 llama.cpp 的 `llama_sampler_dist_apply`：取一个 `[0, 1)` 的随机数，
/// 按累计概率定位候选，直到累计值达到 `r * 总和` 为止。概率为 0 的候选
/// 不参与累计，否则 `r` 恰为 0 时会抽到概率为 0 的 token。累加因浮点误差
/// 始终略小于抽样上界时，兜底返回最后一个非零概率的候选。
fn sample_from_distribution(probabilities: &[f64], rng: &mut StdRng) -> Result<usize, String> {
    let total: f64 = probabilities.iter().sum();
    let threshold = rng.gen::<f64>() * total;
    let mut cumulative = 0.0f64;
    for (index, &probability) in probabilities.iter().enumerate() {
        if probability <= 0.0 {
            continue;
        }
        cumulative += probability;
        if cumulative >= threshold {
            return Ok(index);
        }
    }
    probabilities
        .iter()
        .rposition(|&probability| probability > 0.0)
        .ok_or_else(|| "Sampling failed: the distribution has no positive probability".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 惩罚系数 2.0：非正 logit 乘以 2，正 logit 除以 2（llama.cpp 语义）。
    #[test]
    fn repeat_penalty_divides_positive_and_multiplies_negative() {
        let mut logits = vec![0.0, 0.5, -1.0, 0.0];
        apply_repeat_penalty(&mut logits, &[1, 2], 2.0);
        // id1: 0.5 > 0 -> 0.5 / 2 = 0.25；id2: -1.0 <= 0 -> -1.0 * 2 = -2.0。
        // id0 未出现在 context 中，保持 0.0。
        assert_eq!(logits, vec![0.0, 0.25, -2.0, 0.0]);
    }

    /// 同一个 id 在 context 里出现多次只惩罚一次。
    #[test]
    fn repeat_penalty_ignores_repetition_count() {
        let mut once = vec![1.0];
        apply_repeat_penalty(&mut once, &[0], 4.0);
        let mut thrice = vec![1.0];
        apply_repeat_penalty(&mut thrice, &[0, 0, 0], 4.0);
        assert_eq!(once, vec![0.25]);
        assert_eq!(thrice, vec![0.25]);
    }

    /// 惩罚系数为 1.0 时是空操作。
    #[test]
    fn repeat_penalty_of_one_changes_nothing() {
        let mut logits = vec![1.0, -2.0, 0.5];
        apply_repeat_penalty(&mut logits, &[0, 1, 2], 1.0);
        assert_eq!(logits, vec![1.0, -2.0, 0.5]);
    }

    /// context 中超出词表的 id 不触碰任何 logit。
    #[test]
    fn repeat_penalty_skips_out_of_range_ids() {
        let mut logits = vec![1.0, 2.0];
        apply_repeat_penalty(&mut logits, &[7, -1], 2.0);
        assert_eq!(logits, vec![1.0, 2.0]);
    }

    /// softmax：减最大值后归一化，结果之和为 1。
    #[test]
    fn softmax_is_normalized() {
        let probabilities = softmax(&[1.0, 2.0, 3.0]);
        let total: f64 = probabilities.iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "total={total}");
        // 相对权重 e^1 : e^2 : e^3 = 2.7183 : 7.3891 : 20.0855，总和 30.1929。
        // p2 = 20.0855 / 30.1929 = 0.66524，p1 = 7.3891 / 30.1929 = 0.24473，
        // p0 = 2.7183 / 30.1929 = 0.09003。
        assert!(
            (probabilities[2] - 0.66524).abs() < 1e-4,
            "{:?}",
            probabilities
        );
        assert!(
            (probabilities[1] - 0.24473).abs() < 1e-4,
            "{:?}",
            probabilities
        );
        assert!(
            (probabilities[0] - 0.09003).abs() < 1e-4,
            "{:?}",
            probabilities
        );
    }
}
