//! 连续行主序 f32 张量运算。

use super::Tensor;
use rayon::prelude::*;

/// 张量算子；结果不修改输入或其共享视图。
///
/// 所有方法均校验形状与缓冲区长度；形状乘积或步长溢出时返回错误。
/// 空张量在参数有效时保留运算所得形状，零维张量表示单元素标量。
pub trait TensorOps {
    /// 计算 `[m, k] × [k, n] -> [m, n]`；非二维或内维不匹配时返回错误。
    ///
    /// 按行并行，行内以 i-k-j 顺序进行 f32 乘加；零内维产生全零结果。
    /// 非有限值和算术溢出遵循 IEEE 754，输出容量无法分配时返回错误。
    fn matmul(&self, other: &Tensor) -> Result<Tensor, String>;

    /// 沿最后一维做 RMSNorm，权重缓冲区长度须为该维长度或 1。
    ///
    /// 无最后一维、权重长度错误或 eps 非有限/小于零时返回错误。
    /// 平方、均值和归一化使用 f64；全零切片在 eps=0 时定义为零。
    /// 输入及权重的 NaN/无穷按浮点规则传播，有限权重下零切片始终有限。
    fn rms_norm(&self, weight: &Tensor, eps: f32) -> Result<Tensor, String>;

    /// 沿任意有效轴计算 softmax；轴越界或输入含 NaN 时返回错误。
    ///
    /// 减去切片最大值后以 f64 求指数及累加。全负无穷切片返回均匀分布；
    /// 含正无穷时仅这些位置平分概率，其余位置为零。形状保持不变。
    fn softmax(&self, dim: usize) -> Result<Tensor, String>;

    /// 对 `[seq_len, ..., head_dim]` 的相邻元素对应用 RoPE，保持形状不变。
    ///
    /// 序列位置为 start_pos 加首维索引，第 i 对频率为 theta^(-2i/head_dim)。
    /// 要求至少二维、head_dim 为正偶数且等于末维、theta 为正有限数；
    /// 参数无效或末位置溢出时返回错误。角度和旋转使用 f64，非有限数据按浮点规则传播。
    ///
    /// "相邻元素对"是与 llama.cpp `GGML_ROPE_TYPE_NORMAL` 对齐的选择，不是可自由
    /// 替换的实现细节：改成对半拆分（HuggingFace `rotate_half`）会偏离 llama 架构
    /// 权重，依据与实测影响见 [`crate::transformer::layers`] 模块文档。
    fn rope(&self, head_dim: usize, start_pos: usize, theta: f32) -> Result<Tensor, String>;

    /// 逐元素计算稳定的 SiLU，保持形状；NaN 传播，正无穷映射到正无穷，负无穷映射到负零。
    fn silu(&self) -> Result<Tensor, String>;

    /// 按 NumPy 尾维规则广播后相加；不兼容形状返回错误，数值遵循 IEEE f32 算术。
    fn add(&self, other: &Tensor) -> Result<Tensor, String>;

    /// 按 NumPy 尾维规则广播后计算 self - other；不兼容形状返回错误，数值遵循 IEEE f32 算术。
    fn sub(&self, other: &Tensor) -> Result<Tensor, String>;

    /// 按 NumPy 尾维规则广播后相乘；不兼容形状返回错误，数值遵循 IEEE f32 算术。
    fn mul(&self, other: &Tensor) -> Result<Tensor, String>;

    /// 按 NumPy 尾维规则广播后计算 self / other；不兼容形状返回错误。
    ///
    /// 遵循 IEEE f32 算术：非零数除以正负零得到带符号无穷，0/0 得到 NaN，不额外报错。
    fn div(&self, other: &Tensor) -> Result<Tensor, String>;

    /// 逐元素乘以 factor，保持形状；含非有限 factor 在内均遵循 IEEE f32 算术。
    fn scale(&self, factor: f32) -> Result<Tensor, String>;
}

impl TensorOps for Tensor {
    fn matmul(&self, other: &Tensor) -> Result<Tensor, String> {
        validate(self)?;
        validate(other)?;
        if self.ndim() != 2 || other.ndim() != 2 {
            return Err("MatMul requires two 2D tensors".to_string());
        }
        let (rows, shared) = (self.dims()[0], self.dims()[1]);
        let columns = other.dims()[1];
        if shared != other.dims()[0] {
            return Err(format!(
                "MatMul inner dimensions do not match: {shared} and {}",
                other.dims()[0]
            ));
        }

        let dims = vec![rows, columns];
        let mut data = zeroed_data(checked_size(&dims)?)?;
        if !data.is_empty() && shared != 0 {
            data.par_chunks_mut(columns)
                .enumerate()
                .for_each(|(row_index, output_row)| {
                    let left_row = &self.data()[row_index * shared..(row_index + 1) * shared];
                    for (inner, &left) in left_row.iter().enumerate() {
                        let right_row = &other.data()[inner * columns..(inner + 1) * columns];
                        for (output, &right) in output_row.iter_mut().zip(right_row) {
                            *output += left * right;
                        }
                    }
                });
        }
        Ok(Tensor::new(data, dims))
    }

    fn rms_norm(&self, weight: &Tensor, eps: f32) -> Result<Tensor, String> {
        validate(self)?;
        validate(weight)?;
        let &hidden = self
            .dims()
            .last()
            .ok_or_else(|| "RMSNorm requires at least one dimension".to_string())?;
        if !eps.is_finite() || eps < 0.0 {
            return Err("RMSNorm eps must be finite and nonnegative".to_string());
        }
        if weight.len() != hidden && weight.len() != 1 {
            return Err(format!(
                "RMSNorm weight length must be {hidden} or 1, got {}",
                weight.len()
            ));
        }
        if self.is_empty() {
            return Ok(self.clone());
        }

        let mut data = zeroed_data(self.len())?;
        for (input, output) in self
            .data()
            .chunks_exact(hidden)
            .zip(data.chunks_exact_mut(hidden))
        {
            let square_sum: f64 = input.iter().map(|&x| f64::from(x).powi(2)).sum();
            let denominator = (square_sum / hidden as f64 + f64::from(eps)).sqrt();
            for (index, (&value, result)) in input.iter().zip(output).enumerate() {
                // eps=0 时明确将全零切片归一化为零，避免 0/0。
                let normalized = if denominator == 0.0 {
                    f64::from(value)
                } else {
                    f64::from(value) / denominator
                };
                let weight_index = if weight.len() == 1 { 0 } else { index };
                *result = (normalized * f64::from(weight.data()[weight_index])) as f32;
            }
        }
        Ok(Tensor::new(data, self.dims().to_vec()))
    }

    fn softmax(&self, dim: usize) -> Result<Tensor, String> {
        validate(self)?;
        if dim >= self.ndim() {
            return Err(format!(
                "Softmax axis {dim} is out of bounds for rank {}",
                self.ndim()
            ));
        }
        if self.data().iter().any(|value| value.is_nan()) {
            return Err("Softmax input must not contain NaN".to_string());
        }
        if self.is_empty() {
            return Ok(self.clone());
        }

        let inner = self.strides()[dim];
        let block = self.dims()[dim] * inner;
        let mut data = zeroed_data(self.len())?;
        for (input, output) in self
            .data()
            .chunks_exact(block)
            .zip(data.chunks_exact_mut(block))
        {
            for offset in 0..inner {
                softmax_slice(&input[offset..], &mut output[offset..], inner);
            }
        }
        Ok(Tensor::new(data, self.dims().to_vec()))
    }

    fn rope(&self, head_dim: usize, start_pos: usize, theta: f32) -> Result<Tensor, String> {
        validate(self)?;
        if self.ndim() < 2
            || head_dim == 0
            || head_dim & 1 != 0
            || self.dims().last() != Some(&head_dim)
        {
            return Err(
                "RoPE requires shape [seq_len, ..., head_dim] with a positive even head_dim"
                    .to_string(),
            );
        }
        if !theta.is_finite() || theta <= 0.0 {
            return Err("RoPE theta must be finite and positive".to_string());
        }
        let sequence_len = self.dims()[0];
        if sequence_len != 0 {
            start_pos
                .checked_add(sequence_len - 1)
                .ok_or_else(|| "RoPE last position overflows usize".to_string())?;
        }
        if self.is_empty() {
            return Ok(self.clone());
        }

        let frequencies: Vec<f64> = (0..head_dim / 2)
            .map(|pair| f64::from(theta).powf(-2.0 * pair as f64 / head_dim as f64))
            .collect();
        let sequence_stride = self.len() / sequence_len;
        let mut data = zeroed_data(self.len())?;
        for (sequence, (input, output)) in self
            .data()
            .chunks_exact(sequence_stride)
            .zip(data.chunks_exact_mut(sequence_stride))
            .enumerate()
        {
            let position = (start_pos + sequence) as f64;
            for (pair, frequency) in frequencies.iter().enumerate() {
                // 同一序列位置的所有头共享同一组角度，避免冗余三角运算。
                let (sin, cos) = (position * frequency).sin_cos();
                for (input_head, output_head) in input
                    .chunks_exact(head_dim)
                    .zip(output.chunks_exact_mut(head_dim))
                {
                    let first = f64::from(input_head[2 * pair]);
                    let second = f64::from(input_head[2 * pair + 1]);
                    output_head[2 * pair] = (first * cos - second * sin) as f32;
                    output_head[2 * pair + 1] = (first * sin + second * cos) as f32;
                }
            }
        }
        Ok(Tensor::new(data, self.dims().to_vec()))
    }

    fn silu(&self) -> Result<Tensor, String> {
        validate(self)?;
        let data = self
            .data()
            .iter()
            .map(|&value| {
                if value == f32::NEG_INFINITY {
                    -0.0
                } else if value >= 0.0 {
                    value / (1.0 + (-value).exp())
                } else {
                    let exponential = value.exp();
                    value * exponential / (1.0 + exponential)
                }
            })
            .collect();
        Ok(Tensor::new(data, self.dims().to_vec()))
    }

    fn add(&self, other: &Tensor) -> Result<Tensor, String> {
        elementwise(self, other, |left, right| left + right)
    }

    fn sub(&self, other: &Tensor) -> Result<Tensor, String> {
        elementwise(self, other, |left, right| left - right)
    }

    fn mul(&self, other: &Tensor) -> Result<Tensor, String> {
        elementwise(self, other, |left, right| left * right)
    }

    fn div(&self, other: &Tensor) -> Result<Tensor, String> {
        elementwise(self, other, |left, right| left / right)
    }

    fn scale(&self, factor: f32) -> Result<Tensor, String> {
        validate(self)?;
        let data = self.data().iter().map(|value| value * factor).collect();
        Ok(Tensor::new(data, self.dims().to_vec()))
    }
}

fn checked_size(dims: &[usize]) -> Result<usize, String> {
    // 从末维计算，同时保证 Tensor::new 的后缀步长可表示；末尾零维可消去大前缀。
    dims.iter().rev().try_fold(1usize, |size, &dim| {
        size.checked_mul(dim)
            .ok_or_else(|| format!("Tensor shape product or stride overflows usize: {dims:?}"))
    })
}

fn validate(tensor: &Tensor) -> Result<(), String> {
    let size = checked_size(tensor.dims())?;
    if size != tensor.len() {
        return Err(format!(
            "Invalid tensor shape {:?}: expected {size} elements, got {}",
            tensor.dims(),
            tensor.len()
        ));
    }
    Ok(())
}

fn zeroed_data(len: usize) -> Result<Vec<f32>, String> {
    let mut data = Vec::new();
    data.try_reserve_exact(len)
        .map_err(|error| format!("Cannot allocate {len} tensor elements: {error}"))?;
    data.resize(len, 0.0);
    Ok(data)
}

fn softmax_slice(input: &[f32], output: &mut [f32], stride: usize) {
    let maximum = input
        .iter()
        .step_by(stride)
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let contribution = |&value: &f32| {
        if maximum == f32::NEG_INFINITY {
            1.0
        } else if maximum == f32::INFINITY {
            if value == f32::INFINITY {
                1.0
            } else {
                0.0
            }
        } else {
            (f64::from(value) - f64::from(maximum)).exp()
        }
    };
    let denominator: f64 = input.iter().step_by(stride).map(contribution).sum();
    for (value, result) in input
        .iter()
        .step_by(stride)
        .zip(output.iter_mut().step_by(stride))
    {
        *result = (contribution(value) / denominator) as f32;
    }
}

fn broadcast_shape(left: &[usize], right: &[usize]) -> Result<Vec<usize>, String> {
    let mut dims = vec![1; left.len().max(right.len())];
    let mut left_axes = left.iter().rev();
    let mut right_axes = right.iter().rev();
    for dim in dims.iter_mut().rev() {
        let left_dim = left_axes.next().copied().unwrap_or(1);
        let right_dim = right_axes.next().copied().unwrap_or(1);
        *dim = if left_dim == right_dim || right_dim == 1 {
            left_dim
        } else if left_dim == 1 {
            right_dim
        } else {
            return Err(format!("Cannot broadcast shapes {left:?} and {right:?}"));
        };
    }
    Ok(dims)
}

fn broadcast_strides(tensor: &Tensor, rank: usize) -> Vec<usize> {
    let mut strides = vec![0; rank];
    let offset = rank - tensor.ndim();
    for (axis, &dim) in tensor.dims().iter().enumerate() {
        if dim != 1 {
            strides[offset + axis] = tensor.strides()[axis];
        }
    }
    strides
}

fn elementwise(
    left: &Tensor,
    right: &Tensor,
    operation: impl Fn(f32, f32) -> f32,
) -> Result<Tensor, String> {
    validate(left)?;
    validate(right)?;
    if left.dims() == right.dims() {
        let data = left
            .data()
            .iter()
            .zip(right.data())
            .map(|(&left, &right)| operation(left, right))
            .collect();
        return Ok(Tensor::new(data, left.dims().to_vec()));
    }

    let dims = broadcast_shape(left.dims(), right.dims())?;
    let mut data = zeroed_data(checked_size(&dims)?)?;
    if !data.is_empty() {
        let left_strides = broadcast_strides(left, dims.len());
        let right_strides = broadcast_strides(right, dims.len());
        for (index, result) in data.iter_mut().enumerate() {
            let (mut remaining, mut left_index, mut right_index) = (index, 0, 0);
            for axis in (0..dims.len()).rev() {
                let coordinate = remaining % dims[axis];
                remaining /= dims[axis];
                left_index += coordinate * left_strides[axis];
                right_index += coordinate * right_strides[axis];
            }
            *result = operation(left.data()[left_index], right.data()[right_index]);
        }
    }
    Ok(Tensor::new(data, dims))
}
