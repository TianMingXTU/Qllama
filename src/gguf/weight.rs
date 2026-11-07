//! 共享 GGUF 映射的权重：保持磁盘格式，计算时只解码行或 32 元素块。

use std::ops::Range;
use std::sync::Arc;

use memmap2::Mmap;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;

use super::loader::{checked_product, f16_to_f32};
use super::GgmlType;
use crate::tensor::Tensor;

/// 原始 GGUF 布局的只读权重；持有映射，因此不依赖 GgufFile 的生命周期。
///
/// 维度已反转为行优先顺序。F16 与量化权重不会在加载时扩张为 f32；克隆只
/// 增加映射引用计数。调用方必须保证映射期间原文件不被修改或截断。
#[derive(Debug, Clone)]
pub struct MappedWeight {
    mapping: Arc<Mmap>,
    range: Range<usize>,
    dims: Vec<usize>,
    dtype: GgmlType,
    row_bytes: usize,
}

impl MappedWeight {
    pub(super) fn new(
        mapping: Arc<Mmap>,
        start: usize,
        dims: Vec<usize>,
        dtype: GgmlType,
    ) -> Result<Self, String> {
        if dims.is_empty() || dims.contains(&0) {
            return Err(format!(
                "Weight dimensions must be nonempty and positive: {dims:?}"
            ));
        }
        let count = checked_product(&dims)?;
        let width = *dims.last().expect("nonempty dimensions");
        // Q4_0 块 = f16 scale + 16 字节 nibble；Q8_0 块 = f16 scale + 32 个 i8。
        let block_bytes = match dtype {
            GgmlType::Q4_0 => 18,
            GgmlType::Q8_0 => 34,
            GgmlType::F32 => 0,
            GgmlType::F16 => 0,
            _ => {
                return Err(format!(
                    "Unsupported weight type {dtype:?}; supported: F32, F16, Q4_0, Q8_0"
                ))
            }
        };
        let row_bytes = if block_bytes > 0 {
            if !width.is_multiple_of(32) {
                return Err(format!(
                    "{dtype:?} row width {width} is not a multiple of block size 32"
                ));
            }
            (width / 32).checked_mul(block_bytes)
        } else {
            width.checked_mul(if dtype == GgmlType::F32 { 4 } else { 2 })
        }
        .ok_or_else(|| "Weight row byte length overflows usize".to_string())?;
        let length = (count / width)
            .checked_mul(row_bytes)
            .ok_or_else(|| "Weight byte length overflows usize".to_string())?;
        let end = start
            .checked_add(length)
            .ok_or_else(|| "Weight end offset overflows usize".to_string())?;
        if end > mapping.len() {
            return Err(format!(
                "Weight spans [{start}, {end}) beyond mapped {} bytes",
                mapping.len()
            ));
        }
        Ok(Self {
            mapping,
            range: start..end,
            dims,
            dtype,
            row_bytes,
        })
    }

    /// 行优先逻辑维度；二维矩阵为 `[out_features, in_features]`。
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// 映射中的原始 GGML 类型。
    pub fn dtype(&self) -> GgmlType {
        self.dtype
    }

    /// 该张量实际占用的压缩字节数，不包含 GGUF 元数据与对齐。
    pub fn storage_bytes(&self) -> usize {
        self.range.len()
    }

    /// 仅解码选中行；用于词嵌入，不构造完整反量化矩阵。
    pub fn read_row(&self, row: usize) -> Result<Vec<f32>, String> {
        let width = *self.dims.last().expect("validated dimensions");
        if row >= self.range.len() / self.row_bytes {
            return Err(format!("Weight row {row} is out of range"));
        }
        let bytes = self.row(row);
        let mut values = Vec::with_capacity(width);
        match self.dtype {
            GgmlType::F32 => values.extend(bytes.as_chunks::<4>().0.iter().map(|c| read_f32(c))),
            GgmlType::F16 => values.extend(bytes.as_chunks::<2>().0.iter().map(|c| read_f16(c))),
            GgmlType::Q4_0 | GgmlType::Q8_0 => {
                for block in bytes.chunks_exact(self.block_bytes()) {
                    values.extend(dequantize_block(block, self.dtype));
                }
            }
            _ => unreachable!("validated weight type"),
        }
        debug_assert_eq!(values.len(), width);
        Ok(values)
    }

    /// 计算 `input * weight^T`；输入 `[rows, in]`，输出 `[rows, out]`。
    ///
    /// 权重始终从映射读取；量化临时缓冲仅为栈上的一个 32 元素块。每个输出的
    /// 累加顺序固定为输入特征从左至右，与原有标量 matmul 的顺序一致，因此并行
    /// 划分不改变任何数值结果。
    ///
    /// 并行策略：多行（prefill）按行划分，行间天然独立；单行（decode）没有行级
    /// 并行度，改为把输出列分块交给 Rayon，让 1B 级模型的每步 decode 也能吃满
    /// 线程。列分块大小取 64，摊薄调度开销又不至于让尾块失衡。
    pub fn matmul(&self, input: &Tensor) -> Result<Tensor, String> {
        if self.dims.len() != 2 || input.ndim() != 2 || input.dims()[1] != self.dims[1] {
            return Err(format!(
                "Mapped matmul requires [rows, in] and [out, in], got {:?} and {:?}",
                input.dims(),
                self.dims
            ));
        }
        let (rows, width, columns) = (input.dims()[0], self.dims[1], self.dims[0]);
        let length = rows
            .checked_mul(columns)
            .ok_or_else(|| "Matmul output size overflows usize".to_string())?;
        let mut output = vec![0.0; length];
        if rows == 1 {
            const BLOCK: usize = 64;
            let row_input = &input.data()[..width];
            for start in (0..columns).step_by(BLOCK) {
                let end = (start + BLOCK).min(columns);
                output[start..end]
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(offset, value)| *value = self.dot(row_input, start + offset));
            }
        } else {
            output
                .par_chunks_mut(columns)
                .enumerate()
                .for_each(|(row, result)| {
                    let x = &input.data()[row * width..(row + 1) * width];
                    for (column, value) in result.iter_mut().enumerate() {
                        *value = self.dot(x, column);
                    }
                });
        }
        Ok(Tensor::new(output, vec![rows, columns]))
    }

    fn row(&self, row: usize) -> &[u8] {
        let start = self.range.start + row * self.row_bytes;
        &self.mapping[start..start + self.row_bytes]
    }

    fn block_bytes(&self) -> usize {
        if self.dtype == GgmlType::Q4_0 {
            18
        } else {
            34
        }
    }

    fn dot(&self, input: &[f32], row: usize) -> f32 {
        let bytes = self.row(row);
        let mut sum = 0.0;
        match self.dtype {
            GgmlType::F32 => {
                for (x, value) in input.iter().zip(bytes.as_chunks::<4>().0) {
                    sum += x * read_f32(value);
                }
            }
            GgmlType::F16 => {
                for (x, value) in input.iter().zip(bytes.as_chunks::<2>().0) {
                    sum += x * read_f16(value);
                }
            }
            GgmlType::Q4_0 | GgmlType::Q8_0 => {
                let block_bytes = self.block_bytes();
                for (x, block) in input
                    .as_chunks::<32>()
                    .0
                    .iter()
                    .zip(bytes.chunks_exact(block_bytes))
                {
                    let values = dequantize_block(block, self.dtype);
                    for index in 0..32 {
                        sum += x[index] * values[index];
                    }
                }
            }
            _ => unreachable!("validated weight type"),
        }
        sum
    }
}

fn read_f32(bytes: &[u8]) -> f32 {
    f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}
fn read_f16(bytes: &[u8]) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn dequantize_block(bytes: &[u8], dtype: GgmlType) -> [f32; 32] {
    let scale = read_f16(bytes);
    let mut values = [0.0; 32];
    match dtype {
        GgmlType::Q4_0 => {
            // ggml 的低 nibble 对应前 16 个数，高 nibble 对应后 16 个数，不能交错。
            for (index, &pair) in bytes[2..].iter().enumerate() {
                values[index] = f32::from((pair & 0x0f) as i8 - 8) * scale;
                values[index + 16] = f32::from((pair >> 4) as i8 - 8) * scale;
            }
        }
        GgmlType::Q8_0 => {
            let quants = bytes[2..].as_chunks::<2>().0;
            for (index, pair) in quants.iter().enumerate() {
                values[index * 2] = f32::from(pair[0] as i8) * scale;
                values[index * 2 + 1] = f32::from(pair[1] as i8) * scale;
            }
        }
        _ => unreachable!("validated weight type"),
    }
    values
}
