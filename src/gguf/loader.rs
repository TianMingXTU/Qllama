//! GGUF 权重数据加载。
//!
//! `parser.rs` 的索引解析只记录张量的名称、维度、类型与相对偏移；本模块把
//! `tensor_data_offset` 之后的真实字节读成 `Tensor`。
//!
//! ## 维度顺序（关键约定）
//!
//! GGUF 的张量维度按 ggml 的 `ne[]` 顺序存放（`gguf.cpp` 直接把读到的值写进
//! `info.t.ne[j]`），而 ggml 的 `ne[0]` 是**最快变化**的连续维度
//! （`nb[0] = ggml_type_size(type)`）。`Tensor` 的 `compute_strides` 恰好相反：
//! `dims[0]` 是最慢的外层，`dims[last]` 才是连续维度。
//!
//! 因此加载时必须**反转** GGUF 的 dims。以 `token_embd.weight` 为例，其
//! `ne = {n_embd, n_vocab}`，内存布局是 `n_vocab` 行、每行 `n_embd` 个连续元素；
//! 反转为 `[n_vocab, n_embd]` 后，按 token id 取行才能命中正确区间。
//!
//! 反转之后，权重在内存中的逻辑矩阵就是 `[out_features, in_features]`，
//! 所以 `transformer::layers::Linear` 内部按该方向存储，前向时再转置。

use super::types::{GgmlType, TensorInfo};
use crate::tensor::Tensor;

/// GGUF 半精度与单精度共用的指数域掩码。
const F16_EXPONENT_MASK: u16 = 0x1F;
/// GGUF 半精度尾数掩码。
const F16_MANTISSA_MASK: u16 = 0x03FF;

impl super::GgufFile {
    /// 按名称加载一个 F32 张量。
    ///
    /// 名称不存在、类型不是 F32 或数据段越界时返回错误。
    pub fn load_tensor_f32(&self, name: &str) -> Result<Tensor, String> {
        self.load_typed(name, GgmlType::F32)
    }

    /// 按名称加载一个 F16 张量并转换为 F32。
    ///
    /// 半精度到单精度的转换是精确的：所有半精度值都能在 f32 中无损表示。
    pub fn load_tensor_f16(&self, name: &str) -> Result<Tensor, String> {
        self.load_typed(name, GgmlType::F16)
    }

    /// 通用入口：自动识别 F32/F16，其余类型返回明确错误。
    ///
    /// 量化类型与整数/双精度类型均不在本阶段支持范围内，分别给出不同措辞的
    /// 错误，避免把"暂不支持的量化"和"本就不该出现的类型"混为一谈。
    pub fn load_tensor(&self, name: &str) -> Result<Tensor, String> {
        let info = self.tensor_info(name)?;
        match info.ggml_type {
            GgmlType::F32 => self.load_typed(name, GgmlType::F32),
            GgmlType::F16 => self.load_typed(name, GgmlType::F16),
            other if other.is_quantized() => {
                Err(format!("Quantized type {other:?} not yet supported"))
            }
            other => Err(format!(
                "Unsupported tensor type {other:?}; only F32 and F16 are supported"
            )),
        }
    }

    /// 定位张量信息；名称不存在时给出可读的错误。
    fn tensor_info(&self, name: &str) -> Result<&TensorInfo, String> {
        self.tensor_infos
            .iter()
            .find(|info| info.name == name)
            .ok_or_else(|| format!("Tensor not found: {name}"))
    }

    /// 按期望类型加载并解码张量。
    fn load_typed(&self, name: &str, expected: GgmlType) -> Result<Tensor, String> {
        let info = self.tensor_info(name)?;
        if info.ggml_type != expected {
            return Err(format!(
                "Tensor {name} has type {:?}, expected {expected:?}",
                info.ggml_type
            ));
        }

        let element_size = expected.element_size();
        if element_size == 0 {
            return Err(format!("Type {expected:?} has no scalar element size"));
        }

        // GGUF 的 dims 按 ggml 的 ne[] 顺序存放，ne[0] 最快变化；Tensor 的
        // dims[0] 是最慢外层，故反转。先转换再求积，越界维度在此一次性拦截。
        let dims: Vec<usize> = info
            .dims
            .iter()
            .rev()
            .map(|&dim| {
                usize::try_from(dim)
                    .map_err(|_| format!("Tensor {name} dimension {dim} exceeds usize"))
            })
            .collect::<Result<_, _>>()?;
        let element_count = checked_product(&dims)?;
        let byte_len = element_count
            .checked_mul(element_size)
            .ok_or_else(|| format!("Tensor {name} byte length overflows usize"))?;

        let bytes = self.tensor_bytes(info, byte_len)?;
        let mut data = Vec::new();
        data.try_reserve_exact(element_count)
            .map_err(|error| format!("Cannot allocate {element_count} elements: {error}"))?;
        // byte_len 已按 element_count * element_size 校验，故余部必为空；
        // 下方 data.len() 与 element_count 的自洽检查作为兜底。
        match expected {
            GgmlType::F32 => {
                let (chunks, _) = bytes.as_chunks::<4>();
                data.extend(chunks.iter().map(|chunk| f32::from_le_bytes(*chunk)));
            }
            GgmlType::F16 => {
                let (chunks, _) = bytes.as_chunks::<2>();
                data.extend(
                    chunks
                        .iter()
                        .map(|chunk| f16_to_f32(u16::from_le_bytes(*chunk))),
                );
            }
            // 调用方已按 expected 校验过类型，其余分支不可达。
            other => return Err(format!("Type {other:?} is not decodable as f32")),
        }

        // 解码出的元素数必须与形状自洽；不一致说明上面的偏移或类型判断有误。
        if data.len() != element_count {
            return Err(format!(
                "Tensor {name} decoded {len} elements but its shape {dims:?} requires {element_count}",
                len = data.len()
            ));
        }

        Ok(Tensor::new(data, dims))
    }

    /// 从映射中取出张量数据段的字节切片。
    ///
    /// 偏移与长度均来自不可信的文件内容，全部使用 checked 运算；切片越界时
    /// 返回错误而非 panic。
    fn tensor_bytes(&self, info: &TensorInfo, byte_len: usize) -> Result<&[u8], String> {
        let map = self.tensor_data.as_ref().ok_or_else(|| {
            format!(
                "Tensor data for {} is unavailable: only GgufFile::from_path keeps the mapping",
                info.name
            )
        })?;

        let start = self
            .tensor_data_offset
            .checked_add(info.offset)
            .ok_or_else(|| format!("Tensor {} offset overflows u64", info.name))?;
        let start = usize::try_from(start)
            .map_err(|_| format!("Tensor {} offset exceeds usize range", info.name))?;
        let end = start
            .checked_add(byte_len)
            .ok_or_else(|| format!("Tensor {} end offset overflows usize", info.name))?;

        map.get(start..end).ok_or_else(|| {
            format!(
                "Tensor {} spans [{start}, {end}) which exceeds the mapped {} bytes",
                info.name,
                map.len()
            )
        })
    }
}

/// 计算维度乘积，乘积溢出 usize 时返回错误。
fn checked_product(dims: &[usize]) -> Result<usize, String> {
    dims.iter().try_fold(1usize, |product, &dim| {
        product
            .checked_mul(dim)
            .ok_or_else(|| format!("Tensor shape product overflows usize: {dims:?}"))
    })
}

/// 把 IEEE 754 半精度位模式转换成 f32。
///
/// 手工位运算实现，避免为此新增 crate。半精度为 1 符号位、5 指数位（偏置 15）、
/// 10 尾数位；指数域为 0 表示次正规数（`m × 2^-24`），全 1 表示无穷或 NaN。
///
/// 每个分支在 f32 中都精确可表示：规格化数的尾数只需 11 位，次正规数是 2^-24
/// 的整数倍，指数缩放不改变有效位，因此不存在额外舍入。
fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = (bits >> 10) & F16_EXPONENT_MASK;
    let mantissa = bits & F16_MANTISSA_MASK;

    let magnitude = match exponent {
        0 => {
            if mantissa == 0 {
                0.0
            } else {
                f32::from(mantissa) * 2f32.powi(-24)
            }
        }
        0x1F => {
            if mantissa == 0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => (1.0 + f32::from(mantissa) / 1024.0) * 2f32.powi(i32::from(exponent) - 15),
    };

    sign * magnitude
}

/// 供测试直接验证转换函数，避免每个用例都构造一个 GGUF 文件。
#[cfg(test)]
pub(super) fn f16_to_f32_for_test(bits: u16) -> f32 {
    f16_to_f32(bits)
}

/// 加载器的单元测试只覆盖纯函数；涉及文件与映射的用例见 `tests/gguf_loader.rs`。
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_conversion_matches_hand_computed_values() {
        // 规格化数。
        assert_eq!(f16_to_f32_for_test(0x3C00), 1.0);
        assert_eq!(f16_to_f32_for_test(0xBC00), -1.0);
        assert_eq!(f16_to_f32_for_test(0x4000), 2.0);
        assert_eq!(f16_to_f32_for_test(0xC000), -2.0);
        // (1 + 341/1024) × 2^(13-15) = 1365/4096
        assert_eq!(f16_to_f32_for_test(0x3555), 1365.0 / 4096.0);
        // 最大规格化数：(1 + 1023/1024) × 2^15 = 2047 × 32
        assert_eq!(f16_to_f32_for_test(0x7BFF), 65504.0);
        // 最小规格化数：2^-14
        assert_eq!(f16_to_f32_for_test(0x0400), 2f32.powi(-14));
        // 次正规数：m × 2^-24
        assert_eq!(f16_to_f32_for_test(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32_for_test(0x03FF), 1023.0 * 2f32.powi(-24));
        // 零与无穷。
        assert_eq!(f16_to_f32_for_test(0x0000), 0.0);
        assert!(f16_to_f32_for_test(0x8000).is_sign_negative());
        assert_eq!(f16_to_f32_for_test(0x8000), 0.0);
        assert_eq!(f16_to_f32_for_test(0x7C00), f32::INFINITY);
        assert_eq!(f16_to_f32_for_test(0xFC00), f32::NEG_INFINITY);
        // NaN 保持为 NaN，不静默变成数值。
        assert!(f16_to_f32_for_test(0x7E00).is_nan());
        assert!(f16_to_f32_for_test(0xFFFF).is_nan());
    }

    #[test]
    fn f16_conversion_is_exact_for_every_representable_value() {
        // 穷举全部 65536 个位模式，逐一对照独立参考实现。
        for bits in 0..=u16::MAX {
            let expected = reference_f16_to_f32(bits);
            let actual = f16_to_f32_for_test(bits);
            if expected.is_nan() {
                assert!(actual.is_nan(), "0x{bits:04X} 应为 NaN，得到 {actual}");
            } else {
                assert_eq!(
                    actual, expected,
                    "0x{bits:04X} 应为 {expected}，得到 {actual}"
                );
            }
        }
    }

    /// 独立参考实现：直接按半精度格式定义构造 f32 位模式。
    ///
    /// 与 `f16_to_f32` 的算术路径完全独立，用于穷举交叉验证。
    fn reference_f16_to_f32(bits: u16) -> f32 {
        let sign = (u32::from(bits) & 0x8000) << 16;
        let exponent = u32::from((bits >> 10) & 0x1F);
        let mantissa = u32::from(bits & 0x03FF);

        if exponent == 0 {
            if mantissa == 0 {
                return f32::from_bits(sign);
            }
            // 次正规数 m × 2^-24：设 m 的最高位在位置 p，则值为 (1.f) × 2^(p-24)，
            // f32 指数域为 p + 103。把 m 左移到 f32 的 23 位宽，每移一位指数减一，
            // 故初始值为 112 而非 113。
            let mut shifted = mantissa << 14;
            let mut exponent_field = 112;
            while shifted & 0x0080_0000 == 0 {
                shifted <<= 1;
                exponent_field -= 1;
            }
            return f32::from_bits(
                sign | ((exponent_field as u32) << 23) | (shifted & 0x007F_FFFF),
            );
        }
        if exponent == 0x1F {
            let payload = mantissa << 13;
            return f32::from_bits(sign | 0x7F80_0000 | payload);
        }
        // 规格化数：指数偏置从 15 调整为 127，尾数左移 13 位。
        f32::from_bits(sign | ((exponent + 127 - 15) << 23) | (mantissa << 13))
    }
}
