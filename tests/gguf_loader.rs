//! GGUF 权重加载的集成测试。
//!
//! 全部用例手工编码 GGUF 字节流并写入临时文件，不依赖任何外部模型文件。
//! 张量的 `dims` 按 ggml 的 `ne[]` 顺序书写（`ne[0]` 最快变化），与真实文件一致。

use std::fs;
use std::path::PathBuf;

use qllama::gguf::{GgufFile, GgufValue};

/// 待写入的张量规格。`dims` 为 ggml 的 ne 顺序。
struct TensorSpec {
    name: String,
    dims: Vec<u64>,
    /// GGML 类型的文件编号：0=F32，1=F16，2=Q4_0。
    type_id: u32,
    data: Vec<u8>,
}

impl TensorSpec {
    fn f32(name: &str, dims: Vec<u64>, values: &[f32]) -> Self {
        let mut data = Vec::with_capacity(values.len() * 4);
        for value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        Self {
            name: name.to_string(),
            dims,
            type_id: 0,
            data,
        }
    }

    fn f16(name: &str, dims: Vec<u64>, bits: &[u16]) -> Self {
        let mut data = Vec::with_capacity(bits.len() * 2);
        for value in bits {
            data.extend_from_slice(&value.to_le_bytes());
        }
        Self {
            name: name.to_string(),
            dims,
            type_id: 1,
            data,
        }
    }

    fn raw(name: &str, dims: Vec<u64>, type_id: u32, data: Vec<u8>) -> Self {
        Self {
            name: name.to_string(),
            dims,
            type_id,
            data,
        }
    }
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_string(out: &mut Vec<u8>, value: &str) {
    put_u64(out, value.len() as u64);
    out.extend_from_slice(value.as_bytes());
}

/// 待写入的 metadata 规格。
enum MetadataSpec {
    Str(String),
    U32(u32),
}

fn put_metadata(out: &mut Vec<u8>, key: &str, value: &MetadataSpec) {
    put_string(out, key);
    match value {
        MetadataSpec::Str(s) => {
            put_u32(out, 8); // GGUF 字符串类型
            put_string(out, s);
        }
        MetadataSpec::U32(n) => {
            put_u32(out, 4); // GGUF UINT32 类型
            put_u32(out, *n);
        }
    }
}

/// 把张量规格序列化成完整 GGUF 字节流。
///
/// `alignment` 同时作用于数据段起点与每个张量的起始偏移，与 llama.cpp 一致。
fn build_gguf(
    tensors: &[TensorSpec],
    metadata: &[(&str, MetadataSpec)],
    alignment: u64,
) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(b"GGUF");
    put_u32(&mut body, 3);
    put_u64(&mut body, tensors.len() as u64);
    put_u64(&mut body, metadata.len() as u64);
    for (key, value) in metadata {
        put_metadata(&mut body, key, value);
    }

    // 数据段：每个张量按 alignment 对齐后追加。
    let mut data = Vec::new();
    let mut offsets = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        let pad = (alignment - (data.len() as u64 % alignment)) % alignment;
        data.resize(data.len() + pad as usize, 0u8);
        offsets.push(data.len() as u64);
        data.extend_from_slice(&tensor.data);
    }

    for (tensor, offset) in tensors.iter().zip(&offsets) {
        put_string(&mut body, &tensor.name);
        put_u32(&mut body, tensor.dims.len() as u32);
        for dim in &tensor.dims {
            put_u64(&mut body, *dim);
        }
        put_u32(&mut body, tensor.type_id);
        put_u64(&mut body, *offset);
    }

    let pad = (alignment - (body.len() as u64 % alignment)) % alignment;
    body.resize(body.len() + pad as usize, 0u8);
    body.extend(data);
    body
}

const NO_METADATA: &[(&str, MetadataSpec)] = &[];

/// 每个测试使用独立目录，避免并行执行时互相覆盖。
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("qllama-loader-{name}-{}", std::process::id()));
    fs::create_dir_all(&dir).expect("创建临时目录失败");
    dir
}

fn write_gguf(name: &str, tensors: &[TensorSpec], alignment: u64) -> PathBuf {
    let dir = scratch(name);
    let path = dir.join("model.gguf");
    fs::write(&path, build_gguf(tensors, NO_METADATA, alignment)).expect("写入 GGUF 失败");
    path
}

#[test]
fn mapped_q4_rows_use_separate_low_and_high_nibble_planes() {
    let mut bytes = vec![0x00, 0x3c];
    bytes.extend((0u8..16).map(|i| i | ((15 - i) << 4)));
    let path = write_gguf(
        "mapped-q4-planes",
        &[TensorSpec::raw("w", vec![32, 1], 2, bytes)],
        32,
    );
    let gguf = GgufFile::from_path(&path).unwrap();
    let weight = gguf.load_weight("w").unwrap();
    let expected: Vec<f32> = (-8..8).chain((-8..8).rev()).map(|n| n as f32).collect();
    assert_eq!(weight.read_row(0).unwrap(), expected);
    assert_eq!(weight.storage_bytes(), 18);
    assert_eq!(weight.dims(), &[1, 32]);
}

#[test]
fn mapped_q8_preserves_signed_bytes_and_half_scale() {
    let mut bytes = vec![0x00, 0x38]; // 0.5
    let values: Vec<i8> = (0..32)
        .map(|i| {
            if i == 0 {
                -128
            } else if i == 31 {
                127
            } else {
                i - 16
            }
        })
        .collect();
    bytes.extend(values.iter().map(|&v| v as u8));
    let path = write_gguf(
        "mapped-q8-signed",
        &[TensorSpec::raw("w", vec![32, 1], 8, bytes)],
        32,
    );
    let weight = GgufFile::from_path(&path)
        .unwrap()
        .load_weight("w")
        .unwrap();
    assert_eq!(
        weight.read_row(0).unwrap(),
        values
            .iter()
            .map(|&v| f32::from(v) * 0.5)
            .collect::<Vec<_>>()
    );
    assert_eq!(weight.storage_bytes(), 34);
}

#[test]
fn mapped_quantized_matmul_handles_multiple_rows_without_transposing_storage() {
    let mut bytes = vec![0x00, 0x3c];
    bytes.extend([0xf0; 16]); // -8 前半，+7 后半。
    bytes.extend([0x00, 0x40]);
    bytes.extend([0x87; 16]); // scale=2，前半 -2，后半 0。
    let path = write_gguf(
        "mapped-q4-matmul",
        &[TensorSpec::raw("w", vec![32, 2], 2, bytes)],
        32,
    );
    let weight = GgufFile::from_path(&path)
        .unwrap()
        .load_weight("w")
        .unwrap();
    let input: Vec<f32> = [vec![1.0; 32], vec![2.0; 16], vec![-1.0; 16]].concat();
    let output = weight
        .matmul(&qllama::tensor::Tensor::new(input, vec![2, 32]))
        .unwrap();
    assert_eq!(output.dims(), &[2, 2]);
    assert_eq!(output.data(), &[-16.0, -32.0, -368.0, -64.0]);
}

#[test]
fn mapped_single_row_matmul_preserves_bits_across_threads_and_column_tails() {
    let width = 64;
    for type_id in [0, 1, 2, 8] {
        for columns in [63, 64, 65, 129] {
            let mut bytes = Vec::new();
            for row in 0..columns {
                if type_id <= 1 {
                    for inner in 0..width {
                        let bits = [0x3555u16, 0xbc00, 0x3800, 0x4000][(row + inner) % 4];
                        if type_id == 1 {
                            bytes.extend_from_slice(&bits.to_le_bytes());
                        } else {
                            let value = ((row * 17 + inner * 29) % 101) as f32 / 13.0 - 3.0;
                            bytes.extend_from_slice(&value.to_le_bytes());
                        }
                    }
                } else {
                    for block in 0..width / 32 {
                        let scale = [0x3555u16, 0xbc00, 0x3800][(row + block) % 3];
                        bytes.extend_from_slice(&scale.to_le_bytes());
                        let size = if type_id == 2 { 16 } else { 32 };
                        bytes.extend(
                            (0..size).map(|i| ((row * 17 + block * 29 + i * 31) % 256) as u8),
                        );
                    }
                }
            }
            let path = write_gguf(
                &format!("mapped-thread-{type_id}-{columns}"),
                &[TensorSpec::raw(
                    "w",
                    vec![width as u64, columns as u64],
                    type_id,
                    bytes,
                )],
                32,
            );
            let weight = GgufFile::from_path(&path)
                .unwrap()
                .load_weight("w")
                .unwrap();
            let input: Vec<f32> = (0..width).map(|i| (i as f32 - 31.0) / 11.0).collect();
            let expected: Vec<u32> = (0..columns)
                .map(|row| {
                    let values = weight.read_row(row).unwrap();
                    let mut sum = 0.0f32;
                    for inner in 0..width {
                        sum += input[inner] * values[inner];
                    }
                    sum.to_bits()
                })
                .collect();
            let input = qllama::Tensor::new(input, vec![1, width]);
            for threads in [1, 2, 4, 8] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let actual = pool.install(|| weight.matmul(&input)).unwrap();
                assert_eq!(actual.dims(), &[1, columns]);
                assert_eq!(
                    actual
                        .data()
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    expected,
                    "type={type_id}, columns={columns}, threads={threads}"
                );
            }
        }
    }
}

#[test]
fn mapped_quantized_zero_and_negative_scales_are_respected() {
    for (name, scale, expected) in [("zero", 0x0000u16, 0.0), ("negative", 0xbc00u16, 8.0)] {
        let mut bytes = scale.to_le_bytes().to_vec();
        bytes.extend([0x00; 16]);
        let path = write_gguf(name, &[TensorSpec::raw("w", vec![32, 1], 2, bytes)], 32);
        let weight = GgufFile::from_path(&path)
            .unwrap()
            .load_weight("w")
            .unwrap();
        assert_eq!(weight.read_row(0).unwrap(), vec![expected; 32]);
    }
}

#[test]
fn mapped_weights_keep_the_mapping_alive_and_f16_stays_compressed() {
    let path = write_gguf(
        "mapped-lifetime",
        &[TensorSpec::f16(
            "w",
            vec![2, 2],
            &[0x3c00, 0xc000, 0x3800, 0x4000],
        )],
        32,
    );
    let gguf = GgufFile::from_path(&path).unwrap();
    let weight = gguf.load_weight("w").unwrap();
    drop(gguf);
    assert_eq!(weight.storage_bytes(), 8);
    assert_eq!(weight.read_row(1).unwrap(), vec![0.5, 2.0]);
    let input = qllama::tensor::Tensor::new(vec![2.0, -1.0], vec![1, 2]);
    assert_eq!(weight.matmul(&input).unwrap().data(), &[4.0, -1.0]);
    assert!(weight.read_row(2).is_err());
    assert!(weight
        .matmul(&qllama::tensor::Tensor::new(vec![1.0], vec![1, 1]))
        .is_err());
}

#[test]
fn mapped_weights_reject_truncated_blocks_and_unaligned_rows() {
    for (name, dims, bytes) in [
        ("truncated", vec![32, 1], vec![0u8; 17]),
        ("unaligned", vec![31, 1], vec![0u8; 18]),
    ] {
        let path = write_gguf(name, &[TensorSpec::raw("w", dims, 2, bytes)], 32);
        assert!(GgufFile::from_path(&path)
            .unwrap()
            .load_weight("w")
            .is_err());
    }
}

#[test]
fn mapped_weights_reject_unsupported_types_and_shape_overflow() {
    let path = write_gguf(
        "mapped-unsupported",
        &[TensorSpec::raw("w", vec![256, 1], 14, vec![0u8; 210])],
        32,
    );
    let error = GgufFile::from_path(&path)
        .unwrap()
        .load_weight("w")
        .unwrap_err();
    assert!(error.contains("Q6_K"), "{error}");
    let path = write_gguf(
        "mapped-overflow",
        &[TensorSpec::raw("w", vec![u64::MAX, 2], 0, vec![])],
        32,
    );
    assert!(GgufFile::from_path(&path)
        .unwrap()
        .load_weight("w")
        .is_err());
}

#[test]
fn loads_f32_tensor_values() {
    let path = write_gguf(
        "f32-values",
        &[TensorSpec::f32("w", vec![2, 2], &[1.5, -2.25, 3.0, 4.125])],
        32,
    );
    let gguf = GgufFile::from_path(&path).expect("解析失败");

    let tensor = gguf.load_tensor_f32("w").expect("加载失败");
    assert_eq!(tensor.dims(), &[2, 2]);
    assert_eq!(tensor.data(), &[1.5, -2.25, 3.0, 4.125]);
    assert_eq!(tensor.get(&[1, 0]), Some(3.0));

    fs::remove_dir_all(path.parent().unwrap()).ok();
}

#[test]
fn loads_f16_tensor_and_converts_to_f32() {
    // 0x3C00=1.0, 0xC000=-2.0, 0x0000=0.0, 0x4000=2.0
    let path = write_gguf(
        "f16-values",
        &[TensorSpec::f16(
            "h",
            vec![4],
            &[0x3C00, 0xC000, 0x0000, 0x4000],
        )],
        32,
    );
    let gguf = GgufFile::from_path(&path).expect("解析失败");

    let tensor = gguf.load_tensor_f16("h").expect("加载失败");
    assert_eq!(tensor.dims(), &[4]);
    assert_eq!(tensor.data(), &[1.0, -2.0, 0.0, 2.0]);

    fs::remove_dir_all(path.parent().unwrap()).ok();
}

#[test]
fn f16_conversion_preserves_precision_beyond_f32_rounding() {
    // 0x3555 = (1 + 341/1024) × 2^-2 = 1365/4096，f32 可精确表示。
    let path = write_gguf(
        "f16-precision",
        &[TensorSpec::f16("p", vec![2], &[0x3555, 0x7BFF])],
        32,
    );
    let gguf = GgufFile::from_path(&path).expect("解析失败");

    let tensor = gguf.load_tensor_f16("p").expect("加载失败");
    assert_eq!(tensor.get(&[0]), Some(1365.0 / 4096.0));
    assert_eq!(tensor.get(&[1]), Some(65504.0));

    fs::remove_dir_all(path.parent().unwrap()).ok();
}

#[test]
fn reverses_ggml_dimension_order() {
    // ne = {3, 2}：3 个连续元素为一行，共 2 行。反转为 [2, 3] 后，
    // get(&[0, 1]) 取第一行第二个元素；若不反转会得到 data[3]=40。
    let path = write_gguf(
        "dims-reversal",
        &[TensorSpec::f32(
            "w",
            vec![3, 2],
            &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        )],
        32,
    );
    let gguf = GgufFile::from_path(&path).expect("解析失败");

    let tensor = gguf.load_tensor_f32("w").expect("加载失败");
    assert_eq!(tensor.dims(), &[2, 3]);
    assert_eq!(tensor.get(&[0, 0]), Some(10.0));
    assert_eq!(tensor.get(&[0, 1]), Some(20.0));
    assert_eq!(tensor.get(&[0, 2]), Some(30.0));
    assert_eq!(tensor.get(&[1, 0]), Some(40.0));
    assert_eq!(tensor.get(&[1, 2]), Some(60.0));

    fs::remove_dir_all(path.parent().unwrap()).ok();
}

#[test]
fn reverses_three_dimensional_dimension_order() {
    // ne = {1, 2, 3}：反转为 [3, 2, 1]。若不反转，dims 会是 [1, 2, 3]，
    // get(&[1, 0, 0]) 将算出扁平索引 6，超出 6 个元素的缓冲区。
    let values: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let path = write_gguf(
        "dims-reversal-3d",
        &[TensorSpec::f32("c", vec![1, 2, 3], &values)],
        32,
    );
    let gguf = GgufFile::from_path(&path).expect("解析失败");

    let tensor = gguf.load_tensor_f32("c").expect("加载失败");
    assert_eq!(tensor.dims(), &[3, 2, 1]);
    assert_eq!(tensor.get(&[0, 0, 0]), Some(0.0));
    assert_eq!(tensor.get(&[1, 0, 0]), Some(2.0));
    assert_eq!(tensor.get(&[2, 1, 0]), Some(5.0));

    fs::remove_dir_all(path.parent().unwrap()).ok();
}

#[test]
fn embedding_layout_matches_token_row_lookup() {
    // token_embd 的 ne 是 {n_embd, n_vocab}；反转为 [n_vocab, n_embd] 后，
    // 按 token id 取行必须命中该 id 对应的 n_embd 个连续元素。
    let vocab = 3;
    let n_embd = 2;
    let values: Vec<f32> = (0..(vocab * n_embd)).map(|i| (i + 1) as f32).collect();
    let path = write_gguf(
        "embedding-layout",
        &[TensorSpec::f32(
            "token_embd.weight",
            vec![n_embd as u64, vocab as u64],
            &values,
        )],
        32,
    );
    let gguf = GgufFile::from_path(&path).expect("解析失败");

    let tensor = gguf.load_tensor_f32("token_embd.weight").expect("加载失败");
    assert_eq!(tensor.dims(), &[vocab, n_embd]);
    assert_eq!(tensor.get(&[0, 0]), Some(1.0));
    assert_eq!(tensor.get(&[0, 1]), Some(2.0));
    assert_eq!(tensor.get(&[2, 0]), Some(5.0));
    assert_eq!(tensor.get(&[2, 1]), Some(6.0));

    fs::remove_dir_all(path.parent().unwrap()).ok();
}

#[test]
fn load_tensor_dispatches_on_stored_type() {
    let path = write_gguf(
        "dispatch",
        &[
            TensorSpec::f32("a", vec![2], &[7.0, 8.0]),
            TensorSpec::f16("b", vec![2], &[0x3C00, 0x4000]),
        ],
        32,
    );
    let gguf = GgufFile::from_path(&path).expect("解析失败");

    assert_eq!(gguf.load_tensor("a").unwrap().data(), &[7.0, 8.0]);
    assert_eq!(gguf.load_tensor("b").unwrap().data(), &[1.0, 2.0]);

    fs::remove_dir_all(path.parent().unwrap()).ok();
}

#[test]
fn rejects_unknown_tensor_name() {
    let path = write_gguf("missing-name", &[TensorSpec::f32("w", vec![1], &[1.0])], 32);
    let gguf = GgufFile::from_path(&path).expect("解析失败");

    let error = gguf
        .load_tensor_f32("nope")
        .expect_err("应拒绝不存在的张量");
    assert!(error.contains("nope"), "错误信息应指出名称: {error}");

    fs::remove_dir_all(path.parent().unwrap()).ok();
}

#[test]
fn rejects_quantized_tensor_types() {
    // 文件编号 2 = Q4_0。data 长度任意，加载器应在解码前就按类型拒绝。
    let path = write_gguf(
        "quantized",
        &[TensorSpec::raw("q", vec![32], 2, vec![0u8; 18])],
        32,
    );
    let gguf = GgufFile::from_path(&path).expect("解析失败");

    let error = gguf.load_tensor("q").expect_err("应拒绝量化类型");
    assert!(
        error.contains("Quantized") && error.contains("Q4_0"),
        "错误信息应说明是量化类型: {error}"
    );

    fs::remove_dir_all(path.parent().unwrap()).ok();
}

#[test]
fn rejects_non_floating_tensor_types() {
    // 文件编号 26 = I32，既不是量化也不在本阶段支持范围。
    let path = write_gguf(
        "integer-type",
        &[TensorSpec::raw("i", vec![4], 26, vec![0u8; 16])],
        32,
    );
    let gguf = GgufFile::from_path(&path).expect("解析失败");

    let error = gguf.load_tensor("i").expect_err("应拒绝整数类型");
    assert!(
        error.contains("only F32 and F16"),
        "错误信息应与量化类型区分: {error}"
    );

    fs::remove_dir_all(path.parent().unwrap()).ok();
}

#[test]
fn rejects_type_mismatch_between_loader_and_file() {
    let path = write_gguf(
        "type-mismatch",
        &[TensorSpec::f16("h", vec![2], &[0x3C00, 0x3C00])],
        32,
    );
    let gguf = GgufFile::from_path(&path).expect("解析失败");

    let error = gguf
        .load_tensor_f32("h")
        .expect_err("应按 F32 加载 F16 张量应失败");
    assert!(
        error.contains("F16") && error.contains("F32"),
        "错误应指出类型差异: {error}"
    );

    fs::remove_dir_all(path.parent().unwrap()).ok();
}

#[test]
fn locates_tensors_at_nonzero_offsets() {
    // 第一个张量 5 个 f32 = 20 字节，不对齐到 32，因此第二个张量的 offset 非零。
    let path = write_gguf(
        "nonzero-offset",
        &[
            TensorSpec::f32("first", vec![5], &[1.0, 2.0, 3.0, 4.0, 5.0]),
            TensorSpec::f32("second", vec![2], &[99.0, -99.0]),
        ],
        32,
    );
    let gguf = GgufFile::from_path(&path).expect("解析失败");

    let second = gguf.find_tensor("second").expect("找不到 second");
    assert!(
        second.offset > 0,
        "第二个张量的 offset 应为非零，得到 {}",
        second.offset
    );

    assert_eq!(
        gguf.load_tensor_f32("first").unwrap().data(),
        &[1.0, 2.0, 3.0, 4.0, 5.0]
    );
    assert_eq!(
        gguf.load_tensor_f32("second").unwrap().data(),
        &[99.0, -99.0]
    );

    fs::remove_dir_all(path.parent().unwrap()).ok();
}

#[test]
fn honours_non_default_alignment() {
    // alignment = 64 时，第二个张量的 offset 应向上取齐到 64 的倍数。
    let path = write_gguf(
        "alignment-64",
        &[
            TensorSpec::f32("first", vec![5], &[1.0, 2.0, 3.0, 4.0, 5.0]),
            TensorSpec::f32("second", vec![1], &[42.0]),
        ],
        64,
    );
    let gguf = GgufFile::from_path(&path).expect("解析失败");

    let second = gguf.find_tensor("second").expect("找不到 second");
    assert_eq!(
        second.offset % 64,
        0,
        "offset 应按 64 对齐，得到 {}",
        second.offset
    );
    assert_eq!(gguf.load_tensor_f32("second").unwrap().data(), &[42.0]);

    fs::remove_dir_all(path.parent().unwrap()).ok();
}

#[test]
fn loads_one_dimensional_norm_weights() {
    let path = write_gguf(
        "norm-weight",
        &[TensorSpec::f32("norm.weight", vec![3], &[0.5, 1.5, 2.5])],
        32,
    );
    let gguf = GgufFile::from_path(&path).expect("解析失败");

    let tensor = gguf.load_tensor_f32("norm.weight").expect("加载失败");
    assert_eq!(tensor.dims(), &[3]);
    assert_eq!(tensor.data(), &[0.5, 1.5, 2.5]);

    fs::remove_dir_all(path.parent().unwrap()).ok();
}

#[test]
fn rejects_tensor_whose_data_runs_past_the_end_of_the_file() {
    let mut tensors = vec![TensorSpec::f32("w", vec![4], &[1.0, 2.0, 3.0, 4.0])];
    // 声称有 64 个元素，但只提供 4 个的字节，数据段必然越界。
    tensors.push(TensorSpec::f32("short", vec![64], &[1.0, 2.0, 3.0, 4.0]));
    let path = write_gguf("truncated-data", &tensors, 32);
    let gguf = GgufFile::from_path(&path).expect("索引解析应成功");

    assert!(gguf.load_tensor_f32("w").is_ok(), "完整张量应可加载");
    let error = gguf.load_tensor_f32("short").expect_err("越界张量应被拒绝");
    assert!(error.contains("exceeds"), "错误应说明越界: {error}");

    fs::remove_dir_all(path.parent().unwrap()).ok();
}

#[test]
fn tensor_data_is_unavailable_without_a_file_mapping() {
    let bytes = build_gguf(
        &[TensorSpec::f32("w", vec![2], &[1.0, 2.0])],
        NO_METADATA,
        32,
    );
    let gguf = GgufFile::from_reader(&mut std::io::Cursor::new(bytes)).expect("解析失败");

    let error = gguf.load_tensor_f32("w").expect_err("无映射时应报错");
    assert!(
        error.contains("from_path"),
        "错误应说明只有 from_path 保留映射: {error}"
    );
}

#[test]
fn loaded_tensors_are_independent_copies() {
    let path = write_gguf(
        "independent",
        &[TensorSpec::f32("w", vec![2], &[1.0, 2.0])],
        32,
    );
    let gguf = GgufFile::from_path(&path).expect("解析失败");

    let first = gguf.load_tensor_f32("w").unwrap();
    let mut second = gguf.load_tensor_f32("w").unwrap();
    second.set(&[0], 100.0).unwrap();
    assert_eq!(first.get(&[0]), Some(1.0), "第二次加载不应影响第一次");

    fs::remove_dir_all(path.parent().unwrap()).ok();
}

#[test]
fn mapping_path_agrees_with_reader_path_on_index_and_data() {
    // 加内存映射不能改变索引解析结果，权重数值也必须一致。
    let tensors = [
        TensorSpec::f32("w", vec![2, 2], &[1.0, 2.0, 3.0, 4.0]),
        TensorSpec::f16("b", vec![2], &[0x3C00, 0x4000]),
    ];
    let metadata = [
        (
            "general.architecture",
            MetadataSpec::Str("llama".to_string()),
        ),
        ("llama.block_count", MetadataSpec::U32(3)),
    ];
    let bytes = build_gguf(&tensors, &metadata, 32);

    let dir = scratch("index-parity");
    let path = dir.join("model.gguf");
    fs::write(&path, &bytes).expect("写入失败");

    let from_path = GgufFile::from_path(&path).expect("from_path 解析失败");
    let from_reader =
        GgufFile::from_reader(&mut std::io::Cursor::new(bytes)).expect("from_reader 解析失败");

    assert_eq!(from_path.tensor_count(), from_reader.tensor_count());
    assert_eq!(from_path.metadata_count(), from_reader.metadata_count());
    assert_eq!(
        from_path.tensor_data_offset, from_reader.tensor_data_offset,
        "两条路径算出的数据段起点必须一致"
    );
    // GgufValue 未实现 PartialEq，用模式匹配核对架构字段。
    match from_path.metadata.get("general.architecture") {
        Some(GgufValue::String(arch)) => assert_eq!(arch, "llama"),
        other => panic!("general.architecture 应为字符串，得到 {other:?}"),
    }
    assert_eq!(
        from_path.load_tensor_f32("w").unwrap().data(),
        &[1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(from_path.load_tensor_f16("b").unwrap().data(), &[1.0, 2.0]);

    fs::remove_dir_all(&dir).ok();
}
