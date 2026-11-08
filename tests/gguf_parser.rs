//! GGUF parser integration tests.
//!
//! 用手工构造的 GGUF 字节流验证解析行为，避免依赖外部模型文件。

use std::io::{self, Cursor, Read, Seek, SeekFrom};

use qllama::gguf::{GgmlType, GgufFile, GgufValue};

/// 按 GGUF 规范的 little-endian 字节序写入基础类型。
struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }

    fn u8(&mut self, v: u8) -> &mut Self {
        self.buf.extend_from_slice(&v.to_le_bytes());
        self
    }

    fn u32(&mut self, v: u32) -> &mut Self {
        self.buf.extend_from_slice(&v.to_le_bytes());
        self
    }

    fn u64(&mut self, v: u64) -> &mut Self {
        self.buf.extend_from_slice(&v.to_le_bytes());
        self
    }

    /// GGUF 字符串：u64 长度 + UTF-8 字节。
    fn string(&mut self, s: &str) -> &mut Self {
        self.u64(s.len() as u64);
        self.buf.extend_from_slice(s.as_bytes());
        self
    }

    fn f32(&mut self, v: f32) -> &mut Self {
        self.buf.extend_from_slice(&v.to_le_bytes());
        self
    }

    fn finish(self) -> Vec<u8> {
        self.buf
    }
}

/// 构造一个最小但合法的 GGUF 文件：2 条 metadata + 1 个 tensor。
fn build_minimal_gguf() -> Vec<u8> {
    let mut w = Writer::new();

    // Header
    w.buf.extend_from_slice(b"GGUF");
    w.u32(2); // version
    w.u64(1); // tensor_count
    w.u64(2); // metadata_kv_count

    // Metadata: general.architecture = "llama"
    w.string("general.architecture");
    w.u32(8); // type: STRING
    w.string("llama");

    // Metadata: general.embedding_length = 256
    w.string("general.embedding_length");
    w.u32(4); // type: UINT32
    w.u32(256);

    // Tensor info
    w.string("weight.0");
    w.u32(2); // n_dims
    w.u64(4); // dim 0
    w.u64(4); // dim 1
    w.u32(0); // ggml_type: F32
    w.u64(0); // offset

    // Alignment padding to 32 bytes
    while !w.buf.len().is_multiple_of(32) {
        w.u8(0);
    }

    // Tensor data: 4x4 f32 = 64 bytes
    for i in 0..16 {
        w.f32(i as f32);
    }

    w.finish()
}

#[test]
fn parses_header_and_counts() {
    let bytes = build_minimal_gguf();
    let gguf = GgufFile::from_reader(&mut Cursor::new(bytes)).expect("should parse");

    assert_eq!(&gguf.header.magic, b"GGUF");
    assert_eq!(gguf.header.version, 2);
    assert_eq!(gguf.tensor_count(), 1);
    assert_eq!(gguf.metadata_count(), 2);
}

#[test]
fn parses_metadata_values() {
    let bytes = build_minimal_gguf();
    let gguf = GgufFile::from_reader(&mut Cursor::new(bytes)).expect("should parse");

    match gguf.metadata.get("general.architecture") {
        Some(GgufValue::String(s)) => assert_eq!(s, "llama"),
        other => panic!("unexpected architecture value: {:?}", other),
    }

    match gguf.metadata.get("general.embedding_length") {
        Some(GgufValue::Uint32(n)) => assert_eq!(*n, 256),
        other => panic!("unexpected embedding_length value: {:?}", other),
    }
}

#[test]
fn parses_tensor_info() {
    let bytes = build_minimal_gguf();
    let gguf = GgufFile::from_reader(&mut Cursor::new(bytes)).expect("should parse");

    assert_eq!(gguf.tensor_infos.len(), 1);
    let tensor = &gguf.tensor_infos[0];
    assert_eq!(tensor.name, "weight.0");
    assert_eq!(tensor.dims, vec![4, 4]);
    assert_eq!(tensor.offset, 0);

    let found = gguf
        .find_tensor("weight.0")
        .expect("tensor should be found");
    assert_eq!(found.name, "weight.0");
    assert!(gguf.find_tensor("nonexistent").is_none());
}

#[test]
fn tensor_data_offset_is_aligned() {
    let bytes = build_minimal_gguf();
    let total_len = bytes.len();
    let gguf = GgufFile::from_reader(&mut Cursor::new(bytes)).expect("should parse");

    assert_eq!(gguf.tensor_data_offset % 32, 0);
    assert!(gguf.tensor_data_offset < total_len as u64);
}

#[test]
fn rejects_invalid_magic() {
    let mut bytes = build_minimal_gguf();
    bytes[0] = b'X';

    let result = GgufFile::from_reader(&mut Cursor::new(bytes));
    assert!(result.is_err(), "invalid magic should be rejected");
}

#[test]
fn rejects_truncated_file() {
    let bytes = build_minimal_gguf();
    let truncated = &bytes[..bytes.len() / 2];

    let result = GgufFile::from_reader(&mut Cursor::new(truncated.to_vec()));
    assert!(result.is_err(), "truncated file should be rejected");
}

#[test]
fn parses_metadata_array() {
    let mut w = Writer::new();

    w.buf.extend_from_slice(b"GGUF");
    w.u32(2);
    w.u64(0); // tensor_count
    w.u64(1); // metadata_kv_count

    // tokens = ["a", "b", "c"] (STRING array)
    w.string("tokenizer.ggml.tokens");
    w.u32(9); // type: ARRAY
    w.u32(8); // element type: STRING
    w.u64(3); // array length
    w.string("a");
    w.string("b");
    w.string("c");

    while !w.buf.len().is_multiple_of(32) {
        w.u8(0);
    }

    let gguf = GgufFile::from_reader(&mut Cursor::new(w.finish())).expect("should parse");

    match gguf.metadata.get("tokenizer.ggml.tokens") {
        Some(GgufValue::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            assert!(matches!(arr[0], GgufValue::String(ref s) if s == "a"));
            assert!(matches!(arr[1], GgufValue::String(ref s) if s == "b"));
            assert!(matches!(arr[2], GgufValue::String(ref s) if s == "c"));
        }
        other => panic!("unexpected tokens value: {:?}", other),
    }
}

fn header(tensor_count: u64, metadata_count: u64) -> Writer {
    let mut w = Writer::new();
    w.buf.extend_from_slice(b"GGUF");
    w.u32(3).u64(tensor_count).u64(metadata_count);
    w
}

fn tensor_with_type(wire_type: u32) -> Vec<u8> {
    let mut w = header(1, 0);
    w.string("weight").u32(1).u64(32).u32(wire_type).u64(0);
    w.finish()
}

fn array_header(element_type: u32, count: u64) -> Writer {
    let mut w = header(0, 1);
    w.string("values").u32(9).u32(element_type).u64(count);
    w
}

fn assert_invalid_data(bytes: Vec<u8>) {
    let error = GgufFile::from_reader(&mut Cursor::new(bytes))
        .err()
        .expect("malformed GGUF must be rejected");
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
}

#[test]
fn parses_all_thirteen_metadata_variants_from_literal_tags() {
    let mut w = header(0, 13);
    // 类型编号来自 GGUF 规范，不从生产代码反推，避免两端共同保留编号错误。
    w.string("u8").u32(0).u8(255);
    w.string("i8").u32(1).u8(0x80);
    w.string("u16").u32(2);
    w.buf.extend_from_slice(&65535_u16.to_le_bytes());
    w.string("i16").u32(3);
    w.buf.extend_from_slice(&(-12345_i16).to_le_bytes());
    w.string("u32").u32(4).u32(u32::MAX);
    w.string("i32").u32(5);
    w.buf.extend_from_slice(&(-123456_i32).to_le_bytes());
    w.string("f32").u32(6).f32(-1.25);
    w.string("bool").u32(7).u8(1);
    w.string("string").u32(8).string("模型\0λ");
    w.string("array").u32(9).u32(4).u64(2).u32(7).u32(11);
    w.string("u64").u32(10).u64(u64::MAX);
    w.string("i64").u32(11);
    w.buf.extend_from_slice(&i64::MIN.to_le_bytes());
    w.string("f64").u32(12);
    w.buf.extend_from_slice(&(-9.5_f64).to_le_bytes());

    let gguf = GgufFile::from_reader(&mut Cursor::new(w.finish())).expect("all metadata tags");
    let values = &gguf.metadata;
    assert_eq!(values.len(), 13);
    assert!(matches!(values["u8"], GgufValue::Uint8(255)));
    assert!(matches!(values["i8"], GgufValue::Int8(-128)));
    assert!(matches!(values["u16"], GgufValue::Uint16(65535)));
    assert!(matches!(values["i16"], GgufValue::Int16(-12345)));
    assert!(matches!(values["u32"], GgufValue::Uint32(u32::MAX)));
    assert!(matches!(values["i32"], GgufValue::Int32(-123456)));
    assert!(matches!(values["f32"], GgufValue::Float32(n) if n == -1.25));
    assert!(matches!(values["bool"], GgufValue::Bool(true)));
    assert!(matches!(&values["string"], GgufValue::String(s) if s == "模型\0λ"));
    assert!(matches!(&values["array"], GgufValue::Array(a)
        if matches!(a.as_slice(), [GgufValue::Uint32(7), GgufValue::Uint32(11)])));
    assert!(matches!(values["u64"], GgufValue::Uint64(u64::MAX)));
    assert!(matches!(values["i64"], GgufValue::Int64(i64::MIN)));
    assert!(matches!(values["f64"], GgufValue::Float64(n) if n == -9.5));
}

#[test]
fn parses_all_supported_ggml_wire_types() {
    // ggml.h 的 16..=23 是 IQ 类型；整数类型编号不能由 Rust 枚举顺序推断。
    let cases = [
        (0, GgmlType::F32),
        (1, GgmlType::F16),
        (2, GgmlType::Q4_0),
        (3, GgmlType::Q4_1),
        (6, GgmlType::Q5_0),
        (7, GgmlType::Q5_1),
        (8, GgmlType::Q8_0),
        (9, GgmlType::Q8_1),
        (10, GgmlType::Q2_K),
        (11, GgmlType::Q3_K),
        (12, GgmlType::Q4_K),
        (13, GgmlType::Q5_K),
        (14, GgmlType::Q6_K),
        (15, GgmlType::Q8_K),
        (24, GgmlType::I8),
        (25, GgmlType::I16),
        (26, GgmlType::I32),
        (27, GgmlType::I64),
        (28, GgmlType::F64),
    ];
    for (wire_type, expected) in cases {
        let gguf = GgufFile::from_reader(&mut Cursor::new(tensor_with_type(wire_type)))
            .unwrap_or_else(|error| panic!("wire type {wire_type}: {error}"));
        assert_eq!(gguf.tensor_infos[0].ggml_type, expected);
    }
}

#[test]
fn rejects_removed_unsupported_and_unknown_ggml_wire_types() {
    for wire_type in [
        4,
        5,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        29,
        30,
        31,
        42,
        43,
        u32::MAX,
    ] {
        assert_invalid_data(tensor_with_type(wire_type));
    }
}

#[test]
fn reports_element_size_and_quantization_for_every_public_ggml_variant() {
    let cases = [
        (GgmlType::F32, 4, false),
        (GgmlType::F16, 2, false),
        (GgmlType::Q4_0, 0, true),
        (GgmlType::Q4_1, 0, true),
        (GgmlType::Q5_0, 0, true),
        (GgmlType::Q5_1, 0, true),
        (GgmlType::Q8_0, 0, true),
        (GgmlType::Q8_1, 0, true),
        (GgmlType::Q2_K, 0, true),
        (GgmlType::Q3_K, 0, true),
        (GgmlType::Q4_K, 0, true),
        (GgmlType::Q5_K, 0, true),
        (GgmlType::Q6_K, 0, true),
        (GgmlType::Q8_K, 0, true),
        (GgmlType::I8, 1, false),
        (GgmlType::I16, 2, false),
        (GgmlType::I32, 4, false),
        (GgmlType::I64, 8, false),
        (GgmlType::F64, 8, false),
    ];
    for (data_type, size, quantized) in cases {
        assert_eq!(data_type.element_size(), size, "{data_type:?}");
        assert_eq!(data_type.is_quantized(), quantized, "{data_type:?}");
    }
}

#[test]
fn parses_zero_tensors_and_empty_metadata() {
    let gguf = GgufFile::from_reader(&mut Cursor::new(header(0, 0).finish())).expect("empty index");
    assert_eq!(gguf.tensor_count(), 0);
    assert_eq!(gguf.metadata_count(), 0);
    assert!(gguf.tensor_infos.is_empty());
    assert!(gguf.metadata.is_empty());
    assert!(gguf.find_tensor("missing").is_none());
    assert_eq!(gguf.tensor_data_offset, 32);
}

#[test]
fn accepts_empty_scalar_arrays() {
    for element_type in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12] {
        let gguf = GgufFile::from_reader(&mut Cursor::new(array_header(element_type, 0).finish()))
            .expect("empty array with a supported scalar type");
        assert!(matches!(&gguf.metadata["values"], GgufValue::Array(values) if values.is_empty()));
    }
}

#[test]
fn rejects_nonempty_nested_arrays() {
    let mut w = array_header(9, 1);
    w.u32(4).u64(1).u32(10);
    assert_invalid_data(w.finish());
}

#[test]
fn rejects_empty_nested_arrays() {
    assert_invalid_data(array_header(9, 0).finish());
}

#[test]
fn rejects_unknown_metadata_and_array_element_tags() {
    for tag in [13, u32::MAX] {
        let mut w = header(0, 1);
        w.string("unknown").u32(tag);
        assert_invalid_data(w.finish());
        assert_invalid_data(array_header(tag, 0).finish());
    }
}

#[test]
fn parses_boolean_array_values() {
    let mut w = array_header(7, 2);
    w.u8(0).u8(1);
    let gguf = GgufFile::from_reader(&mut Cursor::new(w.finish())).expect("boolean array");
    assert!(matches!(&gguf.metadata["values"], GgufValue::Array(values)
        if matches!(values.as_slice(), [GgufValue::Bool(false), GgufValue::Bool(true)])));
}

#[test]
fn rejects_noncanonical_booleans_in_scalars_and_arrays() {
    for value in [2, 255] {
        let mut w = header(0, 1);
        w.string("flag").u32(7).u8(value);
        assert_invalid_data(w.finish());
        let mut w = array_header(7, 1);
        w.u8(value);
        assert_invalid_data(w.finish());
    }
}

#[test]
fn honors_general_alignment_including_non_power_of_two() {
    for alignment in [8, 24, 32, 64, 256] {
        let mut w = header(0, 1);
        w.string("general.alignment").u32(4).u32(alignment);
        let position = w.buf.len() as u64;
        let expected = position.div_ceil(u64::from(alignment)) * u64::from(alignment);
        let gguf = GgufFile::from_reader(&mut Cursor::new(w.finish())).expect("valid alignment");
        assert_eq!(gguf.tensor_data_offset, expected, "alignment {alignment}");
    }
}

#[test]
fn rejects_zero_or_non_multiple_of_eight_alignment() {
    for alignment in [0, 1, 7, 12, 20] {
        let mut w = header(0, 1);
        w.string("general.alignment").u32(4).u32(alignment);
        assert_invalid_data(w.finish());
    }
}

#[test]
fn rejects_wrongly_typed_alignment() {
    let mut w = header(0, 1);
    w.string("general.alignment").u32(10).u64(64);
    assert_invalid_data(w.finish());
    let mut w = header(0, 1);
    w.string("general.alignment").u32(8).string("64");
    assert_invalid_data(w.finish());
}

// 用虚拟位置覆盖 u64 边界，不创建巨型文件，也不改变输入字节。
struct ReportedPositionReader {
    bytes: Cursor<Vec<u8>>,
    position: u64,
}

impl Read for ReportedPositionReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.bytes.read(buf)
    }
}

impl Seek for ReportedPositionReader {
    fn seek(&mut self, _position: SeekFrom) -> io::Result<u64> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "only position queries are supported",
        ))
    }

    fn stream_position(&mut self) -> io::Result<u64> {
        Ok(self.position)
    }
}

#[test]
fn rejects_tensor_data_alignment_overflow() {
    let mut reader = ReportedPositionReader {
        bytes: Cursor::new(header(0, 0).finish()),
        position: u64::MAX,
    };
    let error = GgufFile::from_reader(&mut reader)
        .err()
        .expect("alignment overflow");
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
}

#[test]
fn preserves_already_aligned_position_near_u64_maximum() {
    let mut w = header(0, 1);
    w.string("general.alignment").u32(4).u32(8);
    let mut reader = ReportedPositionReader {
        bytes: Cursor::new(w.finish()),
        position: u64::MAX - 7,
    };
    let gguf = GgufFile::from_reader(&mut reader).expect("no padding is required");
    assert_eq!(gguf.tensor_data_offset, u64::MAX - 7);
}

#[test]
fn accepts_string_at_64_mib_limit() {
    let limit = 64 * 1024 * 1024;
    let mut w = header(0, 1);
    w.string("text").u32(8).u64(limit as u64);
    w.buf.resize(w.buf.len() + limit, b'x');
    let gguf = GgufFile::from_reader(&mut Cursor::new(w.finish())).expect("inclusive string limit");
    assert!(matches!(&gguf.metadata["text"], GgufValue::String(text)
        if text.len() == limit && text.bytes().all(|byte| byte == b'x')));
}

#[test]
fn rejects_strings_above_64_mib_limit() {
    for length in [64 * 1024 * 1024 + 1, u64::MAX] {
        let mut w = header(0, 1);
        w.string("text").u32(8).u64(length);
        assert_invalid_data(w.finish());
    }
}

#[test]
fn applies_string_bound_to_metadata_keys_tensor_names_and_array_strings() {
    let length = 64 * 1024 * 1024 + 1;
    let mut key = header(0, 1);
    key.u64(length);
    assert_invalid_data(key.finish());
    let mut tensor = header(1, 0);
    tensor.u64(length);
    assert_invalid_data(tensor.finish());
    let mut array = array_header(8, 1);
    array.u64(length);
    assert_invalid_data(array.finish());
}

#[test]
fn rejects_invalid_utf8_strings() {
    let mut w = header(0, 1);
    w.string("text").u32(8).u64(1).u8(0xff);
    assert_invalid_data(w.finish());
}

#[test]
fn rejects_huge_array_count_without_panicking() {
    let result = GgufFile::from_reader(&mut Cursor::new(array_header(0, u64::MAX).finish()));
    assert!(result.is_err());
}

#[test]
fn rejects_huge_tensor_count_without_panicking() {
    let result = GgufFile::from_reader(&mut Cursor::new(header(u64::MAX, 0).finish()));
    assert!(result.is_err());
}

#[test]
fn rejects_huge_metadata_count_without_panicking() {
    let result = GgufFile::from_reader(&mut Cursor::new(header(0, u64::MAX).finish()));
    assert!(result.is_err());
}

#[test]
fn rejects_dimension_counts_above_ggml_limit() {
    for count in [5, u32::MAX] {
        let mut w = header(1, 0);
        w.string("weight").u32(count);
        // 首个案例提供完整记录，确保拒绝原因是维度上限而非 EOF。
        if count == 5 {
            for _ in 0..count {
                w.u64(1);
            }
            w.u32(0).u64(0);
        }
        assert_invalid_data(w.finish());
    }
}

#[test]
fn parses_four_tensor_dimensions_and_preserves_relative_offset() {
    let mut w = header(1, 0);
    w.string("weight")
        .u32(4)
        .u64(32)
        .u64(4)
        .u64(2)
        .u64(1)
        .u32(2)
        .u64(96);
    let gguf =
        GgufFile::from_reader(&mut Cursor::new(w.finish())).expect("four-dimensional tensor");
    let tensor = &gguf.tensor_infos[0];
    assert_eq!(tensor.n_dims, 4);
    assert_eq!(tensor.dims, [32, 4, 2, 1]);
    assert_eq!(tensor.offset, 96);
}

mod parser_boundaries {
    use super::*;

    fn header(tensor_count: u64, metadata_count: u64) -> Writer {
        let mut writer = Writer::new();
        writer.buf.extend_from_slice(b"GGUF");
        writer.u32(3).u64(tensor_count).u64(metadata_count);
        writer
    }

    fn single_value(tag: u32, payload: &[u8]) -> io::Result<GgufFile> {
        let mut writer = header(0, 1);
        writer.string("test.value").u32(tag);
        writer.buf.extend_from_slice(payload);
        GgufFile::from_reader(&mut Cursor::new(writer.finish()))
    }

    fn assert_invalid_data(result: io::Result<GgufFile>) {
        match result {
            Err(error) => assert_eq!(error.kind(), io::ErrorKind::InvalidData, "{error}"),
            Ok(_) => panic!("invalid GGUF input was accepted"),
        }
    }

    fn assert_value_eq(actual: &GgufValue, expected: &GgufValue) {
        match (actual, expected) {
            (GgufValue::Uint8(a), GgufValue::Uint8(b)) => assert_eq!(a, b),
            (GgufValue::Int8(a), GgufValue::Int8(b)) => assert_eq!(a, b),
            (GgufValue::Uint16(a), GgufValue::Uint16(b)) => assert_eq!(a, b),
            (GgufValue::Int16(a), GgufValue::Int16(b)) => assert_eq!(a, b),
            (GgufValue::Uint32(a), GgufValue::Uint32(b)) => assert_eq!(a, b),
            (GgufValue::Int32(a), GgufValue::Int32(b)) => assert_eq!(a, b),
            (GgufValue::Float32(a), GgufValue::Float32(b)) => assert_eq!(a, b),
            (GgufValue::Bool(a), GgufValue::Bool(b)) => assert_eq!(a, b),
            (GgufValue::String(a), GgufValue::String(b)) => assert_eq!(a, b),
            (GgufValue::Uint64(a), GgufValue::Uint64(b)) => assert_eq!(a, b),
            (GgufValue::Int64(a), GgufValue::Int64(b)) => assert_eq!(a, b),
            (GgufValue::Float64(a), GgufValue::Float64(b)) => assert_eq!(a, b),
            (GgufValue::Array(a), GgufValue::Array(b)) => {
                assert_eq!(a.len(), b.len());
                for (actual, expected) in a.iter().zip(b) {
                    assert_value_eq(actual, expected);
                }
            }
            _ => panic!("unexpected metadata value: {actual:?}, expected {expected:?}"),
        }
    }

    fn scalar_encodings() -> Vec<(u32, Vec<u8>, GgufValue)> {
        let mut string = Writer::new();
        string.string("模型\0token");
        // 编号直接来自 GGUF 规范，不从生产解析器或枚举反推。
        vec![
            (0, vec![255], GgufValue::Uint8(255)),
            (1, vec![128], GgufValue::Int8(-128)),
            (
                2,
                60_000u16.to_le_bytes().to_vec(),
                GgufValue::Uint16(60_000),
            ),
            (
                3,
                (-30_000i16).to_le_bytes().to_vec(),
                GgufValue::Int16(-30_000),
            ),
            (
                4,
                u32::MAX.to_le_bytes().to_vec(),
                GgufValue::Uint32(u32::MAX),
            ),
            (
                5,
                i32::MIN.to_le_bytes().to_vec(),
                GgufValue::Int32(i32::MIN),
            ),
            (
                6,
                (-1.25f32).to_le_bytes().to_vec(),
                GgufValue::Float32(-1.25),
            ),
            (7, vec![1], GgufValue::Bool(true)),
            (
                8,
                string.finish(),
                GgufValue::String("模型\0token".to_string()),
            ),
            (
                10,
                u64::MAX.to_le_bytes().to_vec(),
                GgufValue::Uint64(u64::MAX),
            ),
            (
                11,
                i64::MIN.to_le_bytes().to_vec(),
                GgufValue::Int64(i64::MIN),
            ),
            (
                12,
                1.125f64.to_le_bytes().to_vec(),
                GgufValue::Float64(1.125),
            ),
        ]
    }

    #[test]
    fn parses_all_thirteen_metadata_variants_with_literal_tags() {
        let mut cases = scalar_encodings();
        let mut array = Writer::new();
        array.u32(4).u64(2).u32(123).u32(456);
        cases.push((
            9,
            array.finish(),
            GgufValue::Array(vec![GgufValue::Uint32(123), GgufValue::Uint32(456)]),
        ));

        for (tag, payload, expected) in cases {
            let gguf = single_value(tag, &payload).expect("valid metadata type");
            assert_value_eq(&gguf.metadata["test.value"], &expected);
        }
    }

    #[test]
    fn parses_arrays_of_every_supported_scalar_type() {
        for (tag, payload, expected) in scalar_encodings() {
            let mut array = Writer::new();
            array.u32(tag).u64(2);
            array.buf.extend_from_slice(&payload);
            array.buf.extend_from_slice(&payload);
            let gguf = single_value(9, &array.finish()).expect("valid array element type");
            assert_value_eq(
                &gguf.metadata["test.value"],
                &GgufValue::Array(vec![expected.clone(), expected]),
            );
        }
    }

    #[test]
    fn parses_empty_arrays_of_every_supported_scalar_type() {
        for tag in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12] {
            let mut array = Writer::new();
            array.u32(tag).u64(0);
            let gguf = single_value(9, &array.finish()).expect("empty scalar array");
            assert!(
                matches!(&gguf.metadata["test.value"], GgufValue::Array(values) if values.is_empty())
            );
        }
    }

    #[test]
    fn rejects_nested_arrays_even_when_empty() {
        for count in [0, 1] {
            let mut array = Writer::new();
            array.u32(9).u64(count);
            if count != 0 {
                array.u32(4).u64(0);
            }
            assert_invalid_data(single_value(9, &array.finish()));
        }
    }

    #[test]
    fn rejects_unknown_metadata_tags() {
        for tag in [13, 99, u32::MAX] {
            assert_invalid_data(single_value(tag, &[]));
        }
    }

    #[test]
    fn rejects_unknown_array_tags_even_when_empty() {
        for tag in [13, 99, u32::MAX] {
            for count in [0, 1] {
                let mut array = Writer::new();
                array.u32(tag).u64(count);
                assert_invalid_data(single_value(9, &array.finish()));
            }
        }
    }

    #[test]
    fn parses_boolean_false() {
        let gguf = single_value(7, &[0]).expect("false boolean");
        assert!(matches!(
            gguf.metadata["test.value"],
            GgufValue::Bool(false)
        ));
    }

    #[test]
    fn rejects_non_boolean_bytes_in_scalars_and_arrays() {
        for byte in [2, 255] {
            assert_invalid_data(single_value(7, &[byte]));
            let mut array = Writer::new();
            array.u32(7).u64(1).u8(byte);
            assert_invalid_data(single_value(9, &array.finish()));
        }
    }

    fn tensor_file(wire_type: u32) -> Vec<u8> {
        let mut writer = header(1, 0);
        writer
            .string("weight")
            .u32(2)
            .u64(256)
            .u64(4)
            .u32(wire_type)
            .u64(0);
        writer.finish()
    }

    #[test]
    fn parses_all_supported_ggml_wire_ids() {
        // ggml.h 保留了已删除的 4/5；16..23 是 IQ 系列，整数类型从 24 开始。
        let cases = [
            (0, GgmlType::F32),
            (1, GgmlType::F16),
            (2, GgmlType::Q4_0),
            (3, GgmlType::Q4_1),
            (6, GgmlType::Q5_0),
            (7, GgmlType::Q5_1),
            (8, GgmlType::Q8_0),
            (9, GgmlType::Q8_1),
            (10, GgmlType::Q2_K),
            (11, GgmlType::Q3_K),
            (12, GgmlType::Q4_K),
            (13, GgmlType::Q5_K),
            (14, GgmlType::Q6_K),
            (15, GgmlType::Q8_K),
            (24, GgmlType::I8),
            (25, GgmlType::I16),
            (26, GgmlType::I32),
            (27, GgmlType::I64),
            (28, GgmlType::F64),
        ];
        for (wire_id, expected) in cases {
            let gguf = GgufFile::from_reader(&mut Cursor::new(tensor_file(wire_id)))
                .expect("supported GGML wire ID");
            assert_eq!(
                gguf.tensor_infos[0].ggml_type, expected,
                "wire ID {wire_id}"
            );
        }
    }

    #[test]
    fn rejects_retired_unsupported_and_unknown_ggml_ids() {
        for wire_id in [
            4,
            5,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            29,
            30,
            31,
            42,
            43,
            u32::MAX,
        ] {
            assert_invalid_data(GgufFile::from_reader(&mut Cursor::new(tensor_file(
                wire_id,
            ))));
        }
    }

    #[test]
    fn ggml_element_size_and_quantization_cover_every_public_variant() {
        let cases = [
            (GgmlType::F32, 4, false),
            (GgmlType::F16, 2, false),
            (GgmlType::Q4_0, 0, true),
            (GgmlType::Q4_1, 0, true),
            (GgmlType::Q5_0, 0, true),
            (GgmlType::Q5_1, 0, true),
            (GgmlType::Q8_0, 0, true),
            (GgmlType::Q8_1, 0, true),
            (GgmlType::Q2_K, 0, true),
            (GgmlType::Q3_K, 0, true),
            (GgmlType::Q4_K, 0, true),
            (GgmlType::Q5_K, 0, true),
            (GgmlType::Q6_K, 0, true),
            (GgmlType::Q8_K, 0, true),
            (GgmlType::I8, 1, false),
            (GgmlType::I16, 2, false),
            (GgmlType::I32, 4, false),
            (GgmlType::I64, 8, false),
            (GgmlType::F64, 8, false),
        ];
        for (ggml_type, size, quantized) in cases {
            assert_eq!(ggml_type.element_size(), size, "{ggml_type:?}");
            assert_eq!(ggml_type.is_quantized(), quantized, "{ggml_type:?}");
        }
    }

    #[test]
    fn parses_zero_tensors_and_empty_metadata() {
        let gguf = GgufFile::from_reader(&mut Cursor::new(header(0, 0).finish()))
            .expect("empty GGUF index");
        assert_eq!(gguf.tensor_count(), 0);
        assert_eq!(gguf.metadata_count(), 0);
        assert!(gguf.tensor_infos.is_empty());
        assert!(gguf.metadata.is_empty());
        assert_eq!(gguf.tensor_data_offset, 32);
    }

    #[test]
    fn parses_tensor_index_with_empty_metadata() {
        let gguf = GgufFile::from_reader(&mut Cursor::new(tensor_file(0)))
            .expect("metadata is optional for an index");
        assert!(gguf.metadata.is_empty());
        assert_eq!(gguf.tensor_infos[0].dims, [256, 4]);
    }

    fn alignment_file(alignment: u32) -> Vec<u8> {
        let mut writer = header(0, 1);
        writer.string("general.alignment").u32(4).u32(alignment);
        writer.finish()
    }

    #[test]
    fn uses_general_alignment_including_non_power_of_two_multiples_of_eight() {
        for (alignment, expected_offset) in
            [(8, 64), (24, 72), (64, 64), (128, 128), (1 << 31, 1 << 31)]
        {
            let gguf = GgufFile::from_reader(&mut Cursor::new(alignment_file(alignment)))
                .expect("valid alignment");
            assert_eq!(
                gguf.tensor_data_offset, expected_offset,
                "alignment {alignment}"
            );
        }
    }

    #[test]
    fn rejects_zero_and_non_multiple_of_eight_alignments() {
        for alignment in [0, 1, 3, 12, u32::MAX] {
            assert_invalid_data(GgufFile::from_reader(&mut Cursor::new(alignment_file(
                alignment,
            ))));
        }
    }

    #[test]
    fn rejects_general_alignment_with_wrong_metadata_type() {
        for (tag, payload) in [
            (0, vec![32]),
            (5, 32i32.to_le_bytes().to_vec()),
            (10, 32u64.to_le_bytes().to_vec()),
        ] {
            let mut writer = header(0, 1);
            writer.string("general.alignment").u32(tag);
            writer.buf.extend_from_slice(&payload);
            assert_invalid_data(GgufFile::from_reader(&mut Cursor::new(writer.finish())));
        }
    }

    // 用虚拟起点覆盖 u64 边界，不构造同等大小的文件。
    struct OffsetReader {
        data: Cursor<Vec<u8>>,
        origin: u64,
    }

    impl Read for OffsetReader {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            self.data.read(buf)
        }
    }

    impl Seek for OffsetReader {
        fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
            match position {
                SeekFrom::Current(0) => {
                    self.origin
                        .checked_add(self.data.position())
                        .ok_or_else(|| {
                            io::Error::new(io::ErrorKind::InvalidInput, "virtual position overflow")
                        })
                }
                _ => Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "only position queries are supported",
                )),
            }
        }
    }

    #[test]
    fn rejects_alignment_offset_overflow_without_panicking() {
        let bytes = header(0, 0).finish();
        let origin = u64::MAX - bytes.len() as u64;
        assert_invalid_data(GgufFile::from_reader(&mut OffsetReader {
            data: Cursor::new(bytes),
            origin,
        }));
    }

    #[test]
    fn accepts_already_aligned_position_near_u64_max() {
        let bytes = alignment_file(24);
        let position = u64::MAX - u64::MAX % 24;
        let origin = position - bytes.len() as u64;
        let gguf = GgufFile::from_reader(&mut OffsetReader {
            data: Cursor::new(bytes),
            origin,
        })
        .expect("already aligned position is representable");
        assert_eq!(gguf.tensor_data_offset, position);
    }

    #[test]
    fn rejects_strings_over_64_mib_before_reading_payload() {
        for length in [64 * 1024 * 1024 + 1, u32::MAX as u64 + 1, u64::MAX] {
            assert_invalid_data(single_value(8, &length.to_le_bytes()));
        }
    }

    #[test]
    fn accepts_string_at_64_mib_limit() {
        let length = 64 * 1024 * 1024;
        let mut writer = header(0, 1);
        writer.string("large").u32(8).u64(length as u64);
        writer.buf.resize(writer.buf.len() + length, b'x');
        let gguf = GgufFile::from_reader(&mut Cursor::new(writer.finish()))
            .expect("inclusive string limit");
        match &gguf.metadata["large"] {
            GgufValue::String(value) => {
                assert_eq!(value.len(), length);
                assert!(value.bytes().all(|byte| byte == b'x'));
            }
            other => panic!("unexpected string value: {other:?}"),
        }
    }

    #[test]
    fn rejects_invalid_utf8_in_strings() {
        let mut payload = Writer::new();
        payload.u64(2).u8(0xc3).u8(0x28);
        assert_invalid_data(single_value(8, &payload.finish()));
    }

    #[test]
    fn rejects_truncated_scalar_array_and_string_payloads() {
        for (tag, payload, _) in scalar_encodings() {
            assert!(!payload.is_empty());
            assert!(
                single_value(tag, &payload[..payload.len() - 1]).is_err(),
                "tag {tag}"
            );
        }
        let mut array = Writer::new();
        array.u32(4).u64(2).u32(10);
        assert!(single_value(9, &array.finish()).is_err());
    }

    #[test]
    fn handles_untrusted_array_count_without_eager_allocation() {
        let mut array = Writer::new();
        array.u32(0).u64(u64::MAX);
        assert!(single_value(9, &array.finish()).is_err());
    }

    #[test]
    fn handles_untrusted_tensor_count_without_eager_allocation() {
        assert!(GgufFile::from_reader(&mut Cursor::new(header(u64::MAX, 0).finish())).is_err());
    }

    #[test]
    fn handles_untrusted_metadata_count_without_eager_allocation() {
        assert!(GgufFile::from_reader(&mut Cursor::new(header(0, u64::MAX).finish())).is_err());
    }

    #[test]
    fn rejects_tensor_dimension_count_above_ggml_limit() {
        let mut writer = header(1, 0);
        writer.string("weight").u32(5);
        assert_invalid_data(GgufFile::from_reader(&mut Cursor::new(writer.finish())));
    }
}

mod model_metadata {
    use super::*;
    use qllama::gguf::ModelMetadata;
    use std::collections::HashMap;

    fn llama_metadata() -> HashMap<String, GgufValue> {
        let mut writer = header(0, 11);
        writer.string("general.architecture").u32(8).string("llama");
        writer.string("general.name").u32(8).string("Tiny Llama");
        writer.string("llama.context_length").u32(4).u32(4096);
        writer.string("llama.embedding_length").u32(10).u64(512);
        writer.string("llama.block_count").u32(4).u32(8);
        writer.string("llama.attention.head_count").u32(4).u32(8);
        writer.string("llama.attention.head_count_kv").u32(4).u32(2);
        writer
            .string("llama.attention.layer_norm_rms_epsilon")
            .u32(6)
            .f32(2e-5);
        writer.string("llama.rope.freq_base").u32(6).f32(500_000.0);
        writer.string("tokenizer.ggml.model").u32(8).string("llama");
        writer.string("tokenizer.ggml.tokens").u32(9).u32(8).u64(3);
        writer.string("<unk>").string("<s>").string("</s>");
        GgufFile::from_reader(&mut Cursor::new(writer.finish()))
            .expect("realistic metadata wire keys")
            .metadata
    }

    #[test]
    fn extracts_realistic_llama_metadata_from_gguf() {
        let metadata =
            ModelMetadata::from_gguf_metadata(&llama_metadata()).expect("llama metadata");
        assert_eq!(metadata.architecture, "llama");
        assert_eq!(metadata.name, "Tiny Llama");
        assert_eq!(metadata.context_length, 4096);
        assert_eq!(metadata.embedding_length, 512);
        assert_eq!(metadata.block_count, 8);
        assert_eq!(metadata.attention_head_count, 8);
        assert_eq!(metadata.attention_head_count_kv, 2);
        assert_eq!(metadata.layer_norm_eps, 2e-5);
        assert_eq!(metadata.rope_freq_base, 500_000.0);
        assert_eq!(metadata.vocab_size, 3);
    }

    #[test]
    fn applies_defaults_only_for_absent_optional_fields() {
        let mut values = llama_metadata();
        for key in [
            "general.name",
            "llama.attention.head_count_kv",
            "llama.rope.freq_base",
        ] {
            values.remove(key);
        }
        let metadata =
            ModelMetadata::from_gguf_metadata(&values).expect("optional fields may be absent");
        assert_eq!(metadata.name, "unknown");
        assert_eq!(metadata.attention_head_count_kv, 8);
        assert_eq!(metadata.layer_norm_eps, 2e-5);
        assert_eq!(metadata.rope_freq_base, 10_000.0);
    }

    #[test]
    fn rejects_missing_or_mistyped_required_fields() {
        for key in [
            "general.architecture",
            "llama.context_length",
            "llama.embedding_length",
            "llama.block_count",
            "llama.attention.head_count",
            "llama.attention.layer_norm_rms_epsilon",
            "tokenizer.ggml.tokens",
        ] {
            let mut values = llama_metadata();
            values.remove(key);
            assert!(
                ModelMetadata::from_gguf_metadata(&values).is_err(),
                "missing {key}"
            );
            values.insert(key.to_string(), GgufValue::Bool(true));
            assert!(
                ModelMetadata::from_gguf_metadata(&values).is_err(),
                "mistyped {key}"
            );
        }
    }

    #[test]
    fn rejects_mistyped_optional_fields_instead_of_masking_them_with_defaults() {
        for key in [
            "general.name",
            "llama.attention.head_count_kv",
            "llama.rope.freq_base",
        ] {
            let mut values = llama_metadata();
            values.insert(key.to_string(), GgufValue::Bool(true));
            assert!(
                ModelMetadata::from_gguf_metadata(&values).is_err(),
                "mistyped {key}"
            );
        }
    }

    #[test]
    fn rejects_negative_and_zero_model_dimensions() {
        for key in [
            "llama.context_length",
            "llama.embedding_length",
            "llama.block_count",
            "llama.attention.head_count",
            "llama.attention.head_count_kv",
        ] {
            for value in [
                GgufValue::Int32(-1),
                GgufValue::Int64(i64::MIN),
                GgufValue::Uint64(0),
            ] {
                let mut values = llama_metadata();
                values.insert(key.to_string(), value);
                assert!(
                    ModelMetadata::from_gguf_metadata(&values).is_err(),
                    "invalid {key}"
                );
            }
        }
    }

    #[test]
    fn preserves_positive_signed_integer_metadata_compatibility() {
        let mut values = llama_metadata();
        values.insert("llama.context_length".to_string(), GgufValue::Int64(4096));
        values.insert("llama.block_count".to_string(), GgufValue::Int32(8));
        let metadata =
            ModelMetadata::from_gguf_metadata(&values).expect("positive signed integer dimensions");
        assert_eq!(metadata.context_length, 4096);
        assert_eq!(metadata.block_count, 8);
    }

    #[test]
    fn checks_u64_dimensions_against_target_usize_range() {
        for key in [
            "llama.context_length",
            "llama.embedding_length",
            "llama.block_count",
            "llama.attention.head_count",
            "llama.attention.head_count_kv",
        ] {
            let mut values = llama_metadata();
            values.insert(key.to_string(), GgufValue::Uint64(u32::MAX as u64 + 1));
            let result = ModelMetadata::from_gguf_metadata(&values);
            if usize::BITS < 64 {
                assert!(result.is_err(), "out-of-range {key}");
            } else {
                let metadata = result.expect("representable 64-bit dimension");
                let actual = match key {
                    "llama.context_length" => metadata.context_length,
                    "llama.embedding_length" => metadata.embedding_length,
                    "llama.block_count" => metadata.block_count,
                    "llama.attention.head_count" => metadata.attention_head_count,
                    "llama.attention.head_count_kv" => metadata.attention_head_count_kv,
                    _ => unreachable!(),
                };
                assert_eq!(actual as u64, u32::MAX as u64 + 1);
            }
        }
    }

    #[test]
    fn rejects_nonpositive_nonfinite_and_overflowing_float_parameters() {
        for key in [
            "llama.attention.layer_norm_rms_epsilon",
            "llama.rope.freq_base",
        ] {
            for value in [
                GgufValue::Float32(0.0),
                GgufValue::Float32(-1.0),
                GgufValue::Float32(f32::NAN),
                GgufValue::Float32(f32::INFINITY),
                GgufValue::Float64(f64::MAX),
            ] {
                let mut values = llama_metadata();
                values.insert(key.to_string(), value);
                assert!(
                    ModelMetadata::from_gguf_metadata(&values).is_err(),
                    "invalid {key}"
                );
            }
        }
    }

    #[test]
    fn accepts_representable_float64_parameters() {
        let mut values = llama_metadata();
        values.insert(
            "llama.attention.layer_norm_rms_epsilon".to_string(),
            GgufValue::Float64(0.00002),
        );
        values.insert(
            "llama.rope.freq_base".to_string(),
            GgufValue::Float64(500_000.0),
        );
        let metadata =
            ModelMetadata::from_gguf_metadata(&values).expect("representable float parameters");
        assert_eq!(metadata.layer_norm_eps, 2e-5);
        assert_eq!(metadata.rope_freq_base, 500_000.0);
    }

    #[test]
    fn rejects_empty_or_non_string_vocabulary() {
        for tokens in [
            vec![],
            vec![GgufValue::Uint32(1)],
            vec![
                GgufValue::String("token".to_string()),
                GgufValue::Bool(false),
            ],
        ] {
            let mut values = llama_metadata();
            values.insert(
                "tokenizer.ggml.tokens".to_string(),
                GgufValue::Array(tokens),
            );
            assert!(ModelMetadata::from_gguf_metadata(&values).is_err());
        }
    }
}

mod format_validation {
    use super::*;

    /// 构造指定版本号、无 metadata、无张量的最小文件。
    fn with_version(version: u32) -> Vec<u8> {
        let mut w = Writer::new();
        w.buf.extend_from_slice(b"GGUF");
        w.u32(version).u64(0).u64(0);
        w.finish()
    }

    fn with_keys(keys: &[&str]) -> Vec<u8> {
        let mut w = Writer::new();
        w.buf.extend_from_slice(b"GGUF");
        w.u32(3).u64(0).u64(keys.len() as u64);
        for key in keys {
            w.string(key).u32(4).u32(1);
        }
        w.finish()
    }

    /// 构造若干张量记录，每个名称按给定长度用同一字节填充。
    ///
    /// 这样 `[4, 4]` 产生两个完全同名的张量，用于复现重名场景。
    fn with_tensor_names(names: &[usize]) -> Vec<u8> {
        let mut w = Writer::new();
        w.buf.extend_from_slice(b"GGUF");
        w.u32(3).u64(names.len() as u64).u64(0);
        for &name_len in names {
            w.u64(name_len as u64);
            w.buf.resize(w.buf.len() + name_len, b'a');
            w.u32(1).u64(4).u32(0).u64(0);
        }
        w.finish()
    }

    /// 构造两个名称不同的张量记录，用于确认重名检查不会误拒正常文件。
    fn with_distinct_tensor_names() -> Vec<u8> {
        let mut w = Writer::new();
        w.buf.extend_from_slice(b"GGUF");
        w.u32(3).u64(2).u64(0);
        for name in ["aaaa", "bbbb"] {
            w.string(name).u32(1).u64(4).u32(0).u64(0);
        }
        w.finish()
    }

    #[test]
    fn accepts_gguf_versions_up_to_the_supported_version() {
        for version in [2, 3] {
            let gguf = GgufFile::from_reader(&mut Cursor::new(with_version(version)))
                .unwrap_or_else(|error| panic!("version {version}: {error}"));
            assert_eq!(gguf.header.version, version);
        }
    }

    #[test]
    fn rejects_unsupported_gguf_versions() {
        // 0x30000000 与 0x10000 的低 16 位为 0，上游据此识别大端文件；按小端解析会整体错位。
        for version in [0, 1, 4, 0x0001_0000, 0x3000_0000, u32::MAX] {
            assert_invalid_data(with_version(version));
        }
    }

    #[test]
    fn rejects_duplicate_metadata_keys() {
        // 同名键会被 HashMap 静默覆盖，导致 metadata_count 与实际条目数不一致。
        assert_invalid_data(with_keys(&["general.name", "general.name"]));
    }

    #[test]
    fn accepts_distinct_metadata_keys() {
        let gguf = GgufFile::from_reader(&mut Cursor::new(with_keys(&[
            "general.name",
            "general.version",
        ])))
        .expect("distinct keys");
        assert_eq!(gguf.metadata.len(), 2);
        assert_eq!(gguf.metadata_count(), 2);
    }

    #[test]
    fn rejects_empty_metadata_key() {
        assert_invalid_data(with_keys(&[""]));
    }

    #[test]
    fn accepts_metadata_key_at_the_specification_limit() {
        let key = "k".repeat(65535);
        let gguf = GgufFile::from_reader(&mut Cursor::new(with_keys(&[&key])))
            .expect("key at the 65535-byte limit");
        assert!(gguf.metadata.contains_key(&key));
    }

    #[test]
    fn rejects_metadata_key_above_the_specification_limit() {
        let key = "k".repeat(65536);
        assert_invalid_data(with_keys(&[&key]));
    }

    #[test]
    fn accepts_tensor_name_below_the_ggml_limit() {
        let gguf = GgufFile::from_reader(&mut Cursor::new(with_tensor_names(&[63])))
            .expect("63-byte tensor name is valid");
        assert_eq!(gguf.tensor_infos.len(), 1);
    }

    #[test]
    fn rejects_tensor_name_at_or_above_the_ggml_limit() {
        // ggml.h 的 GGML_MAX_NAME 为 64，上游用 >= 判定，因此 64 字节已超限。
        for name_len in [64, 65] {
            assert_invalid_data(with_tensor_names(&[name_len]));
        }
    }

    #[test]
    fn rejects_duplicate_tensor_names() {
        // 重名会让 find_tensor 只能看到第一个 offset，第二个权重无法通过公开 API 定位。
        assert_invalid_data(with_tensor_names(&[4, 4]));
    }

    #[test]
    fn accepts_distinct_tensor_names() {
        let gguf = GgufFile::from_reader(&mut Cursor::new(with_distinct_tensor_names()))
            .expect("distinct names");
        assert_eq!(gguf.tensor_infos.len(), 2);
        assert_eq!(gguf.tensor_infos[0].name, "aaaa");
        assert_eq!(gguf.tensor_infos[1].name, "bbbb");
    }
}
