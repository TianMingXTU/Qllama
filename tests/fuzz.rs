//! 模糊测试（Phase 3）：GGUF 解析器与分词器对任意/畸形输入的健壮性。
//!
//! 核心不变量：**解析器可以拒绝输入（返回 Err），但绝不能 panic**——
//! 服务进程加载模型时面对的是不可信文件，panic 即 DoS。
//!
//! 用 `#![cfg(feature = "server")]`？不：本文件只依赖核心库，任何 feature
//! 下都应通过，因此不设门控；默认 `cargo test` 即覆盖。

use std::io::Cursor;

use proptest::prelude::*;

use qllama::gguf::GgufFile;
use qllama::tokenizer::Tokenizer;

/// 最小 GGUF 字节流写入器（与 tests/tokenizer.rs 的 Writer 同构，独立
/// 维护以免测试间耦合；只覆盖 fuzz 用到的写入路径）。
struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }
    fn u32(&mut self, value: u32) -> &mut Self {
        self.buf.extend_from_slice(&value.to_le_bytes());
        self
    }
    fn u64(&mut self, value: u64) -> &mut Self {
        self.buf.extend_from_slice(&value.to_le_bytes());
        self
    }
    fn string(&mut self, value: &str) -> &mut Self {
        self.u64(value.len() as u64);
        self.buf.extend_from_slice(value.as_bytes());
        self
    }
}

/// 写一个带类型标签的字符串元数据项。
fn write_string_entry(w: &mut Writer, key: &str, value: &str) {
    w.string(key);
    w.u32(8); // GGUFv2 元数据类型：STRING
    w.string(value);
}

/// 构造一个结构合法（头 + 两个字符串元数据）但内容可控的 GGUF，
/// `tail` 追加在合法结构之后，用于测解析器对后续垃圾的容忍度。
fn structured_gguf(tail: &[u8]) -> Vec<u8> {
    let mut w = Writer::new();
    w.buf.extend_from_slice(b"GGUF");
    w.u32(2); // version
    w.u64(0); // tensor_count
    w.u64(2); // metadata_count
    write_string_entry(&mut w, "general.architecture", "llama");
    write_string_entry(&mut w, "general.name", "fuzz");
    w.buf.extend_from_slice(tail);
    w.buf
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// 任意字节喂给解析器：Ok 或 Err 都行，panic 即失败。
    #[test]
    fn gguf_parser_never_panics_on_arbitrary_bytes(data in proptest::collection::vec(any::<u8>(), 0..4096)) {
        let mut cursor = Cursor::new(data);
        let _ = GgufFile::from_reader(&mut cursor);
    }

    /// 结构合法的文件头 + 任意尾巴：解析器必须干净拒绝或干净接受，
    /// 不允许 panic / 死循环（proptest 默认 512 case，尾部长度受控）。
    #[test]
    fn gguf_parser_never_panics_on_valid_header_plus_garbage(
        tail in proptest::collection::vec(any::<u8>(), 0..2048),
    ) {
        let data = structured_gguf(&tail);
        let mut cursor = Cursor::new(data);
        let _ = GgufFile::from_reader(&mut cursor);
    }

    /// 截断真实结构：合法元数据逐字节截断，解析器要么报错要么给出
    /// 截断前已完整解析的部分，不允许 panic。
    #[test]
    fn gguf_parser_never_panics_on_truncated_valid_file(
        cut in 0usize..160,
    ) {
        let full = structured_gguf(&[]);
        let data = full[..cut.min(full.len())].to_vec();
        let mut cursor = Cursor::new(data);
        let _ = GgufFile::from_reader(&mut cursor);
    }
}

// ---- 分词器 fuzz：固定小词表 + 任意文本 ----

/// 与 tests/tokenizer.rs 相同思路的最小词表（SPM 风格，含 byte fallback）。
const VOCAB: [&str; 8] = ["<s>", "</s>", "a", "b", "ab", "▁", "▁a", "<0x41>"];

fn tiny_tokenizer() -> Tokenizer {
    let mut w = Writer::new();
    w.buf.extend_from_slice(b"GGUF");
    w.u32(2);
    w.u64(0); // tensor_count
    w.u64(5); // metadata_count
    write_string_entry(&mut w, "general.architecture", "llama");
    w.string("tokenizer.ggml.model");
    w.u32(8);
    w.string("gpt2");
    w.string("tokenizer.ggml.tokens");
    w.u32(9); // ARRAY
    w.u32(8); // 元素类型 STRING
    w.u64(VOCAB.len() as u64);
    for token in VOCAB {
        w.string(token);
    }
    w.string("tokenizer.ggml.bos_token_id");
    w.u32(4); // UINT32
    w.u32(0);
    w.string("tokenizer.ggml.eos_token_id");
    w.u32(4);
    w.u32(1);
    while !w.buf.len().is_multiple_of(8) {
        w.buf.push(0);
    }
    let gguf = GgufFile::from_reader(&mut Cursor::new(w.buf.clone())).expect("fixture parses");
    Tokenizer::from_gguf(&gguf).expect("tokenizer loads")
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// 任意 Unicode 文本编码：不 panic；产出的每个 id 都在词表范围内。
    #[test]
    fn tokenizer_encode_arbitrary_text_ids_in_range(text in ".*") {
        let tokenizer = tiny_tokenizer();
        if let Ok(ids) = tokenizer.encode(&text) {
            prop_assert!(ids.iter().all(|&id| id >= 0 && (id as usize) < VOCAB.len()));
        }
    }

    /// 逐 id 流式解码：不 panic；拼接结果（含 finish 冲刷）是合法 String
    ///（类型上保证 UTF-8，这里验证不会 panic 且长度不爆炸）。
    #[test]
    fn tokenizer_stream_decode_never_panics(
        ids in proptest::collection::vec(0usize..VOCAB.len(), 0..64),
    ) {
        let tokenizer = tiny_tokenizer();
        let mut decoder = tokenizer.stream();
        let mut out_len = 0usize;
        for &id in &ids {
            if let Ok(piece) = decoder.push(id as i32) {
                out_len += piece.len();
            }
        }
        out_len += decoder.finish().len();
        // 每 token 最多产出 8 字节（词表里最长 token "▁a" 的 UTF-8 为 3 字节，
        // byte fallback 每 token 1 字节；给宽松上界防内存异常）。
        prop_assert!(out_len <= ids.len() * 8 + 16);
    }
}
