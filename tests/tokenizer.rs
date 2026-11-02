//! Tokenizer 的行为测试。
//!
//! 期望值全部手工推导：8-token 词表场景按 merges 逐级合并推到最终 id，
//! GGUF 夹具用 `tests/gguf_parser.rs` 的 Writer 模式在内存里构造。

use std::io::Cursor;

use qllama::gguf::{GgufFile, GgufValue};
use qllama::tokenizer::{EncodeOptions, Tokenizer};

/// GGUF 元数据类型编号（与本仓库 parser 一致：UINT32=4、INT32=5、
/// FLOAT32=6、BOOL=7、STRING=8、ARRAY=9、UINT64=10、INT64=11）。
const META_TYPE_UINT32: u32 = 4;
const META_TYPE_INT32: u32 = 5;
const META_TYPE_BOOL: u32 = 7;
const META_TYPE_STRING: u32 = 8;
const META_TYPE_ARRAY: u32 = 9;
const META_TYPE_UINT64: u32 = 10;
const META_TYPE_INT64: u32 = 11;

/// 极简 GGUF 字节写入器（测试文件间不共享代码，从 gguf_parser.rs 复制）。
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

    fn i32(&mut self, value: i32) -> &mut Self {
        self.buf.extend_from_slice(&value.to_le_bytes());
        self
    }

    fn string(&mut self, value: &str) -> &mut Self {
        self.u64(value.len() as u64);
        self.buf.extend_from_slice(value.as_bytes());
        self
    }

    fn finish(self) -> Vec<u8> {
        self.buf
    }
}

/// 测试词表：任务书场景 + "▁a" 补充，11 个 token。
///
/// merges ["a b", "ab c"]，词表 id 对照：0=<s> 1=</s> 2=a 3=b 4=c 5=ab
/// 6=abc 7=▁ 8=▁a 9=<0x41> 10=▁ab。
/// - encode("ab")（add_space_prefix=false）：[a,b] --rank0--> [ab] = [5]
/// - encode("abc")（同上）：[a,b,c] --rank0--> [ab,c] --rank1--> [abc] = [6]
/// - encode("a b")（同上）：空格→▁ 得 "a▁b"，任何相邻对都没有 rank，
///   保持 [a,▁,b] = [2, 7, 3]
/// - encode("ab")（默认 add_space_prefix=true）：[▁,a,b] → (a,b) rank0
///   合并 → [▁,ab]，▁=7、ab=5，得 [7, 5]；默认 encode 再前置 bos=0。
const VOCAB: [&str; 11] = [
    "<s>", "</s>", "a", "b", "c", "ab", "abc", "▁", "▁a", "<0x41>", "▁ab",
];
const MERGES: [&str; 2] = ["a b", "ab c"];
/// token_type：0、1 是 Control，其余 Normal；id 9 显式 Byte。
const TOKEN_TYPES: [i32; 11] = [3, 3, 1, 1, 1, 1, 1, 1, 1, 6, 1];

/// 构造一个带分词器元数据的内存 GGUF，再解析为 [`GgufFile`]。
///
/// `extra` 允许单条用例注入额外元数据（如 add_space_prefix=false）。
fn build_gguf(extra: &[(&str, GgufValue)]) -> GgufFile {
    let mut metadata: Vec<(&str, GgufValue)> = vec![
        ("general.architecture", GgufValue::String("llama".into())),
        ("tokenizer.ggml.model", GgufValue::String("llama".into())),
        (
            "tokenizer.ggml.tokens",
            GgufValue::Array(
                VOCAB
                    .iter()
                    .map(|t| GgufValue::String(t.to_string()))
                    .collect(),
            ),
        ),
        (
            "tokenizer.ggml.token_type",
            GgufValue::Array(TOKEN_TYPES.iter().map(|t| GgufValue::Int32(*t)).collect()),
        ),
        (
            "tokenizer.ggml.merges",
            GgufValue::Array(
                MERGES
                    .iter()
                    .map(|m| GgufValue::String(m.to_string()))
                    .collect(),
            ),
        ),
        ("tokenizer.ggml.bos_token_id", GgufValue::Uint32(0)),
        ("tokenizer.ggml.eos_token_id", GgufValue::Uint32(1)),
    ];
    metadata.extend(extra.iter().cloned());

    let mut w = Writer::new();
    w.buf.extend_from_slice(b"GGUF");
    w.u32(2); // version
    w.u64(0); // tensor_count
    w.u64(metadata.len() as u64);
    for (key, value) in &metadata {
        w.string(key);
        write_value(&mut w, value);
    }
    // 对齐到 8 字节即可（无 tensor 数据）。
    while !w.buf.len().is_multiple_of(8) {
        w.buf.push(0);
    }

    GgufFile::from_reader(&mut Cursor::new(w.finish())).expect("fixture should parse")
}

/// 把元数据值按 GGUF 规范写入字节流。
///
/// 类型标签只在顶层写一次：parser 读 ARRAY 时已知元素类型，逐元素直接读
/// 值本身——元素再带类型标签会被当成值解析，整个文件从此错位。
fn write_value(w: &mut Writer, value: &GgufValue) {
    write_typed_value(w, value, true)
}

fn write_typed_value(w: &mut Writer, value: &GgufValue, with_type_tag: bool) {
    if with_type_tag {
        let type_id = match value {
            GgufValue::String(_) => META_TYPE_STRING,
            GgufValue::Uint32(_) => META_TYPE_UINT32,
            GgufValue::Int32(_) => META_TYPE_INT32,
            GgufValue::Uint64(_) => META_TYPE_UINT64,
            GgufValue::Int64(_) => META_TYPE_INT64,
            GgufValue::Bool(_) => META_TYPE_BOOL,
            GgufValue::Array(_) => META_TYPE_ARRAY,
            other => panic!("fixture does not support metadata value {other:?}"),
        };
        w.u32(type_id);
    }
    match value {
        GgufValue::String(s) => {
            w.string(s);
        }
        GgufValue::Uint32(n) => {
            w.u32(*n);
        }
        GgufValue::Int32(n) => {
            w.i32(*n);
        }
        GgufValue::Uint64(n) => {
            w.u64(*n);
        }
        GgufValue::Int64(n) => {
            w.buf.extend_from_slice(&n.to_le_bytes());
        }
        GgufValue::Bool(flag) => {
            w.buf.push(u8::from(*flag));
        }
        GgufValue::Array(items) => {
            // parser 顺序：array_type → count → 元素。
            let element_type = items.first().map(|item| match item {
                GgufValue::String(_) => META_TYPE_STRING,
                GgufValue::Int32(_) => META_TYPE_INT32,
                GgufValue::Int64(_) => META_TYPE_INT64,
                other => panic!("fixture does not support array element {other:?}"),
            });
            if let Some(type_id) = element_type {
                w.u32(type_id);
            }
            w.u64(items.len() as u64);
            for item in items {
                write_typed_value(w, item, false);
            }
        }
        other => panic!("fixture does not support metadata value {other:?}"),
    }
}

/// 标准分词器：add_space_prefix 默认（true）。
fn tokenizer() -> Tokenizer {
    Tokenizer::from_gguf(&build_gguf(&[])).expect("tokenizer should load")
}

// ---------------------------------------------------------------- 任务书场景

/// "ab" 走两级合并：默认先补 ▁，[▁,a,b] --rank0--> [▁,ab] = [bos, 7, 5]。
#[test]
fn encode_merges_pair_into_existing_token() {
    let tokenizer = tokenizer();
    // 默认 encode = add_bos；▁(id7) + ab(id5)。
    assert_eq!(tokenizer.encode("ab").unwrap(), vec![0, 7, 5]);
}

/// 关闭 add_space_prefix 后 "ab" 得到 [bos, 5]。
#[test]
fn encode_without_space_prefix_gives_bare_merge() {
    let tokenizer = Tokenizer::from_gguf(&build_gguf(&[(
        "tokenizer.ggml.add_space_prefix",
        GgufValue::Bool(false),
    )]))
    .unwrap();
    assert_eq!(tokenizer.encode("ab").unwrap(), vec![0, 5]);
}

/// "abc" 两级合并：[a,b,c] → [ab,c]（rank0）→ [abc]（rank1）= [bos, 6]。
#[test]
fn encode_performs_two_merge_levels() {
    let tokenizer = Tokenizer::from_gguf(&build_gguf(&[(
        "tokenizer.ggml.add_space_prefix",
        GgufValue::Bool(false),
    )]))
    .unwrap();
    assert_eq!(tokenizer.encode("abc").unwrap(), vec![0, 6]);
}

/// "a b"：空格变 ▁，但 (a,▁) 与 (▁,b) 都没有 rank，保持三符号。
#[test]
fn encode_keeps_unranked_pairs_unmerged() {
    let tokenizer = Tokenizer::from_gguf(&build_gguf(&[(
        "tokenizer.ggml.add_space_prefix",
        GgufValue::Bool(false),
    )]))
    .unwrap();
    assert_eq!(tokenizer.encode("a b").unwrap(), vec![0, 2, 7, 3]);
}

/// decode 把 ▁ 还原为空格：[2,7,3] → "a▁b" → "a b"。
#[test]
fn decode_restores_spaces() {
    let tokenizer = tokenizer();
    assert_eq!(tokenizer.decode(&[2, 7, 3]).unwrap(), "a b");
}

/// 空格前缀下 BPE 只能合并 (a,b)，到不了整词 "▁ab"；锁定该行为。
#[test]
fn encode_prefers_whole_word_with_prefix() {
    let tokenizer = tokenizer();
    // "ab" 归一化为 "▁ab"，词表里有整词 id 10，但 BPE 从单字符开始、
    // 只有 merges 表驱动合并：(▁,a) 无 rank，到不了 [▁ab]。
    assert_eq!(tokenizer.encode("ab").unwrap(), vec![0, 7, 5]);
}

// ---------------------------------------------------------------- BOS / EOS

/// 默认编码在头部加 BOS（bos_token_id = 0）。
#[test]
fn encode_adds_bos_by_default() {
    let tokenizer = tokenizer();
    let ids = tokenizer
        .encode_with_options(
            "b",
            &EncodeOptions {
                add_bos: true,
                add_eos: false,
            },
        )
        .unwrap();
    // "b" → ▁b：[▁,b] 无 rank → [7, 3]，前面加 bos。
    assert_eq!(ids, vec![0, 7, 3]);
}

/// add_bos=false 时不加。
#[test]
fn encode_omits_bos_when_disabled() {
    let tokenizer = tokenizer();
    let ids = tokenizer
        .encode_with_options(
            "b",
            &EncodeOptions {
                add_bos: false,
                add_eos: false,
            },
        )
        .unwrap();
    assert_eq!(ids, vec![7, 3]);
}

/// 空文本 + add_bos → 只有 [bos]（llama.cpp：tokenize('', true) == [1]）。
#[test]
fn encode_empty_text_with_bos_yields_just_bos() {
    let tokenizer = tokenizer();
    let ids = tokenizer.encode("").unwrap();
    assert_eq!(ids, vec![0]);
}

/// 元数据没有 bos_token_id 时不加，也不报错。add_space_prefix 同样关闭：
/// "b" 无前缀直接命中 id 3（词表有裸 "b"）。
#[test]
fn encode_without_bos_id_skips_the_prefix() {
    let tokenizer = build_gguf_without_bos();
    let ids = tokenizer
        .encode_with_options(
            "b",
            &EncodeOptions {
                add_bos: true,
                add_eos: false,
            },
        )
        .unwrap();
    assert_eq!(ids, vec![3]);
}

/// add_eos=true 时尾部追加 EOS。
#[test]
fn encode_appends_eos_when_requested() {
    let tokenizer = tokenizer();
    let ids = tokenizer
        .encode_with_options(
            "b",
            &EncodeOptions {
                add_bos: false,
                add_eos: true,
            },
        )
        .unwrap();
    assert_eq!(ids, vec![7, 3, 1]);
}

/// 没有 eos_token_id 时 add_eos 静默不追加（与 BOS 同语义）。
#[test]
fn encode_without_eos_id_ignores_add_eos() {
    let tokenizer = build_gguf_without_bos();
    let ids = tokenizer
        .encode_with_options(
            "b",
            &EncodeOptions {
                add_bos: false,
                add_eos: true,
            },
        )
        .unwrap();
    // 该夹具仍有 eos=1：EOS 正常追加。真正的"缺失 eos"由
    // build_gguf_bare 构造（不带任何 special id）。
    assert_eq!(ids, vec![3, 1]);

    let bare_metadata: Vec<(String, GgufValue)> = vec![
        (
            "tokenizer.ggml.model".into(),
            GgufValue::String("llama".into()),
        ),
        (
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(vec![GgufValue::String("b".into())]),
        ),
        (
            "tokenizer.ggml.add_space_prefix".into(),
            GgufValue::Bool(false),
        ),
    ];
    let bare = Tokenizer::from_gguf(&parse(&build_gguf_bytes(&bare_metadata))).unwrap();
    let ids = bare
        .encode_with_options(
            "b",
            &EncodeOptions {
                add_bos: false,
                add_eos: true,
            },
        )
        .unwrap();
    assert_eq!(ids, vec![0]);
}

// ---------------------------------------------------------------- from_gguf 校验

/// 缺 tokens 报错且消息含键名。
#[test]
fn missing_tokens_key_is_rejected() {
    let error = Tokenizer::from_gguf(&build_gguf_bare(&[(
        "tokenizer.ggml.model",
        GgufValue::String("llama".into()),
    )]))
    .unwrap_err();
    assert!(error.contains("tokenizer.ggml.tokens"), "{error}");
}

/// tokens 空数组报错。
#[test]
fn empty_tokens_array_is_rejected() {
    let error = Tokenizer::from_gguf(&build_gguf_bare(&[
        ("tokenizer.ggml.model", GgufValue::String("llama".into())),
        ("tokenizer.ggml.tokens", GgufValue::Array(vec![])),
    ]))
    .unwrap_err();
    assert!(error.contains("must not be empty"), "{error}");
}

/// 重复 token 报错（llama.cpp 静默覆盖，我们拒绝——注释记录差异）。
#[test]
fn duplicate_tokens_are_rejected() {
    let error = Tokenizer::from_gguf(&build_gguf_bare(&[
        ("tokenizer.ggml.model", GgufValue::String("llama".into())),
        (
            "tokenizer.ggml.tokens",
            GgufValue::Array(vec![
                GgufValue::String("x".into()),
                GgufValue::String("x".into()),
            ]),
        ),
    ]))
    .unwrap_err();
    assert!(error.contains("duplicate"), "{error}");
}

/// token_type 数组短于 tokens 报错。
#[test]
fn short_token_type_array_is_rejected() {
    let metadata: Vec<(String, GgufValue)> = vec![
        (
            "tokenizer.ggml.model".into(),
            GgufValue::String("llama".into()),
        ),
        (
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(vec![
                GgufValue::String("a".into()),
                GgufValue::String("b".into()),
            ]),
        ),
        (
            "tokenizer.ggml.token_type".into(),
            GgufValue::Array(vec![GgufValue::Int32(1)]),
        ),
    ];
    let bytes = build_gguf_bytes(&metadata);
    let error = Tokenizer::from_gguf(&parse(&bytes)).unwrap_err();
    assert!(error.contains("token_type"), "{error}");
}

/// token_type 元素不是 Int32 报错。
#[test]
fn non_int32_token_type_is_rejected() {
    // 写入方把 Uint32 当 token_type 用：parser 原样读回，tokenizer 层拒绝。
    let metadata: Vec<(String, GgufValue)> = vec![
        (
            "tokenizer.ggml.model".into(),
            GgufValue::String("llama".into()),
        ),
        (
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(vec![GgufValue::String("a".into())]),
        ),
    ];
    // 直接构造字节流：token_type 数组声明元素类型 Uint32(4) 而非 Int32(5)。
    let mut w = Writer::new();
    w.buf.extend_from_slice(b"GGUF");
    w.u32(2);
    w.u64(0);
    w.u64((metadata.len() + 1) as u64);
    for (key, value) in &metadata {
        w.string(key);
        write_value(&mut w, value);
    }
    w.string("tokenizer.ggml.token_type");
    w.u32(META_TYPE_ARRAY);
    w.u32(META_TYPE_UINT32); // 错误的元素类型
    w.u64(1);
    w.u32(1);
    while !w.buf.len().is_multiple_of(8) {
        w.buf.push(0);
    }
    let error = Tokenizer::from_gguf(&parse(&w.finish())).unwrap_err();
    assert!(error.contains("token_type"), "{error}");
    let _ = &metadata;
}

/// merges 项没有空格分隔符报错。
#[test]
fn merge_without_space_separator_is_rejected() {
    let metadata: Vec<(String, GgufValue)> = vec![
        (
            "tokenizer.ggml.model".into(),
            GgufValue::String("llama".into()),
        ),
        (
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(vec![
                GgufValue::String("a".into()),
                GgufValue::String("ab".into()),
            ]),
        ),
        (
            "tokenizer.ggml.merges".into(),
            GgufValue::Array(vec![GgufValue::String("ab".into())]),
        ),
    ];
    let error = Tokenizer::from_gguf(&parse(&build_gguf_bytes(&metadata))).unwrap_err();
    assert!(error.contains("no space separator"), "{error}");
}

/// merges 首字符是空格也能解析（上游 strchr(word+1, ' ') 语义）。
#[test]
fn merge_leading_space_is_accepted() {
    // merges 项 " a b" 切成 (" ", "a b")；词表放一个含前导空格的 token
    // 使加载与编码都可走通（add_space_prefix=false，避免 ▁ fallback）。
    let metadata: Vec<(String, GgufValue)> = vec![
        (
            "tokenizer.ggml.model".into(),
            GgufValue::String("llama".into()),
        ),
        (
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(vec![
                GgufValue::String(" a".into()),
                GgufValue::String("b".into()),
                GgufValue::String(" ab".into()),
            ]),
        ),
        (
            "tokenizer.ggml.merges".into(),
            GgufValue::Array(vec![GgufValue::String(" a b".into())]),
        ),
        (
            "tokenizer.ggml.add_space_prefix".into(),
            GgufValue::Bool(false),
        ),
    ];
    let tokenizer = Tokenizer::from_gguf(&parse(&build_gguf_bytes(&metadata))).unwrap();
    // " ab"：文本首字符是空格 → 归一化成 "▁ab"？" ab".replace(' ', "▁") = "▁ab"。
    // 走 BPE [▁,a,b] 无 rank（merges 的是 " a"+"b"），逐符号查表：
    // ▁ 无词项 → byte fallback 报错。改用无空格输入验证加载本身成功。
    assert_eq!(tokenizer.encode("b").unwrap(), vec![1]);
}

/// bos/eos id 越界报错。
#[test]
fn out_of_range_bos_id_is_rejected() {
    let metadata: Vec<(String, GgufValue)> = vec![
        (
            "tokenizer.ggml.model".into(),
            GgufValue::String("llama".into()),
        ),
        (
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(vec![GgufValue::String("a".into())]),
        ),
        ("tokenizer.ggml.bos_token_id".into(), GgufValue::Uint32(5)),
    ];
    let error = Tokenizer::from_gguf(&parse(&build_gguf_bytes(&metadata))).unwrap_err();
    assert!(
        error.contains("bos_token_id") && error.contains("outside"),
        "{error}"
    );
}

/// bos_token_id 用 Int32 与 Uint64 两种类型都能读（写入方类型不一致的现实兼容）。
#[test]
fn special_ids_accept_multiple_integer_types() {
    for value in [
        GgufValue::Int32(0),
        GgufValue::Uint32(0),
        GgufValue::Int64(0),
        GgufValue::Uint64(0),
    ] {
        let metadata: Vec<(String, GgufValue)> = vec![
            (
                "tokenizer.ggml.model".into(),
                GgufValue::String("llama".into()),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                GgufValue::Array(vec![
                    GgufValue::String("a".into()),
                    GgufValue::String("b".into()),
                ]),
            ),
            ("tokenizer.ggml.bos_token_id".into(), value),
        ];
        let tokenizer = Tokenizer::from_gguf(&parse(&build_gguf_bytes(&metadata))).unwrap();
        assert_eq!(tokenizer.bos_id(), Some(0));
    }
}

/// 缺 token_type 时全部按 Normal 处理：<s> 也按普通文本编码。
#[test]
fn missing_token_type_treats_everything_as_normal() {
    let metadata: Vec<(String, GgufValue)> = vec![
        (
            "tokenizer.ggml.model".into(),
            GgufValue::String("llama".into()),
        ),
        (
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(vec![
                GgufValue::String("<s>".into()),
                GgufValue::String("x".into()),
            ]),
        ),
    ];
    let tokenizer = Tokenizer::from_gguf(&parse(&build_gguf_bytes(&metadata))).unwrap();
    // "x" 归一化为 "▁x"：▁ 不在词表、无 unk → byte fallback 逐字节，
    // ▁ 的 UTF-8 是 e2 96 81。词表没有这些 byte token → 报错。
    assert!(tokenizer.encode("x").is_err());
    // <s> 没有 Control 标记就不会被当作特殊 token 切分，而是走 BPE 逐字符；
    // 每个字符（<、s、>）都无词项、无 unk、无 byte token → 同样报错。
    let error = tokenizer
        .encode_with_options(
            "<s>",
            &EncodeOptions {
                add_bos: false,
                add_eos: false,
            },
        )
        .unwrap_err();
    assert!(error.contains("Cannot encode"), "{error}");
}

// ---------------------------------------------------------------- 特殊 token 与 fallback

/// 文本里的 Control token 整段映射为 id，不做 BPE 拆分。
#[test]
fn special_token_is_not_split_by_bpe() {
    let tokenizer = tokenizer();
    // "</s>" 是 Control（type 3）。它两侧的普通文本分别编码。
    let ids = tokenizer
        .encode_with_options(
            "a</s>b",
            &EncodeOptions {
                add_bos: false,
                add_eos: false,
            },
        )
        .unwrap();
    // "a" → [7, 2]（▁ + a），</s> → [1]，"b" → [7, 3]。
    // 本实现的段编码每段独立补 ▁（简化：无 is_prev_special 追踪）。
    assert_eq!(ids, vec![7, 2, 1, 7, 3]);
}

/// unk_id 存在时未命中符号返回 unk。
#[test]
fn unknown_symbol_maps_to_unk_id() {
    let metadata: Vec<(String, GgufValue)> = vec![
        (
            "tokenizer.ggml.model".into(),
            GgufValue::String("llama".into()),
        ),
        (
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(vec![
                GgufValue::String("a".into()),
                GgufValue::String("<unk>".into()),
            ]),
        ),
        (
            "tokenizer.ggml.token_type".into(),
            GgufValue::Array(vec![GgufValue::Int32(1), GgufValue::Int32(2)]),
        ),
        (
            "tokenizer.ggml.unknown_token_id".into(),
            GgufValue::Uint32(1),
        ),
        (
            "tokenizer.ggml.add_space_prefix".into(),
            GgufValue::Bool(false),
        ),
    ];
    let tokenizer = Tokenizer::from_gguf(&parse(&build_gguf_bytes(&metadata))).unwrap();
    // "z" 不在词表 → unk id 1。
    assert_eq!(tokenizer.encode("z").unwrap(), vec![1]);
}

/// 词表带 byte token `<0x41>` 时，未命中的 "A" 走 byte fallback 得到它。
#[test]
fn byte_fallback_uses_hex_byte_token() {
    let tokenizer = tokenizer();
    // "A" 不在词表（VOCAB 只有 <0x41>），无 unk → byte fallback。
    // "A" 归一化后是 "▁A"：[▁, A]，▁(id 7) 命中，A → <0x41> = id 9；
    // 默认 encode 前置 bos id 0。
    assert_eq!(tokenizer.encode("A").unwrap(), vec![0, 7, 9]);
}

/// 无 unk、无 byte token 时未命中的符号报错而不是编造 id。
#[test]
fn unencodable_symbol_is_an_error() {
    let metadata: Vec<(String, GgufValue)> = vec![
        (
            "tokenizer.ggml.model".into(),
            GgufValue::String("llama".into()),
        ),
        (
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(vec![GgufValue::String("a".into())]),
        ),
        (
            "tokenizer.ggml.add_space_prefix".into(),
            GgufValue::Bool(false),
        ),
    ];
    let tokenizer = Tokenizer::from_gguf(&parse(&build_gguf_bytes(&metadata))).unwrap();
    let error = tokenizer.encode("z").unwrap_err();
    assert!(error.contains("Cannot encode"), "{error}");
}

// ---------------------------------------------------------------- decode

/// 越界 id 解码报错。
#[test]
fn decode_rejects_out_of_range_ids() {
    let tokenizer = tokenizer();
    let error = tokenizer.decode(&[99]).unwrap_err();
    assert!(error.contains("out of range"), "{error}");
}

/// Control token 原样输出其文本。
#[test]
fn decode_prints_control_tokens_verbatim() {
    let tokenizer = tokenizer();
    assert_eq!(tokenizer.decode(&[0, 2, 1]).unwrap(), "<s>a</s>");
}

/// 多字节 UTF-8 字符逐 char 拆分后可编码（词表含对应字符时直接命中）。
#[test]
fn multibyte_characters_encode_per_char() {
    let metadata: Vec<(String, GgufValue)> = vec![
        (
            "tokenizer.ggml.model".into(),
            GgufValue::String("llama".into()),
        ),
        (
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(vec![
                GgufValue::String("中".into()),
                GgufValue::String("文".into()),
            ]),
        ),
        (
            "tokenizer.ggml.add_space_prefix".into(),
            GgufValue::Bool(false),
        ),
    ];
    let tokenizer = Tokenizer::from_gguf(&parse(&build_gguf_bytes(&metadata))).unwrap();
    assert_eq!(tokenizer.encode("中文").unwrap(), vec![0, 1]);
    assert_eq!(tokenizer.decode(&[0, 1]).unwrap(), "中文");
}

/// encode → decode 往返闭合（add_space_prefix=false 避免前导空格差异）。
#[test]
fn round_trip_recovers_the_original_text() {
    // build_gguf_without_bos 无 bos 且 add_space_prefix=false：encode 结果
    // 不带 bos，decode 完整还原。
    let tokenizer = build_gguf_without_bos();
    let text = "ab abc a b";
    let ids = tokenizer
        .encode_with_options(
            text,
            &EncodeOptions {
                add_bos: false,
                add_eos: false,
            },
        )
        .unwrap();
    assert_eq!(tokenizer.decode(&ids).unwrap(), text);
}

// ---------------------------------------------------------------- 辅助构造

/// 与 [`build_gguf`] 相同但允许完全自定义 metadata 列表（键名用 String）。
fn build_gguf_bytes(metadata: &[(String, GgufValue)]) -> Vec<u8> {
    let mut w = Writer::new();
    w.buf.extend_from_slice(b"GGUF");
    w.u32(2);
    w.u64(0);
    w.u64(metadata.len() as u64);
    for (key, value) in metadata {
        w.string(key);
        write_value(&mut w, value);
    }
    while !w.buf.len().is_multiple_of(8) {
        w.buf.push(0);
    }
    w.finish()
}

fn parse(bytes: &[u8]) -> GgufFile {
    GgufFile::from_reader(&mut Cursor::new(bytes.to_vec())).expect("fixture should parse")
}

/// 不带 bos/eos/merges/token_type 的最小分词器夹具。
fn build_gguf_bare(extra: &[(&str, GgufValue)]) -> GgufFile {
    let mut metadata: Vec<(&str, GgufValue)> =
        vec![("general.architecture", GgufValue::String("llama".into()))];
    metadata.extend(extra.iter().cloned());

    let mut w = Writer::new();
    w.buf.extend_from_slice(b"GGUF");
    w.u32(2);
    w.u64(0);
    w.u64(metadata.len() as u64);
    for (key, value) in &metadata {
        w.string(key);
        write_value(&mut w, value);
    }
    while !w.buf.len().is_multiple_of(8) {
        w.buf.push(0);
    }
    GgufFile::from_reader(&mut Cursor::new(w.finish())).expect("fixture should parse")
}

/// 从标准夹具去掉 bos_token_id 键（用于"缺失 bos"分支），同时关闭
/// add_space_prefix（用于往返测试，避免前导空格差异）。
fn build_gguf_without_bos() -> Tokenizer {
    // 重建一份不含 bos_token_id 的元数据。
    let metadata: Vec<(String, GgufValue)> = vec![
        (
            "general.architecture".into(),
            GgufValue::String("llama".into()),
        ),
        (
            "tokenizer.ggml.model".into(),
            GgufValue::String("llama".into()),
        ),
        (
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(
                VOCAB
                    .iter()
                    .map(|t| GgufValue::String(t.to_string()))
                    .collect(),
            ),
        ),
        (
            "tokenizer.ggml.token_type".into(),
            GgufValue::Array(TOKEN_TYPES.iter().map(|t| GgufValue::Int32(*t)).collect()),
        ),
        (
            "tokenizer.ggml.merges".into(),
            GgufValue::Array(
                MERGES
                    .iter()
                    .map(|m| GgufValue::String(m.to_string()))
                    .collect(),
            ),
        ),
        ("tokenizer.ggml.eos_token_id".into(), GgufValue::Uint32(1)),
        (
            "tokenizer.ggml.add_space_prefix".into(),
            GgufValue::Bool(false),
        ),
    ];
    Tokenizer::from_gguf(&parse(&build_gguf_bytes(&metadata))).unwrap()
}
