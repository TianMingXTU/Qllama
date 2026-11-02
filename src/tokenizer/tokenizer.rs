//! Tokenizer：自然语言文本 ↔ token id 序列。
//!
//! ⚠️ 已知限制（对照 llama.cpp `src/llama-vocab.cpp` 的完整实现）：
//! 1. 真实 Llama GGUF（SPM）没有 `tokenizer.ggml.merges` 而用
//!    `tokenizer.ggml.scores` 驱动合并；本阶段不实现 scores 驱动的 SPM 合并，
//!    对无 merges 的文件退化为"逐字符 + byte fallback"——编码粒度与
//!    llama.cpp 不同，但 id 总能产生、encode→decode 往返闭合。
//! 2. 特殊 token 切分对全部 Control/UserDefined/Unknown 类 token 生效，
//!    没有 llama.cpp 的 `parse_special` 开关。
//! 3. decode 对 byte token（`<0xXX>`）直接输出其文本形式，不做字节级
//!    UTF-8 重组——llama.cpp 的 detokenize 同样逐 token 拼接，多字节字符
//!    由调用方拼接后才合法；`decode(encode("中"))` 会得到
//!    `"<0xE4><0xB8><0xAD>"` 而非 `"中"`。
//! 4. 未实现 precompiled_charsmap（NFC 归一化）等归一化步骤。

use std::collections::HashMap;

use crate::gguf::{GgufFile, GgufValue};

/// token 在 GGUF `tokenizer.ggml.token_type` 里的类别。
///
/// 数值对应 include/llama.h 的 `LLAMA_TOKEN_TYPE_*` 枚举。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl TokenType {
    /// 从 GGUF 里存的整数构造；未知值按 UNDEFINED 处理。
    fn from_ggml_value(value: i32) -> Self {
        match value {
            1 => TokenType::Normal,
            2 => TokenType::Unknown,
            3 => TokenType::Control,
            4 => TokenType::UserDefined,
            5 => TokenType::Unused,
            6 => TokenType::Byte,
            // 0 是 UNDEFINED；写其他值说明文件生成方有误，按未定义对待。
            _ => TokenType::Unused,
        }
    }
}

/// encode 的行为开关；默认与 llama.cpp 对 SPM 模型的 `add_special=true` 一致。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodeOptions {
    /// 在序列头部插入 BOS token（模型元数据给出 bos_token_id 时才生效）。
    pub add_bos: bool,
    /// 在序列尾部追加 EOS token。
    pub add_eos: bool,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            add_bos: true,
            add_eos: false,
        }
    }
}

/// BPE 分词器：从 GGUF 元数据加载词表与合并表。
#[derive(Debug, Clone)]
pub struct Tokenizer {
    token_to_id: HashMap<String, i32>,
    id_to_token: Vec<String>,
    /// 原始 merges 表（调试与展示用）；合并优先级看 [`Self::merge_ranks`]。
    #[allow(dead_code)]
    merges: Vec<(String, String)>,
    merge_ranks: HashMap<(String, String), usize>,
    token_types: Vec<TokenType>,
    bos_id: Option<i32>,
    eos_id: Option<i32>,
    unk_id: Option<i32>,
    /// 是否在文本首部补空格（SPM 语义，llama.cpp 的 add_space_prefix）。
    add_space_prefix: bool,
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer {
    /// 空分词器：没有词表，encode/decode 自然报错。
    pub fn new() -> Self {
        Self {
            token_to_id: HashMap::new(),
            id_to_token: Vec::new(),
            merges: Vec::new(),
            merge_ranks: HashMap::new(),
            token_types: Vec::new(),
            bos_id: None,
            eos_id: None,
            unk_id: None,
            add_space_prefix: true,
        }
    }

    /// 从 GGUF 元数据构造分词器。
    ///
    /// 键名与上游 llama.cpp `src/llama-arch.cpp` 的 LLM_KV_NAMES 表一致。
    /// `tokenizer.ggml.model` 与 `tokenizer.ggml.tokens` 必填；merges、
    /// token_type、bos/eos/unknown id、add_space_prefix 可选。可选键存在但
    /// 类型或范围非法时报错——"缺失"与"写坏"必须区分，写坏的文件不能蒙混。
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let metadata = &gguf.metadata;

        // model 键必填（如 "llama"），但本阶段统一用 BPE-with-merges 算法，
        // 不按类型分派——值只做存在性校验。
        metadata
            .get("tokenizer.ggml.model")
            .and_then(|v| match v {
                GgufValue::String(_) => Some(()),
                _ => None,
            })
            .ok_or_else(|| "Missing or invalid metadata key: tokenizer.ggml.model".to_string())?;

        let tokens: Vec<String> = read_string_array(metadata, "tokenizer.ggml.tokens")?;
        if tokens.is_empty() {
            return Err("Metadata key tokenizer.ggml.tokens must not be empty".to_string());
        }
        // llama.cpp 对重复 token 静默覆盖（后者胜出）；这里直接拒绝——静默
        // 覆盖会让 encode 结果依赖文件里重复项的写法，属于文件级错误。
        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (index, token) in tokens.iter().enumerate() {
            if token_to_id.insert(token.clone(), index as i32).is_some() {
                return Err(format!(
                    "Metadata key tokenizer.ggml.tokens contains duplicate token {token:?} (index {index})"
                ));
            }
        }

        // token_type 可选；存在时长度必须覆盖全部 token（与上游一致）。
        let token_types = match metadata.get("tokenizer.ggml.token_type") {
            None => vec![TokenType::Normal; tokens.len()],
            Some(GgufValue::Array(items)) => {
                let mut types = Vec::with_capacity(items.len());
                for (index, item) in items.iter().enumerate() {
                    let value = match item {
                        GgufValue::Int32(n) => *n,
                        other => {
                            return Err(format!(
                                "Metadata key tokenizer.ggml.token_type element {index} is not Int32: {other:?}"
                            ))
                        }
                    };
                    types.push(TokenType::from_ggml_value(value));
                }
                if types.len() < tokens.len() {
                    return Err(format!(
                        "Metadata key tokenizer.ggml.token_type has {} entries but the vocab has {} tokens",
                        types.len(),
                        tokens.len()
                    ));
                }
                types
            }
            Some(_) => {
                return Err("Missing or invalid metadata key: tokenizer.ggml.token_type".to_string())
            }
        };

        // merges 可选（真实 SPM 文件没有它）。每项 "A B" 按上游
        // strchr(word + 1, ' ') 的语义切分：从第二个字符起找第一个空格，
        // 因此首字符允许是空格。
        let mut merges = Vec::new();
        let mut merge_ranks = HashMap::new();
        match metadata.get("tokenizer.ggml.merges") {
            None => {}
            Some(GgufValue::Array(items)) => {
                for (index, item) in items.iter().enumerate() {
                    let word = match item {
                        GgufValue::String(s) => s,
                        other => {
                            return Err(format!(
                                "Metadata key tokenizer.ggml.merges element {index} is not a string: {other:?}"
                            ))
                        }
                    };
                    // 从第二个字节起找空格，对应上游 strchr(word + 1, ' ')
                    // 的语义：首字符允许是空格。必须按字节扫描（上游就是
                    // 纯字节搜索）而不是 word[1..] 切片——后者要求字节 1
                    // 恰好是字符边界，首字符是多字节 UTF-8（如 GPT-2 merges
                    // 首行的 "Ġ t"、SPM 的 "▁a b"）时直接 panic。UTF-8 的
                    // 续字节与首字节都不会是 0x20，因此命中的空格必在字符
                    // 边界上，两侧切片安全。
                    let split = if word.is_empty() {
                        None
                    } else {
                        word.as_bytes()[1..]
                            .iter()
                            .position(|&byte| byte == b' ')
                            .map(|position| position + 1)
                    };
                    let Some(split) = split else {
                        return Err(format!(
                            "Metadata key tokenizer.ggml.merges element {index} ({word:?}) has no space separator"
                        ));
                    };
                    let pair = (word[..split].to_string(), word[split + 1..].to_string());
                    // 上游 bpe_ranks.emplace 对重复对保留首次出现的 rank
                    // （llama-vocab.cpp L2330）；HashMap::insert 是 last-wins，
                    // 会让同一文件的合并优先级与上游相反。
                    merge_ranks.entry(pair.clone()).or_insert(index);
                    merges.push(pair);
                }
            }
            Some(_) => {
                return Err("Missing or invalid metadata key: tokenizer.ggml.merges".to_string())
            }
        }

        // bos/eos/unk：任何整型都接受（写入方用 Uint32/Int32/Uint64 不一致），
        // 越界 id 报错而不是编码时静默产生越界访问。
        let bos_id = read_optional_id(metadata, "tokenizer.ggml.bos_token_id", tokens.len())?;
        let eos_id = read_optional_id(metadata, "tokenizer.ggml.eos_token_id", tokens.len())?;
        let unk_id = read_optional_id(metadata, "tokenizer.ggml.unknown_token_id", tokens.len())?;

        // 默认 true：llama.cpp 对 SPM 类型设 add_space_prefix=true（L2777），
        // 真实 Llama GGUF 通常不写这个键。
        let add_space_prefix = match metadata.get("tokenizer.ggml.add_space_prefix") {
            None => true,
            Some(GgufValue::Bool(value)) => *value,
            Some(_) => {
                return Err(
                    "Missing or invalid metadata key: tokenizer.ggml.add_space_prefix".to_string(),
                )
            }
        };

        Ok(Self {
            token_to_id,
            id_to_token: tokens,
            merges,
            merge_ranks,
            token_types,
            bos_id,
            eos_id,
            unk_id,
            add_space_prefix,
        })
    }

    /// BOS token id；元数据未提供时为 None。
    pub fn bos_id(&self) -> Option<i32> {
        self.bos_id
    }

    /// EOS token id；元数据未提供时为 None。
    pub fn eos_id(&self) -> Option<i32> {
        self.eos_id
    }

    /// 词表大小。
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// 编码文本；默认行为（加 BOS、不加 EOS）与 llama.cpp 的
    /// `tokenize(text, add_special=true)` 对 SPM 的语义一致。
    pub fn encode(&self, text: &str) -> Result<Vec<i32>, String> {
        self.encode_with_options(text, &EncodeOptions::default())
    }

    /// 按给定开关编码文本。
    pub fn encode_with_options(
        &self,
        text: &str,
        options: &EncodeOptions,
    ) -> Result<Vec<i32>, String> {
        let mut ids = Vec::new();
        // 特殊 token 切分后再逐普通段进 BPE；段间顺序即文本顺序。
        for segment in split_special_tokens(self, text) {
            match segment {
                Segment::Special(id) => ids.push(id),
                Segment::Text(text) => ids.extend(self.encode_bpe_segment(&text)?),
            }
        }

        // llama.cpp 在 add_special 时先加 BOS 再处理片段，最后补 EOS；
        // 空文本 + add_bos 因此得到 [bos]（tokenize('', true) == [1]）。
        if options.add_bos {
            if let Some(bos) = self.bos_id {
                ids.insert(0, bos);
            }
        }
        if options.add_eos {
            if let Some(eos) = self.eos_id {
                ids.push(eos);
            }
        }
        Ok(ids)
    }

    /// 解码 id 序列为文本。
    ///
    /// Control/UserDefined 等特殊 token 原样输出其文本（`<s>` 之类），普通
    /// token 把 `▁` 还原为空格；byte token 直接输出 `<0xXX>` 文本形式
    /// （模块头限制 3）。
    pub fn decode(&self, tokens: &[i32]) -> Result<String, String> {
        let mut text = String::new();
        for &id in tokens {
            let index = self.check_id_range(id)?;
            let token_text = &self.id_to_token[index];
            match self.token_types.get(index) {
                // 特殊类 token 不做 ▁ 还原：它们的文本是字面标记。
                Some(TokenType::Control)
                | Some(TokenType::UserDefined)
                | Some(TokenType::Unknown) => text.push_str(token_text),
                _ => {
                    text.push_str(&token_text.replace('▁', " "));
                }
            }
        }
        Ok(text)
    }

    /// 把一段普通文本走 BPE 流程编码为 id。
    ///
    /// 1. 空格 → `▁`，可选段首补 `▁`（SPM 归一化）；
    /// 2. 按 Unicode 字符拆成初始符号；
    /// 3. 标准 BPE：每轮在所有相邻对里找 merge_ranks 最小的一组合并
    ///    （并列取最左侧），重扫直到无可合并对——每轮 O(n) 线性扫描而非
    ///    优先队列：结果与 llama.cpp 的实现相同（同 rank 序、最左优先），
    ///    仅复杂度差异，且线性重扫天然没有"过期 bigram"问题；
    /// 4. 符号 → id：查表未命中时先 unk，再逐字节 `<0xXX>` fallback，
    ///    仍失败则报错。
    fn encode_bpe_segment(&self, text: &str) -> Result<Vec<i32>, String> {
        let mut normalized = String::with_capacity(text.len() + 3);
        if self.add_space_prefix && !text.is_empty() {
            normalized.push('▁');
        }
        normalized.push_str(&text.replace(' ', "▁"));

        let mut symbols: Vec<String> = normalized.chars().map(|c| c.to_string()).collect();
        loop {
            // (rank, position) 的最小值：rank 并列时 position 最小者在前。
            let mut best: Option<(usize, usize)> = None;
            for position in 0..symbols.len().saturating_sub(1) {
                if let Some(&rank) = self
                    .merge_ranks
                    .get(&(symbols[position].clone(), symbols[position + 1].clone()))
                {
                    if best.is_none() || rank < best.unwrap().0 {
                        best = Some((rank, position));
                    }
                }
            }
            let Some((_, position)) = best else {
                break;
            };
            let merged = format!("{}{}", symbols[position], symbols[position + 1]);
            symbols[position] = merged;
            symbols.remove(position + 1);
        }

        let mut ids = Vec::with_capacity(symbols.len());
        for symbol in &symbols {
            if let Some(&id) = self.token_to_id.get(symbol) {
                ids.push(id);
                continue;
            }
            // ⚠️ 与上游的差异：任务书要求未命中先 unk 再 byte fallback，
            // 本实现照此执行。llama.cpp 的标准 BPE 从不输出 unk——对未命中
            // 符号直接逐字节 fallback（llama-vocab.cpp L798-818，unk 分支
            // 仅 SOPRANO pre-type 生效，unk 输出只存在于 SPM viterbi 路径）。
            // "带 unk 配置 + 词表缺该符号但有 byte token" 的文件两侧结果
            // 不同：这里得 unk，上游得 byte token 序列。若后续实现 scores
            // 驱动的 SPM 合并，需按模型类型分派这条路径。
            if let Some(unk) = self.unk_id {
                ids.push(unk);
                continue;
            }
            // byte fallback：符号逐字节转成 `<0xXX>` 词项（大写 hex，与
            // llama.cpp 的 byte_to_token 一致）。
            for byte in symbol.bytes() {
                let byte_token = format!("<0x{byte:02X}>");
                match self.token_to_id.get(&byte_token) {
                    Some(&id) => ids.push(id),
                    None => {
                        return Err(format!(
                            "Cannot encode symbol {symbol:?}: not in vocab, no unk token, and no byte token {byte_token}"
                        ))
                    }
                }
            }
        }
        Ok(ids)
    }

    /// 校验 id 落在词表范围内并返回其下标。
    fn check_id_range(&self, id: i32) -> Result<usize, String> {
        usize::try_from(id)
            .ok()
            .filter(|&index| index < self.id_to_token.len())
            .ok_or_else(|| {
                format!(
                    "Token id {id} is out of range for a vocab of {} tokens",
                    self.id_to_token.len()
                )
            })
    }

    /// 文本里的字符串是否对应一个 Control/UserDefined/Unknown 类 token。
    fn special_token_id(&self, text: &str) -> Option<i32> {
        self.token_to_id.get(text).copied().filter(|&id| {
            matches!(
                self.token_types[id as usize],
                TokenType::Control | TokenType::UserDefined | TokenType::Unknown
            )
        })
    }
}

/// `encode_with_options` 的输入片段：整段特殊 token 或待 BPE 的普通文本。
enum Segment {
    Special(i32),
    Text(String),
}

/// 把文本按特殊 token 的出现位置切成片段。
///
/// 与 llama.cpp 的 `tokenizer_st_partition` 一致：按 token 文本长度降序
/// 逐个扫描，最长匹配优先（先扫到谁谁切）。
///
/// 空文本的特殊 token 被跳过：`find("")` 恒命中位置 0 且零字节推进，切分
/// 循环会永不终止。上游没有这个问题，是因为它在加载时就把空 token 改名为
/// `[EMPTY_i]`（llama-vocab.cpp L2859-2862）；这里在收集阶段跳过，语义
/// 等价——空 token 的文本本来就不该匹配任何内容。
fn split_special_tokens(tokenizer: &Tokenizer, text: &str) -> Vec<Segment> {
    let mut specials: Vec<(i32, String)> = (0..tokenizer.id_to_token.len() as i32)
        .filter_map(|id| {
            let index = id as usize;
            let token_text = &tokenizer.id_to_token[index];
            if token_text.is_empty() {
                return None;
            }
            tokenizer
                .special_token_id(token_text)
                .map(|special_id| (special_id, token_text.clone()))
        })
        .collect();
    specials.sort_by_key(|(_, token_text)| std::cmp::Reverse(token_text.len()));

    let mut segments = vec![Segment::Text(text.to_string())];
    for (special_id, special_text) in specials {
        let mut next = Vec::new();
        for segment in segments {
            match segment {
                Segment::Special(id) => next.push(Segment::Special(id)),
                Segment::Text(text) => {
                    let mut rest = text.as_str();
                    while let Some(position) = rest.find(special_text.as_str()) {
                        if position > 0 {
                            next.push(Segment::Text(rest[..position].to_string()));
                        }
                        next.push(Segment::Special(special_id));
                        rest = &rest[position + special_text.len()..];
                    }
                    if !rest.is_empty() {
                        next.push(Segment::Text(rest.to_string()));
                    }
                }
            }
        }
        segments = next;
    }
    segments
}

/// 读取字符串数组元数据键。
fn read_string_array(
    metadata: &HashMap<String, GgufValue>,
    key: &str,
) -> Result<Vec<String>, String> {
    match metadata.get(key) {
        None => Err(format!("Missing or invalid metadata key: {key}")),
        Some(GgufValue::Array(items)) => {
            let mut strings = Vec::with_capacity(items.len());
            for (index, item) in items.iter().enumerate() {
                match item {
                    GgufValue::String(s) => strings.push(s.clone()),
                    other => {
                        return Err(format!(
                            "Metadata key {key} element {index} is not a string: {other:?}"
                        ))
                    }
                }
            }
            Ok(strings)
        }
        Some(_) => Err(format!("Missing or invalid metadata key: {key}")),
    }
}

/// 读取一个可选的 token id：缺失返回 None；存在则任何整型都接受且必须
/// 落在词表范围内。
///
/// 上游对"无该特殊 token"的落盘方式是**不写键**（官方转换器对缺失的
/// 特殊 token 跳过写入），对应本函数的 None 分支；GGUF 规范也把这些键
/// 定义为 uint32，没有负数约定——所以键存在但值越界仍然按文件级错误
/// 拒绝。唯一豁免是 -1 及其无符号回绕写法（`u32::MAX`/`u64::MAX`）：
/// llama.h 以 `LLAMA_TOKEN_NULL = -1` 表达"无此 token"，真实世界确实
/// 存在以回绕值落盘 -1 的文件（new_id 为 uint32 时 -1 与 u32::MAX 无法
/// 区分），把它们视为未设置可以加载这类文件，同时不放松对其他越界值
/// 的校验。
fn read_optional_id(
    metadata: &HashMap<String, GgufValue>,
    key: &str,
    vocab_size: usize,
) -> Result<Option<i32>, String> {
    let raw = match metadata.get(key) {
        None => return Ok(None),
        Some(value) => value,
    };
    let is_null_sentinel = match raw {
        GgufValue::Int32(n) => *n == -1,
        GgufValue::Int64(n) => *n == -1,
        GgufValue::Uint32(n) => *n == u32::MAX,
        GgufValue::Uint64(n) => *n == u64::MAX,
        _ => return Err(format!("Missing or invalid metadata key: {key}")),
    };
    if is_null_sentinel {
        return Ok(None);
    }
    let value: i64 = match raw {
        GgufValue::Int32(n) => i64::from(*n),
        GgufValue::Uint32(n) => i64::from(*n),
        GgufValue::Int64(n) => *n,
        GgufValue::Uint64(n) => {
            let n = i64::try_from(*n)
                .map_err(|_| format!("Metadata key {key} is out of i32 range: {n}"))?;
            n
        }
        _ => unreachable!("sentinel check above rejected non-integer values"),
    };
    if value < 0 || value >= vocab_size as i64 {
        return Err(format!(
            "Metadata key {key} is {value}, outside the vocab range [0, {vocab_size})"
        ));
    }
    Ok(Some(value as i32))
}
