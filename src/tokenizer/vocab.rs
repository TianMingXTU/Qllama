//! GGUF 分词：Llama SPM 使用词表分数，GPT-2/StarCoder 保留现有 merges 路径。
//!
//! 特殊 token 按最长字面匹配切分；decode 保留其字面文本。解码先拼接 byte
//! token 的原始字节，再统一转换 UTF-8；不完整或非法序列使用 U+FFFD，避免把
//! 多字节字符的每个 byte token 分别解码。模型需要的额外归一化若未实现则报错。

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

#[derive(Debug, Clone)]
enum TokenizerModel {
    Spm { scores: Vec<f32> },
    MergeBpe,
}

/// 从 GGUF 元数据加载的词表与分词算法。
#[derive(Debug, Clone)]
pub struct Tokenizer {
    model: TokenizerModel,
    token_to_id: HashMap<String, i32>,
    id_to_token: Vec<String>,
    /// merge 对到文件中的序号；序号小者优先合并，重复对保留首次出现的 rank
    /// （上游 bpe_ranks.emplace 的语义，HashMap::insert 的 last-wins 会反过来）。
    merge_ranks: HashMap<(String, String), usize>,
    token_types: Vec<TokenType>,
    bos_id: Option<i32>,
    eos_id: Option<i32>,
    unk_id: Option<i32>,
    /// 是否在文本首部补空格（SPM 语义，llama.cpp 的 add_space_prefix）。
    add_space_prefix: bool,
}

impl Tokenizer {
    /// 从 GGUF 元数据构造分词器。
    ///
    /// 键名与上游 llama.cpp `src/llama-arch.cpp` 的 LLM_KV_NAMES 表一致。
    /// `tokenizer.ggml.model` 与 `tokenizer.ggml.tokens` 必填；Llama SPM 还要求
    /// 与词表等长的有限 Float32 scores。merges、token_type、bos/eos/unknown id、
    /// add_space_prefix 可选。可选键存在但
    /// 类型或范围非法时报错——"缺失"与"写坏"必须区分，写坏的文件不能蒙混。
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let metadata = &gguf.metadata;

        let model_name = match metadata.get("tokenizer.ggml.model") {
            Some(GgufValue::String(name))
                if matches!(name.as_str(), "llama" | "gpt2" | "starcoder") =>
            {
                name.as_str()
            }
            Some(GgufValue::String(name)) => {
                return Err(format!(
                    "Unsupported tokenizer model {name:?}; supported: llama, gpt2, starcoder"
                ))
            }
            _ => return Err("Missing or invalid metadata key: tokenizer.ggml.model".to_string()),
        };
        if metadata.contains_key("tokenizer.ggml.precompiled_charsmap") {
            return Err("Unsupported tokenizer normalization: precompiled_charsmap".to_string());
        }
        match metadata.get("tokenizer.ggml.remove_extra_whitespaces") {
            None | Some(GgufValue::Bool(false)) => {}
            Some(GgufValue::Bool(true)) => {
                return Err(
                    "Unsupported tokenizer normalization: remove_extra_whitespaces=true"
                        .to_string(),
                )
            }
            Some(_) => {
                return Err(
                    "Invalid metadata type: tokenizer.ggml.remove_extra_whitespaces must be Bool"
                        .to_string(),
                )
            }
        }

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

        // merges 可选（真实 SPM 文件没有它）。
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
                    // 分隔符搜索从第二个字节开始，对应上游 strchr(word + 1, ' ')
                    // ——首字符允许是空格。这里按字节扫描而不是 word[1..] 切片：
                    // 切片要求字节 1 恰好在字符边界上，首字符是多字节 UTF-8
                    // （GPT-2 merges 的 "Ġ t"、SPM 的 "▁a b"）时直接 panic。
                    // UTF-8 的续字节不会是 0x20，命中的空格必在字符边界，切片安全。
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
                    merge_ranks.entry(pair).or_insert(index);
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

        let model = if model_name == "llama" {
            let items = match metadata.get("tokenizer.ggml.scores") {
                Some(GgufValue::Array(items)) if items.len() == tokens.len() => items,
                _ => return Err("Metadata key tokenizer.ggml.scores must contain one Float32 score per token for llama SPM".to_string()),
            };
            let scores = items
                .iter()
                .enumerate()
                .map(|(index, value)| match value {
                    GgufValue::Float32(score) if score.is_finite() => Ok(*score),
                    _ => Err(format!(
                        "Metadata key tokenizer.ggml.scores has invalid score at index {index}"
                    )),
                })
                .collect::<Result<Vec<_>, _>>()?;
            TokenizerModel::Spm { scores }
        } else {
            TokenizerModel::MergeBpe
        };
        for (index, token_type) in token_types.iter().take(tokens.len()).enumerate() {
            if *token_type == TokenType::Byte && byte_value(&tokens[index]).is_none() {
                return Err(format!(
                    "Invalid byte token {:?} at index {index}; expected <0xXX>",
                    tokens[index]
                ));
            }
        }
        Ok(Self {
            model,
            token_to_id,
            id_to_token: tokens,
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
        // 特殊 token 隔开普通片段，片段之间不合并。
        for segment in split_special_tokens(self, text) {
            match segment {
                Segment::Special(id) => ids.push(id),
                Segment::Text(text) => match &self.model {
                    TokenizerModel::Spm { scores } => {
                        ids.extend(self.encode_spm_segment(&text, scores)?)
                    }
                    TokenizerModel::MergeBpe => ids.extend(self.encode_bpe_segment(&text)?),
                },
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

    /// 解码 id 序列；特殊标记保留字面文本，`▁` 还原为空格。
    ///
    /// byte token 先拼成原始字节，再统一 UTF-8 解码；非法序列使用 U+FFFD。
    /// SPM 每段普通文本前的 dummy 空格在此移除；与 encode(add_bos=false) 往返
    /// 可恢复原文，字面 U+2581 与空格发生归一化碰撞时除外。自动 BOS 不会被
    /// 删除，因它与输入中的字面 BOS 无法区分。
    /// llama.cpp 的底层接口返回原始字节；需要该契约时使用 token_bytes。
    pub fn decode(&self, tokens: &[i32]) -> Result<String, String> {
        let mut text = String::new();
        let mut fragment = Vec::new();
        for &id in tokens {
            let index = self.check_id_range(id)?;
            let piece = self.token_bytes(id)?;
            if matches!(
                self.token_types[index],
                TokenType::Control | TokenType::UserDefined | TokenType::Unknown
            ) {
                text.push_str(&self.decode_fragment(&fragment));
                fragment.clear();
                text.push_str(&String::from_utf8_lossy(&piece));
            } else {
                fragment.extend_from_slice(&piece);
            }
        }
        text.push_str(&self.decode_fragment(&fragment));
        Ok(text)
    }

    /// 逐 token 增量解码，语义与 [`Tokenizer::decode`] 完全一致。
    pub fn stream(&self) -> StreamingDecoder<'_> {
        StreamingDecoder {
            tokenizer: self,
            pending: Vec::new(),
            at_fragment_head: true,
        }
    }

    fn decode_fragment(&self, bytes: &[u8]) -> String {
        // 空格标记也可能由三个 byte token 组成，必须在完整片段解码后恢复。
        let text = String::from_utf8_lossy(bytes).replace('▁', " ");
        if self.strips_synthetic_space() {
            text.strip_prefix(' ').unwrap_or(&text).to_string()
        } else {
            text
        }
    }

    /// 片段开头的单个空格是否为 encode 注入的合成前缀（须剥掉）。
    pub(crate) fn strips_synthetic_space(&self) -> bool {
        matches!(self.model, TokenizerModel::Spm { .. }) && self.add_space_prefix
    }

    /// 单个 token 的原始字节；多字节字符可能跨 token，调用方须缓冲后解码。
    pub fn token_bytes(&self, id: i32) -> Result<Vec<u8>, String> {
        let index = self.check_id_range(id)?;
        let token = &self.id_to_token[index];
        match self.token_types.get(index) {
            Some(TokenType::Byte) => byte_value(token)
                .map(|byte| vec![byte])
                .ok_or_else(|| format!("Invalid byte token {token:?}")),
            Some(TokenType::Control | TokenType::UserDefined | TokenType::Unknown) => {
                Ok(token.as_bytes().to_vec())
            }
            _ => Ok(token.replace('▁', " ").into_bytes()),
        }
    }

    /// 按 llama.cpp SPM 的分数优先级合并，相同分数取最左相邻对。
    ///
    /// 对照 b12110 llama-vocab.cpp:91–231：合并候选只要求拼接文本存在于词表，
    /// 不过滤 token_type。这里每轮重扫而非保留优先队列，保持同一排序并避免过期项。
    /// 每次合并结果都在词表中，最终未命中的符号只能是未合并的初始字符，因此可
    /// 直接按原始 UTF-8 字节 fallback，等价于上游 resegment 的叶子分支。
    fn encode_spm_segment(&self, text: &str, scores: &[f32]) -> Result<Vec<i32>, String> {
        let mut normalized = String::new();
        if self.add_space_prefix && !text.is_empty() {
            normalized.push('▁');
        }
        normalized.push_str(&text.replace(' ', "▁"));
        let mut symbols: Vec<String> = normalized.chars().map(|c| c.to_string()).collect();
        loop {
            let mut best: Option<(f32, usize, String)> = None;
            for position in 0..symbols.len().saturating_sub(1) {
                let merged = format!("{}{}", symbols[position], symbols[position + 1]);
                if let Some(&id) = self.token_to_id.get(&merged) {
                    let score = scores[id as usize];
                    if best
                        .as_ref()
                        .is_none_or(|(previous, _, _)| score > *previous)
                    {
                        best = Some((score, position, merged));
                    }
                }
            }
            let Some((_, position, merged)) = best else {
                break;
            };
            symbols[position] = merged;
            symbols.remove(position + 1);
        }
        let mut ids = Vec::with_capacity(symbols.len());
        for symbol in symbols {
            if let Some(&id) = self.token_to_id.get(&symbol) {
                ids.push(id);
            } else {
                for byte in symbol.bytes() {
                    let token = format!("<0x{byte:02X}>");
                    let id = self.token_to_id.get(&token).ok_or_else(|| {
                        format!("Cannot encode SPM symbol {symbol:?}: missing byte token {token}")
                    })?;
                    ids.push(*id);
                }
            }
        }
        Ok(ids)
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
            // 保留既有 merges 接口的 UNK 优先契约；它不具备完整 GPT-2
            // 预分词语义，不能用来声明与上游 GPT-2 tokenizer 对齐。
            // Llama SPM 由独立分支处理，未命中时直接 byte fallback。
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

/// 逐 token 增量解码器，由 [`Tokenizer::stream`] 创建。
///
/// 与一次性 [`Tokenizer::decode`] 共享同一套片段规则：连续普通 token 攒成一个
/// 片段，`▁` 一律还原为空格，SPM 片段开头的合成空格剥掉一次；特殊 token 断开
/// 片段并原样输出字面文本。差别只在时机——跨 token 拆散的多字节字符会留在缓冲里
/// 等后续 token，凑齐一个完整字符才输出，因此流式输出永远不会吐出半个字符。
/// 依次拼接 `push` 的返回值、最后接上 `finish`，结果与 `decode` 逐字符相同。
pub struct StreamingDecoder<'tokenizer> {
    tokenizer: &'tokenizer Tokenizer,
    /// 当前片段中尚未输出的字节。
    pending: Vec<u8>,
    /// 本片段还没解析出过任何字符：决定首个字符要不要当合成空格剥掉。
    at_fragment_head: bool,
}

impl StreamingDecoder<'_> {
    /// 喂入一个 token，返回立即可安全输出的文本（可能为空）。
    pub fn push(&mut self, id: i32) -> Result<String, String> {
        let index = self.tokenizer.check_id_range(id)?;
        let piece = self.tokenizer.token_bytes(id)?;
        if matches!(
            self.tokenizer.token_types[index],
            TokenType::Control | TokenType::UserDefined | TokenType::Unknown
        ) {
            // 特殊 token 自成一节：先把片段收尾，再原样输出标记文本。
            let mut text = self.finish();
            text.push_str(&String::from_utf8_lossy(&piece));
            return Ok(text);
        }
        self.pending.extend_from_slice(&piece);
        Ok(self.drain())
    }

    /// 片段结束：吐出剩余字节，按 `decode` 的 lossy 语义渲染凑不齐的尾部并复位。
    ///
    /// 可重复调用；无待输出内容时返回空串。
    pub fn finish(&mut self) -> String {
        let mut text = self.drain();
        if !self.pending.is_empty() {
            let tail = String::from_utf8_lossy(&self.pending).into_owned();
            self.pending.clear();
            text.push_str(&resolve(
                &tail,
                &mut self.at_fragment_head,
                self.tokenizer.strips_synthetic_space(),
            ));
        }
        self.at_fragment_head = true;
        text
    }

    /// 把缓冲里已能定论的字节转成文本，留下一段可能是后续 token 前缀的尾巴。
    ///
    /// 每轮对 `pending[consumed..]` 重新做一次 UTF-8 校验：`valid_up_to` 之前的
    /// 字节必然合法，先吐出再继续；确定无法构成合法序列的字节（`error_len` 为
    /// `Some(n)`）按 lossy 语义出一个 U+FFFD 并跳过；`error_len` 为 `None` 说明
    /// 剩余字节可能是下一个 token 才补全的前缀，原样保留、结束本轮。缓冲里合法
    /// 段、非法序列、不完整尾三类可能交错出现，所以要逐段推进，不能只截尾部。
    fn drain(&mut self) -> String {
        let strips = self.tokenizer.strips_synthetic_space();
        let mut text = String::new();
        let mut consumed = 0usize;
        loop {
            let rest = &self.pending[consumed..];
            match std::str::from_utf8(rest) {
                Ok(valid) => {
                    text.push_str(&resolve(valid, &mut self.at_fragment_head, strips));
                    consumed += valid.len();
                    break;
                }
                Err(error) if error.valid_up_to() > 0 => {
                    let head = &rest[..error.valid_up_to()];
                    let valid = std::str::from_utf8(head).expect("valid_up_to 之前的字节必然合法");
                    text.push_str(&resolve(valid, &mut self.at_fragment_head, strips));
                    consumed += head.len();
                }
                Err(error) => match error.error_len() {
                    Some(bad) => {
                        text.push_str(&resolve("\u{FFFD}", &mut self.at_fragment_head, strips));
                        consumed += bad;
                    }
                    None => break,
                },
            }
        }
        self.pending.drain(..consumed);
        text
    }
}

/// 还原一截已定论的片段文本：`▁` 换空格，片段开头的合成空格剥掉一次。
///
/// `head` 表示这截是否位于片段开头；一旦吐出过任何字符就转为 false，因此
/// 一个片段最多剥一个空格，与 [`Tokenizer::decode`] 的 `strip_prefix` 一致。
fn resolve(text: &str, head: &mut bool, strips_synthetic_space: bool) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        let resolved = if ch == '▁' { ' ' } else { ch };
        if *head {
            *head = false;
            if resolved == ' ' && strips_synthetic_space {
                continue;
            }
        }
        out.push(resolved);
    }
    out
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

fn byte_value(token: &str) -> Option<u8> {
    let hex = token.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
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
