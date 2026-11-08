//! 分词器：文本 ↔ token id，词表与算法参数来自 GGUF 元数据。

mod vocab;

pub use vocab::{EncodeOptions, StreamingDecoder, TokenType, Tokenizer};
