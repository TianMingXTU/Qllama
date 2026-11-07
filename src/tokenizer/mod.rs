//! Tokenizer：文本 ↔ token id（BPE，元数据来自 GGUF）

mod tokenizer;

pub use tokenizer::{EncodeOptions, StreamingDecoder, TokenType, Tokenizer};
