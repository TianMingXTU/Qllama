//! 批量分词验证工具：stdin 为 JSON 字符串数组，stdout 为编码与解码结果。

use qllama::gguf::GgufFile;
use qllama::tokenizer::{EncodeOptions, Tokenizer};
use serde_json::json;

fn main() -> anyhow::Result<()> {
    let model = std::env::args_os().nth(1).ok_or_else(|| {
        anyhow::anyhow!("usage: cargo run --example tokenize -- <model.gguf> < prompts.json")
    })?;
    let gguf = GgufFile::from_path(model)?;
    let tokenizer = Tokenizer::from_gguf(&gguf).map_err(anyhow::Error::msg)?;
    let prompts: Vec<String> = serde_json::from_reader(std::io::stdin().lock())?;
    let mut results = Vec::new();
    for prompt in prompts {
        let ids = tokenizer.encode(&prompt).map_err(anyhow::Error::msg)?;
        let plain = tokenizer
            .encode_with_options(
                &prompt,
                &EncodeOptions {
                    add_bos: false,
                    add_eos: false,
                },
            )
            .map_err(anyhow::Error::msg)?;
        let decoded = tokenizer.decode(&plain).map_err(anyhow::Error::msg)?;
        let raw_bytes = plain
            .iter()
            .map(|&id| tokenizer.token_bytes(id))
            .collect::<Result<Vec<_>, _>>()
            .map_err(anyhow::Error::msg)?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        results.push(json!({"prompt": prompt, "ids": ids, "plain_ids": plain, "decoded": decoded, "raw_bytes": raw_bytes}));
    }
    println!("{}", serde_json::to_string(&results)?);
    Ok(())
}
