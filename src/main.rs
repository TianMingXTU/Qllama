use clap::Parser;
use std::path::PathBuf;

/// Qllama: A Rust-native LLM inference engine
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to GGUF model file
    #[arg(short, long)]
    model: Option<PathBuf>,

    /// Prompt text
    #[arg(short, long)]
    prompt: Option<String>,

    /// Maximum tokens to generate
    #[arg(long, default_value = "256")]
    max_tokens: usize,

    /// Temperature for sampling
    #[arg(long, default_value = "0.8")]
    temperature: f32,

    /// Show model information only
    #[arg(long)]
    info: bool,
}

fn main() -> anyhow::Result<()> {
    // Initialize logger
    env_logger::init();

    let args = Args::parse();

    if let Some(model_path) = &args.model {
        println!("Loading model: {:?}", model_path);

        // Parse GGUF file
        let gguf = qllama::gguf::GgufFile::from_path(model_path)?;

        // Print model information
        print_model_info(&gguf);

        if args.info {
            return Ok(());
        }

        // TODO: Load model and run inference
        println!("\nModel loading and inference not yet implemented.");
    } else {
        println!("Qllama: A Rust-native LLM inference engine");
        println!("\nUsage:");
        println!("  qllama --model <model.gguf> [--prompt <text>]");
        println!("\nOptions:");
        println!("  --info    Show model information only");
        println!("\nExample:");
        println!("  qllama --model llama-2-7b.gguf --prompt \"Hello, world!\"");
    }

    Ok(())
}

/// 单条 metadata 值的紧凑展示。
///
/// 大数组只给出元素个数与首个元素样例：真实模型的 `tokenizer.ggml.tokens`
/// 动辄数万项，完整展开会淹没其余配置。
fn format_value(value: &qllama::gguf::GgufValue) -> String {
    match value {
        qllama::gguf::GgufValue::Array(items) => match items.first() {
            Some(first) => format!("[{} items, e.g. {:?}]", items.len(), first),
            None => "[]".to_string(),
        },
        other => format!("{other:?}"),
    }
}

/// Print model information
fn print_model_info(gguf: &qllama::gguf::GgufFile) {
    println!("\n=== Model Information ===");
    println!("Version: {}", gguf.header.version);
    println!("Tensors: {}", gguf.tensor_count());
    println!("Metadata entries: {}", gguf.metadata_count());

    // 键名本身带命名空间（llama.context_length、tokenizer.ggml.tokens 等），
    // 按 '.' 过滤会把真实模型的全部配置滤掉；改为按条数截断控制输出量。
    // HashMap 的迭代顺序不固定，排序后输出才可复现。
    const MAX_METADATA_ENTRIES: usize = 40;
    println!("\n=== Metadata ===");
    let mut keys: Vec<&String> = gguf.metadata.keys().collect();
    keys.sort();
    for key in keys.iter().take(MAX_METADATA_ENTRIES) {
        println!("  {}: {}", key, format_value(&gguf.metadata[*key]));
    }
    let shown = keys.len().min(MAX_METADATA_ENTRIES);
    if keys.len() > shown {
        println!("  ... and {} more", keys.len() - shown);
    }

    // Print first few tensor names
    println!("\n=== Tensors (first 10) ===");
    for (i, tensor) in gguf.tensor_infos.iter().enumerate() {
        if i >= 10 {
            println!("  ... and {} more", gguf.tensor_count() as usize - 10);
            break;
        }
        println!("  {} {:?} {:?}", tensor.name, tensor.dims, tensor.ggml_type);
    }
}
