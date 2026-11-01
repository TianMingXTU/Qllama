use clap::{CommandFactory, Parser};
use std::path::PathBuf;

use qllama::transformer::Transformer;

/// Qllama: A Rust-native LLM inference engine
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to GGUF model file
    #[arg(short, long)]
    model: Option<PathBuf>,

    /// Show model information only
    #[arg(long)]
    info: bool,

    /// Run a forward pass over these explicit token ids (e.g. "1,15043,29892")
    #[arg(long, value_name = "IDS")]
    tokens: Option<String>,

    /// Greedily decode this many tokens after the prompt
    #[arg(long, value_name = "N", default_value_t = 0)]
    generate: usize,

    /// How many of the highest logits to print for the last position
    #[arg(long, value_name = "N", default_value_t = 5)]
    top_k: usize,

    /// Write the last position's logits to this file, one per line, token id order
    #[arg(long, value_name = "PATH")]
    logits: Option<PathBuf>,

    /// Print only the inference result, skipping the model dump
    #[arg(short, long)]
    quiet: bool,

    /// Prompt text
    ///
    /// 分词器实现之前它只会报错。从 --help 里隐藏，是为了不向使用者宣传一个
    /// 必然失败的入口；保留参数本身则是为了在真的传进来时给出可操作的提示。
    #[arg(short, long, hide = true)]
    prompt: Option<String>,
}

fn main() -> anyhow::Result<()> {
    // Initialize logger
    env_logger::init();

    let args = Args::parse();

    let Some(model_path) = &args.model else {
        // 直接复用 clap 渲染的帮助文本：手写一份 usage 必然与 --help 慢慢不一致。
        Args::command().print_help()?;
        return Ok(());
    };

    // --generate/--logits 离开 --tokens 就没有意义。返回非零而不是打印提示后
    // 退出 0，否则脚本无法从退出码判断这次调用是否真的做了事。
    if args.tokens.is_none() && (args.generate > 0 || args.logits.is_some()) {
        anyhow::bail!("--generate and --logits require --tokens <IDS>");
    }

    let verbose = !args.quiet;
    if verbose {
        println!("Loading model: {model_path:?}");
    }

    // Parse GGUF file
    let gguf = qllama::gguf::GgufFile::from_path(model_path)
        .map_err(|error| anyhow::anyhow!("Cannot read model {}: {error}", model_path.display()))?;

    // Print model information
    if verbose {
        print_model_info(&gguf);
    }

    if args.info {
        // --info 会短路，后面请求的前向与 logits 文件都不会发生；静默忽略会让
        // 使用者以为已经跑过。这里明说，退出码仍为 0（--info 本身成功了）。
        if args.tokens.is_some() || args.generate > 0 || args.logits.is_some() {
            println!(
                "\n--info short-circuits: no forward pass was run and no logits file was written."
            );
            println!("Drop --info to run the forward pass.");
        }
        return Ok(());
    }

    // 分词器尚未实现，--prompt 只能明确报错；静默忽略会让使用者以为跑过了。
    if let Some(prompt) = &args.prompt {
        anyhow::bail!(
            "Tokenizer is not implemented yet, so --prompt {prompt:?} cannot be encoded. \
             Pass explicit token ids instead, e.g. --tokens 1,15043,29892"
        );
    }

    let Some(tokens) = &args.tokens else {
        println!("\nNothing to run. Pass --tokens 1,2,3 to run a forward pass.");
        return Ok(());
    };

    let prompt_ids = parse_tokens(tokens)?;
    if prompt_ids.is_empty() {
        anyhow::bail!("--tokens is empty; at least one token id is required");
    }

    // 库接口统一用 Result<_, String> 传错误，而 anyhow 不为 String 实现 From，
    // 因此在 CLI 边界集中转换一次。
    let model = Transformer::from_gguf(&gguf).map_err(anyhow::Error::msg)?;
    let config = model.config();
    // 标签统一左对齐：逐项写死空格对齐会在增删字段后整体错位。
    let fields = [
        ("hidden_size", config.hidden_size.to_string()),
        ("intermediate_size", config.intermediate_size.to_string()),
        (
            "num_attention_heads",
            config.num_attention_heads.to_string(),
        ),
        ("num_kv_heads", config.num_kv_heads.to_string()),
        ("head_dim", config.head_dim().to_string()),
        ("num_layers", config.num_layers.to_string()),
        ("vocab_size", config.vocab_size.to_string()),
        (
            "max_position_embeddings",
            config.max_position_embeddings.to_string(),
        ),
        ("rms_norm_eps", config.rms_norm_eps.to_string()),
        ("rope_theta", config.rope_theta.to_string()),
    ];
    if verbose {
        println!("\n=== Configuration ===");
        for (label, value) in fields {
            println!("  {label:<24}{value}");
        }
    }

    let logits = model.forward(&prompt_ids, 0).map_err(anyhow::Error::msg)?;
    let (last_row, shape) = last_position_row(&logits)?;
    println!("\n=== Forward ===");
    println!("  input ids: {prompt_ids:?}");
    println!("  logits shape: {shape:?}");
    // 分词器未实现，下面出现的都是 token id 而不是文本；明说一句，避免把 id 当成输出。
    println!("  (tokenizer not implemented yet; token ids, not text)");

    let summary = logit_summary(last_row);
    println!(
        "  last position (index {}): sum={:.6} min={:.6} max={:.6} argmax={}",
        shape[0] - 1,
        summary.sum,
        summary.min,
        summary.max,
        summary.argmax
    );

    for (rank, (token, logit)) in top_k(last_row, args.top_k).iter().enumerate() {
        println!("  #{rank:<3} token {token:<8} logit {logit:.6}");
    }

    if let Some(path) = &args.logits {
        let body: String = last_row
            .iter()
            .map(|value| format!("{value:.9e}\n"))
            .collect();
        std::fs::write(path, body)?;
        // 说清格式：否则使用者要把科学计数法猜成概率还是 log 概率。
        println!(
            "  wrote {} logits to {path:?} (one per line, token id order, {:.9e} format)",
            last_row.len(),
            last_row.first().copied().unwrap_or(0.0)
        );
    }

    if args.generate > 0 {
        let generated = greedy_decode(&model, &prompt_ids, args.generate)?;
        println!("\n=== Greedy generation ===");
        println!("  generated {} token(s): {generated:?}", generated.len());
    }

    Ok(())
}

/// 解析 token id 列表：含逗号时按逗号切，否则按空白切。
///
/// 两种分隔符都接受，因为复制粘贴出来的 id 串两种样子都有。逗号分支对空字段
/// 报错——静默跳过会让前向跑在与使用者所写不同的序列上；空白分支容忍连续空白，
/// 因为手打的空串不表达任何意图。报错时给出序号与原文，而不是笼统的
/// "invalid digit"。
fn parse_tokens(text: &str) -> anyhow::Result<Vec<i32>> {
    let pieces: Vec<&str> = if text.contains(',') {
        text.split(',').collect()
    } else {
        text.split_whitespace().collect()
    };

    let mut ids = Vec::with_capacity(pieces.len());
    for (index, piece) in pieces.iter().enumerate() {
        let piece = piece.trim();
        if piece.is_empty() {
            anyhow::bail!("token #{index} is empty; separate ids with a single ','");
        }
        ids.push(piece.parse::<i32>().map_err(|_| {
            anyhow::anyhow!("token #{index} ({piece:?}) is not a valid 32-bit integer")
        })?);
    }
    Ok(ids)
}

/// 取出 logits 最后一行的切片与整体形状。
///
/// 直接按行宽切 `data()`，省掉一次 `slice` + `reshape` 拷贝。
fn last_position_row(logits: &qllama::tensor::Tensor) -> anyhow::Result<(&[f32], Vec<usize>)> {
    let shape = logits.dims().to_vec();
    let [seq_len, vocab_size] = shape.as_slice() else {
        anyhow::bail!("logits must be 2D [seq_len, vocab_size], got {shape:?}");
    };
    if *seq_len == 0 {
        anyhow::bail!("logits have no positions");
    }
    let start = (seq_len - 1) * vocab_size;
    let row = logits
        .data()
        .get(start..start + vocab_size)
        .ok_or_else(|| anyhow::anyhow!("logits buffer is shorter than its shape {shape:?}"))?;
    Ok((row, shape))
}

struct LogitSummary {
    sum: f64,
    min: f32,
    max: f32,
    argmax: usize,
}

/// 汇总一行 logits 的统计量，用于跨实现对照。
///
/// `sum` 用 f64 累加：f32 累加到几万个元素时舍入误差会超过实现间差异，
/// 反而掩盖真实偏差。
fn logit_summary(row: &[f32]) -> LogitSummary {
    let mut sum = 0.0f64;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut argmax = 0usize;
    for (index, &value) in row.iter().enumerate() {
        sum += f64::from(value);
        min = min.min(value);
        if value > max {
            max = value;
            argmax = index;
        }
    }
    LogitSummary {
        sum,
        min,
        max,
        argmax,
    }
}

/// 取 logits 最大的前 `k` 个 `(token_id, logit)`。
///
/// 并列时按 id 升序，保证输出可复现；`total_cmp` 让 NaN 也有确定次序。
fn top_k(row: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indices: Vec<usize> = (0..row.len()).collect();
    indices.sort_by(|&left, &right| {
        row[right]
            .total_cmp(&row[left])
            .then_with(|| left.cmp(&right))
    });
    indices
        .into_iter()
        .take(k)
        .map(|index| (index, row[index]))
        .collect()
}

/// 贪心解码：每步取上一轮最后位置的 argmax 作为下一 token。
///
/// 无 KV Cache，因此每步都重新前向整段序列，代价是 O(n²)；仅作端到端联调，
/// 接上 KV Cache 后改为增量前向。
fn greedy_decode(model: &Transformer, prompt: &[i32], steps: usize) -> anyhow::Result<Vec<i32>> {
    let mut ids = prompt.to_vec();
    let mut generated = Vec::with_capacity(steps);
    for _ in 0..steps {
        let logits = model.forward(&ids, 0).map_err(anyhow::Error::msg)?;
        let (last_row, _) = last_position_row(&logits)?;
        let next = logit_summary(last_row).argmax;
        let next = i32::try_from(next)
            .map_err(|_| anyhow::anyhow!("argmax {next} does not fit in i32"))?;
        generated.push(next);
        ids.push(next);
    }
    Ok(generated)
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
