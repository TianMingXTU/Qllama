use clap::{CommandFactory, Parser};
use std::io::Write;
use std::num::NonZeroUsize;
use std::path::PathBuf;

use qllama::sampler::{Sampler, SamplerConfig};
use qllama::tokenizer::{EncodeOptions, StreamingDecoder, Tokenizer};
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

    /// Run a forward pass over these explicit token ids (e.g. "1,2,3")
    ///
    /// 与 --prompt 二选一。模型缺少分词器元数据时，这是唯一能用的入口。
    #[arg(long, value_name = "IDS")]
    tokens: Option<String>,

    /// Prompt text to encode and continue from
    ///
    /// 需要模型带分词器元数据（tokenizer.ggml.tokens 等）。
    #[arg(short, long, value_name = "TEXT")]
    prompt: Option<String>,

    /// How many new tokens to generate after the prompt
    ///
    /// 采样生成，上限为 N；提前遇到 EOS 会更早停止。
    #[arg(
        long,
        visible_alias = "max-tokens",
        value_name = "N",
        default_value_t = 0
    )]
    generate: usize,

    /// Sampling temperature; 0 means greedy
    #[arg(long, value_name = "F", default_value_t = 0.8)]
    temperature: f32,

    /// Keep only the K highest-probability tokens when sampling; 0 disables the limit
    #[arg(long, value_name = "N", default_value_t = 40)]
    top_k: usize,

    /// Nucleus sampling cutoff; 1.0 disables the limit
    #[arg(long, value_name = "F", default_value_t = 0.95)]
    top_p: f32,

    /// Penalty applied to tokens that already appeared in the context
    #[arg(long, value_name = "F", default_value_t = 1.1)]
    repeat_penalty: f32,

    /// Seed for the sampling RNG; makes the output reproducible
    #[arg(long, value_name = "SEED")]
    seed: Option<u64>,

    /// Context budget in positions; defaults to the model's context_length
    ///
    /// 同时决定 KV cache 容量与生成上限。超出模型标称值会被拒绝。
    #[arg(long, value_name = "N")]
    ctx_size: Option<NonZeroUsize>,

    /// Worker threads for the matrix products; defaults to the CPU count
    ///
    /// 只影响并行划分，不改变每个输出的累加顺序，因此同参数下结果逐位一致。
    #[arg(long, value_name = "N")]
    threads: Option<NonZeroUsize>,

    /// Buffer the whole generated text and print it after generation
    ///
    /// 默认逐 token 流式输出；--logits 与采样参数不受影响。
    #[arg(long)]
    no_stream: bool,

    /// Do not prepend the BOS token to the encoded prompt
    #[arg(long)]
    no_bos: bool,

    /// Write the last position's logits to this file, one per line, token id order
    #[arg(long, value_name = "PATH")]
    logits: Option<PathBuf>,

    /// Print only the inference result, skipping the model dump
    #[arg(short, long)]
    quiet: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // --threads 只影响 matmul 的并行划分，不改累加顺序；显式局部池与
    // bench_inference 一致，避免污染全局池语义。默认路径不动任何既有行为。
    match args.threads {
        Some(threads) => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads.get())
                .build()
                .map_err(anyhow::Error::msg)?;
            pool.install(|| run(&args))
        }
        None => run(&args),
    }
}

fn run(args: &Args) -> anyhow::Result<()> {
    let Some(model_path) = &args.model else {
        // 直接复用 clap 渲染的帮助文本：手写一份 usage 必然与 --help 慢慢不一致。
        Args::command().print_help()?;
        return Ok(());
    };

    // --prompt 与 --tokens 是同一个输入的两个来源，同时给出会让"到底跑了哪段
    // 序列"变得不确定，因此直接拒绝而不是挑一个执行。
    if args.prompt.is_some() && args.tokens.is_some() {
        anyhow::bail!("--prompt and --tokens are mutually exclusive; pass exactly one");
    }

    // --generate/--logits 离开输入就没有意义。返回非零而不是打印提示后
    // 退出 0，否则脚本无法从退出码判断这次调用是否真的做了事。
    if args.prompt.is_none()
        && args.tokens.is_none()
        && (args.generate > 0 || args.logits.is_some())
    {
        anyhow::bail!("--generate and --logits require --prompt <TEXT> or --tokens <IDS>");
    }

    let verbose = !args.quiet;
    if verbose {
        println!("Loading model: {model_path:?}");
    }

    let gguf = qllama::gguf::GgufFile::from_path(model_path)
        .map_err(|error| anyhow::anyhow!("Cannot read model {}: {error}", model_path.display()))?;

    if verbose {
        print_model_info(&gguf);
    }

    if args.info {
        // --info 会短路，后面请求的前向与 logits 文件都不会发生；静默忽略会让
        // 使用者以为已经跑过。这里明说，退出码仍为 0（--info 本身成功了）。
        if args.tokens.is_some()
            || args.prompt.is_some()
            || args.generate > 0
            || args.logits.is_some()
        {
            println!(
                "\n--info short-circuits: no forward pass was run and no logits file was written."
            );
            println!("Drop --info to run the forward pass.");
        }
        return Ok(());
    }

    // 分词器只在需要把文本变成 id 时才加载：--tokens 路径不需要它，模型缺少
    // 分词器元数据时也不该阻断显式 id 的用法。
    let mut tokenizer = None;
    let prompt_ids = match (&args.prompt, &args.tokens) {
        (Some(prompt), None) => {
            let loaded = Tokenizer::from_gguf(&gguf).map_err(|error| {
                anyhow::anyhow!(
                    "Cannot encode --prompt {prompt:?}: {error}. \
                     The model has no usable tokenizer metadata; \
                     pass explicit token ids instead, e.g. --tokens 1,2,3"
                )
            })?;
            let options = EncodeOptions {
                add_bos: !args.no_bos,
                add_eos: false,
            };
            let ids = loaded
                .encode_with_options(prompt, &options)
                .map_err(anyhow::Error::msg)?;
            if ids.is_empty() {
                anyhow::bail!("--prompt {prompt:?} encodes to no tokens");
            }
            tokenizer = Some(loaded);
            ids
        }
        (None, Some(tokens)) => {
            let ids = parse_tokens(tokens)?;
            if ids.is_empty() {
                anyhow::bail!("--tokens is empty; at least one token id is required");
            }
            ids
        }
        (None, None) => {
            println!(
                "\nNothing to run. Pass --prompt \"some text\" or --tokens 1,2,3 to run a forward pass."
            );
            return Ok(());
        }
        (Some(_), Some(_)) => unreachable!("互斥检查已在上游完成"),
    };

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

    // 上下文预算在第一次前向之前算清并校验：跑到一半才发现放不下，前面的
    // token 已经吐给用户了，收都收不回。
    let ctx_size = match args.ctx_size {
        Some(requested) => {
            let requested = requested.get();
            if requested > config.max_position_embeddings {
                anyhow::bail!(
                    "--ctx-size {requested} exceeds the model's context length {}",
                    config.max_position_embeddings
                );
            }
            requested
        }
        None => config.max_position_embeddings,
    };
    if prompt_ids.len() >= ctx_size && args.generate > 0 {
        anyhow::bail!(
            "Context budget exceeded: {} prompt tokens already fill --ctx-size {ctx_size}; \
             there is no room for a new token. Raise --ctx-size (model maximum {}), \
             shorten the prompt, or pass --no-bos.",
            prompt_ids.len(),
            config.max_position_embeddings
        );
    }

    let logits = model.forward(&prompt_ids, 0).map_err(anyhow::Error::msg)?;
    let (last_row, shape) = last_position_row(&logits)?;
    println!("\n=== Forward ===");
    println!("  input ids: {prompt_ids:?}");
    if let Some(tokenizer) = &tokenizer {
        let text = tokenizer.decode(&prompt_ids).map_err(anyhow::Error::msg)?;
        println!("  input text: {text:?}");
    } else {
        // 没有分词器时下面出现的都是 token id 而不是文本；明说一句，避免把 id 当成输出。
        println!("  (no tokenizer metadata in this model; token ids, not text)");
    }
    println!("  logits shape: {shape:?}");

    let summary = logit_summary(last_row);
    println!(
        "  last position (index {}): sum={:.6} min={:.6} max={:.6} argmax={}",
        shape[0] - 1,
        summary.sum,
        summary.min,
        summary.max,
        summary.argmax
    );

    // 按规范仅在未采样时打印前 5 个 logits（数量固定为 5、与 --top-k 解耦，
    // --top-k 现在是采样参数）：采样时惩罚系数会改变实际采到的 token，把
    // 裸 argmax 展示出来容易让人误当作生成依据。
    if args.generate == 0 {
        for (rank, (token, logit)) in top_k(last_row, 5).iter().enumerate() {
            println!("  #{rank:<3} token {token:<8} logit {logit:.6}");
        }
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
        // 放不下的部分收缩到预算：静默少生成会被当成模型输出就这么短，所以
        // 生成结束后用 budget_limited 在停止原因里明说。ctx_size 已校验。
        let steps = args.generate.min(ctx_size - prompt_ids.len());
        let budget_limited = steps < args.generate;

        let sampler_config = SamplerConfig {
            temperature: args.temperature,
            top_k: args.top_k,
            top_p: args.top_p,
            repeat_penalty: args.repeat_penalty,
        };
        let mut sampler = match args.seed {
            Some(seed) => Sampler::with_seed(sampler_config, seed),
            None => Sampler::new(sampler_config),
        };
        // 只有 --prompt 路径才有分词器，也就才知道 EOS 是哪个 id。
        let eos_id = tokenizer.as_ref().and_then(Tokenizer::eos_id);
        // 输出方式三选一：流式（默认，有分词器）、缓冲（--no-stream）、纯 id。
        let mut stream = tokenizer.as_ref().map(Tokenizer::stream);
        let output = match (&mut stream, args.no_stream) {
            (Some(decoder), false) => OutputMode::Stream(decoder),
            (Some(_), true) => OutputMode::Buffered,
            (None, _) => OutputMode::IdsOnly,
        };
        let text_output = output.streams_text();
        if text_output {
            println!("\n=== Generation ===");
            std::io::stdout().flush()?;
        }
        let outcome = generate(
            &model,
            &mut sampler,
            &prompt_ids,
            steps,
            ctx_size,
            eos_id,
            output,
        )?;
        // generate 只见到收缩后的步数；"被预算截断"这个事实只有调用方知道。
        let stop_reason = if outcome.stop_reason == StopReason::ContextBudget
            || (budget_limited && outcome.stop_reason == StopReason::RequestedLimit)
        {
            StopReason::ContextBudget
        } else {
            outcome.stop_reason
        };
        if text_output {
            println!("\n=== Generation summary ===");
        } else {
            println!("\n=== Generation ===");
        }
        println!("  prompt ids: {prompt_ids:?}");
        println!(
            "  generated {} token(s): {:?}",
            outcome.tokens.len(),
            outcome.tokens
        );
        println!("  {stop_reason}");
        if let Some(tokenizer) = &tokenizer {
            if args.no_stream {
                let text = tokenizer
                    .decode(&outcome.tokens)
                    .map_err(anyhow::Error::msg)?;
                println!("  text: {text:?}");
            }
        } else {
            println!("  (no tokenizer metadata; the ids above are the output, not text)");
        }
    }

    Ok(())
}

/// 生成循环结束时可观察到的原因。
#[derive(Clone, Copy, PartialEq, Eq)]
enum StopReason {
    /// 用满 --generate 请求的步数。
    RequestedLimit,
    /// 采到 EOS，提前结束。
    EndOfSequence,
    /// 上下文预算耗尽（收缩后的步数跑完，或 --ctx-size 只够这些）。
    ContextBudget,
}

impl std::fmt::Display for StopReason {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RequestedLimit => {
                write!(formatter, "stopped after the requested --generate limit")
            }
            Self::EndOfSequence => {
                write!(
                    formatter,
                    "stopped at end-of-sequence before the --generate limit"
                )
            }
            Self::ContextBudget => write!(
                formatter,
                "reached the context budget (--ctx-size) before the --generate limit"
            ),
        }
    }
}

struct GenerationOutcome {
    tokens: Vec<i32>,
    stop_reason: StopReason,
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

/// 生成的输出方式：由分词器有无与 --no-stream 共同决定。
///
/// 流式路径每采出一个 token 就立即解码写出；缓冲路径保持旧契约，生成结束后
/// 打印 `text:` 行；id 路径没有分词器，没有文本可吐，只输出 id 摘要。
enum OutputMode<'a> {
    Stream(&'a mut StreamingDecoder<'a>),
    Buffered,
    IdsOnly,
}

impl OutputMode<'_> {
    fn streams_text(&self) -> bool {
        matches!(self, Self::Stream(_))
    }
}

/// 采样生成：prefill 整段 prompt，之后每步只前向一个新 token。
///
/// 两阶段结构对应 KV Cache 的两种用法：
/// 1. prefill：`forward_with_cache(prompt, 0, cache)` 一次算完 prompt 全部
///    位置的 K/V 并写入 cache，取末行 logits 采出第一个 token；
/// 2. decode：之后每步只喂单个新 token，`start_pos` 随已生成长度前进，
///    K/V 与 cache 中的历史拼接——每步代价从 O(n²) 降到 O(n)。
///
/// cache 按 `ctx_size` 预算分配（CLI 已保证不超过模型标称值，也保证
/// prompt.len() + steps 放得下）；容量语义见 `KvCache::new`。
///
/// `eos_id` 给出时，遇到它立即停止且不计入生成结果——llama.cpp 同样不把 EOS
/// 计入输出文本。`None` 表示模型没有 EOS，生成到 `steps` 为止。"steps 被上下文
/// 预算收缩过"只有调用方知道，停止原因由调用方最终裁定；这里只区分
/// EOS 提前停止与跑满请求步数。
fn generate(
    model: &Transformer,
    sampler: &mut Sampler,
    prompt: &[i32],
    steps: usize,
    ctx_size: usize,
    eos_id: Option<i32>,
    mut output: OutputMode<'_>,
) -> anyhow::Result<GenerationOutcome> {
    let config = model.config();
    let mut kv_cache = qllama::kv_cache::KvCache::new(
        config.num_layers,
        ctx_size,
        config.num_kv_heads,
        config.head_dim(),
    );

    let mut ids = prompt.to_vec();
    let mut generated = Vec::with_capacity(steps);
    // 第一批是整段 prompt（位置 0 起），之后每批只有一个新 token。
    let mut batch = prompt.to_vec();
    let mut position = 0usize;
    let mut stop_reason = StopReason::RequestedLimit;
    for _ in 0..steps {
        let logits = model
            .forward_with_cache(&batch, position, &mut kv_cache)
            .map_err(anyhow::Error::msg)?;
        let (last_row, _) = last_position_row(&logits)?;
        // context 传当前完整序列：重复惩罚要看到 prompt 里的每个 token。
        let next = sampler
            .sample_slice(last_row, &ids)
            .map_err(anyhow::Error::msg)?;
        if Some(next) == eos_id {
            stop_reason = StopReason::EndOfSequence;
            break;
        }
        generated.push(next);
        ids.push(next);
        if let OutputMode::Stream(decoder) = &mut output {
            let text = decoder.push(next).map_err(anyhow::Error::msg)?;
            write_stdout(&text)?;
        }
        position += batch.len();
        batch = vec![next];
    }
    if let OutputMode::Stream(decoder) = &mut output {
        // 收尾：把解码器里可能滞留的不完整序列按 lossy 语义吐出，避免结尾
        // 少字。EOS 停止时片段早已 flush，这里只是幂等的第二次 flush。
        let tail = decoder.finish();
        if !tail.is_empty() {
            write_stdout(&tail)?;
        }
    }
    Ok(GenerationOutcome {
        tokens: generated,
        stop_reason,
    })
}

/// 流式正文逐 token 写出并立即 flush；这是"流式"与"结束后打印"的本质区别。
fn write_stdout(text: &str) -> anyhow::Result<()> {
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(text.as_bytes())?;
    stdout.flush()?;
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
