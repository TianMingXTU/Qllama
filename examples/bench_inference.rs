//! CPU 推理基准：分别测量加载、KV Cache prefill 与固定步数的单 token decode。
//!
//! 分词与 cache 分配不计入前向耗时；generated_ids 首项来自 prefill，后续每项
//! 对应一次计时内的 decode。固定负载不因 EOS 提前停止。

#![forbid(unsafe_code)]

use std::io::{Read, Write};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{ensure, Context};
use clap::Parser;
use qllama::gguf::GgufFile;
use qllama::kv_cache::KvCache;
use qllama::tokenizer::Tokenizer;
use qllama::{Tensor, Transformer};
use serde_json::{json, Value};

#[derive(Debug, Parser)]
#[command(about = "Measure model loading, cached prefill and single-token CPU decode")]
struct Args {
    /// GGUF model file
    #[arg(long)]
    model: PathBuf,

    /// Number of threads in an explicit local Rayon pool (must be positive)
    #[arg(long, default_value = "8")]
    threads: NonZeroUsize,

    /// KV cache capacity; defaults to the model's maximum context
    #[arg(long)]
    ctx_size: Option<NonZeroUsize>,

    /// Text to encode using the model's tokenizer, including its default BOS
    #[arg(long, default_value = "Once upon a time", conflicts_with = "tokens")]
    prompt: String,

    /// Explicit comma-separated token ids; bypasses the tokenizer
    #[arg(long, value_name = "IDS")]
    tokens: Option<String>,

    /// Exactly this many single-token forwards after prefill; EOS is ignored
    #[arg(long, default_value_t = 32)]
    steps: usize,

    /// Flush the final JSON, then wait for a byte on stdin before exiting
    #[arg(long)]
    hold_for_memory: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads.get())
        .build()
        .context("Cannot create the benchmark's Rayon thread pool")?;
    let report = pool.install(|| benchmark(&args))?;

    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, &report)?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    if args.hold_for_memory {
        std::io::stdin()
            .lock()
            .read_exact(&mut [0u8])
            .context("--hold-for-memory requires a byte on stdin to release the benchmark")?;
    }
    Ok(())
}

fn benchmark(args: &Args) -> anyhow::Result<Value> {
    let load_started = Instant::now();
    let gguf = GgufFile::from_path(&args.model)
        .with_context(|| format!("Cannot read model {}", args.model.display()))?;
    let model = Transformer::from_gguf(&gguf)
        .map_err(anyhow::Error::msg)
        .context("Cannot load Transformer weights")?;
    let load_seconds = load_started.elapsed().as_secs_f64();
    let config = model.config();
    let ctx_size = args
        .ctx_size
        .map(NonZeroUsize::get)
        .unwrap_or(config.max_position_embeddings);
    ensure!(
        ctx_size <= config.max_position_embeddings,
        "--ctx-size {ctx_size} exceeds model maximum {}",
        config.max_position_embeddings
    );

    let prompt_ids = match &args.tokens {
        Some(tokens) => parse_tokens(tokens)?,
        None => Tokenizer::from_gguf(&gguf)
            .map_err(anyhow::Error::msg)
            .context("Cannot load tokenizer; use --tokens for explicit ids")?
            .encode(&args.prompt)
            .map_err(anyhow::Error::msg)
            .context("Cannot encode --prompt")?,
    };
    drop(gguf);
    ensure!(
        !prompt_ids.is_empty(),
        "The input must contain at least one token"
    );
    for (index, &id) in prompt_ids.iter().enumerate() {
        ensure!(
            id >= 0 && (id as usize) < config.vocab_size,
            "Input token #{index} ({id}) is outside the vocabulary [0, {})",
            config.vocab_size
        );
    }
    let required_context = prompt_ids
        .len()
        .checked_add(args.steps)
        .context("Prompt length plus --steps overflows the context size")?;
    // 在 prefill 前校验整个负载，避免部分执行后才发现 decode 放不下。
    ensure!(
        required_context <= ctx_size,
        "Context budget exceeded: {} prompt tokens + {} decode steps require \
         {required_context} positions, but --ctx-size is {ctx_size}",
        prompt_ids.len(),
        args.steps
    );
    let mut cache = KvCache::new(
        config.num_layers,
        ctx_size,
        config.num_kv_heads,
        config.head_dim(),
    );

    let prefill_started = Instant::now();
    let logits = model
        .forward_with_cache(&prompt_ids, 0, &mut cache)
        .map_err(anyhow::Error::msg)
        .context("Prefill failed")?;
    let prefill_seconds = prefill_started.elapsed().as_secs_f64();
    let mut next_id = argmax_last(&logits)?;
    drop(logits);
    // 首 token 在计时前选出；decode_steps 始终只数单 token forward。
    let mut generated_ids = vec![next_id];
    let decode_started = Instant::now();
    for step in 0..args.steps {
        let position = prompt_ids.len() + step;
        let logits = model
            .forward_with_cache(&[next_id], position, &mut cache)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("Decode step {} at position {position} failed", step + 1))?;
        next_id = argmax_last(&logits)?;
        generated_ids.push(next_id);
    }
    let decode_seconds = if args.steps == 0 {
        0.0
    } else {
        decode_started.elapsed().as_secs_f64()
    };

    Ok(json!({
        "model": args.model,
        "config": {
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_attention_heads": config.num_attention_heads,
            "num_kv_heads": config.num_kv_heads,
            "head_dim": config.head_dim(),
            "num_layers": config.num_layers,
            "vocab_size": config.vocab_size,
            "max_position_embeddings": config.max_position_embeddings,
            "rms_norm_eps": config.rms_norm_eps,
            "rope_theta": config.rope_theta,
        },
        "threads": rayon::current_num_threads(),
        "ctx_size": ctx_size,
        "prompt_ids": prompt_ids,
        "load_seconds": load_seconds,
        "prefill_seconds": prefill_seconds,
        "prefill_tokens": prompt_ids.len(),
        "prefill_tokens_per_second": tokens_per_second(prompt_ids.len(), prefill_seconds),
        "decode_seconds": decode_seconds,
        "decode_steps": args.steps,
        "decode_tokens_per_second": tokens_per_second(args.steps, decode_seconds),
        "generated_ids": generated_ids,
    }))
}

fn parse_tokens(text: &str) -> anyhow::Result<Vec<i32>> {
    text.split(',')
        .enumerate()
        .map(|(index, piece)| {
            let piece = piece.trim();
            ensure!(!piece.is_empty(), "--tokens entry #{index} is empty");
            piece.parse::<i32>().with_context(|| {
                format!("--tokens entry #{index} ({piece:?}) is not a 32-bit integer")
            })
        })
        .collect()
}

fn argmax_last(logits: &Tensor) -> anyhow::Result<i32> {
    let &[positions, vocab_size] = logits.dims() else {
        anyhow::bail!("Expected 2D logits, got {:?}", logits.dims());
    };
    ensure!(
        positions > 0 && vocab_size > 0,
        "Logits contain no token scores"
    );
    let row = logits
        .data()
        .get((positions - 1) * vocab_size..positions * vocab_size)
        .context("Logits buffer does not match its shape")?;
    let mut best = 0;
    for (id, &value) in row.iter().enumerate() {
        ensure!(
            value.is_finite(),
            "Non-finite logit for token {id}: {value}"
        );
        // 严格大于使平局时选择较小 id，保证 greedy 结果可复现。
        if value > row[best] {
            best = id;
        }
    }
    i32::try_from(best).context("Selected token id exceeds the 32-bit token range")
}

fn tokens_per_second(tokens: usize, seconds: f64) -> f64 {
    if seconds > 0.0 {
        tokens as f64 / seconds
    } else {
        0.0
    }
}
