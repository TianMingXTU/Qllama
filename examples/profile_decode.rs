//! 用真实量化模型分解增量前向耗时；预热和 prefill 不进入 decode 分解表。

#![forbid(unsafe_code)]

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{ensure, Context};
use clap::Parser;
use qllama::kv_cache::KvCache;
use qllama::profile::{self, Stage};
use qllama::tokenizer::Tokenizer;
use qllama::{GgufFile, Tensor, Transformer};
use serde_json::json;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value = "8")]
    threads: NonZeroUsize,
    #[arg(long, default_value_t = 2048)]
    ctx_size: usize,
    #[arg(long, default_value_t = 16)]
    steps: usize,
    #[arg(long, default_value_t = 3)]
    warmup: usize,
    #[arg(long, default_value_t = 5)]
    runs: usize,
    #[arg(long, default_value = "Once upon a time")]
    prompt: String,
    /// 只分解单层 attention；历史为合成 K/V，实际占满指定上下文。
    #[arg(long)]
    attention_only: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    ensure!(args.steps > 0, "--steps must be positive");
    ensure!(
        args.warmup >= 3 && args.runs >= 5,
        "Require >=3 warmups and >=5 measured runs"
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads.get())
        .build()?;
    pool.install(|| run(&args))
}

fn run(args: &Args) -> anyhow::Result<()> {
    let gguf = GgufFile::from_path(&args.model).map_err(anyhow::Error::msg)?;
    let model = Transformer::from_gguf(&gguf).map_err(anyhow::Error::msg)?;
    ensure!(
        args.ctx_size > 0 && args.ctx_size <= model.config().max_position_embeddings,
        "Invalid context capacity"
    );
    if args.attention_only {
        return profile_attention(args, &model);
    }
    let ids = Tokenizer::from_gguf(&gguf)
        .map_err(anyhow::Error::msg)?
        .encode(&args.prompt)
        .map_err(anyhow::Error::msg)?;
    let config = model.config();
    ensure!(!ids.is_empty(), "Prompt must encode at least one token");
    ensure!(
        args.ctx_size <= config.max_position_embeddings,
        "Context exceeds model maximum"
    );
    ensure!(
        ids.len()
            .checked_add(args.steps)
            .is_some_and(|n| n <= args.ctx_size),
        "Prompt plus decode steps exceeds context"
    );
    let mut cache = KvCache::new(
        config.num_layers,
        args.ctx_size,
        config.num_kv_heads,
        config.head_dim(),
    );
    let mut rows = Vec::new();
    for run in 0..args.warmup + args.runs {
        cache.clear();
        let prefill = model
            .forward_with_cache(&ids, 0, &mut cache)
            .map_err(anyhow::Error::msg)?;
        let mut token = argmax(&prefill)?;
        let measured = run >= args.warmup;
        if measured {
            profile::start(config.num_layers);
        }
        let started = Instant::now();
        for step in 0..args.steps {
            let logits = model
                .forward_with_cache(&[token], ids.len() + step, &mut cache)
                .map_err(anyhow::Error::msg)?;
            token = argmax(&logits)?;
        }
        let wall_seconds = started.elapsed().as_secs_f64();
        if measured {
            rows.push(snapshot_row(
                profile::finish(),
                run - args.warmup + 1,
                args.steps,
                wall_seconds,
            ));
        }
        eprintln!(
            "profile run {}/{} complete (warmup={})",
            run + 1,
            args.warmup + args.runs,
            !measured
        );
    }
    println!(
        "{}",
        serde_json::to_string(&json!({
            "model": args.model,
            "threads": args.threads.get(),
            "ctx_size": args.ctx_size,
            "prompt_ids": ids,
            "history_start": ids.len(),
            "history_end": ids.len() + args.steps,
            "temperature": 0,
            "warmup": args.warmup,
            "runs": rows,
            "allocation_scope": "instrumented Qllama tensor/Arc/metadata/temporary Vec allocations; excludes Rayon and allocator internals",
        }))?
    );
    Ok(())
}

fn snapshot_row(
    snapshot: profile::Snapshot,
    run: usize,
    steps: usize,
    wall_seconds: f64,
) -> serde_json::Value {
    let mut stages = Vec::new();
    let mut allocations = snapshot.model.allocations;
    for (layer, timings) in std::iter::once((None, &snapshot.model)).chain(
        snapshot
            .layers
            .iter()
            .enumerate()
            .map(|(index, timings)| (Some(index), timings)),
    ) {
        for stage in Stage::ALL {
            let index = stage as usize;
            if timings.calls[index] != 0 {
                stages.push(json!({
                    "layer": layer,
                    "stage": stage.name(),
                    "nanoseconds": timings.nanoseconds[index] as u64,
                    "calls": timings.calls[index],
                }));
            }
        }
        if layer.is_some() {
            allocations += timings.allocations;
        }
    }
    json!({
        "run": run,
        "decode_steps": steps,
        "wall_seconds": wall_seconds,
        "allocations": allocations,
        "allocations_per_layer": snapshot.layers.iter().map(|v| v.allocations).collect::<Vec<_>>(),
        "stages": stages,
    })
}

fn profile_attention(args: &Args, model: &Transformer) -> anyhow::Result<()> {
    let config = model.config();
    let history = args.ctx_size - 1;
    let positions = (0..history)
        .map(|i| i32::try_from(i).context("Position exceeds i32"))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let shape = vec![history, config.num_kv_heads, config.head_dim()];
    let count: usize = shape.iter().product();
    let k = Tensor::new(
        (0..count)
            .map(|i| (i % 251) as f32 / 512.0 - 0.25)
            .collect(),
        shape.clone(),
    );
    let v = Tensor::new(
        (0..count)
            .map(|i| (i % 239) as f32 / 512.0 - 0.25)
            .collect(),
        shape,
    );
    let hidden = Tensor::new(
        (0..config.hidden_size)
            .map(|i| (i % 127) as f32 / 512.0)
            .collect(),
        vec![1, config.hidden_size],
    );
    let attention = model.attention_layer(0).context("No layer 0")?;
    let mut cache = KvCache::new(1, args.ctx_size, config.num_kv_heads, config.head_dim());
    let mut rows = Vec::new();
    for run in 0..args.warmup + args.runs {
        cache.clear();
        cache
            .update(0, &k, &v, &positions)
            .map_err(anyhow::Error::msg)?;
        let measured = run >= args.warmup;
        if measured {
            profile::start(1);
        }
        let started = Instant::now();
        let output = attention
            .forward_with_cache(&hidden, history, &mut cache, 0)
            .map_err(anyhow::Error::msg)?;
        std::hint::black_box(&output);
        let elapsed = started.elapsed().as_secs_f64();
        if measured {
            rows.push(snapshot_row(
                profile::finish(),
                run - args.warmup + 1,
                1,
                elapsed,
            ));
        }
        ensure!(
            output.data().iter().all(|v| v.is_finite()),
            "Non-finite attention output"
        );
    }
    println!(
        "{}",
        json!({
            "kind": "single_layer_attention_synthetic_history",
            "model": args.model,
            "threads": args.threads.get(),
            "ctx_size": args.ctx_size,
            "actual_history_tokens": history,
            "warmup": args.warmup,
            "runs": rows,
            "note": "Real layer weights; synthetic RoPE-space K, V and hidden states. Not full-model long-prompt decode throughput.",
        })
    );
    Ok(())
}

fn argmax(logits: &Tensor) -> anyhow::Result<i32> {
    let width = *logits.dims().last().context("Missing logits dimension")?;
    let row = &logits.data()[logits.len() - width..];
    let mut best = 0;
    for (id, &value) in row.iter().enumerate() {
        ensure!(value.is_finite(), "Non-finite logit for id {id}");
        if value > row[best] {
            best = id;
        }
    }
    i32::try_from(best).context("Token id exceeds i32")
}
