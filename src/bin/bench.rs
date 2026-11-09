//! CPU 基准：真实模型增量推理，以及使用真实压缩权重的算子诊断。

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::hint::black_box;
use std::io::{BufWriter, Write};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{ensure, Context};
use clap::{Parser, ValueEnum};
use qllama::gguf::{GgmlType, GgufFile, MappedWeight};
use qllama::kv_cache::KvCache;
use qllama::tokenizer::Tokenizer;
use qllama::transformer::Linear;
use qllama::{Tensor, Transformer};

const QUERY_WEIGHT: &str = "blk.0.attn_q.weight";
const DEQUANT_ROWS: usize = 32;
const HASH_SEED: u64 = 0xcbf2_9ce4_8422_2325;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Mode {
    Inference,
    Kernels,
    All,
}

#[derive(Debug, Parser)]
#[command(about = "Emit raw CPU benchmark samples as TSV; requires the bench feature")]
struct Args {
    /// GGUF model file
    #[arg(long)]
    model: PathBuf,

    /// Worker count in an explicit local Rayon pool
    #[arg(long, default_value = "1")]
    threads: NonZeroUsize,

    /// Cache capacity; defaults to the model's maximum context
    #[arg(long)]
    ctx_size: Option<NonZeroUsize>,

    /// Exactly this many NEW ids; the first comes from prefill, EOS is ignored
    #[arg(long, default_value = "128")]
    steps: NonZeroUsize,

    /// Prompt to encode, including the tokenizer's default BOS
    #[arg(long, default_value = "Once upon a time", conflicts_with = "tokens")]
    prompt: String,

    /// Explicit comma-separated ids; bypasses the tokenizer
    #[arg(long, value_name = "IDS")]
    tokens: Option<String>,

    /// Full warmup runs for every metric (at least 3)
    #[arg(long, default_value_t = 3)]
    warmup: usize,

    /// Full measured runs for every metric (at least 5)
    #[arg(long, default_value_t = 5)]
    runs: usize,

    /// Inference, synthetic-activation kernel diagnostics, or both
    #[arg(long, value_enum, default_value = "all")]
    mode: Mode,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    ensure!(args.warmup >= 3, "--warmup must be at least 3");
    ensure!(args.runs >= 5, "--runs must be at least 5");
    args.warmup
        .checked_add(args.runs)
        .context("--warmup plus --runs overflows usize")?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads.get())
        .build()
        .context("Cannot create the benchmark's Rayon thread pool")?;
    pool.install(|| benchmark(&args))
}

fn benchmark(args: &Args) -> anyhow::Result<()> {
    let gguf = GgufFile::from_path(&args.model)
        .with_context(|| format!("Cannot read model {}", args.model.display()))?;
    let model = Transformer::from_gguf(&gguf).map_err(anyhow::Error::msg)?;
    let config = model.config();
    let prompt = match &args.tokens {
        Some(text) => parse_tokens(text)?,
        None => Tokenizer::from_gguf(&gguf)
            .map_err(anyhow::Error::msg)
            .context("Cannot load tokenizer; pass --tokens for explicit ids")?
            .encode(&args.prompt)
            .map_err(anyhow::Error::msg)?,
    };
    ensure!(
        !prompt.is_empty(),
        "The prompt must contain at least one id"
    );
    for (index, &id) in prompt.iter().enumerate() {
        ensure!(
            id >= 0 && (id as usize) < config.vocab_size,
            "Prompt id #{index} ({id}) is outside [0, {})",
            config.vocab_size
        );
    }
    let ctx_size = args
        .ctx_size
        .map(NonZeroUsize::get)
        .unwrap_or(config.max_position_embeddings);
    ensure!(
        ctx_size <= config.max_position_embeddings,
        "--ctx-size {ctx_size} exceeds model maximum {}",
        config.max_position_embeddings
    );
    ensure!(
        i32::try_from(ctx_size - 1).is_ok(),
        "--ctx-size exceeds the cache's i32 position range"
    );
    let required_context = prompt
        .len()
        .checked_add(args.steps.get() - 1)
        .context("Prompt length plus --steps - 1 overflows usize")?;
    // 首个新 id 来自 prefill；只有其余 id 需要占用新的前向位置。
    ensure!(
        required_context <= ctx_size,
        "Context budget exceeded: {} prompt tokens + {} decode forwards require \
         {required_context} positions, but --ctx-size is {ctx_size}; no truncation is performed",
        prompt.len(),
        args.steps.get() - 1
    );

    // 算子前提也在任何计时之前检查，避免 all 模式交付半份成功结果。
    let weight = if matches!(args.mode, Mode::Kernels | Mode::All) {
        let weight = gguf.load_weight(QUERY_WEIGHT).map_err(anyhow::Error::msg)?;
        ensure!(
            matches!(weight.dtype(), GgmlType::Q4_0 | GgmlType::Q8_0),
            "Kernel diagnostics require Q4_0/Q8_0 {QUERY_WEIGHT}, got {:?}",
            weight.dtype()
        );
        ensure!(
            weight.dims().len() == 2 && weight.dims()[0] >= DEQUANT_ROWS,
            "{QUERY_WEIGHT} must be a matrix with at least {DEQUANT_ROWS} rows"
        );
        ensure!(
            model.attention_layer(0).is_some(),
            "Model has no attention layer 0"
        );
        Some(weight)
    } else {
        None
    };

    let mut report = Reporter {
        args,
        ctx_size,
        prompt_tokens: prompt.len(),
        output: BufWriter::new(std::io::stdout()),
        checksums: HashMap::new(),
    };
    writeln!(report.output, "#\tschema_version\t1")?;
    writeln!(
        report.output,
        "#\tchecksum_algorithm\tfnv1a64_little_endian"
    )?;
    writeln!(
        report.output,
        "#\tmodel_max_context\t{}",
        config.max_position_embeddings
    )?;
    writeln!(report.output, "#\tvocab_size\t{}", config.vocab_size)?;
    writeln!(report.output, "#\thidden_size\t{}", config.hidden_size)?;
    writeln!(report.output, "#\tnum_layers\t{}", config.num_layers)?;
    writeln!(report.output, "#\tnum_kv_heads\t{}", config.num_kv_heads)?;
    writeln!(report.output, "#\thead_dim\t{}", config.head_dim())?;
    report.ids("prompt_ids", &prompt)?;
    writeln!(report.output, "metric\tthreads\tctx_size\tprompt_tokens\tgenerated_tokens\tdecode_steps\trun\twarmup\tseconds\tunits\tchecksum")?;

    if matches!(args.mode, Mode::Inference | Mode::All) {
        run_inference(&model, &prompt, &mut report)?;
    }
    if let Some(weight) = weight {
        run_kernels(&model, &weight, &mut report)?;
    }
    report.output.flush()?;
    Ok(())
}

struct Reporter<'a> {
    args: &'a Args,
    ctx_size: usize,
    prompt_tokens: usize,
    output: BufWriter<std::io::Stdout>,
    checksums: HashMap<&'static str, u64>,
}

impl Reporter<'_> {
    fn record(
        &mut self,
        metric: &'static str,
        run: usize,
        seconds: f64,
        units: usize,
        checksum: u64,
    ) -> anyhow::Result<()> {
        ensure!(
            self.checksums
                .get(metric)
                .is_none_or(|expected| *expected == checksum),
            "{metric} output changed on run {run}"
        );
        self.checksums.insert(metric, checksum);
        let (prompt_tokens, generated_tokens, decode_steps) = match metric {
            "prefill" | "decode" => (
                self.prompt_tokens,
                self.args.steps.get(),
                self.args.steps.get() - 1,
            ),
            _ => (0, 0, 0),
        };
        writeln!(
            self.output,
            "{metric}\t{}\t{}\t{prompt_tokens}\t{generated_tokens}\t{decode_steps}\t{run}\t{}\t{seconds:.12e}\t{units}\t{checksum:016x}",
            self.args.threads,
            self.ctx_size,
            u8::from(run <= self.args.warmup),
        )?;
        Ok(())
    }

    fn tensor(
        &mut self,
        metric: &'static str,
        run: usize,
        units: usize,
        operation: impl FnOnce() -> Result<Tensor, String>,
    ) -> anyhow::Result<()> {
        let started = Instant::now();
        let output = operation().map_err(anyhow::Error::msg)?;
        black_box(&output);
        let seconds = started.elapsed().as_secs_f64();
        self.record(metric, run, seconds, units, tensor_checksum(&output)?)
    }

    fn ids(&mut self, name: &str, ids: &[i32]) -> anyhow::Result<()> {
        let text = ids.iter().map(i32::to_string).collect::<Vec<_>>().join(",");
        writeln!(self.output, "#\t{name}\t{text}")?;
        Ok(())
    }
}

fn run_inference(
    model: &Transformer,
    prompt: &[i32],
    report: &mut Reporter<'_>,
) -> anyhow::Result<()> {
    writeln!(
        report.output,
        "#\tinference_timing\tforward_and_finite_greedy_argmax"
    )?;
    let config = model.config();
    let mut cache = KvCache::new(
        config.num_layers,
        report.ctx_size,
        config.num_kv_heads,
        config.head_dim(),
    );
    let mut expected_ids = None;
    for run in 1..=report.args.warmup + report.args.runs {
        cache.clear();
        let mut generated = Vec::with_capacity(report.args.steps.get());
        let started = Instant::now();
        let logits = model
            .forward_with_cache(black_box(prompt), 0, &mut cache)
            .map_err(anyhow::Error::msg)?;
        let mut next_id = argmax_last(&logits)?;
        let prefill_seconds = started.elapsed().as_secs_f64();
        generated.push(next_id);
        let prefill_checksum = tensor_checksum(&logits)?;
        drop(logits);

        let started = Instant::now();
        for step in 0..report.args.steps.get() - 1 {
            let logits = model
                .forward_with_cache(&[next_id], prompt.len() + step, &mut cache)
                .map_err(anyhow::Error::msg)
                .with_context(|| format!("Decode forward {} failed", step + 1))?;
            next_id = argmax_last(&logits)?;
            generated.push(next_id);
        }
        let decode_steps = report.args.steps.get() - 1;
        let decode_seconds = if decode_steps == 0 {
            0.0
        } else {
            started.elapsed().as_secs_f64()
        };
        ensure!(
            cache.seq_len() == prompt.len() + decode_steps,
            "Cache history length mismatch"
        );
        match &expected_ids {
            Some(expected) => ensure!(*expected == generated, "Generated ids changed on run {run}"),
            None => {
                report.ids("generated_ids", &generated)?;
                expected_ids = Some(generated.clone());
            }
        }
        let checksum = hash_bytes(HASH_SEED, generated.iter().flat_map(|id| id.to_le_bytes()));
        report.record(
            "prefill",
            run,
            prefill_seconds,
            prompt.len(),
            prefill_checksum,
        )?;
        report.record("decode", run, decode_seconds, decode_steps, checksum)?;
        report.output.flush()?;
    }
    Ok(())
}

fn run_kernels(
    model: &Transformer,
    weight: &MappedWeight,
    report: &mut Reporter<'_>,
) -> anyhow::Result<()> {
    let (rows, width) = (weight.dims()[0], weight.dims()[1]);
    // 只扩张这一张 Q 矩阵作算子对照，不构造稠密模型或替代生产前向。
    let mut expanded = Vec::with_capacity(rows * width);
    for row in 0..rows {
        expanded.extend(weight.read_row(row).map_err(anyhow::Error::msg)?);
    }
    let dense =
        Linear::new(Tensor::new(expanded, vec![rows, width]), None).map_err(anyhow::Error::msg)?;
    let vector = synthetic(vec![1, width], 11)?;
    let matrix = synthetic(vec![8, width], 11)?;
    let selected_rows: Vec<usize> = (0..DEQUANT_ROWS)
        .map(|index| index * (rows / DEQUANT_ROWS))
        .collect();
    let config = model.config();
    let history_len = report.ctx_size - 1;
    let positions = (0..history_len)
        .map(|position| position as i32)
        .collect::<Vec<_>>();
    // 历史 K 已视为 RoPE 空间的合成数据；该测量不代表真实 prompt 的激活分布。
    let history_k = synthetic(
        vec![history_len, config.num_kv_heads, config.head_dim()],
        29,
    )?;
    let history_v = synthetic(
        vec![history_len, config.num_kv_heads, config.head_dim()],
        71,
    )?;
    let hidden = synthetic(vec![1, config.hidden_size], 11)?;
    let attention = model
        .attention_layer(0)
        .context("Model has no attention layer 0")?;
    let mut cache = KvCache::new(1, report.ctx_size, config.num_kv_heads, config.head_dim());

    writeln!(report.output, "#\tmatrix_tensor\t{QUERY_WEIGHT}")?;
    writeln!(report.output, "#\tmatrix_dtype\t{:?}", weight.dtype())?;
    writeln!(report.output, "#\tmatrix_rows\t{rows}")?;
    writeln!(report.output, "#\tmatrix_columns\t{width}")?;
    writeln!(
        report.output,
        "#\tkernel_activations\tsynthetic_deterministic"
    )?;
    writeln!(
        report.output,
        "#\tattention_history\tsynthetic_rope_space_k_and_synthetic_v"
    )?;
    writeln!(report.output, "#\tattention_history_tokens\t{history_len}")?;
    writeln!(report.output, "#\tattention_query_position\t{history_len}")?;
    writeln!(report.output, "#\tdense_scope\tselected_matrix_kernel_only")?;
    writeln!(
        report.output,
        "#\tdequantization_includes\tread_row_allocation"
    )?;
    let selected = selected_rows
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    writeln!(report.output, "#\tdequantization_rows\t{selected}")?;
    for run in 1..=report.args.warmup + report.args.runs {
        report.tensor("mapped_gemv", run, 1, || weight.matmul(black_box(&vector)))?;
        report.tensor("dense_gemv", run, 1, || dense.forward(black_box(&vector)))?;
        report.tensor("mapped_gemm", run, 8, || weight.matmul(black_box(&matrix)))?;
        report.tensor("dense_gemm", run, 8, || dense.forward(black_box(&matrix)))?;

        cache.clear();
        cache
            .update(0, &history_k, &history_v, &positions)
            .map_err(anyhow::Error::msg)?;
        report.tensor("attention_cached", run, 1, || {
            attention.forward_with_cache(black_box(&hidden), history_len, &mut cache, 0)
        })?;
        ensure!(
            cache.seq_len() == report.ctx_size,
            "Attention did not use full cache capacity"
        );

        let mut decoded = Vec::with_capacity(DEQUANT_ROWS);
        let started = Instant::now();
        for &row in &selected_rows {
            decoded.push(black_box(
                weight
                    .read_row(black_box(row))
                    .map_err(anyhow::Error::msg)?,
            ));
        }
        let seconds = started.elapsed().as_secs_f64();
        let mut checksum = HASH_SEED;
        for values in decoded {
            checksum = float_checksum(checksum, &values)?;
        }
        report.record("dequantize_rows", run, seconds, DEQUANT_ROWS, checksum)?;
        report.output.flush()?;
    }
    Ok(())
}

fn parse_tokens(text: &str) -> anyhow::Result<Vec<i32>> {
    text.split(',')
        .enumerate()
        .map(|(index, part)| {
            let part = part.trim();
            ensure!(!part.is_empty(), "--tokens entry #{index} is empty");
            part.parse()
                .with_context(|| format!("Invalid --tokens entry #{index}: {part:?}"))
        })
        .collect()
}

fn argmax_last(logits: &Tensor) -> anyhow::Result<i32> {
    let &[positions, vocabulary] = logits.dims() else {
        anyhow::bail!(
            "Expected [positions, vocabulary] logits, got {:?}",
            logits.dims()
        );
    };
    ensure!(positions > 0 && vocabulary > 0, "Logits contain no scores");
    let row = logits
        .data()
        .get((positions - 1) * vocabulary..positions * vocabulary)
        .context("Logits buffer does not match its shape")?;
    let mut best = 0;
    for (id, &score) in row.iter().enumerate() {
        ensure!(score.is_finite(), "Non-finite logit for id {id}: {score}");
        // 相等时保留较小 id，固定 greedy 的平局行为。
        if score > row[best] {
            best = id;
        }
    }
    i32::try_from(best).context("Selected id exceeds i32")
}

fn synthetic(dims: Vec<usize>, salt: usize) -> anyhow::Result<Tensor> {
    let count = dims
        .iter()
        .try_fold(1usize, |count, &dim| count.checked_mul(dim))
        .context("Synthetic tensor shape overflows usize")?;
    let values = (0..count)
        .map(|index| ((index.wrapping_mul(17).wrapping_add(salt) % 251) as f32 - 125.0) / 512.0)
        .collect();
    Ok(Tensor::new(values, dims))
}

fn tensor_checksum(tensor: &Tensor) -> anyhow::Result<u64> {
    float_checksum(HASH_SEED, tensor.data())
}

fn float_checksum(seed: u64, values: &[f32]) -> anyhow::Result<u64> {
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "Non-finite kernel/logit output"
    );
    Ok(hash_bytes(
        seed,
        values.iter().flat_map(|value| value.to_le_bytes()),
    ))
}

fn hash_bytes(seed: u64, bytes: impl Iterator<Item = u8>) -> u64 {
    bytes.fold(seed, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
    })
}
