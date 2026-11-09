//! 性能改动的逐值红线：一次整段前向对照全部 oracle 位置，并保存可逐位比较的结果。

#![forbid(unsafe_code)]

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

use anyhow::{ensure, Context};
use clap::Parser;
use qllama::{GgufFile, Transformer};
use serde_json::{json, Value};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    reference: PathBuf,
    #[arg(long)]
    save: Option<PathBuf>,
    #[arg(long)]
    baseline: Option<PathBuf>,
    #[arg(long, default_value_t = 4)]
    threads: usize,
    #[arg(long, default_value_t = 1e-4)]
    tolerance: f64,
}

fn top5(row: &[f64]) -> Vec<usize> {
    let mut ids: Vec<usize> = (0..row.len()).collect();
    ids.sort_unstable_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
    ids.truncate(5);
    ids
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    ensure!(args.threads > 0, "Threads must be positive");
    ensure!(
        args.tolerance.is_finite() && args.tolerance >= 0.0,
        "Invalid tolerance"
    );
    let mut reader = BufReader::new(File::open(&args.reference)?).lines();
    let metadata: Value = serde_json::from_str(&reader.next().context("Empty reference")??)?;
    let ids: Vec<i32> = serde_json::from_value(metadata["ids"].clone())?;
    let width = metadata["n_vocab"].as_u64().context("Missing n_vocab")? as usize;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()?;
    let gguf = GgufFile::from_path(&args.model).map_err(anyhow::Error::msg)?;
    let model = Transformer::from_gguf(&gguf).map_err(anyhow::Error::msg)?;
    let logits = pool
        .install(|| model.forward(&ids, 0))
        .map_err(anyhow::Error::msg)?;
    ensure!(
        logits.dims() == [ids.len(), width],
        "Unexpected logits shape"
    );

    let mut bytes = Vec::with_capacity(logits.len() * 4);
    for value in logits.data() {
        ensure!(value.is_finite(), "Non-finite output");
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let mut baseline_max = 0.0f64;
    let mut baseline_bit_mismatches = 0;
    if let Some(path) = args.baseline {
        let baseline = std::fs::read(path)?;
        ensure!(baseline.len() == bytes.len(), "Baseline dimensions differ");
        for (old, new) in baseline
            .as_chunks::<4>()
            .0
            .iter()
            .zip(bytes.as_chunks::<4>().0)
        {
            if old != new {
                baseline_bit_mismatches += 1;
            }
            let old = f32::from_le_bytes(*old);
            let new = f32::from_le_bytes(*new);
            ensure!(old.is_finite(), "Baseline contains a non-finite value");
            baseline_max = baseline_max.max((f64::from(old) - f64::from(new)).abs());
        }
    }
    if let Some(path) = args.save {
        File::create(path)?.write_all(&bytes)?;
    }

    let mut maximum = 0.0f64;
    let mut total = 0.0f64;
    let mut count = 0usize;
    let mut positions = 0usize;
    let mut argmax_mismatches = 0;
    let mut top5_mismatches = 0;
    let mut seen = vec![false; ids.len()];
    for line in reader {
        let reference: Value = serde_json::from_str(&line?)?;
        if reference["engine"] != "llama.cpp" {
            continue;
        }
        let position = reference["position"].as_u64().context("Missing position")? as usize;
        ensure!(
            position < ids.len() && !seen[position],
            "Invalid/duplicate oracle position"
        );
        seen[position] = true;
        let expected: Vec<f64> = serde_json::from_value(reference["values"].clone())?;
        ensure!(
            expected.len() == width && expected.iter().all(|v| v.is_finite()),
            "Invalid reference vector"
        );
        let actual: Vec<f64> = logits.data()[position * width..(position + 1) * width]
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        let actual_top = top5(&actual);
        let expected_top = top5(&expected);
        argmax_mismatches += usize::from(actual_top[0] != expected_top[0]);
        let mut actual_set = actual_top;
        let mut expected_set = expected_top;
        actual_set.sort_unstable();
        expected_set.sort_unstable();
        top5_mismatches += usize::from(actual_set != expected_set);
        let actual_peak = actual.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let expected_peak = expected.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        for (actual, expected) in actual.iter().zip(&expected) {
            let deviation = ((actual - actual_peak) - (expected - expected_peak)).abs();
            maximum = maximum.max(deviation);
            total += deviation;
            count += 1;
        }
        positions += 1;
    }
    ensure!(
        positions == ids.len(),
        "Missing oracle positions ({positions}/{})",
        ids.len()
    );
    println!(
        "{}",
        json!({
            "model": args.model,
            "threads": args.threads,
            "positions": positions,
            "values": count,
            "max_abs_dev": maximum,
            "mean_abs_dev": total / count as f64,
            "argmax_mismatches": argmax_mismatches,
            "top5_set_mismatches": top5_mismatches,
            "baseline_max_abs_dev": baseline_max,
            "baseline_bit_mismatches": baseline_bit_mismatches,
        })
    );
    ensure!(
        maximum <= args.tolerance && argmax_mismatches == 0 && top5_mismatches == 0,
        "Oracle comparison regressed"
    );
    ensure!(
        baseline_bit_mismatches == 0,
        "Output differs from the pre-optimization baseline"
    );
    Ok(())
}
