//! 真实 GGUF 模型的对照脚手架：默认不运行，需显式 opt-in。
//!
//! 用法（指向一个 **F32/F16 权重**的模型）：
//!
//! ```text
//! QLLAMA_REAL_MODEL=/path/to/model.gguf cargo test --test real_model -- --ignored --nocapture
//! ```
//!
//! 环境变量未设置时所有用例直接返回（不失败），因此不会进入默认 CI。
//!
//! 范围说明：这里验证"加载、分词、前向有限、增量与整段一致"四件事，都是
//! **实现自洽性**检查，不涉及外部参照。与 llama.cpp 的数值对照（argmax、
//! top-16、逐值偏差、贪心链、分词粒度）在 `tests/llamacpp_golden.rs`，
//! 冻结的参考数据与生成命令见 `tests/golden/`（TSV 头部即来源说明）。
//!
//! **量化模型不在支持范围**：Q4_0 等量化张量的加载会显式报错（
//! `src/gguf/loader.rs` 的 `load_tensor`），本脚手架不会偷偷反量化后继续
//! 对照——那测的是另一条代码路径。

use std::path::PathBuf;

use qllama::gguf::GgufFile;
use qllama::tokenizer::Tokenizer;
use qllama::transformer::Transformer;

/// 真实模型路径；未设置 `QLLAMA_REAL_MODEL` 时返回 `None`（用例跳过）。
fn model_path() -> Option<PathBuf> {
    std::env::var_os("QLLAMA_REAL_MODEL").map(PathBuf::from)
}

/// 用例入口：没有真实模型时打印说明并跳过。
fn load_model() -> Option<(PathBuf, GgufFile)> {
    let path = model_path()?;
    let gguf = GgufFile::from_path(&path)
        .unwrap_or_else(|error| panic!("无法读取 {}: {error}", path.display()));
    Some((path, gguf))
}

#[test]
#[ignore = "需要 QLLAMA_REAL_MODEL 指向真实 GGUF；量化模型应被显式拒绝"]
fn real_model_loads_without_quantization_errors() {
    let Some((path, gguf)) = load_model() else {
        eprintln!("跳过：未设置 QLLAMA_REAL_MODEL");
        return;
    };
    let model = Transformer::from_gguf(&gguf)
        .unwrap_or_else(|error| panic!("加载失败（量化模型应在此显式报错）: {error}"));
    let config = model.config();
    println!("模型: {}", path.display());
    println!(
        "hidden={} layers={} heads={} kv_heads={} head_dim={} vocab={} max_pos={}",
        config.hidden_size,
        config.num_layers,
        config.num_attention_heads,
        config.num_kv_heads,
        config.head_dim(),
        config.vocab_size,
        config.max_position_embeddings
    );
    assert!(config.hidden_size > 0 && config.vocab_size > 0);
}

#[test]
#[ignore = "需要 QLLAMA_REAL_MODEL 指向真实 GGUF"]
fn real_model_tokenizer_encodes_a_prompt() {
    let Some((_, gguf)) = load_model() else {
        eprintln!("跳过：未设置 QLLAMA_REAL_MODEL");
        return;
    };
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("模型缺少可用的分词器元数据");
    let ids = tokenizer
        .encode("The capital of France is")
        .expect("编码失败");
    assert!(!ids.is_empty(), "编码结果不应为空");
    let vocab = tokenizer.vocab_size();
    for &id in &ids {
        assert!(
            (0..vocab as i32).contains(&id),
            "id {id} 越界（vocab={vocab}）"
        );
    }
    println!("ids: {ids:?}");
    println!("decoded: {:?}", tokenizer.decode(&ids));
}

#[test]
#[ignore = "需要 QLLAMA_REAL_MODEL 指向真实 GGUF"]
fn real_model_forward_is_finite_and_argmax_is_stable() {
    let Some((_, gguf)) = load_model() else {
        eprintln!("跳过：未设置 QLLAMA_REAL_MODEL");
        return;
    };
    let model = Transformer::from_gguf(&gguf).expect("加载失败");
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("加载分词器失败");
    let ids = tokenizer
        .encode("The capital of France is")
        .expect("编码失败");

    let logits = model.forward(&ids, 0).expect("整段前向失败");
    assert!(
        logits.data().iter().all(|value| value.is_finite()),
        "logits 含非有限值"
    );
    let columns = logits.dims()[1];
    let last = &logits.data()[(ids.len() - 1) * columns..ids.len() * columns];
    let argmax = last
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .expect("logits 非空");
    println!("末位置 argmax = {argmax}，前 5: {:?}", top5(last));
}

/// 取一行 logits 的前 5 大 `(id, logit)`，仅用于人工对照。
fn top5(row: &[f32]) -> Vec<(usize, f32)> {
    let mut indices: Vec<usize> = (0..row.len()).collect();
    indices.sort_by(|&left, &right| row[right].total_cmp(&row[left]));
    indices
        .into_iter()
        .take(5)
        .map(|index| (index, row[index]))
        .collect()
}

#[test]
#[ignore = "需要 QLLAMA_REAL_MODEL 指向真实 GGUF"]
fn real_model_incremental_matches_full_last_position() {
    // 真实模型上的增量/整段一致性：逐 token decode 后与整段前向的末行
    // logits 比较。真实权重下两条路径的差异应仍在浮点噪声量级
    // （打印最大偏差，供与 llama.cpp 对照时判断阈值）。
    let Some((_, gguf)) = load_model() else {
        eprintln!("跳过：未设置 QLLAMA_REAL_MODEL");
        return;
    };
    let model = Transformer::from_gguf(&gguf).expect("加载失败");
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("加载分词器失败");
    let ids = tokenizer
        .encode("The capital of France is")
        .expect("编码失败");
    let config = model.config();

    let full = model.forward(&ids, 0).expect("整段前向失败");
    let columns = full.dims()[1];
    let expected = &full.data()[(ids.len() - 1) * columns..ids.len() * columns];

    let mut cache = qllama::kv_cache::KvCache::new(
        config.num_layers,
        config.max_position_embeddings,
        config.num_kv_heads,
        config.head_dim(),
    );
    let mut logits = model
        .forward_with_cache(&ids[..1], 0, &mut cache)
        .expect("prefill 失败");
    for (position, &id) in ids[1..].iter().enumerate() {
        logits = model
            .forward_with_cache(&[id], position + 1, &mut cache)
            .unwrap_or_else(|error| panic!("位置 {} 增量前向失败: {error}", position + 1));
    }
    let actual = &logits.data()[logits.len() - columns..];

    let max_diff = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("末行 logits 最大偏差 = {max_diff:.3e}");
    assert!(
        max_diff < 1e-3,
        "增量与整段路径偏差 {max_diff:.3e} 超出预期量级"
    );
}
