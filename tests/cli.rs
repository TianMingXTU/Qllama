//! CLI 用户路径的集成测试。
//!
//! 这些用例站在使用者角度驱动真实的 `qllama` 可执行文件：喂一个符合 llama.cpp
//! 规范的 GGUF 文件，检查退出码与输出内容。张量命名与 metadata 键名均对照上游
//! `llama-arch.cpp` 的 LLM_TENSOR / LLM_KV 映射表，因此夹具不是任意字节拼凑。

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use qllama::gguf::{GgufFile, GgufValue, ModelMetadata};

const N_EMBD: usize = 32;
const N_FF: usize = 64;
const N_LAYER: usize = 2;
const VOCAB: usize = 16;
const ALIGNMENT: u64 = 32;

/// 夹具中的一条 metadata 记录。
enum Value {
    U32(u32),
    F32(f32),
    Str(String),
    Tokens(Vec<String>),
}

struct TensorSpec {
    name: String,
    dims: Vec<u64>,
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_f32(out: &mut Vec<u8>, value: f32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_string(out: &mut Vec<u8>, value: &str) {
    put_u64(out, value.len() as u64);
    out.extend_from_slice(value.as_bytes());
}

fn put_value(out: &mut Vec<u8>, value: &Value) {
    match value {
        Value::U32(v) => {
            put_u32(out, 4);
            put_u32(out, *v);
        }
        Value::F32(v) => {
            put_u32(out, 6);
            put_f32(out, *v);
        }
        Value::Str(v) => {
            put_u32(out, 8);
            put_string(out, v);
        }
        // 数组：type 9 + 元素类型 + 元素个数 + 各元素。
        Value::Tokens(items) => {
            put_u32(out, 9);
            put_u32(out, 8);
            put_u64(out, items.len() as u64);
            for item in items {
                put_string(out, item);
            }
        }
    }
}

/// 生成一个微型 Llama 架构的 GGUF 文件。
fn write_fixture(path: &Path) {
    let metadata: Vec<(&str, Value)> = vec![
        ("general.architecture", Value::Str("llama".to_string())),
        ("general.name", Value::Str("qllama-cli-fixture".to_string())),
        ("llama.context_length", Value::U32(128)),
        ("llama.embedding_length", Value::U32(N_EMBD as u32)),
        ("llama.block_count", Value::U32(N_LAYER as u32)),
        ("llama.feed_forward_length", Value::U32(N_FF as u32)),
        ("llama.attention.head_count", Value::U32(4)),
        ("llama.attention.head_count_kv", Value::U32(4)),
        ("llama.attention.layer_norm_rms_epsilon", Value::F32(1e-5)),
        ("llama.rope.freq_base", Value::F32(10000.0)),
        ("tokenizer.ggml.model", Value::Str("llama".to_string())),
        (
            "tokenizer.ggml.tokens",
            Value::Tokens((0..VOCAB).map(|i| format!("tok{i}")).collect()),
        ),
    ];

    // token_embd.weight 的 dims 必须按 ggml 的 ne[] 顺序写，即 {n_embd, n_vocab}；
    // 早期写成 {n_vocab, n_embd} 虽然不影响本文件的解析用例，但会让任何按
    // llama.cpp 约定解释该文件的代码拿到错误形状。
    let mut tensors = vec![TensorSpec {
        name: "token_embd.weight".to_string(),
        dims: vec![N_EMBD as u64, VOCAB as u64],
    }];
    for il in 0..N_LAYER {
        tensors.extend([
            TensorSpec {
                name: format!("blk.{il}.attn_norm.weight"),
                dims: vec![N_EMBD as u64],
            },
            TensorSpec {
                name: format!("blk.{il}.attn_q.weight"),
                dims: vec![N_EMBD as u64, N_EMBD as u64],
            },
            TensorSpec {
                name: format!("blk.{il}.attn_k.weight"),
                dims: vec![N_EMBD as u64, N_EMBD as u64],
            },
            TensorSpec {
                name: format!("blk.{il}.attn_v.weight"),
                dims: vec![N_EMBD as u64, N_EMBD as u64],
            },
            TensorSpec {
                name: format!("blk.{il}.attn_output.weight"),
                dims: vec![N_EMBD as u64, N_EMBD as u64],
            },
            TensorSpec {
                name: format!("blk.{il}.ffn_norm.weight"),
                dims: vec![N_EMBD as u64],
            },
            TensorSpec {
                name: format!("blk.{il}.ffn_gate.weight"),
                dims: vec![N_EMBD as u64, N_FF as u64],
            },
            TensorSpec {
                name: format!("blk.{il}.ffn_up.weight"),
                dims: vec![N_EMBD as u64, N_FF as u64],
            },
            TensorSpec {
                name: format!("blk.{il}.ffn_down.weight"),
                dims: vec![N_FF as u64, N_EMBD as u64],
            },
        ]);
    }
    tensors.push(TensorSpec {
        name: "output_norm.weight".to_string(),
        dims: vec![N_EMBD as u64],
    });
    tensors.push(TensorSpec {
        name: "output.weight".to_string(),
        dims: vec![N_EMBD as u64, VOCAB as u64],
    });

    let mut body = Vec::new();
    body.extend_from_slice(b"GGUF");
    put_u32(&mut body, 3);
    put_u64(&mut body, tensors.len() as u64);
    put_u64(&mut body, metadata.len() as u64);
    for (key, value) in &metadata {
        put_string(&mut body, key);
        put_value(&mut body, value);
    }

    // 数据区内每个张量同样按 alignment 对齐，与 llama.cpp 的写入方式一致。
    let mut data = Vec::new();
    let mut offsets = Vec::with_capacity(tensors.len());
    for tensor in &tensors {
        let pad = (ALIGNMENT - (data.len() as u64 % ALIGNMENT)) % ALIGNMENT;
        data.extend(std::iter::repeat_n(0u8, pad as usize));
        offsets.push(data.len() as u64);
        let count: usize = tensor.dims.iter().map(|&d| d as usize).product();
        for index in 0..count {
            put_f32(&mut data, 0.001 * (index as f32 + 1.0));
        }
    }

    for (tensor, offset) in tensors.iter().zip(&offsets) {
        put_string(&mut body, &tensor.name);
        put_u32(&mut body, tensor.dims.len() as u32);
        for dim in &tensor.dims {
            put_u64(&mut body, *dim);
        }
        put_u32(&mut body, 0); // GGML_TYPE_F32
        put_u64(&mut body, *offset);
    }

    let pad = (ALIGNMENT - (body.len() as u64 % ALIGNMENT)) % ALIGNMENT;
    body.extend(std::iter::repeat_n(0u8, pad as usize));

    let mut bytes = body;
    bytes.extend(data);
    fs::write(path, bytes).expect("写入夹具失败");
}

/// 每个测试使用独立目录，避免并行执行时互相覆盖。
fn fixture_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("qllama-cli-{name}-{}", std::process::id()));
    fs::create_dir_all(&dir).expect("创建临时目录失败");
    dir
}

fn run_cli(model: &Path, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_qllama"))
        .arg("--model")
        .arg(model)
        .args(extra)
        .output()
        .expect("启动 qllama 失败")
}

fn stdout_of(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

#[test]
fn info_prints_llama_configuration_and_tensor_index() {
    let dir = fixture_dir("info");
    let model = dir.join("tiny-llama.gguf");
    write_fixture(&model);

    let output = run_cli(&model, &["--info"]);
    assert!(output.status.success(), "退出码应为 0: {:?}", output.status);
    let stdout = stdout_of(&output);

    // 这些键全部带命名空间，曾被按 '.' 过滤而整段消失。
    for expected in [
        "general.architecture",
        "llama.context_length",
        "llama.embedding_length",
        "llama.block_count",
        "llama.attention.head_count",
        "llama.attention.layer_norm_rms_epsilon",
        "llama.rope.freq_base",
        "tokenizer.ggml.tokens",
    ] {
        assert!(stdout.contains(expected), "输出缺少 {expected}:\n{stdout}");
    }
    // 词表只打印摘要，不展开 16 个元素。
    assert!(
        stdout.contains("tokenizer.ggml.tokens: [16 items"),
        "词表应为摘要形式:\n{stdout}"
    );
    assert!(
        stdout.contains("blk.0.attn_q.weight"),
        "缺少张量列表:\n{stdout}"
    );
    assert!(
        stdout.contains("... and 11 more"),
        "缺少张量截断提示:\n{stdout}"
    );

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn info_output_is_deterministic_across_runs() {
    let dir = fixture_dir("determinism");
    let model = dir.join("tiny-llama.gguf");
    write_fixture(&model);

    let first = stdout_of(&run_cli(&model, &["--info"]));
    let second = stdout_of(&run_cli(&model, &["--info"]));
    assert_eq!(first, second, "两次运行输出不一致，元数据顺序未固定");

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn model_metadata_extracts_hyperparameters_from_fixture() {
    let dir = fixture_dir("metadata");
    let model = dir.join("tiny-llama.gguf");
    write_fixture(&model);

    let gguf = GgufFile::from_path(&model).expect("解析夹具失败");
    let metadata = ModelMetadata::from_gguf_metadata(&gguf.metadata).expect("提取模型超参数失败");

    assert_eq!(metadata.architecture, "llama");
    assert_eq!(metadata.name, "qllama-cli-fixture");
    assert_eq!(metadata.context_length, 128);
    assert_eq!(metadata.embedding_length, N_EMBD);
    assert_eq!(metadata.block_count, N_LAYER);
    assert_eq!(metadata.attention_head_count, 4);
    assert_eq!(metadata.attention_head_count_kv, 4);
    assert_eq!(metadata.layer_norm_eps, 1e-5);
    assert_eq!(metadata.rope_freq_base, 10000.0);
    assert_eq!(metadata.vocab_size, VOCAB);

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn rejects_non_gguf_file_with_nonzero_exit() {
    let dir = fixture_dir("bad-magic");
    let model = dir.join("not-a-model.txt");
    fs::write(&model, b"plain text, definitely not GGUF").expect("写入失败");

    let output = run_cli(&model, &["--info"]);
    assert!(!output.status.success(), "非法文件应以非零码退出");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Invalid GGUF magic"),
        "错误信息应指出魔数问题: {stderr}"
    );

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn rejects_missing_file_with_nonzero_exit() {
    let dir = fixture_dir("missing");
    let output = run_cli(&dir.join("does-not-exist.gguf"), &["--info"]);
    assert!(!output.status.success(), "缺失文件应以非零码退出");

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn fixture_values_survive_the_wire_round_trip() {
    // 确认夹具写入与解析器读取一致：metadata 值类型与张量维度原样回来。
    let dir = fixture_dir("round-trip");
    let model = dir.join("tiny-llama.gguf");
    write_fixture(&model);

    let gguf = GgufFile::from_path(&model).expect("解析夹具失败");
    assert_eq!(gguf.tensor_count(), 21);
    assert_eq!(gguf.metadata_count(), 12);

    match gguf.metadata.get("tokenizer.ggml.tokens") {
        Some(GgufValue::Array(items)) => {
            assert_eq!(items.len(), VOCAB);
            assert!(matches!(items[0], GgufValue::String(_)));
        }
        other => panic!("词表类型不符: {other:?}"),
    }

    let q = gguf
        .find_tensor("blk.0.attn_q.weight")
        .expect("找不到 attn_q");
    assert_eq!(q.dims, vec![N_EMBD as u64, N_EMBD as u64]);
    assert!(gguf.find_tensor("blk.9.attn_q.weight").is_none());

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn generate_and_logits_require_explicit_token_ids() {
    // 退出码必须非零：脚本要靠它判断这次调用到底有没有跑前向。
    let dir = fixture_dir("needs-tokens");
    let model = dir.join("tiny-llama.gguf");
    write_fixture(&model);

    let output = run_cli(&model, &["--generate", "4"]);
    assert!(
        !output.status.success(),
        "--generate 缺少 --tokens 时应以非零码退出"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--tokens"),
        "错误信息应指出缺少 --tokens: {stderr}"
    );

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn rejects_an_empty_field_inside_the_token_list() {
    // "1,,2" 静默按 [1,2] 跑，会让前向作用在与使用者所写不同的序列上。
    let dir = fixture_dir("empty-field");
    let model = dir.join("tiny-llama.gguf");
    write_fixture(&model);

    let output = run_cli(&model, &["--tokens", "1,,2"]);
    assert!(!output.status.success(), "空字段应以非零码退出");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("token #1 is empty"),
        "错误信息应指出空字段的位置: {stderr}"
    );

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn info_announces_that_it_short_circuits_the_forward_pass() {
    // --info 与推理参数同时给出时，前向和 logits 文件都不会发生，必须明说。
    let dir = fixture_dir("info-short-circuit");
    let model = dir.join("tiny-llama.gguf");
    write_fixture(&model);
    let logits = dir.join("never-written.txt");

    let output = run_cli(
        &model,
        &[
            "--info",
            "--tokens",
            "1,2",
            "--logits",
            logits.to_str().unwrap(),
        ],
    );
    assert!(
        output.status.success(),
        "--info 本身应成功: {:?}",
        output.status
    );
    let stdout = stdout_of(&output);
    assert!(
        stdout.contains("short-circuits"),
        "应提示 --info 短路了前向:\n{stdout}"
    );
    assert!(!logits.exists(), "--info 不应写入 logits 文件");

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn token_list_accepts_both_commas_and_whitespace() {
    let dir = fixture_dir("separators");
    let model = dir.join("tiny-llama.gguf");
    write_fixture(&model);

    for raw in ["1,2", "1, 2", "1 2", " 1  2 "] {
        let stdout = stdout_of(&run_cli(&model, &["--tokens", raw, "--top-k", "1"]));
        assert!(
            stdout.contains("input ids: [1, 2]"),
            "{raw:?} 应解析为 [1, 2]:\n{stdout}"
        );
    }

    fs::remove_dir_all(&dir).ok();
}
