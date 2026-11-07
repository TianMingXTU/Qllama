//! CLI 用户路径的集成测试。
//!
//! 这些用例站在使用者角度驱动真实的 `qllama` 可执行文件：喂一个符合 llama.cpp
//! 规范的 GGUF 文件，检查退出码与输出内容。张量命名与 metadata 键名均对照上游
//! `llama-arch.cpp` 的 LLM_TENSOR / LLM_KV 映射表，因此夹具不是任意字节拼凑。

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use qllama::gguf::{GgufFile, GgufValue, ModelMetadata};
use qllama::sampler::{Sampler, SamplerConfig};
use qllama::tokenizer::{EncodeOptions, Tokenizer};
use qllama::transformer::Transformer;

const N_EMBD: usize = 32;
const N_FF: usize = 64;
const N_LAYER: usize = 2;
const VOCAB: usize = 16;
const ALIGNMENT: u64 = 32;

/// 夹具中的一条 metadata 记录。
enum Value {
    U32(u32),
    F32(f32),
    Bool(bool),
    Str(String),
    Tokens(Vec<String>),
    Ints(Vec<i32>),
}

struct TensorSpec {
    name: String,
    dims: Vec<u64>,
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_i32(out: &mut Vec<u8>, value: i32) {
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
        Value::Bool(v) => {
            put_u32(out, 7);
            out.push(u8::from(*v));
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
        Value::Ints(items) => {
            put_u32(out, 9);
            put_u32(out, 5);
            put_u64(out, items.len() as u64);
            for item in items {
                put_i32(out, *item);
            }
        }
    }
}

/// 生成一个微型 Llama 架构的 GGUF 文件。
///
/// `with_tokenizer` 决定是否写入分词器元数据：关掉它可以构造"模型没有分词器"
/// 的场景，用来验证 --prompt 的报错路径是否给出了 --tokens 的逃生门。
fn write_fixture(path: &Path) {
    write_fixture_with(path, true);
}

/// 见 [`write_fixture`] 的 `with_tokenizer` 参数说明。
fn write_fixture_with(path: &Path, with_tokenizer: bool) {
    let mut metadata: Vec<(&str, Value)> = vec![
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
    ];
    if with_tokenizer {
        // 词表长度必须等于 token_embd.weight 的行数：TransformerConfig 会拒绝
        // 两者不一致的文件。16 个槽位 = t/o/k/5 四个单字符 + to/tok/tok5 三个
        // 合并产物 + tok7..tok15 九个占位（从 7 编号避免与链里的 tok5 重名）。
        // merges 让 "tok5" 有确定的编码路径：t+o→to → to+k→tok → tok+5→tok5
        // = id 6。add_space_prefix=false：词表没有 ▁，开着会把前导空格编成 unk。
        metadata.extend([
            ("tokenizer.ggml.model", Value::Str("gpt2".to_string())),
            (
                "tokenizer.ggml.tokens",
                Value::Tokens(
                    // 链符号在前（id 0..6），占位符 tok7..tok15 在后。
                    ["t", "o", "k", "5", "to", "tok", "tok5"]
                        .iter()
                        .map(|s| s.to_string())
                        .chain((7..VOCAB).map(|i| format!("tok{i}")))
                        .collect(),
                ),
            ),
            // token_type：1=NORMAL（上游 LLAMA_TOKEN_TYPE 枚举）。
            ("tokenizer.ggml.token_type", Value::Ints(vec![1; VOCAB])),
            (
                "tokenizer.ggml.merges",
                Value::Tokens(
                    ["t o", "to k", "tok 5"]
                        .into_iter()
                        .map(String::from)
                        .collect(),
                ),
            ),
            ("tokenizer.ggml.bos_token_id", Value::U32(0)),
            ("tokenizer.ggml.eos_token_id", Value::U32(1)),
            ("tokenizer.ggml.add_space_prefix", Value::Bool(false)),
        ]);
    }

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
    assert_eq!(gguf.metadata_count(), 17);

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

#[test]
fn prompt_is_encoded_and_generated_text_is_printed() {
    // --prompt 是文字生成闭环的主入口：文本进、token id 出、再解码回文本。
    // 默认走流式；--no-stream 保留"生成结束后一次性打印 text:"的旧契约。
    let dir = fixture_dir("prompt-generate");
    let model = dir.join("tiny-llama.gguf");
    write_fixture(&model);

    let output = run_cli(
        &model,
        &[
            "--prompt",
            "tok5",
            "--generate",
            "3",
            "--temperature",
            "0",
            "--no-stream",
        ],
    );
    assert!(
        output.status.success(),
        "退出码应为 0: {:?}\n{}",
        output.status,
        stdout_of(&output)
    );
    let stdout = stdout_of(&output);
    // BOS(id 0) 自动加在头部；"tok5" 经 merges 链 t+o→to→tok+5 合并为 id 6。
    assert!(
        stdout.contains("input ids: [0, 6]"),
        "应把 tok5 编码为 [0, 6]:\n{stdout}"
    );
    assert!(
        stdout.contains("generated 3 token(s)"),
        "应生成 3 个 token:\n{stdout}"
    );
    assert!(stdout.contains("text:"), "应打印解码后的文本:\n{stdout}");

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn streamed_generation_matches_the_one_shot_decode() {
    // 流式是默认路径：逐 token 输出的正文必须与对同一串 id 一次性 decode 的
    // 结果完全一致，且旧的 "text:" 缓冲行不再重复打印。
    let dir = fixture_dir("stream-generation");
    let model = dir.join("tiny-llama.gguf");
    write_fixture(&model);

    let output = run_cli(
        &model,
        &["--prompt", "tok5", "--generate", "3", "--temperature", "0"],
    );
    assert!(
        output.status.success(),
        "退出码应为 0: {:?}\n{}",
        output.status,
        stdout_of(&output)
    );
    let stdout = stdout_of(&output);
    let body = streamed_body(&stdout);
    // 只看摘要段：Forward 段本来就有一行 "input text:"，那不是缓冲的生成文本。
    let summary = stdout
        .split("=== Generation summary ===")
        .nth(1)
        .expect("流式模式应打印生成摘要");
    assert!(
        !summary.contains("text:"),
        "流式模式不应再打印缓冲 text: 行:\n{summary}"
    );
    assert!(
        summary.contains("stopped after the requested --generate limit"),
        "摘要应说明停止原因:\n{summary}"
    );

    // 期望值用库侧独立计算：同一夹具、同一 decode 契约。
    let gguf = GgufFile::from_path(&model).expect("读取夹具失败");
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("加载分词器失败");
    let ids = generated_ids(&stdout);
    assert_eq!(
        body,
        tokenizer.decode(&ids).expect("解码生成 id 失败"),
        "流式正文应与一次性 decode 一致"
    );

    fs::remove_dir_all(&dir).ok();
}

/// 提取流式正文：`=== Generation ===` 段头之后、`=== Generation summary ===`
/// 之前的全部文本（去掉结尾为可读性补的换行）。
fn streamed_body(stdout: &str) -> String {
    let after_header = stdout
        .split("=== Generation ===\n")
        .nth(1)
        .expect("输出应包含 Generation 段");
    let body = after_header
        .split("=== Generation summary ===")
        .next()
        .expect("流式输出后应有摘要段");
    body.strip_suffix('\n').unwrap_or(body).to_string()
}

/// 解析 "generated N token(s): [a, b, c]" 中的 id 列表，逐元素比对而不是
/// 子串匹配——子串会把 id 11 误判成 id 1。
fn generated_ids(stdout: &str) -> Vec<i32> {
    let line = stdout
        .lines()
        .find(|line| line.trim_start().starts_with("generated "))
        .expect("输出缺少 generated 行");
    let list = line
        .split_once('[')
        .and_then(|(_, rest)| rest.split_once(']'))
        .map(|(inside, _)| inside.trim())
        .expect("generated 行应给出 id 列表");
    if list.is_empty() {
        Vec::new()
    } else {
        list.split(',')
            .map(|value| value.trim().parse::<i32>().expect("id 应可解析"))
            .collect()
    }
}

#[test]
fn top5_logit_summary_prints_only_when_not_sampling() {
    // 规范（spec-cli.md）：top-5 logits 摘要仅在未采样（generate==0）时打印
    // ——采样时惩罚系数会改变实际采到的 token，裸 argmax 展示容易误导。
    let dir = fixture_dir("top5-summary");
    let model = dir.join("tiny-llama.gguf");
    write_fixture(&model);

    let sampling = stdout_of(&run_cli(
        &model,
        &["--prompt", "tok5", "--generate", "2", "--temperature", "0"],
    ));
    assert!(
        !sampling.contains("#0"),
        "采样时不应打印 top-5 摘要:\n{sampling}"
    );

    let not_sampling = stdout_of(&run_cli(
        &model,
        &["--prompt", "tok5", "--temperature", "0"],
    ));
    assert!(
        not_sampling.contains("#0") && not_sampling.contains("logit"),
        "未采样时应保留 top-5 摘要:\n{not_sampling}"
    );

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn prompt_and_tokens_are_mutually_exclusive() {
    // 两个输入来源同时给出时，"到底跑了哪段序列"变得不确定，必须拒绝。
    let dir = fixture_dir("prompt-and-tokens");
    let model = dir.join("tiny-llama.gguf");
    write_fixture(&model);

    let output = run_cli(&model, &["--prompt", "tok5", "--tokens", "1,2"]);
    assert!(
        !output.status.success(),
        "--prompt 与 --tokens 同时给出应以非零码退出"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("mutually exclusive"),
        "错误信息应指出互斥: {stderr}"
    );

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn generate_without_any_input_is_rejected() {
    let dir = fixture_dir("needs-input");
    let model = dir.join("tiny-llama.gguf");
    write_fixture(&model);

    let output = run_cli(&model, &["--generate", "2"]);
    assert!(
        !output.status.success(),
        "既无 --prompt 也无 --tokens 时应以非零码退出"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--prompt") && stderr.contains("--tokens"),
        "错误信息应指出两个可用入口: {stderr}"
    );

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn prompt_on_a_model_without_tokenizer_metadata_points_at_tokens() {
    // 模型没有分词器元数据是常见情形（量化工具可能不写 tokenizer 段），
    // 错误信息必须给出可操作的替代方案，而不是一句 "not implemented"。
    let dir = fixture_dir("prompt-no-tokenizer");
    let model = dir.join("tiny-llama.gguf");
    write_fixture_with(&model, false);

    let output = run_cli(&model, &["--prompt", "tok5"]);
    assert!(
        !output.status.success(),
        "缺分词器元数据时 --prompt 应以非零码退出"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--tokens"),
        "错误信息应提示改用 --tokens: {stderr}"
    );

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_fixed_seed_makes_sampling_reproducible() {
    // 同一 seed 必须给出同一串 token，否则脚本无法复现实验。
    let dir = fixture_dir("seed-reproducible");
    let model = dir.join("tiny-llama.gguf");
    write_fixture(&model);

    let args = ["--prompt", "tok5", "--generate", "5", "--seed", "42"];
    let first = stdout_of(&run_cli(&model, &args));
    let second = stdout_of(&run_cli(&model, &args));
    assert_eq!(first, second, "同一 seed 的两次运行输出应完全一致");

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn temperature_zero_is_greedy_regardless_of_the_seed() {
    // 贪心路径不消耗随机数，换 seed 不能改变结果。
    let dir = fixture_dir("greedy-seed");
    let model = dir.join("tiny-llama.gguf");
    write_fixture(&model);

    let first = stdout_of(&run_cli(
        &model,
        &[
            "--prompt",
            "tok5",
            "--generate",
            "4",
            "--temperature",
            "0",
            "--seed",
            "1",
        ],
    ));
    let second = stdout_of(&run_cli(
        &model,
        &[
            "--prompt",
            "tok5",
            "--generate",
            "4",
            "--temperature",
            "0",
            "--seed",
            "999",
        ],
    ));
    let generated = |stdout: &str| {
        stdout
            .lines()
            .find_map(|line| line.trim().strip_prefix("generated "))
            .unwrap_or_else(|| panic!("输出缺少 generated 行:\n{stdout}"))
            .to_string()
    };
    assert_eq!(
        generated(&first),
        generated(&second),
        "贪心采样不应受 seed 影响"
    );

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn logits_file_holds_one_parseable_value_per_vocab_entry() {
    let dir = fixture_dir("logits-content");
    let model = dir.join("tiny-llama.gguf");
    write_fixture(&model);
    let logits = dir.join("logits.txt");

    let output = run_cli(
        &model,
        &["--tokens", "1,2", "--logits", logits.to_str().unwrap()],
    );
    assert!(output.status.success(), "退出码应为 0: {:?}", output.status);
    let body = fs::read_to_string(&logits).expect("读取 logits 文件失败");
    let values: Vec<f32> = body
        .lines()
        .map(|line| {
            line.parse::<f32>()
                .unwrap_or_else(|e| panic!("无法解析 {line:?}: {e}"))
        })
        .collect();
    assert_eq!(values.len(), VOCAB, "行数应等于词表大小");
    assert!(
        values.iter().all(|value| value.is_finite()),
        "logits 不应含 NaN 或 Inf"
    );

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn generation_stops_at_the_requested_token_count() {
    let dir = fixture_dir("generate-count");
    let model = dir.join("tiny-llama.gguf");
    write_fixture(&model);

    let output = run_cli(&model, &["--tokens", "2", "--generate", "4"]);
    assert!(output.status.success(), "退出码应为 0: {:?}", output.status);
    let stdout = stdout_of(&output);
    assert!(
        stdout.contains("generated 4 token(s)"),
        "应生成 4 个 token:\n{stdout}"
    );

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn top_k_above_the_vocab_size_is_accepted() {
    // top-k 是"至多保留 K 个"，K 大于词表大小时等价于不限制，不该报错。
    let dir = fixture_dir("top-k-large");
    let model = dir.join("tiny-llama.gguf");
    write_fixture(&model);

    let output = run_cli(
        &model,
        &["--tokens", "2", "--generate", "2", "--top-k", "999"],
    );
    assert!(
        output.status.success(),
        "top-k 超过词表大小应被接受: {:?}\n{}",
        output.status,
        stdout_of(&output)
    );

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn no_bos_flag_keeps_the_bos_token_out_of_the_prompt() {
    // BOS 的有无会改变整个序列的语义，必须能从命令行控制。
    let dir = fixture_dir("no-bos");
    let model = dir.join("tiny-llama.gguf");
    write_fixture(&model);

    let with_bos = stdout_of(&run_cli(&model, &["--prompt", "tok5", "--generate", "1"]));
    assert!(
        with_bos.contains("input ids: [0, 6]"),
        "默认应加上 BOS:\n{with_bos}"
    );

    let without_bos = stdout_of(&run_cli(
        &model,
        &["--prompt", "tok5", "--generate", "1", "--no-bos"],
    ));
    assert!(
        without_bos.contains("input ids: [6]"),
        "--no-bos 时不应加 BOS:\n{without_bos}"
    );

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn generation_stops_early_when_the_model_emits_eos() {
    // EOS 必须终止生成。仓库内 golden 夹具（tests/fixtures/tiny-llama.gguf，
    // tok5 编码为 [0, 8]）的贪心链在温度 0 下第 43 步采到 EOS(id 1)——实测
    // --generate 46/47 都只生成 43 个 token。请求 46 个，断言生成数严格小于
    // 请求值且列表不含 EOS：只有提前停止分支被执行，这个断言才可能通过
    // （此前用 --generate 8 的写法 8 步内碰不到 EOS，删掉停止逻辑测试照样
    // 全绿，属于零覆盖）。
    let model = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("tiny-llama.gguf");

    let output = run_cli(
        &model,
        &[
            "--prompt",
            "tok5",
            "--generate",
            "46",
            "--temperature",
            "0",
            "--no-stream",
        ],
    );
    assert!(output.status.success(), "退出码应为 0: {:?}", output.status);
    let stdout = stdout_of(&output);
    let ids = generated_ids(&stdout);
    // 提前停止的直接证据：生成数严格小于请求的上限。仅靠循环上限不可能
    // 产生这个结果——若 EOS 停止被移除，这里会生成满 46 个并失败。
    assert!(
        ids.len() < 46,
        "应在采到 EOS 时提前停止（生成数应严格小于 46）:\n{stdout}"
    );
    assert!(
        !ids.contains(&1),
        "EOS(id 1) 不应出现在生成结果里:\n{stdout}"
    );
    // 停止原因必须显式：只报"生成了 43 个"会让使用者以为请求被截断了。
    assert!(
        stdout.contains("stopped at end-of-sequence"),
        "应报告 EOS 停止:\n{stdout}"
    );
}

#[test]
fn ctx_size_smaller_than_the_prompt_is_rejected_before_the_forward() {
    // 放不下就是放不下：必须在前向之前显式报错并说明差额，而不是跑半截再
    // 被 KV cache 的越界信息打断。
    let dir = fixture_dir("ctx-too-small");
    let model = dir.join("tiny-llama.gguf");
    write_fixture(&model);

    let output = run_cli(
        &model,
        &[
            "--tokens",
            "1,2,3,4,5,6",
            "--generate",
            "4",
            "--ctx-size",
            "6",
        ],
    );
    assert!(
        !output.status.success(),
        "prompt + 生成超出 --ctx-size 应以非零码退出"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--ctx-size 6") && stderr.contains("6 prompt tokens"),
        "错误信息应给出预算与需求: {stderr}"
    );
    assert!(
        !stdout_of(&output).contains("=== Forward ==="),
        "预算检查必须先于前向:\n{}",
        stdout_of(&output)
    );

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn ctx_size_larger_than_the_model_maximum_is_rejected() {
    let dir = fixture_dir("ctx-too-large");
    let model = dir.join("tiny-llama.gguf");
    write_fixture(&model);

    let output = run_cli(&model, &["--tokens", "1", "--ctx-size", "1024"]);
    assert!(!output.status.success(), "超出模型标称上下文应报错");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("1024") && stderr.contains("128"),
        "错误信息应同时给出请求值与模型上限: {stderr}"
    );

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn ctx_size_limits_generation_and_says_so() {
    // 夹具标称上下文 128。把 --ctx-size 压到刚好放得下 prompt+3 个新 token，
    // 第 4 步必须由预算检查拦下并说明原因；不能静默少生成。
    let dir = fixture_dir("ctx-limited");
    let model = dir.join("tiny-llama.gguf");
    write_fixture(&model);

    let output = run_cli(
        &model,
        &[
            "--tokens",
            "1,2",
            "--generate",
            "10",
            "--ctx-size",
            "5",
            "--no-stream",
        ],
    );
    assert!(
        output.status.success(),
        "预算内生成完成后应正常退出: {:?}\n{}",
        output.status,
        stdout_of(&output)
    );
    let stdout = stdout_of(&output);
    let ids = generated_ids(&stdout);
    assert_eq!(ids.len(), 3, "只应生成预算允许的 3 个 token:\n{stdout}");
    assert!(
        stdout.contains("reached the context budget"),
        "应报告上下文耗尽:\n{stdout}"
    );

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn thread_count_does_not_change_the_output() {
    // 线程数只影响 matmul 的划分方式，不影响每个输出的累加顺序（见
    // MappedWeight::matmul 的注释），因此结果必须逐字节一致。
    let dir = fixture_dir("threads-parity");
    let model = dir.join("tiny-llama.gguf");
    write_fixture(&model);

    let base = ["--prompt", "tok5", "--generate", "4", "--temperature", "0"];
    let one = stdout_of(&run_cli(&model, &[&base[..], &["--threads", "1"]].concat()));
    let many = stdout_of(&run_cli(&model, &[&base[..], &["--threads", "4"]].concat()));
    assert_eq!(one, many, "--threads 不应改变输出");

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn zero_threads_is_rejected_by_the_argument_parser() {
    let dir = fixture_dir("threads-zero");
    let model = dir.join("tiny-llama.gguf");
    write_fixture(&model);

    let output = run_cli(&model, &["--tokens", "1", "--threads", "0"]);
    assert!(!output.status.success(), "--threads 0 应被拒绝");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--threads"), "错误应指向参数: {stderr}");

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn cached_generation_matches_full_sequence_greedy() {
    // 阶段 5 红线 10：--generate 现在走 KV Cache 增量路径，贪心（温度 0）下
    // 生成的 token 序列必须与"每步重算整段序列"的旧路径逐个相同。
    //
    // 对照基准在测试内独立重放旧循环（整段 forward + 同参数采样），CLI 二进制
    // 则跑新路径。链长取到 EOS（43 步）：链越长，两条路径一旦有微小数值差
    // 被放大成不同 argmax 的机会越大。
    //
    // **测试边界（诚实声明）**：本用例只比对 token id 序列，不比对 logits。
    // 实测"读回后二次旋转 K"这类缺陷造成的 logits 偏差（1e-2 量级）小于
    // 夹具的 top1-top2 间隔，贪心链可能一步不分叉——logits 级的牙在
    // tests/transformer_forward.rs 的一致性测试里（1e-5 逐元素断言）。
    // 同理，若 generate 被改回"每步整段 forward"，本用例因两条路径数值一致
    // 仍会通过：它锁的是可观察输出契约（红线 10），不是实现路径本身。
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("tiny-llama.gguf");
    // 请求 46 步；夹具贪心链第 43 步采到 EOS(id 1)，两条路径都应在此停止。
    let steps = 46;
    let eos_id = 1;

    // 参考序列：整段前向 + 同参数贪心采样（旧 generate 的循环结构，EOS 语义一致）。
    let gguf = GgufFile::from_path(&model_path).expect("读取夹具失败");
    let model = Transformer::from_gguf(&gguf).expect("加载模型失败");
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("加载分词器失败");
    let prompt = tokenizer
        .encode_with_options(
            "tok5",
            &EncodeOptions {
                add_bos: true,
                add_eos: false,
            },
        )
        .expect("编码失败");
    let sampler_config = SamplerConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        // 与 CLI 显式传参一致。注意重复惩罚会改变贪心 argmax：1.0（无惩罚）
        // 时该夹具的贪心链 46 步内碰不到 EOS，只有 1.1 才复现 golden 记录的
        // 第 43 步 EOS 链。
        repeat_penalty: 1.1,
    };
    let mut sampler = Sampler::new(sampler_config);
    let mut ids = prompt.clone();
    let mut expected = Vec::with_capacity(steps);
    for _ in 0..steps {
        let logits = model.forward(&ids, 0).expect("整段前向失败");
        let columns = logits.dims()[1];
        let last = &logits.data()[(ids.len() - 1) * columns..ids.len() * columns];
        let next = sampler.sample_slice(last, &ids).expect("采样失败");
        if next == eos_id {
            break;
        }
        expected.push(next);
        ids.push(next);
    }
    // 参考链本身必须真的提前停止，否则本用例退化成"跑满步数"的弱对照
    assert!(
        expected.len() < steps,
        "参考整段循环应在 {steps} 步内采到 EOS，实际生成 {} 个",
        expected.len()
    );

    // CLI 走增量路径，参数与参考一致（温度 0 即贪心，不涉及 RNG）。
    let output = run_cli(
        &model_path,
        &[
            "--prompt",
            "tok5",
            "--generate",
            &steps.to_string(),
            "--temperature",
            "0",
            "--top-k",
            "0",
            "--top-p",
            "1",
            "--repeat-penalty",
            "1.1",
        ],
    );
    assert!(output.status.success(), "退出码应为 0: {:?}", output.status);
    let generated = stdout_of(&output)
        .lines()
        .find_map(|line| line.trim().strip_prefix("generated "))
        .expect("输出缺少 generated 行")
        .to_string();
    let list = generated
        .split_once('[')
        .and_then(|(_, rest)| rest.split_once(']'))
        .map(|(inside, _)| inside.trim())
        .expect("generated 行应给出 id 列表");
    let actual: Vec<i32> = if list.is_empty() {
        Vec::new()
    } else {
        list.split(',')
            .map(|value| value.trim().parse::<i32>().expect("id 应可解析"))
            .collect()
    };

    assert_eq!(
        actual, expected,
        "增量路径与整段路径的贪心序列不一致（夹具 tok5 贪心链至 EOS）"
    );
}
