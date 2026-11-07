//! 与 llama.cpp 数值对照的回归基准测试（阶段 7）。
//!
//! 数据来源见 `tests/golden/llamacpp_stories15M_b12109_f32cache.tsv` 的头部注释：
//! llama.cpp b12109 预编译 win-cpu-x64，经 `llama-server` 取全词表 f32 logprob。
//! **这些数字是回归基准，不是 oracle**：它锁定的是"2026-11-04 我们与 llama.cpp
//! 对齐到什么程度"。模型权重或参考版本变化后必须用头部记录的命令重新生成。
//!
//! 前两个用例不依赖模型，在 CI 里跑结构自洽性；其余需要真实模型，显式 opt-in：
//!
//! ```text
//! QLLAMA_REAL_MODEL=/path/to/stories15M.gguf \
//! cargo test --test llamacpp_golden -- --ignored --nocapture
//! ```
//!
//! 所有期望值来自 llama.cpp 输出或手工推导，不从被测实现反推。

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use qllama::gguf::GgufFile;
use qllama::tokenizer::{EncodeOptions, Tokenizer};
use qllama::transformer::Transformer;

/// golden TSV 每个位置保留的条数（由导出脚本 `--golden-top` 决定）。
const GOLDEN_TOP: usize = 16;
/// 逐值偏差上限。实测最大偏差 5.198e-05、平均 6.550e-06，留两个数量级余量：
/// RoPE 配对方式错乱一类的真实错误会把偏差抬到 9.12，不会被误判为通过。
const TOLERANCE: f32 = 2.0e-3;
/// 冻结的 position 列表，必须与 `--positions` 导出参数一致。
const GOLDEN_POSITIONS: [usize; 6] = [0, 10, 30, 52, 66, 74];

/// golden 文件里一个位置的数据：`(token id, shifted logprob)` 按 rank 升序。
#[derive(Debug)]
struct GoldenRow {
    position: usize,
    entries: Vec<(i32, f32)>,
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(name)
}

/// 读取 golden TSV，同时返回头部注释里的来源信息。
///
/// 头部 `# key: value` 与数据行共用一个解析入口，避免"文件能读但来源丢失"。
fn read_golden() -> Result<(HashMap<String, String>, Vec<GoldenRow>), String> {
    let text = std::fs::read_to_string(fixture("llamacpp_stories15M_b12109_f32cache.tsv"))
        .map_err(|error| format!("无法读取 golden 文件: {error}"))?;
    let mut provenance = HashMap::new();
    let mut grouped: HashMap<usize, Vec<(usize, i32, f32)>> = HashMap::new();
    for line in text.lines() {
        if let Some(comment) = line.strip_prefix('#') {
            if let Some((key, value)) = comment.trim().split_once(": ") {
                provenance.insert(key.to_string(), value.to_string());
            }
            continue;
        }
        if line.trim().is_empty() {
            continue;
        }
        let columns: Vec<&str> = line.split('\t').collect();
        let [Some(position), Some(rank), Some(token_id), Some(shifted)] = [
            columns.first(),
            columns.get(1),
            columns.get(2),
            columns.get(3),
        ] else {
            return Err(format!("golden 行列数不足: {line}"));
        };
        grouped
            .entry(position.parse().map_err(|e| format!("{line}: {e}"))?)
            .or_default()
            .push((
                rank.parse().map_err(|e| format!("{line}: {e}"))?,
                token_id.parse().map_err(|e| format!("{line}: {e}"))?,
                shifted.parse().map_err(|e| format!("{line}: {e}"))?,
            ));
    }
    let mut rows: Vec<GoldenRow> = grouped
        .into_iter()
        .map(|(position, mut ranked)| {
            ranked.sort_by_key(|(rank, _, _)| *rank);
            for (index, entry) in ranked.iter().enumerate() {
                assert_eq!(
                    entry.0, index,
                    "位置 {position} 的 rank 不连续（第 {index} 项标为 {}）",
                    entry.0
                );
            }
            GoldenRow {
                position,
                entries: ranked
                    .into_iter()
                    .map(|(_, token_id, shifted)| (token_id, shifted))
                    .collect(),
            }
        })
        .collect();
    rows.sort_by_key(|row| row.position);
    Ok((provenance, rows))
}

/// 读取冻结的输入 token id 序列（一行一个）。
fn read_ids() -> Result<Vec<i32>, String> {
    let text = std::fs::read_to_string(fixture("llamacpp_stories15M_ids.txt"))
        .map_err(|error| format!("无法读取 golden ids 文件: {error}"))?;
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            line.trim()
                .parse::<i32>()
                .map_err(|error| format!("ids 行 {line:?} 解析失败: {error}"))
        })
        .collect()
}

/// 取一行里前 k 大的 `(id, value)`，值降序、并列时 id 升序。
///
/// 并列时的打破规则与 golden 导出脚本一致（按 id 升序），
/// 否则 top-16 集合相同但顺序不同会产生假失败。
fn top_k_desc(row: &[f32], k: usize) -> Vec<(i32, f32)> {
    let mut indices: Vec<usize> = (0..row.len()).collect();
    indices.sort_by(|&left, &right| row[right].total_cmp(&row[left]).then(left.cmp(&right)));
    indices
        .into_iter()
        .take(k)
        .map(|index| (index as i32, row[index]))
        .collect()
}

/// 减去本行最大值，得到跨实现可比的量。
///
/// `logit_i - logit_max == log_softmax_i - log_softmax_max`，归一化常数抵消，
/// 因此 llama.cpp 的 logprob 与我们 CLI 输出的原始 logits 可以直接逐值相减。
fn shift_by_max(row: &[f32]) -> Vec<f32> {
    let peak = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    row.iter().map(|value| value - peak).collect()
}

/// 真实模型 + golden 头部指纹校验。未设置环境变量时返回 `None`（用例跳过）。
fn load_verified_model() -> Option<(GgufFile, Vec<i32>, Vec<GoldenRow>)> {
    let path = PathBuf::from(std::env::var_os("QLLAMA_REAL_MODEL")?)
        .canonicalize()
        .expect("QLLAMA_REAL_MODEL 指向的文件不存在");
    let (provenance, rows) = read_golden().expect("golden 文件解析失败");
    let ids = read_ids().expect("ids 文件解析失败");

    // 只比字节数：足以挡住"指向了另一个模型"这个真实失误，又不引入哈希依赖。
    // 完整 sha256 记在头部 `model_sha256`，需要时可人工核对。
    let expected_bytes: u64 = provenance["model_bytes"]
        .parse()
        .expect("golden 头部 model_bytes 应为整数");
    let actual_bytes = std::fs::metadata(&path).expect("无法 stat 模型").len();
    assert_eq!(
        actual_bytes, expected_bytes,
        "模型字节数与 golden 头部不符，基准与模型对不上（{} vs {}）",
        actual_bytes, expected_bytes
    );

    let gguf = GgufFile::from_path(&path).expect("GGUF 解析失败");
    Some((gguf, ids, rows))
}

/// 末位置那一行 logits。
fn last_row(logits: &qllama::tensor::Tensor, sequence_len: usize) -> Vec<f32> {
    let columns = logits.dims()[1];
    assert_eq!(logits.dims()[0], sequence_len);
    logits.data()[(sequence_len - 1) * columns..sequence_len * columns].to_vec()
}

#[test]
fn golden_fixture_is_self_consistent_and_sized() {
    // 结构损坏（列错位、rank 不连续、缺来源、值序颠倒）必须在这里暴露，
    // 而不是等到真机比对时表现为"莫名偏差"。
    let (provenance, rows) = read_golden().expect("golden 解析失败");
    for key in [
        "model",
        "model_bytes",
        "model_sha256",
        "llamacpp",
        "command",
        "request",
        "measured",
    ] {
        assert!(
            provenance.contains_key(key),
            "golden 头部缺少来源字段 {key}，这份数据无法追溯"
        );
    }
    assert!(
        provenance["llamacpp"].contains("b12109"),
        "应记录 llama.cpp 构建号: {}",
        provenance["llamacpp"]
    );
    // f16 KV cache 会把偏差抬到 1e-2 量级，命令里必须能看到 f32 cache。
    assert!(
        provenance["command"].contains("-ctk f32") && provenance["command"].contains("-ctv f32"),
        "golden 必须来自精确 f32 cache 配置: {}",
        provenance["command"]
    );

    let ids = read_ids().expect("ids 解析失败");
    assert_eq!(ids.len(), 75);
    assert_eq!(
        ids[0], 1,
        "首个 id 应为 BOS(<s>)=1，与 stories15M 元数据一致"
    );
    for &id in &ids {
        assert!((0..32_000).contains(&id), "id {id} 越界（n_vocab=32000）");
    }

    assert_eq!(
        rows.iter().map(|row| row.position).collect::<Vec<_>>(),
        GOLDEN_POSITIONS.to_vec()
    );
    for row in &rows {
        assert_eq!(
            row.entries.len(),
            GOLDEN_TOP,
            "位置 {} 应有 {GOLDEN_TOP} 条",
            row.position
        );
        assert!(
            row.entries[0].1.to_bits() == 0.0f32.to_bits(),
            "位置 {} rank 0 的 shifted logprob 必须恰为 0（它减的是自身最大值）",
            row.position
        );
        for (rank, entry) in row.entries.iter().enumerate() {
            assert!(
                entry.1.is_finite(),
                "位置 {} rank {rank} 含非有限值",
                row.position
            );
            if rank > 0 {
                assert!(
                    row.entries[rank - 1].1 > entry.1,
                    "位置 {} 的 shifted 值必须严格递减: {:?} -> {:?}",
                    row.position,
                    row.entries[rank - 1],
                    entry
                );
            }
        }
        let unique: HashSet<i32> = row.entries.iter().map(|entry| entry.0).collect();
        assert_eq!(
            unique.len(),
            row.entries.len(),
            "位置 {} 的 top-{GOLDEN_TOP} 有重复 id",
            row.position
        );
        // position 必须是合法前缀长度-1。
        assert!(row.position < ids.len());
    }
}

#[test]
fn golden_positions_match_prefix_lengths() {
    // golden 的 position 语义是"前缀 ids[..=position] 的末位置"，与
    // `llama-server` 请求里 `prompt: ids[:p+1]` 一致。这里把该约定钉住：
    // 每个冻结 position 都能取到对应长度的前缀，且互不重复。
    let ids = read_ids().expect("ids 解析失败");
    let (_provenance, rows) = read_golden().expect("golden 解析失败");
    let unique: HashSet<usize> = rows.iter().map(|row| row.position).collect();
    assert_eq!(unique.len(), rows.len(), "position 不应重复");
    for row in &rows {
        let prefix = &ids[..=row.position];
        assert_eq!(prefix.len(), row.position + 1);
        assert_eq!(prefix[0], ids[0]);
    }
    assert_eq!(
        rows.iter().map(|row| row.position).max().unwrap(),
        ids.len() - 1,
        "最大 position 应等于 ids 长度减一"
    );
}

#[test]
#[ignore = "需要 QLLAMA_REAL_MODEL 指向 golden 头部记录的那个模型"]
fn golden_logits_match_reference_for_every_frozen_position() {
    let Some((gguf, ids, rows)) = load_verified_model() else {
        eprintln!("跳过：未设置 QLLAMA_REAL_MODEL");
        return;
    };
    let model = Transformer::from_gguf(&gguf).expect("模型加载失败");
    let mut worst = 0.0f32;
    for row in &rows {
        let prefix = &ids[..=row.position];
        let logits = model.forward(prefix, 0).expect("整段前向失败");
        let shifted = shift_by_max(&last_row(&logits, prefix.len()));

        let argmax = top_k_desc(&shifted, 1)[0].0;
        assert_eq!(
            argmax, row.entries[0].0,
            "位置 {} 的 argmax 与 llama.cpp 不一致",
            row.position
        );
        // 实测两侧 top-16 的 id 与顺序完全一致，因此这里连次序一起断言。
        let ours: Vec<i32> = top_k_desc(&shifted, GOLDEN_TOP)
            .iter()
            .map(|entry| entry.0)
            .collect();
        let golden: Vec<i32> = row.entries.iter().map(|entry| entry.0).collect();
        assert_eq!(
            ours, golden,
            "位置 {} 的 top-{GOLDEN_TOP} 顺序与 llama.cpp 不一致",
            row.position
        );

        let mut position_worst = 0.0f32;
        for &(token_id, expected) in &row.entries {
            let deviation = (shifted[token_id as usize] - expected).abs();
            position_worst = position_worst.max(deviation);
            assert!(
                deviation <= TOLERANCE,
                "位置 {} token {token_id} 偏差 {deviation:.3e} 超出容差 {TOLERANCE:.1e}",
                row.position
            );
        }
        worst = worst.max(position_worst);
        println!(
            "位置 {:>2}  argmax {:>5}  该位置最大偏差 {:.3e}",
            row.position, row.entries[0].0, position_worst
        );
    }
    println!(
        "{} 个位置全部通过，最大逐值偏差 {worst:.3e}（容差 {TOLERANCE:.1e}）",
        rows.len()
    );
}

#[test]
#[ignore = "需要 QLLAMA_REAL_MODEL 指向 golden 头部记录的那个模型"]
fn golden_greedy_chain_matches_reference_for_sixteen_steps() {
    // 贪心链比单点 logits 更严：每步 argmax 都会改变后续输入，任何算子误解
    // 都会在几步内跑偏（RoPE 配对方式错乱时首步 argmax 即从 13 翻到 450）。
    let Some((gguf, ids, _rows)) = load_verified_model() else {
        eprintln!("跳过：未设置 QLLAMA_REAL_MODEL");
        return;
    };
    let model = Transformer::from_gguf(&gguf).expect("模型加载失败");
    // llama.cpp b12109 对 ids 前 10 个 token（"<s> Once upon a time there was a
    // little cat"）的 16 步贪心续写，`--temp 0 --top-k 1 --seed 1`。
    // 该链在两种 KV cache 配置下完全一致：默认 `-ctk f16 -ctv f16`（端口 8975，
    // `--no-repack`）与精确的 `-ctk f32 -ctv f32`（端口 8976），所以它不受
    // golden logits 那个 1e-2 量级 cache 噪声的影响。
    const EXPECTED: [i32; 16] = [
        29889, 450, 6635, 471, 1407, 9074, 14793, 322, 5131, 304, 1284, 1554, 304, 17545, 29889,
        1105,
    ];
    let mut sequence: Vec<i32> = ids[..10].to_vec();
    let mut chain = Vec::new();
    for (step, &expected) in EXPECTED.iter().enumerate() {
        let logits = model.forward(&sequence, 0).expect("整段前向失败");
        let next = top_k_desc(&last_row(&logits, sequence.len()), 1)[0].0;
        assert_eq!(
            next,
            expected,
            "第 {} 步贪心 token 与 llama.cpp 不一致",
            step + 1
        );
        chain.push(next);
        sequence.push(next);
    }
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("分词器加载失败");
    println!(
        "贪心链 {} 个 token 逐位一致: {chain:?}\ndecoded: {:?}",
        chain.len(),
        tokenizer.decode(&chain)
    );
}

#[test]
#[ignore = "需要 QLLAMA_REAL_MODEL 指向 golden 头部记录的那个模型"]
fn tokenizer_matches_llamacpp_ids_for_frozen_prompts() {
    // 保留阶段 7 的三个固定 prompt 和全部既有性质断言；SPM 实现后进一步要求
    // 与当时 b12109 原始输出逐 id 相同，不再只检查解码等价与长度下界。
    // 期望 ids 不变，tests/golden 的冻结数据也不变。
    let Some((gguf, _ids, _rows)) = load_verified_model() else {
        eprintln!("跳过：未设置 QLLAMA_REAL_MODEL");
        return;
    };
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("分词器加载失败");
    let cases: [(&str, &[i32]); 3] = [
        (
            "Once upon a time, a little cat sat by the window and watched the rain.",
            &[
                1, 9038, 2501, 263, 931, 29892, 263, 2217, 6635, 3290, 491, 278, 3474, 322, 20654,
                278, 17251, 29889,
            ][..],
        ),
        (
            "The cat slept on the mat.",
            &[1, 450, 6635, 12844, 415, 373, 278, 1775, 29889][..],
        ),
        (
            "hello world 1234567890",
            &[
                1, 22172, 3186, 29871, 29896, 29906, 29941, 29946, 29945, 29953, 29955, 29947,
                29929, 29900,
            ][..],
        ),
    ];
    for (prompt, llamacpp_ids) in cases {
        let ours = tokenizer
            .encode_with_options(prompt, &EncodeOptions::default())
            .expect("编码失败");

        assert_eq!(
            ours, llamacpp_ids,
            "prompt {prompt:?}: SPM token ids must match llama.cpp"
        );

        // 保留原有文本级等价断言。
        assert_eq!(
            tokenizer.decode(&ours).expect("解码 ours 失败"),
            tokenizer
                .decode(llamacpp_ids)
                .expect("解码 llama.cpp 序列失败"),
            "prompt {prompt:?}: 两侧 id 必须解码成同一文本"
        );

        // 原有长度下界仍须成立，新增的逐 id 相等断言进一步收紧该契约。
        assert!(
            ours.len() >= llamacpp_ids.len(),
            "prompt {prompt:?}: 预期我们的序列不短于 llama.cpp（ours {} vs cpp {}）",
            ours.len(),
            llamacpp_ids.len()
        );
        println!(
            "分词对照 {prompt:?}: ours {} 个 id vs llama.cpp {} 个（文本等价）",
            ours.len(),
            llamacpp_ids.len()
        );
    }

    // 性质 2：byte/单字符 token 的索引与 llama.cpp 一致——数字串没有任何
    // 多字符词片段可匹配，两侧都只能走单 token，所以这段必须逐 id 相同。
    // 三条 prompt 里只有第三条以全数字结尾，因此单独取它验证。
    let digits_tail: &[i32] = &[
        29896, 29906, 29941, 29946, 29945, 29953, 29955, 29947, 29929, 29900,
    ];
    let ours_digits = tokenizer
        .encode_with_options(cases[2].0, &EncodeOptions::default())
        .expect("编码失败");
    assert_eq!(
        ours_digits[ours_digits.len() - digits_tail.len()..],
        digits_tail[..],
        "数字串的 id 序列应与 llama.cpp 逐 id 相同"
    );
    let frozen = cases[2].1;
    assert_eq!(
        frozen[frozen.len() - digits_tail.len()..],
        digits_tail[..],
        "冻结的 llama.cpp id 尾部应与记录的数字段一致"
    );
}
