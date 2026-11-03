//! KvCache 读写契约测试：形状/位置校验、事务性与 clear 语义。
//!
//! 预期值全部来自写入时使用的确定性模式（`staged_tensor`），不依赖任何实现细节。

use qllama::kv_cache::KvCache;
use qllama::Tensor;

/// 生成 `[len, heads, dim]` 的 KV 张量，元素值 = token*100 + head*10 + dim。
///
/// `token_offset` 让不同批次写进同一 cache 的数据仍可区分，用来验证
/// update 到底写到了哪个绝对位置。
fn staged_tensor(len: usize, heads: usize, dim: usize, token_offset: usize) -> Tensor {
    let mut data = Vec::with_capacity(len * heads * dim);
    for token in 0..len {
        for head in 0..heads {
            for d in 0..dim {
                data.push(((token_offset + token) * 100 + head * 10 + d) as f32);
            }
        }
    }
    Tensor::new(data, vec![len, heads, dim])
}

/// 取 cache 的 K 张量并断言每个坐标的值。
fn assert_k_at(cache: &KvCache, layer: usize, positions: &[i32], heads: usize, dim: usize) {
    let (k, v) = cache.get(layer, positions).expect("get 应成功");
    let expected_dims = vec![positions.len(), heads, dim];
    assert_eq!(k.dims(), expected_dims.as_slice());
    assert_eq!(v.dims(), expected_dims.as_slice());
    for (row, &position) in positions.iter().enumerate() {
        for head in 0..heads {
            for d in 0..dim {
                let coords = [row, head, d];
                let expected = (position as usize * 100 + head * 10 + d) as f32;
                assert_eq!(k.get(&coords), Some(expected), "K {coords:?}");
                assert_eq!(v.get(&coords), Some(expected), "V {coords:?}");
            }
        }
    }
}

#[test]
fn update_then_get_returns_written_values() {
    let mut cache = KvCache::new(2, 16, 4, 8);
    let k = staged_tensor(3, 4, 8, 0);
    let v = staged_tensor(3, 4, 8, 0);
    cache.update(0, &k, &v, &[0, 1, 2]).expect("连续写入应成功");
    assert_eq!(cache.seq_len(), 3);
    assert_k_at(&cache, 0, &[0, 1, 2], 4, 8);
}

#[test]
fn second_batch_appends_after_current_len() {
    let mut cache = KvCache::new(1, 16, 2, 4);
    cache
        .update(
            0,
            &staged_tensor(2, 2, 4, 0),
            &staged_tensor(2, 2, 4, 0),
            &[0, 1],
        )
        .expect("首批写入应成功");
    cache
        .update(
            0,
            &staged_tensor(2, 2, 4, 2),
            &staged_tensor(2, 2, 4, 2),
            &[2, 3],
        )
        .expect("续写应成功");
    assert_eq!(cache.seq_len(), 4);
    // token_offset=2 让第二三批的 token 值分别为 200/300 起头，断言 verify 落点
    assert_k_at(&cache, 0, &[0, 1, 2, 3], 2, 4);
    let (k, _) = cache.get(0, &[2]).expect("取单个位置应成功");
    assert_eq!(k.get(&[0, 0, 0]), Some(200.0));
}

#[test]
fn get_with_empty_positions_returns_zero_len() {
    let mut cache = KvCache::new(1, 8, 2, 4);
    cache
        .update(
            0,
            &staged_tensor(1, 2, 4, 0),
            &staged_tensor(1, 2, 4, 0),
            &[0],
        )
        .expect("写入应成功");
    let (k, v) = cache.get(0, &[]).expect("空 positions 应成功");
    assert_eq!(k.dims(), [0, 2, 4]);
    assert_eq!(v.dims(), [0, 2, 4]);
}

#[test]
fn clear_resets_prefix_and_hides_stale_data() {
    let mut cache = KvCache::new(1, 8, 2, 4);
    cache
        .update(
            0,
            &staged_tensor(2, 2, 4, 0),
            &staged_tensor(2, 2, 4, 0),
            &[0, 1],
        )
        .expect("写入应成功");
    cache.clear();
    assert_eq!(cache.seq_len(), 0);
    let error = cache
        .get(0, &[0])
        .expect_err("clear 后读任何位置都应被拒绝");
    assert!(
        error.contains("written prefix"),
        "错误应说明越过了已写前缀，实际: {error}"
    );
    // 清空后应从 0 重新开始，而不是把 current_len 卡在旧值
    cache
        .update(
            0,
            &staged_tensor(1, 2, 4, 0),
            &staged_tensor(1, 2, 4, 0),
            &[0],
        )
        .expect("clear 后重新 prefill 应成功");
    assert_eq!(cache.seq_len(), 1);
}

#[test]
fn update_rejects_shape_mismatches() {
    let mut cache = KvCache::new(1, 8, 2, 4);
    let good = staged_tensor(1, 2, 4, 0);

    // KV 头数不符：用 4 个头写 2 个头的 cache
    let wide = staged_tensor(1, 4, 4, 0);
    let error = cache
        .update(0, &wide, &good, &[0])
        .expect_err("头数不符应报错");
    assert!(
        error.contains("[1, 4, 4]") && error.contains("[_, 2, 4]"),
        "实际: {error}"
    );

    // head_dim 不符
    let deep = staged_tensor(1, 2, 6, 0);
    cache
        .update(0, &deep, &good, &[0])
        .expect_err("head_dim 不符应报错");

    // K/V 互相不一致
    cache
        .update(0, &good, &deep, &[0])
        .expect_err("K/V 形状不一致应报错");

    // 不是三维
    let flat = Tensor::new(vec![1.0; 8], vec![2, 4]);
    cache
        .update(0, &flat, &good, &[0])
        .expect_err("非三维 K 应报错");
}

#[test]
fn update_rejects_positions_length_mismatch() {
    let mut cache = KvCache::new(1, 8, 2, 4);
    let k = staged_tensor(2, 2, 4, 0);
    let v = staged_tensor(2, 2, 4, 0);
    let error = cache
        .update(0, &k, &v, &[0])
        .expect_err("位置数与新 token 数不符应报错");
    assert!(
        error.contains("2 token(s)") && error.contains("1 position"),
        "实际: {error}"
    );
}

#[test]
fn update_rejects_gap_backward_and_double_prefill() {
    let mut cache = KvCache::new(1, 8, 2, 4);
    cache
        .update(
            0,
            &staged_tensor(2, 2, 4, 0),
            &staged_tensor(2, 2, 4, 0),
            &[0, 1],
        )
        .expect("首批写入应成功");

    // 从 current_len=2 起跳洞
    let error = cache
        .update(
            0,
            &staged_tensor(1, 2, 4, 0),
            &staged_tensor(1, 2, 4, 0),
            &[3],
        )
        .expect_err("跳洞应报错");
    assert!(error.contains("must continue"), "实际: {error}");

    // 回退写旧位置
    cache
        .update(
            0,
            &staged_tensor(1, 2, 4, 0),
            &staged_tensor(1, 2, 4, 0),
            &[1],
        )
        .expect_err("回退应报错");

    // 重复 prefill（位置 0 已被占用）
    cache
        .update(
            0,
            &staged_tensor(1, 2, 4, 0),
            &staged_tensor(1, 2, 4, 0),
            &[0],
        )
        .expect_err("重复 prefill 应报错");

    // 批次内乱序同样不行：严格递增才允许
    cache
        .update(
            0,
            &staged_tensor(2, 2, 4, 0),
            &staged_tensor(2, 2, 4, 0),
            &[2, 4],
        )
        .expect_err("批次内乱序应报错");
}

#[test]
fn update_rejects_out_of_range_and_negative_positions() {
    let mut cache = KvCache::new(1, 4, 2, 4);
    // 先填满 4 个位置，current_len == 容量
    cache
        .update(
            0,
            &staged_tensor(4, 2, 4, 0),
            &staged_tensor(4, 2, 4, 0),
            &[0, 1, 2, 3],
        )
        .expect("填满容量应成功");

    // 位置 4：连续性上正好接在末尾，越界只能由容量检查拦下
    let error = cache
        .update(
            0,
            &staged_tensor(1, 2, 4, 0),
            &staged_tensor(1, 2, 4, 0),
            &[4],
        )
        .expect_err("位置等于容量应报错");
    assert!(error.contains("capacity 4"), "实际: {error}");

    // 远大于容量在空 cache 上先被连续性检查拦下（跳洞），同样拒绝
    let mut fresh = KvCache::new(1, 4, 2, 4);
    fresh
        .update(
            0,
            &staged_tensor(1, 2, 4, 0),
            &staged_tensor(1, 2, 4, 0),
            &[99],
        )
        .expect_err("远超容量应报错");

    // 负数：若被 as usize 截断成巨数同样会越界，但必须按负位置报错而不是绕回
    fresh
        .update(
            0,
            &staged_tensor(1, 2, 4, 0),
            &staged_tensor(1, 2, 4, 0),
            &[-1],
        )
        .expect_err("负位置应报错");
}

#[test]
fn update_rejects_unknown_layer() {
    let mut cache = KvCache::new(1, 8, 2, 4);
    let error = cache
        .update(
            1,
            &staged_tensor(1, 2, 4, 0),
            &staged_tensor(1, 2, 4, 0),
            &[0],
        )
        .expect_err("layer_idx 越界应报错");
    assert!(
        error.contains("layer index 1") && error.contains("1 layers"),
        "实际: {error}"
    );
}

#[test]
fn failed_update_does_not_advance_current_len() {
    let mut cache = KvCache::new(1, 8, 2, 4);
    cache
        .update(
            0,
            &staged_tensor(2, 2, 4, 0),
            &staged_tensor(2, 2, 4, 0),
            &[0, 1],
        )
        .expect("首批写入应成功");

    // 形状错误发生在任何写入之前
    cache
        .update(
            0,
            &staged_tensor(1, 4, 4, 0),
            &staged_tensor(1, 2, 4, 0),
            &[2],
        )
        .expect_err("形状错误应被拒绝");
    assert_eq!(cache.seq_len(), 2, "失败的 update 不得推进 current_len");

    // 前缀仍然只有 [0, 1]，位置 2 不可见
    cache
        .get(0, &[2])
        .expect_err("失败的 update 不得留下可见数据");
    assert_k_at(&cache, 0, &[0, 1], 2, 4);

    // 被拒绝的位置 2 之后仍能正常写入
    cache
        .update(
            0,
            &staged_tensor(1, 2, 4, 2),
            &staged_tensor(1, 2, 4, 2),
            &[2],
        )
        .expect("被拒绝后继续写应成功");
    assert_eq!(cache.seq_len(), 3);
    assert_k_at(&cache, 0, &[0, 1, 2], 2, 4);
}

#[test]
fn get_rejects_negative_out_of_range_and_unknown_layer() {
    let mut cache = KvCache::new(2, 8, 2, 4);
    cache
        .update(
            0,
            &staged_tensor(2, 2, 4, 0),
            &staged_tensor(2, 2, 4, 0),
            &[0, 1],
        )
        .expect("写入应成功");

    cache.get(0, &[-1]).expect_err("负位置应报错");
    cache.get(0, &[8]).expect_err("超出容量应报错");
    cache.get(0, &[2]).expect_err("越过已写前缀应报错");
    let error = cache.get(2, &[0]).expect_err("layer_idx 越界应报错");
    assert!(error.contains("layer index 2"), "实际: {error}");
}

#[test]
fn layers_are_independent() {
    let mut cache = KvCache::new(2, 8, 2, 4);
    cache
        .update(
            0,
            &staged_tensor(2, 2, 4, 0),
            &staged_tensor(2, 2, 4, 0),
            &[0, 1],
        )
        .expect("layer 0 写入应成功");
    cache
        .update(
            1,
            &staged_tensor(1, 2, 4, 0),
            &staged_tensor(1, 2, 4, 0),
            &[0],
        )
        .expect("layer 1 写入应成功");

    // seq_len 只看 layer 0；layer 1 只有 1 个位置，位置 1 不可见
    assert_eq!(cache.seq_len(), 2);
    assert_k_at(&cache, 1, &[0], 2, 4);
    cache.get(1, &[1]).expect_err("layer 1 的位置 1 未写入");
}

#[test]
fn cache_without_layers_reports_zero_and_rejects_access() {
    let mut cache = KvCache::new(0, 8, 2, 4);
    assert_eq!(cache.seq_len(), 0);
    cache.get(0, &[0]).expect_err("没有层时 get 应报错");
    cache
        .update(
            0,
            &staged_tensor(1, 2, 4, 0),
            &staged_tensor(1, 2, 4, 0),
            &[0],
        )
        .expect_err("没有层时 update 应报错");
}

#[test]
fn clear_resets_every_layer_even_when_lengths_diverge() {
    // 各层 current_len 可以不同（见 layers_are_independent），clear 必须逐层重置。
    // 若退化为"只清首层"，layer 1 的旧前缀仍可被 get 读到——正是 clear 文档
    // 承诺"stale data is unreachable"被破坏的场景。
    let mut cache = KvCache::new(2, 8, 2, 4);
    cache
        .update(
            0,
            &staged_tensor(3, 2, 4, 0),
            &staged_tensor(3, 2, 4, 0),
            &[0, 1, 2],
        )
        .expect("layer 0 写入应成功");
    cache
        .update(
            1,
            &staged_tensor(1, 2, 4, 0),
            &staged_tensor(1, 2, 4, 0),
            &[0],
        )
        .expect("layer 1 写入应成功");

    cache.clear();
    assert_eq!(cache.seq_len(), 0);
    cache
        .get(0, &[0])
        .expect_err("clear 后 layer 0 的旧数据应不可达");
    cache
        .get(1, &[0])
        .expect_err("clear 后 layer 1 的旧数据应不可达");

    // 两层都能从 0 重新 prefill
    for layer in 0..2 {
        cache
            .update(
                layer,
                &staged_tensor(1, 2, 4, 0),
                &staged_tensor(1, 2, 4, 0),
                &[0],
            )
            .expect("clear 后重新写入应成功");
    }
    assert_eq!(cache.seq_len(), 1);
    assert_k_at(&cache, 1, &[0], 2, 4);
}

#[test]
fn get_returns_rows_in_request_order_and_allows_duplicates() {
    // 行序契约：第 i 行对应 positions[i]。乱序请求若被实现排序/聚簇，
    // 会拿到错误的行序；重复位置应返回重复行而不是去重。
    let mut cache = KvCache::new(1, 8, 2, 4);
    cache
        .update(
            0,
            &staged_tensor(3, 2, 4, 0),
            &staged_tensor(3, 2, 4, 0),
            &[0, 1, 2],
        )
        .expect("写入应成功");

    let (k, v) = cache.get(0, &[2, 0]).expect("乱序请求应成功");
    assert_eq!(k.dims(), [2, 2, 4]);
    // 第 0 行必须是位置 2 的数据（200 起），第 1 行是位置 0（0 起）
    for head in 0..2 {
        for d in 0..4 {
            assert_eq!(k.get(&[0, head, d]), Some((200 + head * 10 + d) as f32));
            assert_eq!(k.get(&[1, head, d]), Some((head * 10 + d) as f32));
            assert_eq!(v.get(&[0, head, d]), Some((200 + head * 10 + d) as f32));
        }
    }

    let (k_dup, _) = cache.get(0, &[1, 1]).expect("重复位置应成功");
    assert_eq!(k_dup.dims(), [2, 2, 4]);
    for head in 0..2 {
        for d in 0..4 {
            let expected = Some((100 + head * 10 + d) as f32);
            assert_eq!(k_dup.get(&[0, head, d]), expected, "重复行第 0 行");
            assert_eq!(k_dup.get(&[1, head, d]), expected, "重复行第 1 行");
        }
    }
}

#[test]
fn empty_batch_update_is_a_noop() {
    // new_len == 0 是合法空操作：不写数据、不推进 current_len、不报错。
    // 空批会经 forward_with_cache 的 embed/注意力空路径到达这里（例如
    // 调用方传入空 token 列表），语义必须固定，而不是靠实现恰好没崩。
    let mut cache = KvCache::new(1, 8, 2, 4);
    cache
        .update(
            0,
            &staged_tensor(2, 2, 4, 0),
            &staged_tensor(2, 2, 4, 0),
            &[0, 1],
        )
        .expect("首批写入应成功");

    let empty = staged_tensor(0, 2, 4, 0);
    cache
        .update(0, &empty, &empty, &[])
        .expect("空批应作为空操作成功");
    assert_eq!(cache.seq_len(), 2, "空批不得推进 current_len");
    assert_k_at(&cache, 0, &[0, 1], 2, 4);
}
