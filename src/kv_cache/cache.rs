//! KV Cache：缓存并增量复用各注意力层的 K/V 张量。
//!
//! 缓存存的是**已施加 RoPE** 的 K 与未施加位置编码的 V。这条约定来自 RoPE 的核心
//! 性质"注意力只依赖相对位置"：增量路径在写入前把新段旋转一次，读回时**绝不再旋转**
//! ——旧 K 在它自己的写入时刻已经带上了对应位置的旋转，重复旋转等于叠加两份旋转。
//! llama.cpp 的 `k_cache` 采用同一约定。
//!
//! 单层缓存形状为 `[max_seq_len, num_kv_heads, head_dim]`，头数按模型的 GQA 分组取
//! **KV 头数**（例如 32 个 query 头共享 8 个 KV 头的配置，每组 4 个 query 头复用
//! 同一个 KV 头；Llama-3-8B 即此配置）。MHA 模型两者相等（Llama-7B 为 32/32）。
//!
//! 容量成本：layers × max_seq_len × num_kv_heads × head_dim × 2(K+V) × 4 字节。
//! Llama-7B 规模（32 层、cap 2048、32 KV 头、head_dim 128、f32）约 2 GB；
//! 同层数/上下文但只有 8 个 KV 头的 GQA 模型约 512 MiB。因此容量必须按实际上下文
//! 长度预分配，不要默认取模型标称最大值；一次性小量生成时把容量缩到
//! prompt 长度 + 生成量更经济。

use crate::tensor::Tensor;

/// 单个层的 KV Cache。
struct KvCacheLayer {
    /// `[max_seq_len, num_kv_heads, head_dim]`，已施加 RoPE 的 K
    k_cache: Tensor,
    /// `[max_seq_len, num_kv_heads, head_dim]`，原始 V，无位置编码
    v_cache: Tensor,
    /// 已写入的前缀长度，从位置 0 起连续计数
    current_len: usize,
}

/// 全部层的 KV Cache，各层独立维护 `current_len`。
pub struct KvCache {
    layers: Vec<KvCacheLayer>,
    max_seq_len: usize,
    /// KV 头数（GQA 下不是 query 头数），见 `new`
    num_kv_heads: usize,
    head_dim: usize,
}

impl KvCache {
    /// 创建 `num_layers` 层 × `[max_seq_len, num_kv_heads, head_dim]` 的 KV Cache。
    ///
    /// `num_kv_heads` 的语义是 **KV 头数**，不是配置里的 query 头数
    /// （`num_attention_heads`）：cache 存的是 K/V。GQA 模型（如 32 query 头 / 8 KV 头）
    /// 传 query 头数既浪费内存（4 倍），也无法与 Attention 写出的 K
    /// `[new_len, num_kv_heads, head_dim]` 形状匹配，`update` 会直接以形状错误拒绝。
    /// 调用方应取 `TransformerConfig::num_kv_heads`。
    pub fn new(
        num_layers: usize,
        max_seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let layers = (0..num_layers)
            .map(|_| KvCacheLayer {
                k_cache: Tensor::zeros(vec![max_seq_len, num_kv_heads, head_dim]),
                v_cache: Tensor::zeros(vec![max_seq_len, num_kv_heads, head_dim]),
                current_len: 0,
            })
            .collect();

        Self {
            layers,
            max_seq_len,
            num_kv_heads,
            head_dim,
        }
    }

    /// 把一批新 K/V 写入 `layer_idx` 层的指定绝对位置。
    ///
    /// `k`/`v` 必须是 `[new_len, num_kv_heads, head_dim]` 且头数与 head_dim 和 cache
    /// 一致；`positions.len()` 必须等于 `new_len`；位置从该层当前 `current_len` 起连续
    /// 递增、无空洞（`positions[i] == current_len + i`），且每个值非负、小于
    /// `max_seq_len`。越界、回退、跳洞一律报错，绝不静默截断或写到回绕后的位置。
    /// prefill（从 0 起）与 decode（接在 `current_len` 后）遵循同一条规则。
    ///
    /// `new_len == 0`（空批，positions 也为空）是合法的空操作：不写任何数据，
    /// `current_len` 不变。
    ///
    /// **事务性**：所有校验都在任何写入之前完成，写入不可能中途失败；`current_len`
    /// 只在整批写完后推进。即便某个防御性写入错误被触发，由于 `current_len` 未推进，
    /// 刚写入的残余行也落在前缀之外，对 `get` 永远不可见。
    pub fn update(
        &mut self,
        layer_idx: usize,
        k: &Tensor,
        v: &Tensor,
        positions: &[i32],
    ) -> Result<(), String> {
        if layer_idx >= self.layers.len() {
            return Err(format!(
                "KV cache layer index {layer_idx} out of range (cache has {} layers)",
                self.layers.len()
            ));
        }
        let layer = &mut self.layers[layer_idx];

        // 形状校验：k/v 必须匹配 cache 的 `[_, num_kv_heads, head_dim]`。
        // GQA 下用 query 头数误建的 cache 会在这里被拦下，而不是读的时候才发现。
        let k_dims = k.dims();
        if k_dims.len() != 3 || k_dims[1] != self.num_kv_heads || k_dims[2] != self.head_dim {
            return Err(format!(
                "K shape {k_dims:?} does not match cache [_, {}, {}]",
                self.num_kv_heads, self.head_dim
            ));
        }
        let v_dims = v.dims();
        if v_dims != k_dims {
            return Err(format!("K/V shape mismatch: K {k_dims:?}, V {v_dims:?}"));
        }
        let new_len = k_dims[0];

        // 位置校验，全部先于写入：
        // - 长度必须与 new_len 一致，否则调用方意图不明确
        // - 必须从 current_len 接续（同时覆盖回退、跳洞、重复 prefill）
        // - 越界用有符号先判负，避免负数 as usize 后变成巨数
        if positions.len() != new_len {
            return Err(format!(
                "K batch is {new_len} token(s) but got {} position(s)",
                positions.len()
            ));
        }
        for (index, &position) in positions.iter().enumerate() {
            let position = usize::try_from(position)
                .map_err(|_| format!("positions[{index}] = {position} is negative"))?;
            let expected = layer.current_len.checked_add(index).ok_or_else(|| {
                format!(
                    "positions[{index}] overflows usize from current_len {}",
                    layer.current_len
                )
            })?;
            if position != expected {
                return Err(format!(
                    "positions[{index}] = {position} must continue from current_len \
                     (expected {expected}); skipping, backward, or double writes are not allowed"
                ));
            }
            if position >= self.max_seq_len {
                return Err(format!(
                    "positions[{index}] = {position} exceeds cache capacity {}",
                    self.max_seq_len
                ));
            }
        }

        for (token, &position) in positions.iter().enumerate() {
            let position = position as usize;
            for head in 0..self.num_kv_heads {
                for dim in 0..self.head_dim {
                    let coords = [token, head, dim];
                    // 形状已在前面的校验中锁定，按坐标读取不会失败；
                    // 若出现内部不一致，报错而不是 panic。
                    let k_value = k.get(&coords).ok_or_else(|| {
                        format!("K is missing coordinate [{token}, {head}, {dim}]")
                    })?;
                    let v_value = v.get(&coords).ok_or_else(|| {
                        format!("V is missing coordinate [{token}, {head}, {dim}]")
                    })?;
                    layer.k_cache.set(&[position, head, dim], k_value)?;
                    layer.v_cache.set(&[position, head, dim], v_value)?;
                }
            }
        }

        // 整批写完，此时才推进长度
        if let Some(&last) = positions.last() {
            layer.current_len = layer.current_len.max(last as usize + 1);
        }
        Ok(())
    }

    /// 读出指定位置的 K/V，返回 `[positions.len(), num_kv_heads, head_dim]`。
    ///
    /// 位置必须落在 `[0, current_len)`：前缀之外是初始零数组的内容，直接喂给注意力
    /// 等同于伪造"token 什么都没关注到"的 K/V，因此报错而不是静默返回零。空
    /// `positions` 返回 `[0, num_kv_heads, head_dim]` 空张量（空批注意力是合法的
    /// 空操作）。
    ///
    /// **行序契约**：返回的第 i 行对应 `positions[i]`，与请求顺序一致；允许乱序、
    /// 允许重复位置（重复请求返回重复行）。布局与整段路径 reshape 出的
    /// `[seq, num_kv_heads, head_dim]` 完全一致，调用方无需特殊处理。
    ///
    /// 性能说明：本阶段按位置逐行复制（每个位置拷 `num_kv_heads * head_dim` 个元素），
    /// 历史读出的成本是 O(total_len) 拷贝；性能阶段可改为非连续视图避免拷贝。
    pub fn get(&self, layer_idx: usize, positions: &[i32]) -> Result<(Tensor, Tensor), String> {
        if layer_idx >= self.layers.len() {
            return Err(format!(
                "KV cache layer index {layer_idx} out of range (cache has {} layers)",
                self.layers.len()
            ));
        }
        let layer = &self.layers[layer_idx];

        // 每个位置读一整行（num_kv_heads * head_dim 个元素）。cache 的扁平布局就是
        // 按 [position, head, dim] 行主序，因此逐位置做连续切片拷贝即可；边界已校验。
        let row = self.num_kv_heads * self.head_dim;
        let mut k_data = Vec::with_capacity(positions.len() * row);
        let mut v_data = Vec::with_capacity(positions.len() * row);
        for (index, &position) in positions.iter().enumerate() {
            let position = usize::try_from(position)
                .map_err(|_| format!("positions[{index}] = {position} is negative"))?;
            if position >= layer.current_len {
                return Err(format!(
                    "positions[{index}] = {position} is at or beyond the written prefix \
                     (current_len {})",
                    layer.current_len
                ));
            }
            let start = position * row;
            let end = start + row;
            k_data.extend_from_slice(
                layer
                    .k_cache
                    .data()
                    .get(start..end)
                    .ok_or_else(|| format!("K cache is missing rows {start}..{end}"))?,
            );
            v_data.extend_from_slice(
                layer
                    .v_cache
                    .data()
                    .get(start..end)
                    .ok_or_else(|| format!("V cache is missing rows {start}..{end}"))?,
            );
        }

        let dims = vec![positions.len(), self.num_kv_heads, self.head_dim];
        Ok((Tensor::new(k_data, dims.clone()), Tensor::new(v_data, dims)))
    }

    /// 清空所有层已写入的前缀，把 `current_len` 重置为 0。
    ///
    /// 只重置长度，不清零 K/V 缓冲：`clear` 之后 `get` 会按 `current_len` 规则拒绝
    /// 全部读取，旧数据不可达；后续 `update` 会逐位置覆盖。
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.current_len = 0;
        }
    }

    /// 首层已写入的前缀长度。
    ///
    /// 正常使用中每层按同一顺序、同一位置更新，各层该值一致；用它做边界断言比
    /// 逐层查询更直接。没有任何层的 cache 返回 0。
    pub fn seq_len(&self) -> usize {
        self.layers.first().map(|l| l.current_len).unwrap_or(0)
    }
}
