//! Transformer 基础层。
//!
//! ## 权重布局约定（关键）
//!
//! GGUF 中二维权重的 `dims` 按 ggml 的 `ne[]` 顺序存放，`ne[0]` 是输入特征数、
//! `ne[1]` 是输出特征数（见 `gguf::loader` 的说明）。加载器按内存原样反转为
//! `[out_features, in_features]` 后交给本模块。
//!
//! [`Linear`] 在构造时**一次性**转置为 `[in_features, out_features]`，使前向
//! 只需一次 `matmul`。选择在构造时转置而非每次前向转置，是因为后者会对每个
//! token 都复制一遍权重矩阵；构造期的一次性代价可以忽略。
//!
//! ## RoPE 配对方式（关键，已用 llama.cpp 与真实模型确认）
//!
//! [`Attention`] 通过 [`crate::tensor::TensorOps::rope`] 施加旋转位置编码，
//! 该实现旋转**相邻元素对** `(x[0],x[1]), (x[2],x[3]), ...`，即 llama.cpp 的
//! `GGML_ROPE_TYPE_NORMAL`。以下游源码为据（llama.cpp tag `b12110`，逐条核对过）：
//!
//! - `src/llama-model.cpp:3601` `llama_model_rope_type()` 里，llama 架构的 case
//!   列表起于 `:3631`（`case LLM_ARCH_LLAMA:` 及同组的 LLADA/LLAMA4/DECI...），
//!   `LLM_ARCH_LLAMA_EMBED` 在 `:3671`，整组统一的
//!   `return LLAMA_ROPE_TYPE_NORM;` 在 `:3678` —— llama 架构取 NORMAL，不是 NEOX。
//! - `ggml/src/ggml-cpu/ops.cpp:6419-6420` `case GGML_ROPE_TYPE_NORMAL:
//!   rotate_pairs<T>(n_dims, /*n_offset=*/1, cache, src + n_offs, dst_data + n_offs,
//!   /*scale=*/1);`；对照 `:6425` 的 NEOX 分支传 `n_offset = n_dims/2`。
//! - `ggml/src/ggml-cpu/ops.cpp:6276-6292` `rotate_pairs()`：`scale = 1` 使
//!   `ic = i0/scale = i0`，于是旋转的两个元素是 `src[i0]` 与 `src[i0 + 1]`
//!   ——相邻；`scale = 2` 时 `ic = i0/2`，两个元素相距 `n_offset = n_dims/2`。
//! - `ggml/include/ggml.h:1910-1913` 的示例表把这一点说得很直白：
//!   `head size 8` 下 `NORMAL n_dims=8 → [cscscscs]`，`NEOX n_dims=8 → [ccccssss]`。
//! - `ggml/include/ggml.h:250-251` 定义 `GGML_ROPE_TYPE_NORMAL 0`、
//!   `GGML_ROPE_TYPE_NEOX 2`。
//!
//! HuggingFace 的 Llama 参考实现用 `rotate_half` 的**对半拆分**
//! `(x[0],x[d/2]), (x[1],x[d/2+1]), ...`，等价于上游的 NEOX 分支，与 llama.cpp
//! 的 llama 模型不同。阶段 7 在真实训练模型（stories15M）上实测过这个区别：
//! 把本仓库实现改成 half-split 配对后，末位置 logits 的 argmax 从 13 翻到 450、
//! 逐元素最大偏差 9.12（判别性用例见 `tests/tensor_ops.rs` 的
//! `rope_rejects_hf_rotate_half_pairing`；模型指纹与复现命令记录在
//! `tests/golden/` 的 TSV 头部）。
//!
//! 角度本身仍可能有微小差别：上游 `ggml_rope_cache_init`（`ops.cpp:6190-6201`）用
//! f32 累乘 `theta *= theta_scale`（`ops.cpp:6356` 的
//! `theta_scale = powf(freq_base, -2.0f/n_dims)`），本实现在 f64 下直接取幂
//! `theta^(-2i/n_dims)`；差值在 f32 舍入量级，实测不影响 top-16 排序。
//! 分母 `n_dims` 即被旋转的维度数：本项目的配置层拒绝 `rope.dimension_count !=
//! head_dim` 的 partial RoPE（见 `transformer::config`），因此两者恒等。

use crate::gguf::MappedWeight;
use crate::kv_cache::KvCache;
use crate::tensor::{Tensor, TensorOps};
use rayon;

/// 词嵌入：按 token id 取权重行。
///
/// 权重布局为 `[vocab_size, hidden_size]`，与 GGUF 的 `token_embd.weight`
/// 反转维度后的形状一致。
#[derive(Debug)]
pub struct Embedding {
    /// `[vocab_size, hidden_size]`
    weight: Tensor,
}

impl Embedding {
    /// 构造词嵌入层；权重必须是二维。
    pub fn new(weight: Tensor) -> Result<Self, String> {
        if weight.ndim() != 2 {
            return Err(format!(
                "Embedding weight must be 2D, got rank {}",
                weight.ndim()
            ));
        }
        Ok(Self { weight })
    }

    /// 按 id 查表，输入 `[seq_len]`，输出 `[seq_len, hidden_size]`。
    ///
    /// id 为负数或超出词表范围时返回错误。空输入返回 `[0, hidden_size]`。
    pub fn forward(&self, input_ids: &[i32]) -> Result<Tensor, String> {
        let (vocab_size, hidden_size) = (self.weight.dims()[0], self.weight.dims()[1]);
        let mut data = Vec::with_capacity(input_ids.len() * hidden_size);

        for (position, &id) in input_ids.iter().enumerate() {
            let index = usize::try_from(id)
                .map_err(|_| format!("Token id {id} at position {position} is negative"))?;
            if index >= vocab_size {
                return Err(format!(
                    "Token id {id} at position {position} exceeds vocabulary size {vocab_size}"
                ));
            }
            let start = index * hidden_size;
            let row = self
                .weight
                .data()
                .get(start..start + hidden_size)
                .ok_or_else(|| format!("Embedding row {index} is out of range"))?;
            data.extend_from_slice(row);
        }

        Ok(Tensor::new(data, vec![input_ids.len(), hidden_size]))
    }

    /// 权重引用，供测试与调试使用。
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
}

/// 全连接层。
///
/// 构造时接收 GGUF 内存布局的 `[out_features, in_features]`，内部转置为
/// `[in_features, out_features]` 以便前向直接 `matmul`。具体理由见模块文档。
#[derive(Debug)]
pub struct Linear {
    /// `[in_features, out_features]`，由构造时的权重转置而来。
    weight: Tensor,
    /// `[out_features]`，可选。
    bias: Option<Tensor>,
}

impl Linear {
    /// 构造全连接层。
    ///
    /// `weight` 须为二维且形状是 `[out_features, in_features]`；`bias` 若给出，
    /// 其长度须等于 `out_features`。
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Result<Self, String> {
        if weight.ndim() != 2 {
            return Err(format!(
                "Linear weight must be 2D, got rank {}",
                weight.ndim()
            ));
        }
        let out_features = weight.dims()[0];
        if let Some(bias) = &bias {
            if bias.len() != out_features {
                return Err(format!(
                    "Linear bias length {} does not match out_features {out_features}",
                    bias.len()
                ));
            }
        }
        // 一次性转置：前向不再复制权重。
        let transposed = weight.transpose(0, 1)?;
        Ok(Self {
            weight: transposed,
            bias,
        })
    }

    /// 输入 `[seq_len, in_features]`，输出 `[seq_len, out_features]`。
    ///
    /// 借助 NumPy 风格广播，`[seq_len, in]` 的输入加上 `[out]` 的偏置即可得到
    /// `[seq_len, out]`。本阶段 `matmul` 只接受二维输入，因此高维输入直接报错，
    /// 由调用方先 `reshape`。
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, String> {
        if input.ndim() != 2 {
            return Err(format!(
                "Linear input must be 2D [seq_len, in_features], got rank {}",
                input.ndim()
            ));
        }
        let in_features = input.dims()[input.ndim() - 1];
        if in_features != self.weight.dims()[0] {
            return Err(format!(
                "Linear input has {in_features} features but the weight expects {}",
                self.weight.dims()[0]
            ));
        }
        let output = input.matmul(&self.weight)?;
        match &self.bias {
            Some(bias) => output.add(bias),
            None => Ok(output),
        }
    }

    /// 权重引用（已转置为 `[in_features, out_features]`），供测试使用。
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
}

// 既有 Tensor 层保留公开接口；映射权重通过枚举接入同一前向计算，不伪造 Tensor。
#[derive(Debug)]
pub(crate) enum Projection {
    Dense(Linear),
    Mapped(MappedWeight),
}

impl From<Linear> for Projection {
    fn from(value: Linear) -> Self {
        Self::Dense(value)
    }
}

impl Projection {
    pub(crate) fn mapped(weight: MappedWeight) -> Result<Self, String> {
        if weight.dims().len() != 2 {
            return Err(format!("Mapped weight must be 2D, got {:?}", weight.dims()));
        }
        Ok(Self::Mapped(weight))
    }

    pub(crate) fn dims(&self) -> [usize; 2] {
        match self {
            Self::Dense(layer) => [layer.weight().dims()[0], layer.weight().dims()[1]],
            Self::Mapped(weight) => [weight.dims()[1], weight.dims()[0]],
        }
    }

    pub(crate) fn forward(&self, input: &Tensor) -> Result<Tensor, String> {
        match self {
            Self::Dense(layer) => layer.forward(input),
            Self::Mapped(weight) => weight.matmul(input),
        }
    }
}

#[derive(Debug)]
pub(crate) enum TokenEmbedding {
    Dense(Embedding),
    Mapped(MappedWeight),
}

impl TokenEmbedding {
    pub(crate) fn dims(&self) -> &[usize] {
        match self {
            Self::Dense(layer) => layer.weight().dims(),
            Self::Mapped(weight) => weight.dims(),
        }
    }

    pub(crate) fn forward(&self, ids: &[i32]) -> Result<Tensor, String> {
        match self {
            Self::Dense(layer) => layer.forward(ids),
            Self::Mapped(weight) => {
                let width = weight.dims()[1];
                let mut values = Vec::with_capacity(ids.len() * width);
                for (position, &id) in ids.iter().enumerate() {
                    let row = usize::try_from(id)
                        .map_err(|_| format!("Token id {id} at position {position} is negative"))?;
                    if row >= weight.dims()[0] {
                        return Err(format!(
                            "Token id {id} at position {position} exceeds vocabulary size {}",
                            weight.dims()[0]
                        ));
                    }
                    values.extend(weight.read_row(row)?);
                }
                Ok(Tensor::new(values, vec![ids.len(), width]))
            }
        }
    }
}

/// RMSNorm 层。
#[derive(Debug)]
pub struct RmsNorm {
    /// `[hidden_size]`
    weight: Tensor,
    eps: f32,
}

impl RmsNorm {
    /// 构造归一化层；权重长度须与待归一化的最后一维一致。
    pub fn new(weight: Tensor, eps: f32) -> Result<Self, String> {
        if weight.ndim() != 1 {
            return Err(format!(
                "RmsNorm weight must be 1D, got rank {}",
                weight.ndim()
            ));
        }
        Ok(Self { weight, eps })
    }

    /// 沿最后一维归一化，输入输出形状相同。
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, String> {
        input.rms_norm(&self.weight, self.eps)
    }

    /// 权重引用，供测试使用。
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
}

/// 多头注意力。
///
/// 提供两条前向路径：
/// - [`Attention::forward`]：整段序列一次算完，因果掩码作用在
///   `[seq_len, seq_len]` 方阵上。它是增量路径的对照基准，行为保持不变。
/// - [`Attention::forward_with_cache`]：只算新 token，K/V 经 KV Cache 与历史
///   拼接，因果掩码是带 `start_pos` 偏移的矩形。
///
/// ## `start_pos` 的语义
///
/// RoPE 是旋转，故 q·k 只依赖**相对**位置：R(θ_q)q · R(θ_k)k = q·R(θ_k-θ_q)k。
/// 整段序列一起平移时相对位置不变，输出随之不变——所以整段路径中 `start_pos`
/// 不改变结果。但增量路径中缓存里的 K/V 已按各自的历史位置旋转过，`start_pos`
/// 决定了新 query/新 K 的绝对旋转角，也决定因果掩码的偏移，是真实参与计算的
/// 输入。两条路径都把它定义为"首 token 的绝对位置"。
#[derive(Debug)]
pub struct Attention {
    wq: Projection,
    wk: Projection,
    wv: Projection,
    wo: Projection,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_theta: f32,
}

impl Attention {
    /// 构造注意力层。
    ///
    /// `num_heads` 须能被 `num_kv_heads` 整除（分组查询注意力）；
    /// `head_dim * num_heads` 须等于 Q/O 投影的输出特征数。
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        wq: Linear,
        wk: Linear,
        wv: Linear,
        wo: Linear,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_theta: f32,
    ) -> Result<Self, String> {
        Self::from_projections(
            [wq.into(), wk.into(), wv.into(), wo.into()],
            [num_heads, num_kv_heads, head_dim],
            rope_theta,
        )
    }

    pub(crate) fn from_projections(
        projections: [Projection; 4],
        heads: [usize; 3],
        rope_theta: f32,
    ) -> Result<Self, String> {
        let [wq, wk, wv, wo] = projections;
        let [num_heads, num_kv_heads, head_dim] = heads;
        if num_heads == 0 || num_kv_heads == 0 || head_dim == 0 {
            return Err("Attention head counts and head_dim must be positive".to_string());
        }
        if !num_heads.is_multiple_of(num_kv_heads) {
            return Err(format!(
                "num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}"
            ));
        }
        let q_features = wq.dims()[1];
        let kv_features = wk.dims()[1];
        if q_features != num_heads * head_dim {
            return Err(format!(
                "Q projection outputs {q_features} features but num_heads*head_dim is {}",
                num_heads * head_dim
            ));
        }
        if kv_features != num_kv_heads * head_dim {
            return Err(format!(
                "K projection outputs {kv_features} features but num_kv_heads*head_dim is {}",
                num_kv_heads * head_dim
            ));
        }
        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_theta,
        })
    }

    /// 前向计算，输入 `[seq_len, hidden_size]`，输出 `[seq_len, hidden_size]`。
    ///
    /// 流程见模块文档与上游 `llama-graph.cpp` 的 `build_attn_mha`：
    /// 投影 → 拆头 → RoPE → 转成 head 主序 → 逐头缩放点积 + 因果掩码 + softmax
    /// → 加权和 → 合并头 → 输出投影。
    pub fn forward(&self, hidden_states: &Tensor, start_pos: usize) -> Result<Tensor, String> {
        if hidden_states.ndim() != 2 {
            return Err(format!(
                "Attention input must be 2D [seq_len, hidden_size], got rank {}",
                hidden_states.ndim()
            ));
        }
        let seq_len = hidden_states.dims()[0];
        let head_dim = self.head_dim;

        // 1. Q/K/V 投影。
        let q = self.wq.forward(hidden_states)?;
        let k = self.wk.forward(hidden_states)?;
        let v = self.wv.forward(hidden_states)?;

        // 2. 拆头：把最后一维的特征切成 num_heads 段，每段 head_dim。
        let q = q.reshape(vec![seq_len, self.num_heads, head_dim])?;
        let k = k.reshape(vec![seq_len, self.num_kv_heads, head_dim])?;
        let v = v.reshape(vec![seq_len, self.num_kv_heads, head_dim])?;

        // 3. RoPE。约定见模块文档的醒目提示。
        let q = q.rope(head_dim, start_pos, self.rope_theta)?;
        let k = k.rope(head_dim, start_pos, self.rope_theta)?;

        // 4. 转到 head 主序，便于按头取二维矩阵做 matmul。
        let q = q.transpose(0, 1)?;
        let k = k.transpose(0, 1)?;
        let v = v.transpose(0, 1)?;

        // 5. 逐头计算。matmul 只接受二维，因此这里按头切出子矩阵。
        let group = self.num_heads / self.num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        #[cfg(feature = "profile")]
        crate::profile::allocations(1);
        let mut heads = Vec::with_capacity(self.num_heads);
        for head in 0..self.num_heads {
            let q_h = head_matrix(&q, head, seq_len, head_dim)?;
            // GQA：query 头 h 复用 KV 头 h / group，与 ggml 的
            // flash attention（ik2 = iq2 / rk2）及 HF 的 repeat_kv 一致。
            let kv_head = head / group;
            let k_h = head_matrix(&k, kv_head, seq_len, head_dim)?;
            let v_h = head_matrix(&v, kv_head, seq_len, head_dim)?;

            let scores = q_h.matmul(&k_h.transpose(0, 1)?)?;
            let scores = scores.scale(scale)?;
            let scores = apply_causal_mask(&scores)?;
            let probabilities = scores.softmax(1)?;
            heads.push(probabilities.matmul(&v_h)?);
        }

        // 6. 合并头：第 h 个头占据输出特征的 [h*head_dim, (h+1)*head_dim) 区间，
        //    与上游 ggml_permute + ggml_cont_2d 的结果一致。按 (位置, 头) 的
        //    顺序直接铺开即可得到 [seq, hidden]，省掉一次转置拷贝。
        let mut merged_data = Vec::with_capacity(seq_len * self.num_heads * head_dim);
        for position in 0..seq_len {
            for head in &heads {
                let start = position * head_dim;
                merged_data.extend_from_slice(&head.data()[start..start + head_dim]);
            }
        }
        let merged = Tensor::new(merged_data, vec![seq_len, self.num_heads * head_dim]);

        // 7. 输出投影。
        self.wo.forward(&merged)
    }

    /// 增量前向：只计算新 token 的 Q/K/V，K/V 写入 KV Cache 后与历史拼接。
    ///
    /// 与 `forward` 的三处关键差异（除此之外算子序列完全一致）：
    /// 1. RoPE 历史不重算：`k` 只旋转本次新段（rope 的 `start_pos` 参数即绝对
    ///    位置），写入 cache 后原样读回；历史 K 在它自己的写入时刻已旋转过，
    ///    再旋转一次等于叠加两份旋转，结果必然错。
    /// 2. 因果掩码带偏移：分数矩阵是 `[new_len, total_len]` 矩形，第 i 个 query
    ///    只能看见绝对位置 `<= start_pos + i` 的 key，掩码条件含 offset。
    /// 3. 注意力键长是 `total_len = start_pos + new_len`，cache 读回的 K/V 头数
    ///    是 `num_kv_heads`（GQA 分组在取头时处理，与整段路径同一公式）。
    ///
    /// 一致性：prefill 阶段（`start_pos == 0`）算子序列与 `forward` 逐项相同；
    /// decode 阶段 softmax 只多出 `-inf → exp → 0` 的加零项，不改变数值。
    /// 两条路径的逐值一致性由 `tests/transformer_forward.rs` 的对照测试锁定。
    pub fn forward_with_cache(
        &self,
        hidden_states: &Tensor,
        start_pos: usize,
        kv_cache: &mut KvCache,
        layer_idx: usize,
    ) -> Result<Tensor, String> {
        if hidden_states.ndim() != 2 {
            return Err(format!(
                "Attention input must be 2D [seq_len, hidden_size], got rank {}",
                hidden_states.ndim()
            ));
        }
        let new_len = hidden_states.dims()[0];
        let head_dim = self.head_dim;

        let q = profile_call!(Query, self.wq.forward(hidden_states))?;
        let k = profile_call!(Key, self.wk.forward(hidden_states))?;
        let v = profile_call!(Value, self.wv.forward(hidden_states))?;

        let q = q.reshape(vec![new_len, self.num_heads, head_dim])?;
        let k = k.reshape(vec![new_len, self.num_kv_heads, head_dim])?;
        let v = v.reshape(vec![new_len, self.num_kv_heads, head_dim])?;
        let (q, k) = profile_call!(Rope, {
            (
                q.rope(head_dim, start_pos, self.rope_theta)?,
                k.rope(head_dim, start_pos, self.rope_theta)?,
            )
        });

        #[cfg(feature = "profile")]
        let attention_span = crate::profile::Span::new(crate::profile::Stage::Attention);

        // 4. 写入 cache，再按"全部历史 + 新段"读回。绝对位置换算 i32 的
        //    溢出与越界都由 KvCache 校验，这里先把转换错误显式报出。
        #[cfg(feature = "profile")]
        crate::profile::allocations(usize::from(new_len > 0));
        let positions: Vec<i32> = (0..new_len)
            .map(|offset| {
                let position = start_pos
                    .checked_add(offset)
                    .ok_or_else(|| format!("position {start_pos} + {offset} overflows usize"))?;
                i32::try_from(position).map_err(|_| {
                    format!("position {position} exceeds the i32 range the cache uses")
                })
            })
            .collect::<Result<Vec<i32>, String>>()?;
        kv_cache.update(layer_idx, &k, &v, &positions)?;

        let total_len = start_pos + new_len;
        #[cfg(feature = "profile")]
        crate::profile::allocations(usize::from(total_len > 0));
        let all_positions: Vec<i32> = (0..total_len)
            .map(|position| {
                i32::try_from(position).map_err(|_| {
                    format!("position {position} exceeds the i32 range the cache uses")
                })
            })
            .collect::<Result<Vec<i32>, String>>()?;
        let (k, v) = kv_cache.get(layer_idx, &all_positions)?;

        // 5. 转到 head 主序，与整段路径同构。
        let q = q.transpose(0, 1)?;
        let k = k.transpose(0, 1)?;
        let v = v.transpose(0, 1)?;

        // 6. 逐头计算：查询是 new_len 行，键是 total_len 列。
        let group = self.num_heads / self.num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        #[cfg(feature = "profile")]
        crate::profile::allocations(1);
        let mut heads = Vec::with_capacity(self.num_heads);
        for head in 0..self.num_heads {
            let q_h = head_matrix(&q, head, new_len, head_dim)?;
            let kv_head = head / group;
            let k_h = head_matrix(&k, kv_head, total_len, head_dim)?;
            let v_h = head_matrix(&v, kv_head, total_len, head_dim)?;

            let scores = q_h.matmul(&k_h.transpose(0, 1)?)?;
            let scores = scores.scale(scale)?;
            let scores = apply_causal_mask_offset(&scores, start_pos)?;
            let probabilities = scores.softmax(1)?;
            heads.push(probabilities.matmul(&v_h)?);
        }

        // 7. 合并头：布局与整段路径一致，只是行数是 new_len。
        let mut merged_data = Vec::with_capacity(new_len * self.num_heads * head_dim);
        for position in 0..new_len {
            for head in &heads {
                let start = position * head_dim;
                merged_data.extend_from_slice(&head.data()[start..start + head_dim]);
            }
        }
        let merged = Tensor::new(merged_data, vec![new_len, self.num_heads * head_dim]);

        #[cfg(feature = "profile")]
        drop(attention_span);
        profile_call!(Output, self.wo.forward(&merged))
    }
}

/// 从 `[heads, seq_len, head_dim]` 中取出第 `head` 个头的二维矩阵。
fn head_matrix(
    tensor: &Tensor,
    head: usize,
    seq_len: usize,
    head_dim: usize,
) -> Result<Tensor, String> {
    tensor
        .slice(0, head, head + 1)?
        .reshape(vec![seq_len, head_dim])
}

/// 施加因果掩码：分数矩阵中 `j > i` 的位置置为 `-inf`。
///
/// 只处理整段序列前向（无 KV Cache），因此查询与键的绝对位置相差同样的
/// `start_pos`，掩码条件与 `start_pos` 无关。带 offset 的增量场景走
/// [`apply_causal_mask_offset`]。
///
/// 保留方阵守卫：这里只有整段路径调用，输入恒为方阵；未来若有人误把增量的
/// 矩形矩阵接到本函数，方阵守卫会立刻报错，而不是按 `j > i` 静默错切历史列。
fn apply_causal_mask(scores: &Tensor) -> Result<Tensor, String> {
    if scores.ndim() != 2 || scores.dims()[0] != scores.dims()[1] {
        return Err(format!(
            "Causal mask expects a square 2D score matrix, got {:?}",
            scores.dims()
        ));
    }
    let seq_len = scores.dims()[0];
    let mut masked = scores.clone();
    for query in 0..seq_len {
        for key in (query + 1)..seq_len {
            masked.set(&[query, key], f32::NEG_INFINITY)?;
        }
    }
    Ok(masked)
}

/// 因果掩码的带偏移形式：查询行 `i` 对应绝对位置 `query_offset + i`，
/// 只能看见绝对位置 `<= query_offset + i` 的键列。
///
/// 增量前向的分数矩阵是 `[new_len, total_len]` 矩形：query 只覆盖新段，
/// key 覆盖整段历史，两侧位置差了一个 `start_pos`。此时**不能**沿用整段
/// 路径的 `j > i`——那会把历史列整体错切，把 query 该看的 key 掩掉、
/// 不该看的未来 key 漏掉。
///
/// 要求 `cols >= rows + query_offset`：键列比因果窗口还短，说明调用方给的
/// 位置关系不自洽，报错而不是静默减少可见列。
fn apply_causal_mask_offset(scores: &Tensor, query_offset: usize) -> Result<Tensor, String> {
    if scores.ndim() != 2 {
        return Err(format!(
            "Causal mask expects a 2D score matrix, got rank {}",
            scores.ndim()
        ));
    }
    let (rows, cols) = (scores.dims()[0], scores.dims()[1]);
    if cols < rows + query_offset {
        return Err(format!(
            "Causal mask got {:?}: {cols} key column(s) cannot cover \
             {rows} query row(s) starting at offset {query_offset}",
            scores.dims()
        ));
    }
    let mut masked = scores.clone();
    for query in 0..rows {
        for key in (query_offset + query + 1)..cols {
            masked.set(&[query, key], f32::NEG_INFINITY)?;
        }
    }
    Ok(masked)
}

/// SwiGLU 前馈网络。
///
/// `down_proj(silu(gate_proj(x)) * up_proj(x))`，与 llama.cpp 的
/// `LLM_FFN_SILU` + `LLM_FFN_PAR` 组合一致。
#[derive(Debug)]
pub struct Mlp {
    gate_proj: Projection,
    up_proj: Projection,
    down_proj: Projection,
}

impl Mlp {
    /// 构造前馈网络。
    ///
    /// `gate_proj` 与 `up_proj` 的形状须一致，且 `down_proj` 的输入特征数
    /// 须等于二者的输出特征数。
    pub fn new(gate_proj: Linear, up_proj: Linear, down_proj: Linear) -> Result<Self, String> {
        Self::from_projections(gate_proj.into(), up_proj.into(), down_proj.into())
    }

    pub(crate) fn from_projections(
        gate_proj: Projection,
        up_proj: Projection,
        down_proj: Projection,
    ) -> Result<Self, String> {
        let gate_shape = gate_proj.dims();
        if gate_shape != up_proj.dims() {
            return Err(format!(
                "gate_proj {:?} and up_proj {:?} must have the same shape",
                gate_shape,
                up_proj.dims()
            ));
        }
        let intermediate = gate_shape[1];
        if down_proj.dims()[0] != intermediate {
            return Err(format!(
                "down_proj expects {} input features but the intermediate size is {intermediate}",
                down_proj.dims()[0]
            ));
        }
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// 输入 `[seq_len, hidden_size]`，输出 `[seq_len, hidden_size]`。
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, String> {
        // gate 和 up 投影相互独立，使用 rayon::join 并行执行以节省时间
        let (gate, up) = rayon::join(
            || profile_call!(Gate, self.gate_proj.forward(input)),
            || profile_call!(Up, self.up_proj.forward(input)),
        );
        let gate = gate?;
        let up = up?;
        let hidden = profile_call!(Activation, gate.silu()?.mul(&up))?;
        profile_call!(Down, self.down_proj.forward(&hidden))
    }
}

/// 单个 Transformer 层。
#[derive(Debug)]
pub struct TransformerBlock {
    /// 注意力前的归一化。
    pub input_layernorm: RmsNorm,
    /// 自注意力。
    pub self_attn: Attention,
    /// 前馈网络前的归一化。
    pub post_attention_layernorm: RmsNorm,
    /// SwiGLU 前馈网络。
    pub mlp: Mlp,
}

impl TransformerBlock {
    /// 前向计算，输入输出均为 `[seq_len, hidden_size]`。
    ///
    /// 残差结构与上游 `models/llama.cpp` 一致：先加注意力残差，再加前馈残差。
    pub fn forward(&self, hidden_states: &Tensor, start_pos: usize) -> Result<Tensor, String> {
        let normed = self.input_layernorm.forward(hidden_states)?;
        let attended = self.self_attn.forward(&normed, start_pos)?;
        let residual = hidden_states.add(&attended)?;

        let normed = self.post_attention_layernorm.forward(&residual)?;
        let fed_forward = self.mlp.forward(&normed)?;
        residual.add(&fed_forward)
    }

    /// 增量前向：`hidden_states` 只含新 token 的隐状态，形状
    /// `[new_len, hidden_size]`。
    ///
    /// 残差连接与前馈部分与 `forward` 逐行相同（Mlp 无状态，直接复用），
    /// 差异全部集中在注意力的 cache 读写上，见
    /// [`Attention::forward_with_cache`]。`layer_idx` 是本层在
    /// [`KvCache`] 中的槽位下标，由调用方按层序传入。
    pub fn forward_with_cache(
        &self,
        hidden_states: &Tensor,
        start_pos: usize,
        kv_cache: &mut KvCache,
        layer_idx: usize,
    ) -> Result<Tensor, String> {
        let normed = profile_call!(RmsNorm, self.input_layernorm.forward(hidden_states))?;
        let attended = self
            .self_attn
            .forward_with_cache(&normed, start_pos, kv_cache, layer_idx)?;
        let residual = profile_call!(Residual, hidden_states.add(&attended))?;

        let normed = profile_call!(RmsNorm, self.post_attention_layernorm.forward(&residual))?;
        let fed_forward = self.mlp.forward(&normed)?;
        profile_call!(Residual, residual.add(&fed_forward))
    }
}
