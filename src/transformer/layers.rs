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
//! ## RoPE 约定（关键，接真实模型前必须复核）
//!
//! [`Attention`] 通过 [`crate::tensor::TensorOps::rope`] 施加旋转位置编码，
//! 该实现旋转**相邻元素对** `(x[0],x[1]), (x[2],x[3]), ...`，即 llama.cpp 的
//! `GGML_ROPE_TYPE_NORM`，已由上游源码确认是 `LLM_ARCH_LLAMA` 的行为。
//!
//! 但 HuggingFace 的 Llama 参考实现普遍使用 `rotate_half` 的**对半拆分**风格
//! `(x[0],x[d/2]), (x[1],x[d/2+1]), ...`，两者在 `d > 2` 时结果不同。
//!
//! **接真实 GGUF 模型时若输出乱码或数值对不上，首先检查此处。**

use crate::tensor::{Tensor, TensorOps};

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
/// 本阶段不含 KV Cache：一次前向处理整段序列，因果掩码直接作用在
/// `[seq_len, seq_len]` 的分数矩阵上。
///
/// ## `start_pos` 为何在本阶段不改变输出
///
/// RoPE 是旋转，故 q·k 只依赖**相对**位置：R(θ_q)q · R(θ_k)k = q·R(θ_k-θ_q)k。
/// 整段序列一起平移时相对位置不变，输出随之不变。接入 KV Cache 后，缓存中的
/// K/V 已按各自的历史位置旋转过，start_pos 才会真正参与新 query 的计算——因此
/// 该参数现在就按首 token 的绝对位置传入，避免后续再改语义。
#[derive(Debug)]
pub struct Attention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
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
        if num_heads == 0 || num_kv_heads == 0 || head_dim == 0 {
            return Err("Attention head counts and head_dim must be positive".to_string());
        }
        if !num_heads.is_multiple_of(num_kv_heads) {
            return Err(format!(
                "num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}"
            ));
        }
        let q_features = wq.weight().dims()[1];
        let kv_features = wk.weight().dims()[1];
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
/// `start_pos`，掩码条件与 `start_pos` 无关。
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

/// SwiGLU 前馈网络。
///
/// `down_proj(silu(gate_proj(x)) * up_proj(x))`，与 llama.cpp 的
/// `LLM_FFN_SILU` + `LLM_FFN_PAR` 组合一致。
#[derive(Debug)]
pub struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    /// 构造前馈网络。
    ///
    /// `gate_proj` 与 `up_proj` 的形状须一致，且 `down_proj` 的输入特征数
    /// 须等于二者的输出特征数。
    pub fn new(gate_proj: Linear, up_proj: Linear, down_proj: Linear) -> Result<Self, String> {
        let gate_shape = gate_proj.weight().dims().to_vec();
        if gate_shape != up_proj.weight().dims().to_vec() {
            return Err(format!(
                "gate_proj {:?} and up_proj {:?} must have the same shape",
                gate_shape,
                up_proj.weight().dims()
            ));
        }
        let intermediate = gate_shape[1];
        if down_proj.weight().dims()[0] != intermediate {
            return Err(format!(
                "down_proj expects {} input features but the intermediate size is {intermediate}",
                down_proj.weight().dims()[0]
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
        let gate = self.gate_proj.forward(input)?;
        let up = self.up_proj.forward(input)?;
        let hidden = gate.silu()?.mul(&up)?;
        self.down_proj.forward(&hidden)
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
}
