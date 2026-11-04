"""用 NumPy/f64 独立重写 Llama 架构前向，作为 Qllama 与 llama.cpp 之外的第三方对照。

动机：项目里已有的 f64 参考实现与本仓库共用同一批 Rust 语义假设，llama.cpp 又是
另一个团队的实现；当"我们 vs llama.cpp"出现 1e-2 量级偏差时，需要一个**独立写出**
的、精度明确更高的实现来判断偏差属于谁。本脚本只做一件事：按 llama.cpp 的 Llama
架构语义（相邻对 RoPE、GQA 分组、SwiGLU、RMSNorm 把 eps 加在均值之内）在 f64 下
算末行 logits，然后与两侧比较。

如果 f64 参考与 llama.cpp 的偏差和它与 Qllama 的偏差同量级，说明残差是 f32
累加精度差异（正常）；如果 f64 参考明显更靠近其中一侧，另一侧就有需要定位的问题。

依赖：``numpy``、``gguf``（``pip install gguf``），仅用于分析，不是生产依赖。

用法::

    python scripts/numpy_forward_reference.py \\
        --model path/to/stories15M.gguf \\
        --ids-file ids.txt --positions 10,30,52,66,69,74 \\
        --server http://127.0.0.1:8975/completion

`--server` 给出时同时与 llama.cpp 对照；省略时只打印 f64 参考自身的结果。
"""

import argparse
import json
import sys
import urllib.request

import numpy as np
from gguf import GGUFReader


def load_model(path):
    """读取 GGUF，返回 (hparams dict, 权重 dict)，权重全部转 f64。

    gguf 里二维权重按 `ne = [in_features, out_features]` 存储，gguf-py 的 `.data`
    已经反转为 `(out_features, in_features)` 的行主序数组，因此线性层一律
    `y = x @ W.T`（与 llama.cpp `build_lora_mm` 的语义一致）。
    """
    reader = GGUFReader(path)

    def field(key):
        # gguf-py 把字段拆成 [偏移, 名称, 类型, 数量, ..., 值] 的 parts 列表，
        # 对标量而言最后一个元素才是值本身。
        entry = reader.fields.get(key)
        return None if entry is None else np.asarray(entry.parts[-1]).reshape(-1)

    def scalar(key, default=None):
        value = field(key)
        return default if value is None else int(value[0])

    def number(key, default):
        value = field(key)
        return default if value is None else float(value[0])

    weights = {}
    for tensor in reader.tensors:
        array = np.asarray(tensor.data)
        if array.dtype != np.float32:
            raise ValueError("本参考实现只支持 F32 权重，%s 是 %s" % (tensor.name, array.dtype))
        weights[tensor.name] = array.astype(np.float64)

    n_embd = scalar("llama.embedding_length")
    n_head = scalar("llama.attention.head_count")
    lm_head = weights.get("output.weight", weights["token_embd.weight"])
    hparams = {
        "n_layers": scalar("llama.block_count"),
        "n_vocab": lm_head.shape[0],
        "n_embd": n_embd,
        "n_head": n_head,
        # 与 llama.cpp 一致：缺失时 n_head_kv = n_head（llama-model.cpp:1556-1559）。
        "n_head_kv": scalar("llama.attention.head_count_kv", n_head),
        "n_ffn": scalar("llama.feed_forward_length"),
        # GGUF 缺省即整头旋转，与 llama.cpp 的 n_rot = n_embd_head_size 默认一致。
        "n_rot": scalar("llama.rope.dimension_count", n_embd // n_head),
        "rms_eps": number("llama.attention.layer_norm_rms_epsilon", 1e-5),
        "rope_theta": number("llama.rope.freq_base", 10000.0),
        "bos": scalar("tokenizer.ggml.bos_token_id", 1),
    }
    return hparams, weights


def rms_norm(x, weight, eps):
    """RMSNorm：`x_i / sqrt(mean(x^2) + eps) * w_i`，eps 加在均值之内。"""
    scale = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / scale) * weight


def rope(positions, hparams, n_rot):
    """按 llama.cpp `GGML_ROPE_TYPE_NORMAL` 生成 (cos, sin)，形状 [n, n_rot//2]。

    上游 `rotate_pairs(n_dims, n_offset=1, ..., scale=1)` 旋转相邻对，第 i 对的角频率
    是 `theta^(-2i/n_dims)`（`ggml_rope_cache_init` 里以 `theta *= theta_scale` 累乘，
    本函数直接用幂，精度更高，差值在 f32 舍入量级）。分母用被旋转维度数 `n_rot`
    而不是 head_dim，与上游 `powf(freq_base, -2.0f/n_dims)` 一致；Qllama 的配置层
    拒绝 `n_rot != head_dim` 的 partial RoPE，因此两者在本模型上取值相同。
    """
    freqs = hparams["rope_theta"] ** (-2.0 * np.arange(n_rot // 2, dtype=np.float64) / n_rot)
    angles = positions[:, None].astype(np.float64) * freqs[None, :]
    return np.cos(angles), np.sin(angles)


def rotate_rows(x, cos, sin, n_rot):
    """对 [n, heads, head_dim] 的前 n_rot 维做相邻对旋转，其余维原样保留。"""
    rotated = x[..., :n_rot]
    pairs = rotated.reshape(rotated.shape[0], rotated.shape[1], n_rot // 2, 2)
    first, second = pairs[..., 0], pairs[..., 1]
    out = np.empty_like(pairs)
    out[..., 0] = first * cos[:, None, :] - second * sin[:, None, :]
    out[..., 1] = first * sin[:, None, :] + second * cos[:, None, :]
    rotated = out.reshape(x.shape[0], x.shape[1], n_rot)
    if n_rot == x.shape[-1]:
        return rotated
    return np.concatenate([rotated, x[..., n_rot:]], axis=-1)


def forward(hparams, weights, ids):
    """f64 前向，返回 [n, n_vocab] 的 logits。"""
    n = len(ids)
    head_dim = hparams["n_embd"] // hparams["n_head"]
    n_rot = head_dim if hparams["n_rot"] in (0, None) else hparams["n_rot"]
    x = weights["token_embd.weight"][np.asarray(ids)].copy()
    positions = np.arange(n)
    cos, sin = rope(positions, hparams, n_rot)
    kv_groups = hparams["n_head"] // hparams["n_head_kv"]

    for layer in range(hparams["n_layers"]):
        prefix = "blk.%d." % layer
        h = rms_norm(x, weights[prefix + "attn_norm.weight"], hparams["rms_eps"])
        q = (h @ weights[prefix + "attn_q.weight"].T).reshape(n, hparams["n_head"], head_dim)
        k = (h @ weights[prefix + "attn_k.weight"].T).reshape(n, hparams["n_head_kv"], head_dim)
        v = (h @ weights[prefix + "attn_v.weight"].T).reshape(n, hparams["n_head_kv"], head_dim)
        q, k = rotate_rows(q, cos, sin, n_rot), rotate_rows(k, cos, sin, n_rot)

        merged = np.zeros((n, hparams["n_head"], head_dim), dtype=np.float64)
        mask = np.triu(np.ones((n, n), dtype=bool), 1)
        for head in range(hparams["n_head"]):
            kv_head = head // kv_groups
            scores = (q[:, head] @ k[:, kv_head].T) / np.sqrt(head_dim)
            scores = np.where(mask, -np.inf, scores)
            scores -= scores.max(axis=-1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(axis=-1, keepdims=True)
            merged[:, head] = probs @ v[:, kv_head]
        attn = merged.reshape(n, hparams["n_embd"]) @ weights[prefix + "attn_output.weight"].T
        x = x + attn

        h = rms_norm(x, weights[prefix + "ffn_norm.weight"], hparams["rms_eps"])
        gate = h @ weights[prefix + "ffn_gate.weight"].T
        up = h @ weights[prefix + "ffn_up.weight"].T
        x = x + ((gate / (1.0 + np.exp(-gate))) * up) @ weights[prefix + "ffn_down.weight"].T

    x = rms_norm(x, weights["output_norm.weight"], hparams["rms_eps"])
    return x @ weights["output.weight"].T


def fetch_llamacpp(url, ids, positions, n_vocab):
    """取 llama.cpp 的 f32 logprob 行并转成 `logit - logit_max`。"""
    rows = {}
    for position in positions:
        body = json.dumps({
            "prompt": ids[: position + 1], "n_predict": 0, "temperature": 1.0,
            "top_k": 0, "top_p": 1.0, "min_p": 0.0, "typ_p": 1.0, "repeat_penalty": 1.0,
            "seed": 1, "n_probs": 1 << 20, "cache_prompt": False,
        }).encode("utf-8")
        request = urllib.request.Request(url, data=body,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.loads(response.read().decode("utf-8"))
        top = payload["completion_probabilities"][-1]["top_logprobs"]
        vector = np.full(n_vocab, -np.inf, dtype=np.float64)
        for item in top:
            vector[item["id"]] = item["logprob"]
        if np.isfinite(vector).sum() != n_vocab:
            raise ValueError("位置 %d 只取到 %d/%d 个 logprob"
                             % (position, np.isfinite(vector).sum(), n_vocab))
        rows[position] = vector - vector.max()
    return rows


def top_k(vector, k=5):
    order = np.argsort(-vector, kind="stable")[:k]
    return [(int(i), float(vector[i])) for i in order]


def greedy(hparams, weights, ids, steps):
    """f64 贪心解码，返回新生成的 token id 列表。

    这是本脚本自身的正确性校验：只有实现完全正确，才可能复现 llama.cpp 的贪心链
    （每一步都依赖上一步的 argmax，任何算子/布局/eps 位置的误解都会很快跑偏）。
    """
    sequence = list(ids)
    generated = []
    for _ in range(steps):
        logits = forward(hparams, weights, sequence)
        last = logits[-1]
        token = int(np.argmax(last))
        generated.append(token)
        sequence.append(token)
    return generated


def run_qllama(qllama_bin, model, ids, out_path=None):
    """驱动 Qllama CLI 取末行 logits（f32 实现侧）。

    输出文件默认落在系统临时目录并随即删除：早先版本写到当前目录，
    在仓库里跑一次就留下一个 `qllama_ref.txt`。
    """
    import os
    import subprocess
    import tempfile

    scratch = out_path or os.path.join(tempfile.mkdtemp(prefix="qllama-ref-"), "logits.txt")
    command = [os.path.abspath(qllama_bin), "--model", os.path.abspath(model),
               "--tokens", ",".join(str(i) for i in ids), "--logits", os.path.abspath(scratch)]
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("qllama 退出码 %d: %s" % (proc.returncode, proc.stderr))
    with open(scratch, encoding="utf-8") as handle:
        values = np.array([float(line) for line in handle if line.strip()], dtype=np.float64)
    if out_path is None:
        os.unlink(scratch)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--ids-file", required=True, help="逗号分隔的 token id 文件")
    parser.add_argument("--positions", default="all", help="all 或 10,30,52")
    parser.add_argument("--server", help="llama-server /completion 地址（可选）")
    parser.add_argument("--greedy", type=int, default=0, help="f64 贪心解码步数（自校验用）")
    parser.add_argument("--qllama-bin", help="qllama 可执行文件，加入三方比较")
    args = parser.parse_args()

    ids = [int(x) for x in open(args.ids_file, encoding="utf-8").read().replace("\n", ",").split(",") if x.strip()]
    hparams, weights = load_model(args.model)
    print("hparams:", {k: v for k, v in hparams.items() if k != "n_vocab"},
          "n_vocab=%d" % hparams["n_vocab"])

    if args.greedy:
        chain = greedy(hparams, weights, ids, args.greedy)
        print("f64 参考贪心链 (%d): %s" % (args.greedy, chain))

    positions = list(range(len(ids))) if args.positions == "all" else [int(p) for p in args.positions.split(",")]
    logits = forward(hparams, weights, ids)
    print("f64 参考 logits 形状 %s" % (logits.shape,))

    ref = fetch_llamacpp(args.server, ids, positions, hparams["n_vocab"]) if args.server else {}
    print("三方逐值偏差（各自减最大值；f64 = 本脚本，rust = Qllama，cpp = llama.cpp）：")
    for position in positions:
        ours = logits[position]
        ours = ours - ours.max()
        parts = ["pos %-3d f64 top-5 %s" % (position, [i for i, _ in top_k(ours)])]
        if position in ref:
            cpp = ref[position]
            finite = np.isfinite(cpp)
            parts.append("f64↔cpp %.3e" % np.abs(ours - cpp)[finite].max())
        if args.qllama_bin:
            rust = run_qllama(args.qllama_bin, args.model, ids[: position + 1])
            rust = rust - rust.max()
            parts.append("rust top-5 %s" % [i for i, _ in top_k(rust)])
            parts.append("f64↔rust %.3e" % np.abs(ours - rust).max())
            if position in ref:
                parts.append("rust↔cpp %.3e" % np.abs(rust - ref[position])[np.isfinite(ref[position])].max())
        print("  " + " | ".join(parts))


if __name__ == "__main__":
    main()
