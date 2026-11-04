#!/usr/bin/env python3
"""比较 Qllama 与 llama.cpp 在同一 token 序列上产出的 logits。

两侧输入必须是**完全相同的 token id 序列**，否则比较的是分词器而不是算子：

* Qllama 侧：``qllama --model M.gguf --tokens <ids> --logits out.txt``
  写出末位置一行 logits，每行一个 f32（``%.9e``），按 token id 升序。
* llama.cpp 侧：``llama-perplexity --model M.gguf -f p.txt -c 64 -b 8
  --save-all-logits out.bin``，本脚本解析其自定义二进制格式。

llama.cpp logits 文件格式（b12110 ``tools/perplexity/perplexity.cpp``）：

    偏移  长度                          内容
    0     8 字节                        魔数 "_logits_"                (:466)
    8     int32 × 3                     n_ctx, n_vocab, n_chunk        (:467, :521-523)
    20    int32 × (n_chunk * n_ctx)     每 chunk 实际参与计算的 token   (:522)
    ...   每 chunk ``n_ctx - n_ctx/2 - 1`` 行，行布局见下

每行是 ``nv = 2 * ((n_vocab + 1) // 2) + 4`` 个 uint16：前 4 个 uint16 是两个
float32 ``(scale, min_log_prob)``，其后 ``n_vocab`` 个量化码。编码由
``log_softmax()`` 完成（:93-108）：``scale = (max_logit - min_logit) / 65535``，
``code_i = round((logit_i - min_logit) / scale)``，因此

    logit_i - logit_max = (code_i - 65535) * scale

码 65535 按构造恰好落在 ``max_logit`` 上，所以**不需要知道 logsumexp 就能还原
相对 logits**。注意 :83 处 ``min_logit = max(min_logit, max_logit - 16)``：低于
``max - 16`` 的 logits 全部塌缩成码 0，无法逐值区分，本脚本把这些位置从偏差统计
中排除并单独计数。

相对 logits 而非原始 logits 是唯一可比的量：``logit_i - logit_max`` 与两侧的
softmax 归一化常数无关，也不受 Qllama 只写单行、llama.cpp 按 chunk 写多行的影响。

两种参考来源，精度差别很大，优先用第一种：

* **llama-server**（f32，逐值全精度）：``/completion`` 接受 token id 数组，
  ``n_predict=0`` 时只对最后一个 prompt token 产出 logits，``n_probs`` 足够大时
  返回整个词表的 ``log_softmax`` 值。取两条侧都同意的 ``logit - logit_max`` 即可
  逐值比较（``log_softmax`` 只是 logits 减同一个常数）。
* **llama-perplexity dump**（65536 级线性量化，本模型实测步长 1e-5）：见上面的
  文件格式说明。受"输入必须 ≥ 2×n_ctx"限制，只有 chunk 后半段的位置有 dump。

用法::

    # 全行扫描（server 参考，脚本自己驱动 Qllama）
    python scripts/compare_logits.py \\
        --server http://127.0.0.1:8971/completion \\
        --ids 1,450,7483,310,3444,338 --positions all \\
        --qllama-bin target/release/qllama.exe --model path/to/model.gguf

    # 单行比较（Qllama 已把末位置 logits 写到文本文件）
    python scripts/compare_logits.py --llamacpp llcpp_logits.bin --row -1 \\
        --qllama qllama63.txt [--tolerance 1e-2]

    # 导出回归基准 TSV（含来源注释）
    python scripts/compare_logits.py --server ... --ids ... --positions 5,62 \\
        --export-golden tests/golden/llamacpp_xxx.tsv \\
        --provenance model tiny-f32.gguf --provenance command "curl ..."

退出码：0 表示 argmax 与 top-5 一致且逐值偏差在容差内；1 表示存在不一致；
2 表示输入文件或 HTTP 响应解析失败。
"""

import argparse
import os
import struct
import sys
import tempfile

MAGIC = b"_logits_"
TOP_K = 5


def load_llamacpp_rows(path):
    """解析 llama.cpp ``--save-all-logits`` 文件。

    Args:
        path: logits 二进制文件路径。

    Returns:
        列表，每个元素为 ``(chunk_index, position, prompt_ids, shifted_logits)``，
        其中 ``shifted_logits`` 是 ``logit - max_logit`` 的 float 列表，
        ``position`` 是该行对应的绝对位置（0 基，即该行是在预测 ``position + 1``）。

    Raises:
        ValueError: 魔数不符或文件长度与声明的 chunk/vocab 尺寸不一致。
    """
    with open(path, "rb") as handle:
        blob = handle.read()
    if blob[:8] != MAGIC:
        raise ValueError("不是 llama.cpp logits 文件（魔数 %r）" % blob[:8])
    n_ctx, n_vocab, n_chunk = struct.unpack_from("<iii", blob, 8)
    tokens = struct.unpack_from("<%di" % (n_chunk * n_ctx), blob, 20)
    nv = 2 * ((n_vocab + 1) // 2) + 4
    row_bytes = nv * 2
    rows_per_chunk = n_ctx - n_ctx // 2 - 1
    first = n_ctx // 2
    expected = 20 + 4 * n_chunk * n_ctx + rows_per_chunk * n_chunk * row_bytes
    if len(blob) != expected:
        raise ValueError(
            "文件长度 %d 与按 n_ctx=%d/n_vocab=%d/n_chunk=%d 推算的 %d 不符"
            % (len(blob), n_ctx, n_vocab, n_chunk, expected)
        )

    rows = []
    offset = 20 + 4 * n_chunk * n_ctx
    for chunk in range(n_chunk):
        # 每个 chunk 独占一个上下文窗口：KV cache 在 chunk 之间被清空，位置从 0
        # 重新编号，因此 chunk k 的第 i 行的绝对位置就是 first + i，前缀是
        # tokens[k*n_ctx : k*n_ctx + first + i + 1]。
        chunk_ids = tokens[chunk * n_ctx : (chunk + 1) * n_ctx]
        for i in range(rows_per_chunk):
            base = offset + (chunk * rows_per_chunk + i) * row_bytes
            scale, _min_log_prob = struct.unpack_from("<ff", blob, base)
            codes = struct.unpack_from("<%dH" % n_vocab, blob, base + 8)
            shifted = [(c - 65535.0) * scale for c in codes]
            rows.append((chunk, first + i, chunk_ids, shifted))
    return rows


def load_qllama_row(path, n_vocab=None):
    """读取 Qllama ``--logits`` 文本输出并按需校验长度。

    Args:
        path: 每行一个 ``%.9e`` 的文本文件。
        n_vocab: 期望的向量长度；None 表示不校验长度（行数即结果）。

    Returns:
        ``logit`` 原始值的 float 列表。

    Raises:
        ValueError: 行数与 n_vocab 不符（给出时）或存在无法解析的行。
    """
    with open(path, "r", encoding="utf-8") as handle:
        values = [float(line) for line in handle if line.strip()]
    if n_vocab is not None and len(values) != n_vocab:
        raise ValueError("Qllama logits 行数 %d != n_vocab %d" % (len(values), n_vocab))
    return values


def top_k(logits, k=TOP_K):
    """返回前 k 名的 ``(token_id, logit)``，按 logit 降序、id 升序打破平局。"""
    order = sorted(range(len(logits)), key=lambda i: (-logits[i], i))
    return [(i, logits[i]) for i in order[:k]]


def compare(shifted_ours, shifted_ref, tolerance):
    """比较两侧相对 logits，返回结论字典。

    两侧输入都已减去各自的最大值，因此差值直接可比；``shifted_ref`` 中码 0 的位置
    是 llama.cpp 侧被 16 的窗口截断的结果（真实 logit 只会更低），不参与逐值统计。

    Args:
        shifted_ours: Qllama 的 ``logit - max``。
        shifted_ref: llama.cpp 还原的 ``logit - max``。
        tolerance: 逐值绝对偏差容差。

    Returns:
        dict：argmax/top-5 一致性、参与统计的元素数、落在窗口下沿的数、最大/平均偏差、
        超出容差的数量。
    """
    ours_top = top_k(shifted_ours)
    ref_top = top_k(shifted_ref)
    # 码 0 的位置即 shifted 的最小值：可能是真实 argmin（logits 跨度 < 16 时），也可能
    # 是被 16 窗口截断的一整批 token。两种情况下 llama.cpp 都给不出确定值，一律排除。
    floor = min(shifted_ref)
    diffs = []
    at_floor = 0
    for ours, ref in zip(shifted_ours, shifted_ref):
        if ref == floor:
            at_floor += 1
            continue
        diffs.append(abs(ours - ref))
    over = [d for d in diffs if d > tolerance]
    # 归一化偏差：跨实现比较时 logits 的绝对尺度由模型决定（玩具模型 logits 只有
    # ±0.3，真实模型可以跨几十），因此除以自己的 logit 跨度（shifted 里 0 - floor）
    # 才能横向比较"这一致性有多硬"。
    span = -floor if floor < 0 else float("nan")
    worst = max(range(len(diffs)), key=lambda i: diffs[i]) if diffs else None
    return {
        "argmax_ours": ours_top[0][0],
        "argmax_ref": ref_top[0][0],
        "argmax_match": ours_top[0][0] == ref_top[0][0],
        "top5_ours": ours_top,
        "top5_ref": ref_top,
        "top5_order_match": [i for i, _ in ours_top] == [i for i, _ in ref_top],
        "top5_set_match": {i for i, _ in ours_top} == {i for i, _ in ref_top},
        "compared": len(diffs),
        "at_floor": at_floor,
        "max_abs_dev": max(diffs) if diffs else float("nan"),
        "mean_abs_dev": sum(diffs) / len(diffs) if diffs else float("nan"),
        "over_tolerance": len(over),
        "logit_span": span,
        "rel_max_abs_dev": (max(diffs) / span) if (diffs and span) else float("nan"),
        "worst_token": worst,
    }


def fetch_server_rows(url, ids, positions, timeout=120):
    """从 llama-server 取指定位置的完整词表 logprob 行。

    走 `/completion` 且 `n_predict=0`，llama.cpp 只对最后一个 prompt token 产出
    logits；要第 p 个位置的行，就把 `ids[:p+1]` 当 prompt 发一次。采样链配成
    `temperature=1, top_k=0, top_p=1, min_p=0`，即不做任何缩放与截断，于是返回的
    `logprob` 就是 `log_softmax(logits)`，两侧同减最大值后与 logits 差完全等价。

    Args:
        url: 例如 ``http://127.0.0.1:8971/completion``。
        ids: 与 Qllama 侧共用的 token id 序列。
        positions: 要取哪些位置的行（0 基，行内容是"看完 0..position 之后的预测"）。
        timeout: 单次 HTTP 超时秒数。

    Returns:
        与 ``load_llamacpp_rows`` 同形状的行列表，chunk 恒为 0。
    """
    import json
    import urllib.request

    rows = []
    n_vocab = None
    for position in positions:
        if position + 1 > len(ids):
            raise ValueError("位置 %d 超出 %d 个 id" % (position, len(ids)))
        body = json.dumps({
            "prompt": ids[: position + 1],
            "n_predict": 0,
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "typ_p": 1.0,
            "repeat_penalty": 1.0,
            "seed": 1,
            "n_probs": 1 << 20,
            "cache_prompt": False,
        }).encode("utf-8")
        request = urllib.request.Request(url, data=body,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        entries = payload["completion_probabilities"]
        if not entries:
            raise ValueError("位置 %d 没有返回 logits（n_predict=0 路径失效？）" % position)
        top = entries[-1]["top_logprobs"]
        if not top:
            raise ValueError("位置 %d 的 top_logprobs 为空" % position)
        if n_vocab is None:
            n_vocab = len(top)
        shifted = [float("-inf")] * n_vocab
        got = 0
        for item in top:
            token_id = item["id"]
            if token_id < n_vocab:
                shifted[token_id] = item["logprob"]
                got += 1
        if got != n_vocab:
            raise ValueError("位置 %d 只取到 %d/%d 个 logprob，无法做全词表比较"
                             % (position, got, n_vocab))
        # log_softmax 是 logits 减同一个常数，去掉常数项即可与 Qllama 侧对齐。
        peak = max(shifted)
        rows.append((0, position, ids, [v - peak for v in shifted]))
    return rows


def run_qllama(qllama_bin, model, ids, out_path):
    """调用 Qllama CLI 跑一遍前向，返回末位置 logits 行。

    Args:
        qllama_bin: ``qllama`` 可执行文件路径。
        model: GGUF 模型路径。
        ids: 与 llama.cpp 完全一致的 token id 序列。
        out_path: ``--logits`` 输出临时文件路径。

    Returns:
        ``logit`` 原始值的 float 列表。

    Raises:
        RuntimeError: CLI 退出码非 0（把 stderr 一并抛出，避免只看到缺文件）。
        ValueError: 输出行数与词表不符。
    """
    import subprocess  # 只在需要驱动 CLI 时才引入，单行比较模式不依赖它

    if os.path.exists(out_path):
        os.remove(out_path)
    command = [os.path.abspath(qllama_bin), "--model", os.path.abspath(model),
               "--tokens", ",".join(str(i) for i in ids),
               "--logits", os.path.abspath(out_path)]
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("qllama 退出码 %d\n%s\n%s"
                           % (proc.returncode, " ".join(command), proc.stderr))
    return load_qllama_row(out_path, None)


def sweep(rows, qllama_bin, model, tolerance, scratch, force_bos=None):
    """对每一行参考跑一次 Qllama 前向并比较。

    逐位置扫描的意义：RoPE 配对、位置偏移这类错误会随位置累积放大，只测末行容易
    漏掉"前半段对、后半段漂"的实现。

    Args:
        rows: 参考行列表，元素为 ``(chunk, position, ids, shifted_logits)``。
        qllama_bin: Qllama 可执行文件路径。
        model: GGUF 路径。
        tolerance: 逐值绝对偏差容差。
        scratch: 临时 logits 文件路径。
        force_bos: 非 None 时把前缀首元素替换成该 id（perplexity dump 的语义）；
            None 表示原样使用 ids（server 直接吃 token 数组，不做替换）。

    Returns:
        ``(每行结果列表, 汇总字典)``。
    """
    per_row = []
    for _chunk, position, chunk_ids, ref in rows:
        prefix = list(chunk_ids[: position + 1])
        if force_bos is not None:
            # llama.cpp 的 perplexity 把每个 chunk 的 tokens[0] 临时替换成 BOS
            # （perplexity.cpp:561-566），还原成 Qllama 输入时必须照做。
            prefix[0] = force_bos
        ours_raw = run_qllama(qllama_bin, model, prefix, scratch)
        max_ours = max(ours_raw)
        ours = [v - max_ours for v in ours_raw]
        result = compare(ours, ref, tolerance)
        result["position"] = position
        result["n_prefix"] = len(prefix)
        per_row.append(result)
        print("  pos %-4d argmax %-6s top-5 %-6s 最大偏差 %.3e (相对 %.3e) 平均 %.3e 超容差 %d"
              % (position,
                 "一致" if result["argmax_match"] else "不一致",
                 "一致" if result["top5_order_match"] else "不一致",
                 result["max_abs_dev"], result["rel_max_abs_dev"],
                 result["mean_abs_dev"], result["over_tolerance"]))
    summary = {
        "rows": len(per_row),
        "argmax_mismatch": sum(1 for r in per_row if not r["argmax_match"]),
        "top5_set_mismatch": sum(1 for r in per_row if not r["top5_set_match"]),
        "top5_order_mismatch": sum(1 for r in per_row if not r["top5_order_match"]),
        "over_tolerance": sum(r["over_tolerance"] for r in per_row),
        "max_abs_dev": max((r["max_abs_dev"] for r in per_row), default=float("nan")),
        "rel_max_abs_dev": max((r["rel_max_abs_dev"] for r in per_row),
                               default=float("nan")),
        "mean_abs_dev": (sum(r["mean_abs_dev"] for r in per_row) / len(per_row))
        if per_row else float("nan"),
    }
    return per_row, summary


def print_summary(summary, tolerance):
    """打印扫描汇总并返回是否通过。"""
    print("汇总: 行数 %d，argmax 不一致 %d，top-5 集合不一致 %d，排序不一致 %d"
          % (summary["rows"], summary["argmax_mismatch"], summary["top5_set_mismatch"],
             summary["top5_order_mismatch"]))
    print("      逐值最大偏差 %.3e（相对跨度 %.3e），平均 %.3e，超出容差 %.1e 的总数 %d"
          % (summary["max_abs_dev"], summary["rel_max_abs_dev"], summary["mean_abs_dev"],
             tolerance, summary["over_tolerance"]))
    ok = (summary["argmax_mismatch"] == 0 and summary["top5_set_mismatch"] == 0
          and summary["top5_order_mismatch"] == 0 and summary["over_tolerance"] == 0)
    print("结论: %s" % ("通过" if ok else "不通过"))
    return ok


def parse_positions(spec, n_ids):
    """解析位置表达式：``all`` 或 ``0,5,10-12``。"""
    if spec == "all":
        return list(range(n_ids))
    positions = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, high = part.split("-", 1)
            positions.extend(range(int(low), int(high) + 1))
        else:
            positions.append(int(part))
    return positions


def write_golden(path, rows, top_count, provenance):
    """把参考行导出为回归基准 TSV（只保留每行 top-K）。

    列：position、rank、token_id、shifted（``logit - max_logit``）。头部以 ``#``
    写清来源与命令，因为**这份数据只用于回归，不是 oracle**：模型权重一旦变化，
    必须重新生成，而重新生成的命令必须能从文件本身查到。

    Args:
        path: 输出路径。
        rows: 参考行列表。
        top_count: 每行保留的条数。
        provenance: ``(key, value)`` 列表，逐条写成 ``# key: value``。

    Returns:
        写出的数据行数。
    """
    written = 0
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("# llama.cpp 数值对照的回归基准（不是 oracle）。\n")
        for key, value in provenance:
            handle.write("# %s: %s\n" % (key, value))
        handle.write("# 列: position\trank\ttoken_id\tshifted_logprob(logit - max_logit)\n")
        handle.write("# 仅保留每行 top-%d；全词表逐值对照需要重新运行 llama.cpp，命令见上。\n"
                     % top_count)
        for _chunk, position, _ids, ref in rows:
            for rank, (token_id, value) in enumerate(top_k(ref, top_count)):
                handle.write("%d\t%d\t%d\t%.9g\n" % (position, rank, token_id, value))
                written += 1
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--llamacpp",
                        help="llama-perplexity --save-all-logits 输出的二进制文件")
    source.add_argument("--server",
                        help="llama-server 的 /completion 地址，如 http://127.0.0.1:8971/completion")
    parser.add_argument("--ids", help="配合 --server：与 Qllama 侧完全一致的 token id 序列")
    parser.add_argument("--positions", default="all",
                        help="配合 --server：要比较的位置，all 或 0,5,10-12")
    parser.add_argument("--qllama", help="单行模式：已生成的 Qllama --logits 文本文件")
    parser.add_argument("--qllama-bin", help="扫描模式：qllama 可执行文件")
    parser.add_argument("--model", help="扫描模式：GGUF 模型路径")
    parser.add_argument("--chunk", type=int, default=0, help="dump 模式取第几个 chunk")
    parser.add_argument("--row", default="all",
                        help="dump 模式取哪一行：下标、-1 末行、all 全部")
    parser.add_argument("--bos", type=int, default=1,
                        help="dump 模式下 llama.cpp 在 chunk 首位置强制写入的 BOS id")
    parser.add_argument("--tolerance", type=float, default=1e-2, help="逐值绝对偏差容差")
    parser.add_argument("--export-golden", help="把参考行导出为回归基准 TSV 后退出")
    parser.add_argument("--golden-top", type=int, default=64, help="导出时每行保留的条数")
    parser.add_argument("--provenance", action="append", nargs=2, default=[],
                        metavar=("KEY", "VALUE"), help="导出时写入头注释的来源信息，可重复")
    args = parser.parse_args(argv)

    if args.server:
        if not args.ids:
            print("--server 需要同时给出 --ids", file=sys.stderr)
            return 2
        ids = [int(x) for x in args.ids.split(",") if x.strip()]
        try:
            rows = fetch_server_rows(args.server, ids,
                                     parse_positions(args.positions, len(ids)))
        except (ValueError, KeyError) as exc:
            print("llama-server 返回无法解析: %r" % (exc,), file=sys.stderr)
            return 2
        force_bos = None
    else:
        try:
            rows = load_llamacpp_rows(args.llamacpp)
        except ValueError as exc:
            print("解析失败: %s" % exc, file=sys.stderr)
            return 2
        rows = [r for r in rows if r[0] == args.chunk]
        if not rows:
            print("文件里没有 chunk %d" % args.chunk, file=sys.stderr)
            return 2
        if rows[0][2][0] != args.bos:
            print("警告: chunk 首 token 是 %d 而不是 --bos %d，两侧输入的 id 可能对不上"
                  % (rows[0][2][0], args.bos), file=sys.stderr)
        force_bos = args.bos
        if args.row != "all":
            rows = [rows[int(args.row)]]

    if args.export_golden:
        count = write_golden(args.export_golden, rows, args.golden_top, args.provenance)
        print("已导出 %d 行参考数据到 %s（%d 个位置 × top-%d）"
              % (count, args.export_golden, len(rows), args.golden_top))
        return 0

    if args.qllama_bin:
        if not args.model:
            print("--qllama-bin 需要同时给出 --model", file=sys.stderr)
            return 2
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as handle:
            scratch = handle.name
        print("扫描 %d 个位置（每位置一次 Qllama 前向）" % len(rows))
        try:
            _per_row, summary = sweep(rows, args.qllama_bin, args.model, args.tolerance,
                                      scratch, force_bos)
        finally:
            # run_qllama 每轮都会先删除旧文件，临时文件可能已不存在。
            if os.path.exists(scratch):
                os.unlink(scratch)
        return 0 if print_summary(summary, args.tolerance) else 1

    if not args.qllama:
        print("需要 --qllama <文本文件> 或 --qllama-bin <可执行文件>", file=sys.stderr)
        return 2
    if len(rows) != 1:
        print("单行模式需要把位置限制到 1 个（--row N 或 --positions N）", file=sys.stderr)
        return 2
    _chunk, position, chunk_ids, ref = rows[0]
    ours_raw = load_qllama_row(args.qllama, len(ref))
    max_ours = max(ours_raw)
    ours = [v - max_ours for v in ours_raw]

    # 红线要求可复现：把两侧实际使用的 prompt 前缀完整打出来。
    prefix = chunk_ids[: position + 1]
    if force_bos is not None:
        prefix = [force_bos] + list(prefix[1:])
    result = compare(ours, ref, args.tolerance)
    print("位置 %d（前缀 %d 个 token）" % (position, len(prefix)))
    print("  ids: %s" % ",".join(str(t) for t in prefix))
    print("  Qllama argmax=%d, llama.cpp argmax=%d -> %s"
          % (result["argmax_ours"], result["argmax_ref"],
             "一致" if result["argmax_match"] else "不一致"))
    print("  top-5 Qllama:    %s"
          % " ".join("%d(%.6f)" % (i, v) for i, v in result["top5_ours"]))
    print("  top-5 llama.cpp: %s"
          % " ".join("%d(%.6f)" % (i, v) for i, v in result["top5_ref"]))
    print("  top-5 集合 %s, 排序 %s"
          % ("一致" if result["top5_set_match"] else "不一致",
             "一致" if result["top5_order_match"] else "不一致"))
    print("  逐值：参与比较 %d 个 token，不可判 %d 个，最大绝对偏差 %.3e（logit 跨度 %.3f，"
          "相对 %.3e），平均 %.3e，超出容差 %.1e 的个数 %d（最大偏差出现在 token %s）"
          % (result["compared"], result["at_floor"], result["max_abs_dev"],
             result["logit_span"], result["rel_max_abs_dev"], result["mean_abs_dev"],
             args.tolerance, result["over_tolerance"], result["worst_token"]))
    print("  Qllama 原始末行最大值 %.9g（仅作记录，跨实现不可直接比）" % max_ours)
    ok = (result["argmax_match"] and result["top5_set_match"]
          and result["top5_order_match"] and result["over_tolerance"] == 0)
    print("结论: %s" % ("通过" if ok else "不通过"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
