#!/usr/bin/env python3
"""量化偏差归因：同一 vectors JSONL 证据里比较四个组合。

用法：python -I scripts/attribute_quant_deviation.py <f32.vectors.jsonl> <q8.vectors.jsonl> <q4.vectors.jsonl>

对每个冻结位置计算 shifted（减行最大值）行：
- our-F32 vs oracle-F32（协议基线，应 <= 1e-4）
- our-Q vs oracle-Q（同一量化文件，两侧唯一差异是存储/计算方式）
- oracle-Q vs oracle-F32（oracle 自身引入的量化影响，含激活 q8_0 化）
- our-Q vs our-F32（我们权重-only 反量化的自身代价）

若 our-Q vs oracle-Q 的偏差量级与 oracle-Q vs oracle-F32 相当，而 our-Q vs our-F32
明显更小，则可归因为 oracle 在量化 GEMV 中把 f32 激活一并量化到 q8_0
（ggml_vec_dot_q4_0_q8_0 / q8_0_q8_0），不是我们的反量化错位。
"""

import json
import math
import sys
from pathlib import Path


def rows(path):
    """position -> engine -> raw vector。"""
    table = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("kind") == "metadata":
                continue
            table.setdefault(record["position"], {})[record["engine"]] = record["values"]
    return table


def shift(values):
    peak = max(values)
    return [value - peak for value in values]


def stats(left, right):
    diffs = [abs(a - b) for a, b in zip(left, right)]
    return max(diffs), math.fsum(diffs) / len(diffs)


def top_ids(values):
    return sorted(range(len(values)), key=lambda i: (-values[i], i))[:5]


def main():
    if not sys.flags.isolated:
        raise SystemExit("run with python -I")
    paths = [Path(argument) for argument in sys.argv[1:]]
    if len(paths) != 3:
        raise SystemExit(__doc__)
    f32, q8, q4 = (rows(path) for path in paths)
    positions = sorted(f32)
    print(f"{'pos':>4} {'ours-Q vs oracle-Q':>22} {'oracle-Q vs oracle-F32':>26} {'ours-Q vs ours-F32':>20} {'F32 baseline':>14} top5")
    for name, table in (("Q8_0", q8), ("Q4_0", q4)):
        print(f"--- {name} ---")
        for position in positions:
            our_f32 = shift(f32[position]["qllama"])
            oracle_f32 = shift(f32[position]["llama.cpp"])
            our_q = shift(table[position]["qllama"])
            oracle_q = shift(table[position]["llama.cpp"])
            same = stats(our_q, oracle_q)
            oracle_side = stats(oracle_q, oracle_f32)
            our_side = stats(our_q, our_f32)
            baseline = stats(our_f32, oracle_f32)
            order = "OK" if top_ids(our_q) == top_ids(oracle_q) else "DIFF"
            argmax = "OK" if top_ids(our_q)[0] == top_ids(oracle_q)[0] else "DIFF"
            print(f"{position:>4} {same[0]:>12.3e}/{same[1]:.2e} {oracle_side[0]:>16.3e}/{oracle_side[1]:.2e} "
                  f"{our_side[0]:>14.3e}/{our_side[1]:.2e} {baseline[0]:>10.2e} argmax={argmax} top5={order}")


if __name__ == "__main__":
    main()
