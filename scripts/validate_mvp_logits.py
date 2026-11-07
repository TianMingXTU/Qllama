#!/usr/bin/env python3
"""用独立运行的 llama-server 对照 Qllama 的固定 token 末行 logits。

用法：python -I scripts/validate_mvp_logits.py --model MODEL.gguf
    --qllama target/release/qllama.exe --oracle-server PATH/llama-server.exe
    --ids-file ids.txt --positions all --ctx-size 128 --threads 1
    --tolerance 1e-4 --output evidence.json

位置从 0 开始；位置 p 的输入是 ids[:p + 1]。比较量为各自的 value - max，
因此完整 log_softmax 行与原始 logits 行可逐值对照，包括最小值。每个位置都须满足
最大绝对偏差 <= tolerance、argmax 一致且 top-5 顺序一致，量化模型也不例外。
原始向量按 token id 顺序写到 OUTPUT_STEM.vectors.jsonl，不导出 golden。
退出码：0 全部通过，1 数值或排序不一致，2 输入、进程或协议错误，130 用户中断。
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


STARTUP_TIMEOUT = 180
HTTP_TIMEOUT = 120
PROCESS_TIMEOUT = 600
STOP_TIMEOUT = 10

# 沿用 compare_logits.py 已验证的 /completion 配方，关闭采样变换与 prompt 缓存。
COMPLETION_PARAMS = {
    "n_predict": 0,
    "temperature": 1.0,
    "top_k": 0,
    "top_p": 1.0,
    "min_p": 0.0,
    "typ_p": 1.0,
    "repeat_penalty": 1.0,
    "seed": 1,
    "cache_prompt": False,
    "post_sampling_probs": False,
}


def positive_int(value):
    """解析正整数命令行参数。"""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def tolerance_value(value):
    """拒绝负数、NaN 和无穷容差，避免把失败误判为通过。"""
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="local GGUF model")
    parser.add_argument("--qllama", type=Path, required=True, help="Qllama CLI executable")
    parser.add_argument("--oracle-server", type=Path, required=True,
                        help="llama-server executable, not an existing server URL")
    parser.add_argument("--ids-file", type=Path, required=True,
                        help="UTF-8 file containing one nonnegative token ID per line")
    parser.add_argument("--positions", default="all",
                        help="all, or distinct zero-based positions such as 0,5,74")
    parser.add_argument("--ctx-size", type=positive_int, default=128,
                        help="maximum compared prefix length (default: 128); oracle -c >= 512")
    parser.add_argument("--threads", type=positive_int, default=1,
                        help="oracle CPU threads and Qllama RAYON_NUM_THREADS (default: 1)")
    parser.add_argument("--tolerance", type=tolerance_value, default=1e-4,
                        help="maximum absolute deviation after subtracting each peak (default: 1e-4)")
    parser.add_argument("--output", type=Path, required=True, help="JSON evidence path")
    args = parser.parse_args(argv)
    if not sys.flags.isolated:
        parser.error("run this script with python -I before reading models")
    return args


def read_ids(path):
    """严格读取一行一个 i32 token id，不插入或替换特殊 token。"""
    ids = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        text = line.strip()
        if not re.fullmatch(r"[0-9]+", text):
            raise ValueError(f"{path}:{line_number}: expected exactly one nonnegative ID")
        token_id = int(text)
        if token_id > 2**31 - 1:
            raise ValueError(f"{path}:{line_number}: ID exceeds the Qllama i32 range")
        ids.append(token_id)
    if not ids:
        raise ValueError("the IDs file is empty")
    return ids


def parse_positions(spec, n_ids):
    """解析 all 或逗号列表，拒绝重复、空字段和越界位置。"""
    if spec == "all":
        return list(range(n_ids))
    pieces = [piece.strip() for piece in spec.split(",")]
    if any(not re.fullmatch(r"[0-9]+", piece) for piece in pieces):
        raise ValueError("--positions must be all or a comma-separated list of positions")
    positions = [int(piece) for piece in pieces]
    if len(set(positions)) != len(positions):
        raise ValueError("--positions contains duplicates")
    if any(position >= n_ids for position in positions):
        raise ValueError(f"--positions is outside the IDs file (length {n_ids})")
    return positions


def sha256(path):
    """分块计算文件指纹，避免把模型整体读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_line(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()


def request_json(opener, url, body=None, timeout=HTTP_TIMEOUT):
    data = None if body is None else json.dumps(body, allow_nan=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with opener.open(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def wait_ready(child, opener, base_url):
    """在有界时间内等待自己的子进程就绪，并及时报告提前退出。"""
    deadline = time.monotonic() + STARTUP_TIMEOUT
    last_error = "no response"
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError(f"llama-server exited before readiness: {child.returncode}")
        try:
            payload = request_json(opener, base_url + "/health",
                                   timeout=min(1.0, max(0.01, deadline - time.monotonic())))
            if isinstance(payload, dict) and payload.get("status") == "ok":
                if child.poll() is not None:
                    raise RuntimeError(f"llama-server exited during readiness: {child.returncode}")
                return
            last_error = f"unexpected health response: {payload!r}"
        except (OSError, RuntimeError, ValueError) as exc:
            last_error = str(exc)
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    raise RuntimeError(f"llama-server readiness timed out after {STARTUP_TIMEOUT}s: {last_error}")


@contextmanager
def owned_server(command, env, log_path, provenance):
    """只终止本次 Popen 创建的进程；文件日志避免 stdout 管道塞满。"""
    with log_path.open("wb") as log:
        child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT)
        provenance["pid"] = child.pid
        try:
            yield child
        finally:
            try:
                if child.poll() is None:
                    child.terminate()
                try:
                    child.wait(timeout=STOP_TIMEOUT)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=STOP_TIMEOUT)
            finally:
                provenance["exit_code_after_cleanup"] = child.poll()
                log.flush()
                provenance["log"] = log_path.read_text(encoding="utf-8", errors="replace")
                # /props 给出可用 slot 上下文；日志还可记录底层分配的 512 最小窗口。
                match = re.search(r"llama_context:\s+n_ctx\s+=\s+(\d+)", provenance["log"])
                provenance["allocated_ctx_size_from_log"] = int(match[1]) if match else None


def positive_field(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    return value


def finite_value(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number, got {value!r}")
    return float(value)


def oracle_row(payload, n_vocab, prefix_length):
    """按独立 metadata 的词表大小验证完整 ID 覆盖，保留包括最小值的所有 logprob。"""
    if not isinstance(payload, dict):
        raise ValueError("/completion did not return an object")
    if payload.get("truncated") is not False:
        raise ValueError("/completion did not confirm an untruncated prompt")
    if payload.get("tokens_evaluated") != prefix_length:
        raise ValueError("/completion tokens_evaluated differs from the exact ID prefix length")
    entries = payload.get("completion_probabilities")
    if not isinstance(entries, list) or len(entries) != 1:
        raise ValueError("n_predict=0 must return exactly one completion_probabilities row")
    top = entries[0]["top_logprobs"]
    if not isinstance(top, list) or len(top) != n_vocab:
        count = len(top) if isinstance(top, list) else None
        raise ValueError(f"oracle returned {count}/{n_vocab} vocabulary entries")
    values = [None] * n_vocab
    for item in top:
        token_id = item["id"]
        if type(token_id) is not int or not 0 <= token_id < n_vocab:
            raise ValueError(f"oracle returned an out-of-range/noninteger token ID: {token_id!r}")
        if values[token_id] is not None:
            raise ValueError(f"oracle returned duplicate token ID {token_id}")
        values[token_id] = finite_value(item["logprob"], f"oracle token {token_id}")
    if any(value is None for value in values):
        raise ValueError("oracle response does not cover the entire vocabulary")
    return values


def run_qllama(command, env, output, n_vocab, evidence):
    """运行现有 CLI 并严格读取末行；不依赖尚未提供的线程或上下文参数。"""
    output.unlink(missing_ok=True)
    evidence["argv"] = command
    proc = subprocess.run(command, env=env, stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=PROCESS_TIMEOUT)
    evidence.update(return_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"Qllama exited with code {proc.returncode}; see row.qllama.stderr")
    values = []
    with output.open("r", encoding="utf-8") as handle:
        for token_id, line in enumerate(handle):
            values.append(finite_value(float(line), f"Qllama token {token_id}"))
    if len(values) != n_vocab:
        raise ValueError(f"Qllama returned {len(values)}/{n_vocab} logits")
    return values


def compare_rows(ours_raw, oracle_raw, tolerance):
    """对齐两侧常数偏移后比较全部值；归一化分母是 oracle 的整行跨度。"""
    if not ours_raw or len(ours_raw) != len(oracle_raw):
        raise ValueError("comparison requires two equally sized, nonempty vocabulary rows")
    ours_peak, oracle_peak = max(ours_raw), max(oracle_raw)
    ours = [finite_value(value - ours_peak, "shifted Qllama logit") for value in ours_raw]
    oracle = [finite_value(value - oracle_peak, "shifted oracle logprob") for value in oracle_raw]
    diffs = [abs(left - right) for left, right in zip(ours, oracle)]
    worst = max(range(len(diffs)), key=diffs.__getitem__)
    max_dev = diffs[worst]
    mean_dev = math.fsum(diffs) / len(diffs)
    span = -min(oracle)
    ours_top = sorted(range(len(ours)), key=lambda token_id: (-ours[token_id], token_id))[:5]
    oracle_top = sorted(range(len(oracle)), key=lambda token_id: (-oracle[token_id], token_id))[:5]
    argmax_match = ours_top[0] == oracle_top[0]
    top5_match = ours_top == oracle_top
    return {
        "passed": max_dev <= tolerance and argmax_match and top5_match,
        "compared": len(diffs),
        "excluded_values": 0,
        "max_abs_dev": max_dev,
        "mean_abs_dev": mean_dev,
        "over_tolerance": sum(diff > tolerance for diff in diffs),
        "worst_token_id": worst,
        "worst_shifted_ours": ours[worst],
        "worst_shifted_oracle": oracle[worst],
        "oracle_logit_span": span,
        # 常数 oracle 行没有可用的归一化尺度；用 null 表示未定义，禁止写 Infinity。
        "rel_max_abs_dev": max_dev / span if span else (0.0 if max_dev == 0 else None),
        "rel_mean_abs_dev": mean_dev / span if span else (0.0 if mean_dev == 0 else None),
        "argmax_ours": ours_top[0],
        "argmax_oracle": oracle_top[0],
        "argmax_match": argmax_match,
        "top5_ours_ids": ours_top,
        "top5_oracle_ids": oracle_top,
        "top5_order_match": top5_match,
        "minimum_ours_ids": [i for i, value in enumerate(ours) if value == min(ours)],
        "minimum_oracle_ids": [i for i, value in enumerate(oracle) if value == min(oracle)],
    }


def validate(args, report, vectors_path):
    """收集实时 oracle、逐位置 CLI 结果及可重放的完整来源信息。"""
    ids = read_ids(args.ids_file)
    positions = parse_positions(args.positions, len(ids))
    report.update(ids=ids, positions=positions, position_index_base=0)
    max_prefix = max(positions) + 1
    if max_prefix > args.ctx_size:
        raise ValueError(f"longest selected prefix ({max_prefix}) exceeds --ctx-size {args.ctx_size}")
    report["model"].update(sha256=sha256(args.model), size_bytes=args.model.stat().st_size)
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith("LLAMA_ARG_")}
    report["environment"] = {
        "removed_llama_arg_keys": sorted(key for key in os.environ
                                         if key.upper().startswith("LLAMA_ARG_")),
        "python": sys.version,
        "platform": platform.platform(),
        "isolated_mode": bool(sys.flags.isolated),
        "qllama_overrides": {"RAYON_NUM_THREADS": str(args.threads)},
    }
    ours_env = dict(env, RAYON_NUM_THREADS=str(args.threads))
    version_argv = [str(args.oracle_server), "--version"]
    version = subprocess.run(version_argv, env=env, stdin=subprocess.DEVNULL, capture_output=True,
                             text=True, encoding="utf-8", errors="replace", timeout=30)
    oracle = report["oracle"]
    oracle.update(version_argv=version_argv, version_return_code=version.returncode,
                  version_stdout=version.stdout, version_stderr=version.stderr,
                  version="\n".join(part.strip() for part in (version.stdout, version.stderr)
                                    if part.strip()))
    if version.returncode != 0 or not oracle["version"]:
        raise RuntimeError("could not obtain the actual llama-server version")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    # llama.cpp 可将小上下文向上取整到 512；分别记录请求值、实传值与实际 slot 值。
    oracle_ctx_argument = max(512, args.ctx_size)
    command = [str(args.oracle_server), "--model", str(args.model),
               "--host", "127.0.0.1", "--port", str(port),
               "--ctx-size", str(oracle_ctx_argument), "--parallel", "1",
               "--threads", str(args.threads), "--threads-batch", str(args.threads),
               "-ngl", "0", "-ctk", "f32", "-ctv", "f32", "--flash-attn", "off",
               "--no-repack", "--offline", "--no-context-shift"]
    oracle.update(argv=command, base_url=base_url, ctx_size_argument=oracle_ctx_argument,
                  startup_timeout_seconds=STARTUP_TIMEOUT, http_timeout_seconds=HTTP_TIMEOUT)
    # 本地对照不应经过用户的 HTTP 代理。
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with tempfile.TemporaryDirectory(prefix="qllama-logits-") as scratch:
        scratch = Path(scratch)
        with owned_server(command, env, scratch / "oracle.log", oracle) as child:
            wait_ready(child, opener, base_url)
            models = request_json(opener, base_url + "/v1/models")
            props = request_json(opener, base_url + "/props")
            model_entries = models["data"]
            if not isinstance(model_entries, list) or len(model_entries) != 1:
                raise ValueError("owned server must expose exactly one loaded model")
            meta = model_entries[0]["meta"]
            n_vocab = positive_field(meta["n_vocab"], "/v1/models meta.n_vocab")
            actual_ctx = positive_field(props["default_generation_settings"]["n_ctx"],
                                        "/props default_generation_settings.n_ctx")
            oracle.update(model_metadata=meta, build_info=props.get("build_info"),
                          model_path=props["model_path"], actual_slot_ctx_size=actual_ctx)
            if Path(props["model_path"]).resolve() != args.model:
                raise ValueError("owned server model_path differs from --model")
            if max_prefix + 1 >= actual_ctx:
                # server-context.cpp 在 process_token 里按 `prompt.n_tokens()+1 >= slot.n_ctx`
                # 置 truncated；提前失败而不是让 oracle_row 在扫描中途拒绝。
                raise ValueError(f"longest prefix ({max_prefix}) needs slot room >= {max_prefix + 2}; "
                                 f"actual oracle slot n_ctx is {actual_ctx}")
            if any(token_id >= n_vocab for token_id in ids):
                raise ValueError(f"IDs file contains token IDs outside vocabulary size {n_vocab}")
            report["n_vocab"] = n_vocab
            params = dict(COMPLETION_PARAMS, n_probs=max(1 << 20, n_vocab))
            oracle["completion_url"] = base_url + "/completion"
            oracle["completion_parameters"] = params
            with vectors_path.open("w", encoding="utf-8", newline="\n") as vectors:
                report["raw_vectors"]["created"] = True
                write_json_line(vectors, {"kind": "metadata", "schema_version": 1,
                                         "model_sha256": report["model"]["sha256"],
                                         "n_vocab": n_vocab, "ids": ids, "positions": positions,
                                         "ordering": "vector index is token ID; no values omitted"})
                for position in positions:
                    row = {"position": position, "prefix_length": position + 1, "status": "error"}
                    report["rows"].append(row)
                    if child.poll() is not None:
                        raise RuntimeError(f"llama-server exited during sweep: {child.returncode}")
                    request = dict(params, prompt=ids[:position + 1])
                    payload = request_json(opener, oracle["completion_url"], request)
                    reference = oracle_row(payload, n_vocab, position + 1)
                    row["oracle_tokens_evaluated"] = payload["tokens_evaluated"]
                    row["oracle_truncated"] = payload["truncated"]
                    write_json_line(vectors, {"position": position, "engine": "llama.cpp",
                                             "value_type": "raw_logprob", "request": request,
                                             "values": reference})
                    output = scratch / "qllama-logits.txt"
                    ours_command = [str(args.qllama), "--model", str(args.model), "--tokens",
                                    ",".join(map(str, ids[:position + 1])),
                                    "--logits", str(output), "--quiet"]
                    row["qllama"] = {}
                    ours = run_qllama(ours_command, ours_env, output, n_vocab, row["qllama"])
                    write_json_line(vectors, {"position": position, "engine": "qllama",
                                             "value_type": "raw_logit", "values": ours})
                    row.update(compare_rows(ours, reference, args.tolerance))
                    row["status"] = "passed" if row["passed"] else "failed"
                    print(f"pos {position}: {row['status']} max={row['max_abs_dev']:.6e} "
                          f"mean={row['mean_abs_dev']:.6e} argmax={row['argmax_match']} "
                          f"top5={row['top5_order_match']}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    for name in ("model", "qllama", "oracle_server", "ids_file", "output"):
        setattr(args, name, getattr(args, name).resolve())
    vectors_path = args.output.with_name(args.output.stem + ".vectors.jsonl")
    protected = {args.model, args.qllama, args.oracle_server, args.ids_file, Path(__file__).resolve()}
    golden = Path(__file__).resolve().parents[1] / "tests" / "golden"
    for output in (args.output, vectors_path):
        if output in protected or output == golden or golden in output.parents:
            print(f"refusing to overwrite inputs or write evidence into tests/golden: {output}",
                  file=sys.stderr)
            return 2
    report = {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "error",
        "passed": False,
        "exit_code": 2,
        "model": {"path": str(args.model)},
        "ids_file": str(args.ids_file),
        "positions_spec": args.positions,
        "ctx_size_requested": args.ctx_size,
        "threads": args.threads,
        "comparison": {
            "values": "each side subtracts its own maximum; every vocabulary value is compared",
            "tolerance": args.tolerance,
            "acceptance": "max_abs_dev <= tolerance AND argmax_match AND top5_order_match",
            "relative_deviation_denominator": "oracle max(logprob) - min(logprob)",
            "zero_span_relative_deviation": "0 for equal rows; null otherwise",
            "top5_tie_break": "ascending token ID",
            "quantized_policy": "same acceptance criteria; no automatic tolerance relaxation",
        },
        "oracle": {"executable": str(args.oracle_server)},
        "qllama": {"executable": str(args.qllama), "timeout_seconds": PROCESS_TIMEOUT,
                   "ctx_size_note": "current CLI has no --ctx-size; requested size bounds selected prefixes"},
        "raw_vectors": {"path": str(vectors_path), "format": "JSONL", "created": False},
        "rows": [],
    }
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for path in (args.model, args.qllama, args.oracle_server, args.ids_file):
            if not path.is_file():
                raise ValueError(f"input is not a file: {path}")
        validate(args, report, vectors_path)
        passed = all(row["passed"] for row in report["rows"])
        report.update(status="passed" if passed else "failed", passed=passed,
                      exit_code=0 if passed else 1)
    except KeyboardInterrupt:
        report.update(error="interrupted", exit_code=130)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    completed = [row for row in report["rows"] if "passed" in row]
    report["summary"] = {
        "requested_positions": len(report.get("positions", [])),
        "completed_positions": len(completed),
        "compared_values": sum(row["compared"] for row in completed),
        "max_abs_dev": max((row["max_abs_dev"] for row in completed), default=None),
        "mean_abs_dev": (math.fsum(row["mean_abs_dev"] for row in completed) / len(completed)
                         if completed else None),
        "over_tolerance": sum(row["over_tolerance"] for row in completed),
        "argmax_mismatches": sum(not row["argmax_match"] for row in completed),
        "top5_order_mismatches": sum(not row["top5_order_match"] for row in completed),
    }
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    try:
        if report["raw_vectors"]["created"]:
            report["raw_vectors"]["sha256"] = sha256(vectors_path)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
                               + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"could not save JSON evidence: {exc}", file=sys.stderr)
        return 2
    print(f"{report['status'].upper()}: {args.output}")
    if "error" in report:
        print(report["error"], file=sys.stderr)
    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
