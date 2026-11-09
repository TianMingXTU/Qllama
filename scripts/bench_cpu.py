#!/usr/bin/env python3
"""串行运行 CPU 基准，将原始 TSV、元数据和统计量原子写入 JSON。

python -I scripts/bench_cpu.py --model MODEL.gguf --output EXISTING_DIR/cpu.json
每个完整组合都会更新检查点；注意力算子使用合成激活和满容量合成历史。
"""

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time


WORKSPACE = Path(__file__).resolve().parents[1]
COLUMNS = (
    "metric", "threads", "ctx_size", "prompt_tokens", "generated_tokens",
    "decode_steps", "run", "warmup", "seconds", "units", "checksum",
)
INTEGER_COLUMNS = (
    "threads", "ctx_size", "prompt_tokens", "generated_tokens", "decode_steps",
    "run", "units",
)
KERNEL_UNITS = {
    "mapped_gemv": 1,
    "dense_gemv": 1,
    "mapped_gemm": 8,
    "dense_gemm": 8,
    "attention_cached": 1,
    "dequantize_rows": 32,
}
INTEGER_METADATA = (
    "model_max_context", "vocab_size", "hidden_size", "num_layers",
    "num_kv_heads", "head_dim", "matrix_rows", "matrix_columns",
    "attention_history_tokens", "attention_query_position",
)
SPREAD_THRESHOLD = 0.05
SAMPLE_INTERVAL = 0.05


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def positive_list(text):
    try:
        values = [positive_int(part.strip()) for part in text.split(",")]
    except (ValueError, argparse.ArgumentTypeError) as error:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from error
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("duplicate values are not allowed")
    return values


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    binary_name = "bench.exe" if os.name == "nt" else "bench"
    parser.add_argument("--bench", type=Path, default=WORKSPACE / "target" / "release" / binary_name)
    parser.add_argument("--threads", type=positive_list, default="1,2,4,8")
    parser.add_argument("--ctx-sizes", type=positive_list, default="128,2048")
    parser.add_argument("--steps", type=positive_int, default=32)
    parser.add_argument("--mode", choices=("inference", "kernels", "all"), default="all")
    parser.add_argument("--warmup", type=positive_int, default=3)
    parser.add_argument("--runs", type=positive_int, default=5)
    parser.add_argument("--full-generation", action="store_true",
                        help="also generate exactly 128 ids at the largest context, for each thread count")
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("--prompt")
    prompt.add_argument("--tokens", help="explicit comma-separated ids; bypass the tokenizer")
    args = parser.parse_args()
    if args.warmup < 3 or args.runs < 5:
        parser.error("--warmup must be at least 3 and --runs must be at least 5")
    args.model = args.model.resolve(strict=True)
    args.bench = args.bench.resolve(strict=True)
    args.output = args.output.resolve()
    if not args.model.is_file() or not args.bench.is_file():
        parser.error("--model and --bench must be files")
    if not args.output.parent.is_dir() or args.output.is_dir():
        parser.error("--output must name a file in an existing directory")
    if args.output in (args.model, args.bench):
        parser.error("--output must differ from the model and benchmark binary")
    return args


def file_identity(path):
    info = path.stat()
    return info.st_size, info.st_mtime_ns, info.st_dev, info.st_ino


def sha256_file(path):
    before = file_identity(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if file_identity(path) != before:
        raise RuntimeError(f"File changed while computing SHA256: {path}")
    return digest.hexdigest()


def command_output(command):
    result = subprocess.run(command, cwd=WORKSPACE, check=True, capture_output=True,
                            text=True, encoding="utf-8")
    return result.stdout.strip()


def atomic_checkpoint(output, report):
    """临时文件与目标同目录，替换时不会暴露半份 JSON。"""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         prefix=f".{output.name}.", suffix=".tmp",
                                         dir=output.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


class WindowsMemorySampler:
    def __init__(self, pid):
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.CloseHandle.restype = wintypes.BOOL
        psapi.GetProcessMemoryInfo.argtypes = (
            wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD,
        )
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        self.close_handle = kernel.CloseHandle
        self.get_memory = psapi.GetProcessMemoryInfo
        self.handle = kernel.OpenProcess(0x0400 | 0x0010, False, pid)
        self.peak = None
        self.samples = 0
        self.error = None
        if not self.handle:
            self.failed()

    def failed(self):
        self.error = str(ctypes.WinError(ctypes.get_last_error()))
        print(f"Working-set sampling unavailable: {self.error}", file=sys.stderr, flush=True)

    def sample(self):
        if not self.handle or self.error is not None:
            return
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not self.get_memory(self.handle, ctypes.byref(counters), counters.cb):
            self.failed()
            return
        self.samples += 1
        self.peak = max(self.peak or 0, int(counters.PeakWorkingSetSize))

    def close(self):
        if self.handle:
            if not self.close_handle(self.handle):
                raise ctypes.WinError(ctypes.get_last_error())
            self.handle = None

    def report(self):
        # 无退出前 hold 协议，末次采样之后的峰值可能遗漏，不能声称精确进程峰值。
        return {
            "method": "sampled_windows_peak_working_set",
            "sampled_peak_working_set_bytes": self.peak,
            "is_lower_bound": True,
            "sample_interval_seconds": SAMPLE_INTERVAL,
            "sample_count": self.samples,
            "error": self.error,
        }


def invoke(command):
    # 重定向到临时文件，采样期间不依赖管道容量，也不启动并行的基准进程。
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        started = time.perf_counter()
        process = subprocess.Popen(command, cwd=WORKSPACE, stdin=subprocess.DEVNULL,
                                   stdout=stdout_file, stderr=stderr_file)
        sampler = None
        try:
            if os.name == "nt":
                sampler = WindowsMemorySampler(process.pid)
            while True:
                if sampler is not None:
                    sampler.sample()
                try:
                    process.wait(timeout=SAMPLE_INTERVAL)
                    break
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            if process.poll() is None:
                process.kill()
            process.wait()
            raise
        finally:
            if sampler is not None:
                sampler.close()
        wall_seconds = time.perf_counter() - started
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read().decode("utf-8")
        stderr = stderr_file.read().decode("utf-8")
    if process.returncode != 0:
        raise RuntimeError(f"Benchmark exited with code {process.returncode}\n{stderr}\n{stdout}")
    memory = sampler.report() if sampler is not None else {
        "method": "unsupported_platform",
        "sampled_peak_working_set_bytes": None,
        "is_lower_bound": True,
        "sample_count": 0,
    }
    return stdout, stderr, wall_seconds, memory


def parse_ids(text, name):
    values = [int(part) for part in text.split(",")]
    if not values or any(value < 0 or value > 2**31 - 1 for value in values):
        raise ValueError(f"Invalid {name}: {text!r}")
    return values


def generated_checksum(ids):
    checksum = 0xCBF29CE484222325
    for token in ids:
        for byte in token.to_bytes(4, "little", signed=True):
            checksum = ((checksum ^ byte) * 0x100000001B3) & (2**64 - 1)
    return f"{checksum:016x}"


def parse_samples(text, case, warmup, runs):
    metadata = {}
    samples = []
    header_seen = False
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line:
            continue
        fields = line.split("\t")
        if fields[0] == "#":
            if len(fields) != 3 or not fields[1] or fields[1] in metadata:
                raise ValueError(f"Malformed or duplicate metadata at TSV line {line_number}")
            metadata[fields[1]] = fields[2]
            continue
        if not header_seen:
            if tuple(fields) != COLUMNS:
                raise ValueError(f"Unexpected TSV header at line {line_number}: {line!r}")
            header_seen = True
            continue
        if len(fields) != len(COLUMNS):
            raise ValueError(f"Wrong column count at TSV line {line_number}: {line!r}")
        sample = dict(zip(COLUMNS, fields))
        for field in INTEGER_COLUMNS:
            sample[field] = int(sample[field])
            if sample[field] < 0:
                raise ValueError(f"Negative {field} at TSV line {line_number}")
        if sample["warmup"] not in ("0", "1"):
            raise ValueError(f"Invalid warmup flag at TSV line {line_number}")
        sample["warmup"] = sample["warmup"] == "1"
        sample["seconds"] = float(sample["seconds"])
        if not math.isfinite(sample["seconds"]) or sample["seconds"] < 0:
            raise ValueError(f"Invalid duration at TSV line {line_number}")
        if re.fullmatch(r"[0-9a-f]{16}", sample["checksum"]) is None:
            raise ValueError(f"Invalid checksum at TSV line {line_number}")
        samples.append(sample)
    if not header_seen or metadata.get("schema_version") != "1":
        raise ValueError("Missing TSV header or unsupported schema_version")
    if metadata.get("checksum_algorithm") != "fnv1a64_little_endian":
        raise ValueError("Unsupported checksum algorithm")
    for key in INTEGER_METADATA:
        if key in metadata:
            metadata[key] = int(metadata[key])
    prompt = parse_ids(metadata["prompt_ids"], "prompt_ids")
    metadata["prompt_ids"] = prompt
    vocabulary = metadata["vocab_size"]
    if any(token >= vocabulary for token in prompt):
        raise ValueError("Prompt ids exceed the vocabulary")
    if len(prompt) + case["steps"] - 1 > case["ctx_size"]:
        raise ValueError("Reported prompt exceeds the requested context budget")
    if case["ctx_size"] > metadata["model_max_context"]:
        raise ValueError("Reported model maximum is below the requested context size")

    metrics = set()
    if case["mode"] in ("inference", "all"):
        metrics.update(("prefill", "decode"))
        ids = parse_ids(metadata["generated_ids"], "generated_ids")
        if len(ids) != case["steps"] or any(token >= vocabulary for token in ids):
            raise ValueError("Generated ids do not match the requested generation count/vocabulary")
        metadata["generated_ids"] = ids
        if metadata.get("inference_timing") != "forward_and_finite_greedy_argmax":
            raise ValueError("Unexpected inference timing scope")
    if case["mode"] in ("kernels", "all"):
        metrics.update(KERNEL_UNITS)
        expected_metadata = {
            "matrix_tensor": "blk.0.attn_q.weight",
            "kernel_activations": "synthetic_deterministic",
            "attention_history": "synthetic_rope_space_k_and_synthetic_v",
            "attention_history_tokens": case["ctx_size"] - 1,
            "attention_query_position": case["ctx_size"] - 1,
            "dense_scope": "selected_matrix_kernel_only",
            "dequantization_includes": "read_row_allocation",
        }
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise ValueError("Kernel metadata does not describe the required production/synthetic workload")
        if metadata.get("matrix_dtype") not in ("Q4_0", "Q8_0"):
            raise ValueError("Kernel matrix is not a supported compressed weight")
        rows = parse_ids(metadata["dequantization_rows"], "dequantization_rows")
        if len(rows) != 32 or len(set(rows)) != 32 or max(rows) >= metadata["matrix_rows"]:
            raise ValueError("Expected 32 distinct valid dequantization rows")
        metadata["dequantization_rows"] = rows

    seen = set()
    checksums = {}
    for sample in samples:
        metric, run = sample["metric"], sample["run"]
        key = metric, run
        if metric not in metrics or not 1 <= run <= warmup + runs or key in seen:
            raise ValueError(f"Unexpected or duplicate sample: {key}")
        seen.add(key)
        if sample["threads"] != case["threads"] or sample["ctx_size"] != case["ctx_size"]:
            raise ValueError(f"Sample belongs to a different thread/context combination: {key}")
        if sample["warmup"] != (run <= warmup):
            raise ValueError(f"Wrong warmup flag: {key}")
        if metric in ("prefill", "decode"):
            counts = len(prompt), case["steps"], case["steps"] - 1
            units = len(prompt) if metric == "prefill" else case["steps"] - 1
        else:
            counts, units = (0, 0, 0), KERNEL_UNITS[metric]
        actual_counts = tuple(sample[field] for field in (
            "prompt_tokens", "generated_tokens", "decode_steps"))
        if actual_counts != counts or sample["units"] != units:
            raise ValueError(f"Wrong work counts: {key}")
        if (units > 0 and sample["seconds"] <= 0) or (units == 0 and sample["seconds"] != 0):
            raise ValueError(f"Duration does not match measured work: {key}")
        if metric in checksums and sample["checksum"] != checksums[metric]:
            raise ValueError(f"Output changed across repeated {metric} runs")
        checksums[metric] = sample["checksum"]
    expected = {(metric, run) for metric in metrics for run in range(1, warmup + runs + 1)}
    if seen != expected:
        raise ValueError(f"Incomplete samples; missing {sorted(expected - seen)}")
    if "decode" in checksums and checksums["decode"] != generated_checksum(metadata["generated_ids"]):
        raise ValueError("Decode checksum does not match the emitted generated ids")
    return metadata, samples


def distribution(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def summarize(samples):
    summary = {}
    for metric in sorted({sample["metric"] for sample in samples}):
        formal = [sample for sample in samples if sample["metric"] == metric and not sample["warmup"]]
        seconds = distribution([sample["seconds"] for sample in formal])
        spread = ((seconds["max"] - seconds["min"]) / seconds["median"]
                  if seconds["median"] > 0 else None)
        result = {
            "formal_samples": len(formal),
            "warmup_samples": sum(sample["metric"] == metric and sample["warmup"] for sample in samples),
            "seconds": seconds,
            "spread": spread,
            "spread_over_5_percent": spread is not None and spread > SPREAD_THRESHOLD,
        }
        rate_name = "tokens_per_second" if metric in ("prefill", "decode") else "rows_per_second"
        result[rate_name] = (distribution([sample["units"] / sample["seconds"] for sample in formal])
                             if formal[0]["units"] else None)
        summary[metric] = result
    return summary


def print_summary(case, summary):
    label = f"{case['case']} threads={case['threads']} ctx={case['ctx_size']} steps={case['steps']}"
    for metric, result in summary.items():
        rate_name = "tokens_per_second" if metric in ("prefill", "decode") else "rows_per_second"
        rate = result[rate_name]
        throughput = f", median {rate['median']:.3f} {rate_name}" if rate is not None else ""
        spread = f"{result['spread']:.1%}" if result["spread"] is not None else "n/a"
        flag = " [spread > 5%]" if result["spread_over_5_percent"] else ""
        print(f"{label} {metric}: median {result['seconds']['median']:.6f}s"
              f"{throughput}, spread {spread}{flag}", flush=True)


def main():
    args = arguments()
    cases = [
        {"case": "capacity", "threads": threads, "ctx_size": context,
         "steps": args.steps, "mode": args.mode}
        for threads in args.threads for context in args.ctx_sizes
    ]
    if args.full_generation:
        cases.extend({"case": "full_generation", "threads": threads,
                      "ctx_size": max(args.ctx_sizes), "steps": 128, "mode": "inference"}
                     for threads in args.threads)
    model_identity = file_identity(args.model)
    model_sha = sha256_file(args.model)
    bench_sha = sha256_file(args.bench)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": {"path": str(args.model), "size_bytes": model_identity[0],
                  "sha256": model_sha, "sha256_verified_after_runs": False},
        "benchmark": {"path": str(args.bench), "sha256": bench_sha},
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": sys.version,
        "rustc": command_output(["rustc", "--version"]),
        "git_revision": command_output(["git", "rev-parse", "HEAD"]),
        "git_tracked_files_dirty": bool(command_output([
            "git", "status", "--porcelain", "--untracked-files=no"])),
        "warmup": args.warmup,
        "runs": args.runs,
        "spread_threshold": SPREAD_THRESHOLD,
        "prompt": args.prompt if args.prompt is not None else "Once upon a time",
        "explicit_tokens": args.tokens,
    }
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    report = {"schema_version": 1, "status": "running", "metadata": metadata,
              "planned_combinations": cases, "combinations": []}
    atomic_checkpoint(args.output, report)
    active_case = None
    try:
        for index, case in enumerate(cases, 1):
            active_case = case
            print(f"[{index}/{len(cases)}] {case}", file=sys.stderr, flush=True)
            if file_identity(args.model) != model_identity:
                raise RuntimeError("Model file changed between benchmark combinations")
            command = [str(args.bench), "--model", str(args.model),
                       "--threads", str(case["threads"]), "--ctx-size", str(case["ctx_size"]),
                       "--steps", str(case["steps"]), "--mode", case["mode"],
                       "--warmup", str(args.warmup), "--runs", str(args.runs)]
            if args.tokens is not None:
                command.extend(("--tokens", args.tokens))
            elif args.prompt is not None:
                command.extend(("--prompt", args.prompt))
            stdout, stderr, wall_seconds, memory = invoke(command)
            parsed_metadata, samples = parse_samples(stdout, case, args.warmup, args.runs)
            summary = summarize(samples)
            report["combinations"].append({
                **case, "command": command, "metadata": parsed_metadata,
                "raw_samples": samples, "summary": summary, "memory": memory,
                "process_wall_seconds": wall_seconds, "stderr": stderr,
            })
            atomic_checkpoint(args.output, report)
            print_summary(case, summary)
            print(f"[{index}/{len(cases)}] completed; checkpoint {args.output}",
                  file=sys.stderr, flush=True)
        active_case = None
        if file_identity(args.model) != model_identity or sha256_file(args.model) != model_sha:
            raise RuntimeError("Model SHA256 changed during the benchmark")
        if sha256_file(args.bench) != bench_sha:
            raise RuntimeError("Benchmark binary SHA256 changed during the benchmark")
        metadata["model"]["sha256_verified_after_runs"] = True
        report["status"] = "complete"
        atomic_checkpoint(args.output, report)
    except BaseException as error:
        report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        report["failed_combination"] = active_case
        report["error"] = f"{type(error).__name__}: {error}"
        atomic_checkpoint(args.output, report)
        print(f"{report['status']}: {report['error']}; completed combinations retained in {args.output}",
              file=sys.stderr, flush=True)
        raise
    print(f"Saved {len(report['combinations'])} complete combinations to {args.output}", flush=True)


if __name__ == "__main__":
    main()
