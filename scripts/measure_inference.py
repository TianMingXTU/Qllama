#!/usr/bin/env python3
"""运行 bench_inference 并采集 Windows 进程峰值工作集。

用法：python -I scripts/measure_inference.py --bench target/release/examples/bench_inference.exe
    --model MODEL.gguf --threads 8 --steps 32 [--hold] [--output perf.json]

--hold 让基准进程在输出 JSON 后阻塞，父进程此时用 GetProcessMemoryInfo 读取
PeakWorkingSetSize（操作系统高水位，非采样估计），随后放行并检查退出码。
"""

import argparse
import ctypes
import ctypes.wintypes
import json
import subprocess
import sys
from pathlib import Path

# wintypes 没有导出 SIZE_T；64 位平台上就是 c_size_t。
SIZE_T = ctypes.c_size_t


class ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.wintypes.DWORD),
        ("PageFaultCount", ctypes.wintypes.DWORD),
        ("PeakWorkingSetSize", SIZE_T),
        ("WorkingSetSize", SIZE_T),
        ("QuotaPeakPagedPoolUsage", SIZE_T),
        ("QuotaPagedPoolUsage", SIZE_T),
        ("QuotaPeakNonPagedPoolUsage", SIZE_T),
        ("QuotaNonPagedPoolUsage", SIZE_T),
        ("PagefileUsage", SIZE_T),
        ("PeakPagefileUsage", SIZE_T),
    ]


def peak_working_set(process):
    """读取子进程的峰值工作集字节；失败时返回 None 而不是猜一个值。"""
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(ProcessMemoryCounters)
    handle = ctypes.windll.kernel32.OpenProcess(
        0x0400 | 0x0010, False, process.pid)  # PROCESS_QUERY_INFORMATION | PROCESS_VM_READ
    if not handle:
        return None
    try:
        if not ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.PeakWorkingSetSize)
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hold", action="store_true")
    known, passthrough = parser.parse_known_args()
    command = [str(known.bench.resolve())] + passthrough
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE if known.hold else subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    peak = None
    if known.hold:
        report = json.loads(process.stdout.readline())
        peak = peak_working_set(process)
        if peak is None:
            print("could not read PeakWorkingSetSize; check process permissions",
                  file=sys.stderr)
        # Windows 管道的缓冲写入需要走底层句柄，TextIO 的 flush 会报 Invalid argument。
        process.stdin.buffer.write(b"\n")
        process.stdin.buffer.flush()
        process.stdin.close()
    stdout, stderr = process.communicate(timeout=1800)
    if process.returncode != 0:
        print(stderr or stdout, file=sys.stderr)
        raise SystemExit(process.returncode)
    if not known.hold:
        report = json.loads(stdout)
    report["peak_working_set_bytes"] = peak
    if peak is not None:
        report["peak_working_set_gib"] = round(peak / (1024 ** 3), 3)
    known.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "model", "threads", "load_seconds", "prefill_tokens_per_second",
        "decode_tokens_per_second", "peak_working_set_bytes")}, indent=2))


if __name__ == "__main__":
    main()
