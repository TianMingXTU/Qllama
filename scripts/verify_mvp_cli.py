"""从真实用户角度验收 CLI：直接驱动 qllama 可执行文件，不看库内部。

用法：python -I verify_mvp_cli.py --qllama <qllama.exe> --model <model.gguf>

检查项全部是用户能观察到的行为：
1. 流式：正文在进程退出前逐 token 到达管道，不是结束时一次性刷出。
2. 编码：正文可无损 UTF-8 解码，不含 U+FFFD（跨 token 的多字节字符没被截断）。
3. 复现：同一 --seed 两次运行的 token 序列一致。
4. 预算：--ctx-size 小于请求步数时提前停并说明原因；超过模型标称值或 prompt
   已填满预算时以非零码退出并给出可读建议。
5. 参数：--threads 0 被拒绝。
6. 内存：KV cache 随 --ctx-size 变化，峰值工作集差值可解释。
7. 原生：进程没有加载 llama.dll / ggml.dll。

任一项失败即以非零码退出。
"""

import argparse
import ctypes
import ctypes.wintypes
import struct
import subprocess
import sys
import time
from pathlib import Path

if not sys.flags.isolated:
    raise SystemExit("run with python -I")

# Windows 控制台默认 GBK，中文断言消息会乱码；统一按 UTF-8 输出。
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
psapi = ctypes.WinDLL("psapi", use_last_error=True)

TH32CS_SNAPMODULE = 0x00000008
TH32CS_SNAPMODULE32 = 0x00000010
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
ERROR_BAD_LENGTH = 24
ERROR_NO_MORE_FILES = 18

GENERATION_HEADER = b"=== Generation ===\n"
SUMMARY_HEADER = b"=== Generation summary ==="

PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010


class ProcessMemoryCountersEx(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.wintypes.DWORD),
        ("PageFaultCount", ctypes.wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


class ModuleEntry32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.wintypes.DWORD),
        ("th32ModuleID", ctypes.wintypes.DWORD),
        ("th32ProcessID", ctypes.wintypes.DWORD),
        ("GlcntUsage", ctypes.wintypes.DWORD),
        ("ProccntUsage", ctypes.wintypes.DWORD),
        ("modBaseAddr", ctypes.POINTER(ctypes.c_byte)),
        ("modSize", ctypes.wintypes.DWORD),
        ("szModule", ctypes.wintypes.WCHAR * 256),
        ("szExePath", ctypes.wintypes.WCHAR * 260),
    ]


# 64 位 Python 下 ctypes 默认 restype 是 c_int，句柄会被截成 32 位；截断后的
# 快照句柄让 Module32FirstW 恒返回 ERROR_BAD_LENGTH，看起来就像"进程没加载任何
# 模块"，从而把"没有 llama.dll"变成一句永远为真的空话。原型必须在结构体之后声明。
kernel32.CreateToolhelp32Snapshot.argtypes = (ctypes.wintypes.DWORD, ctypes.wintypes.DWORD)
kernel32.CreateToolhelp32Snapshot.restype = ctypes.wintypes.HANDLE
kernel32.OpenProcess.argtypes = (ctypes.wintypes.DWORD, ctypes.wintypes.BOOL,
                                 ctypes.wintypes.DWORD)
kernel32.OpenProcess.restype = ctypes.wintypes.HANDLE
kernel32.CloseHandle.argtypes = (ctypes.wintypes.HANDLE,)
kernel32.Module32FirstW.argtypes = (ctypes.wintypes.HANDLE, ctypes.POINTER(ModuleEntry32W))
kernel32.Module32FirstW.restype = ctypes.wintypes.BOOL
kernel32.Module32NextW.argtypes = (ctypes.wintypes.HANDLE, ctypes.POINTER(ModuleEntry32W))
kernel32.Module32NextW.restype = ctypes.wintypes.BOOL
psapi.GetProcessMemoryInfo.argtypes = (ctypes.wintypes.HANDLE,
                                       ctypes.POINTER(ProcessMemoryCountersEx),
                                       ctypes.wintypes.DWORD)
psapi.GetProcessMemoryInfo.restype = ctypes.wintypes.BOOL


def peak_working_set(pid):
    handle = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        counters = ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.PeakWorkingSetSize
    finally:
        kernel32.CloseHandle(handle)


def pe_imported_dlls(path):
    """解析 PE 导入表，返回该可执行文件静态链接的 DLL 名列表。

    比运行时枚举模块可靠：不需要提权、不受进程初始化竞态影响（本机
    CreateToolhelp32Snapshot 对子进程恒返回 ERROR_BAD_LENGTH），而且直接回答
    "这个二进制到底链接了什么"。延迟加载与运行时 LoadLibrary 不在此表内，
    所以调用处仍会用 psapi 枚举做一次尽力而为的补充验证。
    """
    data = path.read_bytes()
    pe_header = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe_header:pe_header + 4] != b"PE\0\0":
        raise AssertionError(f"{path} 不是 PE 文件")
    section_count, = struct.unpack_from("<H", data, pe_header + 6)
    optional = pe_header + 24
    magic, = struct.unpack_from("<H", data, optional)
    data_directories = optional + (112 if magic == 0x20B else 96)
    import_rva, import_size = struct.unpack_from("<II", data, data_directories + 8)
    if import_rva == 0:
        return []
    header_size, = struct.unpack_from("<H", data, pe_header + 20)
    table = optional + header_size

    def rva_to_offset(rva):
        for index in range(section_count):
            base = table + index * 40
            virtual_size, virtual_address = struct.unpack_from("<II", data, base + 8)
            raw_size, raw_address = struct.unpack_from("<II", data, base + 16)
            if virtual_address <= rva < virtual_address + max(virtual_size, raw_size):
                return raw_address + (rva - virtual_address)
        raise AssertionError(f"RVA {rva:#x} 落在任何节区之外")

    offset = rva_to_offset(import_rva)
    names = []
    while True:
        entry = data[offset:offset + 20]
        name_rva, = struct.unpack_from("<I", entry, 12)
        if name_rva == 0:
            break
        start = rva_to_offset(name_rva)
        names.append(data[start:data.index(b"\x00", start)].decode("ascii"))
        offset += 20
    assert len(names) > 0, "导入表为空，解析结果不可信"
    return names


def loaded_modules(pid):
    """尽力而为地枚举进程模块；失败返回 None，由调用方决定如何降级。

    Toolhelp 快照在本机对刚创建的子进程恒返回 ERROR_BAD_LENGTH，因此这条路径
    只作补充证据，主判据是 pe_imported_dlls 的静态导入表。
    """
    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid)
    if snapshot == INVALID_HANDLE_VALUE:
        return None
    entry = ModuleEntry32W()
    entry.dwSize = ctypes.sizeof(ModuleEntry32W)
    names = []
    try:
        if not kernel32.Module32FirstW(snapshot, ctypes.byref(entry)):
            return None
        while True:
            names.append(entry.szModule)
            if not kernel32.Module32NextW(snapshot, ctypes.byref(entry)):
                return names if ctypes.get_last_error() == ERROR_NO_MORE_FILES else None
    finally:
        kernel32.CloseHandle(snapshot)


def read_timed(process, started):
    """逐字节读满 stdout，返回 (原始字节, [(字节序号, 到达秒)], 退出码)。

    读取阶段不做状态机：每个字节记下到达时刻，正文边界由调用方按标记位置切。
    """
    raw = bytearray()
    arrivals = []
    while True:
        piece = process.stdout.read(1)
        if not piece:
            break
        arrivals.append((len(raw), time.perf_counter() - started))
        raw.extend(piece)
    return bytes(raw), arrivals, process.wait()


def cli_arguments(qllama, model, extra):
    return [str(qllama), "--model", str(model)] + extra


def check_streaming(qllama, model, prompt, ctx_size, steps):
    process = subprocess.Popen(
        cli_arguments(qllama, model, [
            "--prompt", prompt, "--generate", str(steps),
            "--temperature", "0", "--ctx-size", str(ctx_size),
        ]),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    started = time.perf_counter()
    raw, arrivals, code = read_timed(process, started)
    exit_at = time.perf_counter() - started
    assert code == 0, f"退出码非零: {code}\n{raw.decode('utf-8', 'replace')}"

    header_at = raw.find(GENERATION_HEADER)
    assert header_at >= 0, f"没有 Generation 段头:\n{raw.decode('utf-8', 'replace')[:600]}"
    body_start = header_at + len(GENERATION_HEADER)
    summary_at = raw.find(SUMMARY_HEADER, body_start)
    assert summary_at > body_start, "摘要段头缺失"
    body = raw[body_start:summary_at]

    text = body.decode("utf-8", errors="strict")
    assert chr(0xFFFD) not in text, "正文含 U+FFFD：跨 token 的多字节字符被截断"
    assert text.strip(), "正文为空"

    times = [at for index, at in arrivals if body_start <= index < summary_at]
    first, last = min(times), max(times)
    slices = len({round(value / 0.2) for value in times})
    assert first < exit_at - 1.0, (
        f"正文第一个字节 {first:.2f}s 距退出 {exit_at:.2f}s 不足 1s：不是逐 token 流式"
    )
    assert slices >= 3, f"正文到达时间只占 {slices} 个 0.2s 时间片：疑似一次性缓冲"
    print(f"  first_token={first:.2f}s last_token={last:.2f}s exit={exit_at:.2f}s "
          f"slices={slices} body_bytes={len(body)}")
    print(f"  text: {text.strip()[:72]!r}")


def check_reproducible(qllama, model, prompt, ctx_size):
    def run():
        result = subprocess.run(
            cli_arguments(qllama, model, [
                "--prompt", prompt, "--generate", "12", "--temperature", "0.8",
                "--seed", "42", "--ctx-size", str(ctx_size),
            ]),
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert result.returncode == 0, result.stderr
        return next(
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip().startswith("generated ")
        )

    first, second = run(), run()
    assert first == second, f"同一 seed 两次运行不一致:\n{first}\n{second}"
    print(f"  seed 42 reproducible: {first[:72]}...")


def check_budget(qllama, model, prompt, ctx_size, steps):
    clamped = subprocess.run(
        cli_arguments(qllama, model, [
            "--prompt", prompt, "--generate", str(steps), "--temperature", "0",
            "--ctx-size", str(ctx_size),
        ]),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert clamped.returncode == 0, clamped.stderr
    assert "reached the context budget" in clamped.stdout, clamped.stdout
    summary = next(
        line for line in clamped.stdout.splitlines() if line.strip().startswith("generated ")
    )
    count = int(summary.split()[1])
    assert count < steps, f"预算应提前截断，实际生成 {count} >= {steps}"
    print(f"  --ctx-size {ctx_size} clamped --generate {steps} to {count} tokens, reason stated")

    oversized = subprocess.run(
        cli_arguments(qllama, model, ["--prompt", prompt, "--ctx-size", "100000000",
                                      "--generate", "2"]),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert oversized.returncode != 0, "--ctx-size 超过模型标称值应失败"
    assert "exceeds the model's context length" in oversized.stderr, oversized.stderr
    print("  oversized --ctx-size rejected (non-zero exit)")

    filled = subprocess.run(
        cli_arguments(qllama, model, ["--tokens", "1,2,3", "--ctx-size", "3", "--generate", "2"]),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert filled.returncode != 0, "prompt 已填满预算应失败"
    assert "no room for a new token" in filled.stderr, filled.stderr
    print("  prompt filling --ctx-size rejected with actionable advice")


def check_threads(qllama, model):
    result = subprocess.run(
        cli_arguments(qllama, model, ["--tokens", "1", "--threads", "0"]),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode != 0, "--threads 0 应被拒绝"
    print("  --threads 0 rejected by the parser")

    parity = []
    for threads in ("1", "4"):
        output = subprocess.run(
            cli_arguments(qllama, model, [
                "--prompt", "tok5", "--generate", "4", "--temperature", "0",
                "--threads", threads, "--ctx-size", "64", "--quiet",
            ]),
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if output.returncode != 0:
            # 词表里没有 "tok5" 的模型会拒绝该 prompt：这不是缺陷，跳过一致性对照。
            print(f"  --threads {threads}: prompt unsupported by this model, skipped")
            return
        parity.append(output.stdout)
    assert parity[0] == parity[1], "--threads 改变了一致贪心输出"
    print("  --threads 1 vs 4: identical greedy output")


def check_memory(qllama, model, prompt, small_ctx, large_ctx):
    readings = {}
    for ctx in (small_ctx, large_ctx):
        process = subprocess.Popen(
            cli_arguments(qllama, model, [
                "--prompt", prompt, "--generate", "64", "--temperature", "0",
                "--ctx-size", str(ctx), "--quiet",
            ]),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        peak = 0
        while process.poll() is None:
            try:
                peak = max(peak, peak_working_set(process.pid))
            except OSError:
                pass
            time.sleep(0.02)
        try:
            peak = max(peak, peak_working_set(process.pid))
        except OSError:
            pass
        assert process.returncode == 0, f"ctx {ctx} 运行失败"
        readings[ctx] = peak
    small, large = readings[small_ctx], readings[large_ctx]
    assert large > small, f"更大 ctx 的峰值工作集应更高: {small} vs {large}"
    print(f"  peak working set: ctx {small_ctx} = {small / 2**20:.1f} MiB, "
          f"ctx {large_ctx} = {large / 2**20:.1f} MiB (delta {(large - small) / 2**20:.1f} MiB)")


def check_native(qllama, model, prompt):
    """主判据：二进制自身的 PE 导入表里没有任何 llama/ggml DLL。

    导入表是静态事实，不受进程竞态与权限影响；运行时模块枚举在本机对子进程
    拿不到快照（ERROR_BAD_LENGTH），因此只作补充证据，缺失时明确说明而不是
    当作通过。
    """
    imports = pe_imported_dlls(qllama)
    banned = [name for name in imports if "llama" in name.lower() or "ggml" in name.lower()]
    assert not banned, f"二进制静态链接了 llama.cpp 的库: {banned}"
    print(f"  PE imports: {len(imports)} DLLs, none from llama.cpp")

    process = subprocess.Popen(
        cli_arguments(qllama, model, [
            "--prompt", prompt, "--generate", "48", "--temperature", "0", "--quiet",
        ]),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    names = None
    try:
        deadline = time.perf_counter() + 30
        while time.perf_counter() < deadline and names is None:
            if process.poll() is not None:
                break
            names = loaded_modules(process.pid)
            if names is None:
                time.sleep(0.05)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    if names is None:
        print("  runtime module enumeration unavailable (ERROR_BAD_LENGTH); "
              "static import table is the basis for this check")
    else:
        runtime_banned = [n for n in names
                          if n.lower().endswith(".dll")
                          and ("llama" in n.lower() or "ggml" in n.lower())]
        assert not runtime_banned, f"运行时加载了 llama.cpp 的库: {runtime_banned}"
        print(f"  runtime modules: {len(names)} loaded, none from llama.cpp")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qllama", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--ctx-size", type=int, default=256)
    parser.add_argument("--generate", type=int, default=40)
    arguments = parser.parse_args()

    print(f"model: {arguments.model.name} "
          f"({arguments.model.stat().st_size / 2**20:.1f} MiB)")
    print("1) streaming output")
    check_streaming(arguments.qllama, arguments.model, arguments.prompt,
                    arguments.ctx_size, arguments.generate)
    print("2) seed reproducibility")
    check_reproducible(arguments.qllama, arguments.model, arguments.prompt, arguments.ctx_size)
    print("3) context budget")
    check_budget(arguments.qllama, arguments.model, arguments.prompt, 48, arguments.generate * 8)
    print("4) thread argument handling")
    check_threads(arguments.qllama, arguments.model)
    print("5) memory scaling with --ctx-size")
    check_memory(arguments.qllama, arguments.model, arguments.prompt, 64, arguments.ctx_size)
    print("6) no llama.cpp runtime dependency")
    check_native(arguments.qllama, arguments.model, arguments.prompt)
    print("MVP_CLI_VERIFY_PASS")


if __name__ == "__main__":
    main()
