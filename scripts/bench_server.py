#!/usr/bin/env python3
"""Qllama RC 非流式 HTTP 推理的服务端并发基准采集。

只访问 --base-url（以及可选的第二基址），不跟随重定向、不使用代理。
输出仅包含聚合指标与结构信息，不写出提示词、生成文本或响应体；
逐请求记录只用于聚合，不写入结果文件。非流式请求没有真实首 token
时间，逐请求 TTFT 一律用总延迟代理并标记 ttft_is_proxy=true。

用法::

    python scripts/bench_server.py --base-url http://127.0.0.1:8080
"""

import argparse
import json
import math
import os
import platform
import re
import socket
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_CONCURRENCY = "1,8,32,64"
DEFAULT_OUTPUT = "target/reports/bench_server.json"
SEED = 20261114
STATEMENT = "尚不能宣称性能优于竞品"

METRIC_RSS = "qllama_rss_bytes"
METRIC_HIT_NAMES = ("qllama_kv_cache_hit_total", "qllama_kv_cache_hits_total")
METRIC_MISS_NAMES = ("qllama_kv_cache_miss_total", "qllama_kv_cache_misses_total")
METRIC_TTFT = "qllama_time_to_first_token_seconds"
TTFT_LABEL_FILTER = {"quantile": "0.95"}
TTFT_NOTE = (
    "非流式请求无法观测真实首 token 时间；逐请求 ttft_seconds 等于总延迟且 "
    "ttft_is_proxy=true。仅当 /metrics 提供 "
    'qllama_time_to_first_token_seconds{quantile="0.95"} 时记录服务端 p95，'
    "否则 p95_seconds 为 null，结论仍以代理值为限。"
)
PROMPT_NOTE = "由 --ctx 派生的中性重复词输入；未写入结果"

_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:\\.|[^"\\])*)"')
_METRIC_NAME_RE = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")


class Reply:
    """一次 HTTP 调用的最小结果；不携带可输出的响应文本。"""

    __slots__ = ("status", "raw", "data", "error")

    def __init__(self, status: int, raw: bytes, data: Any, error: Optional[str]) -> None:
        self.status = status
        self.raw = raw
        self.data = data
        self.error = error


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """拒绝重定向，保证所有请求留在配置的基址上。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D802
        return None


def _validate_base_url(value: str) -> urllib.parse.SplitResult:
    parts = urllib.parse.urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("base_url_invalid")
    try:
        _ = parts.port
    except ValueError as exc:
        raise ValueError("base_url_invalid") from exc
    if parts.query or parts.fragment:
        raise ValueError("base_url_invalid")
    return parts


def _safe_netloc(parts: urllib.parse.SplitResult) -> str:
    """去掉 userinfo，避免把 URL 内凭据写进结果。"""
    host = parts.hostname or ""
    if ":" in host and not host.startswith("["):
        host = "[" + host + "]"
    if parts.port is not None:
        host = "%s:%d" % (host, parts.port)
    return host


def _safe_base_url(parts: urllib.parse.SplitResult) -> str:
    return urllib.parse.urlunsplit(
        (parts.scheme, _safe_netloc(parts), parts.path.rstrip("/"), "", "")
    )


def _decode_json(raw: bytes) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8", "replace"))
    except (ValueError, TypeError):
        return None


class Client:
    """仅访问指定基址的 urllib 客户端。"""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str],
        tenant: Optional[str],
        timeout: float,
    ) -> None:
        self._parts = _validate_base_url(base_url)
        self._netloc = _safe_netloc(self._parts)
        self._prefix = self._parts.path.rstrip("/")
        self._api_key = api_key
        self._tenant = tenant
        self._timeout = timeout
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect()
        )

    @property
    def safe_base_url(self) -> str:
        return _safe_base_url(self._parts)

    def url(self, path: str) -> str:
        return urllib.parse.urlunsplit(
            (self._parts.scheme, self._netloc, self._prefix + path, "", "")
        )

    def get(self, path: str, accept: str) -> Reply:
        return self._request("GET", path, None, accept)

    def post_json(self, path: str, payload: Dict[str, Any]) -> Reply:
        return self._request("POST", path, payload, "application/json")

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]],
        accept: str,
    ) -> Reply:
        headers = {"Accept": accept}
        if self._api_key:
            headers["Authorization"] = "Bearer " + self._api_key
        if self._tenant:
            headers["X-Qllama-Tenant"] = self._tenant
        data = None
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(
            self.url(path), data=data, headers=headers, method=method
        )
        try:
            with self._opener.open(req, timeout=self._timeout) as resp:
                raw = resp.read()
                return Reply(int(resp.getcode() or 0), raw, _decode_json(raw), None)
        except urllib.error.HTTPError as exc:
            try:
                raw = exc.read()
            except Exception:
                raw = b""
            return Reply(int(exc.code), raw, _decode_json(raw), None)
        except (socket.timeout, TimeoutError):
            return Reply(0, b"", None, "timeout")
        except urllib.error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, (socket.timeout, TimeoutError)):
                return Reply(0, b"", None, "timeout")
            errno = getattr(reason, "errno", None)
            if isinstance(reason, ConnectionRefusedError) or errno in (61, 111, 10061):
                return Reply(0, b"", None, "connection_refused")
            return Reply(0, b"", None, "network_error")
        except OSError:
            return Reply(0, b"", None, "network_error")
        except Exception:
            return Reply(0, b"", None, "client_error")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _unescape_label(value: str) -> str:
    def replace(match: "re.Match[str]") -> str:
        token = match.group(1)
        if token == "n":
            return "\n"
        return token

    return re.sub(r"\\(.)", replace, value)


def _parse_prometheus(raw: bytes) -> Dict[str, List[Tuple[Dict[str, str], float]]]:
    """解析 Prometheus 文本；保留指标名、标签与数值，不保留 HELP 文本。"""
    samples: Dict[str, List[Tuple[Dict[str, str], float]]] = {}
    for line in raw.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        labels: Dict[str, str] = {}
        if "{" in line:
            name_part, remainder = line.split("{", 1)
            if "}" not in remainder:
                continue
            label_part, value_part = remainder.split("}", 1)
            name = name_part.strip()
            labels = {
                match.group(1): _unescape_label(match.group(2))
                for match in _LABEL_RE.finditer(label_part)
            }
        else:
            fields = line.split(None, 1)
            if len(fields) != 2:
                continue
            name, value_part = fields[0], fields[1]
        if not _METRIC_NAME_RE.match(name):
            continue
        tokens = value_part.strip().split()
        if not tokens:
            continue
        try:
            value = float(tokens[0])
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        samples.setdefault(name, []).append((labels, value))
    return samples


def _metric_series(
    samples: Dict[str, List[Tuple[Dict[str, str], float]]],
    name: str,
    label_filter: Optional[Dict[str, str]] = None,
) -> List[float]:
    values: List[float] = []
    for labels, value in samples.get(name, []):
        if label_filter and any(labels.get(key) != wanted for key, wanted in label_filter.items()):
            continue
        values.append(value)
    return values


def _sum_metric(
    samples: Dict[str, List[Tuple[Dict[str, str], float]]], name: str
) -> Optional[float]:
    values = _metric_series(samples, name)
    return float(sum(values)) if values else None


def _counter_delta(
    before: Dict[str, List[Tuple[Dict[str, str], float]]],
    after: Dict[str, List[Tuple[Dict[str, str], float]]],
    names: Sequence[str],
) -> Dict[str, Any]:
    """计算计数器增量；缺失或回退时不臆造数值。"""
    for name in names:
        before_value = _sum_metric(before, name)
        after_value = _sum_metric(after, name)
        if before_value is None or after_value is None:
            return {
                "metric": name,
                "before": before_value,
                "after": after_value,
                "delta": None,
                "note": "metric_missing",
            }
        delta = after_value - before_value
        if delta < 0:
            return {
                "metric": name,
                "before": before_value,
                "after": after_value,
                "delta": None,
                "note": "counter_reset_or_nonmonotonic",
            }
        return {
            "metric": name,
            "before": before_value,
            "after": after_value,
            "delta": delta,
            "note": "delta_of_counter_sum",
        }
    return {
        "metric": None,
        "before": None,
        "after": None,
        "delta": None,
        "note": "metric_missing",
    }


def _gauge_after(
    before: Dict[str, List[Tuple[Dict[str, str], float]]],
    after: Dict[str, List[Tuple[Dict[str, str], float]]],
    name: str,
) -> Dict[str, Any]:
    values = _metric_series(after, name)
    before_values = _metric_series(before, name)
    result: Dict[str, Any] = {
        "metric": name,
        "series_count": len(values),
        "before": None,
        "after": None,
        "delta": None,
        "note": "metric_missing",
    }
    if len(before_values) == 1:
        result["before"] = before_values[0]
    if len(values) == 1:
        result["after"] = values[0]
        result["note"] = "single_series"
        if result["before"] is not None:
            result["delta"] = values[0] - before_values[0]
    elif len(values) > 1:
        result["note"] = "multiple_series_no_aggregation"
    return result


def _ttft_metric(
    after: Dict[str, List[Tuple[Dict[str, str], float]]]
) -> Dict[str, Any]:
    values = _metric_series(after, METRIC_TTFT, TTFT_LABEL_FILTER)
    if len(values) == 1:
        p95 = values[0]
        source = "server_metric"
        note = "single_matching_series"
    elif len(values) == 0:
        p95 = None
        source = "proxy_only"
        note = "metric_missing"
    else:
        p95 = None
        source = "proxy_only"
        note = "multiple_matching_series_no_aggregation"
    return {
        "p95_seconds": p95,
        "source": source,
        "metric": METRIC_TTFT,
        "label_filter": dict(TTFT_LABEL_FILTER),
        "series_count": len(values),
        "note": note,
        "p95_is_proxy": p95 is None,
        "documentation": TTFT_NOTE,
    }


def _snapshot_metrics(client: Client) -> Tuple[Dict[str, List[Tuple[Dict[str, str], float]]], Optional[str]]:
    reply = client.get("/metrics", "text/plain")
    if reply.error:
        return {}, reply.error
    if reply.status != 200:
        return {}, "metrics_http_%d" % reply.status
    return _parse_prometheus(reply.raw), None


def _percentile(values: Sequence[float], p: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _latency_stats(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    values = [float(record["latency_seconds"]) for record in records]
    if not values:
        return {
            "count": 0,
            "mean_seconds": None,
            "p50_seconds": None,
            "p95_seconds": None,
            "p99_seconds": None,
            "variance_seconds2": None,
            "variance_kind": "sample",
        }
    variance = statistics.variance(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean_seconds": statistics.mean(values),
        "p50_seconds": _percentile(values, 0.50),
        "p95_seconds": _percentile(values, 0.95),
        "p99_seconds": _percentile(values, 0.99),
        "variance_seconds2": variance,
        "variance_kind": "sample",
    }


def _prompt_word_count(ctx: int) -> int:
    return max(1, ctx // 2)


def _completion_body(model: Optional[str], ctx: int, steps: int) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "prompt": " ".join(["context"] * _prompt_word_count(ctx)),
        "max_tokens": steps,
        "seed": SEED,
        "stream": False,
        "temperature": 0,
    }
    if model:
        body["model"] = model
    return body


def _run_request(client: Client, body: Dict[str, Any]) -> Dict[str, Any]:
    """逐请求采集：状态、延迟、TTFT 代理、生成 token 数、通用错误码。"""
    start = time.perf_counter()
    reply = client.post_json("/v1/completions", body)
    latency = time.perf_counter() - start
    completion_tokens = 0.0
    error = reply.error
    if reply.status == 200 and error is None:
        usage = reply.data.get("usage") if isinstance(reply.data, dict) else None
        if isinstance(usage, dict) and _is_number(usage.get("completion_tokens")):
            completion_tokens = float(usage["completion_tokens"])
        else:
            error = "missing_usage"
    elif reply.status == 200:
        error = error or "invalid_json"
    elif error is None:
        error = "http_status_%d" % reply.status
    return {
        "status": reply.status,
        "latency_seconds": latency,
        "ttft_seconds": latency,
        "ttft_is_proxy": True,
        "completion_tokens": completion_tokens,
        "error": error,
    }

def _batch(client: Client, body: Dict[str, Any], workers: int) -> List[Dict[str, Any]]:
    """用恰好 workers 个线程发出 workers 个并发请求。"""
    records: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_run_request, client, body) for _ in range(workers)
        ]
        for future in as_completed(futures):
            try:
                records.append(future.result())
            except Exception:
                records.append(
                    {
                        "status": 0,
                        "latency_seconds": 0.0,
                        "ttft_seconds": 0.0,
                        "ttft_is_proxy": True,
                        "completion_tokens": 0.0,
                        "error": "client_error",
                    }
                )
    return records


def _sample_summary(records: Sequence[Dict[str, Any]], wall_time: float) -> Dict[str, Any]:
    status_counts: Dict[str, int] = {}
    for record in records:
        key = str(record["status"])
        status_counts[key] = status_counts.get(key, 0) + 1
    tokens = sum(float(record["completion_tokens"]) for record in records)
    errors = sum(1 for record in records if record["error"] is not None)
    throughput = tokens / wall_time if wall_time > 0 else None
    return {
        "request_count": len(records),
        "wall_time_seconds": wall_time,
        "completion_tokens": tokens,
        "tok_s": throughput,
        "error_count": errors,
        "status_counts": status_counts,
    }


def _run_workload(client: Client, workers: int, args: argparse.Namespace) -> Dict[str, Any]:
    body = _completion_body(args.model, args.ctx, args.steps)
    for _ in range(args.warmup):
        _batch(client, body, workers)
    before, before_error = _snapshot_metrics(client)

    sample_summaries: List[Dict[str, Any]] = []
    all_records: List[Dict[str, Any]] = []
    for _ in range(args.samples):
        start = time.perf_counter()
        records = _batch(client, body, workers)
        wall_time = time.perf_counter() - start
        all_records.extend(records)
        sample_summaries.append(_sample_summary(records, wall_time))
    after, after_error = _snapshot_metrics(client)

    total_wall = sum(item["wall_time_seconds"] for item in sample_summaries)
    total_tokens = sum(float(item["completion_tokens"]) for item in sample_summaries)
    total_errors = sum(int(item["error_count"]) for item in sample_summaries)
    status_counts: Dict[str, int] = {}
    for item in sample_summaries:
        for key, count in item["status_counts"].items():
            status_counts[key] = status_counts.get(key, 0) + count
    latency = _latency_stats(all_records)
    ttft = _ttft_metric(after)

    return {
        "concurrency": workers,
        "warmup_request_count": workers * args.warmup,
        "sample_round_count": args.samples,
        "request_count": len(all_records),
        "samples": sample_summaries,
        "tok_s": total_tokens / total_wall if total_wall > 0 else None,
        "completion_tokens_total": total_tokens,
        "wall_time_seconds": total_wall,
        "error_count": total_errors,
        "error_rate": total_errors / len(all_records) if all_records else None,
        "status_counts": status_counts,
        "latency": latency,
        "variance": latency["variance_seconds2"],
        "variance_kind": latency["variance_kind"],
        "ttft_is_proxy": ttft["p95_is_proxy"],
        "ttft": ttft,
        "metrics": {
            "before_error": before_error,
            "after_error": after_error,
            "rss_bytes": _gauge_after(before, after, METRIC_RSS),
            "kv_cache_hit": _counter_delta(before, after, METRIC_HIT_NAMES),
            "kv_cache_miss": _counter_delta(before, after, METRIC_MISS_NAMES),
        },
        "request_records": all_records,
    }


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("需要正整数") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("需要正整数")
    return value


def _nonnegative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("需要非负整数") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("需要非负整数")
    return value


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("需要正数") from exc
    if value <= 0 or not math.isfinite(value):
        raise argparse.ArgumentTypeError("需要正数")
    return value


def _concurrency_list(text: str) -> List[int]:
    values: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part.isdigit():
            raise argparse.ArgumentTypeError("并发列表必须是逗号分隔的正整数")
        value = int(part)
        if value <= 0:
            raise argparse.ArgumentTypeError("并发列表必须是逗号分隔的正整数")
        values.append(value)
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("并发列表不得为空或重复")
    return values


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="采集 Qllama RC 服务端并发基准")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="主服务基址")
    parser.add_argument("--api-key", default=None, help="可选的 Bearer 凭据")
    parser.add_argument("--tenant", default=None, help="可选的租户请求头值")
    parser.add_argument("--model", default=None, help="可选的模型名")
    parser.add_argument(
        "--concurrency",
        default=DEFAULT_CONCURRENCY,
        type=_concurrency_list,
        help="逗号分隔的并发梯度，默认 1,8,32,64",
    )
    parser.add_argument("--ctx", type=_positive_int, default=128, help="上下文长度参数")
    parser.add_argument("--steps", type=_positive_int, default=16, help="每次生成上限")
    parser.add_argument("--warmup", type=_nonnegative_int, default=2, help="预热轮数")
    parser.add_argument("--samples", type=_positive_int, default=5, help="采样轮数")
    parser.add_argument("--timeout", type=_positive_float, default=120.0, help="单次请求超时秒数")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="JSON 摘要输出路径")
    parser.add_argument(
        "--baseline-no-cont-batching",
        default=None,
        help="可选的串行基线服务基址；省略时不测量基线",
    )
    return parser.parse_args(argv)


def _hardware() -> Dict[str, Any]:
    affinity: Optional[List[int]] = None
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": affinity,
        "cpu_affinity_available": affinity is not None,
        "note": "仅占位；未采集 CPU 型号、内存容量、模型与量化信息",
    }


def _method() -> Dict[str, Any]:
    return {
        "throughput": "sum(completion_tokens) / sum(sample_wall_time)",
        "wall_time": "每轮从提交 N 个请求到全部完成",
        "latency_percentile_method": "linear_interpolation",
        "variance_kind": "sample",
        "ttft": TTFT_NOTE,
        "metrics_counter_aggregation": "同名多序列求和；缺失、回退或多序列分位值不臆造",
        "request_records": "逐请求采集 status/latency/ttft_proxy/completion_tokens/error；不含响应文本",
        "baseline": "相同参数与并发梯度在第二个基址重跑；未配置则不测量",
        "comparison_scope": "仅比较两个已配置服务，不涉及 Ollama/llama.cpp 或其他竞品",
    }


def _speedup(
    primary: Sequence[Dict[str, Any]], baseline: Sequence[Dict[str, Any]]
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    comparisons: List[Dict[str, Any]] = []
    for primary_result, baseline_result in zip(primary, baseline):
        primary_rate = primary_result["tok_s"]
        baseline_rate = baseline_result["tok_s"]
        measured = (
            primary_rate is not None
            and baseline_rate is not None
            and primary_rate > 0
            and baseline_rate > 0
            and primary_result["error_rate"] < 1.0
            and baseline_result["error_rate"] < 1.0
        )
        value = primary_rate / baseline_rate if measured else None
        comparisons.append(
            {
                "concurrency": primary_result["concurrency"],
                "primary_tok_s": primary_rate,
                "baseline_tok_s": baseline_rate,
                "speedup": value,
                "both_measured": bool(measured),
            }
        )
    values = [item["speedup"] for item in comparisons if item["speedup"] is not None]
    if not values:
        return None, comparisons
    return (
        {
            "by_concurrency": comparisons,
            "mean": statistics.mean(values),
            "note": "仅两个已配置服务之间的吞吐比值，不构成竞品性能结论",
        },
        comparisons,
    )


def _emit(summary: Dict[str, Any], output_path: str) -> bool:
    try:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written = True
    except OSError:
        summary["output_error"] = "write_failed"
        written = False
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    summary: Dict[str, Any] = {
        "hardware": _hardware(),
        "params": {},
        "method": _method(),
        "results": [],
        "baseline": "not_measured",
        "comparison": STATEMENT,
        "speedup": None,
        "statement": STATEMENT,
        "errors": [],
    }
    try:
        primary_client = Client(args.base_url, args.api_key, args.tenant, args.timeout)
    except ValueError:
        summary["error"] = "base_url_invalid"
        written = _emit(summary, args.output)
        return 2 if written else 1
    baseline_client = None
    if args.baseline_no_cont_batching:
        try:
            baseline_client = Client(
                args.baseline_no_cont_batching, args.api_key, args.tenant, args.timeout
            )
        except ValueError:
            summary["error"] = "baseline_base_url_invalid"
            written = _emit(summary, args.output)
            return 2 if written else 1

    summary["params"] = {
        "base_url": primary_client.safe_base_url,
        "baseline_url": baseline_client.safe_base_url if baseline_client else None,
        "api_key_provided": bool(args.api_key),
        "tenant_provided": bool(args.tenant),
        "model_provided": bool(args.model),
        "concurrency_values": list(args.concurrency),
        "ctx": args.ctx,
        "steps": args.steps,
        "warmup_rounds": args.warmup,
        "sample_rounds": args.samples,
        "seed": SEED,
        "timeout_seconds": args.timeout,
        "endpoint": "/v1/completions",
        "stream": False,
        "temperature": 0,
        "prompt_word_count": _prompt_word_count(args.ctx),
        "prompt_note": PROMPT_NOTE,
        "output": str(args.output),
    }

    results: List[Dict[str, Any]] = []
    for workers in args.concurrency:
        results.append(_run_workload(primary_client, workers, args))
    summary["results"] = results

    baseline_results: List[Dict[str, Any]] = []
    if baseline_client:
        for workers in args.concurrency:
            baseline_results.append(_run_workload(baseline_client, workers, args))
        speedup, comparisons = _speedup(results, baseline_results)
        summary["speedup"] = speedup
        summary["baseline"] = {
            "base_url": baseline_client.safe_base_url,
            "status": "measured",
            "results": baseline_results,
        }
        summary["comparison"] = {
            "statement": STATEMENT,
            "speedup": speedup,
            "by_concurrency": comparisons,
            "note": "仅比较两个已配置服务，不涉及 Ollama/llama.cpp 或其他竞品",
        }

    error_names = set()
    for workload in results + baseline_results:
        if workload["error_count"] > 0:
            error_names.add("request_errors_present")
        for key in ("before_error", "after_error"):
            if workload["metrics"][key]:
                error_names.add("metrics_unavailable")
    summary["errors"] = sorted(error_names)
    summary["workload_status"] = "completed_with_errors" if "request_errors_present" in error_names else "completed"

    written = _emit(summary, args.output)
    if not written or "request_errors_present" in error_names:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
