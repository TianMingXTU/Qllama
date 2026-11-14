#!/usr/bin/env python3
"""Qllama RC 的 OpenAI 兼容接口与 Ollama 原生接口一致性校验。

脚本只访问 --base-url 指定的服务，不跟随重定向、不使用代理；所有检查结果
只记录状态码、结构是否合规与固定描述，不写出提示词或生成内容。
任何检查失败都返回非零退出码；--strict 与默认模式采用同一退出策略。

用法::

    python scripts/conformance_openai.py --base-url http://127.0.0.1:8080
"""

import argparse
import json
import math
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_OUTPUT = "target/reports/conformance.json"
SEED = 20261114
MAX_TOKENS = 8

# 内部固定输入。禁止把这些字符串写入 detail、输出文件、日志或异常信息。
PROMPT_SHORT = "hello"
PROMPT_JSON = "Return one JSON object with a boolean field named ok."
PROMPT_TOOLS = "Use the available function if it is needed."
PROMPT_TEMPLATE = "hello"

USAGE_KEYS = (
    "prompt_eval_count",
    "eval_count",
    "total_duration",
    "load_duration",
    "prompt_eval_duration",
    "eval_duration",
)
REQUIRED_USAGE_KEYS = USAGE_KEYS


class Reply:
    """一次 HTTP 调用的结果，只保留校验所需的最小信息。"""

    __slots__ = ("status", "headers", "raw", "data", "transport_error")

    def __init__(
        self,
        status: int,
        headers: Dict[str, str],
        raw: bytes,
        data: Any,
        transport_error: Optional[str],
    ) -> None:
        self.status = status
        self.headers = headers
        self.raw = raw
        self.data = data
        self.transport_error = transport_error

    @property
    def content_type(self) -> str:
        return self.headers.get("content-type", "")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """拒绝重定向，避免请求被引导到 --base-url 之外的地址。"""

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
    """去掉 userinfo，只保留主机与端口，避免把凭据写进结果。"""
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


def _decode_json(raw: bytes, headers: Dict[str, str]) -> Any:
    """解析 JSON 响应；SSE 与非 JSON 内容返回 None。"""
    if not raw or "event-stream" in headers.get("content-type", "").lower():
        return None
    try:
        return json.loads(raw.decode("utf-8", "replace"))
    except (ValueError, TypeError):
        return None


class Client:
    """仅访问指定 base-url 的 urllib 客户端。"""

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

    def request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        accept: str = "application/json",
        auth: bool = True,
    ) -> Reply:
        headers = {"Accept": accept}
        if auth and self._api_key:
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
                status = int(resp.getcode() or 0)
                resp_headers = {
                    k.lower(): v for k, v in resp.headers.items()
                }
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            resp_headers = {
                k.lower(): v for k, v in (exc.headers or {}).items()
            }
            try:
                raw = exc.read()
            except Exception:
                raw = b""
        except (socket.timeout, TimeoutError):
            return Reply(0, {}, b"", None, "timeout")
        except urllib.error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, (socket.timeout, TimeoutError)):
                return Reply(0, {}, b"", None, "timeout")
            errno = getattr(reason, "errno", None)
            if isinstance(reason, ConnectionRefusedError) or errno in (61, 111, 10061):
                return Reply(0, {}, b"", None, "connection_refused")
            return Reply(0, {}, b"", None, "network_error")
        except OSError:
            return Reply(0, {}, b"", None, "network_error")
        except Exception:
            return Reply(0, {}, b"", None, "client_error")
        return Reply(status, resp_headers, raw, _decode_json(raw, resp_headers), None)


def _as_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def _error_shape(reply: Reply) -> Dict[str, Any]:
    """解析 OpenAI 错误形状；message 仅用于内部标记匹配，不写出。"""
    data = reply.data if isinstance(reply.data, dict) else {}
    err = data.get("error")
    if isinstance(err, dict):
        return {
            "has_error": True,
            "code": _as_text(err.get("code")),
            "type": _as_text(err.get("type")),
            "message": _as_text(err.get("message")),
        }
    if isinstance(err, str):
        return {"has_error": True, "code": None, "type": None, "message": err}
    if any(key in data for key in ("code", "type", "message")):
        return {
            "has_error": True,
            "code": _as_text(data.get("code")),
            "type": _as_text(data.get("type")),
            "message": _as_text(data.get("message")),
        }
    return {"has_error": False, "code": None, "type": None, "message": None}


def _flatten(value: Any, depth: int = 0) -> str:
    """把响应结构展开成内部检索串；结果不输出。"""
    if depth > 6:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            parts.append(_flatten(key, depth + 1))
            parts.append(_flatten(item, depth + 1))
        return " ".join(parts)
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten(item, depth + 1) for item in value)
    return ""


def _unsupported_info(reply: Reply) -> Dict[str, Any]:
    """定位 qllama.unsupported 声明，并收集内部标记检索文本。"""
    shape = _error_shape(reply)
    data = reply.data if isinstance(reply.data, dict) else {}
    payloads: List[Any] = []
    locations: List[str] = []
    if "qllama.unsupported" in data:
        payloads.append(data["qllama.unsupported"])
        locations.append("top_level")
    err = data.get("error")
    if isinstance(err, dict) and "qllama.unsupported" in err:
        payloads.append(err["qllama.unsupported"])
        locations.append("error")
    code_text = " ".join(x for x in (shape["code"], shape["type"]) if x)
    code_signal = "qllama.unsupported" in code_text.lower() or bool(payloads)
    parts = [_flatten(payload).lower() for payload in payloads]
    if code_signal:
        parts.append((shape["message"] or "").lower())
        parts.append(code_text.lower())
    return {
        "has_error": shape["has_error"],
        "code_signal": code_signal,
        "locations": locations,
        "text": " ".join(parts),
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numeric_usage(usage: Any) -> bool:
    if not isinstance(usage, dict):
        return False
    return all(_is_number(usage.get(key)) for key in ("prompt_tokens", "completion_tokens", "total_tokens"))


def _choice_text(reply: Reply, chat: bool) -> Optional[str]:
    data = reply.data
    if not isinstance(data, dict):
        return None
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    if chat:
        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            return content if isinstance(content, str) else None
        delta = first.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            return content if isinstance(content, str) else None
        return None
    text = first.get("text")
    return text if isinstance(text, str) else None


def _completion_ok(reply: Reply, chat: bool) -> bool:
    data = reply.data
    if not isinstance(data, dict) or not _numeric_usage(data.get("usage")):
        return False
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    first = choices[0]
    if not isinstance(first, dict):
        return False
    if chat:
        message = first.get("message")
        return isinstance(message, dict) and isinstance(message.get("content"), str)
    return isinstance(first.get("text"), str)


def _json_choice_ok(reply: Reply, chat: bool, required_key: Optional[str]) -> bool:
    if reply.status != 200:
        return False
    text = _choice_text(reply, chat)
    if not isinstance(text, str):
        return False
    try:
        parsed = json.loads(text.strip())
    except (ValueError, TypeError):
        return False
    if required_key is None:
        return True
    return isinstance(parsed, dict) and required_key in parsed


def _tool_calls_valid(reply: Reply) -> bool:
    data = reply.data
    if not isinstance(data, dict):
        return False
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    first = choices[0]
    if not isinstance(first, dict):
        return False
    message = first.get("message")
    if not isinstance(message, dict):
        message = first.get("delta")
    if not isinstance(message, dict):
        return False
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return False
    for call in calls:
        if not isinstance(call, dict) or call.get("type") != "function":
            return False
        if not isinstance(call.get("id"), str) or not call["id"]:
            return False
        function = call.get("function")
        if not isinstance(function, dict):
            return False
        if not isinstance(function.get("name"), str) or not function["name"]:
            return False
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            return False
        try:
            parsed = json.loads(arguments)
        except (ValueError, TypeError):
            return False
        if not isinstance(parsed, dict):
            return False
    return True


def _apply_template_ok(reply: Reply) -> bool:
    data = reply.data
    return (
        isinstance(data, dict)
        and isinstance(data.get("prompt"), str)
        and _is_number(data.get("token_count"))
    )


def _parse_sse(raw: bytes) -> Dict[str, Any]:
    """解析 SSE，只保留结构化事件与终止标记，不累积生成文本。"""
    text = raw.decode("utf-8", "replace")
    events: List[Dict[str, Any]] = []
    data_lines: List[str] = []
    event_name = ""
    done = False

    def flush() -> None:
        nonlocal data_lines, event_name, done
        if not data_lines:
            event_name = ""
            return
        payload = "\n".join(data_lines).strip()
        data_lines = []
        name = event_name.strip().lower()
        event_name = ""
        if name == "keepalive" or payload.startswith(":") or payload == "keepalive":
            return
        if payload == "[DONE]":
            done = True
            return
        try:
            obj = json.loads(payload)
        except (ValueError, TypeError):
            return
        if isinstance(obj, dict):
            events.append(obj)

    for line in text.splitlines():
        if line == "":
            flush()
            if done:
                break
        elif line.startswith(":"):
            continue
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip(" "))
        elif line.startswith("event:"):
            event_name = line[6:].strip()
        else:
            continue
    flush()
    return {"events": events, "done": done}


def _sse_summary(reply: Reply) -> Dict[str, Any]:
    parsed = _parse_sse(reply.raw)
    events = parsed["events"]
    usage = None
    usage_index = -1
    finish_present = False
    error_chunk = False
    for index, event in enumerate(events):
        if isinstance(event.get("error"), (dict, str)):
            error_chunk = True
        candidate = event.get("usage")
        if isinstance(candidate, dict):
            usage = candidate
            usage_index = index
        choices = event.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if isinstance(choice, dict) and choice.get("finish_reason"):
                    finish_present = True
    return {
        "content_type_ok": reply.content_type.lower().startswith("text/event-stream"),
        "done": parsed["done"],
        "chunk_count": len(events),
        "finish_reason_present": finish_present,
        "usage_present": usage is not None,
        "usage_valid": _numeric_usage(usage),
        "usage_in_final_chunk": usage_index == len(events) - 1 and usage_index >= 0,
        "error_chunk": error_chunk,
    }


def _parse_ollama_usage(raw: bytes, content_type: str) -> Dict[str, Any]:
    """解析 Ollama JSON 或 NDJSON 响应的末帧 usage，不保留生成文本。"""
    text = raw.decode("utf-8", "replace")
    objects: List[Dict[str, Any]] = []
    fmt = "json"
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError("not_object")
        objects.append(obj)
    except (ValueError, TypeError):
        fmt = "ndjson"
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(item, dict):
                objects.append(item)
    final = None
    for obj in reversed(objects):
        if obj.get("done") is True:
            final = obj
            break
    if final is None and objects:
        final = objects[-1]
    present = [key for key in REQUIRED_USAGE_KEYS if final and _is_number(final.get(key))]
    missing = [key for key in REQUIRED_USAGE_KEYS if key not in present]
    return {
        "format": fmt,
        "object_count": len(objects),
        "done": bool(final and final.get("done") is True),
        "usage_keys_present": present,
        "missing_required": missing,
        "numeric_ok": not missing,
        "content_type": content_type.split(";", 1)[0].strip().lower(),
    }


def _with_model(model: Optional[str], body: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(body)
    if model:
        result["model"] = model
    return result


def _transport_detail(reply: Reply, label: str) -> str:
    mapping = {
        "connection_refused": "连接被拒绝",
        "timeout": "请求超时",
        "network_error": "网络不可达或连接失败",
        "client_error": "客户端异常",
    }
    return "%s：%s" % (label, mapping.get(reply.transport_error or "", "请求失败"))


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("需要正数") from exc
    if value <= 0 or not math.isfinite(value):
        raise argparse.ArgumentTypeError("需要正数")
    return value


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="校验 Qllama RC 的 OpenAI 兼容与 Ollama 原生接口"
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="被测服务基址")
    parser.add_argument("--api-key", default=None, help="可选的 Bearer 凭据")
    parser.add_argument("--tenant", default=None, help="可选的租户请求头值")
    parser.add_argument("--model", default=None, help="模型名；省略时从 /v1/models 发现")
    parser.add_argument("--timeout", type=_positive_float, default=30.0, help="单次请求超时秒数")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="JSON 摘要输出路径")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="兼容开关；默认模式同样在任何检查失败时返回非零",
    )
    parser.add_argument(
        "--skip",
        default="",
        help="逗号分隔的检查名列表；跳过的检查标记为 skipped=true，不计入失败",
    )
    return parser.parse_args(argv)


def _emit(summary: Dict[str, Any], output_path: str) -> bool:
    """写文件并打印同一份 JSON 摘要；写失败时如实标记。"""
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


def run_checks(client: Client, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[str]]:
    checks: List[Dict[str, Any]] = []
    skipped: List[str] = []
    skip_names = {s.strip() for s in (getattr(args, "skip", "") or "").split(",") if s.strip()}

    def add(name: str, ok: bool, status: int, detail: str) -> None:
        if name in skip_names:
            checks.append({"name": name, "ok": True, "status": int(status), "skipped": True, "detail": "已跳过：%s" % detail})
            skipped.append(name)
            return
        checks.append({"name": name, "ok": bool(ok), "status": int(status), "detail": detail})

    def record(
        name: str,
        reply: Reply,
        expected_status: int,
        extra_ok: bool,
        success_detail: str,
        label: str,
    ) -> None:
        if reply.transport_error:
            add(name, False, reply.status, _transport_detail(reply, label))
            return
        if reply.status != expected_status:
            add(
                name,
                False,
                reply.status,
                "%s：期望 status=%d，实际 %d" % (label, expected_status, reply.status),
            )
            return
        if not extra_ok:
            add(name, False, reply.status, "%s：响应结构不符合要求" % label)
            return
        add(name, True, reply.status, success_detail)

    def unsupported_check(
        name: str,
        reply: Reply,
        expected_status: int,
        markers: Tuple[str, ...],
        label: str,
    ) -> None:
        if reply.transport_error:
            add(name, False, reply.status, _transport_detail(reply, label))
            return
        info = _unsupported_info(reply)
        found = [marker for marker in markers if marker in info["text"]]
        ok = (
            reply.status == expected_status
            and info["has_error"]
            and info["code_signal"]
            and bool(found)
        )
        if ok:
            location = info["locations"][0] if info["locations"] else "error_code"
            detail = "%s：status=%d；%s 含 %s" % (
                label,
                reply.status,
                location,
                "/".join(found),
            )
        else:
            detail = "%s：status=%d；未检出符合要求的 qllama.unsupported 标记" % (
                label,
                reply.status,
            )
        add(name, ok, reply.status, detail)

    def ollama_usage_check(name: str, reply: Reply, label: str) -> None:
        if reply.transport_error:
            add(name, False, reply.status, _transport_detail(reply, label))
            return
        usage = _parse_ollama_usage(reply.raw, reply.content_type)
        ok = reply.status == 200 and usage["done"] and usage["numeric_ok"]
        if ok:
            detail = "%s：status=200；%s；done=true；usage 必需数值键齐全（%d 项）" % (
                label,
                usage["format"],
                len(usage["usage_keys_present"]),
            )
        else:
            detail = "%s：status=%d；usage 校验未通过（done=%s，缺失必需项 %d 个）" % (
                label,
                reply.status,
                str(usage["done"]).lower(),
                len(usage["missing_required"]),
            )
        add(name, ok, reply.status, detail)

    # 模型发现与 /v1/models 检查合并；失败时后续请求不携带 model 字段。
    model = args.model
    models_reply = client.request("GET", "/v1/models")
    models_data = models_reply.data
    first_id = None
    if isinstance(models_data, dict):
        entries = models_data.get("data")
        if isinstance(entries, list) and entries and isinstance(entries[0], dict):
            candidate = entries[0].get("id")
            if isinstance(candidate, str) and candidate:
                first_id = candidate
    if not model and first_id:
        model = first_id
    record(
        "models",
        models_reply,
        200,
        isinstance(models_data, dict) and first_id is not None,
        "模型列表可用；省略 --model 时已取首个模型 id",
        "模型列表",
    )

    health = client.request("GET", "/healthz")
    record("healthz", health, 200, True, "存活探针返回 200", "存活探针")

    metrics = client.request("GET", "/metrics", accept="text/plain")
    record(
        "metrics",
        metrics,
        200,
        b"qllama_requests_total" in metrics.raw,
        "指标文本包含 qllama_requests_total",
        "指标端点",
    )

    capabilities = client.request("GET", "/v1/capabilities")
    capability_data = capabilities.data
    capability_ok = (
        isinstance(capability_data, dict)
        and "unsupported_capabilities" in capability_data
        and "reproducibility" in capability_data
    )
    record(
        "capabilities",
        capabilities,
        200,
        capability_ok,
        "能力声明包含 unsupported_capabilities 与 reproducibility",
        "能力声明",
    )

    completions = client.request(
        "POST",
        "/v1/completions",
        payload=_with_model(
            model,
            {
                "prompt": PROMPT_SHORT,
                "max_tokens": MAX_TOKENS,
                "stream": False,
                "temperature": 0,
                "seed": SEED,
            },
        ),
    )
    record(
        "completions",
        completions,
        200,
        _completion_ok(completions, chat=False),
        "非流式补全返回 200 且 usage 计数完整",
        "文本补全",
    )

    chat = client.request(
        "POST",
        "/v1/chat/completions",
        payload=_with_model(
            model,
            {
                "messages": [{"role": "user", "content": PROMPT_SHORT}],
                "max_tokens": MAX_TOKENS,
                "stream": False,
                "temperature": 0,
                "seed": SEED,
            },
        ),
    )
    record(
        "chat_completions",
        chat,
        200,
        _completion_ok(chat, chat=True),
        "非流式对话补全返回 200 且 usage 计数完整",
        "对话补全",
    )

    stream = client.request(
        "POST",
        "/v1/completions",
        payload=_with_model(
            model,
            {
                "prompt": PROMPT_SHORT,
                "max_tokens": MAX_TOKENS,
                "stream": True,
                "stream_options": {"include_usage": True},
                "temperature": 0,
                "seed": SEED,
            },
        ),
        accept="text/event-stream",
    )
    stream_info = _sse_summary(stream)
    stream_ok = (
        stream.status == 200
        and not stream.transport_error
        and stream_info["content_type_ok"]
        and stream_info["done"]
        and stream_info["finish_reason_present"]
        and stream_info["usage_present"]
        and stream_info["usage_valid"]
        and stream_info["usage_in_final_chunk"]
        and not stream_info["error_chunk"]
    )
    if stream.transport_error:
        add("stream_include_usage", False, stream.status, _transport_detail(stream, "流式 usage"))
    elif stream_ok:
        add(
            "stream_include_usage",
            True,
            stream.status,
            "SSE 含 [DONE]、finish_reason，末帧携带完整 usage",
        )
    else:
        add(
            "stream_include_usage",
            False,
            stream.status,
            "流式 usage：SSE 终止帧或 usage 结构不符合要求",
        )

    json_object = client.request(
        "POST",
        "/v1/completions",
        payload=_with_model(
            model,
            {
                "prompt": PROMPT_JSON,
                "max_tokens": MAX_TOKENS,
                "stream": False,
                "seed": SEED,
                "response_format": {"type": "json_object"},
            },
        ),
    )
    record(
        "response_format_json_object",
        json_object,
        200,
        _json_choice_ok(json_object, chat=False, required_key=None),
        "json_object 输出可解析为 JSON，未写出内容",
        "JSON 输出约束",
    )

    schema = {
        "name": "conformance_object",
        "schema": {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
        },
    }
    json_schema = client.request(
        "POST",
        "/v1/chat/completions",
        payload=_with_model(
            model,
            {
                "messages": [{"role": "user", "content": PROMPT_JSON}],
                "max_tokens": MAX_TOKENS,
                "stream": False,
                "seed": SEED,
                "response_format": {"type": "json_schema", "json_schema": schema},
            },
        ),
    )
    record(
        "response_format_json_schema",
        json_schema,
        200,
        _json_choice_ok(json_schema, chat=True, required_key=None),
        "json_schema 输出可解析为合法 JSON 对象，未写出内容",
        "JSON Schema 约束",
    )

    tools_body = _with_model(
        model,
        {
            "messages": [{"role": "user", "content": PROMPT_TOOLS}],
            "max_tokens": MAX_TOKENS,
            "stream": False,
            "temperature": 0,
            "seed": SEED,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_time",
                        "description": "Read the current time",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
        },
    )
    tools = client.request("POST", "/v1/chat/completions", payload=tools_body)
    if tools.transport_error:
        add("tools_or_unsupported", False, tools.status, _transport_detail(tools, "工具调用"))
    elif tools.status == 200 and _tool_calls_valid(tools):
        add("tools_or_unsupported", True, tools.status, "返回有效 tool_calls")
    elif tools.status == 400 and _unsupported_info(tools)["code_signal"] and _error_shape(tools)["has_error"]:
        add("tools_or_unsupported", True, tools.status, "明确返回 400 qllama.unsupported")
    else:
        add(
            "tools_or_unsupported",
            False,
            tools.status,
            "既未返回有效 tool_calls，也未提供明确的 400 qllama.unsupported",
        )

    template = client.request(
        "POST",
        "/apply-template",
        payload=_with_model(
            model,
            {"messages": [{"role": "user", "content": PROMPT_TEMPLATE}]},
        ),
    )
    record(
        "apply_template",
        template,
        200,
        _apply_template_ok(template),
        "模板响应含 prompt 字段与数值 token_count，未写出 prompt",
        "模板接口",
    )

    best_of = client.request(
        "POST",
        "/v1/completions",
        payload=_with_model(
            model,
            {
                "prompt": PROMPT_SHORT,
                "max_tokens": MAX_TOKENS,
                "stream": False,
                "best_of": 2,
            },
        ),
    )
    unsupported_check(
        "unsupported_best_of",
        best_of,
        400,
        ("best_of",),
        "不支持字段声明",
    )

    embeddings = client.request(
        "POST",
        "/v1/embeddings",
        payload=_with_model(model, {"input": PROMPT_SHORT}),
    )
    unsupported_check(
        "embeddings_unsupported",
        embeddings,
        501,
        ("embedding", "rerank", "endpoint"),
        "Embedding 端点声明",
    )

    n_two = client.request(
        "POST",
        "/v1/completions",
        payload=_with_model(
            model,
            {
                "prompt": PROMPT_SHORT,
                "max_tokens": MAX_TOKENS,
                "stream": False,
                "n": 2,
            },
        ),
    )
    record("n_gt_1", n_two, 400, True, "n=2 被明确拒绝", "n>1 拒绝")

    if args.api_key:
        unauthenticated = client.request("GET", "/v1/models", auth=False)
        record(
            "api_key_required",
            unauthenticated,
            401,
            True,
            "未提供 Bearer 凭据时返回 401",
            "API key 校验",
        )
    else:
        skipped.append("api_key_required")

    version = client.request("GET", "/api/version")
    record(
        "ollama_version",
        version,
        200,
        isinstance(version.data, dict),
        "Ollama 版本接口返回 JSON 对象",
        "Ollama 版本",
    )

    tags = client.request("GET", "/api/tags")
    tags_ok = isinstance(tags.data, dict) and isinstance(tags.data.get("models"), list)
    record(
        "ollama_tags",
        tags,
        200,
        tags_ok,
        "Ollama 标签接口返回模型列表",
        "Ollama 标签",
    )

    ps = client.request("GET", "/api/ps")
    record(
        "ollama_ps",
        ps,
        200,
        isinstance(ps.data, dict),
        "Ollama 进程接口返回 JSON 对象",
        "Ollama 进程",
    )

    show_body = _with_model(model, {})
    if model:
        show_body["name"] = model
    show = client.request("POST", "/api/show", payload=show_body)
    record(
        "ollama_show",
        show,
        200,
        isinstance(show.data, dict),
        "Ollama show 接口返回 JSON 对象",
        "Ollama show",
    )

    options = {"num_predict": MAX_TOKENS, "seed": SEED, "temperature": 0}
    generate = client.request(
        "POST",
        "/api/generate",
        payload=_with_model(
            model,
            {
                "prompt": PROMPT_SHORT,
                "stream": False,
                "options": options,
            },
        ),
    )
    ollama_usage_check("ollama_generate", generate, "Ollama generate")

    chat_native = client.request(
        "POST",
        "/api/chat",
        payload=_with_model(
            model,
            {
                "messages": [{"role": "user", "content": PROMPT_SHORT}],
                "stream": False,
                "options": options,
            },
        ),
    )
    ollama_usage_check("ollama_chat", chat_native, "Ollama chat")

    embed = client.request(
        "POST",
        "/api/embed",
        payload=_with_model(model, {"input": PROMPT_SHORT}),
    )
    if embed.transport_error:
        add("ollama_embed_unsupported", False, embed.status, _transport_detail(embed, "Ollama embed"))
    else:
        shape = _error_shape(embed)
        ok = embed.status == 501 and shape["has_error"]
        detail = "status=501；返回错误体" if ok else "status=%d；未返回符合要求的 501 错误体" % embed.status
        if ok and _unsupported_info(embed)["code_signal"]:
            detail += "；含 unsupported 标记"
        add("ollama_embed_unsupported", ok, embed.status, detail)

    return checks, skipped


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    summary: Dict[str, Any] = {
        "passed": 0,
        "failed": 0,
        "total": 0,
        "checks": [],
        "strict": bool(args.strict),
        "exit_policy": "any_failed_check_nonzero",
        "skipped_checks": [],
        "api_key_provided": bool(args.api_key),
        "tenant_provided": bool(args.tenant),
        "timeout_seconds": args.timeout,
        "output": str(args.output),
    }
    try:
        client = Client(args.base_url, args.api_key, args.tenant, args.timeout)
    except ValueError:
        summary["error"] = "base_url_invalid"
        summary["base_url"] = "invalid"
        written = _emit(summary, args.output)
        return 2 if written else 1

    summary["base_url"] = client.safe_base_url
    checks, skipped = run_checks(client, args)
    summary["checks"] = checks
    summary["skipped_checks"] = skipped
    summary["total"] = len(checks)
    summary["passed"] = sum(1 for check in checks if check["ok"])
    summary["failed"] = summary["total"] - summary["passed"]
    written = _emit(summary, args.output)
    if not written or summary["failed"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
