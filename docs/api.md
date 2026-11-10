# Qllama HTTP API

OpenAI 兼容子集。基址 `http://127.0.0.1:8080`（默认仅回环）。

## 认证

默认无认证。启动加 `--api-key <token>` 后，所有 `/v1/*` 请求须带
`Authorization: Bearer <token>`；缺失或错误返回 401。`/healthz` 与
`/metrics` 永远开放（运维探活用）。

## 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/healthz` | 存活探针，返回 `ok` |
| GET  | `/metrics` | Prometheus 文本格式指标 |
| GET  | `/v1/models` | 已加载模型列表（单个） |
| POST | `/v1/completions` | 文本补全 |
| POST | `/v1/chat/completions` | 对话补全 |

## `/v1/completions`

请求字段：

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `prompt` | string \| string[1] | 必填 | 数组只接受单元素 |
| `max_tokens` | int | 16 | 上限为剩余上下文 |
| `temperature` | float | 1.0 | `<=0` 为贪心（argmax） |
| `top_p` | float | 1.0 | 核采样阈值 |
| `top_k` | int | 0（不限） | 非 OpenAI 标准字段，llama.cpp 风格 |
| `stop` | string \| string[] | 无 | 最多 4 条；解码后文本前缀匹配，命中即截断 |
| `stream` | bool | false | SSE 流式 |
| `seed` | int | 随机 | 可播种采样 |
| `model` | string | — | 必须与加载的模型一致（前缀匹配），否则 404 |

响应（非流式）：

```json
{
  "id": "cmpl-…", "object": "text_completion", "created": 1760000000,
  "model": "stories15M-q4_0",
  "choices": [{"text": "…", "index": 0, "finish_reason": "stop|length"}],
  "usage": {"prompt_tokens": 5, "completion_tokens": 24, "total_tokens": 29}
}
```

## `/v1/chat/completions`

`messages` 数组（`role` + `content`），其余字段同上。响应 choice 用
`message: {role: "assistant", content: …}`，object 为 `chat.completion`。

**模板降级声明（诚实性）**：Qllama 没有 jinja 模板解析器，对话统一用
ChatML 拼法（`<|im_start|>{role}\n{content}<|im_end|>\n` + assistant 起）。
对以 ChatML 训练的模型这是官方模板；对其他模型（如 TinyLlama）
生成质量会低于 llama.cpp 的官方模板路径。模板支持在 roadmap。

## 流式（`"stream": true`）

SSE（`text/event-stream`），每 token 一帧：

```
data: {"id":"cmpl-…","object":"text_completion.chunk","created":…,"model":"…",
       "choices":[{"text":"called","index":0,"finish_reason":null}]}

data: {"…","choices":[{"text":"","index":0,"finish_reason":"length"}]}

data: [DONE]
```

- 15 秒 keepalive 心跳（注释行 `: keepalive`）。
- 结束帧 `finish_reason` 非空，随后固定 `data: [DONE]`。
- 服务端错误以 `data: {"error": {...}}` 帧下发，之后仍以 `[DONE]` 收尾。
- `usage` 与 `stream_options.include_usage` 未实现（见 roadmap）。

客户端断开连接即取消生成（通常在下一个 decode step 停止，≤2s 量级），
KV cache 随之释放。

## 错误体

所有错误为 OpenAI 形状：

```json
{"error": {"message": "…", "type": "invalid_request_error", "code": 400}}
```

| 状态 | type | 场景 |
|------|------|------|
| 400 | invalid_request_error | JSON 解析失败、参数越界、prompt 超上下文 |
| 401 | authentication_error | API key 缺失/错误 |
| 404 | invalid_request_error | 模型名不匹配 |
| 413 | — | 请求体超过 `--max-request-bytes`（axum 默认格式） |
| 429 | rate_limit_error | 队列满；带 `Retry-After: 1` |
| 499 | client_closed_request | 客户端断连导致的取消 |
| 500 | internal_error | 生成循环内部错误 |
| 503 | service_unavailable | KV 字节预算超限 / 分配失败 |
| 504 | timeout | 请求超过 `--request-timeout-secs` |

## 接受但忽略的字段

OpenAI 客户端会发送以下字段，本引擎解析后忽略（不报错）：
`frequency_penalty`、`presence_penalty`、`logprobs`、`logit_bias`、
`response_format`、`tools`、`tool_choice`、`echo`、`best_of`、`user`。
`n > 1` 显式拒绝（串行引擎）。

## 指标（/metrics）

```
qllama_requests_total        累计请求数（counter）
qllama_tokens_generated_total 累计生成 token（counter）
qllama_queue_depth           当前排队数（gauge，不含执行中）
qllama_uptime_seconds        运行时长（gauge）
```

## 串行执行声明

引擎一次只执行一个请求的 decode 循环，其余请求 FIFO 排队——这是排队，
不是 continuous batching。并发吞吐 ≈ 单请求吞吐；并发请求的延迟 =
排队时间 + 自身推理时间。多请求 KV 隔离已由集成测试验证（各请求独立
KV cache 与采样状态，无串台）。
