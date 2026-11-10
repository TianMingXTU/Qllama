# ADR 0004：OpenAI 兼容 API 范围

状态：已接受　日期：2026-11-11

## 支持的端点

| 端点 | 方法 | 说明 |
|---|---|---|
| `/healthz` | GET | 存活探针，200 无 body |
| `/v1/models` | GET | OpenAI 格式对象列表，单模型（`--model` 指定的那个） |
| `/v1/completions` | POST | OpenAI legacy 补全格式 |
| `/v1/chat/completions` | POST | OpenAI chat 格式，支持 `stream` |
| `/metrics` | GET | 文本格式指标（QPS、TTFT、tokens/s、队列深度、p99） |

## `/v1/completions` 与 `/v1/chat/completions` 支持的字段

prompt / messages、`model`（校验与加载的模型一致，不匹配报 404）、
`max_tokens`、`temperature`、`top_p`、`top_k`（OpenAI 没有但 llama 生态常用，
显式支持）、`stop`（字符串或数组，最多 4 条）、`stream`、`seed`、`n`（仅
n=1，n>1 返回 400——串行引擎下 n>1 会排队执行，语义等价但延迟不可预期，
显式拒绝比慢吞吞执行诚实）。

## 显式不支持的请求字段（收到时忽略并在响应 `qllama.unsupported` 中列出）

`tools`、`tool_choice`、`function_call`、`functions`、`logprobs`、
`logit_bias`、`response_format`、`presence_penalty`、`frequency_penalty`。

忽略而非报错的原因：OpenAI SDK 与多数客户端会带上默认值字段；硬报错会把
"能用的功能"变"不能用的客户端"。但必须让调用方知道字段没生效——静默吞掉
违反诚实性红线。`stream_options.include_usage` 支持（返回 usage 块）。

## chat 的降级模式

- GGUF 含 `tokenizer.chat_template`（jinja）：本轮**不解析 jinja**，检测到
  存在但无解析器时走降级并在启动日志与响应头 `x-qllama-chat-mode: fallback`
  声明；完整 jinja 模板进 roadmap。
- 降级格式（文档化，不静默猜测）：`<|im_start|>{role}\n{content}<|im_end|>\n`
  拼接 + `<|im_start|>assistant\n` 结尾。system 消息放在最前。
- 模型无任何 chat 元数据时：`/v1/chat/completions` 仍可用降级格式（用户已
  被文档告知这是通用拼法，不是该模型的官方模板）。

## SSE 协议

- `Content-Type: text/event-stream`，chunk 格式 `data: {json}\n\n`，结束
  `data: [DONE]\n\n`。
- 每 chunk 携带增量 delta（chat：`choices[].delta.content`；completions：
  `choices[].text`），`finish_reason` 在最后一个数据 chunk 上给 `stop`/`length`。
- 每 15s 发一行 `: keepalive\n` 注释心跳，防代理空闲超时。
- 错误体：`{"error": {"type", "message", "code"}}`；400 参数、404 模型、
  429 队列满（附 `Retry-After`）、503 过载/KV 分配失败、500 内部。

## 限制项（服务启动参数）

`--max-request-bytes`（默认 1 MiB）、`--max-prompt-tokens`（默认 ctx 上限）、
`--max-concurrency`（默认 = 队列容量）、`--request-timeout-secs`（默认 300）、
`--queue-capacity`（默认 32）。
