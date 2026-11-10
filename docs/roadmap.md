# Roadmap

三层：已完成 / 下一阶段 / 远期。每项给理由（为什么做或不做）。

## 已完成（MVP → 服务化）

| 能力 | 证据 |
|------|------|
| GGUF 加载（F32/F16/Q8_0/Q4_0，mmap 零拷贝） | tests/gguf_*.rs、docs/mvp-quantization.md |
| Transformer 前向（RoPE/RMSNorm/GQA/SwiGLU）与 llama.cpp 数值对照 | docs/llamacpp-comparison.md，F32 全词表 max 5.2e-05 |
| KV cache 增量推理 | docs/development-plan.md 阶段 5，362 测试 |
| 分词器（SPM/BPE + byte fallback + 流式解码） | tests/tokenizer.rs |
| 采样（temperature/top-k/top-p/repeat-penalty，可播种） | tests/sampler.rs |
| CLI（generate/stream/info/logits） | tests/cli.rs |
| OpenAI 兼容 HTTP 服务（串行调度 + SSE + 取消 + 背压 + 资源保护） | tests/server_integration.rs，ADR 0001-0004 |
| 模糊测试（GGUF 解析器 / 分词器健壮性） | tests/fuzz.rs（proptest） |
| 部署（Dockerfile / compose / systemd / --config） | Dockerfile、deploy/qllama.service |

## 下一阶段（按收益排序）

1. **单请求多线程 decode 的真并行**：当前 GEMV 已并行，但整条 decode
   链在调度线程串行。收益最大的一步是把层间/算子间流水化，
   预期 2-4× 吞吐（63.4% 时间在 MLP，docs/mvp-profile.md）。
   前置：profile 数据复核，不变量是数值对照不回退。
2. **Chat 模板（jinja 子集）**：读取 GGUF 内 `tokenizer.chat_template`
   并渲染。当前降级 ChatML 对非 ChatML 模型语义损失明显。
   范围控制：只实现 llama 系模板用到的 jinja 特性。
3. **continuous batching**：多请求共享 decode 循环（每步组批）。
   串行引擎的吞吐天花板明确，批处理是标准解法。前置：1 完成后
   重估收益；KV 池化（分块）随批处理一起做，替代整块 per-request KV。
4. **流式 usage / stream_options.include_usage**：补齐 OpenAI 流式契约
   的最后一块，客户端兼容性收尾。

## 远期（明确不做/暂不做及原因）

- **GPU/CUDA/Metal 后端**：项目红线（无 GPU 依赖、无 C 绑定），
  除非红线变更否则不进入任何阶段。
- **分布式/多机推理**：单机定位（任务书范围），通信开销与复杂度
  远超当前场景需求。
- **PagedAttention 级 KV 管理**：需要块表 + 写时复制 + 前缀共享，
  复杂度高；在批处理（下一阶段 3）落地后再评估简化版块池。
- **量化训练/权重重打包工具**：推理引擎范围外；消费上游 GGUF 即可。
- **embedding/rerank 端点**：当前模型栈是 decoder-only 生成模型，
  embedding 质量无对照基准，不做无验证的功能。
- **Windows 服务化（Windows Service 包装）**：用户群极小，
  Windows 上推荐 docker compose 或手动运行；有真实需求再加。
