# Qllama Rust native CPU-only text-generation inference image.
# 本 Dockerfile 不应把 target/ 作为构建源依赖；只复制 Cargo 清单与源码，
# 在容器内 cargo build，避免宿主机 target 污染与缓存失效。

# ---- builder ----
FROM rust:1.83-slim AS builder
WORKDIR /build

# 先复制清单以最大化 cargo 缓存命中；Cargo.lock 必须随代码提交。
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src \
 && echo "fn main() {}" > src/main.rs

# 复制真实源码（含 src/bin/server.rs 与 server feature）。
COPY src ./src
COPY tests ./tests

RUN cargo build --locked --release --features server --bin qllama-server

# ---- runtime ----
FROM debian:bookworm-slim AS runtime
WORKDIR /app

# 非 root 用户；与 docker-compose / systemd 的非 root 假设保持一致。
RUN useradd --uid 1001 --no-create-home --shell /usr/sbin/nologin qllama \
 && mkdir -p /models /var/log/qllama \
 && chown -R qllama:qllama /var/log/qllama

COPY --from=builder /build/target/release/qllama-server /usr/local/bin/qllama-server

USER qllama
EXPOSE 8080

# 进程只监听回环或 0.0.0.0 由 --listen 决定；生产经反向代理终止 TLS。
# 审计日志目录已在 runtime 阶段 chown 给 qllama，可写。
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8080/healthz || exit 1

CMD ["qllama-server", \
     "--models-dir", "/models", \
     "--model", "/models/model.gguf", \
     "--listen", "0.0.0.0:8080", \
     "--kv-block-size", "32", \
     "--cont-batching", \
     "--audit-log", "/var/log/qllama/audit.jsonl"]
