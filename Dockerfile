# 多阶段构建：构建产物只有 qllama-server 一个静态依赖的 ELF，运行镜像用 slim。
# 构建上下文需要 src/、Cargo.toml、Cargo.lock（examples/tests 不参与服务构建）。
FROM rust:1-slim AS builder
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src ./src
# 服务二进制只依赖核心库；server feature 打开 tokio/axum/futures。
RUN cargo build --release --features server --bin qllama-server

FROM debian:bookworm-slim
# healthcheck 用 curl 探测 /healthz；不装其他运行时依赖。
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 1001 --create-home qllama

COPY --from=builder /build/target/release/qllama-server /usr/local/bin/qllama-server

# 模型文件挂载点：只读使用，宿主机负责放置 GGUF。
VOLUME ["/models"]
EXPOSE 8080

# 默认仅监听回环——容器内回环不可从宿主机访问，compose/k8s 场景必须
# 显式覆盖 --listen 0.0.0.0:8080（见 docker-compose.yml）。
ENTRYPOINT ["/usr/local/bin/qllama-server"]
CMD ["--model", "/models/model.gguf", "--listen", "0.0.0.0:8080"]

USER qllama

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/healthz || exit 1
