//! `qllama-server`: OpenAI compatible HTTP service entry point.
//!
//! Defaults to loopback `127.0.0.1:8080` and one GGUF model. Configuration
//! comes from `--config`, environment variables and command-line flags, with
//! command-line values taking precedence. V1 does not terminate TLS itself;
//! see `docs/ops-runbook.md` for the reverse-proxy boundary.

#[cfg(feature = "server")]
mod inner {
    use std::{
        collections::HashMap,
        fs,
        net::SocketAddr,
        path::{Path, PathBuf},
        time::Duration,
    };

    use anyhow::{bail, Context, Result};
    use clap::Parser;
    use qllama::server::{
        auth::{self, ApiKey, Role},
        qos, Limits, RuntimeConfig, RuntimeOptions,
    };
    use serde::Deserialize;

    /// Configuration file schema; unknown fields are rejected so that a
    /// misspelled operations setting cannot be silently ignored.
    #[derive(Debug, Default, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct FileConfig {
        model: Option<PathBuf>,
        model_dir: Option<PathBuf>,
        models_dir: Option<PathBuf>,
        listen: Option<String>,
        threads: Option<usize>,
        ctx_size: Option<usize>,
        queue_capacity: Option<usize>,
        request_timeout_secs: Option<u64>,
        max_request_bytes: Option<usize>,
        api_key: Option<String>,
        api_keys: Option<Vec<FileApiKey>>,
        api_key_file: Option<PathBuf>,
        kv_budget_mib: Option<usize>,
        kv_block_size: Option<usize>,
        cont_batching: Option<bool>,
        no_cont_batching: Option<bool>,
        single_request_parallel: Option<bool>,
        no_single_request_parallel: Option<bool>,
        prefix_cache: Option<bool>,
        cors_origins: Option<Vec<String>>,
        audit_log: Option<PathBuf>,
        public_health: Option<bool>,
        public_metrics: Option<bool>,
        tls_cert: Option<PathBuf>,
        tls_key: Option<PathBuf>,
    }

    #[derive(Debug, Clone, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct FileApiKey {
        id: String,
        secret: String,
        #[serde(default = "default_role")]
        role: String,
        #[serde(default = "default_tenant")]
        tenant: String,
        #[serde(default)]
        revoked: bool,
        expires_at: Option<u64>,
        quota: Option<FileQuota>,
    }

    #[derive(Debug, Clone, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct FileQuota {
        max_in_flight: Option<usize>,
        requests_per_minute: Option<usize>,
        tokens_per_minute: Option<usize>,
    }

    fn default_role() -> String {
        "inference".into()
    }

    fn default_tenant() -> String {
        "default".into()
    }

    #[derive(Debug, Parser)]
    #[command(name = "qllama-server", about = "Qllama OpenAI-compatible HTTP server")]
    struct Cli {
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long)]
        models_dir: Option<PathBuf>,
        #[arg(long)]
        listen: Option<String>,
        #[arg(long)]
        threads: Option<usize>,
        #[arg(long)]
        ctx_size: Option<usize>,
        #[arg(long)]
        queue_capacity: Option<usize>,
        #[arg(long)]
        request_timeout_secs: Option<u64>,
        #[arg(long)]
        max_request_bytes: Option<usize>,
        #[arg(long)]
        api_key: Option<String>,
        #[arg(long)]
        api_key_file: Option<PathBuf>,
        #[arg(long)]
        kv_budget_mib: Option<usize>,
        #[arg(long)]
        kv_block_size: Option<usize>,
        #[arg(long)]
        cont_batching: bool,
        #[arg(long)]
        no_cont_batching: bool,
        #[arg(long)]
        single_request_parallel: bool,
        #[arg(long)]
        no_single_request_parallel: bool,
        #[arg(long)]
        prefix_cache: bool,
        #[arg(long, value_delimiter = ',')]
        cors_origins: Option<Vec<String>>,
        #[arg(long)]
        audit_log: Option<PathBuf>,
        #[arg(long)]
        public_health: Option<bool>,
        #[arg(long)]
        public_metrics: Option<bool>,
        #[arg(long)]
        tls_cert: Option<PathBuf>,
        #[arg(long)]
        tls_key: Option<PathBuf>,
    }

    #[derive(Debug)]
    struct ServerArgs {
        model: PathBuf,
        model_dir: Option<PathBuf>,
        listen: SocketAddr,
        threads: Option<usize>,
        ctx_size: Option<usize>,
        limits: Limits,
        api_key: Option<String>,
        runtime: RuntimeConfig,
    }

    impl ServerArgs {
        fn resolve(cli: Cli) -> Result<Self> {
            let file = match &cli.config {
                Some(path) => {
                    let text = fs::read_to_string(path)
                        .with_context(|| format!("cannot read config {}", path.display()))?;
                    serde_json::from_str::<FileConfig>(&text)
                        .with_context(|| format!("cannot parse config {}", path.display()))?
                }
                None => FileConfig::default(),
            };
            if file.tls_cert.is_some()
                || file.tls_key.is_some()
                || cli.tls_cert.is_some()
                || cli.tls_key.is_some()
            {
                bail!(
                    "built-in TLS is not implemented in V1; terminate TLS at a reverse proxy (see docs/ops-runbook.md)"
                );
            }
            if cli.cont_batching && cli.no_cont_batching {
                bail!("--cont-batching and --no-cont-batching cannot be combined");
            }
            if cli.single_request_parallel && cli.no_single_request_parallel {
                bail!(
                    "--single-request-parallel and --no-single-request-parallel cannot be combined"
                );
            }
            let model_dir = cli
                .model_dir
                .or(cli.models_dir)
                .or(file.model_dir)
                .or(file.models_dir);
            let listen = cli
                .listen
                .or(file.listen)
                .unwrap_or_else(|| "127.0.0.1:8080".into())
                .parse::<SocketAddr>()
                .context("listen must be host:port")?;
            let model = cli.model.or(file.model).context("--model is required")?;
            let api_key = cli.api_key.clone().or_else(|| file.api_key.clone());
            let key_file = cli
                .api_key_file
                .clone()
                .or_else(|| file.api_key_file.clone());
            let mut keys = load_api_keys(key_file.as_deref(), file.api_keys.as_deref())?;
            if let Some(secret) = &api_key {
                if keys.is_empty() {
                    keys.push(ApiKey {
                        id: "primary".into(),
                        secret: secret.clone(),
                        role: Role::Admin,
                        tenant: "default".into(),
                        revoked: false,
                        expires_at: None,
                    });
                }
            }
            let mut runtime = RuntimeConfig::default();
            runtime.auth.keys = keys;
            runtime.auth.public_health = cli.public_health.or(file.public_health).unwrap_or(true);
            runtime.auth.public_metrics =
                cli.public_metrics.or(file.public_metrics).unwrap_or(true);
            runtime.auth.cors_origins = cli.cors_origins.or(file.cors_origins).unwrap_or_default();
            runtime.auth.oidc_enabled = false;
            runtime.options = RuntimeOptions {
                public_health: runtime.auth.public_health,
                public_metrics: runtime.auth.public_metrics,
                cors_origins: runtime.auth.cors_origins.clone(),
                oidc_enabled: false,
                continuous_batching: if cli.no_cont_batching {
                    false
                } else if cli.cont_batching {
                    true
                } else {
                    file.cont_batching.unwrap_or(true) && !file.no_cont_batching.unwrap_or(false)
                },
                single_request_parallel: if cli.no_single_request_parallel {
                    false
                } else if cli.single_request_parallel {
                    true
                } else {
                    file.single_request_parallel.unwrap_or(true)
                        && !file.no_single_request_parallel.unwrap_or(false)
                },
                prefix_cache: cli.prefix_cache || file.prefix_cache.unwrap_or(false),
                kv_block_size: cli.kv_block_size.or(file.kv_block_size).unwrap_or(32),
                audit_log: cli.audit_log.clone().or_else(|| file.audit_log.clone()),
                api_key_file: key_file,
                tls_cert: None,
                tls_key: None,
            };
            runtime.default_quota = qos::Quota {
                max_in_flight: 64,
                requests_per_minute: 100_000,
                tokens_per_minute: 100_000_000,
            };
            runtime.per_key_quota = file
                .api_keys
                .as_deref()
                .unwrap_or(&[])
                .iter()
                .filter_map(|key| key.quota.as_ref().map(|quota| (key.id.clone(), quota)))
                .map(|(id, quota)| {
                    let defaults = runtime.default_quota;
                    (
                        id,
                        qos::Quota {
                            max_in_flight: quota.max_in_flight.unwrap_or(defaults.max_in_flight),
                            requests_per_minute: quota
                                .requests_per_minute
                                .unwrap_or(defaults.requests_per_minute),
                            tokens_per_minute: quota
                                .tokens_per_minute
                                .unwrap_or(defaults.tokens_per_minute),
                        },
                    )
                })
                .collect::<HashMap<_, _>>();
            let limits = Limits {
                queue_capacity: cli.queue_capacity.or(file.queue_capacity).unwrap_or(32),
                request_timeout: Duration::from_secs(
                    cli.request_timeout_secs
                        .or(file.request_timeout_secs)
                        .unwrap_or(300),
                ),
                max_request_bytes: cli
                    .max_request_bytes
                    .or(file.max_request_bytes)
                    .unwrap_or(1 << 20),
                kv_byte_budget: cli.kv_budget_mib.or(file.kv_budget_mib).unwrap_or(2048)
                    * 1024
                    * 1024,
            };
            Ok(Self {
                model,
                model_dir,
                listen,
                threads: cli.threads.or(file.threads),
                ctx_size: cli.ctx_size.or(file.ctx_size),
                limits,
                api_key,
                runtime,
            })
        }
    }

    fn load_api_keys(
        path: Option<&Path>,
        configured: Option<&[FileApiKey]>,
    ) -> Result<Vec<ApiKey>> {
        let mut keys = Vec::new();
        if let Some(path) = path {
            let text = fs::read_to_string(path)
                .with_context(|| format!("cannot read API key file {}", path.display()))?;
            keys.extend(auth::parse_key_file(&text).map_err(anyhow::Error::msg)?);
        }
        for key in configured.unwrap_or(&[]) {
            let role = match key.role.as_str() {
                "read" => Role::Read,
                "inference" => Role::Inference,
                "admin" => Role::Admin,
                other => bail!("unknown API key role {other:?}"),
            };
            keys.push(ApiKey {
                id: key.id.clone(),
                secret: key.secret.clone(),
                role,
                tenant: key.tenant.clone(),
                revoked: key.revoked,
                expires_at: key.expires_at,
            });
        }
        Ok(keys)
    }

    /// Loads one GGUF, starts the server and joins the scheduler after drain.
    pub async fn run() -> Result<()> {
        let args = ServerArgs::resolve(Cli::parse())?;
        qllama::tensor::set_single_request_parallel(args.runtime.options.single_request_parallel);
        let engine = qllama::server::Engine::load(
            &args.model,
            args.model_dir.as_deref(),
            args.threads,
            args.ctx_size,
        )
        .map_err(anyhow::Error::msg)?;
        let audit_path = args.runtime.options.audit_log.clone();
        let (state, scheduler) = qllama::server::ServerState::start_with_runtime(
            engine,
            args.limits,
            args.api_key.clone(),
            args.runtime,
        )
        .map_err(anyhow::Error::msg)?;
        eprintln!(
            "{}",
            serde_json::json!({
                "event": "server_started",
                "listen": args.listen.to_string(),
                "audit_log": audit_path,
            })
        );
        qllama::server::serve(state.clone(), args.listen).await?;
        drop(state);
        tokio::task::spawn_blocking(move || scheduler.join())
            .await
            .context("scheduler join task failed")?
            .map_err(|_| anyhow::anyhow!("scheduler thread panicked"))?;
        Ok(())
    }
}

#[cfg(not(feature = "server"))]
fn main() {
    eprintln!("qllama-server requires the `server` feature");
}

#[cfg(feature = "server")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    inner::run().await
}
