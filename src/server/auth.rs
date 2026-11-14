//! 企业级访问控制：API Key 鉴权、角色分级、CORS 判定与链路追踪标识。
//!
//! 本模块只做纯逻辑判定，不接触 HTTP 框架与任何请求体内容；调用方
//! （`server::api`）负责把头部转成 `(name, value)` 列表并消费判定结果。
//! 错误串均为稳定的英文机器标识（便于映射 HTTP 状态码），且保证不含密钥。
//!
//! 与其他模块的关系：`audit` 消费本模块产出的 `AuthContext::key_hash`
//! 做日志假名化；`qos` 按租户/密钥维度限流。

/// 密钥角色，权限按 `Read < Inference < Admin` 单调递增。
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// 只读：模型列表、健康检查等。
    Read,
    /// 推理：在 Read 之上允许补全/对话请求。
    Inference,
    /// 管理：在 Inference 之上允许运维类操作。
    Admin,
}

impl Role {
    /// 判断当前角色是否满足 `required` 要求（允许高角色满足低要求）。
    pub fn allows(self, required: Role) -> bool {
        (self as u8) >= (required as u8)
    }
}

/// 一条 API Key 记录；`secret` 只在鉴权比较时使用，禁止写入日志或错误信息。
#[derive(Clone, Debug)]
pub struct ApiKey {
    /// 密钥的业务标识（日志与限流维度使用，非机密）。
    pub id: String,
    /// 密钥明文。仅用于鉴权比较，不得打印、不得进入错误信息。
    pub secret: String,
    /// 授权角色。
    pub role: Role,
    /// 所属租户，配额与审计按租户维度划分。
    pub tenant: String,
    /// 是否已吊销；吊销后立即拒绝，即使密钥本身正确。
    pub revoked: bool,
    /// 过期时间戳（Unix 秒）；`now >= expires_at` 视为过期，`None` 表示永不过期。
    pub expires_at: Option<u64>,
}

/// 鉴权成功后的上下文，供后续中间件（审计、限流、日志）使用。
#[derive(Clone, Debug)]
pub struct AuthContext {
    /// 命中的密钥 id。
    pub key_id: String,
    /// 密钥的 FNV-1a 假名化哈希（见 [`hash_key`]），用于审计日志；非加密哈希。
    pub key_hash: String,
    /// 所属租户。
    pub tenant: String,
    /// 授权角色。
    pub role: Role,
    /// 链路追踪 id（优先取合法 `traceparent` 头，否则按时间/进程/计数器生成）。
    pub trace_id: String,
    /// 本次请求 id，由时间戳、进程号与进程内原子计数器混合生成。
    pub request_id: String,
}

/// 鉴权配置：密钥清单与公开端点、CORS、OIDC 开关。
#[derive(Clone, Debug)]
pub struct AuthConfig {
    /// 已配置的密钥列表；为空时所有带凭证的请求都会被拒绝。
    pub keys: Vec<ApiKey>,
    /// `/health` 是否免鉴权。
    pub public_health: bool,
    /// `/metrics` 是否免鉴权。
    pub public_metrics: bool,
    /// CORS 允许的 Origin 精确匹配列表；空列表拒绝所有，`*` 放行所有。
    pub cors_origins: Vec<String>,
    /// 是否启用 OIDC。V1 阶段启用时按 [`oidc_status`] 返回 501 契约，绝不绕过密钥鉴权。
    pub oidc_enabled: bool,
}

impl Default for AuthConfig {
    /// 默认无密钥、健康与指标端点公开、CORS 全拒、OIDC 关闭。
    fn default() -> Self {
        Self {
            keys: Vec::new(),
            public_health: true,
            public_metrics: true,
            cors_origins: Vec::new(),
            oidc_enabled: false,
        }
    }
}

/// 解析密钥文件文本。
///
/// 每个非空行格式为 `id:secret[:role][:tenant]`，字段两侧空白会被裁剪；
/// `#` 开头的行与空行跳过。`role` 取 `read|inference|admin`（缺省 `inference`），
/// `tenant` 缺省 `default`。密钥字段本身不支持 `:`（属格式约定，需换值）。
///
/// 返回错误时包含行号与密钥 id（非机密），绝不包含 `secret` 本身。
pub fn parse_key_file(text: &str) -> Result<Vec<ApiKey>, String> {
    let mut keys = Vec::new();
    for (idx, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line_no = idx + 1;
        let parts: Vec<&str> = line.split(':').collect();
        if !(2..=4).contains(&parts.len()) {
            return Err(format!(
                "key file line {line_no}: expected id:secret[:role][:tenant]"
            ));
        }
        let id = parts[0].trim();
        if id.is_empty() {
            return Err(format!("key file line {line_no}: id must not be empty"));
        }
        let secret = parts[1].trim();
        if secret.is_empty() {
            return Err(format!(
                "key file line {line_no}: secret must not be empty (id: {id})"
            ));
        }
        let role = match parts.get(2).map(|r| r.trim()) {
            None => Role::Inference,
            Some("read") => Role::Read,
            Some("inference") => Role::Inference,
            Some("admin") => Role::Admin,
            Some(other) => {
                return Err(format!(
                    "key file line {line_no}: unknown role \"{other}\" (id: {id})"
                ))
            }
        };
        let tenant = parts
            .get(3)
            .map(|t| t.trim())
            .filter(|t| !t.is_empty())
            .unwrap_or("default");
        if keys.iter().any(|k: &ApiKey| k.id == id) {
            return Err(format!(
                "key file line {line_no}: duplicate key id \"{id}\""
            ));
        }
        keys.push(ApiKey {
            id: id.to_string(),
            secret: secret.to_string(),
            role,
            tenant: tenant.to_string(),
            revoked: false,
            expires_at: None,
        });
    }
    Ok(keys)
}

/// 计算密钥的稳定 FNV-1a 64 位哈希（16 位小写十六进制）。
///
/// 这是非加密哈希，仅用于日志与审计中的假名化标识（同一密钥在不同日志
/// 行中可关联、且不泄露明文）；密钥比较不使用它，见 [`constant_time_eq`]。
pub fn hash_key(secret: &str) -> String {
    format!("{:016x}", fnv1a64(secret.as_bytes()))
}

/// FNV-1a 64 位哈希的内部实现（也是 id/trace 生成的混合函数基础）。
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// 常量时间字节比较：比较耗时与内容无关，不因前缀匹配提前返回。
///
/// 长度差异会提前返回 `false`（泄露长度可接受，泄露前缀不可接受）。
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for i in 0..a.len() {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

use std::sync::atomic::AtomicU64;

/// 请求 id/trace id 生成用的进程内单调计数器。
static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(0);

/// 将若干 u64 混合成一个 64 位值（FNV-1a 逐字节），用于派生请求/追踪 id。
fn mix64(values: &[u64]) -> u64 {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fnv1a64(&bytes)
}

/// 由时间戳、进程号与进程内原子计数器派生请求 id 与 32 位小写十六进制
/// 的 W3C trace id（不依赖日期库或随机源；同进程内按顺序不重复）。
fn generate_ids(now: u64) -> (String, String) {
    let counter = REQUEST_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let pid = u64::from(std::process::id());
    let hi = mix64(&[now, pid, counter, 0x514c_6c61_6d61]);
    let lo = mix64(&[hi, now.rotate_left(32) ^ (pid << 1), counter]);
    // W3C 规定 trace_id 不得为全零；mix64 输出全零概率可忽略，仍显式规避。
    let (hi, lo) = if hi == 0 && lo == 0 {
        (1u64, 0u64)
    } else {
        (hi, lo)
    };
    let trace_id = format!("{hi:016x}{lo:016x}");
    let request_id = format!("req-{:016x}", mix64(&[lo, now, counter]));
    (request_id, trace_id)
}

/// 判定 `name` 是否等于 `expected`（ASCII 不区分大小写）。
fn header_eq(name: &str, expected: &str) -> bool {
    name.eq_ignore_ascii_case(expected)
}

/// 鉴权入口：校验 `authorization: Bearer <token>` 头并返回 [`AuthContext`]。
///
/// 判定顺序与错误串（均为稳定英文标识、不含密钥）：
/// 缺失头 -> `missing_authorization`；非 Bearer 方案 -> `malformed_authorization`；
/// 无匹配密钥 -> `invalid_credentials`；已吊销 -> `revoked_key`；
/// 已过期 -> `expired_key`；角色不足 -> `insufficient_role`。
///
/// 比较会对所有密钥执行 [`constant_time_eq`]（不因命中提前退出），
/// 吊销/过期的密钥也参与比较，以维持统一的错误分类与时序特征。
/// `now` 为 Unix 秒；`role` 为端点要求的最低角色。
pub fn authenticate(
    config: &AuthConfig,
    headers: &[(String, String)],
    role: Role,
    now: u64,
) -> Result<AuthContext, String> {
    let header = headers
        .iter()
        .find(|(name, _)| header_eq(name, "authorization"))
        .map(|(_, value)| value);
    let Some(value) = header else {
        return Err("missing_authorization".to_string());
    };
    let token = match value.trim().split_once(' ') {
        Some((scheme, rest)) if header_eq(scheme, "bearer") => rest.trim(),
        _ => return Err("malformed_authorization".to_string()),
    };

    // 常量时间比较所有密钥；先比密钥、再查状态，保证错误分类独立于时序。
    let mut matched: Option<usize> = None;
    for (idx, key) in config.keys.iter().enumerate() {
        if constant_time_eq(token.as_bytes(), key.secret.as_bytes()) {
            matched = Some(idx);
        }
    }
    let Some(idx) = matched else {
        return Err("invalid_credentials".to_string());
    };
    let key = &config.keys[idx];
    if key.revoked {
        return Err("revoked_key".to_string());
    }
    if let Some(expires_at) = key.expires_at {
        if now >= expires_at {
            return Err("expired_key".to_string());
        }
    }
    if !key.role.allows(role) {
        return Err("insufficient_role".to_string());
    }

    let (request_id, generated_trace) = generate_ids(now);
    let trace_id = parse_traceparent(headers).unwrap_or(generated_trace);
    Ok(AuthContext {
        key_id: key.id.clone(),
        key_hash: hash_key(&key.secret),
        tenant: key.tenant.clone(),
        role: key.role,
        trace_id,
        request_id,
    })
}

/// 校验 `is_lower_hex`：仅接受小写十六进制字符。
fn is_lower_hex(s: &str) -> bool {
    !s.is_empty()
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// 解析 W3C `traceparent` 头，格式合法（`00-<32hex>-<16hex>-<2hex>`，小写）
/// 时返回其中的 32 位 trace id，否则返回 `None`（由调用方自行生成）。
pub fn parse_traceparent(headers: &[(String, String)]) -> Option<String> {
    let value = headers
        .iter()
        .find(|(name, _)| header_eq(name, "traceparent"))
        .map(|(_, value)| value.trim())?;
    let parts: Vec<&str> = value.split('-').collect();
    if parts.len() != 4 || parts[0] != "00" {
        return None;
    }
    let (trace_id, parent_id, flags) = (parts[1], parts[2], parts[3]);
    if trace_id.len() != 32 || parent_id.len() != 16 || flags.len() != 2 {
        return None;
    }
    if !is_lower_hex(trace_id) || !is_lower_hex(parent_id) || !is_lower_hex(flags) {
        return None;
    }
    Some(trace_id.to_string())
}

/// CORS Origin 判定：列表含 `*` 时放行所有；否则仅精确匹配；空列表拒绝所有。
pub fn cors_allowed(config: &AuthConfig, origin: &str) -> bool {
    if config.cors_origins.iter().any(|o| o == "*") {
        return true;
    }
    config.cors_origins.iter().any(|o| o == origin)
}

/// OIDC 契约状态：未启用返回 `disabled`；启用返回 `not_implemented`。
///
/// 调用方必须把 `not_implemented` 映射为 HTTP 501 并拒绝请求，绝不能因
/// OIDC 未实现而绕过 API Key 鉴权（显式契约占位，非静默降级）。
pub fn oidc_status(config: &AuthConfig) -> &'static str {
    if config.oidc_enabled {
        "not_implemented"
    } else {
        "disabled"
    }
}
