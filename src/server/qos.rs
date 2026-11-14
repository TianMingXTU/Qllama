//! 租户级服务质量控制：在途并发、固定 60 秒窗口的请求数与 token 数限流。
//!
//! `QosManager` 按 key 维度维护配额状态，`acquire` 成功后返回 `QosPermit`；
//! 调用方（HTTP 层）在请求结束（含错误路径）时 MUST 调用
//! [`QosPermit::release`] 归还在途计数——`QosPermit` 的 `Drop` 无法访问
//! 管理器，不会自动归还，遗漏会导致该 key 的并发额度永久泄漏。
//!
//! 错误串为稳定英文标识（`in_flight` / `request_rate` / `token_quota`），
//! HTTP 层按其映射 429 与 `Retry-After` 头。
//!
//! 与其他模块的关系：`key_id` 来自 `auth::AuthContext`；拒绝原因可被
//! `metrics::Metrics` 记入 `qllama_requests_rejected_total{reason}`。

use std::cell::Cell;
use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, MutexGuard};

/// 速率窗口长度（毫秒）。
const WINDOW_MS: u64 = 60_000;

/// 一份配额定义；`Default` 为保守起步值：4 并发 / 60 请求每分 / 8192 token 每分。
#[derive(Clone, Copy, Debug)]
pub struct Quota {
    /// 该 key 允许的最大同时在途请求数。
    pub max_in_flight: usize,
    /// 60 秒窗口内允许的最大请求数。
    pub requests_per_minute: usize,
    /// 60 秒窗口内允许的最大预估 token 数。
    pub tokens_per_minute: usize,
}

impl Default for Quota {
    fn default() -> Self {
        Self {
            max_in_flight: 4,
            requests_per_minute: 60,
            tokens_per_minute: 8192,
        }
    }
}

/// 单个 key 的运行状态；仅在 [`QosManager`] 的锁内读写。
#[derive(Debug, Default)]
struct TenantState {
    /// 当前在途请求数（由 `acquire`/`release` 维护）。
    in_flight: usize,
    /// 最近一次 `acquire` 的时间戳，用于检测时钟回拨。
    last_ms: Option<u64>,
    /// 窗口内的 `(时间戳毫秒, 预估 token)` 记录。
    window: VecDeque<(u64, usize)>,
}

/// 一份在途请求的许可。调用方 MUST 在请求结束时调用 [`QosPermit::release`]；
/// `Drop` 不做任何归还（许可无法访问管理器），双重 `release` 安全（第二次为空操作）。
#[derive(Debug)]
pub struct QosPermit {
    /// 许可归属的 key id。
    key_id: String,
    /// 本次预估 token 数（信息记录用）。
    tokens: usize,
    /// 是否仍持有在途额度；用 `Cell` 让 `release(&self)` 也能翻转。
    inflight: Cell<bool>,
}

/// key 级配额管理器。锁内状态仅由本模块更新、临界区内无 panic 源。
pub struct QosManager {
    /// 未命中 per-key 覆盖时使用的默认配额。
    defaults: Quota,
    /// 按 key 覆盖的配额。
    per_key: HashMap<String, Quota>,
    /// 各 key 的运行状态。
    state: Mutex<HashMap<String, TenantState>>,
}

/// 锁毒化恢复：状态只由本模块更新且临界区内无 panic 源，
/// 直接复用内部数据比让后续所有请求 panic 更安全。
fn lock_state(
    map: &Mutex<HashMap<String, TenantState>>,
) -> MutexGuard<'_, HashMap<String, TenantState>> {
    map.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl QosManager {
    /// 创建管理器；`per_key` 中的条目优先于 `defaults`。
    pub fn new(defaults: Quota, per_key: HashMap<String, Quota>) -> Self {
        Self {
            defaults,
            per_key,
            state: Mutex::new(HashMap::new()),
        }
    }

    /// 解析 key 的生效配额：per-key 覆盖优先，否则用默认值。
    fn quota_for(&self, key_id: &str) -> Quota {
        self.per_key.get(key_id).copied().unwrap_or(self.defaults)
    }

    /// 申请一次请求额度；成功时占用在途计数并把 `(now_ms, estimated_tokens)`
    /// 记入窗口。失败返回稳定错误串：
    /// `in_flight`（并发满）、`request_rate`（窗口请求数满）、
    /// `token_quota`（窗口 token 数满）。
    ///
    /// 时钟回拨处理：`now_ms` 小于该 key 上次时间戳时直接清空窗口重计，
    /// 不 panic、不产生负值。
    pub fn acquire(
        &self,
        key_id: &str,
        estimated_tokens: usize,
        now_ms: u64,
    ) -> Result<QosPermit, String> {
        let quota = self.quota_for(key_id);
        let mut map = lock_state(&self.state);
        let state = map.entry(key_id.to_string()).or_default();
        if let Some(last) = state.last_ms {
            if now_ms < last {
                state.window.clear();
            }
        }
        state.last_ms = Some(now_ms);
        while let Some(&(ts, _)) = state.window.front() {
            if ts + WINDOW_MS <= now_ms {
                state.window.pop_front();
            } else {
                break;
            }
        }
        if state.in_flight >= quota.max_in_flight {
            return Err("in_flight".to_string());
        }
        if state.window.len() >= quota.requests_per_minute {
            return Err("request_rate".to_string());
        }
        let used: usize = state.window.iter().map(|&(_, tokens)| tokens).sum();
        if used.saturating_add(estimated_tokens) > quota.tokens_per_minute {
            return Err("token_quota".to_string());
        }
        state.in_flight += 1;
        state.window.push_back((now_ms, estimated_tokens));
        Ok(QosPermit {
            key_id: key_id.to_string(),
            tokens: estimated_tokens,
            inflight: Cell::new(true),
        })
    }

    /// 输出某个 key 的配额状态快照（控制台/运维查询用，不含任何业务体）。
    pub fn snapshot(&self, key_id: &str) -> serde_json::Value {
        let quota = self.quota_for(key_id);
        let map = lock_state(&self.state);
        let (in_flight, window_requests, window_tokens) = match map.get(key_id) {
            Some(state) => (
                state.in_flight,
                state.window.len(),
                state.window.iter().map(|&(_, tokens)| tokens).sum(),
            ),
            None => (0, 0, 0),
        };
        serde_json::json!({
            "key_id": key_id,
            "quota": {
                "max_in_flight": quota.max_in_flight,
                "requests_per_minute": quota.requests_per_minute,
                "tokens_per_minute": quota.tokens_per_minute,
            },
            "in_flight": in_flight,
            "window_requests": window_requests,
            "window_tokens": window_tokens,
        })
    }
}

impl QosPermit {
    /// 返回本次请求占用的预估 token 数，供请求元数据与审计核对配额决策。
    pub fn estimated_tokens(&self) -> usize {
        self.tokens
    }

    /// 归还在途额度；`status` 为本次请求的 HTTP 状态码，参数预留用于
    /// 按状态码记账（当前仅递减在途计数）。重复调用安全。
    pub fn release(&self, manager: &QosManager, status: u16) {
        let _ = status;
        if !self.inflight.replace(false) {
            return;
        }
        let mut map = lock_state(&manager.state);
        if let Some(state) = map.get_mut(&self.key_id) {
            state.in_flight = state.in_flight.saturating_sub(1);
        }
    }
}
