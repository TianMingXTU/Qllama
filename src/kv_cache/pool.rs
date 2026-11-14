//! 可分块 KV 池与跨请求前缀缓存账本。
//!
//! 现有 Transformer 前向要求每请求拥有连续 `KvCache`。本模块按固定 block size
//! 把容量对齐、以字节预算记账，并在 lease 释放时归还；前缀缓存只记录 token
//! 序列摘要和复用次数，不跨请求复用可写 KV 数据，避免破坏 seed/digest 可复现性。

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use crate::kv_cache::KvCache;
use crate::transformer::TransformerConfig;

/// 将 token 容量向上对齐到 block size。
pub fn block_aligned_capacity(tokens: usize, block_size: usize) -> usize {
    let block_size = block_size.max(1);
    tokens.div_ceil(block_size) * block_size
}

/// 稳定 token 前缀摘要；使用 FNV-1a 64，不依赖进程随机状态。
pub fn hash_token_prefix(tokens: &[i32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for token in tokens {
        for byte in token.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
    hash
}

/// 前缀缓存键：模型 digest + token 序列 hash。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PrefixKey {
    pub model_digest: String,
    pub token_hash: u64,
}

/// KV 池运行统计；用于 `/metrics` 与诊断端点。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct PoolStats {
    pub allocated_bytes: usize,
    pub max_bytes: usize,
    pub active_leases: usize,
    pub prefix_hits: u64,
    pub prefix_misses: u64,
}

#[derive(Debug)]
struct PoolInner {
    allocated_bytes: usize,
    active_leases: usize,
    prefix_enabled: bool,
    prefix_hits: u64,
    prefix_misses: u64,
    known_prefixes: HashSet<PrefixKey>,
}

/// 固定 block size、引用记账的 KV 容量池。
#[derive(Debug, Clone)]
pub struct KvPool {
    config: TransformerConfig,
    block_size: usize,
    bytes_per_token: usize,
    max_bytes: usize,
    inner: Arc<Mutex<PoolInner>>,
}

impl KvPool {
    /// 按模型配置创建池；超过预算返回 `kv_pool_exhausted`。
    pub fn new(config: &TransformerConfig, block_size: usize, max_bytes: usize) -> Self {
        let head_dim = config.head_dim();
        let bytes_per_token =
            config.num_layers * config.num_kv_heads * head_dim * 2 * std::mem::size_of::<f32>();
        Self {
            config: config.clone(),
            block_size: block_size.max(1),
            bytes_per_token,
            max_bytes,
            inner: Arc::new(Mutex::new(PoolInner {
                allocated_bytes: 0,
                active_leases: 0,
                prefix_enabled: false,
                prefix_hits: 0,
                prefix_misses: 0,
                known_prefixes: HashSet::new(),
            })),
        }
    }

    /// 启用/关闭跨请求前缀缓存账本；默认关闭。
    pub fn enable_prefix_cache(&mut self, enabled: bool) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.prefix_enabled = enabled;
            inner.known_prefixes.clear();
        }
    }

    /// 分配一个对齐后的 KV lease；`Drop` 时归还字节预算。
    pub fn acquire(&self, capacity_tokens: usize) -> Result<KvLease, String> {
        let capacity = block_aligned_capacity(capacity_tokens, self.block_size);
        let bytes = capacity
            .checked_mul(self.bytes_per_token)
            .ok_or_else(|| "kv_pool_size_overflow".to_string())?;
        {
            let mut inner = self
                .inner
                .lock()
                .map_err(|_| "kv_pool_lock_poisoned".to_string())?;
            if inner.allocated_bytes.saturating_add(bytes) > self.max_bytes {
                inner.prefix_misses = inner.prefix_misses.saturating_add(1);
                return Err("kv_pool_exhausted".to_string());
            }
            inner.allocated_bytes += bytes;
            inner.active_leases += 1;
        }
        let cache = std::panic::catch_unwind(|| {
            KvCache::new(
                self.config.num_layers,
                capacity,
                self.config.num_kv_heads,
                self.config.head_dim(),
            )
        });
        let cache = match cache {
            Ok(cache) => cache,
            Err(_) => {
                self.release(bytes);
                return Err("kv_pool_allocation_failed".to_string());
            }
        };
        Ok(KvLease {
            cache,
            bytes,
            inner: Arc::clone(&self.inner),
        })
    }

    /// 查询前缀缓存是否命中；关闭状态下始终 miss，且不改变可写 KV。
    pub fn prefix_hit(&self, key: &PrefixKey) -> bool {
        let Ok(mut inner) = self.inner.lock() else {
            return false;
        };
        let hit = inner.prefix_enabled && inner.known_prefixes.contains(key);
        if hit {
            inner.prefix_hits = inner.prefix_hits.saturating_add(1);
        } else {
            inner.prefix_misses = inner.prefix_misses.saturating_add(1);
        }
        hit
    }

    /// 显式登记命中/未命中；供 API 层把观测同步到 Prometheus。
    pub fn record_prefix_hit(&self, hit: bool, key: Option<PrefixKey>) {
        if let Ok(mut inner) = self.inner.lock() {
            if hit {
                inner.prefix_hits = inner.prefix_hits.saturating_add(1);
            } else {
                inner.prefix_misses = inner.prefix_misses.saturating_add(1);
            }
            if let Some(key) = key {
                inner.known_prefixes.insert(key);
            }
        }
    }

    /// 清空前缀账本，不改变 lease 与字节预算。
    pub fn clear_prefix(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.known_prefixes.clear();
        }
    }

    /// 返回当前池统计。
    pub fn stats(&self) -> PoolStats {
        match self.inner.lock() {
            Ok(inner) => PoolStats {
                allocated_bytes: inner.allocated_bytes,
                max_bytes: self.max_bytes,
                active_leases: inner.active_leases,
                prefix_hits: inner.prefix_hits,
                prefix_misses: inner.prefix_misses,
            },
            Err(_) => PoolStats {
                max_bytes: self.max_bytes,
                ..PoolStats::default()
            },
        }
    }

    fn release(&self, bytes: usize) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.allocated_bytes = inner.allocated_bytes.saturating_sub(bytes);
            inner.active_leases = inner.active_leases.saturating_sub(1);
        }
    }
}

/// 单个请求的 KV 所有权；实现 `DerefMut<Target=KvCache>` 供现有前直接使用。
pub struct KvLease {
    cache: KvCache,
    bytes: usize,
    inner: Arc<Mutex<PoolInner>>,
}

impl std::ops::Deref for KvLease {
    type Target = KvCache;

    fn deref(&self) -> &Self::Target {
        &self.cache
    }
}

impl std::ops::DerefMut for KvLease {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.cache
    }
}

impl Drop for KvLease {
    fn drop(&mut self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.allocated_bytes = inner.allocated_bytes.saturating_sub(self.bytes);
            inner.active_leases = inner.active_leases.saturating_sub(1);
        }
    }
}

/// 便于未来把账本按模型/租户拆分的占位类型；本批不跨请求复用可写 KV。
pub type PrefixLedger = HashMap<PrefixKey, u64>;

#[cfg(test)]
mod tests {
    use super::{block_aligned_capacity, hash_token_prefix, PoolStats};

    #[test]
    fn capacity_aligns_and_hash_is_stable() {
        assert_eq!(block_aligned_capacity(1, 32), 32);
        assert_eq!(block_aligned_capacity(32, 32), 32);
        assert_eq!(block_aligned_capacity(33, 32), 64);
        assert_eq!(hash_token_prefix(&[1, 2, 3]), hash_token_prefix(&[1, 2, 3]));
        assert_ne!(hash_token_prefix(&[1, 2, 3]), hash_token_prefix(&[3, 2, 1]));
        assert_eq!(PoolStats::default().prefix_hits, 0);
    }
}
