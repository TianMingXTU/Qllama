//! 本地 GGUF 模型注册表。
//!
//! 目录扫描仍受 canonicalize + starts_with 边界限制；digest 使用 FNV-1a 64
//! 稳定摘要，用于版本钉死与 `/v1/models` 能力发现。注册表只索引模型文件，
//! 不发起网络拉取，也不把未知模型静默回退到默认模型。

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::gguf::GgufFile;

/// 单个已注册模型的脱敏元数据。
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ModelEntry {
    pub id: String,
    pub path: PathBuf,
    pub digest: String,
    pub size: u64,
    pub architecture: String,
    pub quantization: String,
    pub context_length: usize,
    pub chat_template: Option<String>,
    pub loaded: bool,
}

/// 本地模型目录注册表；`base_dir` 是 canonicalize 后的安全根。
#[derive(Debug)]
pub struct ModelRegistry {
    base_dir: PathBuf,
    entries: HashMap<String, ModelEntry>,
}

impl ModelRegistry {
    /// 扫描 `base_dir` 下 `.gguf` 文件；不存在目录返回空注册表。
    pub fn new(base_dir: &Path) -> Result<Self, String> {
        let base_dir = base_dir
            .canonicalize()
            .map_err(|error| format!("cannot canonicalize model directory: {error}"))?;
        let mut registry = Self {
            base_dir,
            entries: HashMap::new(),
        };
        registry.refresh()?;
        Ok(registry)
    }

    /// 重新扫描目录并记录新模型；卸载状态在已有 entry 上保留。
    pub fn refresh(&mut self) -> Result<(), String> {
        for entry in std::fs::read_dir(&self.base_dir)
            .map_err(|error| format!("cannot read model directory: {error}"))?
        {
            let entry = entry.map_err(|error| format!("cannot read model entry: {error}"))?;
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("gguf") {
                continue;
            }
            let canonical = path
                .canonicalize()
                .map_err(|error| format!("cannot canonicalize model path: {error}"))?;
            if !canonical.starts_with(&self.base_dir) {
                return Err("model_path_outside_registry".to_string());
            }
            let id = canonical
                .file_stem()
                .and_then(|stem| stem.to_str())
                .ok_or_else(|| "invalid_model_filename".to_string())?
                .to_string();
            let loaded = self.entries.get(&id).is_some_and(|entry| entry.loaded);
            let metadata = std::fs::metadata(&canonical)
                .map_err(|error| format!("cannot stat model: {error}"))?;
            let digest = compute_file_digest(&canonical)?;
            let (architecture, quantization, context_length, chat_template) =
                match GgufFile::from_path(&canonical) {
                    Ok(gguf) => (
                        metadata_string(&gguf, "general.architecture")
                            .unwrap_or_else(|| "llama".to_string()),
                        gguf.tensor_infos
                            .first()
                            .map(|tensor| format!("{:?}", tensor.ggml_type))
                            .unwrap_or_else(|| "unknown".to_string()),
                        metadata_usize(&gguf, "llama.context_length").unwrap_or(0),
                        metadata_string(&gguf, "tokenizer.chat_template"),
                    ),
                    Err(_) => ("unknown".to_string(), "unknown".to_string(), 0, None),
                };
            self.entries.insert(
                id.clone(),
                ModelEntry {
                    id,
                    path: canonical,
                    digest,
                    size: metadata.len(),
                    architecture,
                    quantization,
                    context_length,
                    chat_template,
                    loaded,
                },
            );
        }
        Ok(())
    }

    /// 列出当前索引中的模型，按 id 稳定排序。
    pub fn list(&self) -> Vec<ModelEntry> {
        let mut entries: Vec<_> = self.entries.values().cloned().collect();
        entries.sort_by(|left, right| left.id.cmp(&right.id));
        entries
    }

    /// 按 id 或 `id@digest` 精确路由；不匹配返回明确错误。
    pub fn resolve(&self, requested: &str) -> Result<ModelEntry, String> {
        let (id, pinned_digest) = requested
            .split_once('@')
            .map_or((requested, None), |(id, digest)| (id, Some(digest)));
        let entry = self
            .entries
            .get(id)
            .ok_or_else(|| "unknown_model".to_string())?;
        if let Some(expected) = pinned_digest {
            if entry.digest != expected {
                return Err("model_digest_mismatch".to_string());
            }
        }
        Ok(entry.clone())
    }

    /// 标记模型已加载；路径必须位于注册表根内。
    pub fn mark_loaded(&mut self, id: &str) -> Result<(), String> {
        let path = self
            .resolve(id)
            .map(|entry| entry.path)
            .map_err(|_| "unknown_model".to_string())?;
        if !path.starts_with(&self.base_dir) {
            return Err("model_path_outside_registry".to_string());
        }
        if let Some(entry) = self.entries.get_mut(id) {
            entry.loaded = true;
        }
        Ok(())
    }

    /// 标记模型卸载；不删除磁盘文件。
    pub fn unload(&mut self, id: &str) -> Result<(), String> {
        let entry = self
            .entries
            .get_mut(id)
            .ok_or_else(|| "unknown_model".to_string())?;
        entry.loaded = false;
        Ok(())
    }

    /// 当前已加载模型数量。
    pub fn loaded_count(&self) -> usize {
        self.entries.values().filter(|entry| entry.loaded).count()
    }
}

fn metadata_string(gguf: &GgufFile, key: &str) -> Option<String> {
    match gguf.metadata.get(key) {
        Some(crate::gguf::GgufValue::String(value)) => Some(value.clone()),
        _ => None,
    }
}

fn metadata_usize(gguf: &GgufFile, key: &str) -> Option<usize> {
    match gguf.metadata.get(key) {
        Some(crate::gguf::GgufValue::Uint32(value)) => usize::try_from(*value).ok(),
        Some(crate::gguf::GgufValue::Uint64(value)) => usize::try_from(*value).ok(),
        _ => None,
    }
}

/// 以 1 MiB 流式读取计算文件 FNV-1a 64 摘要；超过 256 MiB 返回明确错误。
pub fn compute_file_digest(path: &Path) -> Result<String, String> {
    const LIMIT: u64 = 256 * 1024 * 1024;
    let mut file = File::open(path).map_err(|error| format!("cannot open model: {error}"))?;
    let size = file
        .metadata()
        .map_err(|error| format!("cannot stat model: {error}"))?
        .len();
    if size > LIMIT {
        return Err("model_too_large_for_digest".to_string());
    }
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    // Windows 主线程默认栈只有 1 MiB；1 MiB 的栈数组会让服务在启动时
    // 直接栈溢出。缓冲区放堆上，栈占用与文件大小无关。
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("cannot read model: {error}"))?;
        if read == 0 {
            break;
        }
        for byte in &buffer[..read] {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
    Ok(format!("fnv1a64:{hash:016x}"))
}

#[cfg(test)]
mod tests {
    use super::compute_file_digest;

    #[test]
    fn digest_is_stable_and_prefixed() {
        let mut path = std::env::temp_dir();
        path.push(format!("qllama-digest-{}.bin", std::process::id()));
        std::fs::write(&path, b"abc").unwrap();
        let digest = compute_file_digest(&path).unwrap();
        assert!(digest.starts_with("fnv1a64:"));
        assert_eq!(digest, compute_file_digest(&path).unwrap());
        std::fs::remove_file(path).unwrap();
    }
}
