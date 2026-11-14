//! 未来能力的适配器接口占位。
//!
//! V1 是纯文本生成推理引擎，本模块只定义接口形状，不提供任何伪实现：
//! [`NotImplementedAdapter`] 的所有可失败方法一律返回 `not_implemented`
//! 501 错误。严禁从 decoder logits 派生 embedding/rerank 之类的伪能力；
//! 真实实现由后续阶段以独立适配器接入，接入前契约保持 501。
//!
//! 与其他模块的关系：错误串格式与 `auth`/`qos` 的稳定英文标识约定一致，
//! HTTP 层据此映射 501。

/// 文本向量化适配器（embedding 能力）。
pub trait EmbeddingAdapter {
    /// 对一批文本生成向量；V1 阶段任何实现都必须返回 501 错误。
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String>;
}

/// 重排序适配器（rerank 能力）。
pub trait RerankAdapter {
    /// 按 query 对 docs 打相关性分；V1 阶段任何实现都必须返回 501 错误。
    fn rerank(&self, query: &str, docs: &[String]) -> Result<Vec<f32>, String>;
}

/// 多模态适配器。
pub trait MultimodalAdapter {
    /// 声明支持的多模态类型（如 `image`）；占位实现返回 `none`。
    fn modality(&self) -> &'static str;
}

/// 硬件加速适配器。
pub trait AcceleratorAdapter {
    /// 声明后端标识（如 `cuda`、`metal`）；占位实现返回 `none`。
    fn backend(&self) -> &'static str;
}

/// 模型注册/分发适配器（模型拉取、推送、注册表加载）。
pub trait RegistryAdapter {
    /// 按 id 加载模型；V1 阶段任何实现都必须返回 501 错误。
    fn load(&self, id: &str) -> Result<(), String>;
}

/// 全部能力均未实现的占位适配器。
///
/// 可失败方法返回 `not_implemented: <能力名> 501`；`modality`/`backend`
/// 返回 `none` 以如实暴露"无后端"，绝不冒充 `cpu` 之外的加速器。
#[derive(Debug, Default, Clone, Copy)]
pub struct NotImplementedAdapter;

impl EmbeddingAdapter for NotImplementedAdapter {
    fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        Err("not_implemented: embedding 501".to_string())
    }
}

impl RerankAdapter for NotImplementedAdapter {
    fn rerank(&self, _query: &str, _docs: &[String]) -> Result<Vec<f32>, String> {
        Err("not_implemented: rerank 501".to_string())
    }
}

impl MultimodalAdapter for NotImplementedAdapter {
    fn modality(&self) -> &'static str {
        "none"
    }
}

impl AcceleratorAdapter for NotImplementedAdapter {
    fn backend(&self) -> &'static str {
        "none"
    }
}

impl RegistryAdapter for NotImplementedAdapter {
    fn load(&self, _id: &str) -> Result<(), String> {
        Err("not_implemented: registry 501".to_string())
    }
}
