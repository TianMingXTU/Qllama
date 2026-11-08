//! GGUF v3：索引解析、权重加载与映射存储。

mod loader;
mod parser;
mod types;
mod weight;

pub use parser::GgufFile;
pub use types::{GgmlType, GgufHeader, GgufValue, ModelMetadata, TensorInfo};
pub use weight::MappedWeight;

// 供 transformer::config 复用同一套 metadata 读取语义，避免两处对键类型的判断漂移。
pub(crate) use types::{read_dimension, read_optional_dimension, read_positive_f32};
