//! GGUF format parser
//!
//! This module provides functionality to parse GGUF files and extract
//! model metadata and tensor information.

mod parser;
mod types;

pub use parser::GgufFile;
pub use types::{GgmlType, GgufHeader, GgufValue, ModelMetadata, TensorInfo};

mod loader;
