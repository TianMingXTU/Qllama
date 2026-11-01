//! Tensor operations and data structures

mod ops;
mod tensor;

pub use ops::TensorOps;
pub use tensor::Tensor;

/// Data type for tensor elements
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    I32,
    I64,
}
