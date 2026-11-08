//! f32 张量与其逐算子实现：`data` 存数据与形状，`ops` 提供运算。

mod data;
mod ops;

pub use data::Tensor;
pub use ops::TensorOps;
