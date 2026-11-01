//! Tensor data structure

use super::DType;
use std::sync::Arc;

/// A multi-dimensional array
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Data buffer (shared via Arc for zero-copy)
    data: Arc<Vec<f32>>,
    /// Dimensions
    dims: Vec<usize>,
    /// Strides for each dimension
    strides: Vec<usize>,
    /// Data type
    dtype: DType,
}

impl Tensor {
    /// Create a new tensor from data and dimensions
    pub fn new(data: Vec<f32>, dims: Vec<usize>) -> Self {
        let strides = compute_strides(&dims);
        Self {
            data: Arc::new(data),
            dims,
            strides,
            dtype: DType::F32,
        }
    }

    /// Create a tensor filled with zeros
    pub fn zeros(dims: Vec<usize>) -> Self {
        let total_size: usize = dims.iter().product();
        let data = vec![0.0; total_size];
        Self::new(data, dims)
    }

    /// Create a tensor filled with ones
    pub fn ones(dims: Vec<usize>) -> Self {
        let total_size: usize = dims.iter().product();
        let data = vec![1.0; total_size];
        Self::new(data, dims)
    }

    /// Get tensor dimensions
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Get tensor strides
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Get number of dimensions
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Get total number of elements
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if tensor is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get data type
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get raw data as slice
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Get element at flattened index
    pub fn get_flat(&self, index: usize) -> Option<f32> {
        self.data.get(index).copied()
    }

    /// Set element at flattened index
    pub fn set_flat(&mut self, index: usize, value: f32) {
        if index < self.data.len() {
            let data = Arc::make_mut(&mut self.data);
            data[index] = value;
        }
    }

    /// Reshape tensor (total size must match)
    pub fn reshape(&self, new_dims: Vec<usize>) -> Result<Self, String> {
        let new_size: usize = new_dims.iter().product();
        if new_size != self.len() {
            return Err(format!(
                "Cannot reshape tensor of size {} to size {}",
                self.len(),
                new_size
            ));
        }

        let strides = compute_strides(&new_dims);
        Ok(Self {
            data: Arc::clone(&self.data),
            dims: new_dims,
            strides,
            dtype: self.dtype,
        })
    }

    /// Create a view of the tensor (shares data)
    pub fn view(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
            dims: self.dims.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype,
        }
    }
}

/// Compute strides for given dimensions (row-major)
fn compute_strides(dims: &[usize]) -> Vec<usize> {
    let ndim = dims.len();
    if ndim == 0 {
        return vec![];
    }

    let mut strides = vec![0; ndim];
    strides[ndim - 1] = 1;
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let tensor = Tensor::zeros(vec![2, 3]);
        assert_eq!(tensor.dims(), &[2, 3]);
        assert_eq!(tensor.len(), 6);
        assert_eq!(tensor.ndim(), 2);
    }

    #[test]
    fn test_tensor_strides() {
        let tensor = Tensor::zeros(vec![2, 3, 4]);
        assert_eq!(tensor.strides(), &[12, 4, 1]);
    }

    #[test]
    fn test_reshape() {
        let tensor = Tensor::ones(vec![2, 3]);
        let reshaped = tensor.reshape(vec![3, 2]).unwrap();
        assert_eq!(reshaped.dims(), &[3, 2]);
        assert_eq!(reshaped.len(), 6);
    }

    #[test]
    fn test_reshape_invalid() {
        let tensor = Tensor::ones(vec![2, 3]);
        assert!(tensor.reshape(vec![2, 4]).is_err());
    }
}
