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

    /// 按多维坐标读取元素，如 `get(&[1, 2, 0])`。
    ///
    /// 坐标个数须等于维数；任一坐标越界、或坐标数与维数不符时返回 `None`。
    /// 扁平索引由 `strides` 加权求和得到，读取不复制数据。
    pub fn get(&self, coords: &[usize]) -> Option<f32> {
        let flat = self.flat_index(coords).ok()?;
        self.data.get(flat).copied()
    }

    /// 按多维坐标写入元素，如 `set(&[1, 2, 0], 3.5)`。
    ///
    /// 坐标个数不符、坐标越界或扁平索引超出缓冲区时返回错误。写入经
    /// `Arc::make_mut` 触发写时复制：与其他视图共享 buffer 时先复制，
    /// 不影响那些视图。
    pub fn set(&mut self, coords: &[usize], value: f32) -> Result<(), String> {
        let flat = self.flat_index(coords)?;
        if flat >= self.data.len() {
            return Err(format!(
                "Flat index {flat} is out of bounds for a buffer of {} elements",
                self.data.len()
            ));
        }
        Arc::make_mut(&mut self.data)[flat] = value;
        Ok(())
    }

    /// 交换两个轴，返回**连续存储**的新张量（数据重排，不是视图）。
    ///
    /// 例如 `[seq, heads, dim]` 交换轴 0 与 1 得到 `[heads, seq, dim]`。
    /// 轴越界时返回错误。交换同一轴得到等价副本。
    ///
    /// 本阶段刻意采用数据复制而非非连续 strides 视图：调用方（Attention 的
    /// 多头路径）需要连续缓冲才能复用只接受二维的 `matmul`。非连续视图可避免
    /// 这些拷贝，留待性能阶段实现。
    pub fn transpose(&self, axis1: usize, axis2: usize) -> Result<Tensor, String> {
        let rank = self.ndim();
        if axis1 >= rank || axis2 >= rank {
            return Err(format!(
                "Transpose axes ({axis1}, {axis2}) are out of bounds for rank {rank}"
            ));
        }
        self.check_contiguous_shape("Transpose")?;

        let mut new_dims = self.dims.clone();
        new_dims.swap(axis1, axis2);
        let new_strides = compute_strides(&new_dims);

        let mut data = vec![0.0f32; self.len()];
        let mut coords = vec![0usize; rank];
        for _ in 0..self.len() {
            // coords 始终按旧轴顺序维护：先用旧 strides 定位源元素，
            // 交换两个轴后再用新 strides 定位目标位置。
            let source = coords_flat(&coords, &self.strides);
            coords.swap(axis1, axis2);
            let target: usize = coords.iter().zip(&new_strides).map(|(&c, &s)| c * s).sum();
            data[target] = self.data[source];
            coords.swap(axis1, axis2);
            advance_coords(&mut coords, &self.dims);
        }
        Ok(Self::new(data, new_dims))
    }

    /// 沿某一轴取 `[start, end)` 区间，返回连续存储的新张量（数据复制）。
    ///
    /// 轴越界、`start > end` 或 `end` 超出该轴长度时返回错误。
    /// 区间长度为 0 时返回该轴长度为 0 的空张量。
    pub fn slice(&self, axis: usize, start: usize, end: usize) -> Result<Tensor, String> {
        let rank = self.ndim();
        if axis >= rank {
            return Err(format!(
                "Slice axis {axis} is out of bounds for rank {rank}"
            ));
        }
        let dim = self.dims[axis];
        if start > end || end > dim {
            return Err(format!(
                "Slice range [{start}, {end}) is invalid for axis {axis} of length {dim}"
            ));
        }
        self.check_contiguous_shape("Slice")?;

        let mut new_dims = self.dims.clone();
        new_dims[axis] = end - start;

        let new_size: usize = new_dims.iter().product();
        let mut data = Vec::with_capacity(new_size);
        let mut coords = vec![0usize; rank];
        for _ in 0..new_size {
            // 新坐标与旧坐标仅 axis 上相差 start，其余轴的加权和可直接复用。
            let base = coords_flat(&coords, &self.strides);
            let source = base + start * self.strides[axis];
            data.push(self.data[source]);
            advance_coords(&mut coords, &new_dims);
        }
        Ok(Self::new(data, new_dims))
    }

    /// 校验维度乘积与缓冲区长度一致。
    ///
    /// `Tensor::new` 不校验这一约束，畸形张量的 strides 可能远超缓冲区；
    /// 重排类操作依赖坐标换算不越界，因此先在此拦一道。
    fn check_contiguous_shape(&self, operation: &str) -> Result<(), String> {
        let product: usize = self.dims.iter().product();
        if product != self.len() {
            return Err(format!(
                "{operation} requires the dimension product {product} to match the buffer length {}",
                self.len()
            ));
        }
        Ok(())
    }

    /// 把多维坐标换算成扁平索引。
    ///
    /// 坐标数与维数不符、任一坐标越界或加权求和溢出时返回错误。
    fn flat_index(&self, coords: &[usize]) -> Result<usize, String> {
        if coords.len() != self.dims.len() {
            return Err(format!(
                "Coordinate rank {} does not match tensor rank {}",
                coords.len(),
                self.dims.len()
            ));
        }
        let mut flat = 0usize;
        for (axis, (&coord, &dim)) in coords.iter().zip(self.dims.iter()).enumerate() {
            if coord >= dim {
                return Err(format!(
                    "Coordinate {coord} on axis {axis} is out of bounds for size {dim}"
                ));
            }
            let stride = self.strides[axis];
            flat = flat
                .checked_add(
                    coord.checked_mul(stride).ok_or_else(|| {
                        format!("Index overflow at axis {axis}: {coord} * {stride}")
                    })?,
                )
                .ok_or_else(|| format!("Index overflow while summing axis {axis}"))?;
        }
        Ok(flat)
    }
}

/// 按给定 strides 计算扁平索引；调用方须已保证坐标在界内。
fn coords_flat(coords: &[usize], strides: &[usize]) -> usize {
    coords.iter().zip(strides).map(|(&c, &s)| c * s).sum()
}

/// 按行主序把坐标推进到下一个位置，返回 false 表示已遍历完所有坐标。
fn advance_coords(coords: &mut [usize], dims: &[usize]) -> bool {
    for axis in (0..coords.len()).rev() {
        coords[axis] += 1;
        if coords[axis] < dims[axis] {
            return true;
        }
        coords[axis] = 0;
    }
    false
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

    /// 构造值等于扁平索引的 [2,3,4] 张量，便于按坐标推算期望值。
    fn indexed_2x3x4() -> Tensor {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        Tensor::new(data, vec![2, 3, 4])
    }

    #[test]
    fn get_reads_by_coordinates_in_2d_and_3d() {
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        assert_eq!(matrix.get(&[0, 0]), Some(1.0));
        assert_eq!(matrix.get(&[0, 2]), Some(3.0));
        assert_eq!(matrix.get(&[1, 0]), Some(4.0));
        assert_eq!(matrix.get(&[1, 2]), Some(6.0));

        let cube = indexed_2x3x4();
        // 扁平索引 = i*12 + j*4 + k，与坐标值一致。
        assert_eq!(cube.get(&[0, 0, 0]), Some(0.0));
        assert_eq!(cube.get(&[1, 2, 3]), Some(23.0));
        assert_eq!(cube.get(&[1, 0, 0]), Some(12.0));
    }

    #[test]
    fn get_rejects_out_of_range_and_wrong_rank() {
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(matrix.get(&[2, 0]), None);
        assert_eq!(matrix.get(&[0, 2]), None);
        assert_eq!(matrix.get(&[0]), None);
        assert_eq!(matrix.get(&[0, 0, 0]), None);
    }

    #[test]
    fn set_writes_by_coordinates_and_does_not_alias_shared_views() {
        let mut matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let shared = matrix.view();

        matrix.set(&[1, 1], 99.0).unwrap();
        assert_eq!(matrix.get(&[1, 1]), Some(99.0));
        // 写时复制：共享视图看不到这次写入。
        assert_eq!(shared.get(&[1, 1]), Some(5.0));

        let mut cube = indexed_2x3x4();
        cube.set(&[0, 1, 2], -1.0).unwrap();
        assert_eq!(cube.get(&[0, 1, 2]), Some(-1.0));
        assert_eq!(cube.get(&[0, 1, 3]), Some(7.0));
    }

    #[test]
    fn set_rejects_out_of_range_and_wrong_rank() {
        let mut matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert!(matrix.set(&[2, 0], 1.0).is_err());
        assert!(matrix.set(&[0], 1.0).is_err());
    }

    #[test]
    fn transpose_swaps_axes_of_a_matrix() {
        // [[1,2,3],[4,5,6]] 转置后为 [[1,4],[2,5],[3,6]]。
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let transposed = matrix.transpose(0, 1).unwrap();
        assert_eq!(transposed.dims(), &[3, 2]);
        assert_eq!(transposed.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(transposed.get(&[2, 1]), Some(6.0));
    }

    #[test]
    fn transpose_swaps_outer_and_inner_axes_of_a_cube() {
        let cube = indexed_2x3x4();
        let transposed = cube.transpose(0, 2).unwrap();
        assert_eq!(transposed.dims(), &[4, 3, 2]);
        // 新坐标 (a,b,c) 对应旧坐标 (c,b,a)。
        assert_eq!(transposed.get(&[0, 0, 1]), Some(12.0));
        assert_eq!(transposed.get(&[1, 0, 0]), Some(1.0));
        assert_eq!(transposed.get(&[3, 2, 1]), Some(23.0));
        // 逐点核对全部 24 个元素。
        for a in 0..4 {
            for b in 0..3 {
                for c in 0..2 {
                    assert_eq!(
                        transposed.get(&[a, b, c]),
                        Some((c * 12 + b * 4 + a) as f32),
                        "mismatch at ({a},{b},{c})"
                    );
                }
            }
        }
    }

    #[test]
    fn transpose_is_an_involution_and_same_axis_is_a_copy() {
        let cube = indexed_2x3x4();
        let back = cube.transpose(0, 2).unwrap().transpose(0, 2).unwrap();
        assert_eq!(back.dims(), cube.dims());
        assert_eq!(back.data(), cube.data());

        let same = cube.transpose(1, 1).unwrap();
        assert_eq!(same.data(), cube.data());
    }

    #[test]
    fn slice_takes_a_sub_range_of_a_vector_and_matrix() {
        let vector = Tensor::new((10..=50).step_by(10).map(|v| v as f32).collect(), vec![5]);
        let sliced = vector.slice(0, 1, 4).unwrap();
        assert_eq!(sliced.dims(), &[3]);
        assert_eq!(sliced.data(), &[20.0, 30.0, 40.0]);

        let matrix = Tensor::new((0..20).map(|i| i as f32).collect(), vec![4, 5]);
        let rows = matrix.slice(0, 0, 2).unwrap();
        assert_eq!(rows.dims(), &[2, 5]);
        assert_eq!(rows.data(), &(0..10).map(|i| i as f32).collect::<Vec<_>>());

        let columns = matrix.slice(1, 1, 3).unwrap();
        assert_eq!(columns.dims(), &[4, 2]);
        assert_eq!(
            columns.data(),
            &[1.0, 2.0, 6.0, 7.0, 11.0, 12.0, 16.0, 17.0]
        );
    }

    #[test]
    fn slice_supports_empty_ranges_and_full_ranges() {
        let vector = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let empty = vector.slice(0, 1, 1).unwrap();
        assert_eq!(empty.dims(), &[0]);
        assert!(empty.is_empty());

        let whole = vector.slice(0, 0, 3).unwrap();
        assert_eq!(whole.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn reordering_rejects_invalid_axes_and_ranges() {
        let matrix = Tensor::new((0..6).map(|i| i as f32).collect(), vec![2, 3]);
        assert!(matrix.transpose(0, 2).is_err());
        assert!(matrix.transpose(2, 0).is_err());
        assert!(matrix.slice(2, 0, 1).is_err());
        assert!(matrix.slice(0, 3, 1).is_err());
        assert!(matrix.slice(0, 0, 4).is_err());
    }

    #[test]
    fn reordering_rejects_tensors_whose_shape_disagrees_with_the_buffer() {
        // Tensor::new 不校验维度乘积，此处确认重排操作会拦住这种畸形输入。
        let mut malformed = Tensor::new(vec![1.0, 2.0], vec![2, 2]);
        assert!(malformed.transpose(0, 1).is_err());
        assert!(malformed.slice(0, 0, 1).is_err());
        assert_eq!(malformed.get(&[0, 0]), Some(1.0));
        assert!(malformed.set(&[1, 1], 0.0).is_err());
    }
}
