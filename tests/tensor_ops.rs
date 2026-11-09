//! 用手工常量和独立 f64 参考值检验张量算子的数值及形状契约。

use qllama::tensor::{Tensor, TensorOps};

fn tensor(data: &[f32], dims: &[usize]) -> Tensor {
    Tensor::new(data.to_vec(), dims.to_vec())
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
            "element {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}

fn assert_tensor(actual: &Tensor, dims: &[usize], expected: &[f32]) {
    assert_eq!(actual.dims(), dims);
    assert_close(actual.data(), expected, 1e-6);
}

#[test]
fn matmul_rectangular_hand_calculation() {
    let left = tensor(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    let right = tensor(&[7., 8., 9., 10., 11., 12.], &[3, 2]);
    assert_tensor(
        &left.matmul(&right).unwrap(),
        &[2, 2],
        &[58., 64., 139., 154.],
    );
}

#[test]
fn matmul_identity_preserves_values() {
    let input = tensor(&[-2., 0., 3., 4., -5., 6.], &[2, 3]);
    let identity = tensor(&[1., 0., 0., 0., 1., 0., 0., 0., 1.], &[3, 3]);
    assert_tensor(&input.matmul(&identity).unwrap(), &[2, 3], input.data());
}

#[test]
fn matmul_singleton() {
    assert_tensor(
        &tensor(&[-3.], &[1, 1])
            .matmul(&tensor(&[4.], &[1, 1]))
            .unwrap(),
        &[1, 1],
        &[-12.],
    );
}

#[test]
fn matmul_large_matches_independent_f64_reference() {
    let (rows, shared, columns) = (96, 80, 64);
    let left: Vec<f32> = (0..rows * shared)
        .map(|i| ((i * 17 % 101) as f32 - 50.) / 13.)
        .collect();
    let right: Vec<f32> = (0..shared * columns)
        .map(|i| ((i * 29 % 97) as f32 - 48.) / 11.)
        .collect();
    let mut expected = vec![0.; rows * columns];
    // 参考实现采用 i-j-k 和 f64，与生产实现的循环顺序及精度独立。
    for row in 0..rows {
        for column in 0..columns {
            let mut sum = 0.0_f64;
            for inner in 0..shared {
                sum += f64::from(left[row * shared + inner])
                    * f64::from(right[inner * columns + column]);
            }
            expected[row * columns + column] = sum as f32;
        }
    }
    let actual = Tensor::new(left, vec![rows, shared])
        .matmul(&Tensor::new(right, vec![shared, columns]))
        .unwrap();
    assert_eq!(actual.dims(), &[rows, columns]);
    assert_close(actual.data(), &expected, 3e-5);
}

#[test]
fn matmul_accumulates_sequentially_in_f32() {
    let left = tensor(&[16_777_216., 1., -16_777_216.], &[1, 3]);
    let right = tensor(&[1., 1., 1.], &[3, 1]);
    assert_tensor(&left.matmul(&right).unwrap(), &[1, 1], &[0.]);
}

#[test]
fn single_row_matmul_preserves_deterministic_f32_results() {
    // 当前实现使用 par_chunks_mut(columns)，当 rows==1 时只产生一个 chunk，
    // 退化为单线程 i-k-j 顺序累加；若未来改为列分块并行，此测试保证
    // 任意线程数下结果仍与原累积顺序逐位一致。
    let shared = 257;
    for columns in [63, 64, 65, 129, 513] {
        let input: Vec<f32> = (0..shared)
            .map(|i| ((i * 17 % 101) as f32 - 50.0) / 13.0)
            .collect();
        let weights: Vec<f32> = (0..shared * columns)
            .map(|i| ((i * 29 % 97) as f32 - 48.0) / 11.0)
            .collect();
        let expected: Vec<u32> = (0..columns)
            .map(|column| {
                let mut sum = 0.0f32;
                for inner in 0..shared {
                    sum += input[inner] * weights[inner * columns + column];
                }
                sum.to_bits()
            })
            .collect();
        let input = Tensor::new(input, vec![1, shared]);
        let weights = Tensor::new(weights, vec![shared, columns]);
        for threads in [1, 2, 4, 8] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            let actual = pool.install(|| input.matmul(&weights)).unwrap();
            assert_eq!(actual.dims(), &[1, columns]);
            assert_eq!(
                actual.data().iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                expected,
                "columns={columns}, threads={threads}"
            );
        }
    }
}

#[test]
fn matmul_rejects_inner_dimension_mismatch() {
    assert!(Tensor::ones(vec![2, 3])
        .matmul(&Tensor::ones(vec![2, 2]))
        .is_err());
}

#[test]
fn matmul_requires_two_dimensions_for_both_operands() {
    let matrix = Tensor::ones(vec![2, 2]);
    for invalid in [
        Tensor::ones(vec![2]),
        Tensor::ones(vec![1, 2, 2]),
        tensor(&[1.], &[]),
    ] {
        assert!(invalid.matmul(&matrix).is_err());
        assert!(matrix.matmul(&invalid).is_err());
    }
}

#[test]
fn matmul_zero_rows() {
    assert_tensor(
        &tensor(&[], &[0, 3])
            .matmul(&Tensor::ones(vec![3, 2]))
            .unwrap(),
        &[0, 2],
        &[],
    );
}

#[test]
fn matmul_zero_columns() {
    assert_tensor(
        &Tensor::ones(vec![2, 3])
            .matmul(&tensor(&[], &[3, 0]))
            .unwrap(),
        &[2, 0],
        &[],
    );
}

#[test]
fn matmul_zero_contraction_returns_zeros() {
    assert_tensor(
        &tensor(&[], &[2, 0]).matmul(&tensor(&[], &[0, 3])).unwrap(),
        &[2, 3],
        &[0.; 6],
    );
}

#[test]
fn matmul_rejects_overflowing_output_size() {
    let left = tensor(&[], &[usize::MAX, 0]);
    assert!(left.matmul(&tensor(&[], &[0, 2])).is_err());
}

#[test]
fn matmul_rejects_unrepresentable_buffer_capacity() {
    let rows = isize::MAX as usize / std::mem::size_of::<f32>() + 1;
    assert!(tensor(&[], &[rows, 0])
        .matmul(&tensor(&[], &[0, 1]))
        .is_err());
}

#[test]
fn matmul_nonfinite_values_follow_ieee_arithmetic() {
    let result = tensor(&[f32::INFINITY], &[1, 1])
        .matmul(&tensor(&[0.], &[1, 1]))
        .unwrap();
    assert!(result.data()[0].is_nan());
}

#[test]
fn rms_norm_hand_calculation_with_weights_and_epsilon() {
    let input = tensor(&[3., 4.], &[1, 2]);
    let weight = tensor(&[2., 3.], &[2]);
    // mean(x²) + eps = 12.5 + 3.5 = 16。
    assert_tensor(&input.rms_norm(&weight, 3.5).unwrap(), &[1, 2], &[1.5, 3.]);
}

#[test]
fn rms_norm_normalizes_each_last_dimension_independently() {
    let input = tensor(&[3., 4., 0., 2., -3., -4., 2., 0.], &[2, 2, 2]);
    let result = input.rms_norm(&tensor(&[1., 1.], &[2]), 0.).unwrap();
    assert_tensor(
        &result,
        &[2, 2, 2],
        &[
            0.84852815,
            1.1313709,
            0.,
            std::f32::consts::SQRT_2,
            -0.84852815,
            -1.1313709,
            std::f32::consts::SQRT_2,
            0.,
        ],
    );
}

#[test]
fn rms_norm_accepts_single_weight_for_all_features() {
    assert_tensor(
        &tensor(&[3., 4.], &[2])
            .rms_norm(&tensor(&[2.], &[]), 3.5)
            .unwrap(),
        &[2],
        &[1.5, 2.],
    );
}

#[test]
fn rms_norm_weight_contract_uses_length() {
    assert_tensor(
        &tensor(&[2.; 4], &[4])
            .rms_norm(&tensor(&[1., 2., 3., 4.], &[2, 2]), 0.)
            .unwrap(),
        &[4],
        &[1., 2., 3., 4.],
    );
}

#[test]
fn rms_norm_zero_vectors_remain_finite_with_zero_epsilon() {
    let input = Tensor::zeros(vec![2, 3]);
    for eps in [0., 1e-5] {
        assert_tensor(
            &input.rms_norm(&tensor(&[1.], &[1]), eps).unwrap(),
            &[2, 3],
            &[0.; 6],
        );
    }
}

#[test]
fn rms_norm_avoids_square_overflow_for_extreme_finite_inputs() {
    let input = tensor(&[f32::MAX, -f32::MAX], &[2]);
    assert_tensor(
        &input.rms_norm(&tensor(&[1.], &[1]), 0.).unwrap(),
        &[2],
        &[1., -1.],
    );
}

#[test]
fn rms_norm_preserves_subnormal_scale() {
    let smallest = f32::from_bits(1);
    let input = tensor(&[smallest, -smallest], &[2]);
    assert_tensor(
        &input.rms_norm(&tensor(&[1.], &[1]), 0.).unwrap(),
        &[2],
        &[1., -1.],
    );
}

#[test]
fn rms_norm_single_feature() {
    assert_tensor(
        &tensor(&[-3.], &[1])
            .rms_norm(&tensor(&[2.], &[1]), 0.)
            .unwrap(),
        &[1],
        &[-2.],
    );
}

#[test]
fn rms_norm_rejects_invalid_epsilon() {
    let input = tensor(&[1.], &[1]);
    for eps in [-1., f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(input.rms_norm(&input, eps).is_err());
    }
}

#[test]
fn rms_norm_rejects_missing_axis_and_wrong_weight_length() {
    assert!(tensor(&[1.], &[])
        .rms_norm(&tensor(&[1.], &[1]), 0.)
        .is_err());
    assert!(Tensor::ones(vec![2, 3])
        .rms_norm(&tensor(&[1., 1.], &[2]), 0.)
        .is_err());
}

#[test]
fn rms_norm_handles_empty_hidden_and_empty_outer_dimensions() {
    for dims in [vec![2, 0], vec![0, 3]] {
        let input = Tensor::new(vec![], dims.clone());
        assert_tensor(
            &input.rms_norm(&tensor(&[1.], &[1]), 0.).unwrap(),
            &dims,
            &[],
        );
    }
    assert_tensor(
        &tensor(&[], &[0]).rms_norm(&tensor(&[], &[0]), 0.).unwrap(),
        &[0],
        &[],
    );
}

#[test]
fn rms_norm_nonfinite_values_propagate() {
    let result = tensor(&[f32::INFINITY, 2.], &[2])
        .rms_norm(&tensor(&[1.], &[1]), 0.)
        .unwrap();
    assert!(result.data()[0].is_nan());
    assert_eq!(result.data()[1], 0.);
    assert!(tensor(&[1.], &[1])
        .rms_norm(&tensor(&[f32::NAN], &[1]), 0.)
        .unwrap()
        .data()[0]
        .is_nan());
}

#[test]
fn softmax_last_axis_hand_values() {
    let input = tensor(&[1., 2., 3., 0., 0., 0.], &[2, 3]);
    assert_tensor(
        &input.softmax(1).unwrap(),
        &[2, 3],
        &[
            0.09003057,
            0.24472848,
            0.66524094,
            1. / 3.,
            1. / 3.,
            1. / 3.,
        ],
    );
}

#[test]
fn softmax_first_axis() {
    let input = tensor(&[0., 2., 0., 2.], &[2, 2]);
    assert_tensor(&input.softmax(0).unwrap(), &[2, 2], &[0.5; 4]);
}

#[test]
fn softmax_middle_axis_with_multiple_outer_slices() {
    let input = tensor(
        &[0., 2., 1., 1., 2., 0., 2., 0., 1., 1., 0., 2.],
        &[2, 3, 2],
    );
    let (low, middle, high) = (0.09003057, 0.24472848, 0.66524094);
    assert_tensor(
        &input.softmax(1).unwrap(),
        &[2, 3, 2],
        &[
            low, high, middle, middle, high, low, high, low, middle, middle, low, high,
        ],
    );
}

#[test]
fn softmax_is_stable_for_large_offsets() {
    let expected = [0.09003057, 0.24472848, 0.66524094];
    for values in [[1000., 1001., 1002.], [-1002., -1001., -1000.]] {
        assert_tensor(&tensor(&values, &[3]).softmax(0).unwrap(), &[3], &expected);
    }
}

#[test]
fn softmax_extreme_finite_range() {
    assert_tensor(
        &tensor(&[-f32::MAX, 0., f32::MAX], &[3]).softmax(0).unwrap(),
        &[3],
        &[0., 0., 1.],
    );
}

#[test]
fn softmax_large_slice_matches_f64_normalization() {
    let values: Vec<f32> = (0..16_384).map(|i| -((i % 113) as f32) / 7.).collect();
    let denominator: f64 = values.iter().map(|&value| f64::from(value).exp()).sum();
    let expected: Vec<f32> = values
        .iter()
        .map(|&value| (f64::from(value).exp() / denominator) as f32)
        .collect();
    let actual = Tensor::new(values, vec![16_384]).softmax(0).unwrap();
    assert_close(actual.data(), &expected, 1e-9);
    let sum: f64 = actual.data().iter().map(|&value| f64::from(value)).sum();
    assert!((sum - 1.).abs() < 1e-7);
}

#[test]
fn softmax_all_negative_infinity_is_uniform_per_slice() {
    let input = tensor(&[f32::NEG_INFINITY, 0., f32::NEG_INFINITY, 0.], &[2, 2]);
    assert_tensor(&input.softmax(0).unwrap(), &[2, 2], &[0.5; 4]);
}

#[test]
fn softmax_masked_negative_infinities_have_zero_probability() {
    let input = tensor(&[0., f32::NEG_INFINITY, 0.], &[3]);
    assert_tensor(&input.softmax(0).unwrap(), &[3], &[0.5, 0., 0.5]);
}

#[test]
fn softmax_positive_infinities_share_probability_per_slice() {
    let input = tensor(
        &[
            f32::INFINITY,
            2.,
            f32::INFINITY,
            0.,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ],
        &[2, 3],
    );
    assert_tensor(
        &input.softmax(1).unwrap(),
        &[2, 3],
        &[0.5, 0., 0.5, 0., 1., 0.],
    );
}

#[test]
fn softmax_rejects_nan_even_when_slice_contains_infinity() {
    for values in [[0., f32::NAN], [f32::INFINITY, f32::NAN]] {
        assert!(tensor(&values, &[2]).softmax(0).is_err());
    }
}

#[test]
fn softmax_singleton_axis() {
    assert_tensor(
        &tensor(&[-3., 8.], &[2, 1]).softmax(1).unwrap(),
        &[2, 1],
        &[1., 1.],
    );
}

#[test]
fn softmax_empty_axis_and_empty_outer_dimensions() {
    for (dims, axis) in [(vec![2, 0, 3], 1), (vec![0, 2, 3], 1), (vec![2, 3, 0], 0)] {
        assert_tensor(
            &Tensor::new(vec![], dims.clone()).softmax(axis).unwrap(),
            &dims,
            &[],
        );
    }
}

#[test]
fn softmax_rejects_out_of_bounds_axis_and_scalars() {
    assert!(Tensor::ones(vec![2, 3]).softmax(2).is_err());
    assert!(tensor(&[1.], &[]).softmax(0).is_err());
    assert!(tensor(&[], &[0]).softmax(1).is_err());
}

#[test]
fn silu_hand_values() {
    assert_tensor(
        &tensor(&[-2., -1., 0., 1., 2.], &[5]).silu().unwrap(),
        &[5],
        &[-0.23840584, -0.26894143, 0., 0.7310586, 1.7615942],
    );
}

#[test]
fn silu_large_negative_and_positive_inputs_stay_finite() {
    let input = tensor(&[-1000., 1000., -f32::MAX, f32::MAX], &[4]);
    assert_tensor(&input.silu().unwrap(), &[4], &[0., 1000., 0., f32::MAX]);
}

#[test]
fn silu_nonfinite_limits_and_nan() {
    let result = tensor(&[f32::NEG_INFINITY, f32::INFINITY, f32::NAN], &[3])
        .silu()
        .unwrap();
    assert_eq!(result.data()[0], 0.);
    assert_eq!(result.data()[1], f32::INFINITY);
    assert!(result.data()[2].is_nan());
}

#[test]
fn silu_scalar_and_empty_shape() {
    assert_tensor(&tensor(&[1.], &[]).silu().unwrap(), &[], &[0.7310586]);
    assert_tensor(&tensor(&[], &[2, 0]).silu().unwrap(), &[2, 0], &[]);
}

#[test]
fn add_same_shape() {
    assert_tensor(
        &tensor(&[1., 2., 3., 4.], &[2, 2])
            .add(&tensor(&[5., -2., 1., 6.], &[2, 2]))
            .unwrap(),
        &[2, 2],
        &[6., 0., 4., 10.],
    );
}

#[test]
fn sub_same_shape_preserves_operand_order() {
    assert_tensor(
        &tensor(&[1., 2., 3.], &[3])
            .sub(&tensor(&[4., -1., 2.], &[3]))
            .unwrap(),
        &[3],
        &[-3., 3., 1.],
    );
}

#[test]
fn mul_same_shape() {
    assert_tensor(
        &tensor(&[1., -2., 0.], &[3])
            .mul(&tensor(&[4., 3., 2.], &[3]))
            .unwrap(),
        &[3],
        &[4., -6., 0.],
    );
}

#[test]
fn div_same_shape_preserves_operand_order() {
    assert_tensor(
        &tensor(&[8., -9., 0.], &[3])
            .div(&tensor(&[2., 3., 4.], &[3]))
            .unwrap(),
        &[3],
        &[4., -3., 0.],
    );
}

#[test]
fn division_by_zero_follows_ieee_signs_and_nan() {
    let result = tensor(&[1., -1., 0., 1.], &[4])
        .div(&tensor(&[0., 0., 0., -0.], &[4]))
        .unwrap();
    assert_eq!(result.data()[0], f32::INFINITY);
    assert_eq!(result.data()[1], f32::NEG_INFINITY);
    assert!(result.data()[2].is_nan());
    assert_eq!(result.data()[3], f32::NEG_INFINITY);
}

#[test]
fn basic_arithmetic_keeps_ieee_overflow_and_nan() {
    assert_eq!(
        tensor(&[f32::MAX], &[1])
            .add(&tensor(&[f32::MAX], &[1]))
            .unwrap()
            .data()[0],
        f32::INFINITY
    );
    assert_eq!(
        tensor(&[-f32::MAX], &[1])
            .sub(&tensor(&[f32::MAX], &[1]))
            .unwrap()
            .data()[0],
        f32::NEG_INFINITY
    );
    assert!(tensor(&[f32::INFINITY], &[1])
        .mul(&tensor(&[0.], &[1]))
        .unwrap()
        .data()[0]
        .is_nan());
    assert!(tensor(&[f32::NAN], &[1])
        .add(&tensor(&[1.], &[1]))
        .unwrap()
        .data()[0]
        .is_nan());
}

#[test]
fn broadcasting_rank_zero_scalar_on_both_sides() {
    let scalar = tensor(&[2.], &[]);
    let values = tensor(&[4., 8., 16.], &[3]);
    assert_tensor(&values.add(&scalar).unwrap(), &[3], &[6., 10., 18.]);
    assert_tensor(&scalar.sub(&values).unwrap(), &[3], &[-2., -6., -14.]);
    assert_tensor(&scalar.mul(&values).unwrap(), &[3], &[8., 16., 32.]);
    assert_tensor(&scalar.div(&values).unwrap(), &[3], &[0.5, 0.25, 0.125]);
    assert_tensor(&scalar.add(&scalar).unwrap(), &[], &[4.]);
}

#[test]
fn broadcasting_trailing_row() {
    let matrix = tensor(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    assert_tensor(
        &matrix.add(&tensor(&[10., 20., 30.], &[3])).unwrap(),
        &[2, 3],
        &[11., 22., 33., 14., 25., 36.],
    );
}

#[test]
fn broadcasting_column() {
    let matrix = tensor(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    assert_tensor(
        &matrix.mul(&tensor(&[10., 100.], &[2, 1])).unwrap(),
        &[2, 3],
        &[10., 20., 30., 400., 500., 600.],
    );
}

#[test]
fn broadcasting_rank_expansion_and_noncommutative_order() {
    let left = tensor(&[2., 4.], &[2, 1, 1]);
    let right = tensor(&[1., 2., 4.], &[1, 3]);
    assert_tensor(
        &left.sub(&right).unwrap(),
        &[2, 1, 3],
        &[1., 0., -2., 3., 2., 0.],
    );
    assert_tensor(
        &right.div(&left).unwrap(),
        &[2, 1, 3],
        &[0.5, 1., 2., 0.25, 0.5, 1.],
    );
}

#[test]
fn broadcasting_both_operands_expand_multiple_axes() {
    let left = tensor(&[1., 2.], &[2, 1, 1]);
    let right = tensor(&[10., 20., 30., 40., 50., 60.], &[1, 3, 2]);
    assert_tensor(
        &left.add(&right).unwrap(),
        &[2, 3, 2],
        &[11., 21., 31., 41., 51., 61., 12., 22., 32., 42., 52., 62.],
    );
}

#[test]
fn broadcasting_zero_with_singleton_yields_empty_axis() {
    let empty = tensor(&[], &[2, 0, 3]);
    let row = tensor(&[1., 2., 3.], &[1, 1, 3]);
    assert_tensor(&empty.add(&row).unwrap(), &[2, 0, 3], &[]);
    assert_tensor(&row.sub(&empty).unwrap(), &[2, 0, 3], &[]);
    assert_tensor(&empty.mul(&tensor(&[2.], &[])).unwrap(), &[2, 0, 3], &[]);
    assert_tensor(&empty.div(&empty).unwrap(), &[2, 0, 3], &[]);
}

#[test]
fn broadcasting_rejects_incompatible_dimensions_even_if_empty() {
    for (left, right) in [
        (Tensor::ones(vec![2, 3]), Tensor::ones(vec![2])),
        (tensor(&[], &[0, 3]), Tensor::ones(vec![2, 3])),
    ] {
        assert!(left.add(&right).is_err());
        assert!(left.sub(&right).is_err());
        assert!(left.mul(&right).is_err());
        assert!(left.div(&right).is_err());
    }
}

#[test]
fn broadcasting_checks_output_strides_even_for_empty_tensors() {
    let left = tensor(&[], &[0, usize::MAX, 1]);
    let right = tensor(&[], &[0, 1, usize::MAX]);
    assert!(left.add(&right).is_err());
}

#[test]
fn zero_dimension_before_large_prefix_avoids_spurious_product_overflow() {
    let empty = tensor(&[], &[usize::MAX, 2, 0]);
    assert_tensor(&empty.scale(2.).unwrap(), &[usize::MAX, 2, 0], &[]);
    assert_tensor(
        &empty.add(&tensor(&[1.], &[])).unwrap(),
        &[usize::MAX, 2, 0],
        &[],
    );
}

#[test]
fn scale_preserves_shape_and_uses_ieee_multiplication() {
    assert_tensor(
        &tensor(&[1., -2., 0., 4.], &[2, 2]).scale(-0.5).unwrap(),
        &[2, 2],
        &[-0.5, 1., 0., -2.],
    );
    assert_tensor(&tensor(&[3.], &[]).scale(2.).unwrap(), &[], &[6.]);
    assert_tensor(&tensor(&[], &[0, 2]).scale(f32::NAN).unwrap(), &[0, 2], &[]);
    assert_eq!(
        tensor(&[f32::MAX], &[1]).scale(2.).unwrap().data()[0],
        f32::INFINITY
    );
    assert!(tensor(&[0.], &[1]).scale(f32::INFINITY).unwrap().data()[0].is_nan());
}

#[test]
fn rope_position_zero_is_identity() {
    let input = tensor(&[1., 2., 3., 4.], &[1, 4]);
    assert_tensor(&input.rope(4, 0, 10_000.).unwrap(), &[1, 4], input.data());
}

#[test]
fn rope_uses_interleaved_pairs_and_distinct_frequencies() {
    assert_tensor(
        &tensor(&[1., 0., 0., 1.], &[1, 4]).rope(4, 1, 100.).unwrap(),
        &[1, 4],
        &[0.5403023, 0.84147096, -0.099833414, 0.9950042],
    );
}

#[test]
fn rope_multisequence_multihead_uses_start_position() {
    let input = tensor(
        &[
            1., 0., 0., 1., 0., 1., 1., 0., 1., 0., 0., 1., 0., 1., 1., 0.,
        ],
        &[2, 2, 4],
    );
    assert_tensor(
        &input.rope(4, 1, 100.).unwrap(),
        &[2, 2, 4],
        &[
            0.5403023,
            0.84147096,
            -0.099833414,
            0.9950042,
            -0.84147096,
            0.5403023,
            0.9950042,
            0.099833414,
            -0.41614684,
            0.9092974,
            -0.19866933,
            0.9800666,
            -0.9092974,
            -0.41614684,
            0.9800666,
            0.19866933,
        ],
    );
}

#[test]
fn rope_flattens_all_intermediate_head_dimensions() {
    let input = tensor(&[1., 0., 1., 0., 1., 0., 1., 0.], &[1, 2, 2, 2]);
    assert_tensor(
        &input.rope(2, 1, 10_000.).unwrap(),
        &[1, 2, 2, 2],
        &[
            0.5403023, 0.84147096, 0.5403023, 0.84147096, 0.5403023, 0.84147096, 0.5403023,
            0.84147096,
        ],
    );
}

#[test]
fn rope_preserves_pair_magnitudes_across_many_sequences() {
    let values: Vec<f32> = (0..32 * 3 * 8).map(|i| (i % 31) as f32 - 15.).collect();
    let input = Tensor::new(values, vec![32, 3, 8]);
    const PAIR: usize = 2;
    let result = input.rope(8, 123, 10_000.).unwrap();
    let (before_pairs, _) = input.data().as_chunks::<PAIR>();
    let (after_pairs, _) = result.data().as_chunks::<PAIR>();
    for (before, after) in before_pairs.iter().zip(after_pairs) {
        let norm_before = f64::from(before[0]).powi(2) + f64::from(before[1]).powi(2);
        let norm_after = f64::from(after[0]).powi(2) + f64::from(after[1]).powi(2);
        assert!((norm_before - norm_after).abs() <= 2e-7 * norm_before.max(1.));
    }
}

#[test]
fn rope_handles_extreme_finite_values_at_zero_position() {
    let input = tensor(&[f32::MAX, -f32::MAX], &[1, 2]);
    assert_tensor(&input.rope(2, 0, 10_000.).unwrap(), &[1, 2], input.data());
}

#[test]
fn rope_rejects_invalid_rank_and_head_dimension() {
    assert!(tensor(&[1., 2.], &[2]).rope(2, 0, 100.).is_err());
    assert!(tensor(&[1.], &[]).rope(2, 0, 100.).is_err());
    for head_dim in [0, 1, 3, 6] {
        assert!(Tensor::ones(vec![2, 4]).rope(head_dim, 0, 100.).is_err());
    }
    assert!(tensor(&[1., 2., 3.], &[1, 3]).rope(3, 0, 100.).is_err());
}

#[test]
fn rope_rejects_nonpositive_and_nonfinite_theta() {
    for theta in [0., -1., f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(Tensor::ones(vec![1, 2]).rope(2, 0, theta).is_err());
    }
}

#[test]
fn rope_accepts_positive_subnormal_theta() {
    let result = Tensor::ones(vec![1, 4])
        .rope(4, 1, f32::from_bits(1))
        .unwrap();
    assert!(result.data().iter().all(|value| value.is_finite()));
}

#[test]
fn rope_checks_last_position_overflow() {
    assert!(Tensor::ones(vec![2, 2]).rope(2, usize::MAX, 100.).is_err());
    assert!(Tensor::ones(vec![1, 2]).rope(2, usize::MAX, 100.).is_ok());
    assert!(Tensor::ones(vec![2, 2])
        .rope(2, usize::MAX - 1, 100.)
        .is_ok());
}

#[test]
fn rope_empty_shapes_preserve_shape_with_valid_parameters() {
    for dims in [vec![0, 4], vec![3, 0, 4]] {
        assert_tensor(
            &Tensor::new(vec![], dims.clone()).rope(4, 2, 100.).unwrap(),
            &dims,
            &[],
        );
    }
    assert!(tensor(&[], &[0, 4]).rope(4, usize::MAX, 100.).is_ok());
    assert!(tensor(&[], &[2, 0, 4]).rope(4, usize::MAX, 100.).is_err());
}

#[test]
fn rope_nonfinite_data_propagates() {
    let result = tensor(&[f32::NAN, 1.], &[1, 2]).rope(2, 1, 100.).unwrap();
    assert!(result.data().iter().all(|value| value.is_nan()));
}

#[test]
fn empty_inputs_still_validate_operation_parameters() {
    let empty = tensor(&[], &[0, 4]);
    assert!(empty.rms_norm(&tensor(&[1., 1.], &[2]), 0.).is_err());
    assert!(empty.rms_norm(&tensor(&[1.], &[1]), f32::NAN).is_err());
    assert!(empty.rope(2, 0, 100.).is_err());
    assert!(empty.rope(4, 0, -1.).is_err());
    assert!(empty.softmax(2).is_err());
}

#[test]
fn unary_operations_reject_shape_buffer_mismatch() {
    for malformed in [
        tensor(&[1.], &[1, 2]),
        tensor(&[1., 2., 3.], &[1, 2]),
        tensor(&[], &[]),
    ] {
        assert!(malformed.silu().is_err());
        assert!(malformed.scale(2.).is_err());
        assert!(malformed.softmax(0).is_err());
        assert!(malformed.rms_norm(&tensor(&[1.], &[1]), 0.).is_err());
        assert!(malformed.rope(2, 0, 100.).is_err());
    }
}

#[test]
fn binary_operations_reject_malformed_either_operand() {
    let valid = tensor(&[1., 2.], &[1, 2]);
    let malformed = tensor(&[1.], &[1, 2]);
    for (left, right) in [(&valid, &malformed), (&malformed, &valid)] {
        assert!(left.add(right).is_err());
        assert!(left.sub(right).is_err());
        assert!(left.mul(right).is_err());
        assert!(left.div(right).is_err());
    }
    assert!(malformed.matmul(&tensor(&[1., 2.], &[2, 1])).is_err());
    assert!(tensor(&[1., 2.], &[2, 1]).matmul(&malformed).is_err());
}

#[test]
fn rms_norm_rejects_malformed_weight_even_with_valid_buffer_length() {
    let malformed = tensor(&[1., 2.], &[3]);
    assert!(tensor(&[3., 4.], &[2]).rms_norm(&malformed, 0.).is_err());
}

#[test]
fn operations_reject_overflowing_input_shape() {
    // 最末维步长可构造，但总元素数溢出；不能让算子依赖构造函数校验。
    let malformed = tensor(&[], &[usize::MAX, 2]);
    assert!(malformed.silu().is_err());
    assert!(malformed.scale(1.).is_err());
    assert!(malformed.softmax(1).is_err());
    assert!(malformed.rms_norm(&tensor(&[1.], &[1]), 0.).is_err());
    assert!(malformed.rope(2, 0, 100.).is_err());
    assert!(malformed.add(&tensor(&[1.], &[])).is_err());
    assert!(malformed.matmul(&tensor(&[1., 1.], &[2, 1])).is_err());
}

#[test]
fn operations_do_not_modify_shared_input_buffers() {
    let input = tensor(&[1., 2., 3., 4.], &[2, 2]);
    let shared = input.view();
    let other = tensor(&[2.], &[]);
    let _ = input.add(&other).unwrap();
    let _ = input.sub(&other).unwrap();
    let _ = input.mul(&other).unwrap();
    let _ = input.div(&other).unwrap();
    let _ = input.scale(3.).unwrap();
    let _ = input.silu().unwrap();
    let _ = input.softmax(1).unwrap();
    let _ = input.rms_norm(&other, 1e-5).unwrap();
    let _ = input.rope(2, 1, 100.).unwrap();
    let _ = input.matmul(&input).unwrap();
    assert_tensor(&input, &[2, 2], &[1., 2., 3., 4.]);
    assert_tensor(&shared, &[2, 2], &[1., 2., 3., 4.]);
}

/// 固定 README 中的示例，确保文档片段与实现保持一致。
#[test]
fn readme_broadcast_and_softmax_example() {
    let x = tensor(&[1., 2., 3., 4.], &[2, 2]);
    let bias = tensor(&[10., 20.], &[2]);
    let y = x.add(&bias).unwrap();
    assert_tensor(&y, &[2, 2], &[11., 22., 13., 24.]);

    let probabilities = y.softmax(1).unwrap();
    assert_eq!(probabilities.dims(), &[2, 2]);
    assert_close(
        probabilities.data(),
        &[1.6701422e-5, 0.9999833, 1.6701422e-5, 0.9999833],
        1e-5,
    );
}

/// llama.cpp `GGML_ROPE_TYPE_NORMAL` 配对方式的独立参考实现。
///
/// 翻译的是上游两条链：`ggml/src/ggml.c:4327` 的 `ggml_rope_impl()` 决定调用哪个
/// CPU 算子，`ggml/src/ggml-cpu/ops.cpp:6295` 的 `ggml_compute_forward_rope_flt()`
/// 在 `:6419-6420` 用 `rotate_pairs<T>(n_dims, 1, cache, src + n_offs,
/// dst_data + n_offs, 1)` 处理 NORMAL 分支；`rotate_pairs()` 的签名与函数体在
/// `:6276-6292`（参数是 `n, n_offset, cache, src_data, dst_data, scale=2`，
/// NORMAL 显式传 `n_offset=1, scale=1`）。于是第 i 对取 **相邻** 元素
/// `(x[2i], x[2i+1])`，角频率 `theta^(-2i/n_dims)`。
///
/// 本函数只按上述定义用 f64 计算，不调用被测代码，因此可以充当 `rope` 的第二意见。
fn llama_cpp_normal_reference(x: &[f32], head_dim: usize, pos: usize, theta: f64) -> Vec<f32> {
    let mut output = Vec::with_capacity(x.len());
    for (sequence, chunk) in x.chunks_exact(head_dim).enumerate() {
        let position = (pos + sequence) as f64;
        for pair in 0..head_dim / 2 {
            let frequency = theta.powf(-2.0 * pair as f64 / head_dim as f64);
            let (sin, cos) = (position * frequency).sin_cos();
            let first = f64::from(chunk[2 * pair]);
            let second = f64::from(chunk[2 * pair + 1]);
            output.push((first * cos - second * sin) as f32);
            output.push((first * sin + second * cos) as f32);
        }
    }
    output
}

/// HF `rotate_half`（llama.cpp 的 NEOX 分支）配对方式：`(x[i], x[i + head_dim/2])`。
/// 用于判别性断言——它必须与我们的实现**不相等**。
fn hf_rotate_half_reference(x: &[f32], head_dim: usize, pos: usize, theta: f64) -> Vec<f32> {
    let mut output = vec![0.0f32; x.len()];
    let half = head_dim / 2;
    for (sequence, chunk) in x.chunks_exact(head_dim).enumerate() {
        let position = (pos + sequence) as f64;
        for pair in 0..half {
            let frequency = theta.powf(-2.0 * pair as f64 / head_dim as f64);
            let (sin, cos) = (position * frequency).sin_cos();
            let first = f64::from(chunk[pair]);
            let second = f64::from(chunk[pair + half]);
            output[sequence * head_dim + pair] = (first * cos - second * sin) as f32;
            output[sequence * head_dim + pair + half] = (first * sin + second * cos) as f32;
        }
    }
    output
}

/// 判别测试①：`rope` 必须逐元素等于 llama.cpp NORM 分支的独立翻译。
///
/// head_dim=8、pos=1..2、theta=10000 下各对使用明显不同的频率，
/// 使相邻对与半距对两种配对方式给出不同结果（见判别测试②）。
#[test]
fn rope_matches_llamacpp_normal_pairing_reference() {
    // 形状 [2, 8]：2 个序列位置 × 1 个头 × head_dim 8，参考实现按位置递增取角度。
    let input = tensor(
        &[
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        ],
        &[2, 8],
    );
    let actual = input.rope(8, 1, 10_000.).unwrap();
    let expected = llama_cpp_normal_reference(input.data(), 8, 1, 10_000.);
    assert_tensor(&actual, &[2, 8], &expected);
}

/// 判别测试②：同一输入下，我们的 `rope` 与 HF `rotate_half`（NEOX）必须**不相等**。
///
/// 这是阶段 7 用真实训练模型（stories15M，llama 架构）确认过的结论：
/// llama 架构取 `LLAMA_ROPE_TYPE_NORM`，配对方式错误会让末行 logits argmax 翻转、
/// 逐元素偏差达 9.12（tiny-f32 随机权重夹具对该错误几乎不敏感，偏差只有 5.8e-04，
/// 所以本用例用放大差异的手工输入固定性质，不依赖任何模型；模型指纹与复现命令
/// 见 `tests/golden/` 的 TSV 头部）。
#[test]
fn rope_rejects_hf_rotate_half_pairing() {
    let input = tensor(
        &[
            -2.0, 0.33333334, -1.6666667, 0.6666667, -1.3333334, 1.0, -1.0, 1.3333334,
        ],
        &[1, 8],
    );
    let actual = input.rope(8, 1, 10_000.).unwrap();
    let normal = llama_cpp_normal_reference(input.data(), 8, 1, 10_000.);
    let neox = hf_rotate_half_reference(input.data(), 8, 1, 10_000.);

    // 两种配对方式本身必须可区分，否则本用例没有判别力。
    let separation = normal
        .iter()
        .zip(&neox)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(separation > 0.5, "参考实现之间差异过小: {separation}");

    assert_tensor(&actual, &[1, 8], &normal);
    let distance_to_neox = actual
        .data()
        .iter()
        .zip(&neox)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        distance_to_neox > 0.5,
        "rope 结果不应接近 HF rotate_half: 最大偏差 {distance_to_neox}"
    );
}
