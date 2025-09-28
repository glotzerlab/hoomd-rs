// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![expect(
    clippy::missing_docs_in_private_items,
    reason = "benches don't need public documentation"
)]

//! Benchmark Matrix

use divan::{self, Bencher, black_box, counter::ItemsCount};
use rand::{Rng, SeedableRng, rngs::StdRng};

use hoomd_linear_algebra::{Invertible, MatMul, matrix::Matrix};

fn main() {
    divan::main();
}

/// Creates a matrix of size N x M with random f64 elements.
fn create_random_matrix<const N: usize, const M: usize, R: Rng>(rng: &mut R) -> Matrix<N, M> {
    let rows = std::array::from_fn(|_| std::array::from_fn(|_| rng.random::<f64>()));
    Matrix { rows }
}

/// Creates a pair of random square matrices of size N x N.
fn create_random_matrix_pair<const N: usize, R: Rng>(rng: &mut R) -> (Matrix<N, N>, Matrix<N, N>) {
    (create_random_matrix(rng), create_random_matrix(rng))
}

/// Dimensions for general square matrix benchmarks.
const SQUARE_DIMENSIONS: &[usize] = &[2, 3, 4, 8, 16, 64];
/// Dimensions for determinant benchmarks, which are O(n!)
const DETERMINANT_DIMENSIONS: &[usize] = &[2, 3, 4, 5, 6, 7, 8];

#[divan::bench(consts = SQUARE_DIMENSIONS)]
fn matmul_matn<const N: usize>(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(42);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| create_random_matrix_pair::<N, _>(&mut rng))
        .bench_local_values(|(a, b)| black_box(a.matmul(&b)));
}

#[divan::bench(consts = DETERMINANT_DIMENSIONS)]
fn det_matn<const N: usize>(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(42);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| create_random_matrix::<N, N, _>(&mut rng))
        .bench_local_values(|a| black_box(a.determinant()));
}

#[divan::bench]
fn det_mat3_fast(bencher: Bencher) {
    #[expect(clippy::many_single_char_names, reason = "clarity")]
    fn det33(mat: &Matrix<3, 3>) -> f64 {
        let [[a, b, c], [d, e, f], [g, h, i]] = mat.rows;
        a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    }
    let mut rng = StdRng::seed_from_u64(42);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| create_random_matrix::<3, 3, _>(&mut rng))
        .bench_local_values(|a| black_box(det33(&a)));
}

#[divan::bench]
fn det_mat4_fast(bencher: Bencher) {
    #[expect(clippy::many_single_char_names, reason = "clarity")]
    fn det44(mat: &Matrix<4, 4>) -> f64 {
        let [[a, b, c, d], [e, f, g, h], [i, j, k, l], [m, n, o, p]] = mat.rows;

        a * (f * (k * p - l * o) - g * (j * p - l * n) + h * (j * o - k * n))
            - b * (e * (k * p - l * o) - g * (i * p - l * m) + h * (i * o - k * m))
            + c * (e * (j * p - l * n) - f * (i * p - l * m) + h * (i * n - j * m))
            - d * (e * (j * o - k * n) - f * (i * o - k * m) + g * (i * n - j * m))
    }
    let mut rng = StdRng::seed_from_u64(42);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| create_random_matrix::<4, 4, _>(&mut rng))
        .bench_local_values(|a| black_box(det44(&a)));
}

#[divan::bench]
fn inverse_mat2(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(42);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| create_random_matrix::<2, 2, _>(&mut rng))
        .bench_local_values(|a| black_box(a.inverse()));
}

#[divan::bench]
fn svd_mat2(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(42);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| create_random_matrix::<2, 2, _>(&mut rng))
        .bench_local_values(|a| black_box(a.svd()));
}
