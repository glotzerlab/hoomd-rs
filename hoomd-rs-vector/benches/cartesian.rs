// Copyright (c) 2024 The Regents of the University of Michigan.
// Part of hoomd_rs, released under the BSD 3-Clause License.

use divan::counter::ItemsCount;
use divan::{self, Bencher};
use rand::{thread_rng, Rng};

use hoomd_rs_vector::{CartesianVector, CartesianVector3, Vector};

fn main() {
    divan::main();
}

fn create_random_vector_pair<const N: usize>() -> (CartesianVector<N>, CartesianVector<N>) {
    let mut rng = thread_rng();
    (
        CartesianVector::from(std::array::from_fn::<_, N, _>(|_| rng.gen::<f64>())),
        CartesianVector::from(std::array::from_fn::<_, N, _>(|_| rng.gen::<f64>())),
    )
}
fn create_random_vec3() -> CartesianVector3 {
    let mut rng = thread_rng();
    CartesianVector3::from(std::array::from_fn::<_, 3, _>(|_| rng.gen::<f64>()))
}

const DIMENSIONS: &[usize] = &[2, 3, 8, 16, 32, 128];

#[divan::bench(consts = DIMENSIONS)]
fn create_vecn_tryfrom_vec<const N: usize>(bencher: Bencher) {
    let mut rng = thread_rng();

    // TODO: replace rng.gen with something in a more reasonable range
    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| (0..N).map(|_| rng.gen::<f64>()).collect::<Vec<f64>>())
        .bench_local_values(|vec| {
            let _ = CartesianVector::<N>::try_from(vec);
        });
}

#[divan::bench(consts = DIMENSIONS)]
fn add_vecn<const N: usize>(bencher: Bencher) {
    let mut result = CartesianVector::default();
    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(create_random_vector_pair::<N>)
        .bench_local_values(|(a, b)| {
            result += a + b;
        });
}

#[divan::bench(consts = DIMENSIONS)]
fn dot_vecn<const N: usize>(bencher: Bencher) {
    let mut result: f64 = 0.0;
    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(create_random_vector_pair::<N>)
        .bench_local_values(|(a, b)| {
            result += a.dot(&b);
        });
}

#[divan::bench]
fn add_vec3(bencher: Bencher) {
    let mut result = CartesianVector3::default();
    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| (create_random_vec3(), create_random_vec3()))
        .bench_local_values(|(a, b)| result += a + b);
}

#[divan::bench]
fn dot_vec3(bencher: Bencher) {
    let mut result = 0.0;
    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| (create_random_vec3(), create_random_vec3()))
        .bench_local_values(|(a, b)| result += a.dot(&b));
}
