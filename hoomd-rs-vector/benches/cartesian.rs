// Copyright (c) 2024 The Regents of the University of Michigan.
// Part of hoomd_rs, released under the BSD 3-Clause License.

use divan::counter::ItemsCount;
use divan::{self, Bencher};
use std::iter::repeat;
// use rand::distributions::Uniform;
use rand::{thread_rng, Rng};

use hoomd_rs_vector::{CartesianVector, CartesianVector3, Vector};

fn main() {
    divan::main();
}

fn create_vectors<const N: usize>(n_vectors: usize) -> Vec<CartesianVector<N>> {
    Vec::from_iter(repeat(CartesianVector::default()).take(n_vectors))
}
fn create_vec3s(n_vectors: usize) -> Vec<CartesianVector3> {
    Vec::from_iter(repeat(CartesianVector3::default()).take(n_vectors))
}

const N_VECTORS: usize = 200_000;
const DIMENSIONS: &[usize] = &[2, 3, 8, 16, 32, 128];

#[divan::bench(consts = DIMENSIONS)]
fn create_vecn<const N: usize>(bencher: Bencher) {
    let mut rng = thread_rng();

    // TODO: replace rng.gen with something in a more reasonable range
    let random_arrays = (0..N_VECTORS)
        .map(|_| std::array::from_fn(|_| rng.gen::<f64>()))
        .collect::<Vec<[f64; N]>>();

    bencher
        .counter(ItemsCount::from(N_VECTORS))
        .bench_local(|| {
            for a in &random_arrays {
                let mut _temp = CartesianVector::from(*a);
            }
        });
}

#[divan::bench(consts = DIMENSIONS)]
fn add_vecn<const N: usize>(bencher: Bencher) {
    let vectors = create_vectors::<N>(N_VECTORS);

    let mut result = CartesianVector::default();
    bencher
        .counter(ItemsCount::from(N_VECTORS))
        .bench_local(|| {
            for a in &vectors {
                result += *a + *a;
            }
        });
}

#[divan::bench(consts = DIMENSIONS)]
fn dot_vecn<const N: usize>(bencher: Bencher) {
    let vectors = create_vectors::<N>(N_VECTORS);

    let mut result = 0.0;
    bencher
        .counter(ItemsCount::from(N_VECTORS))
        .bench_local(|| {
            for a in &vectors {
                result += a.dot(a);
            }
        });
}

#[divan::bench]
fn add_vec3(bencher: Bencher) {
    let vectors = create_vec3s(N_VECTORS);

    let mut result = CartesianVector3::default();
    bencher
        .counter(ItemsCount::from(N_VECTORS))
        .bench_local(|| {
            for a in &vectors {
                result += *a + *a;
            }
        });
}

#[divan::bench]
fn dot_vec3(bencher: Bencher) {
    let vectors = create_vec3s(N_VECTORS);

    let mut result = 0.0;
    bencher
        .counter(ItemsCount::from(N_VECTORS))
        .bench_local(|| {
            for a in &vectors {
                result += a.dot(a);
            }
        });
}
