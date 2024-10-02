// Copyright (c) 2024 The Regents of the University of Michigan.
// Part of hoomd_rs, released under the BSD 3-Clause License.

use divan::counter::ItemsCount;
use divan::{self, Bencher};
use std::iter::repeat;

use hoomd_rs_vector::CartesianVector;

fn main() {
    divan::main();
}

fn create_vectors<const N: usize>(n_vectors: usize) -> Vec<CartesianVector<N>> {
    Vec::from_iter(repeat(CartesianVector::default()).take(n_vectors))
}

const N_VECTORS: usize = 100_000;
const DIMENSIONS: &[usize] = &[2, 3, 8, 16, 32, 128];

#[divan::bench(consts = DIMENSIONS)]
fn add<const N: usize>(bencher: Bencher) {
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
