// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![expect(
    clippy::missing_docs_in_private_items,
    reason = "benches don't need public documentation"
)]

//! Benchmark Angle

use divan::{self, Bencher, black_box, counter::ItemsCount};
use hoomd_rand::Counter;
use hoomd_vector::{Angle, Cartesian, Rotate, RotationMatrix};
use rand::{RngExt, SeedableRng, rngs::StdRng};
// use threefry::ThreeFry2x64Rng;

fn main() {
    divan::main();
}

#[divan::bench]
fn rotate(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    let a: Angle = rng.random();

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| -> Cartesian<2> { rng.random::<Cartesian<2>>() })
        .bench_local_values(|vec| black_box(a.rotate(&vec)));
}

#[divan::bench]
fn rotate_matrix(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    let a: Angle = rng.random();
    let matrix = RotationMatrix::from(a);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| -> Cartesian<2> { rng.random::<Cartesian<2>>() })
        .bench_local_values(|vec| black_box(matrix.rotate(&vec)));
}

#[divan::bench]
fn gen_random(bencher: Bencher) {
    let mut rng = Counter::new(0, 0, 0).make_rng();

    bencher
        .counter(ItemsCount::from(1_u32))
        .bench_local(|| black_box(rng.random::<Angle>()));
}
