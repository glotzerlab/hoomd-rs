// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![allow(missing_docs)]
#![allow(clippy::missing_docs_in_private_items)]

/*! Benchmark Angle */

use divan::counter::ItemsCount;
use divan::{self, black_box, Bencher};
use rand::{rngs::StdRng, Rng, SeedableRng};

use hoomd_vector::{Angle, Cartesian, Rotate, RotationMatrix};

fn main() {
    divan::main();
}

#[divan::bench]
fn rotate(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    let a: Angle = rng.gen();

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| -> Cartesian<2> { rng.gen::<Cartesian<2>>() })
        .bench_local_values(|vec| black_box(a.rotate(&vec)));
}

#[divan::bench]
fn rotate_matrix(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    let a: Angle = rng.gen();
    let matrix = RotationMatrix::from(a);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| -> Cartesian<2> { rng.gen::<Cartesian<2>>() })
        .bench_local_values(|vec| black_box(matrix.rotate(&vec)));
}

#[divan::bench]
fn gen_random(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .bench_local(|| black_box(rng.gen::<Angle>()));
}
