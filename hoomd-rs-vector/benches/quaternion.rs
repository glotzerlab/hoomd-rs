// Copyright (c) 2024 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![allow(missing_docs)]
#![allow(clippy::missing_docs_in_private_items)]

/*! Benchmark Quaternion */

use divan::counter::ItemsCount;
use divan::{self, black_box, Bencher};
use rand::{rngs::StdRng, Rng, SeedableRng};

use hoomd_rs_vector::rotation::Quaternion;
use hoomd_rs_vector::vector::Cartesian;
use hoomd_rs_vector::Rotate;

fn main() {
    divan::main();
}

#[divan::bench]
fn rotate(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    let q: Quaternion = rng.gen();

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| -> Cartesian<3> { rng.gen::<Cartesian<3>>() })
        .bench_local_values(|vec| black_box(q.rotate(&vec)));
}

#[divan::bench]
fn rotate_precomputed(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    let a: Quaternion = rng.gen();
    let precomputed = a.to_precomputed();

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| -> Cartesian<3> { rng.gen::<Cartesian<3>>() })
        .bench_local_values(|vec| black_box(precomputed.rotate(&vec)));
}

#[divan::bench]
fn gen_random(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .bench_local(|| black_box(rng.gen::<Quaternion>()));
}
