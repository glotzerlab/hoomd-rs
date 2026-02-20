// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![expect(
    clippy::missing_docs_in_private_items,
    reason = "benches don't need public documentation"
)]

//! Benchmark Biquaternion

use divan::{self, Bencher, black_box, counter::ItemsCount};
use rand::{RngExt, SeedableRng, rngs::StdRng};

use hoomd_manifold::{HyperbolicRotate, HyperbolicRotationMatrix, Minkowski, UnitBiquaternion};

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    divan::main();
}

#[cfg(not(target_arch = "wasm32"))]
#[divan::bench]
fn hyperbolic_rotate(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    let q: UnitBiquaternion = rng.random();

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| -> Minkowski<4> { rng.random::<Minkowski<4>>() })
        .bench_local_values(|vec| black_box(q.hyperbolic_rotate(&vec)));
}

#[cfg(not(target_arch = "wasm32"))]
#[divan::bench]
fn hyperbolic_rotate_matrix(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    let a: UnitBiquaternion = rng.random();
    let matrix = HyperbolicRotationMatrix::from(a);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| -> Minkowski<4> { rng.random::<Minkowski<4>>() })
        .bench_local_values(|vec| black_box(matrix.hyperbolic_rotate(&vec)));
}

#[cfg(not(target_arch = "wasm32"))]
#[divan::bench]
fn gen_random(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .bench_local(|| black_box(rng.random::<UnitBiquaternion>()));
}
