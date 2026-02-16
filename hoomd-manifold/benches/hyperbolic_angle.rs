// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![expect(
    clippy::missing_docs_in_private_items,
    reason = "benches don't need public documentation"
)]

//! Benchmark `HyperbolicAngle`

use divan::{self, Bencher, black_box, counter::ItemsCount};
use rand::{Rng, SeedableRng, rngs::StdRng};

use hoomd_manifold::{HyperbolicAngle, HyperbolicRotate, HyperbolicRotationMatrix, Minkowski};

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    divan::main();
}

#[cfg(not(target_arch = "wasm32"))]
#[divan::bench]
fn hyperbolic_rotate_matrix(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    let a: HyperbolicAngle = rng.random();
    let matrix = HyperbolicRotationMatrix::from(a);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| -> Minkowski<3> { rng.random::<Minkowski<3>>() })
        .bench_local_values(|angles| black_box(matrix.hyperbolic_rotate(&angles)));
}

#[cfg(not(target_arch = "wasm32"))]
#[divan::bench]
fn gen_random(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .bench_local(|| black_box(rng.random::<HyperbolicAngle>()));
}
