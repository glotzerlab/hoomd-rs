// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![expect(
    clippy::missing_docs_in_private_items,
    reason = "benches don't need public documentation"
)]

//! Benchmark `LennardJones`

use divan::{self, Bencher, black_box, counter::ItemsCount};
use rand::{Rng, SeedableRng, rngs::StdRng};

use hoomd_interaction::pairwise::{IsotropicEnergy, IsotropicForce, LennardJones};

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    divan::main();
}

#[divan::bench]
fn energy(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    let epsilon: f64 = rng.random();
    let sigma: f64 = rng.random();
    let lj: LennardJones = LennardJones { epsilon, sigma };

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| -> f64 { rng.random::<f64>() })
        .bench_local_values(|r| black_box(lj.energy(r)));
}

#[divan::bench]
fn force(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    let epsilon: f64 = rng.random();
    let sigma: f64 = rng.random();
    let lj: LennardJones = LennardJones { epsilon, sigma };

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| -> f64 { rng.random::<f64>() })
        .bench_local_values(|r| black_box(lj.force(r)));
}
