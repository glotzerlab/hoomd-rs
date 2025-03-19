// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![expect(
    clippy::missing_docs_in_private_items,
    reason = "benches don't need public documentation"
)]

/*! Benchmark `ChaCha` implementations */

use divan::counter::ItemsCount;
use divan::{self, Bencher, black_box};
use rand::{Rng, SeedableRng};

fn main() {
    divan::main();
}

const N: &[usize] = &[1, 4, 8, 16, 32];

#[divan::bench(consts = N)]
fn bench_rand_chacha<const N: usize>(bencher: Bencher) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(N))
        .bench_local(|| black_box(core::array::from_fn::<_, N, _>(|_| rng.random::<f64>())));
}

#[divan::bench(consts = N)]
fn bench_chacha20<const N: usize>(bencher: Bencher) {
    let mut rng = chacha20::ChaCha8Rng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(N))
        .bench_local(|| black_box(core::array::from_fn::<_, N, _>(|_| rng.random::<f64>())));
}
