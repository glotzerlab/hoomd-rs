// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![expect(
    clippy::missing_docs_in_private_items,
    reason = "benches don't need public documentation"
)]

//! Benchmark `ChaCha` implementations

use divan::{self, Bencher, black_box, counter::ItemsCount};
use rand::{Rng, SeedableRng};

use hoomd_utility::random::Counter;

fn main() {
    divan::main();
}

const N: &[usize] = &[1, 4, 8, 16, 32];

#[divan::bench(consts = N)]
fn bench_rand_chacha<const N: usize>(bencher: Bencher) {
    bencher.counter(ItemsCount::from(N)).bench_local(|| {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(black_box(1));
        black_box(core::array::from_fn::<_, N, _>(|_| rng.random::<f64>()))
    });
}

#[divan::bench(consts = N)]
fn bench_chacha20<const N: usize>(bencher: Bencher) {
    bencher.counter(ItemsCount::from(N)).bench_local(|| {
        let mut rng = chacha20::ChaCha8Rng::seed_from_u64(black_box(1));
        black_box(core::array::from_fn::<_, N, _>(|_| rng.random::<f64>()))
    });
}

#[divan::bench(consts = N)]
fn bench_counter<const N: usize>(bencher: Bencher) {
    bencher.counter(ItemsCount::from(N)).bench_local(|| {
        let mut rng = Counter::new(black_box(10), black_box(11), black_box(12))
            .index(black_box(13))
            .make_rng();
        black_box(core::array::from_fn::<_, N, _>(|_| rng.random::<f64>()))
    });
}
