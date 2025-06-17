// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![expect(
    clippy::missing_docs_in_private_items,
    reason = "benches don't need public documentation"
)]

/*! Benchmark Minkowski Vector */

use divan::counter::ItemsCount;
use divan::{self, Bencher, black_box};
use rand::distr::Uniform;
use rand::{Rng, SeedableRng, rngs::StdRng};

use hoomd_manifold::{Minkowski, Hyperboloid};

fn main() {
    divan::main();
}

fn create_random_vector_pair<const N: usize, R: Rng>(rng: &mut R) -> (Minkowski<N>, Minkowski<N>) {
    (rng.random::<Minkowski<N>>(), rng.random::<Minkowski<N>>())
}

const DIMENSIONS: &[usize] = &[2, 3, 8, 16, 32, 128];

#[divan::bench]
fn hyperboloid_distance_vec3(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| create_random_vector_pair::<3, _>(&mut rng))
        .bench_local_values(|(a, b)| black_box(a.hyperbolic_distance(&b, 1.0)));
}

#[divan::bench]
fn hyperboloid_distance_vec4(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| create_random_vector_pair::<4, _>(&mut rng))
        .bench_local_values(|(a, b)| black_box(a.hyperbolic_distance(&b, 1.0)));
}

#[divan::bench(consts = DIMENSIONS)]
fn gen_random<const N: usize>(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .bench_local(|| black_box(rng.random::<Minkowski<N>>()));
}