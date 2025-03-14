// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![expect(clippy::missing_docs_in_private_items, reason = "benches don't need public documentation")]
#![expect(clippy::expect_used, reason = "benches can use expect without individual reasons")]

/*! Benchmark Cartesian */

use divan::counter::ItemsCount;
use divan::{self, Bencher, black_box};
use rand::distr::Uniform;
use rand::{Rng, SeedableRng, rngs::StdRng};

use hoomd_vector::{Cartesian, Cross, Vector};

fn main() {
    divan::main();
}

fn create_random_vector_pair<const N: usize, R: Rng>(rng: &mut R) -> (Cartesian<N>, Cartesian<N>) {
    (rng.random::<Cartesian<N>>(), rng.random::<Cartesian<N>>())
}

const DIMENSIONS: &[usize] = &[2, 3, 8, 16, 32, 128];

#[divan::bench(consts = DIMENSIONS)]
fn create_vecn_tryfrom_vec<const N: usize>(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    let range = Uniform::new(-100.0, 100.0).expect("a valid distribution");
    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| -> Vec<f64> { (&mut rng).sample_iter(range).take(N).collect() })
        .bench_local_values(|vec| black_box(Cartesian::<N>::try_from(vec)));
}

#[divan::bench(consts = DIMENSIONS)]
fn add_vecn<const N: usize>(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| create_random_vector_pair::<N, _>(&mut rng))
        .bench_local_values(|(a, b)| black_box(a + b));
}

#[divan::bench(consts = DIMENSIONS)]
fn sub_vecn<const N: usize>(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| create_random_vector_pair::<N, _>(&mut rng))
        .bench_local_values(|(a, b)| black_box(a - b));
}

#[divan::bench(consts = DIMENSIONS)]
fn mul_vecn<const N: usize>(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| create_random_vector_pair::<N, _>(&mut rng))
        .bench_local_values(|(a, b)| {
            black_box(a * b.coordinates[0]);
        });
}

#[divan::bench(consts = DIMENSIONS)]
fn div_vecn<const N: usize>(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| create_random_vector_pair::<N, _>(&mut rng))
        .bench_local_values(|(a, b)| black_box(a / b.coordinates[0]));
}

#[divan::bench(consts = DIMENSIONS)]
fn dot_vecn<const N: usize>(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| create_random_vector_pair::<N, _>(&mut rng))
        .bench_local_values(|(a, b)| black_box(a.dot(&b)));
}

#[divan::bench]
fn cross_vec3(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| create_random_vector_pair::<3, _>(&mut rng))
        .bench_local_values(|(a, b)| black_box(a.cross(&b)));
}

#[divan::bench(consts = DIMENSIONS)]
fn gen_random<const N: usize>(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .bench_local(|| black_box(rng.random::<Cartesian<N>>()));
}
