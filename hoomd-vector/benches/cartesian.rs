// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![expect(
    clippy::missing_docs_in_private_items,
    reason = "benches don't need public documentation"
)]

//! Benchmark Cartesian

use divan::{self, Bencher, black_box, counter::ItemsCount};
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Cartesian, Cross, InnerProduct, distribution::Ball};
use rand::{Rng, SeedableRng, distr::Distribution, distr::Uniform, rngs::StdRng};
use rand_distr::StandardNormal;

fn main() {
    divan::main();
}

fn create_random_vector_pair<const N: usize, R: Rng>(rng: &mut R) -> (Cartesian<N>, Cartesian<N>) {
    (rng.random::<Cartesian<N>>(), rng.random::<Cartesian<N>>())
}

fn create_random_ball<const N: usize, R: Rng>(rng: &mut R) -> Ball {
    Ball {
        radius: rng
            .random_range(1e-3..1000.0)
            .try_into()
            .expect("hard-coded constant should be positive"),
    }
}

const DIMENSIONS: &[usize] = &[2, 3, 8, 16, 32, 128];
const BALL_DIMENSIONS: &[usize] = &[2, 3, 4, 8, 16];

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

#[divan::bench(consts = BALL_DIMENSIONS)]
fn gen_ball_rejection<const N: usize>(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| Ball {
            radius: PositiveReal::default(),
        })
        .bench_local_values(|ball| {
            black_box::<Cartesian<N>>(ball.sample(&mut rng));
        });
}
}
