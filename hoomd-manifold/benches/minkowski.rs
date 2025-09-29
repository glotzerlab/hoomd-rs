// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![expect(
    clippy::missing_docs_in_private_items,
    reason = "benches don't need public documentation"
)]

//! Benchmark Minkowski Vector

use divan::{self, Bencher, black_box, counter::ItemsCount};
use rand::{Rng, SeedableRng, rngs::StdRng};

use hoomd_manifold::{
    HyperbolicAngle, HyperbolicRotate, HyperbolicRotationMatrix, Hyperboloid, Minkowski,
};
use hoomd_vector::Metric;

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    divan::main();
}

fn create_random_hyperboloid<R: Rng>(rng: &mut R) -> Minkowski<3> {
    let v = rng.random::<HyperbolicAngle>();
    let matrix = HyperbolicRotationMatrix::from(v);
    let origin = Minkowski::from([0.0, 0.0, 1.0]);
    matrix.hyperbolic_rotate(&origin)
}

fn create_random_hyperboloid_pair<R: Rng>(rng: &mut R) -> (Minkowski<3>, Minkowski<3>) {
    (
        create_random_hyperboloid::<_>(rng),
        create_random_hyperboloid::<_>(rng),
    )
}

const DIMENSIONS: &[usize] = &[2, 3, 8, 16, 32, 128];

#[cfg(not(target_arch = "wasm32"))]
#[divan::bench]
fn hyperboloid_distance_vec3(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| create_random_hyperboloid_pair::<_>(&mut rng))
        .bench_local_values(|(a, b)| {
            black_box(Hyperboloid::from(a, 1.0).distance(&Hyperboloid::from(b, 1.0)))
        });
}

#[cfg(not(target_arch = "wasm32"))]
#[divan::bench]
fn to_poincare_vec3(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| create_random_hyperboloid::<_>(&mut rng))
        .bench_local_values(|a| black_box(Hyperboloid::from(a, 1.0).to_poincare()));
}

#[cfg(not(target_arch = "wasm32"))]
#[divan::bench(consts = DIMENSIONS)]
fn gen_random_minkowski<const N: usize>(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .bench_local(|| black_box(rng.random::<Minkowski<N>>()));
}

#[cfg(not(target_arch = "wasm32"))]
#[divan::bench]
fn gen_random_hyperboloid(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .bench_local(|| black_box(create_random_hyperboloid::<_>(&mut rng)));
}
