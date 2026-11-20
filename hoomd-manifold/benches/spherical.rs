// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![expect(
    clippy::missing_docs_in_private_items,
    reason = "benches don't need public documentation"
)]

//! Benchmark Minkowski Vector

use divan::{self, Bencher, black_box, counter::ItemsCount};
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng, distr::Uniform};
use rand_distr::StandardNormal;

use hoomd_manifold::{
    Spherical
};
use hoomd_vector::{Cartesian, Metric, Rotate, Versor};

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    divan::main();
}

fn create_random_2sphere<R: Rng>(rng: &mut R) -> Spherical<3> {
    let north_pole = Cartesian::from([0.0,0.0,1.0]);
    let v: Versor = rng.random();
    let transformed_pole = v.rotate(&north_pole);
    Spherical::from_cartesian_coordinates(transformed_pole)
}

fn create_random_3sphere<R: Rng>(rng: &mut R) -> Spherical<4> {
    let v: Versor = rng.random();
    Spherical::<4>::from_versor(v)
}

fn create_random_2sphere_pair<R: Rng>(rng: &mut R) -> (Spherical<3>, Spherical<3>) {
    (
        create_random_2sphere::<_>(rng),
        create_random_2sphere::<_>(rng),
    )
}

fn create_random_3sphere_pair<R: Rng>(rng: &mut R) -> (Spherical<4>, Spherical<4>) {
    (
        create_random_3sphere::<_>(rng),
        create_random_3sphere::<_>(rng),
    )
}

fn translate_2sphere_point<R: Rng>(rng: &mut R, max_distance: f64) -> Spherical<3> {
    let sph = create_random_2sphere::<_>(rng);
    let coord = sph.coordinates();
    let d = rng.sample(
        Uniform::new(0.0, max_distance).expect("max distance must be positive"),
    );
    // create a tangent vector at v
    let mut tangent: [f64; 3] = std::array::from_fn(|_| rng.sample(StandardNormal));
    let tang_v: f64 = (0..3).map(|i| tangent[i] * coord[i]).sum();
    for i in 0..3 {
        tangent[i] -= tang_v * coord[i];
    }
    let tang_norm = tangent.iter().map(|x| x * x).sum::<f64>().sqrt();

    // Riemann exponential map
    let (sn, cs) = d.sin_cos();
    let new: [f64; 3] = std::array::from_fn(|i| cs * coord[i] + sn * tangent[i] / tang_norm);
    Spherical::<3>::from_cartesian_coordinates(new.into())
}

fn translate_3sphere_point<R: Rng>(rng: &mut R, max_distance: f64) -> Spherical<4> {
    let sph = create_random_3sphere::<_>(rng);
    let coord = sph.coordinates();
    let displacement = max_distance * rng.random::<f64>();
    let (sn, cs) = displacement.sin_cos();

    let vec: [f64; 3] = std::array::from_fn(|_| rng.sample(StandardNormal));
    // based on algebraic simplifiation of quaternion "square root"
    let sg = sn / (vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]).sqrt();
    let k = coord[1] * vec[0] + coord[2] * vec[1] + coord[3] * vec[2];
    let a = 1.0 + coord[0];
    // clip for instability near south pole
    let k_rescale = if a > 1e-12 { k / a } else { 0.0 };
    let new = [
        cs * coord[0] - sg * k,
        cs * coord[1] + sg * (vec[0] - k_rescale * coord[1]),
        cs * coord[2] + sg * (vec[1] - k_rescale * coord[2]),
        cs * coord[3] + sg * (vec[2] - k_rescale * coord[3]),
    ];
    Spherical::<4>::from_cartesian_coordinates(new.into())
}

#[cfg(not(target_arch = "wasm32"))]
#[divan::bench]
fn s2_distance(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| create_random_2sphere_pair::<_>(&mut rng))
        .bench_local_values(|(a, b)| {
            black_box(
                a.distance(&b)
            )
        });
}

#[cfg(not(target_arch = "wasm32"))]
#[divan::bench]
fn s3_distance(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| create_random_3sphere_pair::<_>(&mut rng))
        .bench_local_values(|(a, b)| {
            black_box(
                a.distance(&b)
            )
        });
}

#[cfg(not(target_arch = "wasm32"))]
#[divan::bench]
fn gen_random_2sphere(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .bench_local(|| black_box(create_random_2sphere::<_>(&mut rng)));
}

#[cfg(not(target_arch = "wasm32"))]
#[divan::bench]
fn gen_random_3sphere(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .bench_local(|| black_box(create_random_3sphere::<_>(&mut rng)));
}

const MAX_STEPSIZES: &[i32] = &[-1,-2,-3,-4];

#[cfg(not(target_arch = "wasm32"))]
#[divan::bench(consts = MAX_STEPSIZES)]
fn translation_2sphere<const D: i32>(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .bench_local(|| black_box(translate_2sphere_point::<_>(&mut rng, 10_f64.powi(D))));
}

#[cfg(not(target_arch = "wasm32"))]
#[divan::bench(consts = MAX_STEPSIZES)]
fn translation_3sphere<const D: i32>(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .bench_local(|| black_box(translate_3sphere_point::<_>(&mut rng, 10_f64.powi(D))));
}
