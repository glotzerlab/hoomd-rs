// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![allow(missing_docs)]
#![allow(clippy::missing_docs_in_private_items)]

/*! Benchmark overlaps*/

use divan::counter::ItemsCount;
use divan::{self, black_box, Bencher};
use hoomd_geom::{
    poly::ConvexPolytope,
    xenocollide::{collide2d, collide3d},
    Cuboid, IntersectsAt, Sphere,
};
use rand::distributions::Uniform;
use rand::{rngs::StdRng, Rng, SeedableRng};

use hoomd_vector::{Angle, Cartesian, Rotate, Rotation, RotationMatrix, Versor};

fn main() {
    divan::main();
}

fn create_sphere_pair<const N: usize, R: Rng>(rng: &mut R) -> (Sphere<N>, Sphere<N>) {
    (
        Sphere::from(rng.gen_range(0.0..10.0)),
        Sphere::from(rng.gen_range(0.0..10.0)),
    )
}
fn create_cuboid_pair<const N: usize, R: Rng>(rng: &mut R) -> (Cuboid<N>, Cuboid<N>) {
    (
        Cuboid::from(rng.gen::<Cartesian<N>>() * 10.0),
        Cuboid::from(rng.gen::<Cartesian<N>>() * 10.0),
    )
}

fn create_polygon_pair<const N: usize>() -> (ConvexPolytope<2>, ConvexPolytope<2>) {
    (ConvexPolytope::from(N), ConvexPolytope::from(N))
}

fn create_offset_2d<R: Rng>(rng: &mut R) -> (Cartesian<2>, Angle) {
    (
        rng.gen::<Cartesian<2>>() * 10.0,
        Angle::from(rng.gen_range((-2.0 * std::f64::consts::PI)..(2.0 * std::f64::consts::PI))),
    )
}
fn create_offset<const N: usize, R: Rng>(rng: &mut R) -> (Cartesian<N>, RotationMatrix<N>) {
    (rng.gen::<Cartesian<N>>() * 10.0, RotationMatrix::default())
}
fn create_offset_3d<R: Rng>(rng: &mut R) -> (Cartesian<3>, Versor) {
    (rng.gen::<Cartesian<3>>() * 10.0, rng.gen())
}

const DIMENSIONS: &[usize] = &[2, 3, 4];
const NUM_VERTICES: &[usize] = &[3, 8, 16, 64, 256];

#[divan::bench(consts = DIMENSIONS)]
fn sphere_fast_nd<const N: usize>(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| {
            (
                create_sphere_pair::<N, _>(&mut rng),
                create_offset::<N, _>(&mut rng),
            )
        })
        .bench_local_values(|((s0, s1), (t, r))| black_box(s0.intersects_at(&s1, &t, &r)));
}

#[divan::bench]
fn cuboid_fast_2d(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| {
            (
                create_cuboid_pair::<2, _>(&mut rng),
                create_offset_2d(&mut rng),
                Angle::identity(),
            )
        })
        .bench_local_values(|((c0, c1), (t, _), r)| black_box(c0.intersects_at(&c1, &t, &r)));
}

#[divan::bench]
fn sphere_xenocollide_2d(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| {
            (
                create_sphere_pair::<2, _>(&mut rng),
                create_offset_2d(&mut rng),
            )
        })
        .bench_local_values(|((s0, s1), (t, r))| black_box(collide2d(&s0, &s1, &t, &r)));
}

#[divan::bench(sample_size = 10_000)]
fn sphere_xenocollide_3d(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| {
            (
                create_sphere_pair::<3, _>(&mut rng),
                create_offset_3d(&mut rng),
            )
        })
        .bench_local_values(|((s0, s1), (t, r))| black_box(collide3d(&s0, &s1, &t, &r)));
}

/// Note this is not 1:1 with the naive test, as this uses oriented cuboids!
#[divan::bench]
fn cuboid_xenocollide_2d(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| {
            (
                create_cuboid_pair::<2, _>(&mut rng),
                create_offset_2d(&mut rng),
            )
        })
        .bench_local_values(|((c0, c1), (t, r))| black_box(collide2d(&c0, &c1, &t, &r)));
}

#[divan::bench(consts = NUM_VERTICES, )]
fn polygon_xenocollide_2d<const N: usize>(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| (create_polygon_pair::<N>(), create_offset_2d(&mut rng)))
        .bench_local_values(|((p0, p1), (t, r))| black_box(collide2d(&p0, &p1, &t, &r)));
}
