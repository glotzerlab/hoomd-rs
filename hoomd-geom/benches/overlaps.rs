// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![allow(missing_docs)]
#![allow(clippy::missing_docs_in_private_items)]

/*! Benchmark overlaps*/

use divan::counter::ItemsCount;
use divan::{self, Bencher, black_box};
use hoomd_geom::Simplex3;
use hoomd_geom::{
    Cuboid, IntersectsAt, Sphere,
    poly::ConvexPolytope,
    xenocollide::{collide2d, collide3d},
};
use rand::{Rng, SeedableRng, rngs::StdRng};

use hoomd_vector::{Angle, Cartesian, Rotation, RotationMatrix, Versor};

#[inline(never)]
fn asm_collide3d() {
    let mut rng = StdRng::seed_from_u64(1);
    let (p0, p1) = create_dipyramid_pair::<4, _>(&mut rng, 10.0);
    let (t, r) = create_offset_3d(&mut rng);
    collide3d(&p0, &p1, &t, &r);
}

fn main() {
    asm_collide3d();
    divan::main();
}

fn create_sphere_pair<const N: usize, R: Rng>(rng: &mut R) -> (Sphere<N>, Sphere<N>) {
    (
        Sphere::from(rng.random_range(0.0..10.0)),
        Sphere::from(rng.random_range(0.0..10.0)),
    )
}
fn create_cuboid_pair<const N: usize, R: Rng>(rng: &mut R) -> (Cuboid<N>, Cuboid<N>) {
    (
        Cuboid::from(rng.random::<Cartesian<N>>() * 10.0),
        Cuboid::from(rng.random::<Cartesian<N>>() * 10.0),
    )
}
fn create_simplex_pair<R: Rng>(rng: &mut R) -> (Simplex3, Simplex3) {
    (
        Simplex3::from([
            rng.random::<Cartesian<3>>() * 10.0,
            rng.random::<Cartesian<3>>() * 10.0,
            rng.random::<Cartesian<3>>() * 10.0,
            rng.random::<Cartesian<3>>() * 10.0,
        ]),
        Simplex3::from([
            rng.random::<Cartesian<3>>() * 10.0,
            rng.random::<Cartesian<3>>() * 10.0,
            rng.random::<Cartesian<3>>() * 10.0,
            rng.random::<Cartesian<3>>() * 10.0,
        ]),
    )
}
/// Create a pair of N-dipyramids with random half-heights between 0 and h_max
fn create_dipyramid_pair<const N: usize, R: Rng>(
    rng: &mut R,
    h_max: f64,
) -> (ConvexPolytope<3>, ConvexPolytope<3>) {
    let base = ConvexPolytope::<2>::from(N);
    (
        ConvexPolytope::<3>::from(
            base.vertices
                .iter()
                .map(|x| Cartesian::from([x[0], x[1], 0.0]))
                .chain([
                    [0.0, 0.0, rng.random_range(0.0..h_max)].into(),
                    [0.0, 0.0, -rng.random_range(0.0..h_max)].into(),
                ])
                .collect::<Vec<_>>(),
        ),
        ConvexPolytope::<3>::from(
            base.vertices
                .iter()
                .map(|x| Cartesian::from([x[0], x[1], 0.0]))
                .chain([
                    [0.0, 0.0, rng.random_range(0.0..h_max)].into(),
                    [0.0, 0.0, -rng.random_range(0.0..h_max)].into(),
                ])
                .collect::<Vec<_>>(),
        ),
    )
}

fn create_polygon_pair<const N: usize>() -> (ConvexPolytope<2>, ConvexPolytope<2>) {
    (ConvexPolytope::from(N), ConvexPolytope::from(N))
}

fn create_offset_2d<R: Rng>(rng: &mut R) -> (Cartesian<2>, Angle) {
    (
        rng.random::<Cartesian<2>>() * 10.0,
        Angle::from(rng.random_range((-2.0 * std::f64::consts::PI)..(2.0 * std::f64::consts::PI))),
    )
}

fn create_offset<const N: usize, R: Rng>(rng: &mut R) -> (Cartesian<N>, RotationMatrix<N>) {
    (
        rng.random::<Cartesian<N>>() * 10.0,
        RotationMatrix::default(),
    )
}
fn create_offset_3d<R: Rng>(rng: &mut R) -> (Cartesian<3>, Versor) {
    (rng.random::<Cartesian<3>>() * 10.0, rng.random())
}

const DIMENSIONS: &[usize] = &[2, 3, 4];
const NUM_VERTICES: &[usize] = &[3, 4, 8, 16, 64, 256];

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
#[divan::bench]
fn cuboid_xenocollide_3d(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| {
            (
                create_cuboid_pair::<3, _>(&mut rng),
                create_offset_3d(&mut rng),
                Versor::identity(),
            )
        })
        .bench_local_values(|((c0, c1), (t, _), r)| black_box(collide3d(&c0, &c1, &t, &r)));
}

#[divan::bench(consts = NUM_VERTICES, )]
fn polygon_xenocollide_2d<const N: usize>(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| (create_polygon_pair::<N>(), create_offset_2d(&mut rng)))
        .bench_local_values(|((p0, p1), (t, r))| black_box(collide2d(&p0, &p1, &t, &r)));
}

#[divan::bench(consts = NUM_VERTICES, )]
fn dipyramid_xenocollide_3d<const N: usize>(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| {
            (
                create_dipyramid_pair::<N, _>(&mut rng, 10.0),
                create_offset_3d(&mut rng),
            )
        })
        .bench_local_values(|((p0, p1), (t, r))| black_box(collide3d(&p0, &p1, &t, &r)));
}

#[divan::bench]
fn simplex_xenocollide_3d(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| (create_simplex_pair(&mut rng), create_offset_3d(&mut rng)))
        .bench_local_values(|((t0, t1), (t, r))| black_box(collide3d(&t0, &t1, &t, &r)));
}
#[divan::bench]
fn simplex_fast(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| (create_simplex_pair(&mut rng), create_offset_3d(&mut rng)))
        .bench_local_values(|((t0, t1), (t, r))| black_box(t0.intersects_at(&t1, &t, &r)));
}
