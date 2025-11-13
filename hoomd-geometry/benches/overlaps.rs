// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![expect(
    clippy::missing_docs_in_private_items,
    reason = "benches don't need public documentation"
)]
#![expect(clippy::unwrap_used, reason = "benches can use unwrap where needed")]
#![expect(clippy::wildcard_imports, reason = "simplifies code")]

//! Benchmark overlaps

use divan::{self, Bencher, black_box, counter::ItemsCount};
use hoomd_geometry::{
    Convex, IntersectsAt,
    shape::{
        Capsule, ConvexPolytope, Cylinder, Hypercuboid, Hyperellipsoid, Hypersphere, Simplex3,
    },
    xenocollide::{collide2d, collide3d},
};
use hoomd_vector::{Angle, Cartesian, Versor};
use rand::{Rng, SeedableRng, rngs::StdRng};

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

fn shapes_to_convex<S>(tup: (S, S)) -> (Convex<S>, Convex<S>) {
    (Convex(tup.0), Convex(tup.1))
}

fn create_sphere_pair<const N: usize, R: Rng>(rng: &mut R) -> (Hypersphere<N>, Hypersphere<N>) {
    (
        Hypersphere::<N> {
            radius: rng
                .random_range(0.0..10.0)
                .try_into()
                .expect("test value is a positive real"),
        },
        Hypersphere::<N> {
            radius: rng
                .random_range(0.0..10.0)
                .try_into()
                .expect("test value is a positive real"),
        },
    )
}

fn create_cuboid_pair<const N: usize, R: Rng>(rng: &mut R) -> (Hypercuboid<N>, Hypercuboid<N>) {
    (
        Hypercuboid {
            edge_lengths: (rng.random::<Cartesian<N>>() * 10.0).coordinates.map(|x| {
                (x + 11.0)
                    .try_into()
                    .expect("test value is a positive real")
            }),
        },
        Hypercuboid {
            edge_lengths: (rng.random::<Cartesian<N>>() * 10.0).coordinates.map(|x| {
                (x + 11.0)
                    .try_into()
                    .expect("test value is a positive real")
            }),
        },
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
/// Create a pair of N-dipyramids with random half-heights between 0 and `h_max`
fn create_dipyramid_pair<const N: usize, R: Rng>(
    rng: &mut R,
    h_max: f64,
) -> (ConvexPolytope<3>, ConvexPolytope<3>) {
    let base = ConvexPolytope::<2>::regular(N);
    (
        ConvexPolytope::<3>::with_vertices(
            base.vertices()
                .iter()
                .map(|x| Cartesian::from([x[0], x[1], 0.0]))
                .chain([
                    [0.0, 0.0, rng.random_range(0.0..h_max)].into(),
                    [0.0, 0.0, -rng.random_range(0.0..h_max)].into(),
                ]),
        )
        .unwrap(),
        ConvexPolytope::<3>::with_vertices(
            base.vertices()
                .iter()
                .map(|x| Cartesian::from([x[0], x[1], 0.0]))
                .chain([
                    [0.0, 0.0, rng.random_range(0.0..h_max)].into(),
                    [0.0, 0.0, -rng.random_range(0.0..h_max)].into(),
                ]),
        )
        .unwrap(),
    )
}

fn create_polygon_pair<const N: usize>() -> (ConvexPolytope<2>, ConvexPolytope<2>) {
    (ConvexPolytope::regular(N), ConvexPolytope::regular(N))
}

fn create_ellipsoid_pair<const N: usize, R: Rng>(
    rng: &mut R,
) -> (Hyperellipsoid<N>, Hyperellipsoid<N>) {
    (
        Hyperellipsoid::with_semi_axes((rng.random::<Cartesian<N>>() * 10.0).coordinates.map(
            |x| {
                (x + 11.0)
                    .try_into()
                    .expect("test value is a positive real")
            },
        )),
        Hyperellipsoid::with_semi_axes((rng.random::<Cartesian<N>>() * 10.0).coordinates.map(
            |x| {
                (x + 11.0)
                    .try_into()
                    .expect("test value is a positive real")
            },
        )),
    )
}

fn create_offset_2d<R: Rng>(rng: &mut R) -> (Cartesian<2>, Angle) {
    (
        rng.random::<Cartesian<2>>() * 10.0,
        Angle::from(rng.random_range((-2.0 * std::f64::consts::PI)..(2.0 * std::f64::consts::PI))),
    )
}

fn create_offset<const N: usize, R: Rng>(rng: &mut R) -> Cartesian<N> {
    rng.random::<Cartesian<N>>() * 10.0
}
fn create_offset_3d<R: Rng>(rng: &mut R) -> (Cartesian<3>, Versor) {
    (rng.random::<Cartesian<3>>() * 10.0, rng.random())
}

const DIMENSIONS: &[usize] = &[2, 3, 4];
const NUM_VERTICES: &[usize] = &[3, 4, 8, 16, 64, 256];

#[divan::bench_group]
mod sphere {
    use super::*;

    #[divan::bench(consts = DIMENSIONS)]
    fn fast_nd<const N: usize>(bencher: Bencher) {
        let mut rng = StdRng::seed_from_u64(1);

        bencher
            .counter(ItemsCount::from(1_u32))
            .with_inputs(|| {
                (
                    create_sphere_pair::<N, _>(&mut rng),
                    create_offset::<N, _>(&mut rng),
                )
            })
            .bench_local_values(|((s0, s1), t)| black_box(s0.intersects(&s1, &t)));
    }

    #[divan::bench]
    fn xenocollide_2d(bencher: Bencher) {
        let mut rng = StdRng::seed_from_u64(1);

        bencher
            .counter(ItemsCount::from(1_u32))
            .with_inputs(|| {
                (
                    shapes_to_convex(create_sphere_pair::<2, _>(&mut rng)),
                    create_offset_2d(&mut rng),
                )
            })
            .bench_local_values(|((s0, s1), (t, r))| black_box(collide2d(&s0, &s1, &t, &r)));
    }

    #[divan::bench(sample_size = 10_000)]
    fn xenocollide_3d(bencher: Bencher) {
        let mut rng = StdRng::seed_from_u64(1);

        bencher
            .counter(ItemsCount::from(1_u32))
            .with_inputs(|| {
                (
                    shapes_to_convex(create_sphere_pair::<3, _>(&mut rng)),
                    create_offset_3d(&mut rng),
                )
            })
            .bench_local_values(|((s0, s1), (t, r))| black_box(collide3d(&s0, &s1, &t, &r)));
    }
}

#[divan::bench_group]
mod cuboid {
    use super::*;

    #[divan::bench(consts = DIMENSIONS)]
    fn aligned_nd<const N: usize>(bencher: Bencher) {
        let mut rng = StdRng::seed_from_u64(1);

        bencher
            .counter(ItemsCount::from(1_u32))
            .with_inputs(|| {
                (
                    create_cuboid_pair::<N, _>(&mut rng),
                    create_offset::<N, _>(&mut rng),
                )
            })
            .bench_local_values(|((s0, s1), t)| black_box(s0.intersects_aligned(&s1, &t)));
    }

    #[divan::bench]
    fn xenocollide_2d(bencher: Bencher) {
        let mut rng = StdRng::seed_from_u64(1);

        bencher
            .counter(ItemsCount::from(1_u32))
            .with_inputs(|| {
                (
                    shapes_to_convex(create_cuboid_pair::<2, _>(&mut rng)),
                    create_offset_2d(&mut rng),
                )
            })
            .bench_local_values(|((c0, c1), (t, r))| black_box(c0.intersects_at(&c1, &t, &r)));
    }

    #[divan::bench]
    fn xenocollide_3d(bencher: Bencher) {
        let mut rng = StdRng::seed_from_u64(1);

        bencher
            .counter(ItemsCount::from(1_u32))
            .with_inputs(|| {
                (
                    shapes_to_convex(create_cuboid_pair::<3, _>(&mut rng)),
                    create_offset_3d(&mut rng),
                )
            })
            .bench_local_values(|((c0, c1), (t, r))| black_box(c0.intersects_at(&c1, &t, &r)));
    }
}

#[divan::bench_group()]
mod polytopes {

    use super::*;

    #[divan::bench(consts = NUM_VERTICES)]
    fn polygon_2d<const N: usize>(bencher: Bencher) {
        let mut rng = StdRng::seed_from_u64(1);

        bencher
            .counter(ItemsCount::from(1_u32))
            .with_inputs(|| {
                (
                    shapes_to_convex(create_polygon_pair::<N>()),
                    create_offset_2d(&mut rng),
                )
            })
            .bench_local_values(|((p0, p1), (t, r))| black_box(collide2d(&p0, &p1, &t, &r)));
    }

    #[divan::bench(consts = NUM_VERTICES)]
    fn dipyramid_3d<const N: usize>(bencher: Bencher) {
        let mut rng = StdRng::seed_from_u64(1);

        bencher
            .counter(ItemsCount::from(1_u32))
            .with_inputs(|| {
                (
                    shapes_to_convex(create_dipyramid_pair::<N, _>(&mut rng, 10.0)),
                    create_offset_3d(&mut rng),
                )
            })
            .bench_local_values(|((p0, p1), (t, r))| black_box(collide3d(&p0, &p1, &t, &r)));
    }
}

#[divan::bench_group]
mod simplex {
    use super::*;

    #[divan::bench]
    fn xenocollide_3d(bencher: Bencher) {
        let mut rng = StdRng::seed_from_u64(1);

        bencher
            .counter(ItemsCount::from(1_u32))
            .with_inputs(|| {
                (
                    shapes_to_convex(create_simplex_pair(&mut rng)),
                    create_offset_3d(&mut rng),
                )
            })
            .bench_local_values(|((t0, t1), (t, r))| black_box(collide3d(&t0, &t1, &t, &r)));
    }

    #[divan::bench]
    fn fast_3d(bencher: Bencher) {
        let mut rng = StdRng::seed_from_u64(1);

        bencher
            .counter(ItemsCount::from(1_u32))
            .with_inputs(|| (create_simplex_pair(&mut rng), create_offset_3d(&mut rng)))
            .bench_local_values(|((t0, t1), (t, r))| black_box(t0.intersects_at(&t1, &t, &r)));
    }
}

#[divan::bench_group]
mod ellipsoid {
    use super::*;

    #[divan::bench]
    fn xenocollide_2d(bencher: Bencher) {
        let mut rng = StdRng::seed_from_u64(1);

        bencher
            .counter(ItemsCount::from(1_u32))
            .with_inputs(|| {
                (
                    shapes_to_convex(create_ellipsoid_pair::<2, _>(&mut rng)),
                    create_offset_2d(&mut rng),
                )
            })
            .bench_local_values(|((t0, t1), (t, r))| black_box(collide2d(&t0, &t1, &t, &r)));
    }

    #[divan::bench]
    fn fast_2d(bencher: Bencher) {
        let mut rng = StdRng::seed_from_u64(1);

        bencher
            .counter(ItemsCount::from(1_u32))
            .with_inputs(|| {
                (
                    create_ellipsoid_pair::<2, _>(&mut rng),
                    create_offset_2d(&mut rng),
                )
            })
            .bench_local_values(|((e0, e1), (t, r))| black_box(e0.intersects_at(&e1, &t, &r)));
    }

    #[divan::bench]
    fn xenocollide_3d(bencher: Bencher) {
        let mut rng = StdRng::seed_from_u64(1);

        bencher
            .counter(ItemsCount::from(1_u32))
            .with_inputs(|| {
                (
                    shapes_to_convex(create_ellipsoid_pair::<3, _>(&mut rng)),
                    create_offset_3d(&mut rng),
                )
            })
            .bench_local_values(|((t0, t1), (t, r))| black_box(collide3d(&t0, &t1, &t, &r)));
    }

    #[divan::bench]
    fn fast_3d(bencher: Bencher) {
        let mut rng = StdRng::seed_from_u64(1);

        bencher
            .counter(ItemsCount::from(1_u32))
            .with_inputs(|| {
                (
                    create_ellipsoid_pair::<3, _>(&mut rng),
                    create_offset_3d(&mut rng),
                )
            })
            .bench_local_values(|((e0, e1), (t, r))| black_box(e0.intersects_at(&e1, &t, &r)));
    }
}

fn create_cylinder_pair<R: Rng>(rng: &mut R) -> (Cylinder, Cylinder) {
    (
        Cylinder {
            radius: rng
                .random_range(0.0..10.0)
                .try_into()
                .expect("test value is a positive real"),
            height: rng
                .random_range(0.0..10.0)
                .try_into()
                .expect("test value is a positive real"),
        },
        Cylinder {
            radius: rng
                .random_range(0.0..10.0)
                .try_into()
                .expect("test value is a positive real"),
            height: rng
                .random_range(0.0..10.0)
                .try_into()
                .expect("test value is a positive real"),
        },
    )
}

#[divan::bench_group]
mod cylinder {
    use super::*;

    #[divan::bench]
    fn infinite_3d(bencher: Bencher) {
        let mut rng = StdRng::seed_from_u64(1);

        bencher
            .counter(ItemsCount::from(1_u32))
            .with_inputs(|| (create_cylinder_pair(&mut rng), create_offset_3d(&mut rng)))
            .bench_local_values(|((c0, c1), (t, r))| {
                black_box(c0.intersects_at_infinite(&c1, &t, &r))
            });
    }
}

fn create_capsule_pair<R: Rng>(rng: &mut R) -> (Capsule<3>, Capsule<3>) {
    (
        Capsule {
            radius: rng
                .random_range(0.0..10.0)
                .try_into()
                .expect("test value is a positive real"),
            height: rng
                .random_range(0.0..10.0)
                .try_into()
                .expect("test value is a positive real"),
        },
        Capsule {
            radius: rng
                .random_range(0.0..10.0)
                .try_into()
                .expect("test value is a positive real"),
            height: rng
                .random_range(0.0..10.0)
                .try_into()
                .expect("test value is a positive real"),
        },
    )
}
#[divan::bench_group]
mod capsule {
    use super::*;

    #[divan::bench]
    fn fast_3d(bencher: Bencher) {
        let mut rng = StdRng::seed_from_u64(1);

        bencher
            .counter(ItemsCount::from(1_u32))
            .with_inputs(|| (create_capsule_pair(&mut rng), create_offset_3d(&mut rng)))
            .bench_local_values(|((c0, c1), (t, r))| black_box(c0.intersects_at(&c1, &t, &r)));
    }

    #[divan::bench]
    fn xenocollide_3d(bencher: Bencher) {
        let mut rng = StdRng::seed_from_u64(1);

        bencher
            .counter(ItemsCount::from(1_u32))
            .with_inputs(|| {
                (
                    shapes_to_convex(create_capsule_pair(&mut rng)),
                    create_offset_3d(&mut rng),
                )
            })
            .bench_local_values(|((c0, c1), (t, r))| black_box(c0.intersects_at(&c1, &t, &r)));
    }
}
