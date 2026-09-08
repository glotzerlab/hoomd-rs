// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![expect(
    clippy::missing_docs_in_private_items,
    reason = "benches don't need public documentation"
)]
#![expect(clippy::wildcard_imports, reason = "simplifies code")]

//! Benchmark overlaps

use divan::{self, Bencher, black_box, counter::ItemsCount};
use hoomd_geometry::SupportMapping;
use hoomd_geometry::{
    Convex, IntersectsAt,
    shape::{
        Capsule, ConvexPolytope, Cylinder, Hypercuboid, Hyperellipsoid, Hypersphere, Simplex3,
    },
    xenocollide::{collide2d, collide3d, collide4d},
};
use hoomd_vector::{Angle, Cartesian, InnerProduct, Rotate, RotationMatrix, Versor};
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};

fn main() {
    divan::main();
}
/// A direction uniform on the unit (N-1)-sphere.
#[inline]
fn unit_direction<const N: usize, R: Rng>(rng: &mut R) -> Cartesian<N> {
    let (unit, _) = rng
        .random::<Cartesian<N>>()
        .to_unit()
        .expect("random vector is non-zero");
    *unit.get()
}

fn shapes_to_convex<S>(tup: (S, S)) -> (Convex<S>, Convex<S>) {
    (Convex(tup.0), Convex(tup.1))
}

fn create_dipyramid(n: usize) -> ConvexPolytope<3> {
    let base = ConvexPolytope::<2>::regular(n);
    ConvexPolytope::<3>::with_vertices(
        base.vertices()
            .iter()
            .map(|x| Cartesian::from([x[0], x[1], 0.0]))
            .chain([[0.0, 0.0, 0.5].into(), [0.0, 0.0, -0.5].into()]),
    )
    .expect("constructed polytope should be valid")
}

fn sample_offset_angle<R: Rng>(rng: &mut R, r_min: f64, r_max: f64) -> (Cartesian<2>, Angle) {
    (sample_offset(rng, r_min, r_max), rng.random())
}

fn sample_offset_versor<R: Rng>(rng: &mut R, r_min: f64, r_max: f64) -> (Cartesian<3>, Versor) {
    (sample_offset(rng, r_min, r_max), rng.random())
}

fn sample_offset<const N: usize, R: Rng>(rng: &mut R, r_min: f64, r_max: f64) -> Cartesian<N> {
    loop {
        let v = rng.random::<Cartesian<N>>() * r_max;
        let v_norm = v.norm();
        if v_norm >= r_min && v_norm < r_max {
            break v;
        }
    }
}
fn create_offset_3d<R: Rng>(rng: &mut R) -> (Cartesian<3>, Versor) {
    (rng.random::<Cartesian<3>>() * 10.0, rng.random())
}

const DIMENSIONS: &[usize] = &[2, 3, 4];
const NUM_VERTICES: &[usize] = &[3, 4, 8, 16, 50];
const DIPYRAMID_VERTICES: &[usize] = &[5, 6, 10, 18, 52];

#[divan::bench_group]
mod sphere {
    use super::*;

    #[divan::bench(consts = DIMENSIONS)]
    fn fast_nd<const N: usize>(bencher: Bencher) {
        let mut rng = StdRng::seed_from_u64(1);
        let shape = Hypersphere::<N> {
            radius: 0.5.try_into().expect("hard-coded value is positive"),
        };

        bencher
            .counter(ItemsCount::from(1_u32))
            .with_inputs(|| sample_offset::<N, _>(&mut rng, 0.9, 1.0))
            .bench_local_values(|t| black_box(shape.intersects(&shape, &t)));
    }

    #[divan::bench]
    fn xenocollide_2d(bencher: Bencher) {
        let mut rng = StdRng::seed_from_u64(1);
        let shape = Convex(Hypersphere::<2> {
            radius: 0.5.try_into().expect("hard-coded value is positive"),
        });

        bencher
            .counter(ItemsCount::from(1_u32))
            .with_inputs(|| sample_offset_angle(&mut rng, 0.9, 1.0))
            .bench_local_values(|(t, r)| black_box(collide2d(&shape, &shape, &t, &r)));
    }

    #[divan::bench(sample_size = 10_000)]
    fn xenocollide_3d(bencher: Bencher) {
        let mut rng = StdRng::seed_from_u64(1);
        let shape = Convex(Hypersphere::<3> {
            radius: 0.5.try_into().expect("hard-coded value is positive"),
        });

        bencher
            .counter(ItemsCount::from(1_u32))
            .with_inputs(|| sample_offset_versor(&mut rng, 0.9, 1.0))
            .bench_local_values(|(t, r)| black_box(collide3d(&shape, &shape, &t, &r)));
    }
}

#[divan::bench_group]
mod cuboid {
    use super::*;

    #[divan::bench(consts = DIMENSIONS)]
    fn aligned_nd<const N: usize>(bencher: Bencher) {
        let mut rng = StdRng::seed_from_u64(1);
        let shape = Hypercuboid::with_equal_edges(
            0.5.try_into().expect("hard coded value should be positive"),
        );

        bencher
            .counter(ItemsCount::from(1_u32))
            .with_inputs(|| sample_offset::<N, _>(&mut rng, 0.8, 1.5))
            .bench_local_values(|t| black_box(shape.intersects_aligned(&shape, &t)));
    }

    #[divan::bench]
    fn xenocollide_2d(bencher: Bencher) {
        let mut rng = StdRng::seed_from_u64(1);
        let shape = Convex(Hypercuboid::with_equal_edges(
            0.5.try_into().expect("hard coded value should be positive"),
        ));

        bencher
            .counter(ItemsCount::from(1_u32))
            .with_inputs(|| sample_offset_angle(&mut rng, 0.8, 1.5))
            .bench_local_values(|(t, r)| black_box(collide2d(&shape, &shape, &t, &r)));
    }

    #[divan::bench]
    fn xenocollide_3d(bencher: Bencher) {
        let mut rng = StdRng::seed_from_u64(1);
        let shape = Convex(Hypercuboid::with_equal_edges(
            0.5.try_into().expect("hard coded value should be positive"),
        ));

        bencher
            .counter(ItemsCount::from(1_u32))
            .with_inputs(|| sample_offset_versor(&mut rng, 0.8, 1.5))
            .bench_local_values(|(t, r)| black_box(collide3d(&shape, &shape, &t, &r)));
    }
}

/// Get a loose upper bound for the separation distance of `a` and `b` along `u`.
fn contact_bound<S, Q, const N: usize>(a: &S, b: &S, orientation: Q, u: &Cartesian<N>) -> f64
where
    S: SupportMapping<Cartesian<N>>,
    Q: Copy,
    RotationMatrix<N>: From<Q>,
{
    let q = RotationMatrix::<N>::from(orientation);
    let q_inv = q.inverted();
    let b_support = q.rotate(&b.support_mapping(&q_inv.rotate(&(-*u))));
    a.support_mapping(u).dot(u) - b_support.dot(u)
}

/// Relative contact distance for shallow penetrations and near misses.
const SPAN: (f64, f64) = (-1.0e-3, 1.0e-3);

/// Bisection to find the last-overlap point along a ray.
fn bisect(hi: f64, overlaps: impl Fn(f64) -> bool) -> f64 {
    let mut lo = 0.0_f64;
    let mut hi = hi;
    for _ in 0..32 {
        let mid = lo.midpoint(hi);
        if overlaps(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Generate one 2D near-boundary configuration
fn boundary_input<S>(inner: &S, shape: &Convex<S>, rng: &mut StdRng) -> (Cartesian<2>, Angle)
where
    S: SupportMapping<Cartesian<2>> + Clone,
{
    let u = unit_direction(rng);
    let angle: Angle = rng.random();
    let delta = SPAN.0 + rng.random::<f64>() * (SPAN.1 - SPAN.0);
    let bound = contact_bound(inner, &inner.clone(), angle, &u);
    let contact = bisect(bound, |t| collide2d(shape, shape, &(u * t), &angle));
    (u * (contact * (1.0 + delta)), angle)
}

/// Generate one 3D near-boundary configuration
fn boundary_input_3d<S>(inner: &S, shape: &Convex<S>, rng: &mut StdRng) -> (Cartesian<3>, Versor)
where
    S: SupportMapping<Cartesian<3>> + Clone,
{
    let u = unit_direction(rng);
    let versor: Versor = rng.random();
    let delta = SPAN.0 + rng.random::<f64>() * (SPAN.1 - SPAN.0);
    let bound = contact_bound(inner, &inner.clone(), versor, &u);
    let contact = bisect(bound, |t| collide3d(shape, shape, &(u * t), &versor));
    (u * (contact * (1.0 + delta)), versor)
}

/// Generate one 4D near-boundary configuration
fn boundary_input_4d<S>(
    inner: &S,
    shape: &Convex<S>,
    rng: &mut StdRng,
) -> (Cartesian<4>, RotationMatrix<4>)
where
    S: SupportMapping<Cartesian<4>> + Clone,
{
    let q = RotationMatrix::<4>::default(); // TODO: orient! (PR #346 DoubleVersor)
    let u = unit_direction(rng);
    let delta = SPAN.0 + rng.random::<f64>() * (SPAN.1 - SPAN.0);
    let bound = contact_bound(inner, &inner.clone(), q, &u);
    let contact = bisect(bound, |t| collide4d(shape, shape, &(u * t), &q));
    (u * (contact * (1.0 + delta)), q)
}

#[divan::bench_group()]
mod polytopes {
    use hoomd_geometry::shape::ConvexSurfaceMesh2d;

    use super::*;

    #[divan::bench(consts = NUM_VERTICES)]
    fn polygon_2d<const N: usize>(bencher: Bencher) {
        let shape = Convex(ConvexPolytope::<2>::regular(N));
        let mut rng = StdRng::seed_from_u64(1);

        bencher
            .counter(ItemsCount::from(1_u32))
            .with_inputs(|| boundary_input(&shape.0, &shape, &mut rng))
            .bench_local_values(|(t, r)| black_box(collide2d(&shape, &shape, &t, &r)));
    }

    #[divan::bench(consts = NUM_VERTICES)]
    fn polygon_2d_fast<const N: usize>(bencher: Bencher) {
        let shape = ConvexSurfaceMesh2d::try_from(ConvexPolytope::<2>::regular(N))
            .expect("regular polygon should be a valid surface mesh");
        let proxy = Convex(ConvexPolytope::<2>::regular(N));
        let mut rng = StdRng::seed_from_u64(1);

        bencher
            .counter(ItemsCount::from(1_u32))
            .with_inputs(|| boundary_input(&proxy.0, &proxy, &mut rng))
            .bench_local_values(|(t, r)| black_box(shape.intersects_at(&shape, &t, &r)));
    }

    #[divan::bench(consts = DIPYRAMID_VERTICES)]
    fn dipyramid_3d<const N: usize>(bencher: Bencher) {
        let shape = Convex(create_dipyramid(N));
        let mut rng = StdRng::seed_from_u64(1);

        bencher
            .counter(ItemsCount::from(1_u32))
            .with_inputs(|| boundary_input_3d(&shape.0, &shape, &mut rng))
            .bench_local_values(|(t, r)| black_box(collide3d(&shape, &shape, &t, &r)));
    }

    #[divan::bench]
    fn hypercube_4d(bencher: Bencher) {
        let shape = Convex(ConvexPolytope::<4, 16>::hypercube());
        let mut rng = StdRng::seed_from_u64(1);

        bencher
            .counter(ItemsCount::from(1_u32))
            .with_inputs(|| boundary_input_4d(&shape.0, &shape, &mut rng))
            .bench_local_values(|(t, q)| black_box(collide4d(&shape, &shape, &t, &q)));
    }
}

#[divan::bench_group]
mod simplex {
    use super::*;

    #[divan::bench]
    fn xenocollide_3d(bencher: Bencher) {
        let mut rng = StdRng::seed_from_u64(1);
        let shape = Convex(Simplex3::from([
            Cartesian::from([-0.5, -0.5, -0.5]),
            [-0.5, -0.5, 0.5].into(),
            [-0.5, 0.5, 0.5].into(),
            [0.5, 0.5, 0.5].into(),
        ]));

        bencher
            .counter(ItemsCount::from(1_u32))
            .with_inputs(|| sample_offset_versor(&mut rng, 0.8, 1.5))
            .bench_local_values(|(t, r)| black_box(collide3d(&shape, &shape, &t, &r)));
    }

    #[divan::bench]
    fn fast_3d(bencher: Bencher) {
        let mut rng = StdRng::seed_from_u64(1);
        let shape = Simplex3::from([
            Cartesian::from([-0.5, -0.5, -0.5]),
            [-0.5, -0.5, 0.5].into(),
            [-0.5, 0.5, 0.5].into(),
            [0.5, 0.5, 0.5].into(),
        ]);

        bencher
            .counter(ItemsCount::from(1_u32))
            .with_inputs(|| sample_offset_versor(&mut rng, 0.8, 1.5))
            .bench_local_values(|(t, r)| black_box(shape.intersects_at(&shape, &t, &r)));
    }
}

#[divan::bench_group]
mod ellipsoid {
    use super::*;

    #[divan::bench]
    fn xenocollide_2d(bencher: Bencher) {
        let mut rng = StdRng::seed_from_u64(1);
        let shape = Convex(Hyperellipsoid::<2>::with_semi_axes([
            0.5.try_into().expect("hard-coded value should be positive"),
            0.25.try_into()
                .expect("hard-coded value should be positive"),
        ]));

        bencher
            .counter(ItemsCount::from(1_u32))
            .with_inputs(|| sample_offset_angle(&mut rng, 0.4, 1.0))
            .bench_local_values(|(t, r)| black_box(collide2d(&shape, &shape, &t, &r)));
    }

    #[divan::bench]
    fn fast_2d(bencher: Bencher) {
        let mut rng = StdRng::seed_from_u64(1);
        let shape = Hyperellipsoid::<2>::with_semi_axes([
            0.5.try_into().expect("hard-coded value should be positive"),
            0.25.try_into()
                .expect("hard-coded value should be positive"),
        ]);

        bencher
            .counter(ItemsCount::from(1_u32))
            .with_inputs(|| sample_offset_angle(&mut rng, 0.4, 1.0))
            .bench_local_values(|(t, r)| black_box(shape.intersects_at(&shape, &t, &r)));
    }

    #[divan::bench]
    fn xenocollide_3d(bencher: Bencher) {
        let mut rng = StdRng::seed_from_u64(1);
        let shape = Hyperellipsoid::<3>::with_semi_axes([
            0.5.try_into().expect("hard-coded value should be positive"),
            0.25.try_into()
                .expect("hard-coded value should be positive"),
            0.25.try_into()
                .expect("hard-coded value should be positive"),
        ]);

        bencher
            .counter(ItemsCount::from(1_u32))
            .with_inputs(|| sample_offset_versor(&mut rng, 0.4, 1.0))
            .bench_local_values(|(t, r)| black_box(collide3d(&shape, &shape, &t, &r)));
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

#[divan::bench_group]
mod support_mapping {
    use super::*;
    use hoomd_geometry::SupportMapping;

    #[divan::bench(consts = DIPYRAMID_VERTICES)]
    fn dipyramid<const N: usize>(bencher: Bencher) {
        let shape = create_dipyramid(N);
        let mut rng = StdRng::seed_from_u64(1);
        let directions: Vec<Cartesian<3>> = (0..1024).map(|_| rng.random()).collect();

        bencher
            .counter(ItemsCount::from(1024_u32))
            .with_inputs(|| directions.clone())
            .bench_local_values(|dirs| {
                for d in black_box(dirs) {
                    black_box(shape.support_mapping(&d));
                }
            });
    }
}
