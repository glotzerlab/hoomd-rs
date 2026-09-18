// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![expect(
    clippy::missing_docs_in_private_items,
    reason = "benches don't need public documentation"
)]
#![expect(clippy::wildcard_imports, reason = "simplifies code")]
#![expect(clippy::cast_possible_truncation, reason = "N is small")]

//! Benchmark 2D and 3D convex hulls.

use divan::{self, Bencher, black_box, counter::ItemsCount};
use hoomd_geometry::{
    ConvexHull, platonic,
    shape::{Hypercuboid, Hypersphere},
};
use hoomd_vector::{Cartesian, Rotate, Versor};
use rand::{RngExt, SeedableRng, distr::Distribution, rngs::StdRng};

fn main() {
    divan::main();
}

const NUM_POINTS: &[usize] = &[10, 100, 1_000];

/// Create random points in the unit N-cube
fn create_random_points<const N: usize>(n: usize, rng: &mut StdRng) -> Vec<Cartesian<N>> {
    let cube = Hypercuboid::<N>::with_equal_edges(1.0.try_into().expect("1.0 is positive"));
    (0..n).map(|_| cube.sample(rng)).collect()
}

/// Create random points uniformly distributed *in* the unit N-ball
fn create_ball_points<const N: usize>(n: usize, rng: &mut StdRng) -> Vec<Cartesian<N>> {
    let sphere = Hypersphere::<N>::with_radius(1.0.try_into().expect("1.0 is positive"));
    (0..n).map(|_| sphere.sample(rng)).collect()
}

/// Create points uniformly distributed *on* a circle
fn create_circle_points(n: usize) -> Vec<Cartesian<2>> {
    (0..n)
        .map(|i| {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            Cartesian::from([angle.cos(), angle.sin()])
        })
        .collect()
}

/// Create points densely on a square boundary
fn create_square_boundary(n_per_edge: usize) -> Vec<Cartesian<2>> {
    let mut pts = Vec::with_capacity(4 * n_per_edge);
    for i in 0..n_per_edge {
        let t = i as f64 / (n_per_edge - 1) as f64;
        pts.push(Cartesian::from([t, 0.0])); // bottom
        pts.push(Cartesian::from([1.0, t])); // right
        pts.push(Cartesian::from([t, 1.0])); // top
        pts.push(Cartesian::from([0.0, t])); // left
    }
    pts
}

/// Create points on the surface of the unit sphere (Fibonacci lattice)
fn create_sphere_surface_points(n: usize) -> Vec<Cartesian<3>> {
    let golden_angle = std::f64::consts::PI * (5.0_f64.sqrt() - 1.0);
    (0..n)
        .map(|i| {
            let z = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
            let r = (1.0 - z * z).sqrt();
            let theta = golden_angle * i as f64;
            Cartesian::from([r * theta.cos(), r * theta.sin(), z])
        })
        .collect()
}

/// Create points densely on the surface of a cube with edge length 2
fn create_cube_surface_points(n_per_edge: usize) -> Vec<Cartesian<3>> {
    let mut pts = Vec::with_capacity(6 * n_per_edge * n_per_edge);
    let t: Vec<f64> = (0..n_per_edge)
        .map(|i| 2.0 * i as f64 / (n_per_edge - 1) as f64 - 1.0)
        .collect();
    for &a in &t {
        for &b in &t {
            pts.push(Cartesian::from([a, b, 1.0]));
            pts.push(Cartesian::from([a, b, -1.0]));
            pts.push(Cartesian::from([a, 1.0, b]));
            pts.push(Cartesian::from([a, -1.0, b]));
            pts.push(Cartesian::from([1.0, a, b]));
            pts.push(Cartesian::from([-1.0, a, b]));
        }
    }
    pts
}

/// Benchmark a hull over a fixed point set
fn bench_fixed_points(points: &[Cartesian<3>], bencher: Bencher) {
    let points: Vec<Cartesian<3>> = points.to_vec();
    bencher
        .counter(ItemsCount::from(points.len() as u32))
        .with_inputs(|| points.clone())
        .bench_local_values(|pts| black_box(Cartesian::<3>::convex_hull(&pts)));
}

#[divan::bench_group]
mod random {
    use super::*;

    #[divan::bench(consts = NUM_POINTS)]
    fn unit_square<const N: usize>(bencher: Bencher) {
        let mut rng = StdRng::seed_from_u64(42);

        bencher
            .counter(ItemsCount::from(N as u32))
            .with_inputs(|| create_random_points::<2>(N, &mut rng))
            .bench_local_values(|pts| black_box(Cartesian::<2>::convex_hull(&pts)));
    }

    #[divan::bench(consts = NUM_POINTS)]
    fn unit_cube<const N: usize>(bencher: Bencher) {
        let mut rng = StdRng::seed_from_u64(42);

        bencher
            .counter(ItemsCount::from(N as u32))
            .with_inputs(|| create_random_points::<3>(N, &mut rng))
            .bench_local_values(|pts| black_box(Cartesian::<3>::convex_hull(&pts)));
    }

    #[divan::bench(consts = NUM_POINTS)]
    fn unit_disc<const N: usize>(bencher: Bencher) {
        let mut rng = StdRng::seed_from_u64(42);

        bencher
            .counter(ItemsCount::from(N as u32))
            .with_inputs(|| create_ball_points::<2>(N, &mut rng))
            .bench_local_values(|pts| black_box(Cartesian::<2>::convex_hull(&pts)));
    }

    #[divan::bench(consts = NUM_POINTS)]
    fn unit_ball<const N: usize>(bencher: Bencher) {
        let mut rng = StdRng::seed_from_u64(42);

        bencher
            .counter(ItemsCount::from(N as u32))
            .with_inputs(|| create_ball_points::<3>(N, &mut rng))
            .bench_local_values(|pts| black_box(Cartesian::<3>::convex_hull(&pts)));
    }
}

#[divan::bench_group]
mod boundaries {
    use super::*;

    #[divan::bench(consts = NUM_POINTS)]
    fn circle_boundary<const N: usize>(bencher: Bencher) {
        bencher
            .counter(ItemsCount::from(N as u32))
            .with_inputs(|| create_circle_points(N))
            .bench_local_values(|pts| black_box(Cartesian::<2>::convex_hull(&pts)));
    }

    #[divan::bench(consts = NUM_POINTS)]
    fn square_boundary<const N: usize>(bencher: Bencher) {
        bencher
            .counter(ItemsCount::from(N as u32))
            .with_inputs(|| create_square_boundary(N / 4 + 1))
            .bench_local_values(|pts| black_box(Cartesian::<2>::convex_hull(&pts)));
    }

    #[divan::bench(consts = NUM_POINTS)]
    fn sphere_boundary<const N: usize>(bencher: Bencher) {
        bencher
            .counter(ItemsCount::from(N as u32))
            .with_inputs(|| create_sphere_surface_points(N))
            .bench_local_values(|pts| black_box(Cartesian::<3>::convex_hull(&pts)));
    }

    #[divan::bench(consts = NUM_POINTS)]
    fn cube_boundary<const N: usize>(bencher: Bencher) {
        // The cube surface holds 6 k^2 points for k per edge: choose k so the total ~ N
        #[allow(
            clippy::cast_sign_loss,
            reason = "the square root of a positive count is positive"
        )]
        let n_per_edge = (f64::sqrt(N as f64 / 6.0) as usize).max(2);
        bencher
            .counter(ItemsCount::from((6 * n_per_edge * n_per_edge) as u32))
            .with_inputs(|| create_cube_surface_points(n_per_edge))
            .bench_local_values(|pts| black_box(Cartesian::<3>::convex_hull(&pts)));
    }
}

#[divan::bench_group]
mod platonic_solids {
    use super::*;

    #[divan::bench]
    fn tetrahedron(bencher: Bencher) {
        bench_fixed_points(&platonic::tetrahedron(), bencher);
    }

    #[divan::bench]
    fn octahedron(bencher: Bencher) {
        bench_fixed_points(&platonic::octahedron(), bencher);
    }

    #[divan::bench]
    fn cube(bencher: Bencher) {
        bench_fixed_points(&platonic::cube(), bencher);
    }

    #[divan::bench]
    fn icosahedron(bencher: Bencher) {
        bench_fixed_points(&platonic::icosahedron(), bencher);
    }

    #[divan::bench]
    fn dodecahedron(bencher: Bencher) {
        bench_fixed_points(&platonic::dodecahedron(), bencher);
    }

    #[divan::bench]
    fn rotated_dodecahedron(bencher: Bencher) {
        let mut rng = StdRng::seed_from_u64(42);
        let versor: Versor = rng.random();
        let rotated: Vec<Cartesian<3>> = platonic::dodecahedron()
            .iter()
            .map(|p| versor.rotate(p))
            .collect();
        bench_fixed_points(&rotated, bencher);
    }
}
