// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![expect(
    clippy::missing_docs_in_private_items,
    reason = "benches don't need public documentation"
)]
#![expect(clippy::wildcard_imports, reason = "simplifies code")]
#![expect(clippy::cast_possible_truncation, reason = "N is small")]

//! Benchmark 2D convex hull (Graham scan)

use divan::{self, Bencher, black_box, counter::ItemsCount};
use hoomd_geometry::hull_2d_grahamscan;
use hoomd_vector::Cartesian;
use rand::{RngExt, SeedableRng, rngs::StdRng};

fn main() {
    divan::main();
}

const NUM_POINTS: &[usize] = &[10, 100, 1_000, 10_000];

/// Create random points in the unit square
fn create_random_points(n: usize, rng: &mut StdRng) -> Vec<Cartesian<2>> {
    (0..n)
        .map(|_| Cartesian::from([rng.random::<f64>(), rng.random::<f64>()]))
        .collect()
}

/// Create points uniformly distributed on a circle
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

#[divan::bench_group]
mod random {
    use super::*;

    #[divan::bench(consts = NUM_POINTS)]
    fn unit_square<const N: usize>(bencher: Bencher) {
        let mut rng = StdRng::seed_from_u64(42);

        bencher
            .counter(ItemsCount::from(N as u32))
            .with_inputs(|| create_random_points(N, &mut rng))
            .bench_local_values(|mut pts| black_box(hull_2d_grahamscan(&mut pts)));
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
            .bench_local_values(|mut pts| black_box(hull_2d_grahamscan(&mut pts)));
    }

    #[divan::bench(consts = NUM_POINTS)]
    fn square_boundary<const N: usize>(bencher: Bencher) {
        bencher
            .counter(ItemsCount::from(N as u32))
            .with_inputs(|| create_square_boundary(N / 4 + 1))
            .bench_local_values(|mut pts| black_box(hull_2d_grahamscan(&mut pts)));
    }
}
