// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![expect(
    clippy::missing_docs_in_private_items,
    reason = "benches don't need public documentation"
)]
#![expect(clippy::unwrap_used, reason = "benches can use unwrap where needed")]

//! Benchmark hyperbolic overlaps

use divan::{self, Bencher, black_box, counter::ItemsCount};
use hoomd_geometry::hyperbolic_overlap::{HyperbolicConvexPolytope, SeparatingPlanes};
use hoomd_manifold::{Hyperbolic, HyperbolicDisk, Minkowski};
use rand::{Rng, distr::Distribution, rngs::StdRng, SeedableRng};

use hoomd_vector::Angle;

fn main() {
    divan::main();
}

const RHO : f64 = 1.0; 
fn create_2d_oriented_point_pair<R: Rng>(rng: &mut R) -> (Hyperbolic<3>, Angle, Hyperbolic<3>, Angle) {
    let initial_spacing = 1.4;
    let sample_disk = HyperbolicDisk {
        disk_radius: initial_spacing.try_into().expect("positive number"),
        point: Hyperbolic::<3>::from_minkowski_coordinates(
            Minkowski::from([
                0.00001,
                0.00001,
                f64::sqrt(2.0 * (0.00001_f64).powi(2) + RHO.powi(2)),
            ]),
            RHO,
            ),
    };
    (
        Hyperbolic::from_minkowski_coordinates(
                    *sample_disk.sample(rng).point(),
                    RHO,
                ),
        rng.random(),
        Hyperbolic::from_minkowski_coordinates(
                    *sample_disk.sample(rng).point(),
                    RHO,
                ),
        rng.random()
    )
}

const NUM_VERTICES: &[usize] = &[3, 4, 8, 16, 64];

#[divan::bench(consts = NUM_VERTICES, )]
fn hyperbolic_polygon_overlap<const N: usize>(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);
    let n_gon = HyperbolicConvexPolytope::<3>::regular(N, 0.3, RHO);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| {
            create_2d_oriented_point_pair(&mut rng)
        })
        .bench_local_values(|(p0,r0, p1, r1)| black_box(n_gon.intersects_at(&p0, &r0, &p1, &r1)));
}