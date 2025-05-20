// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![expect(
    clippy::missing_docs_in_private_items,
    reason = "benches don't need public documentation"
)]
#![expect(
    clippy::expect_used,
    reason = "benches can use expect without individual reasons"
)]

/*! Benchmark `LennardJones` */

use divan::counter::ItemsCount;
use divan::{self, Bencher, black_box};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::f64::consts::PI;

use hoomd_interaction::pairwise::angular_mask::Patch;
use hoomd_interaction::pairwise::{AngularMask, AnisotropicEnergy, LennardJones};
use hoomd_vector::{Angle, Cartesian, Versor};

fn main() {
    divan::main();
}

#[divan::bench]
fn energy_2d(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    let epsilon: f64 = rng.random();
    let sigma: f64 = rng.random();
    let lj: LennardJones = LennardJones { epsilon, sigma };

    let masks = [
        Patch {
            director: [1.0, 0.0]
                .try_into()
                .expect("hard-coded vector should have non-zero length"),
            cos_delta: (PI / 8.0).cos(),
        },
        Patch {
            director: [-1.0, 0.0]
                .try_into()
                .expect("hard-coded vector should have non-zero length"),
            cos_delta: (PI / 16.0).cos(),
        },
        Patch {
            director: [0.0, 1.0]
                .try_into()
                .expect("hard-coded vector should have non-zero length"),
            cos_delta: (PI / 16.0).cos(),
        },
        Patch {
            director: [0.0, -1.0]
                .try_into()
                .expect("hard-coded vector should have non-zero length"),
            cos_delta: (PI / 16.0).cos(),
        },
    ];

    let angular_mask = AngularMask::new(lj, masks, masks);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| -> (Cartesian<2>, Angle) {
            (rng.random::<Cartesian<2>>(), rng.random::<Angle>())
        })
        .bench_local_values(|(r_ij, angle)| black_box(angular_mask.energy(&r_ij, &angle)));
}

#[divan::bench]
fn energy_3d(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    let epsilon: f64 = rng.random();
    let sigma: f64 = rng.random();
    let lj: LennardJones = LennardJones { epsilon, sigma };

    let masks = [
        Patch { 
            director: [1.0, 0.0, 0.0]
                .try_into()
                .expect("hard-coded vector should have non-zero length"),
            cos_delta: (PI / 16.0).cos(),
        },
        Patch { 
            director: [-1.0, 0.0, 0.0]
                .try_into()
                .expect("hard-coded vector should have non-zero length"),
            cos_delta: (PI / 16.0).cos(),
        },
        Patch { 
            director: [0.0, 1.0, 0.0]
                .try_into()
                .expect("hard-coded vector should have non-zero length"),
            cos_delta: (PI / 16.0).cos(),
        },
        Patch { 
            director: [0.0, -1.0, 0.0]
                .try_into()
                .expect("hard-coded vector should have non-zero length"),
            cos_delta: (PI / 16.0).cos(),
        },
    ];

    let angular_mask = AngularMask::new(lj, masks, masks);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| -> (Cartesian<3>, Versor) {
            (rng.random::<Cartesian<3>>(), rng.random::<Versor>())
        })
        .bench_local_values(|(r_ij, angle)| black_box(angular_mask.energy(&r_ij, &angle)));
}
