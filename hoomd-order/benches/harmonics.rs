// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! ...
use hoomd_order::SphericalHarmonic;
use hoomd_vector::{Cartesian, InnerProduct};
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};

/// Generate a random point on the unit sphere.
#[inline]
fn random_unit_point<R: Rng>(rng: &mut R) -> Cartesian<3> {
    let (x, y, z) = (rng.random(), rng.random(), rng.random());
    Cartesian::from([x, y, z])
}
fn main() {
    divan::main();
}

/// Measure per-point performance at each l.
#[divan::bench(
    consts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    sample_count = 1_000,
    sample_size = 10_000,
)]
fn recurrence<const L: usize>(bencher: divan::Bencher<'_, '_>) {
    let sh = SphericalHarmonic::<L>::new();
    let mut rng = StdRng::seed_from_u64(1);
    bencher
        .with_inputs(|| {
            let point = random_unit_point::<StdRng>(&mut rng);
            (point.to_unit().expect("non-zero point")).0
        })
        .bench_local_values(|xyz| sh.eval(xyz));
}
