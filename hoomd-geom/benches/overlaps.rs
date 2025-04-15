// Profile overlap checks between various geometric primitives.

#![allow(missing_docs)]
#![allow(clippy::missing_docs_in_private_items)]

/*! Benchmark overlaps*/

use divan::counter::ItemsCount;
use divan::{self, black_box, Bencher};
use hoomd_geom::{xenocollide::collide2d, IntersectsAt, Sphere};
use rand::distributions::Uniform;
use rand::{rngs::StdRng, Rng, SeedableRng};

use hoomd_vector::{Angle, Cartesian, RotationMatrix};

fn main() {
    divan::main();
}

fn create_sphere_pair<const N: usize, R: Rng>(rng: &mut R) -> (Sphere<N>, Sphere<N>) {
    (
        Sphere::from(rng.gen_range(0f64..100f64)),
        Sphere::from(rng.gen_range(0f64..100f64)),
    )
}

fn create_offset_2d<R: Rng>(rng: &mut R) -> (Cartesian<2>, Angle) {
    (
        rng.gen::<Cartesian<2>>() * 100f64,
        Angle::from(rng.gen_range((-2.0 * std::f64::consts::PI)..(2.0 * std::f64::consts::PI))),
    )
}

const DIMENSIONS: &[usize] = &[1, 2, 3, 4];

#[divan::bench]
fn sphere_overlap_fast_2d(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| {
            (
                create_sphere_pair::<2, _>(&mut rng),
                create_offset_2d(&mut rng),
            )
        })
        .bench_local_values(|((s0, s1), (t, r))| black_box(s0.intersects_at(&s1, &t, &r)));
}
#[divan::bench()]
fn sphere_overlap_xenocollide_2d(bencher: Bencher) {
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
