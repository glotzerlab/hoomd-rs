//! ...
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
use rsph::spherical_harmonic;

/// Generate a tuple of uniform random values in [0, 1].
#[inline]
fn create_random_tuple<R: Rng>(rng: &mut R) -> (f64, f64, f64) {
    (rng.random(), rng.random(), rng.random())
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
    let mut rng = StdRng::seed_from_u64(1);
    bencher
        .with_inputs(|| create_random_tuple::<StdRng>(&mut rng))
        .bench_local_values(|(x, y, z)| spherical_harmonic::<L>(x, y, z));
}

/// Spherical-harmonic core for l = 6. Returns [p_6^m * Q_6^m] for m = 0..6.
#[inline(always)]
#[must_use]
pub fn sph_core_l6(x: f64, y: f64, z: f64) -> [f64; 7] {
    let x2 = x * x;
    let y2 = y * y;
    let z2 = z * z;
    let r2 = x2 + y2;

    let z4 = z2 * z2;
    let r4 = r2 * r2;
    let rz2 = r2 * z2;

    let m6 = 0.683_184_105_191_914_3;

    let m5 = 2.366_619_162_231_752 * z;

    let m4 = 5.045_649_007_287_242 * z2 - 0.504_564_900_728_724_2 * r2;

    let m3 = (7.369_642_076_119_389 * z2 - 2.763_615_778_544_770_6 * r2) * z;

    let m2 = 7.369_642_076_119_388 * (z4 - rz2) + 0.460_602_629_757_461_75 * r4;

    let m1 = (4.660_970_900_149_851 * z4 - 11.652_427_250_374_629 * rz2
        + 2.913_106_812_593_657_2 * r4)
        * z;

    let m0 = (1.017_107_236_282_054_8 * z2 - 7.628_304_272_115_411 * r2) * z4
        + (5.721_228_204_086_558 * z2 - 0.317_846_011_338_142_1 * r2) * r4;

    [m0, m1, m2, m3, m4, m5, m6]
}

/// Measure per-point performance for hardcoded l=6.
#[divan::bench(sample_count = 1_000, sample_size = 10_000)]
fn l6_real_hardcoded(bencher: divan::Bencher<'_, '_>) {
    let mut rng = StdRng::seed_from_u64(1);
    bencher
        .with_inputs(|| create_random_tuple::<StdRng>(&mut rng))
        .bench_local_values(|(x, y, z)| sph_core_l6(x, y, z));
}
