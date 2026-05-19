use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};
use rsph::spherical_harmonic;

/// Generate a tuple of uniform random values in [0, 1].
#[inline]
fn create_random_tuple<R: Rng>(rng: &mut R) -> (f64, f64, f64) {
    (rng.random(), rng.random(), rng.random())
}
fn main() {
    divan::main();
}

#[divan::bench(
    args = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    sample_count = 1_000,
    sample_size = 10_000,
)]
fn bench_rsph(bencher: divan::Bencher<'_, '_>, l: usize) {
    let mut rng = StdRng::seed_from_u64(1);
    bencher
        .with_inputs(|| create_random_tuple::<StdRng>(&mut rng))
        .bench_local_values(|(x, y, z)| spherical_harmonic(l, x, y, z));
}
