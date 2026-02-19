// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use core::convert::Infallible;

use rand::{SeedableRng,
    rand_core::{TryRng, utils}};
use serde::{Deserialize, Serialize};

use crate::util::read_le_u64;

/// The "Small Fast Chaotic" PRNG, originally designed by Chris Doty-Humphrey.
///
/// This PRNG holds 256 bits of state (including a 64-bit counter) and generates 64 bits
/// of output with each step. The minimum cycle length is $`2^{64}`$, and the expected
/// period is $`~2^{255}`$. Independent seeds are guaranteed to not collide within the
/// first $`2^{64}`$ steps, although the actual time to collision may be much longer.
///
/// This specific implementation passes [PractRand] with >2TB of output for each of
/// several low-entropy seeds, and >1TB for sets of 8 and 64 parallel streams whose seed
/// differs by only a single bit.
///
/// ## Seeding (construction)
///
/// This generator implements the [`SeedableRng`] trait. Any method may be used,
/// and the results are guaranteed to be [portable].
///
/// Seeding the generator with `seed_from_u64` is often the most convenient and
/// guarantees good pseudorandom statistics.
/// ```
/// use hoomd_rand::SFC64;
/// use rand::SeedableRng;
/// let rng = SFC64::seed_from_u64(42);
/// ```
/// See also [Seeding RNGs] in the Rust Rand book.
///
/// ## Generation
///
/// The generators implements [`Rng`] and thus also `Rng`.
/// See also the [Random Values] chapter in the Rust Rand book.
///
/// [portable]: https://rust-random.github.io/book/crate-reprod.html
/// [Seeding RNGs]: https://rust-random.github.io/book/guide-seeding.html
/// [Random Values]: https://rust-random.github.io/book/guide-values.html
/// [PractRand]: https://pracrand.sourceforge.net

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SFC64 {
    /// The internal state of the PRNG.
    state: [u64; 3],
    /// The current step of the PRNG. This value is incremented with each output, but
    /// can be set independently without issues.
    counter: u64,
}

impl SFC64 {
    /// Set up the PRNG from four u64 values, discarding the first 12 outputs to avoid
    /// correlations between similar seeds.
    #[inline]
    pub(crate) fn initialize(a: u64, b: u64, c: u64, counter: u64) -> Self {
        let mut rng = Self {
            state: [a, b, c],
            counter,
        };
        (0..12).for_each(|_| {
            rng.step();
        });
        rng
    }
    /// Increment the RNG forward, returning the generated result.
    #[inline]
    fn step(&mut self) -> u64 {
        const BARREL_SHIFT: u32 = 24;
        const RSHIFT: u64 = 11;
        const LSHIFT: u64 = 3;
        let out = self.state[0]
            .wrapping_add(self.state[1])
            .wrapping_add(self.counter);
        self.state[0] = self.state[1] ^ (self.state[1] >> RSHIFT);
        self.state[1] = self.state[2].wrapping_add(self.state[2] << LSHIFT);
        self.state[2] = self.state[2].rotate_left(BARREL_SHIFT).wrapping_add(out);
        self.counter = self.counter.wrapping_add(1); // Weyl increment of 1
        out
    }
}

impl TryRng for SFC64 {
    type Error = Infallible;

    #[inline]
    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        Ok(self.step())
    }
    #[inline]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the truncation is intended"
    )]
    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        Ok(self.step() as u32)
    }
    #[inline]
    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
        utils::fill_bytes_via_next_word(dst, || Ok(self.step()))
    }
}
impl SeedableRng for SFC64 {
    type Seed = [u8; 24];
    #[inline]
    fn from_seed(seed: Self::Seed) -> Self {
        let seed = &mut seed.as_slice();
        Self::initialize(read_le_u64(seed), read_le_u64(seed), read_le_u64(seed), 0)
    }
    #[inline]
    fn seed_from_u64(state: u64) -> Self {
        Self::initialize(state, 0, 0, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rstest::rstest;

    // To generate test data:
    // ```python
    // from numpy.random import SFC64
    // rng = SFC64()
    // rng.state = {
    // "bit_generator": "SFC64",
    // "state": {"state":np.array([0,0,0,1], dtype=np.uint64)},
    // "has_uint32": 0,
    // "uinteger": 0
    // }
    // Generator(rng).integers(
    //     np.uint64(2**64-1), size=30, dtype=np.uint64, endpoint=True
    // )
    //
    // array([                   1,                    2,                   12,
    //                   150994975,     2533275243454595,     7601064794802825,
    //           81067317779357880,   939904496648135862,  8358283255545778679,
    //         3766502869832372394,  3749490695777847966, 10585829324311968283,
    // # Done warming up =====================================================
    //         4237781876154851393, 17705428440413258140,  1322197197711907681,
    //          822724228132957142,  2474202602039083746,  5912426283212852001,
    //        15821317571115833757, 10375476962501160791, 16772721532102950986,
    //        10344472337605944266, 17980158625826311727,  5068054200821024954,
    //         7387740087731720141,  3233965506304800125,   805889469043559033,
    //         5406730423683535818, 15071469935640508508,  4156074611580516626],
    //       dtype=uint64)

    // ```

    #[rustfmt::skip]
    const SFC64_REFERENCE_OUTPUT: [u64; 17] = [
        4_237_781_876_154_851_393,  17_705_428_440_413_258_140,
        1_322_197_197_711_907_681,     822_724_228_132_957_142,
        2_474_202_602_039_083_746,   5_912_426_283_212_852_001,
        15_821_317_571_115_833_757, 10_375_476_962_501_160_791,
        16_772_721_532_102_950_986, 10_344_472_337_605_944_266,
        17_980_158_625_826_311_727,  5_068_054_200_821_024_954,
        7_387_740_087_731_720_141,   3_233_965_506_304_800_125,
        805_889_469_043_559_033,     5_406_730_423_683_535_818,
        15_071_469_935_640_508_508
    ];

    #[rstest::fixture]
    fn large_uniform_sample() -> Vec<u64> {
        const N: u32 = 2u32.pow(23);
        let mut rng = SFC64::initialize(456_981, 0xcafe, 9_345_663_908, 123_456_789);
        (0..N).map(|_| rng.next_u64()).collect::<Vec<_>>()
    }

    #[rstest::fixture]
    fn zeros(large_uniform_sample: Vec<u64>) -> u32 {
        large_uniform_sample
            .iter()
            .fold(0, |acc, x| acc + x.count_zeros())
    }
    #[rstest::fixture]
    fn ones(large_uniform_sample: Vec<u64>) -> u32 {
        large_uniform_sample
            .iter()
            .fold(0, |acc, x| acc + x.count_ones())
    }

    #[test]
    fn test_sfc64_against_numpy() {
        let mut rng = SFC64::seed_from_u64(0);
        (0..17).for_each(|i| {
            assert_eq!(
                rng.next_u64(),
                SFC64_REFERENCE_OUTPUT[i],
                "failed at index {i}"
            );
        });
    }

    /// This test is quite slow, but tests that our generation of the ~2^17th value
    /// matches the reference impl
    ///
    /// NOTE: set state to [0,0,0,1], then
    /// `rg.integers(np.uint64(2**64-1), size=2**(17)+12, dtype=np.uint64, endpoint=True)[-1]`
    #[test]
    fn test_sfc64_deep() {
        let mut rng = SFC64::seed_from_u64(0);
        (0..(2u64.pow(17) - 1)).for_each(|_| {
            rng.next_u64();
        });
        assert_eq!(rng.next_u64(), 4_977_758_738_274_538_201);
    }

    #[rstest]
    fn test_uniformity_mean(large_uniform_sample: Vec<u64>, zeros: u32, ones: u32) {
        let standard_error = ((4 * 64 * large_uniform_sample.len()) as f64)
            .sqrt()
            .recip();
        approxim::assert_abs_diff_eq!(
            f64::from(zeros) / f64::from(ones),
            1.0,
            epsilon = standard_error
        );
    }
    #[rstest]
    fn test_uniformity_variance(large_uniform_sample: Vec<u64>, ones: u32) {
        let n_bits: u128 = large_uniform_sample.len() as u128 * 64;

        let variance = (u128::from(ones) * (n_bits - u128::from(ones))) as f64
            / (n_bits * (n_bits - 1)) as f64;
        let epsilon = (n_bits as f64).sqrt().recip(); // approximate
        approxim::assert_abs_diff_eq!(variance, 0.25, epsilon = epsilon);
    }

    #[rstest]
    fn test_autocorrelation(
        large_uniform_sample: Vec<u64>,
        #[values(1, 2, 4, 8, 64, 256, 65536)] lag: usize,
    ) {
        let n = large_uniform_sample.len();
        let sample_f64: Vec<f64> = large_uniform_sample.iter().map(|&x| x as f64).collect();

        let mean = sample_f64.iter().sum::<f64>() / n as f64;
        let var = sample_f64.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let cov = (0..n - lag)
            .map(|i| (sample_f64[i] - mean) * (sample_f64[i + lag] - mean))
            .sum::<f64>()
            / (n - lag) as f64;
        let autocorr = cov / var;

        let epsilon = 3.0 / (n as f64).sqrt();
        approxim::assert_abs_diff_eq!(autocorr, 0.0, epsilon = epsilon);
    }
}
