// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use rand::{RngCore, SeedableRng};
use rand_core::impls;

use crate::backends::{read_u64_le_unchecked, rotl_u64};

/// The "Small Fast Chaotic" PRNG, originally designed by Chris Doty-Humphrey.
///
/// This PRNG holds 256 bits of state (including a 64-bit counter) and generates 64 bits
/// of output with each step. The minimum cycle length is $`2^64`$, and the expected
/// period is $`~2^255`$. Independent seeds are guaranteed to not collide within the
/// first $`2^64`$ steps, although the actual time to collision may be much longer.
pub struct SFC64Rng {
    /// The internal state of the PRNG.
    state: [u64; 3],
    /// The current step of the PRNG. This value is incremented with each output, but
    /// can be set independently without issues.
    counter: u64,
}

impl SFC64Rng {
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
        // println!("{:?} : {:?}", self.counter, self.state);
        let out = self.state[0] + self.state[1] + self.counter;
        self.state[0] = self.state[1] ^ (self.state[1] >> RSHIFT);
        self.state[1] = self.state[2] + (self.state[2] << LSHIFT);
        self.state[2] = rotl_u64(self.state[2], BARREL_SHIFT) + out;
        self.counter += 1;
        println!("  : {out}");
        out
    }

    /// Initialize the PRNG from 192 bits of state and a u64 counter.
    #[inline]
    #[must_use]
    pub fn from_state_and_counter(state: [u64; 3], counter: u64) -> Self {
        Self::initialize(state[0], state[1], state[2], counter)
    }
}

impl RngCore for SFC64Rng {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.step()
    }
    #[inline]
    fn next_u32(&mut self) -> u32 {
        impls::next_u32_via_fill(self)
    }
    #[inline]
    fn fill_bytes(&mut self, dst: &mut [u8]) {
        impls::fill_bytes_via_next(self, dst);
    }
}
impl SeedableRng for SFC64Rng {
    type Seed = [u8; 32];
    #[inline]
    fn from_seed(seed: Self::Seed) -> Self {
        Self::initialize(
            read_u64_le_unchecked(seed, 0..8),
            read_u64_le_unchecked(seed, 8..16),
            read_u64_le_unchecked(seed, 16..24),
            read_u64_le_unchecked(seed, 24..32),
        )
    }
    #[inline]
    fn seed_from_u64(state: u64) -> Self {
        Self::initialize(0, 0, 0, state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    // ```python
    // from numpy.random import SFC64
    // rng = SFC64()
    // rng.state = {
    // "bit_generator": "SFC64",
    // "state": {"state":np.array([0,0,0,0], dtype=np.uint64)},
    // "has_uint32": 0,
    // "uinteger": 0
    // }
    // Generator(rng).integers(np.uint64(2**64-1), size=29, dtype=np.uint64)
    //
    // np.array([            0,                    1,                   11,
    // 150994974,     2533275243454594,     7601064794802824,
    // 81067317779357879,   939904496648135861,  8358283255545778678,
    // 3766502869832372393,  3749490695777847965, 10585829324311968282,
    // # INITIALIZATION COMPLETE (12 values)
    // 4237781876154851392, 17705428440413258139,  1322197197711907680,
    // 822724228132957141,  2474202602039083745,  5912426283212852000,
    // 15821317571115833756, 10375476962501160790, 16772721532102950985,
    // 10344472337605944265, 17980158625826311726,  5068054200821024953,
    // 7387740087731720140,  3233965506304800124,   805889469043559032,
    // 5406730423683535817, 15071469935640508507], dtype=uint64)
    // ```

    const SFC64_REFERENCE_OUTPUT: [u64; 17] = [
        4_237_781_876_154_851_392,
        17_705_428_440_413_258_139,
        1_322_197_197_711_907_680,
        822_724_228_132_957_141,
        2_474_202_602_039_083_745,
        5_912_426_283_212_852_000,
        15_821_317_571_115_833_756,
        10_375_476_962_501_160_790,
        16_772_721_532_102_950_985,
        10_344_472_337_605_944_265,
        17_980_158_625_826_311_726,
        5_068_054_200_821_024_953,
        7_387_740_087_731_720_140,
        3_233_965_506_304_800_124,
        805_889_469_043_559_032,
        5_406_730_423_683_535_817,
        15_071_469_935_640_508_507,
    ];

    #[test]
    fn test_sfc64_zeros() {
        let mut x = SFC64Rng::seed_from_u64(0);
        (0..17).for_each(|i| {
            assert_eq!(
                x.next_u64(),
                SFC64_REFERENCE_OUTPUT[0],
                "failed at index {i}"
            );
        });
    }

    // This test is quite slow, but tests that our generation of the ~2^26th value
    // matches the reference impl
    // #[test]
    // fn test_threefry2x64_r13_deep() {
    //     let mut x = SFC64Rng::seed_from_u64(0);
    //     (0..(110_423_593 * 2 - 4)).for_each(|_| {
    //         x.next_u64();
    //     });
    //     assert_eq!(x.next_u64(), 9_808_966_926_499_203_172u64);
    // }
}
