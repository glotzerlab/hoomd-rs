use crate::backends::{self, C240, read_u64_le_unchecked};
use rand::SeedableRng;
use rand_core::{
    RngCore,
    block::{BlockRng64, BlockRngCore},
};

/// Key schedule for ``ThreeFry4x64``.
const ROTATION_4X64: [(u32, u32); 8] = [
    (14, 16),
    (52, 57),
    (23, 40),
    (5, 37),
    (25, 33),
    (46, 12),
    (58, 22),
    (32, 32),
];

/// .
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ThreeFry4x64Core<const R: usize> {
    /// .
    seed: [u64; 5],
    /// .
    counter: [u64; 4],
}
impl<const R: usize> BlockRngCore for ThreeFry4x64Core<R> {
    type Item = u64;
    type Results = [u64; 4];

    #[inline]
    fn generate(&mut self, results: &mut Self::Results) {
        (0..R).for_each(|d| {
            if d % 4 == 0 {
                let s = d / 4;
                self.counter[0] = self.counter[0].wrapping_add(self.seed[s % 5]);
                self.counter[1] = self.counter[1].wrapping_add(self.seed[(s + 1) % 5]);
                self.counter[2] = self.counter[2].wrapping_add(self.seed[(s + 2) % 5]);
                self.counter[3] = self.counter[3].wrapping_add(self.seed[(s + 3) % 5] + s as u64);
            }
            backends::mix4x64(&mut self.counter, ROTATION_4X64[d % 8], d);
        });
        if R.is_multiple_of(4) {
            let s = R / 4;
            self.counter[0] = self.counter[0].wrapping_add(self.seed[s % 5]);
            self.counter[1] = self.counter[1].wrapping_add(self.seed[(s + 1) % 5]);
            self.counter[2] = self.counter[2].wrapping_add(self.seed[(s + 2) % 5]);
            self.counter[3] = self.counter[3].wrapping_add(self.seed[(s + 3) % 5] + s as u64);
        }
        *results = self.counter;
    }
}
/// .
pub struct ThreeFry4x64Rng<const R: usize>(BlockRng64<ThreeFry4x64Core<R>>);
impl<const R: usize> SeedableRng for ThreeFry4x64Rng<R> {
    type Seed = [u8; 32];
    #[inline]
    fn from_seed(seed: Self::Seed) -> Self {
        let (k0, k1, k2, k3) = (
            read_u64_le_unchecked(seed, 0..8),
            read_u64_le_unchecked(seed, 8..16),
            read_u64_le_unchecked(seed, 16..24),
            read_u64_le_unchecked(seed, 24..32),
        );
        Self(BlockRng64::new(ThreeFry4x64Core {
            seed: [k0, k1, k2, k3, C240 ^ k0 ^ k1 ^ k2 ^ k3],
            counter: [0u64; 4],
        }))
    }
    #[inline]
    fn seed_from_u64(state: u64) -> Self {
        Self(BlockRng64::new(ThreeFry4x64Core {
            seed: [0, 0, 0, state, C240 ^ state],
            counter: [0u64; 4],
        }))
    }
}
impl<const R: usize> ThreeFry4x64Rng<R> {
    /// TODO
    #[inline]
    pub fn set_stream(&mut self, stream: [u8; 32]) {
        self.0.core.counter[0] = read_u64_le_unchecked(stream, 0..8);
        self.0.core.counter[1] = read_u64_le_unchecked(stream, 8..16);
        self.0.core.counter[2] = read_u64_le_unchecked(stream, 16..24);
        self.0.core.counter[3] = read_u64_le_unchecked(stream, 24..32);
    }
    ///.
    #[inline]
    pub fn set_stream_from_u64(&mut self, stream: u64) {
        self.0.core.counter = [0, 0, 0, stream];
    }
}
impl<const R: usize> RngCore for ThreeFry4x64Rng<R> {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }
    #[inline]
    fn next_u32(&mut self) -> u32 {
        self.0.next_u32()
    }
    #[inline]
    fn fill_bytes(&mut self, dst: &mut [u8]) {
        self.0.fill_bytes(dst);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    // Data generated from the Random123 ThreeFry4x64_rN

    const THREEFRY_ZEROS_R12_OUTPUT: [u64; 17] = [
        29_492_327_419_918_145,
        4_614_276_061_115_832_004,
        16_925_429_801_461_668_750,
        5_660_986_226_915_721_659,
        3_033_024_947_340_166_478,
        6_781_058_749_161_877_118,
        3_351_254_782_774_760_096,
        1_548_122_769_143_434_716,
        3_905_849_510_978_950_725,
        5_004_179_567_602_581_535,
        15_834_097_318_001_325_822,
        2_381_001_677_752_307_526,
        12_816_080_602_167_225_049,
        3_678_248_037_922_036_678,
        17_585_484_401_516_521_053,
        18_190_862_716_655_924_057,
        8_399_446_667_960_079_266,
    ];
    const THREEFRY_ZEROS_R20_OUTPUT: [u64; 17] = [
        657_963_966_844_654_903,
        6_166_588_228_550_287_621,
        5_463_532_747_209_585_884,
        17_161_507_908_560_806_923,
        4_326_716_718_985_618_210,
        14_909_708_087_270_875_011,
        14_557_190_979_261_578_939,
        9_749_013_750_939_284_005,
        2_354_230_655_075_451_029,
        6_149_610_916_980_711_482,
        4_409_333_508_652_010_814,
        3_219_883_965_556_188_688,
        8_873_700_516_375_412_562,
        4_657_066_281_480_209_243,
        11_312_148_763_700_402_601,
        14_501_740_901_229_956_470,
        2_188_744_427_393_466_326,
    ];

    #[test]
    fn test_threefry4x64_r12_zeros() {
        let mut x = ThreeFry4x64Rng::<12>::seed_from_u64(0);
        x.set_stream_from_u64(0);
        (0..17).for_each(|i| assert_eq!(x.next_u64(), THREEFRY_ZEROS_R12_OUTPUT[i], "Index {i}"));
    }
    #[test]
    fn test_threefry4x64_r20_zeros() {
        let mut x = ThreeFry4x64Rng::<20>::seed_from_u64(0);
        x.set_stream_from_u64(0);
        (0..17).for_each(|i| assert_eq!(x.next_u64(), THREEFRY_ZEROS_R20_OUTPUT[i], "Index {i}"));
    }
}
