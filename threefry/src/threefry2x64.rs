//! Asdf.

/// asdf
use crate::backends::{self, C240};
use rand::SeedableRng;
use rand_core::{
    RngCore,
    block::{BlockRng64, BlockRngCore},
};
/// Key schedule for ``ThreeFry2x64``.
const ROTATION_2X64: [u32; 8] = [16, 42, 12, 31, 16, 32, 24, 21];

/// TODO: unsafe if N < slice size
#[inline]
fn read_u64_le_unchecked<const N: usize>(stream: [u8; N], range: std::ops::Range<usize>) -> u64 {
    u64::from_le_bytes(stream[range].try_into().unwrap_or_else(|_| unreachable!()))
}

/// TODO.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ThreeFry2x64Core<const R: usize> {
    /// .
    seed: [u64; 3],
    /// .
    counter: [u64; 2],
}
impl<const R: usize> BlockRngCore for ThreeFry2x64Core<R> {
    type Item = u64;
    type Results = [u64; 2];

    #[inline]
    fn generate(&mut self, results: &mut Self::Results) {
        (0..R).for_each(|d| {
            if d % 4 == 0 {
                let s = d / 4;
                self.counter[0] = self.counter[0].wrapping_add(self.seed[s % 3]);
                self.counter[1] = self.counter[1].wrapping_add(self.seed[(s + 1) % 3] + s as u64);
            }
            backends::mix2x64(&mut self.counter, ROTATION_2X64[d % 8]);
        });
        if R.is_multiple_of(4) {
            let s = R / 4;
            self.counter[0] = self.counter[0].wrapping_add(self.seed[s % 3]);
            self.counter[1] = self.counter[1].wrapping_add(self.seed[(s + 1) % 3] + s as u64);
        }
        *results = self.counter;
    }
}

impl<const R: usize> SeedableRng for ThreeFry2x64Rng<R> {
    type Seed = [u8; 16];
    #[inline]
    fn from_seed(seed: Self::Seed) -> Self {
        let (k0, k1) = (
            read_u64_le_unchecked(seed, 0..8),
            read_u64_le_unchecked(seed, 8..16),
        );
        Self(BlockRng64::new(ThreeFry2x64Core {
            seed: [k0, k1, C240 ^ k0 ^ k1],
            counter: [0u64, 0u64],
        }))
    }
    #[inline]
    fn seed_from_u64(state: u64) -> Self {
        Self(BlockRng64::new(ThreeFry2x64Core {
            seed: [0, state, C240 ^ state],
            counter: [0u64, 0u64],
        }))
    }
}

/// TODO.
pub struct ThreeFry2x64Rng<const R: usize>(BlockRng64<ThreeFry2x64Core<R>>);
impl<const R: usize> ThreeFry2x64Rng<R> {
    /// TODO
    #[inline]
    pub fn set_stream(&mut self, stream: [u8; 16]) {
        self.0.core.counter[0] = read_u64_le_unchecked(stream, 0..8);
        self.0.core.counter[1] = read_u64_le_unchecked(stream, 8..16);
    }
    ///.
    #[inline]
    pub fn set_stream_from_u64(&mut self, stream: u64) {
        self.0.core.counter = [0, stream];
    }
}
impl<const R: usize> RngCore for ThreeFry2x64Rng<R> {
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

    const THREEFRY_ZEROS_R13_OUTPUT: [u64; 17] = [
        17_395_065_817_820_070_077,
        16_798_320_980_932_587_445,
        2_652_519_131_407_607_297,
        13_161_320_466_366_176_746,
        17_030_694_613_443_170_302,
        4_344_906_048_248_695_035,
        5_179_954_432_202_753_853,
        5_285_716_355_519_551_071,
        18_008_838_412_780_928_950,
        11_882_322_564_036_187_161,
        4_029_351_837_281_676_536,
        8_970_127_999_437_897_612,
        2_632_576_919_458_191_698,
        17_148_268_545_001_954_552,
        10_276_295_674_570_553_388,
        7_194_263_575_669_194_817,
        6_906_374_478_066_415_370,
    ];

    const THREEFRY_ZEROS_R20_OUTPUT: [u64; 17] = [
        14_030_652_003_081_164_901,
        8_034_964_082_011_408_461,
        18_186_675_445_314_987_089,
        5_866_305_986_487_634_629,
        12_288_944_918_058_316_875,
        3_922_498_236_977_998_704,
        7_911_565_321_056_494_501,
        2_372_175_586_207_549_487,
        11_068_699_058_490_897_680,
        14_886_644_074_691_433_069,
        7_284_550_854_486_178_372,
        9_417_633_656_741_876_096,
        6_018_047_158_176_981_823,
        7_535_311_300_315_424_768,
        3_924_435_617_810_241_325,
        7_359_613_376_437_193_302,
        12_611_337_494_571_985_063,
    ];

    #[test]
    fn test_threefry2x64_r13_zeros() {
        let mut x = ThreeFry2x64Rng::<13>::seed_from_u64(0);
        x.set_stream_from_u64(0);
        (0..17).for_each(|i| assert_eq!(x.next_u64(), THREEFRY_ZEROS_R13_OUTPUT[i], "Index {i}"));
    }
    #[test]
    fn test_threefry2x64_r20_zeros() {
        let mut x = ThreeFry2x64Rng::<20>::seed_from_u64(0);
        x.set_stream_from_u64(0);
        (0..17).for_each(|i| assert_eq!(x.next_u64(), THREEFRY_ZEROS_R20_OUTPUT[i], "Index {i}"));
    }
}
