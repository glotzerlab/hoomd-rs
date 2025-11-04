use crate::backends::read_u32_le_unchecked;
use crate::backends::read_u64_le_unchecked;
use crate::backends::rotl_u32;
use rand::{RngCore, SeedableRng};
use rand_core::{impls, le::read_u32_into};
use wide::u32x4;

pub struct Tyche4x32Rng {
    a: u32x4,
    b: u32x4,
    seed: u32x4,
    d: u32x4,
}
#[inline]
pub(crate) fn rotl_u32x4(x: u32x4, d: u32) -> u32x4 {
    (x << d) | (x >> (32 - d))
}
/// ⌊2**32⌋/π
const D_INIT: u32 = 1_367_130_551;
impl Tyche4x32Rng {
    /// .
    #[inline]
    fn initialize(step: u64, substep: u32, seed: u32) -> Self {
        let mut rng = Self {
            a: u32x4::from((step >> 32) as u32),         //  upper 32 bits
            b: u32x4::from((step & 0xFFFF_FFFF) as u32), // lower 32 bits
            seed: seed.into(),
            d: u32x4::from(D_INIT ^ substep),
        };
        // Decorrelate the initial state.
        (0..20).for_each(|_| {
            rng.mix();
        });
        rng
    }
    /// .
    #[inline]
    fn chacha_quarterround(&mut self) {
        self.a += self.b;
        self.d = rotl_u32x4(self.d ^ self.a, 16);
        self.seed += self.d;
        self.b = rotl_u32x4(self.b ^ self.seed, 12);
        self.a += self.b;
        self.d = rotl_u32x4(self.d ^ self.a, 8);
        self.seed += self.d;
        self.b = rotl_u32x4(self.b ^ self.seed, 7);
    }
    /// .
    #[inline]
    fn mix(&mut self) -> [u64; 2] {
        self.chacha_quarterround();
        unsafe { std::mem::transmute::<u32x4, [u64; 2]>(self.b) }
    }
}

impl RngCore for Tyche4x32Rng {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        // impls::next_u64_via_u32(self)
        self.mix()[0]
    }
    #[inline]
    fn next_u32(&mut self) -> u32 {
        // self.mix()
        impls::next_u32_via_fill(self)
    }
    #[inline]
    fn fill_bytes(&mut self, dst: &mut [u8]) {
        impls::fill_bytes_via_next(self, dst);
    }
}
impl SeedableRng for Tyche4x32Rng {
    type Seed = [u8; 16];
    #[inline]
    fn from_seed(seed: Self::Seed) -> Self {
        Self::initialize(
            read_u64_le_unchecked(seed, 0..8),
            read_u32_le_unchecked(seed, 8..12),
            read_u32_le_unchecked(seed, 12..16),
        )
    }
    #[inline]
    fn seed_from_u64(state: u64) -> Self {
        Self::initialize(state, 0, 0xAAAA_AAAA)
    }
}
