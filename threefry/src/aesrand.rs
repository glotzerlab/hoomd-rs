//! https://github.com/TheIronBorn/simd_prngs/blob/master/src/prngs/aes_rand.rs
//! Ported to aarch64
use std::arch::aarch64::*;

use rand::{RngCore, SeedableRng};
use rand_core::block::{BlockRng64, BlockRngCore};

/// .
pub struct AESRandCore {
    /// .
    state: uint8x16_t,
}

impl AESRandCore {
    /// # SAFETY: While the array might not be aligned, ``vld1q_u8`` aligns the data
    #[target_feature(enable = "aes")]
    #[inline]
    pub fn gen_array(&mut self) -> [uint8x16_t; 2] {
        // SAFETY: While the array might not be aligned, vld1q_u8c aligns the data
        let increment = unsafe {
            vld1q_u8(
                [
                    0x2f, 0x2b, 0x29, 0x25, 0x1f, 0x1d, 0x17, 0x13, 0x11, 0x0D, 0x0B, 0x07, 0x05,
                    0x03, 0x02, 0x01,
                ]
                .as_ptr(),
            )
        };

        self.state = vaddq_u8(self.state, increment);
        let penultimate = vaesmcq_u8(vaeseq_u8(self.state, increment));
        let penultimate1 = vaesmcq_u8(vaeseq_u8(penultimate, increment));
        // InverseMixColumns + (InvSubBytes + InvShiftRows + AddRoundKey)
        let penultimate2 = vaesimcq_u8(vaesdq_u8(penultimate, increment));
        [penultimate1, penultimate2]
    }
}

impl BlockRngCore for AESRandCore {
    type Item = u64;
    type Results = [u64; 4];

    #[inline]
    fn generate(&mut self, results: &mut Self::Results) {
        // SAFETY: As long as the +aes feature is enabled
        let data = unsafe { self.gen_array() };
        // SAFETY: ???
        *results = unsafe { std::mem::transmute::<[uint8x16_t; 2], [u64; 4]>(data) };
        //         [
        //         vreinterpretq_u64_u8(data[0]),
        //         vreinterpretq_u64_u8(data[1]),
        //     ])
        // };
    }
}
/// .
pub struct AESRandRng(BlockRng64<AESRandCore>);
impl SeedableRng for AESRandRng {
    type Seed = [u8; 16];

    #[inline]
    fn from_seed(seed: Self::Seed) -> Self {
        // SAFETY: ???
        Self(BlockRng64::new(AESRandCore {
            state: unsafe { vld1q_u8(seed.as_ptr()) },
        }))
    }
}

impl RngCore for AESRandRng {
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
