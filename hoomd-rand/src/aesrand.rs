// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Ported to aarch64
use std::arch::aarch64::{
    uint8x16_t, vaddq_u64, vaesdq_u8, vaeseq_u8, vaesimcq_u8, vaesmcq_u8, vld1q_u8,
    vreinterpretq_u8_u64, vreinterpretq_u64_u8,
};

use rand::{
    RngCore, SeedableRng,
    rand_core::block::{BlockRng64, BlockRngCore},
};

/// PRNG using the AES block cipher as an [invertible random mapping](https://www.pcg-random.org/posts/random-invertible-mapping-statistics.html).
///
///
/// On systems where AES operations are implemented in hardware, this generator yields
/// data extremely quickly, with very low latency. This is relatively unique in the
/// ratio of state to output, with a 128 bit state and 256 bits of output per step.
///
/// This implementation is based on:
/// <https://github.com/TheIronBorn/simd_prngs/blob/master/src/prngs/aes_rand.rs>
///
/// Requires arch `aarch64` with `neon` and `aes` features.
pub struct AESRandCore {
    /// Internal state of the prng.
    state: uint8x16_t,
}

/// Byte increment for each step of the rng.
const INCREMENT_BYTES: [u8; 16] = [
    0x2f, 0x2b, 0x29, 0x25, 0x1f, 0x1d, 0x17, 0x13, 0x11, 0x0d, 0x0b, 0x07, 0x05, 0x03, 0x02, 0x01,
];

impl AESRandCore {
    /// Small, fast PRNG that uses AES as a random invertible mapping.
    #[target_feature(enable = "aes")]
    #[inline]
    pub fn gen_array(&mut self) -> [uint8x16_t; 2] {
        // SAFETY: While the array might not be aligned, vld1q_u8 aligns the data
        let increment = unsafe { vld1q_u8(INCREMENT_BYTES.as_ptr()) };

        // Increment the counter as a u64
        self.state = vreinterpretq_u8_u64(vaddq_u64(
            vreinterpretq_u64_u8(self.state),
            vreinterpretq_u64_u8(increment),
        ));
        let penultimate = vaesmcq_u8(vaeseq_u8(self.state, increment));
        let left = vaesmcq_u8(vaeseq_u8(penultimate, increment));
        let right = vaesimcq_u8(vaesdq_u8(penultimate, increment));
        [left, right]
    }
}

impl BlockRngCore for AESRandCore {
    type Item = u64;
    type Results = [u64; 4];

    #[inline]
    fn generate(&mut self, results: &mut Self::Results) {
        // SAFETY: As long as the +aes feature is enabled
        let data = unsafe { self.gen_array() };
        // SAFETY: As long as size_of::<[uint8x16_t; 2]>() == 32
        let bytes: [u8; 32] = unsafe { std::mem::transmute(data) };
        for (i, chunk) in bytes.chunks_exact(8).enumerate() {
            results[i] = u64::from_ne_bytes(chunk.try_into().expect("asdf"));
        }
    }
}
/// PRNG using the AES block cipher as an [invertible random mapping](https://www.pcg-random.org/posts/random-invertible-mapping-statistics.html).
///
///
/// On systems where AES operations are implemented in hardware, this generator yields
/// data extremely quickly, with very low latency. This is relatively unique in the
/// ratio of state to output, with a 128 bit state and 256 bits of output per step.
/// The original (x86-64 implementation) of this method is here:
/// <https://github.com/TheIronBorn/simd_prngs/blob/master/src/prngs/aes_rand.rs>
pub struct AESRand(BlockRng64<AESRandCore>);
impl SeedableRng for AESRand {
    type Seed = [u8; 16];

    #[inline]
    fn from_seed(seed: Self::Seed) -> Self {
        // SAFETY: seed.as_ptr() has all 16 bytes and properly aligned for uint8x16_t
        Self(BlockRng64::new(AESRandCore {
            state: unsafe { vld1q_u8(seed.as_ptr()) },
        }))
    }
}

impl RngCore for AESRand {
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
