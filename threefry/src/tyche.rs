// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use crate::util::{read_u32_le_unchecked, read_u64_le_unchecked, rotl_u32};
use rand::{RngCore, SeedableRng};
use rand_core::impls;

pub struct Tyche4x32Rng {
    a: u32,
    b: u32,
    seed: u32,
    d: u32,
}
/// ⌊2**32⌋/π
const D_INIT: u32 = 1_367_130_551;
impl Tyche4x32Rng {
    /// .
    #[inline]
    fn initialize(step: u64, substep: u32, seed: u32) -> Self {
        let mut rng = Self {
            a: (step >> 32) as u32,         //  upper 32 bits
            b: (step & 0xffff_ffff) as u32, // lower 32 bits
            seed,
            d: D_INIT ^ substep,
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
        self.d = rotl_u32(self.d ^ self.a, 16);
        self.seed += self.d;
        self.b = rotl_u32(self.b ^ self.seed, 12);
        self.a += self.b;
        self.d = rotl_u32(self.d ^ self.a, 8);
        self.seed += self.d;
        self.b = rotl_u32(self.b ^ self.seed, 7);
    }
    /// .
    #[inline]
    fn mix(&mut self) -> u32 {
        self.chacha_quarterround();
        self.b
    }
}

impl RngCore for Tyche4x32Rng {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        impls::next_u64_via_u32(self)
    }
    #[inline]
    fn next_u32(&mut self) -> u32 {
        self.mix()
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
        Self::initialize(state, 0, 0xaaaa_aaaa)
    }
}
